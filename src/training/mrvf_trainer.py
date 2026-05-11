from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any

import torch
from datasets import Dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.training.advantages import grpo_zscore_advantages, loo_advantages
from src.training.config import MRVFConfig
from src.training.data import prepare_mrvf_dataset
from src.training.reference_likelihood import teacher_forced_reference_logps


def _resolve_dtype(name: str) -> torch.dtype | None:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return None


def _sequence_logprob_from_texts(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prefix_texts: list[str],
    completion_texts: list[str],
) -> torch.Tensor:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    sequences = [prefix + completion for prefix, completion in zip(prefix_texts, completion_texts, strict=True)]
    tokenized = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
    prefix_ids = [
        tokenizer(prefix, add_special_tokens=False)["input_ids"]
        for prefix in prefix_texts
    ]
    device = next(model.parameters()).device
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    labels = input_ids[:, 1:]
    logprobs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    token_mask = torch.zeros_like(labels, dtype=logprobs.dtype)
    for row_idx, prefix in enumerate(prefix_ids):
        start = max(len(prefix) - 1, 0)
        token_mask[row_idx, start:] = 1.0
    return (logprobs * token_mask).sum(dim=-1)


class MRVFTrainer:
    def __init__(self, cfg: MRVFConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        self.random = random.Random(cfg.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dtype = _resolve_dtype(cfg.torch_dtype)
        model_kwargs: dict[str, Any] = {}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **model_kwargs).to(self.device)
        self.model.train()

        self.ref_model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **model_kwargs).to(self.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        if cfg.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        self.optimizer = AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        warmup_steps = int(cfg.max_steps * cfg.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=cfg.max_steps,
        )

    def _build_trace_prompt(self, prompt: str) -> str:
        if not self.cfg.use_thinking:
            return f"{prompt}\n{self.cfg.trace_instruction}"
        return (
            f"{prompt}\n"
            "Think briefly using <think>...</think>, and output only the plan inside the think block."
        )

    def _generate_grouped_traces(self, prompts: list[str]) -> list[str]:
        repeated_prompts = [self._build_trace_prompt(prompt) for prompt in prompts for _ in range(self.cfg.num_generations)]
        tokenized = self.tokenizer(repeated_prompts, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                **tokenized,
                do_sample=True,
                max_new_tokens=self.cfg.max_trace_length,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_len = tokenized["input_ids"].shape[1]
        completion_ids = generated[:, prompt_len:]
        return self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    def _compute_kl(self, prefixes: list[str], traces: list[str]) -> torch.Tensor:
        if self.cfg.beta == 0:
            return torch.zeros((), device=self.device)
        with torch.no_grad():
            ref_logp = _sequence_logprob_from_texts(
                model=self.ref_model,
                tokenizer=self.tokenizer,
                prefix_texts=prefixes,
                completion_texts=traces,
            )
        curr_logp = _sequence_logprob_from_texts(
            model=self.model,
            tokenizer=self.tokenizer,
            prefix_texts=prefixes,
            completion_texts=traces,
        )
        return (curr_logp - ref_logp).mean()

    def _compute_losses_for_batch(self, batch_rows: list[dict[str, Any]]) -> tuple[torch.Tensor, dict[str, float]]:
        prompts = [row["prompt"] for row in batch_rows]
        references = [row["references"][: self.cfg.num_reference_samples] for row in batch_rows]
        trace_texts = self._generate_grouped_traces(prompts)
        grouped_refs = [refs for refs in references for _ in range(self.cfg.num_generations)]
        grouped_prompts = [self._build_trace_prompt(prompt) for prompt in prompts for _ in range(self.cfg.num_generations)]

        trace_logprob = _sequence_logprob_from_texts(
            model=self.model,
            tokenizer=self.tokenizer,
            prefix_texts=grouped_prompts,
            completion_texts=trace_texts,
        )

        with torch.no_grad():
            reward_outputs = teacher_forced_reference_logps(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt_texts=grouped_prompts,
                trace_texts=trace_texts,
                references=grouped_refs,
                max_reference_length=self.cfg.max_reference_length,
                answer_prefix=self.cfg.answer_prefix,
                length_normalization=self.cfg.reference_length_normalization,
            )

        ref_outputs = teacher_forced_reference_logps(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_texts=grouped_prompts,
            trace_texts=trace_texts,
            references=grouped_refs,
            max_reference_length=self.cfg.max_reference_length,
            answer_prefix=self.cfg.answer_prefix,
            length_normalization=self.cfg.reference_length_normalization,
        )

        grouped_log_mass = reward_outputs.log_mass.view(len(batch_rows), self.cfg.num_generations)
        if self.cfg.reward_transform == "log_mass":
            rewards = grouped_log_mass.reshape(-1)
        else:
            centered = grouped_log_mass - grouped_log_mass.max(dim=1, keepdim=True).values
            rewards = torch.exp(centered).reshape(-1)

        rewards_for_adv = rewards.detach()
        if self.cfg.advantage_mode == "loo":
            advantages = loo_advantages(rewards_for_adv, self.cfg.num_generations)
        else:
            advantages = grpo_zscore_advantages(rewards_for_adv, self.cfg.num_generations)

        trace_loss = -(advantages * trace_logprob).mean()

        if self.cfg.objective_mode == "mrvf_lite" or self.cfg.reference_loss_coef == 0:
            reference_loss = torch.zeros((), device=self.device)
        elif self.cfg.objective_mode == "log_mass_surrogate":
            reference_loss = -ref_outputs.log_mass.mean()
        else:
            log_mass = ref_outputs.log_mass.view(len(batch_rows), self.cfg.num_generations)
            stabilized = torch.exp(log_mass - log_mass.detach().max(dim=1, keepdim=True).values)
            reference_loss = -stabilized.mean()

        kl_term = self._compute_kl(grouped_prompts, trace_texts)
        loss = (
            self.cfg.trace_loss_coef * trace_loss
            + self.cfg.reference_loss_coef * reference_loss
            + self.cfg.beta * kl_term
        )
        metrics = {
            "loss": float(loss.detach().item()),
            "trace_loss": float(trace_loss.detach().item()),
            "reference_loss": float(reference_loss.detach().item()),
            "kl": float(kl_term.detach().item()),
            "reward_mean": float(rewards_for_adv.mean().item()),
        }
        return loss, metrics

    def train(self, raw_train_dataset: Dataset, raw_eval_dataset: Dataset) -> dict[str, Any]:
        train_dataset = prepare_mrvf_dataset(raw_train_dataset, max_reference_samples=self.cfg.num_reference_samples)
        _ = prepare_mrvf_dataset(raw_eval_dataset, max_reference_samples=self.cfg.num_reference_samples)
        train_rows = [dict(item) for item in train_dataset]
        if not train_rows:
            msg = "Prepared training dataset is empty."
            raise RuntimeError(msg)

        batch_size = max(1, self.cfg.per_device_train_batch_size)
        global_step = 0
        history: list[dict[str, float]] = []
        while global_step < self.cfg.max_steps:
            self.random.shuffle(train_rows)
            for start in range(0, len(train_rows), batch_size):
                batch = train_rows[start : start + batch_size]
                if not batch:
                    continue
                loss, metrics = self._compute_losses_for_batch(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                history.append(metrics)
                global_step += 1
                if global_step >= self.cfg.max_steps:
                    break

        self.model.save_pretrained(self.cfg.output_dir)
        self.tokenizer.save_pretrained(self.cfg.output_dir)
        return {
            "config": asdict(self.cfg),
            "steps": global_step,
            "last_metrics": history[-1] if history else {},
            "mean_loss": float(sum(item["loss"] for item in history) / max(len(history), 1)),
        }

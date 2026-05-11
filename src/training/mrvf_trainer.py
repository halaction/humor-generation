from __future__ import annotations

import random
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.training.advantages import grpo_zscore_advantages, loo_advantages
from src.training.config import MRVFConfig
from src.training.data import prepare_mrvf_dataset
from src.training.generation_utils import extract_completion_ids
from src.training.reference_likelihood import teacher_forced_reference_logps_from_ids


def _resolve_dtype(name: str) -> torch.dtype | None:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return None


@dataclass
class TraceBatch:
    prompt_texts: list[str]
    prompt_ids: list[list[int]]
    trace_ids: list[list[int]]
    trace_texts: list[str]
    references: list[list[str]]


@dataclass
class BatchDebugSample:
    prompt: str
    trace: str
    references: list[str]
    reward: float
    advantage: float
    trace_loss: float
    reference_loss: float


def _build_sequence_logprob_inputs(
    tokenizer: Any,
    prefix_ids: list[list[int]],
    completion_ids: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(prefix) + len(completion) for prefix, completion in zip(prefix_ids, completion_ids, strict=True))
    input_rows: list[list[int]] = []
    attn_rows: list[list[int]] = []
    target_rows: list[list[int]] = []

    for prefix, completion in zip(prefix_ids, completion_ids, strict=True):
        ids = prefix + completion
        pad = max_len - len(ids)
        row = ids + [tokenizer.pad_token_id] * pad
        attn = [1] * len(ids) + [0] * pad
        target = [0] * len(prefix) + [1] * len(completion) + [0] * pad
        input_rows.append(row)
        attn_rows.append(attn)
        target_rows.append(target)

    return (
        torch.tensor(input_rows, dtype=torch.long),
        torch.tensor(attn_rows, dtype=torch.long),
        torch.tensor(target_rows, dtype=torch.long),
    )


def _sequence_logprobs_from_ids(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    labels = input_ids[:, 1:]
    shifted_target_mask = target_mask[:, 1:].to(logits.dtype)
    shifted_attn_mask = attention_mask[:, 1:].to(logits.dtype)
    effective_mask = shifted_target_mask * shifted_attn_mask

    token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (token_logprobs * effective_mask).sum(dim=-1)


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
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **model_kwargs).to(self.device)
        self.model.train()

        if cfg.use_peft:
            self._enable_lora()

        if cfg.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        trainable_params = (param for param in self.model.parameters() if param.requires_grad)
        self.optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        warmup_steps = int(cfg.max_steps * cfg.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=cfg.max_steps,
        )
        self._current_step = 0

    def _enable_lora(self) -> None:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as error:  # pragma: no cover
            msg = "`use_peft=True` requires `peft` package."
            raise RuntimeError(msg) from error

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            bias="none",
            target_modules=list(self.cfg.lora_target_modules),
        )
        self.model = get_peft_model(self.model, peft_config)

    def _build_trace_prompt_text(self, prompt: str) -> str:
        if self.cfg.trace_format == "qwen_chat_thinking":
            messages = [{"role": "user", "content": f"{prompt}\n{self.cfg.trace_instruction}"}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if not self.cfg.use_thinking:
            return f"{prompt}\n{self.cfg.trace_instruction}"
        return (
            f"{prompt}\n"
            "Think briefly using <think>...</think>, and output only the plan inside the think block."
        )

    def _generate_trace_batch(self, prompts: list[str], references: list[list[str]]) -> TraceBatch:
        grouped_prompt_texts: list[str] = []
        grouped_prompt_ids: list[list[int]] = []
        grouped_references: list[list[str]] = []
        for prompt, refs in zip(prompts, references, strict=True):
            prompt_text = self._build_trace_prompt_text(prompt)
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            for _ in range(self.cfg.num_generations):
                grouped_prompt_texts.append(prompt_text)
                grouped_prompt_ids.append(prompt_ids)
                grouped_references.append(refs)

        encoded = self.tokenizer(grouped_prompt_texts, padding=True, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        input_width = input_ids.shape[1]
        grouped_prompt_ids = []
        for row_ids, row_mask in zip(input_ids.tolist(), attention_mask.tolist(), strict=True):
            grouped_prompt_ids.append([token for token, mask in zip(row_ids, row_mask, strict=True) if mask == 1])

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                max_new_tokens=self.cfg.max_trace_length,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        trace_ids: list[list[int]] = []
        trace_texts: list[str] = []
        for idx in range(generated.shape[0]):
            row = generated[idx].tolist()
            completion = extract_completion_ids(
                row,
                input_width=input_width,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            trace_ids.append(completion)
            trace_texts.append(self.tokenizer.decode(completion, skip_special_tokens=True))

        return TraceBatch(
            prompt_texts=grouped_prompt_texts,
            prompt_ids=grouped_prompt_ids,
            trace_ids=trace_ids,
            trace_texts=trace_texts,
            references=grouped_references,
        )

    def _trace_logprobs(self, trace_batch: TraceBatch, model: torch.nn.Module) -> torch.Tensor:
        input_ids, attention_mask, target_mask = _build_sequence_logprob_inputs(
            tokenizer=self.tokenizer,
            prefix_ids=trace_batch.prompt_ids,
            completion_ids=trace_batch.trace_ids,
        )
        return _sequence_logprobs_from_ids(
            model=model,
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            target_mask=target_mask.to(self.device),
        )

    def _compute_kl(self, trace_batch: TraceBatch) -> torch.Tensor:
        del trace_batch
        if self.cfg.use_kl:
            msg = "KL mode is not implemented yet."
            raise RuntimeError(msg)
        return torch.zeros((), device=self.device)

    def _append_sample_log(
        self,
        *,
        step: int,
        sample: BatchDebugSample | None,
    ) -> None:
        if self.cfg.logging_steps <= 0 or step == 0 or step % self.cfg.logging_steps != 0 or sample is None:
            return
        payload = {
            "step": step,
            "prompt": sample.prompt,
            "trace": sample.trace,
            "references": sample.references,
            "reward": sample.reward,
            "advantage": sample.advantage,
            "trace_loss": sample.trace_loss,
            "reference_loss": sample.reference_loss,
        }
        path = Path(self.cfg.sample_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _save_checkpoint(self, step: int) -> None:
        if self.cfg.save_steps <= 0 or step == 0 or step % self.cfg.save_steps != 0:
            return
        checkpoint_dir = Path(self.cfg.output_dir) / f"step-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

    def _compute_losses_for_batch(
        self,
        batch_rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, dict[str, float], BatchDebugSample | None]:
        prompts = [row["prompt"] for row in batch_rows]
        references = [row["references"][: self.cfg.num_reference_samples] for row in batch_rows]
        trace_batch = self._generate_trace_batch(prompts, references)
        grouped_size = len(batch_rows)

        trace_logprob = self._trace_logprobs(trace_batch, self.model)
        prefix_ids_for_refs = [
            prompt_ids + trace_ids + self.tokenizer(self.cfg.answer_prefix, add_special_tokens=False)["input_ids"]
            for prompt_ids, trace_ids in zip(trace_batch.prompt_ids, trace_batch.trace_ids, strict=True)
        ]

        with torch.no_grad():
            reward_outputs = teacher_forced_reference_logps_from_ids(
                model=self.model,
                tokenizer=self.tokenizer,
                prefix_ids_per_sample=prefix_ids_for_refs,
                references=trace_batch.references,
                max_reference_length=self.cfg.max_reference_length,
                length_normalization=self.cfg.reference_length_normalization,
            )
        ref_outputs = teacher_forced_reference_logps_from_ids(
            model=self.model,
            tokenizer=self.tokenizer,
            prefix_ids_per_sample=prefix_ids_for_refs,
            references=trace_batch.references,
            max_reference_length=self.cfg.max_reference_length,
            length_normalization=self.cfg.reference_length_normalization,
        )

        if self.cfg.objective_mode == "log_mass_surrogate":
            grouped_scores_for_reward = reward_outputs.log_mass_normalized.view(grouped_size, self.cfg.num_generations)
        else:
            grouped_scores_for_reward = reward_outputs.log_mass_raw.view(grouped_size, self.cfg.num_generations)

        if self.cfg.reward_transform == "log_mass":
            rewards = grouped_scores_for_reward.reshape(-1)
        else:
            centered = grouped_scores_for_reward - grouped_scores_for_reward.max(dim=1, keepdim=True).values
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
            reference_loss = -ref_outputs.log_mass_normalized.mean()
        else:
            grouped_raw = ref_outputs.log_mass_raw.view(grouped_size, self.cfg.num_generations)
            stabilized = torch.exp(grouped_raw - grouped_raw.detach().max(dim=1, keepdim=True).values)
            reference_loss = -stabilized.mean()

        kl_term = self._compute_kl(trace_batch)
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
        sample: BatchDebugSample | None = None
        if batch_rows:
            sample = BatchDebugSample(
                prompt=batch_rows[0]["prompt"],
                trace=trace_batch.trace_texts[0] if trace_batch.trace_texts else "",
                references=trace_batch.references[0] if trace_batch.references else [],
                reward=float(rewards_for_adv[0].item()) if rewards_for_adv.numel() else 0.0,
                advantage=float(advantages[0].item()) if advantages.numel() else 0.0,
                trace_loss=float(trace_loss.detach().item()),
                reference_loss=float(reference_loss.detach().item()),
            )
        return loss, metrics, sample

    def train(self, raw_train_dataset: Dataset, raw_eval_dataset: Dataset) -> dict[str, Any]:
        train_dataset = prepare_mrvf_dataset(raw_train_dataset, max_reference_samples=self.cfg.num_reference_samples)
        _ = prepare_mrvf_dataset(raw_eval_dataset, max_reference_samples=self.cfg.num_reference_samples)
        train_rows = [dict(item) for item in train_dataset]
        if not train_rows:
            msg = "Prepared training dataset is empty."
            raise RuntimeError(msg)

        batch_size = max(1, self.cfg.per_device_train_batch_size)
        accum = max(1, self.cfg.gradient_accumulation_steps)
        self.optimizer.zero_grad(set_to_none=True)
        global_step = 0
        micro_step = 0
        history: list[dict[str, float]] = []
        self._current_step = 0
        pending_sample: BatchDebugSample | None = None

        while global_step < self.cfg.max_steps:
            self.random.shuffle(train_rows)
            for start in range(0, len(train_rows), batch_size):
                batch = train_rows[start : start + batch_size]
                if not batch:
                    continue
                loss, metrics, sample = self._compute_losses_for_batch(batch)
                (loss / accum).backward()
                history.append(metrics)
                pending_sample = sample
                micro_step += 1
                if micro_step % accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    self._current_step = global_step
                    self._save_checkpoint(global_step)
                    self._append_sample_log(step=global_step, sample=pending_sample)
                    if global_step >= self.cfg.max_steps:
                        break

            if micro_step % accum != 0 and global_step < self.cfg.max_steps:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                global_step += 1
                self._current_step = global_step
                self._save_checkpoint(global_step)
                self._append_sample_log(step=global_step, sample=pending_sample)

        self.model.save_pretrained(self.cfg.output_dir)
        self.tokenizer.save_pretrained(self.cfg.output_dir)
        return {
            "config": asdict(self.cfg),
            "steps": global_step,
            "last_metrics": history[-1] if history else {},
            "mean_loss": float(sum(item["loss"] for item in history) / max(len(history), 1)),
        }

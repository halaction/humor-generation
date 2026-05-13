import types
from pathlib import Path
from typing import Any

import pytest

from src.training.config import MRVFConfig

torch = pytest.importorskip("torch")
from src.training.mrvf_trainer import MRVFTrainer


class DummyTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    padding_side = "left"

    def __call__(
        self,
        text,
        *,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        padding: bool = False,
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        del add_special_tokens
        if isinstance(text, list):
            rows = [self._encode(item, truncation=truncation, max_length=max_length) for item in text]
            max_len = max(len(row) for row in rows)
            padded: list[list[int]] = []
            masks: list[list[int]] = []
            for row in rows:
                pad = max_len - len(row)
                padded.append([self.pad_token_id] * pad + row)
                masks.append([0] * pad + [1] * len(row))
            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor(padded, dtype=torch.long),
                    "attention_mask": torch.tensor(masks, dtype=torch.long),
                }
            return {"input_ids": padded, "attention_mask": masks}
        return {"input_ids": self._encode(text, truncation=truncation, max_length=max_length)}

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(x) for x in ids)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> str:
        del tokenize, add_generation_prompt, enable_thinking
        return "<chat>" + messages[0]["content"] + "</chat>"

    @staticmethod
    def _encode(text: str, *, truncation: bool, max_length: int | None) -> list[int]:
        ids = [(ord(ch) % 19) + 1 for ch in text]
        if truncation and max_length is not None:
            return ids[:max_length]
        return ids


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 64) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 16)
        self.head = torch.nn.Linear(16, vocab_size)
        self.last_generate_input_width: int | None = None
        self.append_tokens: list[int] = [7, 8]

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        hidden = self.embed(input_ids)
        logits = self.head(hidden)
        return types.SimpleNamespace(logits=logits)

    def generate(self, *, input_ids, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        self.last_generate_input_width = input_ids.shape[1]
        append = torch.tensor(
            [self.append_tokens for _ in range(input_ids.shape[0])],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, append], dim=1)

    def save_pretrained(self, save_directory):
        Path(save_directory).mkdir(parents=True, exist_ok=True)


class DummyScheduler:
    def step(self) -> None:
        return None

    def get_last_lr(self) -> list[float]:
        return [1e-3]


def _build_trainer(tmp_path: Path, *, num_generations: int = 2) -> MRVFTrainer:
    cfg = MRVFConfig(
        model_name_or_path="dummy",
        output_dir=str(tmp_path / "ckpt"),
        use_kl=False,
        beta=0.0,
        num_generations=num_generations,
        max_trace_length=8,
        max_reference_length=16,
        num_reference_samples=2,
        objective_mode="log_mass_surrogate",
        reward_transform="log_mass",
        reference_length_normalization="token_mean",
        logging_steps=0,
        save_steps=0,
    )
    cfg.validate()

    trainer = object.__new__(MRVFTrainer)
    trainer.cfg = cfg
    trainer.device = torch.device("cpu")
    trainer.tokenizer = DummyTokenizer()
    trainer.model = DummyModel().to(trainer.device)
    trainer.model.train()
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
    trainer.scheduler = DummyScheduler()
    trainer.random = __import__("random").Random(0)
    trainer._current_step = 0
    trainer._wandb_run = None
    return trainer


def test_generate_trace_batch_uses_generated_suffix_only(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path, num_generations=2)
    prompts = ["short", "this prompt is longer"]
    references = [["r1", "r2"], ["r3", "r4"]]
    trace_batch = trainer._generate_trace_batch(prompts, references)

    assert len(trace_batch.trace_ids) == 4
    assert all(ids == [7, 8] for ids in trace_batch.trace_ids)
    assert len(trace_batch.prompt_ids) == 4
    assert len(trace_batch.prompt_ids[0]) != len(trace_batch.prompt_ids[2])
    assert trace_batch.references[0] == ["r1", "r2"]
    assert trace_batch.references[2] == ["r3", "r4"]


def test_compute_losses_for_batch_backward_has_gradients(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path, num_generations=2)
    batch_rows = [
        {
            "prompt": "Write a joke about cats",
            "references": ["joke one", "joke two"],
        }
    ]
    loss, metrics, sample = trainer._compute_losses_for_batch(batch_rows)
    assert torch.isfinite(loss)
    assert metrics["kl"] == 0.0
    assert "advantage_abs_mean" in metrics
    assert "reward_group_std_mean" in metrics
    assert metrics["trace_token_length_mean"] == 2.0
    assert metrics["trace_token_length_max"] == 2.0
    assert metrics["trace_truncated_fraction"] == 0.0
    assert metrics["closed_think_fraction"] == 0.0
    assert metrics["empty_trace_fraction"] == 0.0
    assert metrics["forced_think_close_fraction"] == 0.0
    assert "effective_reference_prefix_length_mean" in metrics
    assert "effective_reference_prefix_length_max" in metrics
    assert sample is not None
    assert sample.prompt == batch_rows[0]["prompt"]
    assert len(sample.references) == 2
    assert len(sample.traces) == 2
    assert all(trace.trace == "7 8" for trace in sample.traces)
    assert all(trace.trace_token_length == 2 for trace in sample.traces)
    assert all(not trace.is_truncated for trace in sample.traces)
    assert all(not trace.forced_think_close for trace in sample.traces)
    assert all(trace.reference_prefix_length > trace.trace_token_length for trace in sample.traces)
    assert all(isinstance(trace.reward, float) for trace in sample.traces)
    assert all(isinstance(trace.advantage, float) for trace in sample.traces)

    trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = [param.grad for param in trainer.model.parameters() if param.grad is not None]
    assert grads, "Expected non-empty gradient list."
    nonzero = any(torch.any(grad != 0).item() for grad in grads)
    assert nonzero, "Expected at least one nonzero gradient."


def test_group_sample_log_includes_all_generations(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path, num_generations=2)
    trainer.cfg.logging_steps = 1
    trainer.cfg.sample_log_path = str(tmp_path / "samples.jsonl")
    batch_rows = [
        {
            "prompt": "Write a joke about cats",
            "references": ["joke one", "joke two"],
        }
    ]
    _, _, sample = trainer._compute_losses_for_batch(batch_rows)
    trainer._append_sample_log(step=1, sample=sample)

    import json

    rows = [json.loads(line) for line in Path(trainer.cfg.sample_log_path).read_text().splitlines()]
    assert len(rows) == 1
    assert "trace" not in rows[0]
    assert len(rows[0]["traces"]) == 2
    assert {trace["trace_index"] for trace in rows[0]["traces"]} == {0, 1}


def test_trace_prompt_uses_jinja_template_before_qwen_chat(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path, num_generations=2)
    trainer.cfg.trace_format = "qwen_chat_thinking"
    prompt = "Write a joke using the following keywords: banana"

    rendered = trainer._build_trace_prompt_text(prompt)

    assert rendered.startswith("<chat>")
    assert prompt in rendered
    assert "Think freely before answering" in rendered
    assert "Do not force a fixed structure" in rendered
    assert trainer.cfg.trace_instruction not in rendered


def test_forced_think_close_only_affects_reference_prefix(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path, num_generations=2)
    trainer.cfg.force_close_thinking = True
    prompts = ["Write a joke about cats"]
    references = [["joke one", "joke two"]]

    trace_batch = trainer._generate_trace_batch(prompts, references)

    assert all(ids == [7, 8] for ids in trace_batch.trace_ids)
    assert trace_batch.forced_think_close == [True, True]
    suffix_ids = trainer.tokenizer(trainer.cfg.forced_thinking_suffix, add_special_tokens=False)["input_ids"]
    answer_prefix_ids = trainer.tokenizer(trainer.cfg.answer_prefix, add_special_tokens=False)["input_ids"]
    assert suffix_ids
    assert all(
        prefix[-(len(suffix_ids) + len(answer_prefix_ids)) : -len(answer_prefix_ids)] == suffix_ids
        for prefix in trace_batch.reference_prefix_ids
    )
    assert all(
        len(prefix) > len(prompt_ids) + len(trace_ids)
        for prefix, prompt_ids, trace_ids in zip(
            trace_batch.reference_prefix_ids,
            trace_batch.prompt_ids,
            trace_batch.trace_ids,
            strict=True,
        )
    )


def test_forced_think_close_not_added_after_sampled_close(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path, num_generations=2)
    trainer.cfg.force_close_thinking = True
    close_ids = trainer.tokenizer("</think>", add_special_tokens=False)["input_ids"]
    trainer.model.append_tokens = [7, *close_ids, 8]
    prompts = ["Write a joke about cats"]
    references = [["joke one", "joke two"]]

    trace_batch = trainer._generate_trace_batch(prompts, references)

    expected_trace = [7, *close_ids]
    assert all(ids == expected_trace for ids in trace_batch.trace_ids)
    assert trace_batch.forced_think_close == [False, False]
    suffix_ids = trainer.tokenizer(trainer.cfg.forced_thinking_suffix, add_special_tokens=False)["input_ids"]
    assert all(
        suffix_ids != prefix[-(len(suffix_ids) + len(trainer.tokenizer(trainer.cfg.answer_prefix, add_special_tokens=False)["input_ids"])) : -len(trainer.tokenizer(trainer.cfg.answer_prefix, add_special_tokens=False)["input_ids"])]
        for prefix in trace_batch.reference_prefix_ids
    )

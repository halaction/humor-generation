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

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        hidden = self.embed(input_ids)
        logits = self.head(hidden)
        return types.SimpleNamespace(logits=logits)

    def generate(self, *, input_ids, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        self.last_generate_input_width = input_ids.shape[1]
        append = torch.full((input_ids.shape[0], 2), 7, dtype=input_ids.dtype, device=input_ids.device)
        append[:, 1] = 8
        return torch.cat([input_ids, append], dim=1)

    def save_pretrained(self, save_directory):
        Path(save_directory).mkdir(parents=True, exist_ok=True)


class DummyScheduler:
    def step(self) -> None:
        return None


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
    assert sample is not None
    assert sample.prompt == batch_rows[0]["prompt"]
    assert len(sample.references) == 2
    assert isinstance(sample.reward, float)
    assert isinstance(sample.advantage, float)

    trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = [param.grad for param in trainer.model.parameters() if param.grad is not None]
    assert grads, "Expected non-empty gradient list."
    nonzero = any(torch.any(grad != 0).item() for grad in grads)
    assert nonzero, "Expected at least one nonzero gradient."

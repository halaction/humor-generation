import types

import pytest

from src.training.reference_likelihood import teacher_forced_reference_logps

torch = pytest.importorskip("torch")


class DummyTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        truncation=False,
        max_length=None,
        padding=False,
        return_tensors=None,
    ):
        del add_special_tokens
        del padding
        if isinstance(text, list):
            ids = [self._encode(item, truncation=truncation, max_length=max_length) for item in text]
            max_len = max(len(item) for item in ids)
            padded = [item + [self.pad_token_id] * (max_len - len(item)) for item in ids]
            attn = [[1] * len(item) + [0] * (max_len - len(item)) for item in ids]
            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor(padded, dtype=torch.long),
                    "attention_mask": torch.tensor(attn, dtype=torch.long),
                }
            return {"input_ids": padded, "attention_mask": attn}
        return {"input_ids": self._encode(text, truncation=truncation, max_length=max_length)}

    @staticmethod
    def _encode(text: str, *, truncation: bool, max_length: int | None) -> list[int]:
        ids = [(ord(char) % 11) + 1 for char in text]
        if truncation and max_length is not None:
            return ids[:max_length]
        return ids


class DummyLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 16) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 8)
        self.head = torch.nn.Linear(8, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        hidden = self.embed(input_ids)
        logits = self.head(hidden)
        return types.SimpleNamespace(logits=logits)


def test_reference_likelihood_shapes_and_finite() -> None:
    model = DummyLM()
    tokenizer = DummyTokenizer()
    output = teacher_forced_reference_logps(
        model=model,
        tokenizer=tokenizer,
        prompt_texts=["p1", "p2"],
        trace_texts=["t1", "t2"],
        references=[["r1", "r2"], ["r3"]],
        max_reference_length=8,
        answer_prefix="\nA:\n",
        length_normalization="token_mean",
    )
    assert output.ref_logps.shape == (2, 2)
    assert output.ref_lengths.shape == (2, 2)
    assert output.log_mass.shape == (2,)
    assert torch.isfinite(output.log_mass).all()

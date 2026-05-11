from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizerBase


@dataclass
class ReferenceLikelihoodOutput:
    ref_logps: "torch.Tensor"
    ref_lengths: "torch.Tensor"
    log_mass: "torch.Tensor"


def _require_torch() -> "torch":
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        msg = "Torch is required for MRVF reference likelihood computation."
        raise RuntimeError(msg) from error
    return torch


def _normalize_sequence_logps(logps: "torch.Tensor", lengths: "torch.Tensor", mode: str) -> "torch.Tensor":
    torch = _require_torch()
    lengths = lengths.clamp_min(1)
    if mode == "token_mean":
        return logps / lengths
    if mode == "sqrt":
        return logps / torch.sqrt(lengths)
    return logps


def teacher_forced_reference_logps(
    *,
    model: "torch.nn.Module",
    tokenizer: "PreTrainedTokenizerBase",
    prompt_texts: list[str],
    trace_texts: list[str],
    references: list[list[str]],
    max_reference_length: int,
    answer_prefix: str,
    length_normalization: str,
) -> ReferenceLikelihoodOutput:
    torch = _require_torch()
    if len(prompt_texts) != len(trace_texts) or len(prompt_texts) != len(references):
        msg = "`prompt_texts`, `trace_texts`, and `references` must have the same length."
        raise ValueError(msg)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    flat_sequences: list[list[int]] = []
    flat_masks: list[list[int]] = []
    counts: list[int] = []
    for prompt, trace, prompt_refs in zip(prompt_texts, trace_texts, references, strict=True):
        prefix_text = f"{prompt}{trace}{answer_prefix}"
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        per_prompt_count = 0
        for reference in prompt_refs:
            ref_ids = tokenizer(
                reference,
                add_special_tokens=False,
                truncation=True,
                max_length=max_reference_length,
            )["input_ids"]
            if not ref_ids:
                continue
            input_ids = prefix_ids + ref_ids
            ref_mask = [0] * len(prefix_ids) + [1] * len(ref_ids)
            flat_sequences.append(input_ids)
            flat_masks.append(ref_mask)
            per_prompt_count += 1
        counts.append(per_prompt_count)

    if not flat_sequences:
        zeros = torch.zeros(len(prompt_texts), device=next(model.parameters()).device)
        return ReferenceLikelihoodOutput(
            ref_logps=zeros.unsqueeze(-1),
            ref_lengths=zeros.unsqueeze(-1),
            log_mass=zeros,
        )

    max_len = max(len(item) for item in flat_sequences)
    padded_ids = []
    padded_mask = []
    attention = []
    for ids, mask in zip(flat_sequences, flat_masks, strict=True):
        pad_size = max_len - len(ids)
        padded_ids.append(ids + [tokenizer.pad_token_id] * pad_size)
        padded_mask.append(mask + [0] * pad_size)
        attention.append([1] * len(ids) + [0] * pad_size)

    device = next(model.parameters()).device
    input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
    ref_mask = torch.tensor(padded_mask, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention, dtype=torch.long, device=device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    shifted_ref_mask = ref_mask[:, 1:].to(logits.dtype)

    token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    seq_logps = (token_logprobs * shifted_ref_mask).sum(dim=-1)
    seq_lengths = shifted_ref_mask.sum(dim=-1).clamp_min(1)
    seq_scores = _normalize_sequence_logps(seq_logps, seq_lengths, length_normalization)

    per_prompt_scores: list[torch.Tensor] = []
    per_prompt_lengths: list[torch.Tensor] = []
    start = 0
    max_refs = max(max(counts), 1)
    for count in counts:
        if count == 0:
            per_prompt_scores.append(torch.full((max_refs,), float("-inf"), device=device))
            per_prompt_lengths.append(torch.zeros((max_refs,), device=device))
            continue
        chunk_scores = seq_scores[start : start + count]
        chunk_lengths = seq_lengths[start : start + count]
        start += count
        if count < max_refs:
            pad = max_refs - count
            chunk_scores = torch.cat([chunk_scores, torch.full((pad,), float("-inf"), device=device)], dim=0)
            chunk_lengths = torch.cat([chunk_lengths, torch.zeros((pad,), device=device)], dim=0)
        per_prompt_scores.append(chunk_scores)
        per_prompt_lengths.append(chunk_lengths)

    ref_logps = torch.stack(per_prompt_scores, dim=0)
    ref_lengths = torch.stack(per_prompt_lengths, dim=0)
    log_mass = torch.logsumexp(ref_logps, dim=-1)
    return ReferenceLikelihoodOutput(
        ref_logps=ref_logps,
        ref_lengths=ref_lengths,
        log_mass=log_mass,
    )

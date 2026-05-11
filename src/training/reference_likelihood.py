from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizerBase


@dataclass
class ReferenceLikelihoodOutput:
    ref_logps_raw: "torch.Tensor"
    ref_logps_normalized: "torch.Tensor"
    ref_lengths: "torch.Tensor"
    log_mass_raw: "torch.Tensor"
    log_mass_normalized: "torch.Tensor"


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


def teacher_forced_reference_logps_from_ids(
    *,
    model: "torch.nn.Module",
    tokenizer: "PreTrainedTokenizerBase",
    prefix_ids_per_sample: list[list[int]],
    references: list[list[str]],
    max_reference_length: int,
    length_normalization: str,
) -> ReferenceLikelihoodOutput:
    torch = _require_torch()
    if len(prefix_ids_per_sample) != len(references):
        msg = "`prefix_ids_per_sample` and `references` must have the same length."
        raise ValueError(msg)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    flat_sequences: list[list[int]] = []
    flat_masks: list[list[int]] = []
    counts: list[int] = []
    for prefix_ids, prompt_refs in zip(prefix_ids_per_sample, references, strict=True):
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
        zeros = torch.zeros(len(prefix_ids_per_sample), device=next(model.parameters()).device)
        return ReferenceLikelihoodOutput(
            ref_logps_raw=zeros.unsqueeze(-1),
            ref_logps_normalized=zeros.unsqueeze(-1),
            ref_lengths=zeros.unsqueeze(-1),
            log_mass_raw=zeros,
            log_mass_normalized=zeros,
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
    shifted_attn_mask = attention_mask[:, 1:].to(logits.dtype)
    target_mask = shifted_ref_mask * shifted_attn_mask

    token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    seq_logps_raw = (token_logprobs * target_mask).sum(dim=-1)
    seq_lengths = target_mask.sum(dim=-1).clamp_min(1)
    seq_logps_norm = _normalize_sequence_logps(seq_logps_raw, seq_lengths, length_normalization)

    per_prompt_raw: list[torch.Tensor] = []
    per_prompt_norm: list[torch.Tensor] = []
    per_prompt_len: list[torch.Tensor] = []
    start = 0
    max_refs = max(max(counts), 1)
    for count in counts:
        if count == 0:
            per_prompt_raw.append(torch.full((max_refs,), float("-inf"), device=device))
            per_prompt_norm.append(torch.full((max_refs,), float("-inf"), device=device))
            per_prompt_len.append(torch.zeros((max_refs,), device=device))
            continue
        chunk_raw = seq_logps_raw[start : start + count]
        chunk_norm = seq_logps_norm[start : start + count]
        chunk_len = seq_lengths[start : start + count]
        start += count
        if count < max_refs:
            pad = max_refs - count
            pad_vals = torch.full((pad,), float("-inf"), device=device)
            chunk_raw = torch.cat([chunk_raw, pad_vals], dim=0)
            chunk_norm = torch.cat([chunk_norm, pad_vals], dim=0)
            chunk_len = torch.cat([chunk_len, torch.zeros((pad,), device=device)], dim=0)
        per_prompt_raw.append(chunk_raw)
        per_prompt_norm.append(chunk_norm)
        per_prompt_len.append(chunk_len)

    ref_logps_raw = torch.stack(per_prompt_raw, dim=0)
    ref_logps_normalized = torch.stack(per_prompt_norm, dim=0)
    ref_lengths = torch.stack(per_prompt_len, dim=0)
    log_mass_raw = torch.logsumexp(ref_logps_raw, dim=-1)
    log_mass_normalized = torch.logsumexp(ref_logps_normalized, dim=-1)
    return ReferenceLikelihoodOutput(
        ref_logps_raw=ref_logps_raw,
        ref_logps_normalized=ref_logps_normalized,
        ref_lengths=ref_lengths,
        log_mass_raw=log_mass_raw,
        log_mass_normalized=log_mass_normalized,
    )


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
    prefixes = []
    for prompt, trace in zip(prompt_texts, trace_texts, strict=True):
        prefix_text = f"{prompt}{trace}{answer_prefix}"
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        prefixes.append(prefix_ids)
    return teacher_forced_reference_logps_from_ids(
        model=model,
        tokenizer=tokenizer,
        prefix_ids_per_sample=prefixes,
        references=references,
        max_reference_length=max_reference_length,
        length_normalization=length_normalization,
    )

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import config
from src.models import CandidateOutput
from src.paths import DATA_DIR
from src.training.data import prepare_mrvf_dataset


def _resolve_dtype(name: str) -> torch.dtype | str | None:
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return None


def _default_label(model_or_checkpoint: str) -> str:
    return model_or_checkpoint.strip("/").replace("/", "__")


WRAPPER_PATTERNS = (
    re.compile(r"^\s*sure[!,]?\s*", flags=re.IGNORECASE),
    re.compile(
        r"^\s*here(?:'s| is)\s+(?:a|the)\s+joke(?:\s+using(?:\s+the)?\s+(?:keywords?|words?)?.*?)?:\s*",
        flags=re.IGNORECASE | re.DOTALL,
    ),
)
SUFFIX_PATTERNS = (
    re.compile(r"\s*let me know if.*$", flags=re.IGNORECASE | re.DOTALL),
    re.compile(r"\s*\(?note:\s*.*$", flags=re.IGNORECASE | re.DOTALL),
)
DEFAULT_FORCED_THINKING_SUFFIX = (
    "\n\nConsidering the limited time, I will now answer from this reasoning.\n</think>\n\n"
)


def _has_unclosed_think(text: str) -> bool:
    lowered = text.lower()
    last_open = lowered.rfind("<think>")
    if last_open == -1:
        return False
    last_close = lowered.rfind("</think>")
    return last_close < last_open


def _strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    last_open = cleaned.lower().rfind("<think>")
    if last_open != -1:
        cleaned = cleaned[:last_open]
    return cleaned.strip()


def _clean_candidate_text(text: str, *, strip_thinking: bool) -> str:
    cleaned = _strip_thinking(text) if strip_thinking else text.strip()
    for pattern in WRAPPER_PATTERNS:
        cleaned = pattern.sub("", cleaned).strip()
    for pattern in SUFFIX_PATTERNS:
        cleaned = pattern.sub("", cleaned).strip()
    return cleaned


def _nonpad_ids(ids: list[int], pad_token_id: int | None) -> list[int]:
    if pad_token_id is None:
        return ids
    return [token_id for token_id in ids if token_id != pad_token_id]


def _candidate_quality_summary(rows: list[CandidateOutput]) -> dict[str, int | float]:
    texts = [row.text or "" for row in rows]
    lengths = [len(text) for text in texts]
    lowered = [text.lower() for text in texts]
    wrapper_regex = re.compile(r"\b(sure[!,]?|here(?:'s| is).{0,80}joke|let me know)\b", flags=re.IGNORECASE)
    return {
        "rows": len(rows),
        "empty_text_count": sum(length == 0 for length in lengths),
        "text_length_min": min(lengths) if lengths else 0,
        "text_length_mean": float(sum(lengths) / len(lengths)) if lengths else 0.0,
        "text_length_max": max(lengths) if lengths else 0,
        "contains_think_count": sum("<think>" in text for text in lowered),
        "contains_wrapper_count": sum(bool(wrapper_regex.search(text)) for text in texts),
        "contains_note_count": sum("note:" in text for text in lowered),
    }


def _load_model(model_or_checkpoint: str, torch_dtype: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_or_checkpoint, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = _resolve_dtype(torch_dtype)
    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    try:
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(model_or_checkpoint, **kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_or_checkpoint, **kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, tokenizer


def _is_lora_checkpoint(path_or_id: str) -> bool:
    return (Path(path_or_id) / "adapter_config.json").exists()


def _read_lora_base_model(path_or_id: str) -> str | None:
    adapter_config_path = Path(path_or_id) / "adapter_config.json"
    if not adapter_config_path.exists():
        return None
    try:
        data = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    base_model = data.get("base_model_name_or_path")
    return base_model if isinstance(base_model, str) and base_model else None


def _read_lora_rank(path_or_id: str) -> int | None:
    adapter_config_path = Path(path_or_id) / "adapter_config.json"
    if not adapter_config_path.exists():
        return None
    try:
        data = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    rank = data.get("r")
    return int(rank) if isinstance(rank, int) and rank > 0 else None


def _load_vllm(
    *,
    model_or_checkpoint: str,
    base_model_override: str | None,
    tokenizer_name: str,
    torch_dtype: str,
    max_model_len: int | None,
    gpu_memory_utilization: float,
    max_lora_rank: int | None,
) -> tuple[Any, Any, Any | None]:
    try:
        from vllm import LLM
        from vllm.lora.request import LoRARequest
    except ImportError as error:  # pragma: no cover
        msg = "`--generation-backend=vllm` requires the `vllm` package."
        raise RuntimeError(msg) from error

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    lora_request = None
    model_name = model_or_checkpoint
    enable_lora = _is_lora_checkpoint(model_or_checkpoint)
    if enable_lora:
        base_model = base_model_override or _read_lora_base_model(model_or_checkpoint)
        if base_model is None:
            msg = (
                "`--generation-backend=vllm` with a LoRA checkpoint requires adapter_config.json "
                "with base_model_name_or_path, or pass a full base-model path as --vllm-base-model."
            )
            raise ValueError(msg)
        model_name = base_model
        adapter_rank = _read_lora_rank(model_or_checkpoint)
        if max_lora_rank is None:
            max_lora_rank = adapter_rank
        lora_request = LoRARequest("candidate_adapter", 1, model_or_checkpoint)

    dtype = torch_dtype if torch_dtype != "auto" else "auto"
    kwargs: dict[str, Any] = {
        "model": model_name,
        "tokenizer": tokenizer_name,
        "dtype": dtype,
        "enable_lora": enable_lora,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    if max_lora_rank is not None:
        kwargs["max_lora_rank"] = max_lora_rank

    llm = LLM(**kwargs)
    return llm, tokenizer, lora_request


def _format_prompts(tokenizer: Any, prompts: list[str], use_chat_template: bool, enable_thinking: bool) -> list[str]:
    if not use_chat_template:
        return prompts

    formatted = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            )
        except TypeError:
            formatted.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return formatted


def _vllm_generate_texts(
    *,
    llm: Any,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    lora_request: Any | None,
) -> list[str]:
    try:
        from vllm import SamplingParams
    except ImportError as error:  # pragma: no cover
        msg = "`--generation-backend=vllm` requires the `vllm` package."
        raise RuntimeError(msg) from error

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    if lora_request is None:
        outputs = llm.generate(prompts, sampling_params=sampling_params)
    else:
        outputs = llm.generate(prompts, sampling_params=sampling_params, lora_request=lora_request)
    return [output.outputs[0].text for output in outputs]


def _write_candidates(rows: list[CandidateOutput], output_dir: Path, shard_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("model", pa.string()),
            pa.field("text", pa.string()),
        ]
    )
    for idx, start in enumerate(range(0, len(rows), shard_size)):
        chunk = [row.model_dump() for row in rows[start : start + shard_size]]
        table = pa.Table.from_pylist(chunk, schema=schema)
        pq.write_table(table, output_dir / f"part-{idx:04d}.parquet", compression="zstd")


def generate_candidates(args: argparse.Namespace) -> Path:
    model_label = args.model_label or _default_label(args.model_or_checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR / config.candidates.hf_config_name
    split_dir = output_dir / model_label / args.split

    raw_dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=args.split)
    dataset = prepare_mrvf_dataset(raw_dataset, max_reference_samples=1)
    if args.limit is not None:
        dataset = Dataset.from_list([dict(row) for row in dataset.select(range(min(args.limit, len(dataset))))])

    if args.generation_backend == "vllm":
        tokenizer_name = args.vllm_base_model or _read_lora_base_model(args.model_or_checkpoint) or args.model_or_checkpoint
        model, tokenizer, lora_request = _load_vllm(
            model_or_checkpoint=args.model_or_checkpoint,
            base_model_override=args.vllm_base_model,
            tokenizer_name=tokenizer_name,
            torch_dtype=args.torch_dtype,
            max_model_len=args.vllm_max_model_len,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_lora_rank=args.vllm_max_lora_rank,
        )
        device = None
    else:
        model, tokenizer = _load_model(args.model_or_checkpoint, torch_dtype=args.torch_dtype)
        lora_request = None
        device = next(model.parameters()).device

    rows: list[CandidateOutput] = []
    for start in tqdm(range(0, len(dataset), args.batch_size), desc=f"Generate {model_label} candidates"):
        batch = [dict(row) for row in dataset.select(range(start, min(start + args.batch_size, len(dataset))))]
        prompt_texts = _format_prompts(
            tokenizer=tokenizer,
            prompts=[row["prompt"] for row in batch],
            use_chat_template=args.use_chat_template,
            enable_thinking=args.enable_thinking,
        )
        if args.generation_backend == "vllm":
            texts = _vllm_generate_texts(
                llm=model,
                prompts=prompt_texts,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                lora_request=lora_request,
            )
            completion_ids: list[list[int]] | None = None
            encoded = None
        else:
            encoded = tokenizer(prompt_texts, padding=True, add_special_tokens=False, return_tensors="pt").to(device)
            input_width = encoded["input_ids"].shape[1]
            with torch.no_grad():
                generation_kwargs: dict[str, Any] = {
                    "do_sample": args.temperature > 0,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "pad_token_id": tokenizer.pad_token_id,
                }
                if args.temperature > 0:
                    generation_kwargs["temperature"] = args.temperature
                generated = model.generate(**encoded, **generation_kwargs)
            completions = generated[:, input_width:]
            completion_ids = [
                _nonpad_ids(row.tolist(), tokenizer.pad_token_id)
                for row in completions
            ]
            texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        if args.force_close_thinking:
            suffix_ids = tokenizer(args.forced_thinking_suffix, add_special_tokens=False)["input_ids"]
            needs_continuation = [
                idx
                for idx, text in enumerate(texts)
                if _has_unclosed_think(text)
            ]
            if needs_continuation and args.answer_continuation_max_new_tokens > 0:
                if args.generation_backend == "vllm":
                    continuation_prompts = [
                        prompt_texts[idx] + texts[idx] + args.forced_thinking_suffix
                        for idx in needs_continuation
                    ]
                    continuation_texts = _vllm_generate_texts(
                        llm=model,
                        prompts=continuation_prompts,
                        max_tokens=args.answer_continuation_max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        lora_request=lora_request,
                    )
                    for local_idx, dataset_idx in enumerate(needs_continuation):
                        texts[dataset_idx] = (
                            texts[dataset_idx] + args.forced_thinking_suffix + continuation_texts[local_idx]
                        )
                else:
                    if completion_ids is None or encoded is None:
                        msg = "Transformers continuation requires encoded prompts and completion IDs."
                        raise RuntimeError(msg)
                    prompt_ids = []
                    for row_ids, row_mask in zip(
                        encoded["input_ids"].tolist(),
                        encoded["attention_mask"].tolist(),
                        strict=True,
                    ):
                        prompt_ids.append(
                            [token_id for token_id, mask in zip(row_ids, row_mask, strict=True) if mask == 1]
                        )
                    continuation_inputs = [
                        prompt_ids[idx] + completion_ids[idx] + suffix_ids
                        for idx in needs_continuation
                    ]
                    max_len = max(len(item) for item in continuation_inputs)
                    padded = []
                    attention = []
                    for ids in continuation_inputs:
                        pad = max_len - len(ids)
                        padded.append([tokenizer.pad_token_id] * pad + ids)
                        attention.append([0] * pad + [1] * len(ids))
                    cont_encoded = {
                        "input_ids": torch.tensor(padded, dtype=torch.long, device=device),
                        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
                    }
                    with torch.no_grad():
                        cont_kwargs: dict[str, Any] = {
                            "do_sample": args.temperature > 0,
                            "top_p": args.top_p,
                            "max_new_tokens": args.answer_continuation_max_new_tokens,
                            "pad_token_id": tokenizer.pad_token_id,
                        }
                        if args.temperature > 0:
                            cont_kwargs["temperature"] = args.temperature
                        continued = model.generate(**cont_encoded, **cont_kwargs)
                    cont_ids = continued[:, max_len:]
                    cont_completion_ids = [
                        _nonpad_ids(row.tolist(), tokenizer.pad_token_id)
                        for row in cont_ids
                    ]
                    for local_idx, dataset_idx in enumerate(needs_continuation):
                        completion_ids[dataset_idx] = (
                            completion_ids[dataset_idx] + suffix_ids + cont_completion_ids[local_idx]
                        )
                    texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        for row, text in zip(batch, texts, strict=True):
            cleaned = _clean_candidate_text(text, strip_thinking=args.strip_thinking)
            rows.append(
                CandidateOutput(
                    id=int(row["id"]),
                    keywords=list(row["keywords"]),
                    model=model_label,
                    text=cleaned,
                )
            )

    _write_candidates(rows=rows, output_dir=split_dir, shard_size=args.shard_size)
    print({"candidate_quality": _candidate_quality_summary(rows)})
    return split_dir


def main() -> None:
    load_dotenv(Path(".env"))
    parser = argparse.ArgumentParser(description="Generate candidate jokes from a local HF model or checkpoint.")
    parser.add_argument("--model-or-checkpoint", required=True)
    parser.add_argument("--model-label")
    parser.add_argument("--dataset-name", default="halaction/humor-generation")
    parser.add_argument("--dataset-config-name", default="references")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strip-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-close-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--forced-thinking-suffix", default=DEFAULT_FORCED_THINKING_SUFFIX)
    parser.add_argument("--answer-continuation-max-new-tokens", type=int, default=128)
    parser.add_argument("--generation-backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--vllm-base-model")
    parser.add_argument("--vllm-max-model-len", type=int)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-max-lora-rank", type=int)
    args = parser.parse_args()

    split_dir = generate_candidates(args)
    part_count = sum(1 for _ in split_dir.glob("part-*.parquet"))
    print({"candidates_dir": str(split_dir), "part_count": part_count})


if __name__ == "__main__":
    main()

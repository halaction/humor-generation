from __future__ import annotations

import argparse
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

    model, tokenizer = _load_model(args.model_or_checkpoint, torch_dtype=args.torch_dtype)
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
        texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
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
    args = parser.parse_args()

    split_dir = generate_candidates(args)
    part_count = sum(1 for _ in split_dir.glob("part-*.parquet"))
    print({"candidates_dir": str(split_dir), "part_count": part_count})


if __name__ == "__main__":
    main()

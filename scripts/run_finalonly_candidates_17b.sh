#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source .venv-vllm/bin/activate
set -a
[ -f .env ] && source .env
set +a

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

ROOT="${ROOT:-data/candidates_finalonly_17b_200}"
LIMIT="${LIMIT:-200}"

python -m scripts.generate_checkpoint_candidates \
  --generation-backend vllm \
  --model-or-checkpoint Qwen/Qwen3-1.7B \
  --model-label qwen3-17b-base-nothink-finalonly \
  --output-dir "${ROOT}" \
  --split validation \
  --limit "${LIMIT}" \
  --batch-size 32 \
  --max-new-tokens 48 \
  --torch-dtype bfloat16 \
  --no-enable-thinking \
  --vllm-max-model-len 4096

python -m scripts.generate_checkpoint_candidates \
  --generation-backend vllm \
  --model-or-checkpoint Qwen/Qwen3-1.7B \
  --model-label qwen3-17b-base-think-finalonly \
  --output-dir "${ROOT}" \
  --split validation \
  --limit "${LIMIT}" \
  --batch-size 16 \
  --max-new-tokens 256 \
  --torch-dtype bfloat16 \
  --enable-thinking \
  --force-close-thinking \
  --answer-continuation-max-new-tokens 48 \
  --vllm-max-model-len 4096

python -m scripts.generate_checkpoint_candidates \
  --generation-backend vllm \
  --model-or-checkpoint data/checkpoints/qwen3-17b-free-think-r6-finalonly \
  --vllm-base-model Qwen/Qwen3-1.7B \
  --model-label qwen3-17b-mrvf-r6-think-finalonly \
  --output-dir "${ROOT}" \
  --split validation \
  --limit "${LIMIT}" \
  --batch-size 16 \
  --max-new-tokens 256 \
  --torch-dtype bfloat16 \
  --enable-thinking \
  --force-close-thinking \
  --answer-continuation-max-new-tokens 48 \
  --vllm-max-model-len 4096 \
  --vllm-max-lora-rank 32

python -m scripts.generate_checkpoint_candidates \
  --generation-backend vllm \
  --model-or-checkpoint data/checkpoints/qwen3-17b-free-think-r6-finalonly \
  --vllm-base-model Qwen/Qwen3-1.7B \
  --model-label qwen3-17b-mrvf-r6-nothink-finalonly \
  --output-dir "${ROOT}" \
  --split validation \
  --limit "${LIMIT}" \
  --batch-size 32 \
  --max-new-tokens 48 \
  --torch-dtype bfloat16 \
  --no-enable-thinking \
  --vllm-max-model-len 4096 \
  --vllm-max-lora-rank 32

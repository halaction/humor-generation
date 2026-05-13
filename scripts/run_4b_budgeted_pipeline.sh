#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_NAME="${RUN_NAME:-qwen3-4b-mrvf-free-think-r1-budgeted-384x2}"
HF_MODEL_REPO="${HF_MODEL_REPO:-halaction/Qwen3-4B-humor-lora}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-data/checkpoints/qwen3-4b-free-think-r1-budgeted}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-data/candidates_4b_r1_100}"

echo "[4b-r1] started $(date -Is)"
echo "[4b-r1] run_name=${RUN_NAME}"

python -m scripts.train_mrvf --config configs/models/qwen3-4b-free-think-r1-budgeted.yaml

python -m scripts.generate_checkpoint_candidates \
  --model-or-checkpoint Qwen/Qwen3-4B \
  --model-label qwen3-4b-base-nothink \
  --output-dir "${CANDIDATE_ROOT}" \
  --split validation \
  --limit 100 \
  --batch-size 2 \
  --max-new-tokens 128 \
  --torch-dtype bfloat16 \
  --no-enable-thinking

python -m scripts.generate_checkpoint_candidates \
  --model-or-checkpoint Qwen/Qwen3-4B \
  --model-label qwen3-4b-base-think \
  --output-dir "${CANDIDATE_ROOT}" \
  --split validation \
  --limit 100 \
  --batch-size 1 \
  --max-new-tokens 1536 \
  --torch-dtype bfloat16 \
  --enable-thinking \
  --force-close-thinking \
  --answer-continuation-max-new-tokens 128

python -m scripts.generate_checkpoint_candidates \
  --model-or-checkpoint "${CHECKPOINT_DIR}" \
  --model-label qwen3-4b-mrvf-r1-budgeted-think \
  --output-dir "${CANDIDATE_ROOT}" \
  --split validation \
  --limit 100 \
  --batch-size 1 \
  --max-new-tokens 1536 \
  --torch-dtype bfloat16 \
  --enable-thinking \
  --force-close-thinking \
  --answer-continuation-max-new-tokens 128

python -m scripts.generate_checkpoint_candidates \
  --model-or-checkpoint "${CHECKPOINT_DIR}" \
  --model-label qwen3-4b-mrvf-r1-budgeted-nothink \
  --output-dir "${CANDIDATE_ROOT}" \
  --split validation \
  --limit 100 \
  --batch-size 2 \
  --max-new-tokens 128 \
  --torch-dtype bfloat16 \
  --no-enable-thinking

python -m scripts.upload_checkpoint_to_hf \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --repo-id "${HF_MODEL_REPO}" \
  --path-in-repo "runs/${RUN_NAME}/final" \
  --upload-step-checkpoints

echo "[4b-r1] finished $(date -Is)"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_NAME="${RUN_NAME:-qwen3-17b-mrvf-free-think-r5-budgeted-512x2}"
HF_MODEL_REPO="${HF_MODEL_REPO:-halaction/Qwen3-1.7B-humor-lora}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-data/checkpoints/qwen3-17b-free-think-r5-budgeted}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-data/candidates_r5_200}"
EVAL_DIR="${EVAL_DIR:-data/evaluation/qwen3-17b-free-think-r5-budgeted-validation200}"

echo "[r5] started $(date -Is)"
echo "[r5] run_name=${RUN_NAME}"

python -m scripts.train_mrvf --config configs/models/qwen3-17b-free-think-r5-budgeted.yaml

python -m scripts.generate_checkpoint_candidates \
  --model-or-checkpoint Qwen/Qwen3-1.7B \
  --model-label qwen3-17b-base-nothink-r5eval \
  --output-dir "${CANDIDATE_ROOT}" \
  --split validation \
  --limit 200 \
  --batch-size 4 \
  --max-new-tokens 48 \
  --torch-dtype bfloat16 \
  --no-enable-thinking

python -m scripts.generate_checkpoint_candidates \
  --model-or-checkpoint Qwen/Qwen3-1.7B \
  --model-label qwen3-17b-base-think-r5eval \
  --output-dir "${CANDIDATE_ROOT}" \
  --split validation \
  --limit 200 \
  --batch-size 2 \
  --max-new-tokens 2048 \
  --torch-dtype bfloat16 \
  --enable-thinking \
  --force-close-thinking \
  --answer-continuation-max-new-tokens 48

python -m scripts.generate_checkpoint_candidates \
  --model-or-checkpoint "${CHECKPOINT_DIR}" \
  --model-label qwen3-17b-mrvf-free-think-r5-budgeted \
  --output-dir "${CANDIDATE_ROOT}" \
  --split validation \
  --limit 200 \
  --batch-size 2 \
  --max-new-tokens 2048 \
  --torch-dtype bfloat16 \
  --enable-thinking \
  --force-close-thinking \
  --answer-continuation-max-new-tokens 48

python -m scripts.generate_checkpoint_candidates \
  --model-or-checkpoint "${CHECKPOINT_DIR}" \
  --model-label qwen3-17b-mrvf-free-think-r5-budgeted-nothink \
  --output-dir "${CANDIDATE_ROOT}" \
  --split validation \
  --limit 200 \
  --batch-size 4 \
  --max-new-tokens 48 \
  --torch-dtype bfloat16 \
  --no-enable-thinking

python -m scripts.upload_checkpoint_to_hf \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --repo-id "${HF_MODEL_REPO}" \
  --path-in-repo "runs/${RUN_NAME}/final" \
  --upload-step-checkpoints

python -m scripts.evaluate_candidates \
  --candidate-dir "${CANDIDATE_ROOT}/qwen3-17b-base-nothink-r5eval" \
  --candidate-dir "${CANDIDATE_ROOT}/qwen3-17b-base-think-r5eval" \
  --candidate-dir "${CANDIDATE_ROOT}/qwen3-17b-mrvf-free-think-r5-budgeted" \
  --candidate-dir "${CANDIDATE_ROOT}/qwen3-17b-mrvf-free-think-r5-budgeted-nothink" \
  --split validation \
  --output-dir "${EVAL_DIR}" \
  --no-resume

echo "[r5] finished $(date -Is)"

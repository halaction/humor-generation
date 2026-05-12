#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

RUN_NAME="${RUN_NAME:-qwen3-17b-mrvf-lora-a100-r1}"
HF_MODEL_REPO="${HF_MODEL_REPO:-halaction/Qwen3-1.7B-humor-lora}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-data/checkpoints/qwen3-17b-light}"
MRVF_MODEL_LABEL="${MRVF_MODEL_LABEL:-qwen3-17b-mrvf-light-final}"
SKIP_BASELINES="${SKIP_BASELINES:-0}"

echo "[pipeline] started $(date -Is)"
echo "[pipeline] run_name=${RUN_NAME}"
echo "[pipeline] hf_model_repo=${HF_MODEL_REPO}"

if [[ "${SKIP_BASELINES}" != "1" ]]; then
  python -m scripts.generate_checkpoint_candidates \
    --model-or-checkpoint Qwen/Qwen3-1.7B \
    --model-label qwen3-17b-base-nothink \
    --split validation \
    --batch-size 4 \
    --max-new-tokens 128 \
    --torch-dtype bfloat16 \
    --no-enable-thinking

  python -m scripts.generate_checkpoint_candidates \
    --model-or-checkpoint Qwen/Qwen3-1.7B \
    --model-label qwen3-17b-base-think \
    --split validation \
    --batch-size 4 \
    --max-new-tokens 128 \
    --torch-dtype bfloat16 \
    --enable-thinking
else
  echo "[pipeline] skipping baseline candidate generation"
fi

python -m scripts.train_mrvf --config configs/models/qwen3-17b-hpc.yaml

python -m scripts.generate_checkpoint_candidates \
  --model-or-checkpoint "${CHECKPOINT_DIR}" \
  --model-label "${MRVF_MODEL_LABEL}" \
  --split validation \
  --batch-size 4 \
  --max-new-tokens 128 \
  --torch-dtype bfloat16 \
  --no-enable-thinking

python -m scripts.upload_checkpoint_to_hf \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --repo-id "${HF_MODEL_REPO}" \
  --path-in-repo "runs/${RUN_NAME}/final" \
  --upload-step-checkpoints

echo "[pipeline] finished $(date -Is)"

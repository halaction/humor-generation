#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source .venv-training/bin/activate
set -a
[ -f .env ] && source .env
set +a

if [[ -z "${OPENAI_API_KEY:-}" && -n "${OPENROUTER_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="${OPENROUTER_API_KEY}"
fi
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"

python -m scripts.evaluate_candidates \
  --candidate-dir data/candidates_finalonly_4b_100/qwen3-4b-base-nothink-finalonly \
  --candidate-dir data/candidates_finalonly_4b_100/qwen3-4b-base-think-finalonly \
  --candidate-dir data/candidates_finalonly_4b_100/qwen3-4b-mrvf-r2-think-finalonly \
  --candidate-dir data/candidates_finalonly_4b_100/qwen3-4b-mrvf-r2-nothink-finalonly \
  --split validation \
  --output-dir data/evaluation/qwen3-4b-finalonly-gemini-flash-100 \
  --judge-model google/gemini-2.5-flash \
  --judge-temperature 0 \
  --max-parallel-requests 8 \
  --max-retries 4 \
  --no-resume

#!/bin/bash
#SBATCH --job-name=humor-mrvf
#SBATCH --output=train-%j.log
#SBATCH --error=train-%j.err
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail

module purge
module load Python
source activate humor-mrvf

export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=9.0+PTX
export WANDB_MODE=offline
export HF_HOME=/home/$USER/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p logs
hostname
date
nvidia-smi
which python
python -V
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

python -u scripts/train_mrvf.py --config configs/models/qwen3-17b-hpc.yaml

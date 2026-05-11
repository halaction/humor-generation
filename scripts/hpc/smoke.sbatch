#!/bin/bash
#SBATCH --job-name=humor-mrvf-smoke
#SBATCH --output=smoke-%j.log
#SBATCH --error=smoke-%j.err
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00

set -euo pipefail

module purge
module load Python
source activate humor-mrvf
export WANDB_MODE=offline
export HF_HOME=/home/$USER/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

hostname
date
nvidia-smi
which python
python -V
python -u - <<'PY'
import torch
print("python ok")
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
PY

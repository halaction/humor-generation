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

python -u scripts/train_mrvf.py --config configs/training/mrvf_qwen3_1_7b_h200.yaml

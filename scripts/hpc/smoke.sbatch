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

python -u - <<'PY'
import torch
print("python ok")
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
PY

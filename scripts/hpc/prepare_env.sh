#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-humor-mrvf}"
PYTHON_VERSION="${2:-3.12}"

module purge
module load Python

if conda env list | awk '{print $1}' | grep -q "^${ENV_NAME}$"; then
  echo "Environment ${ENV_NAME} already exists."
else
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

source activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install -r requirements/training-hpc.txt

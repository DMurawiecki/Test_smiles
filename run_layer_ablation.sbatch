#!/bin/bash
#SBATCH --job-name=smiles-ablation
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=80:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=rocky
#SBATCH -G 1
#SBATCH -C "type_e|type_f|type_h"

set -euo pipefail

module load Python
conda activate /home/dmuravetskiy/.conda/envs/daniilenv

cd /home/dmuravetskiy/SMILES/SMILES-2026-Hallucination-Detection-main

mkdir -p logs resultscd

test -f aggregation.py
test -f probe.py
test -f splitting.py
test -f experiments/run_layer_ablation.py
test -f data/dataset.csv
test -f requirements.txt

# Keep Hugging Face caches outside the repository.
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false

# The ablation script switches aggregation modes internally.
export PROBE_TYPE="${PROBE_TYPE:-logreg}"

echo "=== DIAG ==="
echo "job id: ${SLURM_JOB_ID}"
echo "hostname: $(hostname)"
echo "workdir: $(pwd)"
which python
python -V
python -c "import torch; print('torch', torch.__version__, 'cuda_ver', torch.version.cuda, 'cuda_ok', torch.cuda.is_available(), 'n', torch.cuda.device_count())"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<empty>}"
echo "PROBE_TYPE=${PROBE_TYPE}"
echo "HF_HOME=${HF_HOME}"
echo "============"

# Set SKIP_PIP_INSTALL=1 if dependencies are already installed in daniilenv.
if [[ "${SKIP_PIP_INSTALL:-0}" != "1" ]]; then
  python -m pip install -r requirements.txt
fi

python experiments/run_layer_ablation.py

echo "Finished at: $(date)"
echo "Results CSV: results/layer_ablation.csv"

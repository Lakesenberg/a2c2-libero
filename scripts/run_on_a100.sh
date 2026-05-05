#!/usr/bin/env bash
# Submit script for the A100 server inference benchmark.
#
# Usage:
#   sbatch scripts/run_on_a100.sh                    # default: synthetic mode
#   sbatch scripts/run_on_a100.sh dataset             # use real LIBERO obs
#   sbatch scripts/run_on_a100.sh env libero_spatial  # full LIBERO env
#
# Adjust SBATCH directives to your cluster.

#SBATCH --job-name=a2c2_inf_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/a2c2_inf_%j.out
#SBATCH --error=logs/a2c2_inf_%j.err

set -euo pipefail

MODE="${1:-synthetic}"
TASK="${2:-libero_spatial}"
POLICY="${POLICY:-lerobot/smolvla_libero}"
HEAD_CKPT="${HEAD_CKPT:-}"

mkdir -p logs

export MUJOCO_GL=egl
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Activate your venv / conda env here:
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate a2c2

cd "$(dirname "$0")/.."

EXTRA_ARGS=""
if [[ -n "${HEAD_CKPT}" ]]; then
  EXTRA_ARGS="--head-ckpt ${HEAD_CKPT}"
fi

echo "== a2c2 inference test =="
echo "mode=$MODE task=$TASK policy=$POLICY"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

case "$MODE" in
  synthetic)
    python scripts/a100_inference_test.py \
      --mode synthetic \
      --policy-path "$POLICY" \
      --ticks 1000 --tick-dt-ms 5 \
      --output logs/a2c2_synthetic_${SLURM_JOB_ID:-local}.json \
      $EXTRA_ARGS
    ;;
  dataset)
    python scripts/a100_inference_test.py \
      --mode dataset \
      --policy-path "$POLICY" \
      --dataset-repo lerobot/libero_spatial \
      --ticks 500 --tick-dt-ms 5 \
      --output logs/a2c2_dataset_${SLURM_JOB_ID:-local}.json \
      $EXTRA_ARGS
    ;;
  env)
    python scripts/a100_inference_test.py \
      --mode env \
      --task "$TASK" \
      --policy-path "$POLICY" \
      --episodes 5 --max-episode-steps 300 \
      --tick-dt-ms 0 \
      --output logs/a2c2_env_${SLURM_JOB_ID:-local}.json \
      $EXTRA_ARGS
    ;;
  *)
    echo "unknown mode: $MODE"; exit 1;;
esac

echo "done."

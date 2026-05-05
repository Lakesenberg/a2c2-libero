#!/usr/bin/env bash
# Run on the 4090 inference machine. Pulls the latest ckpts from the A100
# server (or HuggingFace) and runs the real-time A2C2 pipeline on a real
# LIBERO env, recording success rate and per-tick latency.
set -euo pipefail

# ============ EDIT THESE ============
HF_USER="${HF_USER:-Lakesenberg}"
SERVER_HOST="${SERVER_HOST:-192.168.3.103}"
SERVER_USER="${SERVER_USER:-lakesenberg}"
SERVER_DIR="${SERVER_DIR:-~/a2c2-libero}"
SUITE="${SUITE:-libero_10}"
EPISODES="${EPISODES:-10}"
# Where to load checkpoints from: "hf" or "server"
CKPT_SOURCE="${CKPT_SOURCE:-hf}"
# ====================================

mkdir -p logs ckpts

echo "================================================================"
echo " 4090 inference machine — A2C2 real-time pipeline"
echo " host:    $(hostname)"
echo " gpu:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo " suite:   ${SUITE}, ${EPISODES} eps"
echo " server:  ${SERVER_USER}@${SERVER_HOST}"
echo " ckpt:    ${CKPT_SOURCE}"
echo "================================================================"

# ----- pull checkpoints -----
SMOLVLA_PATH="${HF_USER}/smolvla_${SUITE}"
A2C2_PATH="${HF_USER}/a2c2_${SUITE}"

if [[ "${CKPT_SOURCE}" == "server" ]]; then
  echo "[ckpt] rsync from A100 server ..."
  rsync -avz --partial \
    "${SERVER_USER}@${SERVER_HOST}:${SERVER_DIR}/outputs/a2c2_head_${SUITE}/checkpoint.pt" \
    "ckpts/a2c2_head_${SUITE}.pt"
  rsync -avz --partial \
    "${SERVER_USER}@${SERVER_HOST}:${SERVER_DIR}/outputs/smolvla_${SUITE}/checkpoint" \
    "ckpts/smolvla_${SUITE}/"
  SMOLVLA_PATH="ckpts/smolvla_${SUITE}"
  HEAD_CKPT="ckpts/a2c2_head_${SUITE}.pt"
else
  echo "[ckpt] using HF: ${SMOLVLA_PATH} + ${A2C2_PATH}"
  HEAD_CKPT=""           # let the script pick up the HF version internally
fi

# ----- env -----
export MUJOCO_GL=egl
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# ----- main: full LIBERO env eval with async A2C2 engine -----
echo "[eval] running full env evaluation ..."
EXTRA=""
if [[ -n "${HEAD_CKPT}" && -f "${HEAD_CKPT}" ]]; then
  EXTRA="--head-ckpt ${HEAD_CKPT}"
fi

python scripts/a100_inference_test.py \
  --mode env \
  --task "${SUITE}" \
  --policy-path "${SMOLVLA_PATH}" \
  --episodes "${EPISODES}" \
  --max-episode-steps 300 \
  --tick-dt-ms 0 \
  --output "logs/4090_eval_${SUITE}.json" \
  ${EXTRA} \
  2>&1 | tee "logs/4090_eval_${SUITE}.log"

echo "================================================================"
echo " 4090 evaluation done."
echo " report: logs/4090_eval_${SUITE}.json"
echo "================================================================"

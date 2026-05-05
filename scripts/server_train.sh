#!/usr/bin/env bash
# Run on 192.168.3.103 (A100 server). Trains SmolVLA + A2C2 head, then
# uploads checkpoints back to HuggingFace. After this finishes, the 4090
# inference machine pulls the ckpts via `scripts/sync_from_server.sh`.
set -euo pipefail

# ============ EDIT THESE ============
HF_USER="${HF_USER:-Lakesenberg}"
SUITE="${SUITE:-libero_10}"               # libero_spatial / object / goal / 10
SMOLVLA_OUTPUT="outputs/smolvla_${SUITE}"
A2C2_OUTPUT="outputs/a2c2_head_${SUITE}"
# ====================================

mkdir -p logs outputs

echo "================================================================"
echo " A100 server — SmolVLA + A2C2 training pipeline"
echo " host:  $(hostname)  ($(hostname -I | awk '{print $1}'))"
echo " gpu:   $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo " suite: ${SUITE}"
echo "================================================================"

# ----- Stage 1: SmolVLA fine-tune (skip if you use lerobot/smolvla_libero) -----
if [[ "${SKIP_SMOLVLA:-0}" != "1" ]]; then
  echo "[Stage 1] Fine-tuning SmolVLA on ${SUITE} ..."
  python src/lerobot/scripts/train.py \
    --policy.type=smolvla \
    --policy.load_vlm_weights=true \
    --dataset.repo_id="lerobot/${SUITE}" \
    --batch_size=64 \
    --steps=80000 \
    --policy.repo_id="${HF_USER}/smolvla_${SUITE}" \
    --output_dir="${SMOLVLA_OUTPUT}" \
    --job_name="smolvla_${SUITE}" \
    --wandb.enable=true \
    2>&1 | tee logs/smolvla_${SUITE}.log
fi

# ----- Stage 2: build residual dataset (â_base relabel) -----
echo "[Stage 2] Creating residual dataset ..."
python eval_libero/create_dataset_for_residualpolicy.py \
  --base_repo_id="lerobot/${SUITE}" \
  --base_policy_path="${HF_USER}/smolvla_${SUITE}" \
  --upload_repo_id="${HF_USER}/${SUITE}-smolvla-residual" \
  2>&1 | tee logs/residual_dataset_${SUITE}.log

# ----- Stage 3: train A2C2 residual head -----
echo "[Stage 3] Training A2C2 residual transformer ..."
python src/lerobot/scripts/train_residual_transformer.py \
  --policy.type=residual_transformer \
  --policy.repo_id="${HF_USER}/a2c2_${SUITE}" \
  --batch_size=64 --num_workers=16 \
  --steps=200000 \
  --dataset.repo_id="${HF_USER}/${SUITE}-smolvla-residual" \
  --output_dir="${A2C2_OUTPUT}" \
  --job_name="a2c2_${SUITE}" \
  --wandb.enable=true \
  2>&1 | tee logs/a2c2_${SUITE}.log

# ----- Stage 4: server-side inference smoke test (latency only) -----
echo "[Stage 4] Server-side inference smoke test ..."
export MUJOCO_GL=egl
python scripts/a100_inference_test.py \
  --mode synthetic \
  --policy-path "${HF_USER}/smolvla_${SUITE}" \
  --head-ckpt "${A2C2_OUTPUT}/checkpoint.pt" \
  --ticks 500 --tick-dt-ms 5 \
  --output "logs/server_smoke_${SUITE}.json"

echo "================================================================"
echo " all stages done."
echo " checkpoints:"
echo "   - SmolVLA:  ${SMOLVLA_OUTPUT}/  (also pushed to HF as ${HF_USER}/smolvla_${SUITE})"
echo "   - A2C2:     ${A2C2_OUTPUT}/     (also pushed to HF as ${HF_USER}/a2c2_${SUITE})"
echo " logs/server_smoke_${SUITE}.json has latency stats"
echo "================================================================"

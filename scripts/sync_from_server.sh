#!/usr/bin/env bash
# Pull artifacts (ckpts + logs) from the A100 server to the 4090 machine.
# Run from the 4090.
set -euo pipefail

SERVER_HOST="${SERVER_HOST:-192.168.3.103}"
SERVER_USER="${SERVER_USER:-lakesenberg}"
SERVER_DIR="${SERVER_DIR:-~/a2c2-libero}"

mkdir -p ckpts logs

echo "[1/2] rsync checkpoints"
rsync -avz --partial \
  "${SERVER_USER}@${SERVER_HOST}:${SERVER_DIR}/outputs/" \
  "ckpts/"

echo "[2/2] rsync logs"
rsync -avz --partial \
  "${SERVER_USER}@${SERVER_HOST}:${SERVER_DIR}/logs/" \
  "logs/server_logs/"

echo "done. local ckpts/ now has SmolVLA + A2C2 head."

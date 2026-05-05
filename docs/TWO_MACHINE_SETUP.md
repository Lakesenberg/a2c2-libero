# Two-machine setup — A100 server (192.168.3.103) + 4090 inference

This documents the canonical workflow when you have:

* **服务器 (server)**: A100 at `192.168.3.103` — heavy compute, training,
  dataset relabel, all artifact creation.
* **推理机 (inference machine)**: workstation with RTX 4090 — real-time
  evaluation on LIBERO, low-latency inference loop.

## One-time setup

### On both machines

```bash
git clone https://github.com/Lakesenberg/a2c2-libero.git
cd a2c2-libero
git switch Inference_test
git submodule update --init --recursive

uv venv -p 3.10 && source .venv/bin/activate
uv pip install -e ".[smolvla,test]"
uv pip install -e third_party/libero
uv pip install mujoco==3.3.2
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=egl
```

### On 4090 only — SSH alias for the server

```bash
cat >> ~/.ssh/config <<EOF

Host a100
  Hostname 192.168.3.103
  User lakesenberg
  IdentityFile ~/.ssh/id_rsa
EOF

ssh a100        # test
```

---

## Daily flow

### ① On the A100 server (192.168.3.103) — train everything

```bash
ssh a100
cd ~/a2c2-libero
chmod +x scripts/server_train.sh
SUITE=libero_10 ./scripts/server_train.sh
```

What it does, in order:

1. Fine-tunes SmolVLA on the chosen suite. Pushes to
   `${HF_USER}/smolvla_${SUITE}`.
2. Generates the residual relabel dataset (`â_base` per frame). Pushes to
   `${HF_USER}/${SUITE}-smolvla-residual`.
3. Trains the A2C2 residual transformer head. Pushes to
   `${HF_USER}/a2c2_${SUITE}`.
4. Runs a synthetic-input latency smoke test on the A100 itself, saves
   `logs/server_smoke_${SUITE}.json`.

Set `SKIP_SMOLVLA=1` if you want to keep using
`lerobot/smolvla_libero` and only retrain the A2C2 head.

### ② On the 4090 — pull and evaluate

Two ways to consume the checkpoints:

```bash
# A) pull from HuggingFace (default; works without 192.168.3.103 reachable)
SUITE=libero_10 ./scripts/inference_4090.sh

# B) pull directly from the server (rsync; faster if you have an internal LAN)
SUITE=libero_10 CKPT_SOURCE=server ./scripts/inference_4090.sh
```

What it does:

1. Pulls SmolVLA + A2C2 head ckpts (HF or rsync).
2. Runs `scripts/a100_inference_test.py --mode env` with the async A2C2
   engine on a real LIBERO env.
3. Records success rate + per-tick latency to `logs/4090_eval_${SUITE}.json`.

### Optional: just sync artifacts without running

```bash
./scripts/sync_from_server.sh
```

---

## Variables you'll edit

All three scripts read these via env vars (or hard-coded defaults at the top):

| var          | default          | meaning                                |
|--------------|------------------|----------------------------------------|
| `HF_USER`    | `Lakesenberg`    | your HuggingFace username              |
| `SUITE`      | `libero_10`      | libero_spatial / object / goal / 10    |
| `EPISODES`   | `10`             | trials per task (4090 eval)            |
| `SERVER_HOST`| `192.168.3.103`  | A100 server IP                         |
| `SERVER_USER`| `lakesenberg`    | your account on the A100               |
| `SERVER_DIR` | `~/a2c2-libero`  | repo path on the A100                  |
| `CKPT_SOURCE`| `hf`             | `hf` or `server` (where 4090 pulls)    |

---

## End-to-end command summary

```bash
# A100 server
ssh a100 "cd ~/a2c2-libero && SUITE=libero_10 ./scripts/server_train.sh"

# 4090 inference machine
SUITE=libero_10 CKPT_SOURCE=server ./scripts/inference_4090.sh
cat logs/4090_eval_libero_10.json | python -m json.tool
```

Run the second pair for each suite (`libero_spatial`, `libero_object`,
`libero_goal`, `libero_10`) to populate the main results table in your
thesis.

## Why split this way

* **A100 = training only.** SmolVLA fine-tune (~80k steps, ~12 h) and
  residual head (~200k steps, ~6 h) saturate one A100. No real-time
  constraints.
* **4090 = inference only.** SmolVLA forward at ~80 ms is fine for the
  ~10 Hz upper layer; A2C2 head + control loop run at 200 Hz. The 4090
  has the lowest single-stream latency of the two cards, which matters
  more than raw throughput at deployment.
* **Artifacts cross via HF (default) or rsync (LAN).** Hugging Face acts
  as the source of truth — versioned, reproducible. Rsync is for speed
  when the LAN is reliable.

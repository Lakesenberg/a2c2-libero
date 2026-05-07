# a2c2-libero · Async SmolVLA + A2C2 residual head deployment

Two-machine deployment toolkit for hierarchical SmolVLA + A2C2 residual transformer:
heavy training on an **A100 server**, real-time inference on a **4090 client**.
Branch: [`inference_test_async`](https://github.com/Lakesenberg/a2c2-libero/tree/inference_test_async).

```
┌─────────────────────────────────┐         ┌──────────────────────────────────┐
│  A100 server (192.168.3.103)    │  ckpts  │  4090 inference machine          │
│  ─────────────────────────────  │ ──────▶ │  ──────────────────────────────  │
│  • train SmolVLA                │  scp/HF │  • async A2C2 inference         │
│  • build residual dataset        │         │  • real robot (bi-SO / SO-100)   │
│  • train A2C2 residual head      │         │  • optional LIBERO env eval      │
└─────────────────────────────────┘         └──────────────────────────────────┘
```

---

## Repository layout

```
.
├── src/a2c2_libero/
│   ├── inference/
│   │   ├── async_smolvla.py      # background SmolVLA worker
│   │   ├── a2c2_engine.py        # per-tick A2C2 engine (sync + async)
│   │   ├── utils.py              # build_state, sincos, LatentHook
│   │   └── mocks.py              # MockSmolVLA / MockA2C2Head for tests
│   └── heads/a2c2_head.py        # minimal MLP residual head
├── scripts/
│   ├── server_train.sh           # A100 — full training pipeline
│   ├── inference_4090.sh         # 4090 — eval on LIBERO/real robot
│   ├── sync_from_server.sh       # 4090 — rsync ckpts from A100
│   ├── realrobot_a2c2_inference.py   # 4090 — real-robot inference loop
│   ├── a100_inference_test.py    # latency benchmark using a trained SmolVLA
│   ├── run_on_a100.sh            # SLURM submit wrapper
│   └── run_inference_demo.py     # CPU demo (no GPU, no LIBERO required)
├── tests/                        # 27 unit tests (threading, engine, head)
└── docs/
    ├── TWO_MACHINE_SETUP.md      # canonical deployment workflow
    ├── A100_TEST.md              # A100 latency benchmark guide
    ├── DEBUG_JOURNEY.md          # full debug story (v2.1 dataset → working)
    ├── NEXT_STEPS.md             # post-training inference plan
    ├── STOP_TRAIN_AND_INFER.md   # early-stop + scp + real-robot eval
    └── ...                        # offline / stats / version docs
```

---

## Quick start (after the residual head has been trained on the A100)

### 0. Common bootstrap (both machines)

```bash
git clone https://github.com/Lakesenberg/a2c2-libero.git
cd a2c2-libero
git switch inference_test_async
git submodule update --init --recursive

uv venv -p 3.10 && source .venv/bin/activate
uv pip install -e ".[smolvla]"
uv pip install -e third_party/libero
uv pip install mujoco==3.3.2
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=egl    # headless
```

### 1. SSH alias for the A100 (one-time, run on the 4090)

```bash
cat >> ~/.ssh/config <<EOF

Host a100
  Hostname 192.168.3.103
  User <your_a100_user>
  IdentityFile ~/.ssh/id_rsa
EOF

ssh a100   # smoke test
```

### 2. A100 server — full training pipeline

```bash
ssh a100
cd ~/a2c2-libero
chmod +x scripts/server_train.sh

SUITE=libero_10 ./scripts/server_train.sh
```

This runs, in order:

1. SmolVLA fine-tune (skip with `SKIP_SMOLVLA=1` if you already have one).
2. Generate residual dataset by replaying the frozen SmolVLA over your demo
   data (writes `vla_actions` + `vlm_hidden`).
3. Train the residual transformer head on that residual dataset (~6 h, but
   loss usually converges by step 60 k — feel free to early-stop).
4. Synthetic-input latency smoke test.

Variables you'll edit (env or top of script):

| variable        | default          | meaning                                |
|-----------------|------------------|----------------------------------------|
| `HF_USER`       | `Lakesenberg`    | HuggingFace owner namespace            |
| `SUITE`         | `libero_10`      | libero_spatial / object / goal / 10    |
| `SERVER_HOST`   | `192.168.3.103`  | A100 IP                                |
| `SERVER_USER`   | your A100 user   | account on the A100                    |
| `SKIP_SMOLVLA`  | unset            | set to `1` to reuse an existing ckpt   |

### 2.1 Early-stop the residual head when loss plateaus

```bash
# A100 — once loss is around 0.01-0.012 (≈ 60 k steps)
PID=$(pgrep -f run_train_offline)
kill $PID
sleep 3
LAST=$(ls -t outputs/a2c2_head_v21/checkpoints/ | grep -E '^[0-9]+$' | head -1)
realpath outputs/a2c2_head_v21/checkpoints/$LAST/pretrained_model
# → /root/.../outputs/a2c2_head_v21/checkpoints/060000/pretrained_model
```

### 3. 4090 client — pull checkpoints and run inference

#### 3.1 LIBERO simulator evaluation

```bash
# Pull ckpts from HuggingFace (default; works without server reachability)
SUITE=libero_10 ./scripts/inference_4090.sh

# Or rsync directly from the A100 over LAN
SUITE=libero_10 CKPT_SOURCE=server ./scripts/inference_4090.sh

cat logs/4090_eval_libero_10.json | python -m json.tool
```

#### 3.2 Real robot inference (bi-SO / SO-100 / your robot)

```bash
# Optional: copy ckpts from the server first
mkdir -p outputs

scp -r a100:~/a2c2-libero/outputs/a2c2_head_v21/checkpoints/060000/pretrained_model \
       outputs/a2c2_head_v21
scp -r a100:/root/project/lq/lerobot/outputs/train/smolvla_a2c2_v21/checkpoints/020000/pretrained_model \
       outputs/smolvla_v21

# Robot calibration (one-time)
lerobot-calibrate --robot.type=so100_follower --robot.id=<your_id>
lerobot-find-cameras

# Dry run (no data recording, 3 episodes; remove --no-record to record)
python scripts/realrobot_a2c2_inference.py \
    --smolvla-path outputs/smolvla_v21 \
    --head-ckpt    outputs/a2c2_head_v21/model.safetensors \
    --robot-type   so100_follower \
    --robot-id     <your_robot_id> \
    --action-dim   6 \
    --chunk-size   50 \
    --episodes     3 \
    --task         "pick up the cup" \
    --home-on-start \
    --no-record
```

The script feeds the residual head **all four** training-time fields:
single-step `action`, full `base_action_chunk`, `time_feature`
(sin/cos chunk index), and `vlm_hidden`. Without these the residual
transformer falls back to the base-action-only path — see `docs/DEBUG_JOURNEY.md`.

### 4. Sync artifacts only (no eval)

```bash
# 4090
./scripts/sync_from_server.sh
```

---

## Async timing model

```
Background SmolVLA:   [forward#0===][#1===][#2===][#3===]
                                   ↓      ↓      ↓      ↓
SharedBoard chunk:    None ──────  chunk_0 chunk_1 chunk_2 chunk_3
SharedBoard z:        None ──────  z_0     z_1     z_2     z_3
SharedBoard k:        0    ──────  0..H-1  0..H-1  0..H-1  0..H-1

Main thread A2C2:     hold-pose    step    step    step    step
                                   every tick (5 ms), reads board snapshot
```

The main thread never blocks on SmolVLA. It uses the most recently published
chunk and applies the per-tick A2C2 correction
`a_exec = chunk[k] + π_C(obs_t, chunk[k], chunk_full, τ_k, z, lang)`, then
advances `k`. When SmolVLA finishes a new forward, the worker writes the new
chunk to the board and resets `k = 0`.

---

## Tests / demo (no GPU, no LIBERO required)

```bash
pip install -e ".[test]"
pytest tests/ -v --timeout 30        # 27 tests, ~5 s
pytest tests/ -v -m slow             # add LIBERO smoke (needs MUJOCO_GL=egl)

python scripts/run_inference_demo.py                       # async, 200 ticks
python scripts/run_inference_demo.py --sync --ticks 30     # sync version
python scripts/run_inference_demo.py --smolvla-latency-ms 200  # slow SmolVLA
```

---

## Troubleshooting cheat-sheet

| Symptom | See |
|---|---|
| `lerobot==0.1.7` not found / pip install doesn't take effect | `docs/DEBUG_JOURNEY.md` §1-2 |
| `MaxRetryError: huggingface.co` | `docs/OFFLINE_RESIDUAL_DATASET_v3.md` |
| `OfflineModeIsEnabled: ... refs` | `docs/OFFLINE_RESIDUAL_DATASET_v3.md` |
| `RevisionNotFoundError: ... codebase version` | `docs/DEBUG_JOURNEY.md` §6 |
| `tasks.jsonl: No such file` | `docs/DEBUG_JOURNEY.md` §7 |
| `[normalize-fix] mean is inf` | `docs/FIX_STATS_AND_TRAIN.md` |
| `TypeError: only length-1 arrays...` (vla_actions) | `docs/FIX_STATS_VLA_ACTIONS.md` |
| Residual head loss stuck at NaN | re-run stats fix (`docs/FIX_STATS_VLA_ACTIONS.md`) |

Full chronological story of every error encountered while bringing a v2.1
dataset through the v3.0 LeRobot fork into a successful residual-head
training run is in [`docs/DEBUG_JOURNEY.md`](docs/DEBUG_JOURNEY.md).

---

## References

* **A2C2** — *Leave No Observation Behind: Real-time Correction for VLA Action Chunks*, arXiv:[2509.23224](https://arxiv.org/abs/2509.23224)
* Upstream A2C2 + SmolVLA + LIBERO codebase — [`k1000dai/a2c2-libero`](https://github.com/k1000dai/a2c2-libero)
* A2C2 + Kinetix sim reproduction — [`k1000dai/a2c2-kinetix`](https://github.com/k1000dai/a2c2-kinetix)
* **SmolVLA** — *Affordable VLA*, arXiv:[2506.01844](https://arxiv.org/abs/2506.01844)
* **RTC** — *Real-Time Execution of Action Chunking Flow Policies*, arXiv:[2506.07339](https://arxiv.org/abs/2506.07339)
* **LeRobot** — https://github.com/huggingface/lerobot
* **LIBERO** — https://github.com/Lifelong-Robot-Learning/LIBERO

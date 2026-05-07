# 接下来要做什么 — 从残差数据集到真机 A2C2 推理

当前状态:
- ✅ SmolVLA 训练完成 (`/root/project/lq/lerobot/outputs/train/smolvla_a2c2_v21/checkpoints/020000/pretrained_model`)
- 🔄 残差数据集生成中 (~93%, 25000/26751 帧)
- ⚠️ 数据集 stats 损坏 (mean/std = inf, 训练前必须修)
- ⏳ 残差头训练 (没开始)
- ⏳ 真机 A2C2 推理 (没开始)

所有命令在 A100 (`192.168.3.103`) bash 里执行, 假设 `cd ~/project/lq/a2c2-libero`。

---

## Step 1 · 等数据集生成完 (~10 分钟)

```bash
# 监控
tail -f logs/residual.log | grep -E "Saving episode|Processed|Done"
```

完成标志: 看到 `Saving episode 80` (或最大 episode 数) + 不再有新输出 + python 进程退出。

```bash
# 确认数据集生成完整
DATASET_ROOT=~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual

ls "$DATASET_ROOT/"
ls "$DATASET_ROOT/data/" | wc -l       # parquet 文件数
ls "$DATASET_ROOT/meta/"
cat "$DATASET_ROOT/meta/info.json" | python -m json.tool | head -30
```

---

## Step 2 · 修复 stats inf 问题 (必做, 5 分钟)

日志里反复出现的 `[normalize-fix] mean is inf; ...std is inf;` 说明源数据集 stats 损坏, 残差数据集继承了这个问题。<b>不修的话训练会出 NaN loss。</b>

```bash
DATASET_ROOT=~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual

python << 'PY'
import os, json, glob
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
import pandas as pd
from pathlib import Path

root = Path("/root/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual")

files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
print(f"[stats] found {len(files)} parquet files")

cols_to_stat = ["observation.state", "action", "vla_actions"]
buf = {c: [] for c in cols_to_stat}

for f in files:
    df = pd.read_parquet(f)
    for c in cols_to_stat:
        if c in df.columns:
            arr = np.stack(df[c].values)
            buf[c].append(arr)

stats = {}
for c, lst in buf.items():
    if not lst: continue
    arr = np.concatenate(lst, axis=0).astype(np.float64)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    stats[c] = {
        "mean":  arr.mean(0).tolist(),
        "std":   arr.std(0).clip(1e-6, None).tolist(),
        "min":   arr.min(0).tolist(),
        "max":   arr.max(0).tolist(),
        "count": int(arr.shape[0]),
    }
    print(f"[stats] {c}: shape={arr.shape}  mean={arr.mean(0)[:3]}  std={arr.std(0)[:3]}")

out = root / "meta" / "stats.json"
with open(out, "w") as f:
    json.dump(stats, f, indent=2)
print(f"[stats] wrote {out}")

# v2.1 也写一份 episodes_stats.jsonl (空文件占位即可, 防止 lerobot 找)
es = root / "meta" / "episodes_stats.jsonl"
if not es.exists():
    es.touch()
    print(f"[stats] touched {es}")
PY
```

### 验证 stats 修好了

```bash
cat "$DATASET_ROOT/meta/stats.json" | python -m json.tool | head -40
```

看 `mean`, `std`:
- ✅ 应该是合理数值, 例如 `[0.12, -0.34, 1.5, ...]`
- ❌ 如果还是 `Infinity` / `NaN` / `inf`, 说明 BASE 数据集本身坏了, 见<b>故障 Q1</b>

---

## Step 3 · 训练残差头 (~6 h on A100)

```bash
mkdir -p outputs/a2c2_head_v21 logs

python run_train_offline.py \
  --policy.type=residual_transformer \
  --policy.repo_id=lakesenberg/a2c2_head_v21 \
  --dataset.repo_id=lakesenberg/a2c2_dataset_v21-smolvla-residual \
  --batch_size=64 \
  --num_workers=16 \
  --steps=200000 \
  --output_dir=outputs/a2c2_head_v21 \
  --job_name=a2c2_head_v21 \
  --wandb.enable=false \
  2>&1 | tee logs/train_a2c2_head.log
```

### 监控

```bash
# 看 loss 下降
tail -f logs/train_a2c2_head.log | grep -E "step|loss"
```

预期:
| 步数 | loss 大致 |
|---|---|
| 1k | 0.5 - 1.0 |
| 10k | 0.1 - 0.2 |
| 100k | < 0.05 |
| 200k | 收敛 |

如果 loss 出现 `nan` / `inf` → stats 没修好, 回 Step 2 重做。

### 后台跑 + 关 ssh 不影响

```bash
nohup python run_train_offline.py \
  --policy.type=residual_transformer \
  --policy.repo_id=lakesenberg/a2c2_head_v21 \
  --dataset.repo_id=lakesenberg/a2c2_dataset_v21-smolvla-residual \
  --batch_size=64 --num_workers=16 --steps=200000 \
  --output_dir=outputs/a2c2_head_v21 --wandb.enable=false \
  > logs/train_a2c2_head.log 2>&1 &

echo "PID: $!"
disown
```

---

## Step 4 · 训完后查产物

A100:
```bash
ls outputs/a2c2_head_v21/
ls outputs/a2c2_head_v21/checkpoints/

# 找最新 ckpt
LAST_CKPT=$(ls -t outputs/a2c2_head_v21/checkpoints/ | head -1)
echo "最新 ckpt: outputs/a2c2_head_v21/checkpoints/$LAST_CKPT/pretrained_model"
ls "outputs/a2c2_head_v21/checkpoints/$LAST_CKPT/pretrained_model/"
```

应该看到 `model.safetensors` (或 `pytorch_model.bin`) + `config.json`。

---

## Step 5 · scp 到 4090 推理机

在 4090 终端:

```bash
mkdir -p ~/a2c2-libero/outputs

# 拉 A2C2 残差头
scp -r root@192.168.3.103:/root/project/lq/a2c2-libero/outputs/a2c2_head_v21 \
       ~/a2c2-libero/outputs/

# 拉 SmolVLA ckpt (如果 4090 还没有)
scp -r root@192.168.3.103:/root/project/lq/lerobot/outputs/train/smolvla_a2c2_v21/checkpoints/020000/pretrained_model \
       ~/a2c2-libero/outputs/smolvla_v21

ls ~/a2c2-libero/outputs/
```

---

## Step 6 · 4090 真机 A2C2 推理

### 6.1 先检查机器人 + 相机能用

```bash
cd ~/a2c2-libero

# 标定 (如果没标过)
lerobot-calibrate --robot.type=<你的 robot type, 例如 so100_follower> --robot.id=<你的 id>

# 找相机
lerobot-find-cameras

# 看 robot.id 是什么
ls ~/.cache/huggingface/lerobot/calibration/robots/
```

### 6.2 跑真机 A2C2 推理

```bash
python scripts/realrobot_a2c2_inference.py \
  --smolvla-path ~/a2c2-libero/outputs/smolvla_v21 \
  --head-ckpt    ~/a2c2-libero/outputs/a2c2_head_v21/checkpoints/last/pretrained_model/model.safetensors \
  --robot-type   <你的 robot type> \
  --robot-id     <你的 robot id> \
  --action-dim   6 \
  --chunk-size   50 \
  --episodes     5 \
  --task         "pick up the cup" \
  --home-on-start
```

---

## 故障排查

### Q1. Step 2 算出来 stats 还是 inf/nan

源 BASE 数据集本身有损坏帧。过滤掉:

```bash
DATASET_ROOT=~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual
python << 'PY'
import glob, numpy as np, pandas as pd
from pathlib import Path
root = Path("/root/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual")
for f in sorted(glob.glob(str(root/"data/**/*.parquet"), recursive=True)):
    df = pd.read_parquet(f)
    bad = 0
    for c in ["observation.state", "action"]:
        if c in df.columns:
            arr = np.stack(df[c].values)
            mask = np.isfinite(arr).all(axis=tuple(range(1, arr.ndim)))
            bad += int((~mask).sum())
    if bad:
        print(f"{f}: {bad} bad rows")
PY
```

### Q2. Step 3 训练 loss 一直高 / 不下降

可能原因:
- stats 没修好 (回 Step 2)
- batch_size 太大 OOM, 改 16
- 学习率太高, 加 `--policy.optimizer.lr=1e-5`

```bash
nvidia-smi      # 看显存占用
```

### Q3. ckpt 4090 加载报 missing key

A100 上的 lerobot 跟 4090 上的不是同一个 fork 版本。确认两边 git checkout 同一个 commit:

```bash
# 两边都跑
cd ~/project/lq/a2c2-libero  # 或 ~/a2c2-libero
git rev-parse HEAD
```

不一样的话, 4090 同步到 A100 的 commit:
```bash
git fetch origin && git checkout <a100 上的 commit hash>
```

### Q4. realrobot_a2c2_inference.py 报 LatentHook 找不到 backbone

SmolVLA 内部结构因版本而异。先打印结构:
```bash
python -c "
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
p = SmolVLAPolicy.from_pretrained('outputs/smolvla_v21')
for name, _ in p.named_modules():
    if 'backbone' in name.lower() or 'encoder' in name.lower():
        print(name)
" 2>&1 | head -20
```

把输出贴出来, 我告诉你改哪行。

---

## 时间预算

| 步骤 | 时间 |
|---|---|
| Step 1 等数据集生成完 | ~10 min (剩余) |
| Step 2 修 stats | 5 min |
| Step 3 训残差头 | 6 h (后台跑) |
| Step 4 查产物 | 1 min |
| Step 5 scp 到 4090 | 5 min |
| Step 6 真机推理 (5 ep) | 30 min |
| **合计** | **~7 h** |

---

## 一次到位脚本 (Step 2 → Step 3 → Step 4 串起)

数据集生成完后, 这一段一次贴下去:

```bash
cd ~/project/lq/a2c2-libero
DATASET_ROOT=~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual

# Step 2 修 stats
python << 'PY'
import os, json, glob
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, pandas as pd
from pathlib import Path
root = Path("/root/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual")
files = sorted(glob.glob(str(root/"data/**/*.parquet"), recursive=True))
print(f"found {len(files)} parquet files")
buf = {c: [] for c in ["observation.state","action","vla_actions"]}
for f in files:
    df = pd.read_parquet(f)
    for c in buf:
        if c in df.columns:
            buf[c].append(np.stack(df[c].values))
stats = {}
for c, lst in buf.items():
    if not lst: continue
    arr = np.concatenate(lst,0).astype(np.float64)
    if arr.ndim == 3: arr = arr.reshape(-1, arr.shape[-1])
    stats[c] = {
        "mean": arr.mean(0).tolist(),
        "std":  arr.std(0).clip(1e-6,None).tolist(),
        "min":  arr.min(0).tolist(),
        "max":  arr.max(0).tolist(),
        "count": int(arr.shape[0]),
    }
    print(f"{c}: mean[:3]={arr.mean(0)[:3]}  std[:3]={arr.std(0)[:3]}")
with open(root/"meta"/"stats.json","w") as f: json.dump(stats,f,indent=2)
es = root/"meta"/"episodes_stats.jsonl"
if not es.exists(): es.touch()
print("stats.json written")
PY

# 验证 stats
cat "$DATASET_ROOT/meta/stats.json" | python -m json.tool | head -30

# Step 3 后台启训练
mkdir -p outputs/a2c2_head_v21 logs
nohup python run_train_offline.py \
  --policy.type=residual_transformer \
  --policy.repo_id=lakesenberg/a2c2_head_v21 \
  --dataset.repo_id=lakesenberg/a2c2_dataset_v21-smolvla-residual \
  --batch_size=64 --num_workers=16 --steps=200000 \
  --output_dir=outputs/a2c2_head_v21 --wandb.enable=false \
  > logs/train_a2c2_head.log 2>&1 &
echo "training PID: $!"
disown

# Step 4 跟踪 loss
sleep 30
tail -50 logs/train_a2c2_head.log | grep -E "step|loss"
```

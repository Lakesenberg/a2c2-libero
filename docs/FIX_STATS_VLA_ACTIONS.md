# 修 stats — vla_actions 变长 chunk 报错版

上次修 stats 报错 `TypeError: only length-1 arrays can be converted to Python scalars`,
原因: `vla_actions` 是 chunk 数据 (shape `(H, A)`),不同帧 chunk 长度不同,`np.stack` 失败。

但你已经看到 `observation.state` 和 `action` 的 mean/std 是<b>正常数字</b> (不是 inf):
```
observation.state: mean[:3]=[ -5.49 -24.31  27.67]  std[:3]=[14.37 48.69 43.11]
action:            mean[:3]=[ -5.56 -25.34  26.45]  std[:3]=[14.46 48.44 44.07]
```

→ 数据本身是健康的, 只是脚本对变长 chunk 处理失败。下面是修过的版本。

---

## §1 修过的 stats 脚本 (30 秒)

```bash
cd ~/project/lq/a2c2-libero

python << 'PY'
import os, json, glob
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, pandas as pd
from pathlib import Path

root = Path("/root/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual")
files = sorted(glob.glob(str(root/"data/**/*.parquet"), recursive=True))
print(f"[stats] found {len(files)} parquet files")


def safe_stack(values):
    """Try np.stack; on shape mismatch, flatten each row and concat."""
    try:
        return np.stack(values)
    except (ValueError, TypeError):
        parts = []
        for v in values:
            v = np.asarray(v)
            if v.ndim == 0:
                continue
            if v.ndim == 1:
                parts.append(v.reshape(1, -1))
            else:
                parts.append(v.reshape(-1, v.shape[-1]))
        if not parts:
            return None
        return np.concatenate(parts, axis=0)


cols_to_stat = ["observation.state", "action", "vla_actions"]
buf = {c: [] for c in cols_to_stat}

for f in files:
    df = pd.read_parquet(f)
    for c in cols_to_stat:
        if c in df.columns:
            arr = safe_stack(df[c].values)
            if arr is not None:
                buf[c].append(arr)

stats = {}
for c, lst in buf.items():
    if not lst:
        print(f"[stats] {c}: SKIPPED (no data)")
        continue
    arr = np.concatenate(lst, axis=0).astype(np.float64)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    # 过滤 inf/nan 行
    mask = np.isfinite(arr).all(axis=1)
    if (~mask).any():
        print(f"[stats] {c}: filtered {(~mask).sum()} bad rows out of {len(arr)}")
        arr = arr[mask]
    if arr.size == 0:
        print(f"[stats] {c}: SKIPPED (all rows filtered)")
        continue
    stats[c] = {
        "mean":  arr.mean(0).tolist(),
        "std":   arr.std(0).clip(1e-6, None).tolist(),
        "min":   arr.min(0).tolist(),
        "max":   arr.max(0).tolist(),
        "count": int(arr.shape[0]),
    }
    print(f"[stats] {c}: shape={arr.shape}  mean[:3]={arr.mean(0)[:3]}  std[:3]={arr.std(0)[:3]}")

with open(root/"meta"/"stats.json", "w") as f:
    json.dump(stats, f, indent=2)
es = root/"meta"/"episodes_stats.jsonl"
if not es.exists():
    es.touch()
print(f"[stats] wrote {root/'meta'/'stats.json'}")
PY
```

### 验证

```bash
cat ~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual/meta/stats.json \
  | python -m json.tool | head -40
```

应该看到至少 `observation.state` 和 `action` 两个完整 entry, 每个 mean/std 是数字数组。
`vla_actions` 应该也能成功 (变长 chunk 被 flatten 处理了)。

---

## 修过的版本干了 4 件事

1. **`safe_stack`** — `np.stack` 失败时 flatten 后 concat (处理变长 chunk)
2. **`reshape(-1, action_dim)`** — chunk 数据 (N, H, A) → (N×H, A)
3. **过滤 inf/nan 行** — 个别坏帧不影响整体 stats
4. **空 buffer 跳过** — 不存在的列不会崩

---

## §2 stats OK 后启训练 (后台 ~6 h)

```bash
cd ~/project/lq/a2c2-libero
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

sleep 60
echo "=== 60 秒后日志 ==="
tail -50 logs/train_a2c2_head.log
```

### 监控

```bash
tail -f logs/train_a2c2_head.log | grep -E "step|loss"
watch -n 2 nvidia-smi
```

### 预期 loss 下降

| 步数 | loss 大致 |
|---|---|
| 1k | 0.5 - 1.0 |
| 10k | 0.1 - 0.2 |
| 100k | < 0.05 |
| 200k | 收敛 |

如果出现 `nan` / `inf` → stats 还有问题, 回 §1。

---

## §3 训完查产物

```bash
ls outputs/a2c2_head_v21/checkpoints/
LAST=$(ls -t outputs/a2c2_head_v21/checkpoints/ | head -1)
ls outputs/a2c2_head_v21/checkpoints/$LAST/
```

---

## §4 scp 到 4090 (在 4090 上)

```bash
mkdir -p ~/a2c2-libero/outputs

scp -r root@192.168.3.103:/root/project/lq/a2c2-libero/outputs/a2c2_head_v21 \
       ~/a2c2-libero/outputs/

scp -r root@192.168.3.103:/root/project/lq/lerobot/outputs/train/smolvla_a2c2_v21/checkpoints/020000/pretrained_model \
       ~/a2c2-libero/outputs/smolvla_v21

ls ~/a2c2-libero/outputs/
```

---

## §5 4090 真机推理

```bash
cd ~/a2c2-libero

ls ~/.cache/huggingface/lerobot/calibration/robots/   # 看 robot.id

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

### Q1. stats 修过版本仍报错

把完整 traceback 贴出来。最常见是 parquet 列名跟假定的不一致, 用这条诊断:

```bash
python -c "
import pandas as pd, glob
f = sorted(glob.glob('/root/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual/data/**/*.parquet', recursive=True))[0]
df = pd.read_parquet(f)
print('columns:', list(df.columns))
print()
for c in df.columns:
    sample = df[c].iloc[0]
    print(f'{c}: type={type(sample).__name__}, ', end='')
    if hasattr(sample, 'shape'):
        print(f'shape={sample.shape}, dtype={sample.dtype}')
    else:
        print(f'value sample={str(sample)[:60]}')
"
```

把输出贴回来, 我据此调整 cols_to_stat 列表。

### Q2. 训练 loss 一直高 / 不下降

stats 没修好 (回 §1) 或 batch 太大 OOM 改 32 (`--batch_size=32`)。

### Q3. ckpt 4090 加载缺 key

A100 / 4090 fork commit 不一致, 两边 git rev-parse HEAD 看, 同步同一个 commit。

---

## 时间预算

| 段 | 时间 |
|---|---|
| §1 修 stats | 30 秒 |
| §2 训练 | 6 h (后台) |
| §3 查产物 | 1 min |
| §4 scp | 5 min |
| §5 真机 5 ep | 30 min |

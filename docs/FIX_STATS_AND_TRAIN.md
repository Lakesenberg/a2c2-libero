# 数据集已生成完 — 修 stats + 启训练

状态确认:
- ✅ 残差数据集生成完毕 ("All episodes processed and saved")
- ✅ 75 个 parquet 文件
- ⚠️ stats 损坏 (`[normalize-fix] mean is inf`)

下面 2 段命令: ① 修 stats ② 启训练。
所有命令在 A100 (`192.168.3.103`) bash 里, `cd ~/project/lq/a2c2-libero`。

---

## 第 1 段 · 修 stats (30 秒)

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

buf = {c: [] for c in ["observation.state", "action", "vla_actions"]}
for f in files:
    df = pd.read_parquet(f)
    for c in buf:
        if c in df.columns:
            buf[c].append(np.stack(df[c].values))

stats = {}
for c, lst in buf.items():
    if not lst: continue
    arr = np.concatenate(lst, 0).astype(np.float64)
    if arr.ndim == 3: arr = arr.reshape(-1, arr.shape[-1])
    stats[c] = {
        "mean":  arr.mean(0).tolist(),
        "std":   arr.std(0).clip(1e-6, None).tolist(),
        "min":   arr.min(0).tolist(),
        "max":   arr.max(0).tolist(),
        "count": int(arr.shape[0]),
    }
    print(f"[stats] {c}: mean[:3]={arr.mean(0)[:3]}  std[:3]={arr.std(0)[:3]}")

with open(root/"meta"/"stats.json", "w") as f:
    json.dump(stats, f, indent=2)
es = root/"meta"/"episodes_stats.jsonl"
if not es.exists(): es.touch()
print("[stats] done")
PY

# 验证 mean/std 不再是 inf
cat ~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual/meta/stats.json \
  | python -m json.tool | head -30
```

### 验证标准

输出应该看到 `observation.state`, `action`, `vla_actions` 三个 key, 每个的 `mean` / `std` 是<b>数字数组</b>,
例如:
```json
"observation.state": {
    "mean": [0.123, -0.456, 1.5, ...],
    "std":  [0.05, 0.08, 0.12, ...],
}
```

❌ 看到 `Infinity` / `NaN` / `null` 说明 BASE 数据集本身有坏数据帧, 跳到底部 §故障排查。

---

## 第 2 段 · 启训练 (后台 ~6 h)

`stats.json` 验证 OK 后:

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

### 关闭 ssh 不影响训练

`nohup ... &` + `disown` 让进程脱离当前 shell, ssh 断开不影响。

### 监控

```bash
# loss 实时
tail -f logs/train_a2c2_head.log | grep -E "step|loss"

# GPU
watch -n 2 nvidia-smi
```

### 预期 loss 下降

| 步数 | loss 大致 |
|---|---|
| 1k | 0.5 - 1.0 |
| 10k | 0.1 - 0.2 |
| 100k | < 0.05 |
| 200k | 收敛 |

如果 loss 出现 `nan` / `inf`, 说明 stats 没修好, 回 §1。

---

## 第 3 段 · 训完后查产物 (~6 h 后)

```bash
ls outputs/a2c2_head_v21/
ls outputs/a2c2_head_v21/checkpoints/

LAST_CKPT=$(ls -t outputs/a2c2_head_v21/checkpoints/ | head -1)
echo "最新 ckpt 路径: outputs/a2c2_head_v21/checkpoints/$LAST_CKPT"
ls "outputs/a2c2_head_v21/checkpoints/$LAST_CKPT/pretrained_model/" 2>/dev/null \
   || ls "outputs/a2c2_head_v21/checkpoints/$LAST_CKPT/" 
```

---

## 第 4 段 · scp 到 4090 (在 4090 上跑, 不在 A100)

```bash
mkdir -p ~/a2c2-libero/outputs

scp -r root@192.168.3.103:/root/project/lq/a2c2-libero/outputs/a2c2_head_v21 \
       ~/a2c2-libero/outputs/

# 如果 4090 上还没有 SmolVLA, 也拉过来
scp -r root@192.168.3.103:/root/project/lq/lerobot/outputs/train/smolvla_a2c2_v21/checkpoints/020000/pretrained_model \
       ~/a2c2-libero/outputs/smolvla_v21

ls ~/a2c2-libero/outputs/
```

---

## 第 5 段 · 4090 真机推理

```bash
cd ~/a2c2-libero

# 看 robot.id (从标定文件)
ls ~/.cache/huggingface/lerobot/calibration/robots/

# 跑真机 A2C2 推理
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

### Q1. stats.json 里 mean/std 还是 Infinity

源数据有坏帧, 找出来:

```bash
python << 'PY'
import glob, numpy as np, pandas as pd
from pathlib import Path
root = Path("/root/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual")
for f in sorted(glob.glob(str(root/"data/**/*.parquet"), recursive=True)):
    df = pd.read_parquet(f)
    bad = 0
    for c in ["observation.state", "action", "vla_actions"]:
        if c in df.columns:
            arr = np.stack(df[c].values)
            mask = np.isfinite(arr).all(axis=tuple(range(1, arr.ndim)))
            bad += int((~mask).sum())
    if bad:
        print(f"{f}: {bad} bad rows")
PY
```

输出哪些 parquet 有坏帧。可以选择删除那些 episode 重跑 dataset, 或写个过滤版本。

### Q2. 训练 loss 一直高 / 不下降

- stats 没修好 (回 §1)
- batch_size 太大: 改 `--batch_size=32`
- 学习率: 加 `--policy.optimizer.lr=1e-5`

```bash
nvidia-smi   # 看显存
```

### Q3. ckpt 4090 加载报 missing key

A100 和 4090 上的 lerobot fork 版本不一致。两边 git checkout 同一 commit:
```bash
cd <a2c2-libero>
git rev-parse HEAD
```

### Q4. realrobot_a2c2_inference.py 报 LatentHook 找不到 backbone

打印 SmolVLA 实际结构:
```bash
python -c "
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
p = SmolVLAPolicy.from_pretrained('/path/to/smolvla')
for name, _ in p.named_modules():
    if 'backbone' in name.lower() or 'encoder' in name.lower():
        print(name)
" | head -20
```

把输出贴出来调整 hook 路径。

---

## 时间预算

| 段落 | 时间 |
|---|---|
| §1 修 stats | 30 秒 |
| §2 启训练 | 后台 ~6 h |
| §3 查产物 | 1 分钟 |
| §4 scp 到 4090 | 5 分钟 |
| §5 真机 5 ep | 30 分钟 |
| **合计** | **~7 h** |

---

## 一键脚本 (§1 + §2 串起)

```bash
cd ~/project/lq/a2c2-libero

# § 1 修 stats
python << 'PY'
import os, json, glob
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, pandas as pd
from pathlib import Path
root = Path("/root/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual")
files = sorted(glob.glob(str(root/"data/**/*.parquet"), recursive=True))
print(f"[stats] found {len(files)} parquet files")
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
    stats[c] = {"mean": arr.mean(0).tolist(), "std": arr.std(0).clip(1e-6,None).tolist(),
                "min": arr.min(0).tolist(), "max": arr.max(0).tolist(),
                "count": int(arr.shape[0])}
    print(f"{c}: mean[:3]={arr.mean(0)[:3]}  std[:3]={arr.std(0)[:3]}")
with open(root/"meta"/"stats.json","w") as f: json.dump(stats,f,indent=2)
es = root/"meta"/"episodes_stats.jsonl"
if not es.exists(): es.touch()
print("[stats] done")
PY

# 验证
cat ~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual/meta/stats.json \
  | python -m json.tool | head -30

# § 2 启训练
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
tail -50 logs/train_a2c2_head.log
```

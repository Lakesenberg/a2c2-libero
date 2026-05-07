# 诊断: tail -f 没反应,残差数据集到底跑完没

`tail -f logs/residual.log | grep -E "Saving episode|..."` 没新输出, 不一定是有问题。
按下面的顺序排查。所有命令在 A100 (`192.168.3.103`) bash 里, 假设
`cd ~/project/lq/a2c2-libero`。

---

## Step 0 · 退出当前 tail

按 `Ctrl + C` 退出 `tail -f`,回到提示符再执行下面的诊断。

---

## Step 1 · 跑 4 条诊断命令

```bash
# 1. 进程还在不在?
ps aux | grep run_residual_offline | grep -v grep

# 2. 日志最后 50 行 (不带 grep 过滤, 看真实状态)
tail -50 logs/residual.log

# 3. 日志最后修改时间 (停了多久)
ls -la logs/residual.log

# 4. 数据集生成了多少个 parquet
ls ~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual/data/ 2>/dev/null
ls ~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual/data/chunk-000/ 2>/dev/null | wc -l
```

---

## Step 2 · 根据输出判断状态

### 状态 A · 跑完了 ✅

特征:
- `ps` 没输出
- `tail -50` 末尾看到 `done` / `Saving episode <最大数>` / `Map: 100%`
- `ls .../data/chunk-000/ | wc -l` ≥ episode 总数 (你这个数据集应该 70+)

下一步: 跳到 §3 修 stats, 然后训残差头。

### 状态 B · 中途崩了 ❌

特征:
- `ps` 没输出
- `tail -50` 末尾出现 `Traceback` / `Error` / `Killed` / `OOM`
- parquet 文件数 < episode 总数

排查:
```bash
grep -E "Traceback|Error|Killed|MemoryError|CUDA" logs/residual.log | tail -20
free -h
nvidia-smi
```

如果是 OOM, 跑命令前 `export CUDA_VISIBLE_DEVICES=0`, 减并发后重跑。

### 状态 C · 还在跑 🔄

特征:
- `ps` 输出有 `python run_residual_offline.py`
- `ls -la logs/residual.log` 时间戳是几秒前 (还在写)
- 可能在某个长 episode 处理中, 或在做 video 编码

继续等, 用更宽的监控:
```bash
tail -f logs/residual.log
```
直接 follow 不带 grep, 能看到 SVT-AV1 编码器的输出 (那是 video 编码进度)。

GPU 看负载:
```bash
watch -n 2 nvidia-smi
```

### 状态 D · 卡死了 ⚠️

特征:
- `ps` 显示进程在
- `ls -la logs/residual.log` 时间戳超过 5 分钟没动
- nvidia-smi 利用率 0%

处理:
```bash
# 拿 PID
PID=$(ps aux | grep run_residual_offline | grep -v grep | awk '{print $2}')
echo "PID: $PID"

# 看进程在做什么 (任意一个能用就行)
cat /proc/$PID/status | head -10
ls -la /proc/$PID/cwd
strace -p $PID -e trace=read,write -c 2>&1 | head &
sleep 5; kill %1

# 决定杀掉
kill $PID
sleep 2
ps aux | grep $PID | grep -v grep    # 确认死了
```

杀掉后跳到 §4 看怎么续跑。

---

## Step 3 · 跑完了 → 修 stats

```bash
DATASET_ROOT=~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual

python << 'PY'
import os, json, glob
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, pandas as pd
from pathlib import Path

root = Path("/root/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual")
files = sorted(glob.glob(str(root/"data/**/*.parquet"), recursive=True))
print(f"found {len(files)} parquet files")

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
    print(f"{c}: mean[:3]={arr.mean(0)[:3]}  std[:3]={arr.std(0)[:3]}")

with open(root/"meta"/"stats.json", "w") as f:
    json.dump(stats, f, indent=2)
es = root/"meta"/"episodes_stats.jsonl"
if not es.exists(): es.touch()
print("done")
PY

# 验证
cat "$DATASET_ROOT/meta/stats.json" | python -m json.tool | head -30
```

mean / std 不再是 `Infinity` 即可。然后 →

---

## Step 4 · 跑完了 → 启训练

```bash
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

sleep 30
tail -50 logs/train_a2c2_head.log | grep -E "step|loss"
```

---

## 状态 D / B 的额外: 卡死 / 崩了 后续跑

数据集已有部分 parquet 但不完整。两种选择:

**a) 用现有的部分数据训** (省时间, 假设 70 个 episode 已经够了):

```bash
# 检查现有 episode 数
ls ~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual/data/chunk-000/ | wc -l

# 如果有 50+ episodes, 直接 §3 修 stats + §4 训
```

但要修 `info.json` 里 `total_episodes` / `total_frames`:
```bash
ROOT=~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual
python << 'PY'
import json, pandas as pd, glob
from pathlib import Path
root = Path("/root/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual")
files = sorted(glob.glob(str(root/"data/**/*.parquet"), recursive=True))
total_eps = len(files)
total_frames = sum(len(pd.read_parquet(f)) for f in files)
info = json.load(open(root/"meta"/"info.json"))
info["total_episodes"] = total_eps
info["total_frames"] = total_frames
json.dump(info, open(root/"meta"/"info.json", "w"), indent=2)
print(f"updated: episodes={total_eps}, frames={total_frames}")
PY
```

**b) 从中断处续跑** (慢, 不推荐除非你必须 70+ ep):

需要改 `eval_libero/create_dataset_for_residualpolicy.py` 加一个 skip 已存在的 episode 的逻辑。这个改动较多, 不细写, 建议走 (a)。

---

## 一键诊断 + 自动续跑

```bash
cd ~/project/lq/a2c2-libero

PID=$(ps aux | grep run_residual_offline | grep -v grep | awk '{print $2}')
DS=~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual

echo "=== process status ==="
if [ -n "$PID" ]; then
  echo "RUNNING PID=$PID"
  ps -o pid,etime,pcpu,pmem,cmd -p $PID
else
  echo "NOT RUNNING"
fi

echo "=== log status ==="
ls -la logs/residual.log 2>/dev/null
echo "=== last 20 lines ==="
tail -20 logs/residual.log 2>/dev/null

echo "=== dataset progress ==="
N=$(ls "$DS/data/chunk-000/" 2>/dev/null | wc -l)
echo "parquet files: $N"

echo "=== verdict ==="
if [ -z "$PID" ] && [ "$N" -ge 60 ]; then
  echo "✅ DONE — 跑 §3 修 stats 然后 §4 训练"
elif [ -z "$PID" ] && [ "$N" -lt 60 ]; then
  echo "❌ CRASHED — 看 §2 状态 B"
elif [ -n "$PID" ]; then
  echo "🔄 STILL RUNNING — 等"
fi
```

把这段最后的 verdict 输出贴回来 (DONE / CRASHED / STILL RUNNING) 决定走哪一步。

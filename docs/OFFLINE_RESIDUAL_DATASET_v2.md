# 离线生成残差数据集 + 训练残差头 (v2)

适用: A100 服务器无外网 / 无 huggingface-cli / 想完全跑在本地缓存。
所有命令都在 A100 的 bash shell 里执行,假设你已经 `cd ~/project/lq/a2c2-libero`。

v2 比 v1 多修了 LeRobot 自己的 `get_safe_version` 调用,以及 BASE 数据集找不到的问题。

---

## 0 · 找出你 BASE 数据集真实在哪

```bash
cd ~/project/lq/a2c2-libero

find ~/.cache/huggingface/ -name "tasks.jsonl" 2>/dev/null
find ~/.cache/huggingface/ -name "info.json" -path "*meta*" 2>/dev/null
find ~/project/ -name "info.json" -path "*meta*" 2>/dev/null
```

记下输出。其中一条是你 SmolVLA 训练用的 demo,例如:
```
/root/.cache/huggingface/lerobot/lerobot/a2c2_dataset_v21/meta/tasks.jsonl
/root/project/lq/lerobot/data/xxx/meta/info.json
```

数据集真实路径有 2 种情况, 第 3 步分别处理:
- **A**: 在 `~/.cache/huggingface/lerobot/<owner>/<name>/`
- **B**: 在其他自定义位置

---

## 1 · 写最终版 wrapper `run_residual_offline.py`

替换之前的版本(多 stub 了 LeRobot 自己的版本检查):

```bash
cd ~/project/lq/a2c2-libero
cat > run_residual_offline.py << 'PYEOF'
"""Wrapper: HF + lerobot offline stubs for create_dataset_for_residualpolicy.py"""
import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import huggingface_hub
from huggingface_hub import HfApi
from types import SimpleNamespace


def _noop(*a, **k):
    return SimpleNamespace(
        url="", repo_id="", id="", sha="local",
        siblings=[], branches=[], converts=[], tags=[],
    )


# --- HF stubs ---
HfApi.create_repo = _noop
HfApi.repo_info = _noop
HfApi.list_repo_refs = _noop
HfApi.upload_file = _noop
HfApi.upload_folder = _noop
HfApi.upload_large_folder = _noop
HfApi.create_tag = _noop
HfApi.list_repo_tree = _noop
huggingface_hub.create_repo = _noop
huggingface_hub.repo_info = _noop
huggingface_hub.list_repo_refs = _noop
huggingface_hub.create_tag = _noop

# --- LeRobot stubs ---
import lerobot.datasets.utils as ds_utils
import lerobot.datasets.lerobot_dataset as lrd


def _safe_version(repo_id, version, **kw):
    return version


ds_utils.get_safe_version = _safe_version
if hasattr(lrd, "get_safe_version"):
    lrd.get_safe_version = _safe_version

try:
    import lerobot.datasets.backward_compatibility as bc
    if hasattr(bc, "get_safe_version"):
        bc.get_safe_version = _safe_version
except Exception:
    pass

print("[wrapper] HF + LeRobot version checks stubbed, running ...")
exec(compile(
    open("eval_libero/create_dataset_for_residualpolicy.py").read(),
    "eval_libero/create_dataset_for_residualpolicy.py",
    "exec",
))
PYEOF

ls -la run_residual_offline.py
```

---

## 2 · 改 `eval_libero/create_dataset_for_residualpolicy.py`

### 2.1 必改: UPLOAD_REPO_NAME 用合法 owner 格式

```bash
sed -i '14s|local/|lakesenberg/|' eval_libero/create_dataset_for_residualpolicy.py
sed -n '14p' eval_libero/create_dataset_for_residualpolicy.py
# 期望: UPLOAD_REPO_NAME = "lakesenberg/a2c2_dataset_v21-smolvla-residual"
```

### 2.2 二选一: 改 BASE_REPO_NAME 或加 `root` 参数

**情况 A — 数据集在 HF 缓存里**(例如 find 找到 `~/.cache/huggingface/lerobot/lerobot/a2c2_dataset_v21/`):

```bash
sed -i '13s|BASE_REPO_NAME.*|BASE_REPO_NAME = "lerobot/a2c2_dataset_v21"|' \
  eval_libero/create_dataset_for_residualpolicy.py

sed -n '13p' eval_libero/create_dataset_for_residualpolicy.py
# 期望: BASE_REPO_NAME = "lerobot/a2c2_dataset_v21"
```

如果你 find 找到的 owner 不是 `lerobot/`,把 `lerobot` 换成实际 owner 名。

**情况 B — 数据集在自定义位置**(例如 `/root/project/lq/data/my_demos`):

先看脚本里 LeRobotDataset 怎么构造的:
```bash
grep -n "LeRobotDataset(" eval_libero/create_dataset_for_residualpolicy.py | head
```

通常是:
```python
base_dataset = LeRobotDataset(BASE_REPO_NAME)
```

改成加 root:
```bash
ROOT="/root/project/lq/data/my_demos"   # 改成你实际路径

sed -i "s|LeRobotDataset(BASE_REPO_NAME)|LeRobotDataset(BASE_REPO_NAME, root=\"$ROOT\")|" \
  eval_libero/create_dataset_for_residualpolicy.py

grep -n "LeRobotDataset(BASE_REPO_NAME" eval_libero/create_dataset_for_residualpolicy.py
```

### 2.3 (可选) 检查 robot_type 和 base 一致

第 62 行 `robot_type="panda"`, 如果你 base 数据集不是 panda 改一下:
```bash
sed -n '62p' eval_libero/create_dataset_for_residualpolicy.py

# 改成跟你 base 一样, 例如 "bi_so_arm":
# sed -i '62s|robot_type="panda"|robot_type="bi_so_arm"|' \
#   eval_libero/create_dataset_for_residualpolicy.py
```

不一致也未必崩,但 LeRobot 某些版本会校验。先保持原样跑,失败再改。

---

## 3 · 跑生成残差数据集

```bash
mkdir -p logs
python run_residual_offline.py 2>&1 | tee logs/residual.log
```

正常的话末尾应该看到 `done` 或 `saved last episode`,中间不断打印进度。

预估时间(看 base 大小):
- 50 ep × 300 帧 ≈ 15k frames → 30 min
- 200 ep × 500 帧 ≈ 100k frames → 3 h

如果跑挂了,先看日志末尾:
```bash
tail -40 logs/residual.log
```

---

## 4 · 验证残差数据集生成成功

```bash
# 找新生成的数据集 (UPLOAD_REPO_NAME 对应路径)
ls ~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual/
ls ~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual/data/ | head
ls ~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual/meta/
```

应该看到 `data/`, `meta/`, `info.json`, 可能还有 `videos/`。

---

## 5 · 写训练 wrapper `run_train_offline.py`

```bash
cd ~/project/lq/a2c2-libero
cat > run_train_offline.py << 'PYEOF'
"""Wrapper to train_residual_transformer.py fully offline."""
import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("WANDB_MODE", "offline")

import huggingface_hub
from huggingface_hub import HfApi
from types import SimpleNamespace


def _noop(*a, **k):
    return SimpleNamespace(
        url="", repo_id="", id="", sha="local",
        siblings=[], branches=[], converts=[], tags=[],
    )


HfApi.create_repo = _noop
HfApi.repo_info = _noop
HfApi.list_repo_refs = _noop
HfApi.upload_file = _noop
HfApi.upload_folder = _noop
HfApi.upload_large_folder = _noop
HfApi.create_tag = _noop
HfApi.list_repo_tree = _noop
huggingface_hub.create_repo = _noop
huggingface_hub.repo_info = _noop
huggingface_hub.list_repo_refs = _noop

import lerobot.datasets.utils as ds_utils
import lerobot.datasets.lerobot_dataset as lrd


def _safe_version(repo_id, version, **kw):
    return version


ds_utils.get_safe_version = _safe_version
if hasattr(lrd, "get_safe_version"):
    lrd.get_safe_version = _safe_version
try:
    import lerobot.datasets.backward_compatibility as bc
    if hasattr(bc, "get_safe_version"):
        bc.get_safe_version = _safe_version
except Exception:
    pass

sys.argv = ["train_residual_transformer.py"] + sys.argv[1:]
print("[wrapper] HF + LeRobot version checks stubbed, starting training ...")
exec(compile(
    open("src/lerobot/scripts/train_residual_transformer.py").read(),
    "src/lerobot/scripts/train_residual_transformer.py",
    "exec",
))
PYEOF

ls -la run_train_offline.py
```

---

## 6 · 训练残差头

```bash
mkdir -p outputs/a2c2_head_v21

python run_train_offline.py \
  --policy.type=residual_transformer \
  --policy.repo_id=lakesenberg/a2c2_head_v21 \
  --dataset.repo_id=lakesenberg/a2c2_dataset_v21-smolvla-residual \
  --batch_size=64 \
  --num_workers=16 \
  --steps=200000 \
  --output_dir=outputs/a2c2_head_v21 \
  --job_name=a2c2_head_v21 \
  --wandb.enable=true \
  2>&1 | tee logs/train_a2c2_head.log
```

如果 wandb 起不来或不需要:
```bash
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

监控 loss 下降:
```bash
tail -f logs/train_a2c2_head.log | grep -E "loss|step"
```

A100 ~6 h(数据集小的话 3 h)。

---

## 7 · 训完产物

```bash
ls -la outputs/a2c2_head_v21/
ls -la outputs/a2c2_head_v21/checkpoints/ 2>/dev/null
```

ckpt 路径形如 `outputs/a2c2_head_v21/checkpoints/last/pretrained_model/` 或 `outputs/a2c2_head_v21/checkpoint.pt`。

---

## 8 · scp 到 4090 推理机

在 4090 上:
```bash
mkdir -p ~/a2c2-libero/outputs
scp -r root@192.168.3.103:/root/project/lq/a2c2-libero/outputs/a2c2_head_v21 \
       ~/a2c2-libero/outputs/

ls ~/a2c2-libero/outputs/a2c2_head_v21/
```

---

## 故障排查

### Q1. RevisionNotFoundError / get_safe_version 还在报

确认 wrapper 真的注入了:
```bash
grep "get_safe_version" run_residual_offline.py | wc -l
# 应该 >= 3
```

如果 wrapper 看着对但还报错,可能 lerobot 版本不同, get_safe_version 在别处。试:
```bash
grep -rn "get_safe_version" src/lerobot/datasets/ | head
```

把所有出现位置报回来,我加更多 stub。

### Q2. tasks.jsonl 找不到

数据集结构不完整。看完整结构:
```bash
find ~/.cache/huggingface/lerobot/ -type f -path "*meta*" | head -30
```

如果 `meta/tasks.jsonl` 真的没有, 数据集不是 v3 完整格式, 需要补一个空的:
```bash
DATASET_ROOT="<你 base 数据集真实路径>"
[ ! -f "$DATASET_ROOT/meta/tasks.jsonl" ] && echo '{"task_index":0,"task":"manipulation"}' > "$DATASET_ROOT/meta/tasks.jsonl"
```

### Q3. 数据集格式 v3.0 警告

无害,backward-compatible。如果训练阶段卡住,转换:
```bash
python -m lerobot.datasets.v21.convert_dataset_v20_to_v21 \
  --repo-id=lakesenberg/a2c2_dataset_v21-smolvla-residual
```

### Q4. CUDA OOM

减并发:
```bash
sed -i 's|image_writer_processes=10|image_writer_processes=2|' \
  eval_libero/create_dataset_for_residualpolicy.py
sed -i 's|image_writer_threads=20|image_writer_threads=4|' \
  eval_libero/create_dataset_for_residualpolicy.py
```

### Q5. wandb login 卡住

直接 offline:
```bash
export WANDB_MODE=offline
```

或参数 `--wandb.enable=false`。

---

## 完整一行流程

```bash
# 0. 准备 wrapper
cat > run_residual_offline.py << 'PYEOF'
... (上面 §1 的内容) ...
PYEOF

# 1. 改脚本顶部 3 个变量 + 加 root (如果需要)
sed -i '14s|local/|lakesenberg/|' eval_libero/create_dataset_for_residualpolicy.py
sed -i '13s|BASE_REPO_NAME.*|BASE_REPO_NAME = "lerobot/a2c2_dataset_v21"|' eval_libero/create_dataset_for_residualpolicy.py

# 2. 跑 → 残差数据集
python run_residual_offline.py 2>&1 | tee logs/residual.log

# 3. 验证
ls ~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual/

# 4. 训残差头 (需要 run_train_offline.py wrapper)
python run_train_offline.py \
  --policy.type=residual_transformer \
  --policy.repo_id=lakesenberg/a2c2_head_v21 \
  --dataset.repo_id=lakesenberg/a2c2_dataset_v21-smolvla-residual \
  --batch_size=64 --num_workers=16 --steps=200000 \
  --output_dir=outputs/a2c2_head_v21 --wandb.enable=false \
  2>&1 | tee logs/train_a2c2_head.log
```

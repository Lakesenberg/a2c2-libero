# 离线生成残差数据集 + 训练残差头 (v3)

适用: A100 服务器无外网,LeRobot 内部 `get_safe_version` 也要 stub。
v3 比 v2 改进: monkey-patch 走 sys.modules 暴力扫,所有引用一次替换。
所有命令在 A100 bash 里执行,假设你已经 `cd ~/project/lq/a2c2-libero`。

---

## 0 · 找出 BASE 数据集真实路径

```bash
cd ~/project/lq/a2c2-libero
find ~/.cache/huggingface/ -name "tasks.jsonl" 2>/dev/null
```

记下输出。例如看到:
```
/root/.cache/huggingface/lerobot/lerobot/a2c2_dataset_v21/meta/tasks.jsonl
```

→ owner = `lerobot`, name = `a2c2_dataset_v21` → `BASE_REPO_NAME = "lerobot/a2c2_dataset_v21"`

---

## 1 · 改 `eval_libero/create_dataset_for_residualpolicy.py` 顶部 3 个变量

### 1.1 BASE_REPO_NAME (行 13)

```bash
sed -i '13s|BASE_REPO_NAME.*|BASE_REPO_NAME = "lerobot/a2c2_dataset_v21"|' \
  eval_libero/create_dataset_for_residualpolicy.py
```

### 1.2 UPLOAD_REPO_NAME 改成合法 owner (行 14)

```bash
sed -i '14s|local/|lakesenberg/|' \
  eval_libero/create_dataset_for_residualpolicy.py
```

### 1.3 BASE_POLICY_NAME (行 15) 通常已经对,确认一下

```bash
sed -n '12,16p' eval_libero/create_dataset_for_residualpolicy.py
```

期望输出:
```
BASE_REPO_NAME   = "lerobot/a2c2_dataset_v21"
UPLOAD_REPO_NAME = "lakesenberg/a2c2_dataset_v21-smolvla-residual"
BASE_POLICY_NAME = "/root/project/lq/lerobot/outputs/train/smolvla_a2c2_v21/checkpoints/020000/pretrained_model"
```

---

## 2 · 写 wrapper `run_residual_offline.py` (v3 暴力扫)

```bash
cat > run_residual_offline.py << 'PYEOF'
"""Wrapper: aggressive monkey-patch for offline use.
Walks sys.modules and replaces every get_safe_version / HF API call."""
import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import huggingface_hub
from huggingface_hub import HfApi
import lerobot.datasets.utils
import lerobot.datasets.lerobot_dataset
try:
    import lerobot.datasets.backward_compatibility
except Exception:
    pass

from types import SimpleNamespace


def _safe_version(repo_id, version, **kw):
    return version


def _noop(*a, **k):
    return SimpleNamespace(
        url="", repo_id="", id="", sha="local",
        siblings=[], branches=[], converts=[], tags=[],
    )


for name in ("create_repo", "repo_info", "list_repo_refs", "upload_file",
             "upload_folder", "upload_large_folder", "create_tag",
             "list_repo_tree", "delete_repo", "delete_file"):
    setattr(HfApi, name, _noop)
    if hasattr(huggingface_hub, name):
        setattr(huggingface_hub, name, _noop)

patched = 0
for mod_name, mod in list(sys.modules.items()):
    if mod is None:
        continue
    try:
        if hasattr(mod, "get_safe_version"):
            setattr(mod, "get_safe_version", _safe_version)
            patched += 1
    except Exception:
        pass

print(f"[wrapper] patched get_safe_version in {patched} modules")
print(f"[wrapper] HF API stubbed, running ...")

exec(compile(
    open("eval_libero/create_dataset_for_residualpolicy.py").read(),
    "eval_libero/create_dataset_for_residualpolicy.py",
    "exec",
))
PYEOF

ls -la run_residual_offline.py
```

---

## 3 · 跑生成残差数据集

```bash
mkdir -p logs
python run_residual_offline.py 2>&1 | tee logs/residual.log
```

启动时应该看到 `[wrapper] patched get_safe_version in 2 modules` 之类。

---

## 4 · 验证生成成功

```bash
ls -la ~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual/
ls -la ~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual/data/ | head
ls -la ~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual/meta/
```

---

## 5 · 写训练 wrapper `run_train_offline.py`

```bash
cat > run_train_offline.py << 'PYEOF'
"""Wrapper: aggressive monkey-patch for train_residual_transformer.py."""
import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("WANDB_MODE", "offline")

import huggingface_hub
from huggingface_hub import HfApi
import lerobot.datasets.utils
import lerobot.datasets.lerobot_dataset
try:
    import lerobot.datasets.backward_compatibility
except Exception:
    pass

from types import SimpleNamespace


def _safe_version(repo_id, version, **kw):
    return version


def _noop(*a, **k):
    return SimpleNamespace(
        url="", repo_id="", id="", sha="local",
        siblings=[], branches=[], converts=[], tags=[],
    )


for name in ("create_repo", "repo_info", "list_repo_refs", "upload_file",
             "upload_folder", "upload_large_folder", "create_tag",
             "list_repo_tree", "delete_repo", "delete_file"):
    setattr(HfApi, name, _noop)
    if hasattr(huggingface_hub, name):
        setattr(huggingface_hub, name, _noop)

patched = 0
for mod_name, mod in list(sys.modules.items()):
    if mod is None:
        continue
    try:
        if hasattr(mod, "get_safe_version"):
            setattr(mod, "get_safe_version", _safe_version)
            patched += 1
    except Exception:
        pass

print(f"[wrapper] patched get_safe_version in {patched} modules")
print(f"[wrapper] HF API stubbed, starting training ...")

sys.argv = ["train_residual_transformer.py"] + sys.argv[1:]
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
  --wandb.enable=false \
  2>&1 | tee logs/train_a2c2_head.log
```

监控:
```bash
tail -f logs/train_a2c2_head.log | grep -E "loss|step"
```

---

## 7 · scp 到 4090 推理机

在 4090 上:
```bash
mkdir -p ~/a2c2-libero/outputs
scp -r root@192.168.3.103:/root/project/lq/a2c2-libero/outputs/a2c2_head_v21 \
       ~/a2c2-libero/outputs/

ls ~/a2c2-libero/outputs/a2c2_head_v21/
```

---

## 故障排查

### Q1. 还是报 `RevisionNotFoundError`

wrapper 没生效,看 lerobot 里 `get_safe_version` 是怎么 import 的:

```bash
grep -n "get_safe_version\|from .utils" \
  src/lerobot/datasets/lerobot_dataset.py \
  src/lerobot/datasets/utils.py \
  | head -20
```

把输出贴出来,我看到具体 import 形式后再加 stub。

如果是 `from .utils import get_safe_version` (拷贝引用) — wrapper 的 sys.modules 扫描应该已经能覆盖,因为 `lerobot.datasets.lerobot_dataset` 在 sys.modules 里。

如果是 `import .utils` 然后 `utils.get_safe_version(...)` — 那 stub `lerobot.datasets.utils.get_safe_version` 就够了。

### Q2. 启动时 `[wrapper] patched get_safe_version in 0 modules`

说明 lerobot 模块还没被 import 进 sys.modules。改 wrapper,在 `import lerobot.datasets.utils` 前面加:

```python
import lerobot
import lerobot.datasets
import lerobot.datasets.utils
import lerobot.datasets.lerobot_dataset
```

### Q3. tasks.jsonl 找不到

补一个空的:
```bash
DATASET_ROOT="/root/.cache/huggingface/lerobot/lerobot/a2c2_dataset_v21"
[ ! -f "$DATASET_ROOT/meta/tasks.jsonl" ] && \
  echo '{"task_index":0,"task":"manipulation"}' > "$DATASET_ROOT/meta/tasks.jsonl"
```

### Q4. CUDA OOM

减并发:
```bash
sed -i 's|image_writer_processes=10|image_writer_processes=2|' \
  eval_libero/create_dataset_for_residualpolicy.py
sed -i 's|image_writer_threads=20|image_writer_threads=4|' \
  eval_libero/create_dataset_for_residualpolicy.py
```

### Q5. v3.0 格式警告

无害,可忽略。如果训练阶段报错才转换:
```bash
python -m lerobot.datasets.v21.convert_dataset_v20_to_v21 \
  --repo-id=lakesenberg/a2c2_dataset_v21-smolvla-residual
```

---

## 完整一次到位脚本

复制这段整段贴到 A100 终端,自动完成 1-3 步:

```bash
cd ~/project/lq/a2c2-libero

# 找 BASE
find ~/.cache/huggingface/ -name "tasks.jsonl" 2>/dev/null

# 改两个变量 (按 find 结果调整 lerobot/a2c2_dataset_v21)
sed -i '13s|BASE_REPO_NAME.*|BASE_REPO_NAME = "lerobot/a2c2_dataset_v21"|' \
  eval_libero/create_dataset_for_residualpolicy.py
sed -i '14s|local/|lakesenberg/|' \
  eval_libero/create_dataset_for_residualpolicy.py

# 写 wrapper (复制本文档第 2 节内容)
# ...

# 跑
mkdir -p logs
python run_residual_offline.py 2>&1 | tee logs/residual.log
tail -30 logs/residual.log
```

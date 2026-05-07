# 用旧版 LeRobot (0.3.x / 0.4.x) 训练残差头

可用版本: `0.1.0, 0.3.2, 0.3.3, 0.4.0, 0.4.1, 0.4.2, 0.4.3, 0.4.4` (无 0.1.7)
策略: 先跑诊断脚本找出能加载你数据集的最高版本, 然后按那个版本布局移植代码 + 训残差头。
所有命令在 A100 bash 里执行, 假设 `cd ~/project/lq/a2c2-libero`。

---

## 0 · 创建 conda env

```bash
conda create -n lerobot_v21 python=3.10 -y
conda activate lerobot_v21
pip install num2words accelerate safetensors "transformers>=4.40,<4.52" draccus jsonlines wandb
```

---

## 1 · 诊断: 找出哪个版本能加载你的数据集

```bash
conda activate lerobot_v21

for V in 0.4.4 0.4.0 0.3.3 0.3.2; do
  echo "=== testing lerobot==$V ==="
  pip install -q "lerobot==$V" 2>&1 | tail -3
  python -c "
import os; os.environ['HF_HUB_OFFLINE']='1'
import lerobot
print('lerobot:', lerobot.__version__)
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset('lerobot/a2c2_dataset_v21')
    print('  OK (lerobot.common.datasets), n_eps=', getattr(ds, 'num_episodes', 'N/A'))
except ImportError:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset('lerobot/a2c2_dataset_v21')
        print('  OK (lerobot.datasets), n_eps=', getattr(ds, 'num_episodes', 'N/A'))
    except Exception as e2:
        print('  FAIL:', str(e2)[:200])
except Exception as e:
    print('  FAIL:', str(e)[:200])
" 2>&1
done
```

把第一个 `OK` 的版本号记下来。下文用 `$V` 代表那个版本号 (例如 0.4.4)。

如果四个全 FAIL, 通常是<b>数据集本身的格式问题</b>而不是 lerobot 版本问题, 跳到底部 "故障排查 Q1"。

---

## 2 · 锁定版本 + 看代码目录布局

```bash
V=0.4.4   # 改成上一步成功的版本
pip install -q lerobot==$V

LEROBOT_V21=$(python -c "import lerobot, os; print(os.path.dirname(lerobot.__file__))")
echo "lerobot path: $LEROBOT_V21"

# 看 policies 目录在哪 (v0.3.x: lerobot/common/policies, v0.4.x: 不一定)
ls "$LEROBOT_V21/" | head
[ -d "$LEROBOT_V21/policies" ] && echo "POLICIES_DIR=$LEROBOT_V21/policies"
[ -d "$LEROBOT_V21/common/policies" ] && echo "POLICIES_DIR=$LEROBOT_V21/common/policies"

# 看 datasets import 路径
[ -f "$LEROBOT_V21/common/datasets/lerobot_dataset.py" ] && echo "DATASETS_PATH=lerobot.common.datasets"
[ -f "$LEROBOT_V21/datasets/lerobot_dataset.py" ] && echo "DATASETS_PATH=lerobot.datasets"
```

记下输出里的 `POLICIES_DIR` 和 `DATASETS_PATH` (下文用 `$POL` 和 `$DSP`)。

---

## 3 · 移植 residual_transformer 自定义代码

```bash
# 先在 fork 里找代码
find ~/project/lq/a2c2-libero/src/ -name "*.py" \
  | xargs grep -l "residual_transformer\|ResidualTransformer" 2>/dev/null
```

通常会看到这几个文件:
```
src/lerobot/policies/residual_transformer/__init__.py
src/lerobot/policies/residual_transformer/configuration_residual_transformer.py
src/lerobot/policies/residual_transformer/modeling_residual_transformer.py
src/lerobot/scripts/train_residual_transformer.py
```

把 policy 目录拷到 v2.1 lerobot:

```bash
POL=$LEROBOT_V21/common/policies     # 或 $LEROBOT_V21/policies, 看 §2 输出
cp -r ~/project/lq/a2c2-libero/src/lerobot/policies/residual_transformer "$POL/"
ls "$POL/residual_transformer/"
```

把训练脚本拷过去:

```bash
cp ~/project/lq/a2c2-libero/src/lerobot/scripts/train_residual_transformer.py \
   "$LEROBOT_V21/scripts/"
ls "$LEROBOT_V21/scripts/train_residual_transformer.py"
```

---

## 4 · 把 fork v3 的 import 改成 v2.1 路径

v3 fork: `from lerobot.policies.X` / `from lerobot.datasets.X`
v2.1: `from lerobot.common.policies.X` / `from lerobot.common.datasets.X`

```bash
# 改 residual_transformer policy
find "$POL/residual_transformer/" -name "*.py" -exec \
  sed -i 's|from lerobot\.policies|from lerobot.common.policies|g; s|from lerobot\.datasets|from lerobot.common.datasets|g' {} \;

# 改训练脚本
sed -i 's|from lerobot\.policies|from lerobot.common.policies|g; s|from lerobot\.datasets|from lerobot.common.datasets|g' \
  "$LEROBOT_V21/scripts/train_residual_transformer.py"

# 看是否还有未替换的 import
grep -rn "from lerobot\." "$POL/residual_transformer/" "$LEROBOT_V21/scripts/train_residual_transformer.py" | head
```

如果上面 grep 还有 `from lerobot.X`, 再加对应 sed 替换。

---

## 5 · 注册 residual_transformer 到 factory

```bash
sed -n '1,80p' "$POL/factory.py"
```

找到形如 `elif policy_type == "act":` 的分支, 加一段:

```python
elif policy_type == "residual_transformer":
    from .residual_transformer.modeling_residual_transformer import ResidualTransformerPolicy
    return ResidualTransformerPolicy
```

可以手动 vim/nano 编辑, 或自动 patch:

```bash
python << PY
import os, re
fp = "$POL/factory.py"
src = open(fp).read()
if "residual_transformer" not in src:
    insert = '''
    elif policy_type == "residual_transformer":
        from .residual_transformer.modeling_residual_transformer import ResidualTransformerPolicy
        return ResidualTransformerPolicy
'''
    src = re.sub(r'(\n\s+)(elif policy_type ==)', insert + r'\1\2', src, count=1)
    open(fp, "w").write(src)
    print("patched factory.py")
else:
    print("already patched")
PY
```

确认:
```bash
grep -n "residual_transformer" "$POL/factory.py"
```

---

## 6 · 准备 v21 版本的 create_dataset 脚本

```bash
mkdir -p ~/project/lq/a2c2-libero/eval_libero_v21
cp ~/project/lq/a2c2-libero/eval_libero/create_dataset_for_residualpolicy.py \
   ~/project/lq/a2c2-libero/eval_libero_v21/

# 改 import 路径
sed -i 's|from lerobot\.datasets|from lerobot.common.datasets|g; s|from lerobot\.policies|from lerobot.common.policies|g' \
  ~/project/lq/a2c2-libero/eval_libero_v21/create_dataset_for_residualpolicy.py

# 改 BASE_REPO_NAME (按你 find 的真实路径)
sed -i '13s|BASE_REPO_NAME.*|BASE_REPO_NAME = "lerobot/a2c2_dataset_v21"|' \
  ~/project/lq/a2c2-libero/eval_libero_v21/create_dataset_for_residualpolicy.py

# 改 UPLOAD_REPO_NAME 为合法 owner
sed -i '14s|local/|lakesenberg/|' \
  ~/project/lq/a2c2-libero/eval_libero_v21/create_dataset_for_residualpolicy.py

sed -n '12,16p' ~/project/lq/a2c2-libero/eval_libero_v21/create_dataset_for_residualpolicy.py
```

---

## 7 · v21 wrapper

```bash
cd ~/project/lq/a2c2-libero
cat > run_residual_offline_v21.py << 'PYEOF'
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


for name in ("create_repo", "repo_info", "list_repo_refs", "upload_file",
             "upload_folder", "upload_large_folder", "create_tag",
             "list_repo_tree", "delete_repo", "delete_file"):
    setattr(HfApi, name, _noop)
    if hasattr(huggingface_hub, name):
        setattr(huggingface_hub, name, _noop)

print("[v21 wrapper] HF stubbed, running ...")
exec(compile(
    open("eval_libero_v21/create_dataset_for_residualpolicy.py").read(),
    "eval_libero_v21/create_dataset_for_residualpolicy.py",
    "exec",
))
PYEOF


cat > run_train_offline_v21.py << 'PYEOF'
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


for name in ("create_repo", "repo_info", "list_repo_refs", "upload_file",
             "upload_folder", "upload_large_folder", "create_tag"):
    setattr(HfApi, name, _noop)
    if hasattr(huggingface_hub, name):
        setattr(huggingface_hub, name, _noop)

import lerobot
TRAIN_PATH = os.path.join(os.path.dirname(lerobot.__file__),
                          "scripts", "train_residual_transformer.py")
print(f"[v21 wrapper] running {TRAIN_PATH}")
sys.argv = ["train_residual_transformer.py"] + sys.argv[1:]
exec(compile(open(TRAIN_PATH).read(), TRAIN_PATH, "exec"))
PYEOF
```

---

## 8 · 跑生成残差数据集

```bash
mkdir -p logs
python run_residual_offline_v21.py 2>&1 | tee logs/residual_v21.log
tail -30 logs/residual_v21.log
ls -la ~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual/data/ | head
```

---

## 9 · 跑训练残差头

```bash
mkdir -p outputs/a2c2_head_v21

python run_train_offline_v21.py \
  --policy.type=residual_transformer \
  --policy.repo_id=lakesenberg/a2c2_head_v21 \
  --dataset.repo_id=lakesenberg/a2c2_dataset_v21-smolvla-residual \
  --batch_size=64 \
  --num_workers=16 \
  --steps=200000 \
  --output_dir=outputs/a2c2_head_v21 \
  --job_name=a2c2_head_v21 \
  --wandb.enable=false \
  2>&1 | tee logs/train_a2c2_head_v21.log

# 监控
tail -f logs/train_a2c2_head_v21.log | grep -E "loss|step"
```

---

## 10 · scp 训完 ckpt 到 4090

A100:
```bash
ls outputs/a2c2_head_v21/
```

4090:
```bash
mkdir -p ~/a2c2-libero/outputs
scp -r root@192.168.3.103:/root/project/lq/a2c2-libero/outputs/a2c2_head_v21 \
       ~/a2c2-libero/outputs/
```

---

## 故障排查

### Q1. 4 个版本全 FAIL, 都说找不到 tasks.jsonl

数据集真的缺这个文件, 不是 lerobot 版本问题。先补:
```bash
DATASET_ROOT="/root/.cache/huggingface/lerobot/lerobot/a2c2_dataset_v21"
python << PY
import json, os
root = "$DATASET_ROOT/meta"
info = json.load(open(f"{root}/info.json"))
tasks = info.get("tasks", [])
with open(f"{root}/tasks.jsonl", "w") as f:
    if isinstance(tasks, list) and tasks:
        for i, t in enumerate(tasks):
            if isinstance(t, dict):
                f.write(json.dumps(t) + "\n")
            else:
                f.write(json.dumps({"task_index": i, "task": str(t)}) + "\n")
    else:
        f.write('{"task_index": 0, "task": "manipulation"}\n')
es = f"{root}/episodes_stats.jsonl"
if not os.path.exists(es): open(es, "a").close()
print("done")
PY
```

补完再回到 §1 的诊断脚本重跑。

### Q2. residual_transformer 找不到 / factory 没注册成功

```bash
grep -rn "residual_transformer" $LEROBOT_V21/common/policies/factory.py
```

如果输出空, sed patch 没生效, 用 nano 手动编辑 factory.py 加分支:

```bash
nano $LEROBOT_V21/common/policies/factory.py
# 找 elif policy_type == "act": (或类似), 在它前面加:
#     elif policy_type == "residual_transformer":
#         from .residual_transformer.modeling_residual_transformer import ResidualTransformerPolicy
#         return ResidualTransformerPolicy
```

### Q3. import 错误, 比如 `cannot import name 'XXX' from 'lerobot.common.datasets.utils'`

fork 里的 residual_transformer 用了 v3 lerobot 独有的 utility 函数。看错误 line, 决定是:
- 把 v3 的工具函数从 fork 拷过来到 v2.1 lerobot 对应位置
- 或者改写 residual_transformer 不依赖那个函数

```bash
# 看错误行, 找出缺的符号 NAME
grep -rn "def NAME\|NAME =" ~/project/lq/a2c2-libero/src/lerobot/datasets/
# 找到后拷过去
```

### Q4. dataset features mismatch

v2.1 LeRobotDataset.create() 的 features 字典 schema 跟 v3 不同。看 train_residual_transformer.py 里 features 那段, 可能要把 `{"dtype":"float32", "shape":(7,)}` 改成 `{"dtype":"float32", "shape":(7,), "names":["x"]}` 这样的 v2.1 格式。

---

## 一行总览

```bash
conda create -n lerobot_v21 python=3.10 -y && conda activate lerobot_v21
pip install num2words accelerate safetensors "transformers>=4.40,<4.52" draccus jsonlines wandb

# 找版本
for V in 0.4.4 0.4.0 0.3.3 0.3.2; do echo "=== $V ==="; pip install -q lerobot==$V; python -c "import lerobot, os; os.environ['HF_HUB_OFFLINE']='1'; from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; LeRobotDataset('lerobot/a2c2_dataset_v21'); print('OK')" 2>&1 | tail -3; done

# 锁定 + 移植 (假设 0.4.4 通了)
pip install lerobot==0.4.4
LEROBOT_V21=$(python -c "import lerobot, os; print(os.path.dirname(lerobot.__file__))")
POL=$LEROBOT_V21/common/policies
cp -r ~/project/lq/a2c2-libero/src/lerobot/policies/residual_transformer "$POL/"
cp ~/project/lq/a2c2-libero/src/lerobot/scripts/train_residual_transformer.py "$LEROBOT_V21/scripts/"
find "$POL/residual_transformer/" -name "*.py" -exec sed -i 's|from lerobot\.policies|from lerobot.common.policies|g; s|from lerobot\.datasets|from lerobot.common.datasets|g' {} \;
sed -i 's|from lerobot\.policies|from lerobot.common.policies|g; s|from lerobot\.datasets|from lerobot.common.datasets|g' "$LEROBOT_V21/scripts/train_residual_transformer.py"

# 跑生成 + 训练 (用上面的 wrapper 文件)
python run_residual_offline_v21.py 2>&1 | tee logs/residual_v21.log
python run_train_offline_v21.py --policy.type=residual_transformer \
  --policy.repo_id=lakesenberg/a2c2_head_v21 \
  --dataset.repo_id=lakesenberg/a2c2_dataset_v21-smolvla-residual \
  --batch_size=64 --num_workers=16 --steps=200000 \
  --output_dir=outputs/a2c2_head_v21 --wandb.enable=false \
  2>&1 | tee logs/train_a2c2_head_v21.log
```

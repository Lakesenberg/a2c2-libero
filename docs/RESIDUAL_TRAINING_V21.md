# 用 LeRobot v2.1 训练残差头 (兼容旧格式数据集)

适用: A100 服务器, 数据集是 v2.0/v2.1 格式 (缺 tasks.jsonl 导致 v3.0 加载失败), 想用兼容版本 lerobot 训练残差头。

策略: <b>建独立 conda env, 装 lerobot v2.1 era (0.1.7), 把 fork 里的 residual_transformer 自定义代码移植过来</b>。
现有 v3.0 lerobot fork 不动 (留作 SmolVLA 推理), v2.1 env 只用来训练残差头。

所有命令在 A100 bash 里执行。

---

## 0 · 看一下你 fork 里有哪些 residual 相关代码

```bash
cd ~/project/lq/a2c2-libero
find src/ -name "*residual*" -o -name "*a2c2*" 2>/dev/null
find eval_libero/ -name "*.py" 2>/dev/null
```

记下输出。典型情况:
```
src/lerobot/policies/residual_transformer/
src/lerobot/policies/residual_transformer/__init__.py
src/lerobot/policies/residual_transformer/configuration_residual_transformer.py
src/lerobot/policies/residual_transformer/modeling_residual_transformer.py
src/lerobot/scripts/train_residual_transformer.py
eval_libero/create_dataset_for_residualpolicy.py
```

---

## 1 · 创建 v2.1 conda env

```bash
conda create -n lerobot_v21 python=3.10 -y
conda activate lerobot_v21

# v2.1 era 的 lerobot
pip install "lerobot==0.1.7"

# 残差头训练需要的额外包
pip install num2words "accelerate>=0.26" safetensors "transformers>=4.40,<4.52"
pip install draccus jsonlines wandb
```

确认装好:
```bash
python -c "import lerobot; print('lerobot:', lerobot.__version__); print('path:', lerobot.__file__)"
python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; print('LeRobotDataset OK')"
```

期望输出 `0.1.7` 和 `LeRobotDataset OK`。

---

## 2 · 把 fork 的自定义代码<b>叠加</b>到 v2.1 lerobot

v2.1 lerobot 没有 `residual_transformer` policy, 要把 fork 里的复制过去。

### 2.1 找 v2.1 lerobot 的安装路径

```bash
LEROBOT_V21_PATH=$(python -c "import lerobot, os; print(os.path.dirname(lerobot.__file__))")
echo "v21 lerobot path: $LEROBOT_V21_PATH"
```

期望形如 `/root/miniconda/envs/lerobot_v21/lib/python3.10/site-packages/lerobot`。

### 2.2 拷 residual_transformer policy 代码

```bash
cp -r ~/project/lq/a2c2-libero/src/lerobot/policies/residual_transformer \
      $LEROBOT_V21_PATH/common/policies/

# v2.1 lerobot 的 policy 在 lerobot/common/policies/, 不是 lerobot/policies/
# 如果 fork 是 lerobot/policies/, 拷到 v2.1 的 lerobot/common/policies/
ls $LEROBOT_V21_PATH/common/policies/residual_transformer/
```

### 2.3 拷 train_residual_transformer 脚本

```bash
cp ~/project/lq/a2c2-libero/src/lerobot/scripts/train_residual_transformer.py \
   $LEROBOT_V21_PATH/scripts/

ls $LEROBOT_V21_PATH/scripts/train_residual_transformer.py
```

### 2.4 让 v2.1 lerobot 注册新 policy

v2.1 通常用 factory pattern。看一下:

```bash
grep -rn "ACTPolicy\|DiffusionPolicy" \
  $LEROBOT_V21_PATH/common/policies/factory.py 2>/dev/null | head
```

把 `residual_transformer` 加进去:

```bash
python << PY
import os, re
factory_path = "$LEROBOT_V21_PATH/common/policies/factory.py"
src = open(factory_path).read()

if "residual_transformer" not in src:
    # 在第一个 elif policy_type == 之前加 residual 的分支 (粗略 hack)
    insert = '''
    elif policy_type == "residual_transformer":
        from .residual_transformer.modeling_residual_transformer import ResidualTransformerPolicy
        return ResidualTransformerPolicy
'''
    src = re.sub(r"(elif policy_type ==)", insert + "    \\1", src, count=1)
    open(factory_path, "w").write(src)
    print("patched factory.py")
else:
    print("already patched")
PY
```

如果 factory 路径或形式不对, 看脚本顶部 import 即可:
```bash
head -40 $LEROBOT_V21_PATH/common/policies/factory.py
```

把输出贴回来我帮你写精确 patch。

---

## 3 · 拷 create_dataset_for_residualpolicy.py 脚本

```bash
mkdir -p ~/project/lq/a2c2-libero/eval_libero_v21
cp ~/project/lq/a2c2-libero/eval_libero/create_dataset_for_residualpolicy.py \
   ~/project/lq/a2c2-libero/eval_libero_v21/

# v2.1 的 LeRobotDataset import 路径不同, 改一下
sed -i 's|from lerobot.datasets|from lerobot.common.datasets|g' \
  ~/project/lq/a2c2-libero/eval_libero_v21/create_dataset_for_residualpolicy.py

grep -n "from lerobot" ~/project/lq/a2c2-libero/eval_libero_v21/create_dataset_for_residualpolicy.py
```

---

## 4 · 改脚本顶部 3 个变量

```bash
cd ~/project/lq/a2c2-libero/eval_libero_v21
sed -i '13s|BASE_REPO_NAME.*|BASE_REPO_NAME = "lerobot/a2c2_dataset_v21"|' create_dataset_for_residualpolicy.py
sed -i '14s|local/|lakesenberg/|'                                          create_dataset_for_residualpolicy.py
sed -n '12,16p' create_dataset_for_residualpolicy.py
```

---

## 5 · 写 v2.1 wrapper (offline + monkey-patch)

v2.1 lerobot 没有 `get_safe_version` 这种检查, wrapper 简单很多:

```bash
cd ~/project/lq/a2c2-libero
cat > run_residual_offline_v21.py << 'PYEOF'
"""Wrapper: run create_dataset_for_residualpolicy.py with HF API stubbed,
using v2.1 lerobot."""
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

print("[wrapper v21] HF API stubbed, running ...")
exec(compile(
    open("eval_libero_v21/create_dataset_for_residualpolicy.py").read(),
    "eval_libero_v21/create_dataset_for_residualpolicy.py",
    "exec",
))
PYEOF
```

---

## 6 · 跑生成残差数据集 (在 v2.1 env 里)

```bash
conda activate lerobot_v21
cd ~/project/lq/a2c2-libero

mkdir -p logs
python run_residual_offline_v21.py 2>&1 | tee logs/residual_v21.log
tail -30 logs/residual_v21.log
```

应该比 v3.0 直接成功, 因为 v2.1 不要求 tasks.jsonl 存在。

---

## 7 · 验证生成

```bash
ls -la ~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual/
ls -la ~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual/data/ | head
```

---

## 8 · 写训练 wrapper

```bash
cat > run_train_offline_v21.py << 'PYEOF'
"""Wrapper for train_residual_transformer.py on v2.1 lerobot."""
import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("WANDB_MODE", "offline")

import huggingface_hub
from huggingface_hub import HfApi
from types import SimpleNamespace


def _noop(*a, **k):
    return SimpleNamespace(url="", repo_id="", id="", sha="local",
                           siblings=[], branches=[], converts=[], tags=[])


for name in ("create_repo", "repo_info", "list_repo_refs", "upload_file",
             "upload_folder", "upload_large_folder", "create_tag"):
    setattr(HfApi, name, _noop)
    if hasattr(huggingface_hub, name):
        setattr(huggingface_hub, name, _noop)

import lerobot
TRAIN_PATH = os.path.join(os.path.dirname(lerobot.__file__),
                          "scripts", "train_residual_transformer.py")
print(f"[wrapper v21] running {TRAIN_PATH}")
sys.argv = ["train_residual_transformer.py"] + sys.argv[1:]
exec(compile(open(TRAIN_PATH).read(), TRAIN_PATH, "exec"))
PYEOF
```

---

## 9 · 训练残差头

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
```

监控:
```bash
tail -f logs/train_a2c2_head_v21.log | grep -E "loss|step"
```

---

## 10 · 训完后 scp 给 4090

A100 上:
```bash
ls outputs/a2c2_head_v21/
```

4090 上:
```bash
mkdir -p ~/a2c2-libero/outputs
scp -r root@192.168.3.103:/root/project/lq/a2c2-libero/outputs/a2c2_head_v21 \
       ~/a2c2-libero/outputs/
```

注意: 4090 推理时<b>用的是 v3.0 lerobot fork</b>, 加载 v2.1 训出的 ckpt 应该兼容 (residual_transformer 自定义模型, 不依赖 lerobot 内部数据集 API)。如果加载报错, 给 4090 也建个 v2.1 env 即可。

---

## 故障排查

### Q1. `factory.py` 加 patch 失败 / policy_type 找不到 residual_transformer

直接看 v2.1 factory 文件:
```bash
LEROBOT_V21_PATH=$(python -c "import lerobot, os; print(os.path.dirname(lerobot.__file__))")
sed -n '1,80p' $LEROBOT_V21_PATH/common/policies/factory.py
```

把输出贴回来, 我写精确 patch。

### Q2. import 错误 `cannot import name 'XXX' from lerobot.common.policies.residual_transformer`

fork 里的 residual_transformer 代码可能用了 v3.0 lerobot 的 API。看错误最具体那行, 改成 v2.1 等价的 API。

最常见 import 重定向:
- `from lerobot.policies.X` → `from lerobot.common.policies.X`
- `from lerobot.datasets.X` → `from lerobot.common.datasets.X`

批量改:
```bash
LEROBOT_V21_PATH=$(python -c "import lerobot, os; print(os.path.dirname(lerobot.__file__))")
find $LEROBOT_V21_PATH/common/policies/residual_transformer/ -name "*.py" \
  -exec sed -i 's|from lerobot\.policies|from lerobot.common.policies|g; s|from lerobot\.datasets|from lerobot.common.datasets|g' {} \;
sed -i 's|from lerobot\.policies|from lerobot.common.policies|g; s|from lerobot\.datasets|from lerobot.common.datasets|g' \
  $LEROBOT_V21_PATH/scripts/train_residual_transformer.py
```

### Q3. `tasks.jsonl` 在 v2.1 里也要

v2.1 不要求, 但有些 lerobot 0.1.x 的边缘版本会要。如果还报这个错, 去 v3.0 那个 wrapper (修法 ① 补 tasks.jsonl) 跑一遍, 然后再切到 v2.1 跑。

### Q4. 训练时 dataset features mismatch

v2.1 创建 LeRobotDataset 的 features dict 形式可能跟 v3.0 不同。看 fork 的 train_residual_transformer.py 里读 dataset 的部分, 注释掉 strict 检查或对齐 dtype。

---

## 最简流程总结

```bash
# 1. 装 v2.1 env
conda create -n lerobot_v21 python=3.10 -y && conda activate lerobot_v21
pip install lerobot==0.1.7 num2words accelerate safetensors "transformers>=4.40,<4.52" draccus jsonlines wandb

# 2. 移植自定义代码
LEROBOT_V21=$(python -c "import lerobot, os; print(os.path.dirname(lerobot.__file__))")
cp -r ~/project/lq/a2c2-libero/src/lerobot/policies/residual_transformer $LEROBOT_V21/common/policies/
cp ~/project/lq/a2c2-libero/src/lerobot/scripts/train_residual_transformer.py $LEROBOT_V21/scripts/

# 3. 准备 v21 版本的 create_dataset 脚本
cp -r ~/project/lq/a2c2-libero/eval_libero ~/project/lq/a2c2-libero/eval_libero_v21
sed -i 's|from lerobot.datasets|from lerobot.common.datasets|g' \
  ~/project/lq/a2c2-libero/eval_libero_v21/create_dataset_for_residualpolicy.py
sed -i '13s|BASE_REPO_NAME.*|BASE_REPO_NAME = "lerobot/a2c2_dataset_v21"|' \
  ~/project/lq/a2c2-libero/eval_libero_v21/create_dataset_for_residualpolicy.py
sed -i '14s|local/|lakesenberg/|' \
  ~/project/lq/a2c2-libero/eval_libero_v21/create_dataset_for_residualpolicy.py

# 4. 跑生成 + 训练 (用对应 wrapper)
cd ~/project/lq/a2c2-libero
mkdir -p logs outputs/a2c2_head_v21
python run_residual_offline_v21.py 2>&1 | tee logs/residual_v21.log
python run_train_offline_v21.py \
  --policy.type=residual_transformer \
  --policy.repo_id=lakesenberg/a2c2_head_v21 \
  --dataset.repo_id=lakesenberg/a2c2_dataset_v21-smolvla-residual \
  --batch_size=64 --num_workers=16 --steps=200000 \
  --output_dir=outputs/a2c2_head_v21 --wandb.enable=false \
  2>&1 | tee logs/train_a2c2_head_v21.log
```

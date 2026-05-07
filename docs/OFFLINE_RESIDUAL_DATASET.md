# 离线生成残差数据集 + 训练残差头 — 操作手册

适用于 A100 服务器 (无 / 受限外网),没法连 huggingface.co、装不上 huggingface-cli 的情况。
全程 HF API 全部 stub 掉,数据集只存本地缓存,不 push。

---

## 一次性准备

### 1. 改 `eval_libero/create_dataset_for_residualpolicy.py` 第 14 行

把 `UPLOAD_REPO_NAME` 里的 owner 从 `local/` 改成<b>合法用户名</b>(LeRobot 部分版本会校验 owner 必须是合法 HF 用户名格式,offline 不会真的去 HF 找,只用作本地目录名):

```bash
cd ~/project/lq/a2c2-libero
sed -i '14s|local/|lakesenberg/|' eval_libero/create_dataset_for_residualpolicy.py
sed -n '14p' eval_libero/create_dataset_for_residualpolicy.py
# 期望输出: UPLOAD_REPO_NAME = "lakesenberg/a2c2_dataset_v21-smolvla-residual"
```

### 2. 确认 `BASE_REPO_NAME` / `BASE_POLICY_NAME` 都正确

```bash
sed -n '10,16p' eval_libero/create_dataset_for_residualpolicy.py
```

`BASE_REPO_NAME` = 你训 SmolVLA 的真机 demo 数据集名
`BASE_POLICY_NAME` = 你训出的 SmolVLA ckpt (HF repo 或本地路径)
`UPLOAD_REPO_NAME` = 输出残差数据集名 (合法格式 `<owner>/<name>`)

### 3. 确认 `robot_type` 与 base 数据集一致

第 62 行通常写着 `robot_type="panda"`,如果你 SmolVLA 训练数据集的 `robot_type` 不是 panda,要改成一致。检查 base 数据集 robot_type:

```bash
python -c "
import os; os.environ['HF_HUB_OFFLINE']='1'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
# 把下面 repo_id 换成你 BASE_REPO_NAME 实际值
ds = LeRobotDataset('Lakesenberg/your_base_demo')
print('robot_type:', ds.meta.info.get('robot_type', 'N/A'))
"
```

如果跟 panda 不一致, 改第 62 行:
```bash
sed -i '62s|robot_type="panda"|robot_type="<跟 base 一样>"|' \
  eval_libero/create_dataset_for_residualpolicy.py
```

---

## 步骤 1 · 创建 wrapper 脚本: `run_residual_offline.py`

```bash
cd ~/project/lq/a2c2-libero
cat > run_residual_offline.py << 'PYEOF'
"""Wrapper to run create_dataset_for_residualpolicy.py with all HF network
calls stubbed out, so LeRobotDataset.create() works fully offline."""
import os, sys

# offline env vars
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# monkey-patch huggingface_hub
import huggingface_hub
from huggingface_hub import HfApi
from types import SimpleNamespace


def _noop_create_repo(*args, **kwargs):
    return SimpleNamespace(url="", repo_id=kwargs.get("repo_id", ""))


def _noop_repo_info(*args, **kwargs):
    return SimpleNamespace(id="", sha="local", siblings=[])


def _noop_list_refs(*args, **kwargs):
    return SimpleNamespace(branches=[], converts=[], tags=[])


def _noop_upload(*args, **kwargs):
    return ""


HfApi.create_repo = _noop_create_repo
HfApi.repo_info = _noop_repo_info
HfApi.list_repo_refs = _noop_list_refs
HfApi.upload_file = _noop_upload
HfApi.upload_folder = _noop_upload
HfApi.upload_large_folder = _noop_upload

huggingface_hub.create_repo = _noop_create_repo
huggingface_hub.repo_info = _noop_repo_info
huggingface_hub.list_repo_refs = _noop_list_refs

print("[wrapper] HF API stubbed, running original script ...")
exec(compile(
    open("eval_libero/create_dataset_for_residualpolicy.py").read(),
    "eval_libero/create_dataset_for_residualpolicy.py",
    "exec",
))
PYEOF
```

确认创建好:
```bash
ls -la run_residual_offline.py
```

---

## 步骤 2 · 跑 wrapper 生成残差数据集

```bash
mkdir -p logs
python run_residual_offline.py 2>&1 | tee logs/residual.log
```

预估时间:
- 50 episodes × ~300 帧 = ~15k frames → 30 min
- 200 episodes × ~500 帧 = ~100k frames → 3 h

成功标志: 末尾输出 `done` / `saved episode N` / 没有 stack trace。

### 验证数据集真的生成了

```bash
ls -la ~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual/
ls -la ~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual/data/
ls -la ~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual/meta/
```

应该看到 `data/`、`meta/`、`info.json`、`videos/` 等目录。

---

## 步骤 3 · 创建训练 wrapper: `run_train_offline.py`

为 `train_residual_transformer.py` 也做同样的 stub:

```bash
cd ~/project/lq/a2c2-libero
cat > run_train_offline.py << 'PYEOF'
"""Wrapper to run train_residual_transformer.py fully offline."""
import os, sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", "offline")

import huggingface_hub
from huggingface_hub import HfApi
from types import SimpleNamespace


def _noop(*args, **kwargs):
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

huggingface_hub.create_repo = _noop
huggingface_hub.repo_info = _noop
huggingface_hub.list_repo_refs = _noop

# Pass through CLI args to the training script
sys.argv = ["train_residual_transformer.py"] + sys.argv[1:]

print("[wrapper] HF API stubbed, starting training ...")
exec(compile(
    open("src/lerobot/scripts/train_residual_transformer.py").read(),
    "src/lerobot/scripts/train_residual_transformer.py",
    "exec",
))
PYEOF
```

---

## 步骤 4 · 训练残差头

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

A100 上 ~6 h(~3 h 如果数据集小)。产物:
- `outputs/a2c2_head_v21/` 本地 ckpt
- 不会 push 到 HF (offline)

### 看训练 loss 是否下降

```bash
tail -f logs/train_a2c2_head.log | grep -E "loss|step"
```

正常的话,前 1k 步 loss 从 ~0.5 降到 ~0.1。

---

## 步骤 5 · 把残差头 ckpt scp 到 4090 推理机

A100 上跑完后,在 4090 上拉文件:

```bash
# 在 4090 上
mkdir -p ~/a2c2-libero/outputs/a2c2_head_v21
scp -r <a100_user>@192.168.3.103:~/project/lq/a2c2-libero/outputs/a2c2_head_v21/ \
       ~/a2c2-libero/outputs/

ls ~/a2c2-libero/outputs/a2c2_head_v21/
```

应该看到 `checkpoint.pt` 或 `last/` 目录。

---

## 故障排查

### A. wrapper 还是报 OfflineModeIsEnabled

加更多 stub:

```bash
python -c "
import huggingface_hub.file_download as fd
import huggingface_hub
print('HfApi methods:', [m for m in dir(huggingface_hub.HfApi) if not m.startswith('_')][:30])
"
```

把所有 `list_*` / `*_repo` / `upload_*` / `download_*` 全 stub 掉。

### B. wrapper 起来后报 `KeyError` 或 `ValidationError`

通常是 `robot_type` 与 base 数据集不一致。回到准备阶段第 3 步。

### C. 数据集里 features 的 dtype 对不上

打印对比:
```bash
python -c "
import os; os.environ['HF_HUB_OFFLINE']='1'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('Lakesenberg/your_base_demo')
print(ds.features)
"
```

跟 `eval_libero/create_dataset_for_residualpolicy.py` 第 25-55 行附近 `new_features` 比较, 确保 dtype/shape 一致。

### D. 跑到一半 GPU OOM

```bash
nvidia-smi
```

如果显存满,改 `image_writer_processes` 从 10 减到 2-4:
```bash
sed -i 's|image_writer_processes=10|image_writer_processes=4|' \
  eval_libero/create_dataset_for_residualpolicy.py
```

### E. 训练时 wandb 起不来

wrapper 已经设 `WANDB_MODE=offline`,日志在 `outputs/a2c2_head_v21/wandb/`。如果还报错,直接关 wandb:
```bash
python run_train_offline.py \
  --wandb.enable=false \
  ... 其他参数
```

---

## 关键文件清单

| 文件 | 用途 |
|---|---|
| `run_residual_offline.py` | 跑残差数据集生成 (offline + HF stub) |
| `run_train_offline.py` | 跑残差头训练 (offline + HF stub) |
| `eval_libero/create_dataset_for_residualpolicy.py` | 原脚本,被 wrapper exec |
| `src/lerobot/scripts/train_residual_transformer.py` | 原训练脚本 |
| `~/.cache/huggingface/lerobot/lakesenberg/a2c2_dataset_v21-smolvla-residual/` | 残差数据集本地路径 |
| `outputs/a2c2_head_v21/` | 残差头 ckpt 输出 |
| `logs/residual.log`, `logs/train_a2c2_head.log` | 完整日志 |

---

## 一行总览

```
A100: run_residual_offline.py → 残差数据集
A100: run_train_offline.py    → 残差头 ckpt
4090: scp ckpt → realrobot_a2c2_inference.py 真机推理
```

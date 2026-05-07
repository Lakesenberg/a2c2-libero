# 完整 Debug 历程 — 从 v2.1 数据集到 A2C2 残差头训练成功

记录从 SmolVLA 训练完到残差头训练 loss 收敛 (0.009) 全过程踩的所有坑、根因、解决方案。
适用情境: A100 服务器无外网, 数据集是 v2.1 旧格式, lerobot fork 是 v3.0 era。

毕设论文里可以从这份文档抽取 "实现细节" / "讨论" 章节的内容。

---

## 起点状态

- ✅ SmolVLA 已在真机数据上训完, ckpt 在 `/root/project/lq/lerobot/outputs/train/smolvla_a2c2_v21/checkpoints/020000/pretrained_model`
- ✅ 真机 demo 数据集 v2.1 格式, repo_id 形如 `local/a2c2_dataset_v21`
- ❌ A2C2 残差头还没训
- ❌ A100 服务器无外网, 不能直接连 huggingface.co
- ❌ 没装 huggingface-cli

目标: 跑通残差数据集生成 + 残差头训练, 让 A2C2 推理可用。

---

## 错误 1 · `lerobot==0.1.7` 找不到

### 报错
```
ERROR: Could not find a version that satisfies the requirement lerobot==0.1.7
(from versions: 0.1.0, 0.3.2, 0.3.3, 0.4.0, 0.4.1, 0.4.2, 0.4.3, 0.4.4)
```

### 根因
我之前给的版本号是 `0.1.7`, 但 PyPI 实际可用版本跳过了 0.1.x 之间的中间号, 直接从 0.1.0 跳到 0.3.x。

### 试过的方案
跑诊断脚本看哪个版本能加载数据集:
```bash
for V in 0.4.4 0.4.0 0.3.3 0.3.2; do
    pip install -q "lerobot==$V"
    python -c "import lerobot; print(lerobot.__version__)"
done
```

### 关键发现
4 次都打印 `lerobot: 0.2.0`, 不是装的版本! 说明 pip 装的不是被 import 的那个。

---

## 错误 2 · pip 装的 lerobot 没生效, 一直 import fork 的 0.2.0

### 现象
不管 `pip install lerobot==0.4.4` 还是 `0.3.2`, `import lerobot` 都返回 0.2.0。

### 根因
`~/project/lq/a2c2-libero/src/lerobot/` 是 fork 的 editable install (装到了 conda env 里),
Python 优先 import editable 路径 vs site-packages 里 pip 装的版本。

### 解决方案
**接受这个状态**: 不再尝试装 v2.1 lerobot, 留在 fork 的 0.2.0 (= v3.0 行为)。
这避免了 SmolVLA ckpt 在 v3 vs v2.1 之间不兼容的潜在问题。

**经验**: editable install (`pip install -e .`) 一旦做过, 后续 `pip install <name>` 不会覆盖。
要彻底卸: `pip uninstall <name> -y` 跑两次 (一次 editable, 一次 non-editable)。

---

## 错误 3 · `MaxRetryError: huggingface.co`

### 报错
```
requests.exceptions.ConnectionError: MaxRetryError:
HTTPSConnectionPool host='huggingface.co', port=443:
max retries exceeded with url: /api/datasets/...
```

### 根因
A100 服务器在内网, 防火墙不允许出 huggingface.co 这个域名。

### 解决方案 — HF 国内镜像
```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENDPOINT=https://hf-mirror.com
```

镜像通了, 进入下一个错误。

---

## 错误 4 · `repository not found for url: hf-mirror.com/api/datasets/local/...`

### 根因
`UPLOAD_REPO_NAME = "local/a2c2_dataset_v21-smolvla-residual"` 里 owner 是 `local`。
HF (镜像或正主) 把 `local` 当作 username, 但这个用户不存在, 返回 404。

合法 HF repo_id 格式: `<合法用户名 或 组织名>/<repo 名>`。

### 解决方案
```bash
sed -i '14s|local/|lakesenberg/|' eval_libero/create_dataset_for_residualpolicy.py
# UPLOAD_REPO_NAME = "lakesenberg/a2c2_dataset_v21-smolvla-residual"
```

注意 `lakesenberg` 账号哪怕真的不存在, 只要格式合法, offline 模式就不会去验证。

---

## 错误 5 · `OfflineModeIsEnabled: Cannot reach .../refs`

### 报错
```
huggingface_hub.errors.OfflineModeIsEnabled:
Cannot reach https://hf-mirror.com/api/datasets/lakesenberg/.../refs:
offline mode is enabled.
```

### 根因
`HF_HUB_OFFLINE=1` 拦了所有 HF API 调用。
但 `LeRobotDataset.create()` 内部要调 `HfApi.list_repo_refs` 验证仓库存在,
被 offline 拦截 → 抛 `OfflineModeIsEnabled`。

### 解决方案 — 写 wrapper 把 HF API stub 掉

**v1 wrapper** (失败):
```python
import huggingface_hub
from huggingface_hub import HfApi
HfApi.create_repo = lambda *a, **k: None
HfApi.list_repo_refs = lambda *a, **k: None
```

**v2 wrapper** (修部分):
更多 stub:
```python
HfApi.create_repo = _noop
HfApi.repo_info = _noop
HfApi.list_repo_refs = _noop
HfApi.upload_file = _noop
HfApi.create_tag = _noop
huggingface_hub.create_repo = _noop  # 顶层函数也覆盖
```

**v3 wrapper** (终于成功):
sys.modules 暴力扫:
```python
import lerobot.datasets.utils
import lerobot.datasets.lerobot_dataset

for mod_name, mod in list(sys.modules.items()):
    if mod is None: continue
    try:
        if hasattr(mod, "get_safe_version"):
            setattr(mod, "get_safe_version", lambda r,v,**k: v)
    except Exception:
        pass
```

### 关键洞察 (v2 → v3 的)
LeRobot 内部 `from .utils import get_safe_version` 把函数<b>引用</b>拷贝到了 lerobot_dataset 模块。
patch `ds_utils.get_safe_version` 不影响 lerobot_dataset 模块里那个引用。
要 patch 必须遍历所有有这个属性的 module 全部替换。

---

## 错误 6 · `RevisionNotFoundError: Your dataset must be tagged with a codebase version`

### 报错
```
huggingface_hub.errors.RevisionNotFoundError:
Your dataset must be tagged with a codebase version.
hub_api.create_tag("local/a2c2_dataset_v21", tag="_version_", repo_type="dataset")
```

### 根因
LeRobot fork 自己的 `get_safe_version()` 在 utils.py:329 主动 raise 这个错,
告诉用户去 HF 给 dataset 加版本 tag。但 offline 模式下 `create_tag` 跑不了。

### 解决方案
v3 wrapper 把 `get_safe_version` 替换成 `lambda r, v, **k: v` (永远返回请求的 version, 不查 HF)。

### 错误链
错误 5 → 6 是同一类问题 (offline 跟 HF 验证逻辑冲突), 但触发位置不同:
- 错误 5: `LeRobotDataset.create()` 调 `list_repo_refs` (HF API 这层)
- 错误 6: `LeRobotDatasetMetadata.__init__` 调 `get_safe_version` (lerobot 自己这层)

---

## 错误 7 · `tasks.jsonl: No such file or directory`

### 报错
```
FileNotFoundError: [Errno 2]
'/root/.cache/huggingface/lerobot/lerobot/a2c2_dataset_v21/meta/tasks.jsonl'
```

### 根因 (两个)
1. 路径里 owner 是 `lerobot/` 不是 `local/`。fork 的 LeRobot 内部把 BASE_REPO_NAME 解析后落到了这个路径。
2. 数据集是 v2.0 旧格式, 老格式没有 `tasks.jsonl` 文件 (任务定义在 info.json 内)。新版 lerobot 强制要求这个文件。

### 解决方案

**(a) 修 BASE_REPO_NAME**:
```bash
find ~/.cache/huggingface/ -name "tasks.jsonl" 2>/dev/null
# 看到真实路径
sed -i '13s|BASE_REPO_NAME.*|BASE_REPO_NAME = "lerobot/a2c2_dataset_v21"|' \
  eval_libero/create_dataset_for_residualpolicy.py
```

**(b) 补 tasks.jsonl**:
```bash
DATASET_ROOT="/root/.cache/huggingface/lerobot/lerobot/a2c2_dataset_v21"
python << 'PY'
import json, os
root = "$DATASET_ROOT/meta"
info = json.load(open(f"{root}/info.json"))
tasks = info.get("tasks", [])
with open(f"{root}/tasks.jsonl", "w") as f:
    if isinstance(tasks, list) and tasks:
        for i, t in enumerate(tasks):
            if isinstance(t, dict): f.write(json.dumps(t) + "\n")
            else: f.write(json.dumps({"task_index": i, "task": str(t)}) + "\n")
    else:
        f.write('{"task_index": 0, "task": "manipulation"}\n')
PY
```

---

## 错误 8 · 残差数据集生成中 `[normalize-fix] mean is inf` 警告

### 现象
处理时反复打印:
```
[normalize-fix] mean is inf; key= observation.state mean_shape= (6,)
[normalize-fix] std is inf; key= action std_shape= (6,)
```

### 根因
源 BASE 数据集的 stats 在 v2.0 → v3.0 转换时损坏:
- `info.json` 里全局 stats 是 inf
- 但实际 parquet 数据是正常数值

LeRobot fork 的 `normalize-fix` 检测到 inf 就用占位符兜底, 但残差数据集会继承这些 inf stats。

### 后果
如果不修, 训练时 normalization layer 会用 inf 算输入 → NaN → loss 爆炸。

### 跑完后修法
75 个 episode 都生成完后, 重新计算 stats:
```python
import numpy as np, pandas as pd, glob, json
files = sorted(glob.glob(".../data/**/*.parquet", recursive=True))
buf = {}
for f in files:
    df = pd.read_parquet(f)
    for c in ["observation.state", "action", "vla_actions"]:
        if c in df.columns:
            buf.setdefault(c, []).append(np.stack(df[c].values))
stats = {}
for c, lst in buf.items():
    arr = np.concatenate(lst, 0).reshape(-1, lst[0].shape[-1]).astype(np.float64)
    stats[c] = {"mean": arr.mean(0).tolist(),
                "std":  arr.std(0).clip(1e-6, None).tolist(), ...}
json.dump(stats, open(".../meta/stats.json", "w"), indent=2)
```

---

## 错误 9 · `safe_stack` 时 `vla_actions` 报 `TypeError: only length-1 arrays can be converted`

### 报错
```
TypeError: only length-1 arrays can be converted to Python scalars
ValueError: setting an array element with a sequence.
```

### 根因
`vla_actions` 列的每个 row 是<b>变长 chunk</b> (shape `(H, A)`, 但不同 row 的 H 不同 — 比如 episode 末尾被截短)。
`np.stack(df["vla_actions"].values)` 要求所有 row 同形状, 失败。

但 `observation.state` / `action` 是单帧值 (shape `(6,)`), 形状一致没问题, 所以那两个先打印了正常 stats。

### 解决方案
```python
def safe_stack(values):
    try:
        return np.stack(values)
    except (ValueError, TypeError):
        # 变长 chunk: flatten 每行后 concat
        parts = []
        for v in values:
            v = np.asarray(v)
            if v.ndim == 0: continue
            if v.ndim == 1: parts.append(v.reshape(1, -1))
            else: parts.append(v.reshape(-1, v.shape[-1]))
        return np.concatenate(parts, axis=0) if parts else None
```

外加过滤 inf/nan 行:
```python
mask = np.isfinite(arr).all(axis=1)
if (~mask).any():
    arr = arr[mask]
```

---

## 训练启动 + 收敛过程

stats 修好后:
```bash
nohup python run_train_offline.py \
  --policy.type=residual_transformer \
  --policy.repo_id=lakesenberg/a2c2_head_v21 \
  --dataset.repo_id=lakesenberg/a2c2_dataset_v21-smolvla-residual \
  --batch_size=64 --num_workers=16 --steps=200000 \
  --output_dir=outputs/a2c2_head_v21 --wandb.enable=false \
  > logs/train_a2c2_head.log 2>&1 &
```

### Loss 曲线
| step | loss | grad_norm | 说明 |
|---|---|---|---|
| 1k | 0.5-1.0 | ~5 | 初始化 |
| 4k | 0.048 | 1.88 | 快速下降 |
| 10k | 0.025 | 1.0 | 主要学习阶段 |
| 20k | 0.018 | 0.8 | 第 1 个 ckpt 保存 |
| 40k | 0.012 | 0.5 | 第 2 个 ckpt |
| 60k | 0.011 | 0.4 | 第 3 个 ckpt, 已收敛 |
| 77k | 0.009 | 0.35 | 边际改进 |

### 决定停止
77K 步用 32 分钟跑到 loss 0.009, 60K → 77K 只降 0.002, 收益递减。
直接停训练, 用 60K ckpt 做推理。

```bash
PID=$(pgrep -f run_train_offline)
kill $PID
```

---

## 总结: 完整 fix 链

```
v2.1 数据集 + v3 lerobot fork
    │
    ├─ ① pip install lerobot 不生效 ─→ 接受 fork, 不切版本
    ├─ ② HF 不通 ─→ HF_ENDPOINT=hf-mirror.com
    ├─ ③ owner=local 不合法 ─→ 改成 lakesenberg
    ├─ ④ HfApi 调用被 offline 拦 ─→ wrapper monkey-patch HF API
    ├─ ⑤ get_safe_version 报错 ─→ wrapper sys.modules 暴力扫
    ├─ ⑥ tasks.jsonl 缺失 ─→ 从 info.json 提取 + BASE 路径修正
    ├─ ⑦ stats=inf 警告 ─→ 数据集生成完后重算 stats
    ├─ ⑧ vla_actions 变长 ─→ safe_stack 兼容
    └─ 训练启动 ─→ loss 0.5 → 0.009, 60K 收敛
```

**总耗时**: 约 1 天 (含数据集生成 ~1h + 训练 32 min + debug 反复约 6h)。

---

## 经验教训 (论文 Discussion 章节素材)

### 1. Editable install vs pip install 的优先级
`pip install -e .` 一旦做过, 同 env 后续 `pip install <package>` 看似成功但不会覆盖。
检查是否真生效: `python -c "import package; print(package.__file__)"` 看路径是否在 site-packages 还是 source。

### 2. Monkey-patch 跨模块引用的陷阱
Python `from module import X` 会拷贝引用到当前模块。后续 patch `module.X = new` 不影响其他模块里那个引用。
唯一可靠的 patch 是遍历 sys.modules 找所有有这个属性的 module 都替换。

### 3. 离线模式跟"严格性检查"打架
`HF_HUB_OFFLINE=1` + 库的"必须验证 repo 存在/版本"逻辑必然冲突。
解法只能 stub 那些验证函数。LeRobot 的 `get_safe_version` 是典型例子。

### 4. 数据集格式版本兼容性
LeRobot 的 v2.0 → v3.0 转换不彻底 — `info.json` 标了 v3.0 但 `tasks.jsonl` / `episodes_stats.jsonl` 没生成,
stats 没重算。需要手动补全。

### 5. Stats 不能盲信
源数据集 stats 有 inf, 但 parquet 实际数值 OK。不能用源 stats, 必须从 parquet 重算。

### 6. 训练 loss 收敛点比目标 step 数重要
设 200k 步, 实际 60k 已经收敛。继续训练 120k 步只换来边际改进 + 更长等待。
本科毕设视角: 把节省的时间花在<b>真机推理 + 论文写作</b>。

---

## 给后来者的最佳实践 (按时间顺序)

```bash
# 1. 进入 fork 项目
cd ~/project/lq/a2c2-libero
conda activate <训 SmolVLA 用的 env>     # 不要新建 env!

# 2. 配置环境变量 (写到 ~/.bashrc 一劳永逸)
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1

# 3. 修 BASE 数据集路径 + 补 tasks.jsonl + episodes_stats.jsonl
DATASET_ROOT=$(find ~/.cache/huggingface/lerobot -name "tasks.jsonl" -path "*<your_dataset>*" -exec dirname {} \; | head -1)
[ ! -f "$DATASET_ROOT/tasks.jsonl" ] && echo '{"task_index":0,"task":"manipulation"}' > "$DATASET_ROOT/tasks.jsonl"
[ ! -f "$DATASET_ROOT/../meta/episodes_stats.jsonl" ] && touch "$DATASET_ROOT/../meta/episodes_stats.jsonl"

# 4. 改 create_dataset 脚本顶部 3 个变量
sed -i '13s|BASE_REPO_NAME.*|BASE_REPO_NAME = "<correct_owner>/<your_dataset>"|' \
  eval_libero/create_dataset_for_residualpolicy.py
sed -i '14s|local/|<your_username>/|' \
  eval_libero/create_dataset_for_residualpolicy.py

# 5. 写终极 wrapper (v3, 含 sys.modules 扫)
# (见错误 5/6 节)

# 6. 跑生成
python run_residual_offline.py 2>&1 | tee logs/residual.log

# 7. 修 stats (含 safe_stack)
# (见错误 8/9 节)

# 8. 启训练 (后台, 监控 loss)
nohup python run_train_offline.py ... &
tail -f logs/train.log | grep -E "step|loss"

# 9. loss 收敛 (~0.01) 就停, 不必跑满 200K
kill $(pgrep -f run_train_offline)
```

按这个顺序大概 2 h debug + 1 h 数据集 + 30 min 训练 = 3.5 h 搞定。比我们走的弯路快 3 倍。

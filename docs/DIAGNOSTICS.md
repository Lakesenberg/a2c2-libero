# 推理端诊断手册

如果 `python scripts/realrobot_a2c2_inference.py` 启动报错, 按下面顺序跑诊断 + 修复。
所有命令在 4090 (或任何推理机) 的项目根目录 `~/a2c2-libero/` 里执行。

---

## §0 · 快速健康检查 (一条命令)

```bash
cd ~/a2c2-libero

python - << 'PY'
import importlib, sys
print("=== Python ===")
print(sys.version)
print()
checks = [
    ("a2c2_libero",                     None),
    ("a2c2_libero.heads",                "A2C2MLPHead"),
    ("a2c2_libero.inference.a2c2_engine","A2C2Engine"),
    ("a2c2_libero.inference.utils",      "build_state"),
    ("torch",                            None),
    ("lerobot",                          "__version__"),
    ("lerobot.robots",                   None),
    ("lerobot.robots.so100_follower",    None),
    ("lerobot.policies.smolvla.modeling_smolvla", "SmolVLAPolicy"),
]
for mod_name, attr in checks:
    try:
        m = importlib.import_module(mod_name)
        if attr:
            v = getattr(m, attr, None)
            print(f"  ✓ {mod_name}.{attr} = {v if attr=='__version__' else 'OK'}")
        else:
            print(f"  ✓ {mod_name}")
    except Exception as e:
        print(f"  ✗ {mod_name}: {type(e).__name__}: {e}")
PY
```

期望全是 ✓。每个 ✗ 对应下面一节修法。

---

## §1 · `ModuleNotFoundError: No module named 'a2c2_libero'`

包没装。

```bash
cd ~/a2c2-libero
pip install -e .
# 或 uv:
uv pip install -e .

# 验证
python -c "from a2c2_libero.heads import A2C2MLPHead; print('OK')"
```

如果不想装, 临时:
```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```
写到 `~/.bashrc` 一劳永逸。

---

## §2 · `ModuleNotFoundError: No module named 'lerobot.common'` 或 `lerobot.robots.configs`

lerobot 版本布局不同。先看你实际有什么:

```bash
python - << 'PY'
import lerobot, pkgutil
print('lerobot version:', getattr(lerobot, '__version__', '?'))
print('lerobot path:', lerobot.__file__)
print()
print('--- top-level submodules ---')
for x in pkgutil.iter_modules(lerobot.__path__):
    print('  lerobot.' + x.name)
print()
try:
    import lerobot.robots as r
    print('--- lerobot.robots exports ---')
    print([n for n in dir(r) if not n.startswith('_')])
    print()
    print('--- lerobot.robots submodules ---')
    for x in pkgutil.iter_modules(r.__path__):
        print('  lerobot.robots.' + x.name)
except Exception as e:
    print('lerobot.robots fail:', e)
PY
```

拿这个输出参考下面修法。

### 2a · 新版 lerobot (无 `lerobot.common`)

`scripts/realrobot_a2c2_inference.py` 现在自带 fallback (commit `8305c2d` 起),
当 `make_robot_from_config` 找不到时, 直接 import `lerobot.robots.<robot_type>` 子包。

确保你拉了最新版:
```bash
git fetch origin inference_test_async
git checkout origin/inference_test_async -- scripts/realrobot_a2c2_inference.py
git log -1 --oneline scripts/realrobot_a2c2_inference.py
# 应该看到 8305c2d 或更新的 commit
```

跑推理时第一行会看到:
```
[build_robot] registry import failed (...); falling back to direct per-robot import.
```

这是<b>正常的</b>, 表示走了直接 import 路径。

### 2b · 你的 robot type 跟 lerobot 子包名对不上

诊断输出里 `lerobot.robots.<...>` 列出实际可用的, 例如:

```
lerobot.robots.so100_follower
lerobot.robots.so101_follower
lerobot.robots.bi_so_follower
lerobot.robots.koch_follower
```

`--robot-type` 必须<b>完全等于</b>子包名 (snake_case):
```bash
--robot-type=so100_follower    # ✓
--robot-type=SO100Follower     # ✗ 大小写错
--robot-type=so100             # ✗ 缺 _follower
```

### 2c · 旧版 lerobot (用 `lerobot.common.robots`)

脚本也会自动 fallback。如果还报 `lerobot.common` 找不到, 升级或降级看实际数据集兼容性。本仓库主要测试在新版布局上。

---

## §3 · `LatentHook: could not locate SmolVLA backbone`

SmolVLA 内部模块路径因版本而异。打印实际结构:

```bash
python - << 'PY'
import torch
import importlib
for mod in ("lerobot.policies.smolvla.modeling_smolvla",
            "lerobot.common.policies.smolvla.modeling_smolvla"):
    try:
        m = importlib.import_module(mod)
        SmolVLAPolicy = m.SmolVLAPolicy
        print(f"loaded from {mod}")
        break
    except Exception:
        continue

# 改成你 SmolVLA ckpt 路径
p = SmolVLAPolicy.from_pretrained("outputs/smolvla_v21")

print()
print("=== modules with 'backbone' / 'encoder' in name ===")
for name, mod in p.named_modules():
    if 'backbone' in name.lower() or 'encoder' in name.lower():
        print(f"  {name}  ({type(mod).__name__})")
PY
```

把输出贴出来, 调整 `realrobot_a2c2_inference.py` 里 hook 的 module path。

---

## §4 · `model.safetensors: file not found`

`--head-ckpt` 路径不对。期望:

```
outputs/a2c2_head_v21/
├── model.safetensors        ← --head-ckpt 直接指这个
└── config.json
```

确认:
```bash
ls outputs/a2c2_head_v21/
# 应该看到 model.safetensors 和 config.json
```

如果在嵌套目录里:
```bash
find outputs/ -name "model.safetensors"
```

把找到的<b>文件</b>路径(不是目录)给 `--head-ckpt`。

---

## §5 · `[robot] is_calibrated == False`

机器人没标定过。一次性:

```bash
lerobot-calibrate --robot.type=so100_follower --robot.id=<your_id>
```

之后标定信息存在 `~/.cache/huggingface/lerobot/calibration/robots/<robot_type>/<id>.json`,
不用重做。

查现有标定:
```bash
ls ~/.cache/huggingface/lerobot/calibration/robots/
# 应该看到 robot type 子目录, 里面是 <id>.json
```

---

## §6 · 推理跑起来但 latency p99 > 15 ms

```bash
nvidia-smi    # 确认 GPU 在用
```

可能原因:
- SmolVLA 没异步: 检查脚本里 `engine.start_async()` 调了
- 第一帧冷启动 (前 ~H tick 在等 SmolVLA 第一次 forward): 正常, 跳过最初 0.5s
- 用了 CPU: 命令加 `--device cuda` (如果脚本支持) 或 `CUDA_VISIBLE_DEVICES=0`

精确测:
```bash
python scripts/a100_inference_test.py \
    --mode synthetic \
    --policy-path outputs/smolvla_v21 \
    --head-ckpt   outputs/a2c2_head_v21/model.safetensors \
    --ticks 500 --tick-dt-ms 5 \
    --output logs/4090_synthetic.json

cat logs/4090_synthetic.json | python -m json.tool | head -40
```

---

## §7 · 推理输出全 0 / 抖动

可能 ckpt 加载有问题或 build_state 没传 `base_action_chunk`:

```bash
# 1. 看 commit, 必须 ≥ 30e9624 (修过 base_action_chunk 那个)
git log -1 --oneline scripts/realrobot_a2c2_inference.py
git log -1 --oneline src/a2c2_libero/inference/a2c2_engine.py
git log -1 --oneline src/a2c2_libero/inference/utils.py

# 2. 看 ckpt 实际加载没出错
python - << 'PY'
import torch
sd = torch.load("outputs/a2c2_head_v21/model.safetensors", map_location="cpu") \
     if "model.safetensors".endswith(".pt") else None
if sd is None:
    from safetensors.torch import load_file
    sd = load_file("outputs/a2c2_head_v21/model.safetensors")
print(f"loaded {len(sd)} tensors")
print("first 5 keys:", list(sd.keys())[:5])
PY
```

如果 commit 老了 → `git pull origin inference_test_async`。

---

## §8 · 4090 总是同步最新代码

每次准备跑前一条命令同步:

```bash
cd ~/a2c2-libero
git fetch origin inference_test_async
git checkout origin/inference_test_async -- \
    scripts/realrobot_a2c2_inference.py \
    src/a2c2_libero/inference/utils.py \
    src/a2c2_libero/inference/a2c2_engine.py \
    docs/DIAGNOSTICS.md \
    docs/STOP_TRAIN_AND_INFER.md \
    README.md
```

或暴力一点全分支同步 (会丢本地修改):
```bash
git fetch origin
git reset --hard origin/inference_test_async
```

---

## §9 · 完整诊断报告 (一段贴出全部环境信息)

跑这一段, 把输出贴回去问问题:

```bash
cd ~/a2c2-libero

echo "=== git ==="
git rev-parse HEAD
git status -s | head -10
echo
echo "=== python ==="
python --version
which python
echo
echo "=== installed packages (key ones) ==="
pip list 2>/dev/null | grep -iE "lerobot|torch|safetensors|huggingface|transformers" | head -20
echo
echo "=== lerobot layout ==="
python -c "
import lerobot, pkgutil
print('version:', getattr(lerobot, '__version__', '?'))
print('path:', lerobot.__file__)
for x in pkgutil.iter_modules(lerobot.__path__):
    print('  lerobot.' + x.name)
try:
    import lerobot.robots as r
    for x in pkgutil.iter_modules(r.__path__):
        print('  lerobot.robots.' + x.name)
except Exception as e:
    print('  robots fail:', e)
"
echo
echo "=== ckpts ==="
find outputs/ -maxdepth 4 -name "model.safetensors" 2>/dev/null
find outputs/ -maxdepth 4 -name "config.json" 2>/dev/null
echo
echo "=== robot calibration ==="
ls ~/.cache/huggingface/lerobot/calibration/robots/ 2>/dev/null
echo
echo "=== gpu ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null
```

整段输出贴出来, 任何位置出错都能立刻定位。

---

## §10 · 已知 commit 检查表

| 功能 | 最低 commit | 说明 |
|---|---|---|
| 异步 SmolVLA + A2C2 引擎 | `0521fc4` | 基础架构 |
| 双机部署脚本 | `21ea85b` | server_train.sh / inference_4090.sh |
| 真机推理脚本 | `21b62e9` | realrobot_a2c2_inference.py |
| 残差头接收完整 chunk | `30e9624` | 必需, 没这个真机会漂 |
| README + 双机文档 | `d5b4eda` | 部署文档 |
| 4090-only 流程 | `8482e53` | 离线推理 |
| Robot import fallback | `3cd8324` | 兼容多种 lerobot 布局 |
| 直接 per-robot import fallback | `8305c2d` | 新版 lerobot 无 registry 时 |

跑命令前确认:
```bash
git rev-parse HEAD
git log --oneline -10
```

`8305c2d` 之后的 commit 都 OK。如果在它之前, `git pull origin inference_test_async`。

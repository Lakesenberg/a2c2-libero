# 停止训练 + 真机 A2C2 推理

当前: 训练在 step 77K (loss 0.009, 已收敛), 选择 A 立即停止用现有 ckpt。
下面是停训练 → scp → 4090 真机推理的完整流程。

---

## §1 · A100 上停训练 + 找最新 ckpt

```bash
cd ~/project/lq/a2c2-libero

# 1.1 停训练进程
PID=$(pgrep -f run_train_offline)
echo "killing PID=$PID"
kill $PID
sleep 3
ps aux | grep run_train | grep -v grep      # 期望: 空
```

```bash
# 1.2 看保存了哪些 ckpt
ls outputs/a2c2_head_v21/checkpoints/

# 期望:
#   020000  040000  060000  last
```

```bash
# 1.3 找最新 ckpt 完整路径
LAST=$(ls -t outputs/a2c2_head_v21/checkpoints/ | grep -E '^[0-9]+$' | head -1)
HEAD_PATH="$(realpath outputs/a2c2_head_v21/checkpoints/$LAST/pretrained_model)"
echo "head ckpt: $HEAD_PATH"
ls "$HEAD_PATH"

# 期望:
#   model.safetensors
#   config.json
```

```bash
# 1.4 SmolVLA ckpt 路径 (你之前训的)
SMOLVLA_PATH=/root/project/lq/lerobot/outputs/train/smolvla_a2c2_v21/checkpoints/020000/pretrained_model
ls "$SMOLVLA_PATH"
```

---

## §2 · 4090 上拉两个 ckpt 过去

**注意**: 这一段在 <b>4090 推理机</b> 上跑, 不是 A100。

```bash
# 4090 上
cd ~/a2c2-libero
mkdir -p outputs

# 拉残差头 (替换 060000 为你 §1.3 找到的最新数字)
scp -r root@192.168.3.103:/root/project/lq/a2c2-libero/outputs/a2c2_head_v21/checkpoints/060000/pretrained_model \
       outputs/a2c2_head_v21

# 拉 SmolVLA
scp -r root@192.168.3.103:/root/project/lq/lerobot/outputs/train/smolvla_a2c2_v21/checkpoints/020000/pretrained_model \
       outputs/smolvla_v21

# 确认两个都到位
ls outputs/a2c2_head_v21/   # 应有 model.safetensors + config.json
ls outputs/smolvla_v21/     # 应有 SmolVLA 权重 + config
```

---

## §3 · 4090 上准备机器人

### 3.1 找 robot.id

```bash
ls ~/.cache/huggingface/lerobot/calibration/robots/
```

应该看到机器人类型目录, 例如:
```
so100_follower/
```

进去看 id:
```bash
ls ~/.cache/huggingface/lerobot/calibration/robots/so100_follower/
# my_robot.json   ← <robot id> = my_robot
```

### 3.2 测试相机能用

```bash
lerobot-find-cameras
```

会列出可用相机 + index。记下你要用的相机 (主视角 + 腕部相机)。

### 3.3 测试 robot.get_observation 不卡

```bash
python -c "
from lerobot.robots import make_robot_from_config
from lerobot.robots.configs import RobotConfig
import time
r = make_robot_from_config(RobotConfig.from_kwargs(type='so100_follower', id='<your_id>'))
r.connect()
for _ in range(5):
    t = time.perf_counter()
    obs = r.get_observation()
    print(f'{(time.perf_counter()-t)*1000:.2f}ms keys={list(obs.keys())[:3]}')
r.disconnect()
"
```

期望每行 < 5 ms。

---

## §4 · 4090 跑真机 A2C2 推理 (单臂 6-DoF)

```bash
cd ~/a2c2-libero

# 第一次跑, 强烈建议加 --no-record 先试一遍
python scripts/realrobot_a2c2_inference.py \
  --smolvla-path outputs/smolvla_v21 \
  --head-ckpt    outputs/a2c2_head_v21/model.safetensors \
  --robot-type   so100_follower \
  --robot-id     <your_robot_id> \
  --action-dim   6 \
  --chunk-size   50 \
  --episodes     3 \
  --task         "pick up the cup" \
  --home-on-start \
  --no-record
```

参数解释:
- `--action-dim 6` — SO-100 单臂 6-DoF (你训 SmolVLA 用的)
- `--chunk-size 50` — A2C2 head config 里的 H, 跟训练时一致
- `--no-record` — 第一次不录数据, 只测推理通不通
- `--home-on-start` — episode 开始前自动 home

### 4.1 跑成功之后, 加录制

```bash
python scripts/realrobot_a2c2_inference.py \
  --smolvla-path outputs/smolvla_v21 \
  --head-ckpt    outputs/a2c2_head_v21/model.safetensors \
  --robot-type   so100_follower \
  --robot-id     <your_robot_id> \
  --action-dim   6 \
  --chunk-size   50 \
  --episodes     10 \
  --task         "pick up the cup" \
  --dataset-repo lakesenberg/realrobot_a2c2_eval \
  --home-on-start
```

录制后会在 `~/.cache/huggingface/lerobot/lakesenberg/realrobot_a2c2_eval/` 生成 LeRobotDataset, 可用作后续 DAgger 训练数据。

---

## §5 · 故障排查

### Q1. realrobot_a2c2_inference.py 报 LatentHook 找不到 backbone

打印 SmolVLA 实际结构:
```bash
python -c "
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
p = SmolVLAPolicy.from_pretrained('outputs/smolvla_v21')
for name, _ in p.named_modules():
    if 'backbone' in name.lower() or 'encoder' in name.lower():
        print(name)
" | head -20
```

把输出贴出来, 调整 `scripts/realrobot_a2c2_inference.py` 里 hook 路径。

### Q2. ckpt 加载报 missing key 或 unexpected key

A100 (训) 和 4090 (推) 的 lerobot fork commit 不一致。两边查:
```bash
cd ~/a2c2-libero    # 或对应路径
git rev-parse HEAD
```

不一样的话 4090 同步到 A100 commit:
```bash
git fetch origin
git checkout <a100_commit_hash>
```

### Q3. 真机第一帧动作过大撞东西

第一次跑前<b>必备</b>:
- 周围空旷, 远离障碍
- 急停按钮在手
- `--episodes 1` 先只跑一次, 看动作是否合理

如果动作明显异常, 把 `--head-ckpt` 换成更早的 ckpt (例如 020000), 损失更高但更保守。

### Q4. 推理 latency > 10 ms

```bash
# 加 --benchmark 看每 tick 耗时
python scripts/realrobot_a2c2_inference.py ... --benchmark
```

如果 SmolVLA forward > 100 ms, 4090 GPU 占用看一下:
```bash
nvidia-smi
```

可能 SmolVLA 没用 GPU, 加 `--device cuda` 强制。

### Q5. 推理过程中机器人姿态突然剧烈跳变

可能是 chunk 边界问题。脚本有 SWCF (sliding window cosine fusion) 默认开。如果还跳, 改 `--chunk-size 25` 减半。

---

## §6 · 时间预算

| 段 | 时间 | 备注 |
|---|---|---|
| §1 停训练 + 找 ckpt | 1 min | A100 |
| §2 scp 到 4090 | 5-10 min | 看网速 (LAN ~10 GB/min) |
| §3 准备机器人 | 5 min | 标定/相机/测 obs |
| §4 真机 3 ep no-record | 15 min | 第一次试 |
| §4.1 真机 10 ep + record | 30 min | 收 eval 数据 |
| **合计** | **~1 h** | |

---

## §7 · 论文用的实验数据收集 (可选)

跑完 A2C2 推理后, 你毕设论文需要对比数据:

### 7.1 跑 baseline (SmolVLA only, 不带残差头)

```bash
# 创建一个 dummy head ckpt 让推理脚本认为 head_ckpt 是 0
python -c "
import torch, os
os.makedirs('outputs/dummy_head', exist_ok=True)
state = torch.zeros(1)
torch.save({'dummy': state}, 'outputs/dummy_head/model.safetensors')
"

# 跑 baseline (Δa 全 0, 等于纯 SmolVLA)
# 或者直接用 lerobot-rollout --inference.type=sync
```

更干净的方法: 不要调 A2C2 head, 直接跑 SmolVLA:

```bash
lerobot-rollout \
  --strategy.type=base \
  --strategy.num_episodes=10 \
  --inference.type=sync \
  --policy.path=outputs/smolvla_v21 \
  --robot.type=so100_follower \
  --robot.id=<your_id> \
  --dataset.repo_id=lakesenberg/realrobot_smolvla_only_eval
```

### 7.2 主结果表 (论文里 Table)

| Method | 任务1 | 任务2 | 任务3 | 平均 |
|---|---|---|---|---|
| SmolVLA only | _% | _% | _% | _% |
| **+ A2C2 (ours)** | **_%** | **_%** | **_%** | **_%** |
| Δ | +_pp | +_pp | +_pp | +_pp |

A2C2 应该比 SmolVLA only 高 3-7pp。

---

## §8 · 一行总览 (从 A100 停训练到 4090 真机推理)

A100 (一行):
```bash
cd ~/project/lq/a2c2-libero && PID=$(pgrep -f run_train_offline) && kill $PID && sleep 3 && LAST=$(ls -t outputs/a2c2_head_v21/checkpoints/ | grep -E '^[0-9]+$' | head -1) && echo "head: $(realpath outputs/a2c2_head_v21/checkpoints/$LAST/pretrained_model)"
```

4090 (3 步):
```bash
# 1. 拉 ckpt
scp -r root@192.168.3.103:/root/project/lq/a2c2-libero/outputs/a2c2_head_v21/checkpoints/060000/pretrained_model ~/a2c2-libero/outputs/a2c2_head_v21
scp -r root@192.168.3.103:/root/project/lq/lerobot/outputs/train/smolvla_a2c2_v21/checkpoints/020000/pretrained_model ~/a2c2-libero/outputs/smolvla_v21

# 2. 检查机器人
lerobot-find-cameras
ls ~/.cache/huggingface/lerobot/calibration/robots/

# 3. 跑推理 (no-record 试一遍)
cd ~/a2c2-libero && python scripts/realrobot_a2c2_inference.py \
  --smolvla-path outputs/smolvla_v21 \
  --head-ckpt    outputs/a2c2_head_v21/model.safetensors \
  --robot-type   so100_follower --robot-id <your_id> \
  --action-dim 6 --chunk-size 50 --episodes 3 \
  --task "pick up the cup" --home-on-start --no-record
```

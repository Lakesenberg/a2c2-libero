#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import numpy as np
import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.scripts.eval import eval_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )
    BASE_MODEL_PATH = "k1000dai/smolvla_libero_scratch"
    base_policy = SmolVLAPolicy.from_pretrained(BASE_MODEL_PATH)
    base_policy.to(device)
    base_policy.eval()
    
    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )
    
    def convert_raw_batch_to_residualact(batch):
        """
        Convert the raw batch to the format expected by the residualact policy.
        This includes:
        - Using the first frame of the state horizon for smolVLA.
        - Interpolating the action between time t and t+1.
        - Adding time features.
        
        Input Batch should contain:
        - observation.images.image: (B, T, C, H, W)
        - observation.images.wrist_image: (B, T, C, H, W)
        - observation.state: (B, T, S)
        - action: (B, T+1, A) , +1 for interplate last action
        - action_is_pad: (B, T + 1) boolean tensor indicating padded actions
        - task: (B,) task identifiers
        - language_embedding: (B, D) language embeddings
        
        Output Batch will contain:
        - observation.images.image: (B, C, H, W) - selected frame
        - observation.images.wrist_image: (B, C, H, W) - selected frame
        - observation.state: (B, S) - selected frame
        - action: (B, chunk_size + 1, A) - ( predicted action at time t, interpolated actions from t to t+1 )
        - action_is_pad: (B, chunk_size) - all False (non-padded)
        - task: (B,) - task identifiers
        - time_feature: (B, 2) - time features for the selected frame
        - language_embedding: (B, D) - language embeddings
        """
        batch_size = batch["observation.images.image"].shape[0]
        time_start = time.perf_counter()
        # use first frame of state horizon for smolVLA
        smol_vla_batch = {
            "observation.images.image": torch.stack([batch["observation.images.image"][i, 0] for i in range(batch_size)]).to(device),
            "observation.images.wrist_image": torch.stack([batch["observation.images.wrist_image"][i, 0] for i in range(batch_size)]).to(device),
            "observation.state": torch.stack([batch["observation.state"][i, 0] for i in range(batch_size)]).to(device),
            "task": batch["task"],
        }
        print(f"Convert raw batch to residualact took {time.perf_counter() - time_start:.3f} seconds")
        
        predicted_action_chunk = base_policy.predict_action_chunk(smol_vla_batch)
        
        print(f"Predict action chunk took {time.perf_counter() - time_start:.3f} seconds")
        
        #get random time index from non-padded actions to ensure time_index+1 is also valid
        # Find the last non-padded index for each batch sample
        last_valid_indices = []
        for i in range(batch_size):
            # Find the last False (non-padded) position
            non_pad_mask = ~batch["action_is_pad"][i]  # True for non-padded
            if non_pad_mask.sum() > 1:  # Need at least 2 non-padded actions for interpolation
                last_valid_idx = non_pad_mask.nonzero(as_tuple=True)[0][-1].item()
                last_valid_indices.append(max(0, last_valid_idx - 1))  # -1 to ensure time_index+1 is valid
            else:
                last_valid_indices.append(0)  # Fallback to 0 if not enough data
        
        time_index = torch.stack([
            torch.randint(0, max(1, last_valid_indices[i] + 1), (1,), device=device)[0]
            for i in range(batch_size)
        ])
        
        
        time_feature = torch.stack([
            torch.cos(2 *  np.pi * time_index / base_policy.config.chunk_size),
            torch.sin(2 *  np.pi * time_index / base_policy.config.chunk_size)
        ], dim=1).to(device)
        
        residual_chunk_size = policy.config.chunk_size
        action_t = torch.stack([batch["action"][i, time_index[i]] for i in range(batch_size)]).to(device)
        action_t_plus_1 = torch.stack([batch["action"][i, time_index[i] + 1] for i in range(batch_size)]).to(device)

        #action_t から action_t_plus_1までを chunk_size 個のアクションに線形補間
        ratio = torch.linspace(0, 1, residual_chunk_size, device=device).unsqueeze(0).repeat(batch_size, 1)
        
        # action_tとaction_t_plus_1を(batch_size, 1, action_dim)に拡張
        action_t_expanded = action_t.unsqueeze(1)  # (batch_size, 1, action_dim)
        action_t_plus_1_expanded = action_t_plus_1.unsqueeze(1)  # (batch_size, 1, action_dim)
        
        # ratioを(batch_size, residual_chunk_size, 1)に拡張
        ratio_expanded = ratio.unsqueeze(-1)  # (batch_size, residual_chunk_size, 1)
        
        # 線形補間: action_t * (1-ratio) + action_t_plus_1 * ratio
        action_interpolated = action_t_expanded * (1 - ratio_expanded) + action_t_plus_1_expanded * ratio_expanded
        
        # predicted_action_chunk から time_index のアクションを取得
        predicted_action_time_t = torch.stack([predicted_action_chunk[i, time_index[i]].to(device) for i in range(batch_size)]) # (Batch, action_dim)
        predicted_action_time_t = predicted_action_time_t.unsqueeze(1)  # (batch_size, 1, action_dim)
        
        predicted_action_plus_target_action = torch.cat(
            [
                predicted_action_time_t,  # predicted action chunk
                action_interpolated
            ],
            dim=1,
        ).to(device)
        print(f"Action interpolation took {time.perf_counter() - time_start:.3f} seconds")
        
        
        converted_batch = {
            "observation.images.image": torch.stack([batch["observation.images.image"][i, time_index[i]] for i in range(batch_size)]).to(device),
            "observation.images.wrist_image": torch.stack([batch["observation.images.wrist_image"][i, time_index[i]] for i in range(batch_size)]).to(device),
            "observation.state": torch.stack([batch["observation.state"][i, time_index[i]] for i in range(batch_size)]).to(device),
            "action": predicted_action_plus_target_action,
            "action_is_pad": torch.zeros((batch_size, residual_chunk_size), dtype=torch.bool, device=device),  # All False (non-padded)
            "task": batch["task"],
            "time_feature": time_feature,
            "language_embedding": base_policy.model.language_embeddings.to(device),
        }
        print(f"Batch conversion took {time.perf_counter() - time_start:.3f} seconds")
        return converted_batch

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")
        batch = convert_raw_batch_to_residualact(batch)
        
        ### TODO : add smolVLA inference and batch conversion here
        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if eval_env:
        eval_env.close()
    logging.info("End of training")

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)


if __name__ == "__main__":
    init_logging()
    train()

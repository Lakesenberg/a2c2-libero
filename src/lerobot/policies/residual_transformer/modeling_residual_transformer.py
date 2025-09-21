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
import hashlib
import math
from typing import Dict

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.constants import ACTION
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.residual_transformer.configuration_residual_transformer import (
    ResidualTransformerConfig,
)


def sinusoidal_position_embedding_2d(
    height: int,
    width: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return a (H·W, dim) 2D sinusoidal positional encoding."""
    if dim % 2 != 0:
        raise ValueError(f"Embedding dimension {dim} must be even for 2D encoding.")

    def _positional_encoding(length: int, channels: int) -> Tensor:
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, channels, 2, device=device, dtype=dtype) * (-math.log(10000.0) / channels)
        )
        pe = torch.zeros(length, channels, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    half_dim = dim // 2
    height_encoding = _positional_encoding(height, half_dim)
    width_encoding = _positional_encoding(width, half_dim)

    pos = torch.zeros(height, width, dim, device=device, dtype=dtype)
    pos[:, :, :half_dim] = height_encoding[:, None, :]
    pos[:, :, half_dim:] = width_encoding[None, :, :]
    return pos.view(height * width, dim)


class ResidualTransformerPolicy(PreTrainedPolicy):
    """Refinement policy that predicts the final action given base and context signals."""

    config_class = ResidualTransformerConfig
    name = "residual_transformer"

    def __init__(
        self,
        config: ResidualTransformerConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ) -> None:
        super().__init__(config)
        config.validate_features()

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        self.model = ResidualTransformer(config)

    def get_optim_params(self) -> dict:
        return [{"params": [p for p in self.parameters() if p.requires_grad]}]

    def reset(self) -> None:  # noqa: D401
        """Stateless policy; nothing to reset."""
        return None

    def _prepare_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        batch = self._normalize_action_chunk(batch)
        return batch

    def _normalize_action_chunk(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        chunk = batch.get("base_action_chunk")
        if chunk is None:
            return batch

        buffer_name = "buffer_" + ACTION.replace(".", "_")
        action_buffer = getattr(self.normalize_targets, buffer_name, None)
        if action_buffer is None:
            return batch

        mean = action_buffer.get("mean")
        std = action_buffer.get("std")
        if mean is None or std is None:
            return batch

        # Broadcast normalization statistics over (batch, chunk_length, action_dim)
        target_dtype = chunk.dtype
        target_device = chunk.device
        mean = mean.to(device=target_device, dtype=target_dtype)
        std = std.to(device=target_device, dtype=target_dtype)

        view_shape = (1,) * (chunk.ndim - mean.ndim) + mean.shape
        batch["base_action_chunk"] = (chunk - mean.view(view_shape)) / (std.view(view_shape) + 1e-8)
        return batch

    def forward(self, batch: Dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch = dict(batch)
        batch = self._prepare_batch(batch)

        actions = batch[ACTION]
        if actions.ndim == 2:
            raise ValueError(
                "Training batch is expected to contain both base and target actions with shape (B, 2, action_dim)."
            )
        base_action = actions[:, 0]
        target_action = actions[:, 1]

        residual_norm = self.model(batch, base_action, batch.get("base_action_chunk"))
        target_residual = target_action - base_action
        mse_loss = F.mse_loss(residual_norm, target_residual, reduction="mean")

        return mse_loss, {"mse_loss": mse_loss.item()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, Tensor]) -> Tensor:
        batch = dict(batch)
        batch = self._prepare_batch(batch)

        actions = batch.get(ACTION)
        if actions is None:
            raise ValueError("Batch must contain key 'action' with the base policy action in [:, 0].")
        if actions.ndim == 3 and actions.shape[1] >= 1:
            base_action = actions[:, 0]
        elif actions.ndim == 2:
            base_action = actions
        else:
            raise ValueError("Unexpected action tensor shape. Expected (B, action_dim) or (B, >=1, action_dim).")

        residual_norm = self.model(batch, base_action, batch.get("base_action_chunk"))
        action_norm = base_action + residual_norm
        action = self.unnormalize_outputs({ACTION: action_norm.unsqueeze(1)})[ACTION]
        return action

    @torch.no_grad()
    def select_action(self, batch: Dict[str, Tensor]) -> Tensor:
        return self.predict_action_chunk(batch)[:, 0]


class ResidualTransformer(nn.Module):
    def __init__(self, config: ResidualTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.dim_model = config.dim_model
        self._spatial_pos_cache: dict[tuple[int, int], dict[tuple[str, torch.dtype], Tensor]] = {}

        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
            )
            self.image_out_dim = backbone_model.fc.in_features
            self.image_encoder = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
            self.image_proj = nn.Conv2d(self.image_out_dim, self.dim_model, kernel_size=1)

            if config.freeze_vision_backbone:
                self.image_encoder.eval()
                for param in self.image_encoder.parameters():
                    param.requires_grad = False
        else:
            self.image_encoder = None
            self.image_proj = None

        # Modal projections.
        if self.config.robot_state_feature is not None:
            self.state_proj = nn.Linear(self.config.robot_state_feature.shape[0], self.dim_model)
        else:
            self.state_proj = None
        if self.config.env_state_feature is not None:
            self.env_state_proj = nn.Linear(self.config.env_state_feature.shape[0], self.dim_model)
        else:
            self.env_state_proj = None

        self.action_proj = nn.Linear(self.config.action_feature.shape[0], self.dim_model)
        self.time_proj = nn.Linear(2, self.dim_model)
        vlm_feature = self.config.input_features.get("vlm_hidden")
        if vlm_feature is not None:
            self.vlm_hidden_proj = nn.Linear(vlm_feature.shape[0], self.dim_model)
        else:
            self.vlm_hidden_proj = None
        self.task_proj = nn.Linear(1, self.dim_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.n_encoder_layers)

        self.out_norm = nn.LayerNorm(self.dim_model)
        hidden_dim = self.dim_model
        action_dim = self.config.action_feature.shape[0]
        mlp_hidden = hidden_dim * 2
        dropout = self.config.dropout
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, action_dim),
        )

    def _positional_encoding(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim_model, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / self.dim_model)
        )
        pe = torch.zeros(seq_len, self.dim_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(
        self,
        batch: Dict[str, Tensor],
        base_action: Tensor,
        base_action_chunk: Tensor | None = None,
    ) -> Tensor:
        tokens_main: list[Tensor] = []
        main_lengths: list[int] = []
        chunk_tokens: Tensor | None = None
        chunk_length = 0
        batch_size = base_action.shape[0]
        dtype = base_action.dtype
        device = base_action.device

        cls_token = self.cls_token.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
        tokens_main.append(cls_token)
        main_lengths.append(cls_token.shape[1])

        if base_action_chunk is None:
            base_action_chunk = batch.get("base_action_chunk")
        if base_action_chunk is not None:
            chunk = base_action_chunk.to(device=device, dtype=dtype)
            if chunk.ndim == 2:
                chunk = chunk.unsqueeze(1)
            chunk_tokens = self.action_proj(chunk)
            chunk_length = chunk_tokens.shape[1]

        time_feature = batch.get("time_feature")
        if time_feature is None:
            time_feature = torch.zeros(batch_size, 2, device=device, dtype=dtype)
        else:
            time_feature = time_feature.to(device=device, dtype=dtype)
        time_token = self.time_proj(time_feature).unsqueeze(1)
        tokens_main.append(time_token)
        main_lengths.append(time_token.shape[1])

        if self.state_proj is not None and "observation.state" in batch:
            state_values = batch["observation.state"].to(device=device, dtype=dtype)
            state_token = self.state_proj(state_values).unsqueeze(1)
            tokens_main.append(state_token)
            main_lengths.append(state_token.shape[1])
        if self.env_state_proj is not None and "observation.environment_state" in batch:
            env_values = batch["observation.environment_state"].to(device=device, dtype=dtype)
            env_token = self.env_state_proj(env_values).unsqueeze(1)
            tokens_main.append(env_token)
            main_lengths.append(env_token.shape[1])

        if self.vlm_hidden_proj is not None and "vlm_hidden" in batch:
            hidden_vec = batch["vlm_hidden"].to(device=device, dtype=dtype)
            vlm_token = self.vlm_hidden_proj(hidden_vec).unsqueeze(1)
            tokens_main.append(vlm_token)
            main_lengths.append(vlm_token.shape[1])

        tasks = batch.get("task")
        if tasks is not None:
            if isinstance(tasks, str):
                tasks = [tasks] * batch_size
            task_values = []
            for task in tasks:
                digest = hashlib.sha1(task.encode("utf-8")).digest()
                value = int.from_bytes(digest[:4], "little") / float(0xFFFFFFFF)
                task_values.append(value)
            task_tensor = torch.tensor(task_values, device=device, dtype=dtype).unsqueeze(1)
        else:
            task_tensor = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        task_token = self.task_proj(task_tensor).unsqueeze(1)
        tokens_main.append(task_token)
        main_lengths.append(task_token.shape[1])

        main_total = sum(main_lengths)
        total_len = main_total + chunk_length
        pos = self._positional_encoding(total_len, device, dtype=dtype) if total_len > 0 else None

        if main_total > 0:
            main_tokens = torch.cat(tokens_main, dim=1)
            if pos is not None:
                main_tokens = main_tokens + pos[:, :main_total]
        else:
            main_tokens = torch.empty(batch_size, 0, self.dim_model, device=device, dtype=dtype)

        if chunk_tokens is not None:
            if pos is not None:
                chunk_tokens = chunk_tokens + pos[:, main_total : main_total + chunk_length]
            base_tokens = torch.cat([main_tokens, chunk_tokens], dim=1)
        else:
            base_tokens = main_tokens

        image_tokens_concat: Tensor | None = None
        if self.image_encoder is not None:
            image_tokens = []
            for key in self.config.image_features:
                img = batch[key].to(device=device, dtype=torch.float32)
                features_dict = self.image_encoder(img)
                feature_map = features_dict.get("feature_map")
                if feature_map is None:
                    raise KeyError("Expected vision backbone to return key 'feature_map'.")
                feature_map = feature_map.to(device=device, dtype=torch.float32)
                projected = self.image_proj(feature_map).to(device=device, dtype=dtype)
                bsz, _, height, width = projected.shape
                pos_embed = self._get_spatial_pos_embed(height, width, device=device, dtype=dtype)
                pos_embed = pos_embed.unsqueeze(0).expand(bsz, -1, -1)
                projected = projected.flatten(2).transpose(1, 2)
                image_tokens.append(projected + pos_embed)
            if image_tokens:
                image_tokens_concat = torch.cat(image_tokens, dim=1)

        if image_tokens_concat is not None:
            x = torch.cat([base_tokens, image_tokens_concat], dim=1)
        else:
            x = base_tokens

        x = self.encoder(x)
        cls_state = self.out_norm(x[:, 0])
        residual = self.residual_head(torch.cat([cls_state, base_action], dim=-1))
        action_norm = base_action + residual
        return action_norm

    def _get_spatial_pos_embed(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        cache_key = (height, width)
        per_shape = self._spatial_pos_cache.setdefault(cache_key, {})
        device_key = (str(device), dtype)

        cached = per_shape.get(device_key)
        if cached is not None:
            return cached

        base_key = ("cpu", torch.float32)
        base = per_shape.get(base_key)
        if base is None:
            base = sinusoidal_position_embedding_2d(
                height,
                width,
                self.dim_model,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            per_shape[base_key] = base

        converted = base.to(device=device, dtype=dtype)
        per_shape[device_key] = converted
        return converted

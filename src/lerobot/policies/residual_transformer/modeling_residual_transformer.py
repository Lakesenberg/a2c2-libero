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
import math
from typing import Dict

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.constants import ACTION
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.residual_transformer.configuration_residual_transformer import (
    ResidualTransformerConfig,
)


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

        action_pred = self.model(batch, base_action)
        l1_loss = F.l1_loss(action_pred, target_action, reduction="mean")

        return l1_loss, {"l1_loss": l1_loss.item()}

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

        action_norm = self.model(batch, base_action)
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

        # Vision backbone -> single token per camera via global pooling.
        if self.config.image_features:
            backbone = getattr(torchvision.models, config.vision_backbone)(
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.image_out_dim = backbone.fc.in_features
            self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])  # outputs (B, C, 1, 1)
            self.image_proj = nn.Linear(self.image_out_dim, self.dim_model)
            if config.freeze_vision_backbone:
                self.image_encoder.eval()
                for param in self.image_encoder.parameters():
                    param.requires_grad = False
        else:
            self.image_encoder = None

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
        if self.config.use_language:
            self.language_proj = nn.Linear(960, self.dim_model)
        else:
            self.language_proj = None
        vlm_feature = self.config.input_features.get("vlm_hidden")
        if vlm_feature is not None:
            self.vlm_hidden_proj = nn.Linear(vlm_feature.shape[0], self.dim_model)
        else:
            self.vlm_hidden_proj = None

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
        self.action_head = nn.Linear(self.dim_model, self.config.action_feature.shape[0])

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

    def forward(self, batch: Dict[str, Tensor], base_action: Tensor) -> Tensor:
        tokens = []
        batch_size = base_action.shape[0]
        dtype = base_action.dtype
        device = base_action.device

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens.append(cls_token)

        tokens.append(self.action_proj(base_action).unsqueeze(1))

        time_feature = batch.get("time_feature")
        if time_feature is None:
            time_feature = torch.zeros(batch_size, 2, device=device, dtype=dtype)
        else:
            time_feature = time_feature.to(device=device, dtype=dtype)
        tokens.append(self.time_proj(time_feature).unsqueeze(1))

        if self.state_proj is not None and "observation.state" in batch:
            tokens.append(self.state_proj(batch["observation.state"].to(dtype)).unsqueeze(1))
        if self.env_state_proj is not None and "observation.environment_state" in batch:
            tokens.append(self.env_state_proj(batch["observation.environment_state"].to(dtype)).unsqueeze(1))

        if self.image_encoder is not None:
            image_tokens = []
            for key in self.config.image_features:
                img = batch[key].to(dtype=torch.float32)
                features = self.image_encoder(img).flatten(1)
                image_tokens.append(self.image_proj(features).unsqueeze(1))
            if image_tokens:
                tokens.append(torch.cat(image_tokens, dim=1))

        if self.vlm_hidden_proj is not None and "vlm_hidden" in batch:
            hidden_vec = batch["vlm_hidden"].to(device=device, dtype=dtype)
            tokens.append(self.vlm_hidden_proj(hidden_vec).unsqueeze(1))

        if self.language_proj is not None and "language_embedding" in batch:
            language_emb = batch["language_embedding"].to(dtype=torch.float32)
            language_tokens = self.language_proj(language_emb)
            tokens.append(language_tokens)

        x = torch.cat(tokens, dim=1)
        x = x + self._positional_encoding(x.shape[1], device, dtype=x.dtype)
        x = self.encoder(x)
        cls_state = self.out_norm(x[:, 0])
        action_norm = self.action_head(cls_state)
        return action_norm

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
from typing import Dict

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.constants import ACTION
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.residual_mlp.configuration_residual_mlp import ResidualMLPConfig


class ResidualMLPPolicy(PreTrainedPolicy):
    """Policy head that predicts refined actions using a frozen vision backbone and MLP fusion."""

    config_class = ResidualMLPConfig
    name = "residual_mlp"

    def __init__(
        self,
        config: ResidualMLPConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ) -> None:
        super().__init__(config)
        config.validate_features()

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        self.model = ResidualMLP(config)

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


class ResidualMLP(nn.Module):
    def __init__(self, config: ResidualMLPConfig) -> None:
        super().__init__()
        self.config = config
        self.dim_model = config.dim_model

        # Vision backbone with global pooling per camera.
        if self.config.image_features:
            backbone = getattr(torchvision.models, config.vision_backbone)(
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.image_out_dim = backbone.fc.in_features
            self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])
            self.image_proj = nn.Linear(self.image_out_dim, self.dim_model)
            if config.freeze_vision_backbone:
                self.image_encoder.eval()
                for param in self.image_encoder.parameters():
                    param.requires_grad = False
        else:
            self.image_encoder = None

        self.state_proj = (
            nn.Linear(self.config.robot_state_feature.shape[0], self.dim_model)
            if self.config.robot_state_feature is not None
            else None
        )
        self.env_state_proj = (
            nn.Linear(self.config.env_state_feature.shape[0], self.dim_model)
            if self.config.env_state_feature is not None
            else None
        )

        self.action_proj = nn.Linear(self.config.action_feature.shape[0], self.dim_model)
        self.time_proj = nn.Linear(2, self.dim_model)
        self.language_proj = nn.Linear(960, self.dim_model) if self.config.use_language else None
        vlm_feature = self.config.input_features.get("vlm_context")
        self.vlm_context_proj = (
            nn.Linear(vlm_feature.shape[0], self.dim_model) if vlm_feature is not None else None
        )

        self.feature_norm = nn.LayerNorm(self.dim_model)

        num_feature_vectors = 2  # action + time
        if self.state_proj is not None:
            num_feature_vectors += 1
        if self.env_state_proj is not None:
            num_feature_vectors += 1
        if self.image_encoder is not None:
            num_feature_vectors += len(self.config.image_features)
        if self.language_proj is not None:
            num_feature_vectors += 1
        if self.vlm_context_proj is not None:
            num_feature_vectors += 1

        in_dim = self.dim_model * num_feature_vectors
        hidden_dims = list(self.config.hidden_dims)
        if not hidden_dims:
            hidden_dims = [self.dim_model]

        layers: list[nn.Module] = []
        current_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, self.config.action_feature.shape[0]))
        self.mlp = nn.Sequential(*layers)

    def _encode_images(
        self,
        batch: Dict[str, Tensor],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[Tensor]:
        image_features: list[Tensor] = []
        if self.image_encoder is None:
            return image_features
        for key in self.config.image_features:
            img = batch.get(key)
            if img is None:
                image_features.append(torch.zeros(batch_size, self.dim_model, device=device, dtype=dtype))
                continue
            img = img.to(device=device, dtype=torch.float32)
            features = self.image_encoder(img).flatten(1)
            proj = self.image_proj(features).to(dtype=dtype)
            image_features.append(proj)
        return image_features

    def forward(self, batch: Dict[str, Tensor], base_action: Tensor) -> Tensor:
        batch_size = base_action.shape[0]
        dtype = base_action.dtype
        device = base_action.device

        features: list[Tensor] = []

        action_feat = self.action_proj(base_action)
        features.append(self.feature_norm(action_feat))

        time_feature = batch.get("time_feature")
        if time_feature is None:
            time_feature = torch.zeros(batch_size, 2, device=device, dtype=dtype)
        else:
            time_feature = time_feature.to(device=device, dtype=dtype)
        features.append(self.feature_norm(self.time_proj(time_feature)))

        if self.state_proj is not None:
            state = batch.get("observation.state")
            if state is None:
                state = torch.zeros(batch_size, self.config.robot_state_feature.shape[0], device=device, dtype=dtype)
            else:
                state = state.to(device=device, dtype=dtype)
            features.append(self.feature_norm(self.state_proj(state)))

        if self.env_state_proj is not None:
            env_state = batch.get("observation.environment_state")
            if env_state is None:
                env_state = torch.zeros(
                    batch_size,
                    self.config.env_state_feature.shape[0],
                    device=device,
                    dtype=dtype,
                )
            else:
                env_state = env_state.to(device=device, dtype=dtype)
            features.append(self.feature_norm(self.env_state_proj(env_state)))

        features.extend(
            self.feature_norm(feat)
            for feat in self._encode_images(batch, batch_size, dtype, device)
        )

        if self.language_proj is not None:
            language = batch.get("language_embedding")
            if language is None:
                language_feat = torch.zeros(batch_size, self.dim_model, device=device, dtype=dtype)
            else:
                language = language.to(device=device, dtype=torch.float32)
                language_feat = self.language_proj(language)
                if language_feat.ndim == 3:
                    language_feat = language_feat.mean(dim=1)
                language_feat = language_feat.to(dtype=dtype)
            features.append(self.feature_norm(language_feat))

        if self.vlm_context_proj is not None:
            context = batch.get("vlm_context")
            if context is None:
                context_feat = torch.zeros(batch_size, self.dim_model, device=device, dtype=dtype)
            else:
                context_feat = self.vlm_context_proj(context.to(device=device, dtype=dtype))
            features.append(self.feature_norm(context_feat))

        mlp_input = torch.cat(features, dim=-1)
        action_norm = self.mlp(mlp_input)
        return action_norm

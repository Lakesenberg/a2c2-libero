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
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("residual_transformer")
@dataclass
class ResidualTransformerConfig(PreTrainedConfig):
    """Configuration for a lightweight residual transformer policy.

    The policy takes the current observation (images + robot/environment state), the base policy action
    (provided in `action[:, 0]`), and optional language embeddings to predict an additive residual.
    The final executed action is obtained as `base_action + residual`.
    """

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
            "ENV": NormalizationMode.MEAN_STD,
        }
    )

    # Vision backbone settings.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    freeze_vision_backbone: bool = False

    # Transformer architecture.
    dim_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 10
    dim_feedforward: int = 2048
    dropout: float = 0.1

    # Modality toggles.
    use_language: bool = True

    # Optimizer defaults.
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the torchvision ResNet variants. Got {self.vision_backbone}."
            )

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature and not self.robot_state_feature:
            raise ValueError(
                "ResidualTransformer requires at least one observation modality among images, robot state or environment state."
            )
        if self.action_feature is None:
            raise ValueError("ResidualTransformer requires an action feature in the dataset metadata.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

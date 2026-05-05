"""Minimal A2C2 residual head (MLP variant, ~0.3M parameters).

This is the small head used in A2C2's Kinetix experiments (paper §4). For
LIBERO / bimanual settings the paper recommends the larger Transformer head;
that one is left for follow-up.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..inference.utils import flatten_state_for_mlp


class A2C2MLPHead(nn.Module):
    """A 3-layer MLP that maps state -> Δa.

    The state must be flattened by `flatten_state_for_mlp` before being fed
    to the MLP. The head is intentionally small (~0.3M params) to meet the
    per-tick latency budget.

    Parameters
    ----------
    input_dim : int
        Total flattened state dimension. Must match what the runtime feeder
        produces; mismatches will raise at the first forward.
    action_dim : int
        Output (residual) dimension.
    hidden : int
        MLP width. Default 512 (paper).
    """

    def __init__(self, input_dim: int, action_dim: int, hidden: int = 512) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        x = flatten_state_for_mlp(state) if isinstance(state, dict) else state
        # Squeeze extra batch dim of 1 if present (per-tick inference).
        squeeze_back = x.dim() == 2 and x.shape[0] == 1
        if squeeze_back:
            x = x.squeeze(0)
        out = self.net(x)
        return out

    def loss(self, state: dict[str, torch.Tensor], y_gt: torch.Tensor) -> torch.Tensor:
        """Standard MSE residual loss: ||π_C(s) - (a_gt - a_base)||^2."""
        delta_pred = self.forward(state)
        return ((delta_pred - y_gt) ** 2).mean()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

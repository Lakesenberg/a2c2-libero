"""Tests for the small A2C2 MLP head."""
from __future__ import annotations

import torch

from a2c2_libero.heads import A2C2MLPHead
from a2c2_libero.inference.utils import build_state


INPUT_DIM = 20 + 7 + 2 + 384       # obs + a_base + tau_k + z
ACTION_DIM = 7


def _state():
    obs = {"flat": torch.zeros(20)}
    return build_state(
        obs=obs,
        a_base=torch.zeros(ACTION_DIM),
        tau_k=torch.zeros(2),
        z=torch.zeros(384),
    )


def test_head_forward_shape():
    head = A2C2MLPHead(input_dim=INPUT_DIM, action_dim=ACTION_DIM, hidden=128)
    out = head(_state())
    assert out.shape == (ACTION_DIM,)


def test_head_param_count_small():
    head = A2C2MLPHead(input_dim=INPUT_DIM, action_dim=ACTION_DIM, hidden=512)
    # Should be < 1M for the lightweight variant
    assert head.num_parameters < 1_000_000


def test_head_loss_decreases_after_one_grad_step():
    """Sanity check that the head can fit a constant residual."""
    torch.manual_seed(0)
    head = A2C2MLPHead(input_dim=INPUT_DIM, action_dim=ACTION_DIM, hidden=128)
    optim = torch.optim.SGD(head.parameters(), lr=0.1)
    target = torch.full((ACTION_DIM,), 0.5)

    # Use a non-degenerate state
    obs = {"flat": torch.randn(20)}
    state = build_state(obs, torch.randn(ACTION_DIM), torch.tensor([0.5, 0.5]), torch.randn(384))

    loss0 = head.loss(state, target).item()
    optim.zero_grad()
    head.loss(state, target).backward()
    optim.step()
    loss1 = head.loss(state, target).item()
    assert loss1 < loss0, f"loss did not decrease: {loss0} -> {loss1}"

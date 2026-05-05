"""Tests for sin/cos encoding and state builders."""
from __future__ import annotations

import math

import pytest
import torch

from a2c2_libero.inference.utils import (
    build_state,
    flatten_state_for_mlp,
    sincos_pos_encoding,
)


def test_sincos_at_zero():
    e = sincos_pos_encoding(0, H=50)
    assert torch.allclose(e, torch.tensor([0.0, 1.0]), atol=1e-6)


def test_sincos_at_quarter():
    e = sincos_pos_encoding(12, H=48)         # k/H = 0.25 → 2π·0.25 = π/2
    assert torch.allclose(e, torch.tensor([1.0, 0.0]), atol=1e-6)


def test_sincos_at_half():
    e = sincos_pos_encoding(25, H=50)         # k/H = 0.5 → 2π·0.5 = π
    assert torch.allclose(e, torch.tensor([0.0, -1.0]), atol=1e-6)


def test_sincos_periodic_full():
    """k = H gives the same encoding as k = 0 (period)."""
    a = sincos_pos_encoding(0, H=50)
    b = sincos_pos_encoding(50, H=50)
    assert torch.allclose(a, b, atol=1e-6)


def test_sincos_invalid_H_raises():
    with pytest.raises(ValueError):
        sincos_pos_encoding(0, H=0)


def test_build_state_required_fields():
    obs = {"image": torch.zeros(3, 16, 16)}
    a_base = torch.zeros(7)
    tau_k = torch.zeros(2)
    z = torch.zeros(384)
    s = build_state(obs, a_base, tau_k, z)
    assert "obs" in s and "a_base" in s and "tau_k" in s and "z" in s
    assert "lang_emb" not in s


def test_build_state_optional_lang():
    s = build_state({}, torch.zeros(7), torch.zeros(2), torch.zeros(384), lang_emb=torch.zeros(64))
    assert "lang_emb" in s and s["lang_emb"].shape == (64,)


def test_flatten_state_for_mlp_shape():
    obs = {"flat": torch.zeros(20)}
    s = build_state(obs, torch.zeros(7), torch.zeros(2), torch.zeros(384))
    out = flatten_state_for_mlp(s)
    # 20 (obs) + 7 (a_base) + 2 (tau_k) + 384 (z) = 413
    assert out.shape[-1] == 413

"""Shared utilities for A2C2 inference."""
from __future__ import annotations

import math
from typing import Any

import torch


def sincos_pos_encoding(k: int, H: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """Sin/cos positional encoding of chunk index k in a chunk of length H.

    Returns a 2-D tensor [sin(2*pi*k/H), cos(2*pi*k/H)].
    Matches the formulation used in the A2C2 paper (arXiv:2509.23224, §3.2).
    """
    if H <= 0:
        raise ValueError(f"H must be positive, got {H}")
    angle = 2.0 * math.pi * k / H
    return torch.tensor([math.sin(angle), math.cos(angle)], device=device)


def build_state(
    obs: dict[str, torch.Tensor],
    a_base: torch.Tensor,
    tau_k: torch.Tensor,
    z: torch.Tensor,
    lang_emb: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build the input state dict for the A2C2 head.

    The exact format must match what was used at training time. This is a
    canonical version: future heads may expect different field names; keep this
    function as the single source of truth and re-train if you change it.
    """
    state: dict[str, torch.Tensor] = {
        "obs": obs,
        "a_base": a_base,
        "tau_k": tau_k,
        "z": z,
    }
    if lang_emb is not None:
        state["lang_emb"] = lang_emb
    return state


def flatten_state_for_mlp(state: dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a state dict into a single vector for an MLP head.

    For a Transformer-based head, you would tokenize differently; this helper
    is provided for the lightweight MLP variant of the A2C2 head.
    """
    parts = []
    obs = state["obs"]
    if isinstance(obs, dict):
        # Concatenate all observation tensors after flattening.
        for key in sorted(obs.keys()):
            v = obs[key]
            parts.append(v.reshape(v.shape[0] if v.ndim > 1 else 1, -1) if v.ndim >= 1 else v.reshape(1, -1))
    else:
        parts.append(obs.reshape(obs.shape[0] if obs.ndim > 1 else 1, -1))

    for key in ("a_base", "tau_k", "z", "lang_emb"):
        if key in state:
            v = state[key]
            parts.append(v.reshape(v.shape[0] if v.ndim > 1 else 1, -1))

    # Align batch dims; assume all have same first dim or are 1-d.
    return torch.cat(parts, dim=-1)


class LatentHook:
    """Forward hook to extract a pooled hidden state from a SmolVLA backbone.

    Usage:
        hook = LatentHook(smolvla.model.vlm_backbone.layers[-1])
        smolvla.predict_action_chunk(obs)
        z = hook.latest          # last extracted latent

    The hook stores only the most recent extracted feature; thread-safe access
    is the caller's responsibility.
    """

    def __init__(self, target_module, pooling: str = "cls"):
        self.latest: torch.Tensor | None = None
        self.pooling = pooling
        self._handle = target_module.register_forward_hook(self._hook)

    def _hook(self, module, inputs: tuple[Any, ...], output: Any) -> None:
        h = output[0] if isinstance(output, tuple) else output
        if h.ndim == 3:
            if self.pooling == "cls":
                self.latest = h[:, 0, :].detach()
            elif self.pooling == "mean":
                self.latest = h.mean(dim=1).detach()
            else:
                raise ValueError(f"unknown pooling: {self.pooling}")
        elif h.ndim == 2:
            self.latest = h.detach()
        else:
            raise ValueError(f"unexpected hidden state shape: {tuple(h.shape)}")

    def remove(self) -> None:
        self._handle.remove()

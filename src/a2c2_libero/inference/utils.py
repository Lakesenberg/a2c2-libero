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
    base_action_chunk: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build the input state dict for the A2C2 residual transformer head.

    The exact format must match what was used at training time
    (`src/lerobot/scripts/train_residual_transformer.py`). The trained head
    expects:

      - ``observation.state``     - current proprio (single step)
      - ``observation.image*``    - camera frames
      - ``action``                - the single base action to be executed at this tick
      - ``base_action_chunk``     - the FULL chunk predicted by the base SmolVLA at
                                    the current chunk start (shape [H, action_dim])
      - ``time_feature``          - sin/cos chunk-index encoding (shape [2])
      - ``vlm_hidden``            - cached VLM latent from SmolVLA at the chunk start

    Without ``base_action_chunk`` and ``time_feature`` the residual transformer
    falls back to the single-action token path and refinement quality drops; see
    the upstream README at
    https://github.com/k1000dai/a2c2-libero#residual-transformer .
    """
    state: dict[str, torch.Tensor] = {
        # Legacy nested-obs key (consumed by MLP head and old tests).
        "obs": obs,
        # Backwards-compat aliases.
        "a_base": a_base,
        "tau_k": tau_k,
        "z": z,
    }
    # Flat-key schema matching train_residual_transformer.py contract.
    state["action"] = a_base
    state["time_feature"] = tau_k
    state["vlm_hidden"] = z
    if base_action_chunk is not None:
        state["base_action_chunk"] = base_action_chunk
    if lang_emb is not None:
        state["lang_emb"] = lang_emb
    # Also flatten obs dict keys to top level so the residual-transformer
    # contract keys ("observation.state", "observation.images.*") are
    # directly accessible without unwrapping `state["obs"]`.
    if isinstance(obs, dict):
        for k, v in obs.items():
            if k not in state:
                state[k] = v
    return state


def flatten_state_for_mlp(state: dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a state dict into a single vector for an MLP head.

    For a Transformer-based head, you would tokenize differently; this helper
    is provided for the lightweight MLP variant of the A2C2 head. It only
    consumes the canonical aliases (``a_base``, ``tau_k``, ``z``) plus any
    observation tensors. ``base_action_chunk`` is intentionally skipped here
    because the MLP head doesn't use chunk context.
    """
    parts = []
    obs = state.get("obs")
    if isinstance(obs, dict):
        for key in sorted(obs.keys()):
            v = obs[key]
            parts.append(v.reshape(v.shape[0] if v.ndim > 1 else 1, -1) if v.ndim >= 1 else v.reshape(1, -1))
    elif isinstance(obs, torch.Tensor):
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

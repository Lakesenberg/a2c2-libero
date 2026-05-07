"""A2C2 inference engine combining the async SmolVLA worker and the
per-tick correction head.

Two flavors are provided:

* `A2C2Engine.step_sync(...)`: blocking SmolVLA refresh when no chunk is
  available or chunk is consumed. Simpler, no threads. Suitable for
  simulation eval where wall-clock latency does not matter.

* `A2C2Engine.step_async(...)`: non-blocking; uses a background SmolVLA
  worker. Suitable for real-time control on a robot.
"""
from __future__ import annotations

from typing import Any, Callable

import torch

from .async_smolvla import AsyncSmolVLAWorker, SharedBoard
from .utils import build_state, sincos_pos_encoding


HeadFn = Callable[[dict[str, torch.Tensor]], torch.Tensor]   # state -> Δa


class A2C2Engine:
    """Hierarchical inference engine: SmolVLA (slow) + A2C2 head (per-tick).

    Parameters
    ----------
    smolvla_fn : callable
        `(obs, language) -> (chunk[H,A], z[d_z])`.
    head_fn : callable
        `state_dict -> Δa[A]`. Must be deterministic given the state.
    chunk_size : int
        Length of the SmolVLA action chunk (H).
    action_dim : int
        Action dimension (A).
    safe_action : torch.Tensor | None
        Action returned during cold start when no chunk is available.
        Defaults to zeros of shape (action_dim,).
    device : str | torch.device
        Device for the sin/cos encoding tensor.
    """

    def __init__(
        self,
        smolvla_fn: Callable[[Any, str | None], tuple[torch.Tensor, torch.Tensor]],
        head_fn: HeadFn,
        chunk_size: int,
        action_dim: int,
        safe_action: torch.Tensor | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.smolvla_fn = smolvla_fn
        self.head_fn = head_fn
        self.H = chunk_size
        self.A = action_dim
        self.device = device
        self.safe_action = safe_action if safe_action is not None else torch.zeros(action_dim)

        self.board = SharedBoard()
        self._worker: AsyncSmolVLAWorker | None = None
        self.global_tick = 0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Reset all state. Call at the start of each episode."""
        was_running = self._worker is not None and self._worker.is_running()
        if was_running:
            self._worker.stop()
        self.board = SharedBoard()
        self.global_tick = 0
        if was_running:
            self.start_async()

    def start_async(self) -> None:
        """Spawn the background SmolVLA worker."""
        self._worker = AsyncSmolVLAWorker(
            board=self.board,
            policy_fn=self.smolvla_fn,
            get_global_tick=lambda: self.global_tick,
        )
        self._worker.start()

    def stop_async(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

    # ------------------------------------------------------------------ #
    # Synchronous step: blocking SmolVLA refresh when needed
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step_sync(self, obs: Any, language: str | None = None) -> torch.Tensor:
        """One control tick, blocking on SmolVLA when chunk is exhausted.

        Feeds the residual head the same fields the training pipeline
        produced: single-step ``action`` + full ``base_action_chunk`` +
        ``time_feature`` + ``vlm_hidden``. See README of upstream
        a2c2-libero for the contract.
        """
        chunk = self.board.chunk
        if chunk is None or self.board.k >= self.H:
            chunk, z = self.smolvla_fn(obs, language)
            self.board.install_chunk(chunk, z, self.global_tick)

        k_eff = min(self.board.k, self.H - 1)
        full_chunk = self.board.chunk          # [H, action_dim]
        a_base = full_chunk[k_eff]
        z = self.board.z

        tau_k = sincos_pos_encoding(k_eff, self.H, device=self.device)
        state = build_state(obs, a_base, tau_k, z, base_action_chunk=full_chunk)
        delta = self.head_fn(state)

        self.board.advance_k(max_k=self.H)
        self.global_tick += 1
        return a_base + delta

    # ------------------------------------------------------------------ #
    # Asynchronous step: non-blocking, reads latest chunk from the board
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step_async(self, obs: Any, language: str | None = None) -> torch.Tensor:
        """One control tick, non-blocking. Requires `start_async()` first."""
        if self._worker is None or not self._worker.is_running():
            raise RuntimeError("call start_async() before step_async()")
        self._worker.raise_if_error()

        # Publish latest obs+lang for the background worker.
        self.board.write_obs(obs, language)

        chunk, z, k_now, chunk_id = self.board.snapshot()

        if chunk is None:
            # Cold start: SmolVLA has not produced its first chunk yet.
            self.global_tick += 1
            return self.safe_action.clone()

        # Clip k in case SmolVLA is too slow and chunk has been fully consumed.
        k_eff = min(k_now, self.H - 1)
        a_base = chunk[k_eff]

        tau_k = sincos_pos_encoding(k_eff, self.H, device=self.device)
        # Pass the full chunk (not just a_base) so the residual transformer
        # can attend over the entire predicted trajectory, matching the
        # training-time interface in train_residual_transformer.py.
        state = build_state(obs, a_base, tau_k, z, base_action_chunk=chunk)
        delta = self.head_fn(state)

        self.board.advance_k(max_k=self.H)
        self.global_tick += 1
        return a_base + delta

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    @property
    def chunks_produced(self) -> int:
        return self._worker.forward_count if self._worker is not None else 0

    def diagnostics(self) -> dict[str, Any]:
        with self.board.lock:
            return {
                "global_tick": self.global_tick,
                "chunk_id": self.board.chunk_id,
                "k": self.board.k,
                "chunk_emit_tick": self.board.chunk_emit_tick,
                "chunks_produced": self.chunks_produced,
                "have_chunk": self.board.chunk is not None,
            }

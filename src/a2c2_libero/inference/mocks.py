"""Mock policies and head for testing the A2C2 async pipeline.

These let us verify the threading / state-machine behaviour without loading
SmolVLA weights or running on a GPU.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import torch


class MockSmolVLA:
    """Mock SmolVLA that sleeps for a configurable duration and returns a
    deterministic chunk + latent.

    Each call returns chunk filled with `chunk_value(call_idx)` so tests can
    distinguish chunks from each other.
    """

    def __init__(
        self,
        chunk_size: int = 50,
        action_dim: int = 7,
        latent_dim: int = 384,
        forward_latency_s: float = 0.10,
    ) -> None:
        self.H = chunk_size
        self.A = action_dim
        self.d_z = latent_dim
        self.latency = forward_latency_s

        self._call_count = 0
        self._lock = threading.Lock()

    def __call__(self, obs: Any, language: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        time.sleep(self.latency)
        with self._lock:
            idx = self._call_count
            self._call_count += 1
        # Distinct fill values so tests can assert which chunk is in use.
        chunk = torch.full((self.H, self.A), float(idx), dtype=torch.float32)
        z = torch.full((self.d_z,), float(idx), dtype=torch.float32)
        return chunk, z

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count


class MockA2C2Head:
    """Mock A2C2 head: deterministic, tiny, no learning.

    Returns a Δa whose magnitude is configurable so tests can assert that
    a_exec = a_base + Δa with the expected values.
    """

    def __init__(self, action_dim: int = 7, delta_scale: float = 0.01) -> None:
        self.A = action_dim
        self.delta_scale = delta_scale
        self.call_count = 0

    def __call__(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        self.call_count += 1
        # Make Δa depend on tau_k so we can verify it changes per tick.
        tau_k = state["tau_k"]
        z = state["z"]
        # Simple closed-form so tests can predict outputs:
        # Δa_j = delta_scale * (sin(2πk/H) * j + 0.1 * z[0])
        idx = torch.arange(self.A, dtype=torch.float32)
        return self.delta_scale * (tau_k[0] * idx + 0.1 * z[0])

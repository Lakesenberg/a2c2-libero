"""Asynchronous SmolVLA inference worker.

The high-level policy (SmolVLA) runs in a dedicated background thread and
publishes its action chunks + latent token to a `SharedBoard`. The main
control loop never blocks on SmolVLA; it reads the most recently published
chunk every tick and corrects it via the A2C2 head.

This module is policy-agnostic: anything with a callable interface
`policy(obs) -> (chunk, z)` can be wrapped. A real SmolVLA wrapper and a
mock policy are both used in the tests.
"""
from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch


ChunkAndLatent = tuple[torch.Tensor, torch.Tensor]
PolicyFn = Callable[[Any, Optional[str]], ChunkAndLatent]


@dataclass
class SharedBoard:
    """Mutable shared state between the SmolVLA worker and the main loop.

    All accesses must be guarded by `lock`. Snapshot reads use `.snapshot()`
    which returns a triple `(chunk, z, k)` while holding the lock briefly.
    """

    chunk: torch.Tensor | None = None
    z: torch.Tensor | None = None
    k: int = 0
    chunk_id: int = -1                 # increments each time SmolVLA writes
    chunk_emit_tick: int = -1
    latest_obs: Any = None
    language: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> tuple[torch.Tensor | None, torch.Tensor | None, int, int]:
        with self.lock:
            return self.chunk, self.z, self.k, self.chunk_id

    def write_obs(self, obs: Any, language: str | None) -> None:
        with self.lock:
            self.latest_obs = obs
            self.language = language

    def advance_k(self, max_k: int | None = None) -> int:
        """Increment k. Returns the value before increment.

        If max_k is given, k is clipped at max_k (used for safety when SmolVLA
        is too slow and the chunk has been fully consumed).
        """
        with self.lock:
            cur = self.k
            self.k = cur + 1 if max_k is None else min(cur + 1, max_k)
            return cur

    def install_chunk(self, chunk: torch.Tensor, z: torch.Tensor, tick: int) -> None:
        with self.lock:
            self.chunk = chunk
            self.z = z
            self.k = 0
            self.chunk_id = self.chunk_id + 1
            self.chunk_emit_tick = tick


class AsyncSmolVLAWorker:
    """Background worker that runs `policy_fn` on the latest obs in a loop.

    The worker reads obs+language from `board.latest_obs`/`board.language`,
    calls `policy_fn(obs, language)` which must return `(chunk, z)`, and
    publishes the result to the board. It tracks how many chunks it has
    produced (`chunk_id`).

    Parameters
    ----------
    board : SharedBoard
        Shared state.
    policy_fn : callable
        `(obs, language) -> (chunk_tensor[H, A], z_tensor[d_z])`.
    get_global_tick : callable, optional
        Function returning the current global tick (for emit-tick logging).
        Defaults to a monotonic counter starting at 0.
    poll_interval_s : float
        Sleep when obs is not yet available (cold start). Default 1 ms.
    """

    def __init__(
        self,
        board: SharedBoard,
        policy_fn: PolicyFn,
        get_global_tick: Callable[[], int] | None = None,
        poll_interval_s: float = 1e-3,
    ) -> None:
        self.board = board
        self.policy_fn = policy_fn
        self.get_global_tick = get_global_tick or (lambda: 0)
        self.poll_interval_s = poll_interval_s

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self.forward_count = 0           # incremented each successful forward

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("worker already running")
        self._stop.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 2.0) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=timeout)
        self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def raise_if_error(self) -> None:
        if self._error is not None:
            raise self._error

    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                with self.board.lock:
                    obs = self.board.latest_obs
                    lang = self.board.language
                if obs is None:
                    time.sleep(self.poll_interval_s)
                    continue

                # Deep-copy obs out of the lock so policy_fn can read freely
                # without blocking the main thread on long inference.
                obs_copy = copy.deepcopy(obs)

                # The actual SmolVLA forward (no lock held).
                chunk, z = self.policy_fn(obs_copy, lang)
                emit_tick = self.get_global_tick()
                self.board.install_chunk(chunk, z, emit_tick)
                self.forward_count += 1
        except BaseException as e:           # noqa: BLE001
            self._error = e
            raise

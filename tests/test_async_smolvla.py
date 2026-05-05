"""Tests for the asynchronous SmolVLA worker.

These tests use a `MockSmolVLA` with a configurable forward latency to verify
the threading / shared-board behaviour without loading any real model.

Run with `pytest tests/test_async_smolvla.py -v`.
"""
from __future__ import annotations

import time

import pytest
import torch

from a2c2_libero.inference.async_smolvla import AsyncSmolVLAWorker, SharedBoard
from a2c2_libero.inference.mocks import MockSmolVLA


def _wait_for(condition, timeout_s: float = 5.0, poll_s: float = 0.005):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(poll_s)
    return False


# --------------------------------------------------------------------------- #
def test_cold_start_chunk_is_none():
    """Before any obs is published, the worker should not produce anything."""
    board = SharedBoard()
    smolvla = MockSmolVLA(forward_latency_s=0.05)
    worker = AsyncSmolVLAWorker(board, smolvla)
    worker.start()
    try:
        time.sleep(0.1)             # let it spin
        chunk, z, k, chunk_id = board.snapshot()
        assert chunk is None
        assert z is None
        assert chunk_id == -1
        assert smolvla.call_count == 0
    finally:
        worker.stop()


def test_first_chunk_is_published_after_obs():
    """Once obs is set, the worker should produce a chunk within ~latency."""
    board = SharedBoard()
    smolvla = MockSmolVLA(forward_latency_s=0.05)
    worker = AsyncSmolVLAWorker(board, smolvla)
    worker.start()
    try:
        board.write_obs({"x": torch.zeros(3)}, language="pick the cup")
        ok = _wait_for(lambda: board.chunk is not None, timeout_s=1.0)
        assert ok, "no chunk produced after obs published"
        chunk, z, k, chunk_id = board.snapshot()
        assert chunk.shape == (50, 7)
        assert z.shape == (384,)
        assert k == 0
        assert chunk_id == 0
        # First call values are 0.0
        assert chunk[0, 0].item() == 0.0
    finally:
        worker.stop()


def test_chunk_id_increments_and_k_resets():
    """Repeated forwards should increment chunk_id and reset k each time."""
    board = SharedBoard()
    smolvla = MockSmolVLA(forward_latency_s=0.02)
    worker = AsyncSmolVLAWorker(board, smolvla)
    worker.start()
    try:
        board.write_obs({"x": torch.zeros(3)}, language="task")

        # Wait for first chunk
        assert _wait_for(lambda: board.chunk_id >= 0, timeout_s=1.0)

        # Simulate main loop advancing k
        for _ in range(20):
            board.advance_k(max_k=50)

        # Wait for second chunk to overwrite
        prev_id = board.chunk_id
        assert _wait_for(lambda: board.chunk_id > prev_id, timeout_s=1.0)

        chunk, z, k, chunk_id = board.snapshot()
        assert chunk_id >= 1
        assert k == 0, "k should reset to 0 when new chunk is installed"
    finally:
        worker.stop()


def test_main_loop_does_not_block_on_smolvla():
    """Main loop must remain responsive while SmolVLA is computing.

    With a 200 ms SmolVLA forward and a 5 ms main tick, in 1 s we should be
    able to step ~200 times even though SmolVLA only finishes ~5 forwards.
    """
    board = SharedBoard()
    smolvla = MockSmolVLA(forward_latency_s=0.20)
    worker = AsyncSmolVLAWorker(board, smolvla)
    worker.start()
    try:
        board.write_obs({"x": torch.zeros(3)}, language="task")
        time.sleep(0.05)            # let first forward kick off

        steps = 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.0:
            time.sleep(0.005)       # 5 ms tick
            chunk, z, k, _ = board.snapshot()
            steps += 1

        assert steps > 100, f"main loop only stepped {steps} times in 1 s"
        # 5 forwards in 1 s with 200 ms latency:
        assert 3 <= smolvla.call_count <= 7
    finally:
        worker.stop()


def test_worker_propagates_exceptions():
    """If the policy raises, the worker captures it and stop() returns."""
    board = SharedBoard()

    def raising_policy(obs, lang):
        raise RuntimeError("boom")

    worker = AsyncSmolVLAWorker(board, raising_policy)
    worker.start()
    try:
        board.write_obs({"x": torch.zeros(1)}, language="x")
        # Wait for the worker to die
        assert _wait_for(lambda: not worker.is_running(), timeout_s=1.0)
        with pytest.raises(RuntimeError, match="boom"):
            worker.raise_if_error()
    finally:
        worker.stop()


def test_stop_is_idempotent():
    """stop() called twice (or before start) should be a no-op."""
    board = SharedBoard()
    worker = AsyncSmolVLAWorker(board, MockSmolVLA(forward_latency_s=0.01))
    worker.stop()                   # before start
    worker.start()
    worker.stop()
    worker.stop()                   # second time
    assert not worker.is_running()


def test_double_start_raises():
    board = SharedBoard()
    worker = AsyncSmolVLAWorker(board, MockSmolVLA(forward_latency_s=0.01))
    worker.start()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            worker.start()
    finally:
        worker.stop()

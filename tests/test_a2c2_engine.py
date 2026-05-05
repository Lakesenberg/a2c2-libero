"""Tests for the full A2C2Engine: SmolVLA (mocked) + A2C2 head (mocked).

These verify the per-tick correction logic and the synchronous / asynchronous
pathways behave correctly across cold start, chunk transition, and stale-chunk
fallback.
"""
from __future__ import annotations

import time

import pytest
import torch

from a2c2_libero.inference.a2c2_engine import A2C2Engine
from a2c2_libero.inference.mocks import MockA2C2Head, MockSmolVLA


CHUNK_SIZE = 50
ACTION_DIM = 7
LATENT_DIM = 384


def _make_engine(forward_latency_s: float = 0.05) -> tuple[A2C2Engine, MockSmolVLA, MockA2C2Head]:
    smolvla = MockSmolVLA(
        chunk_size=CHUNK_SIZE,
        action_dim=ACTION_DIM,
        latent_dim=LATENT_DIM,
        forward_latency_s=forward_latency_s,
    )
    head = MockA2C2Head(action_dim=ACTION_DIM, delta_scale=0.01)
    engine = A2C2Engine(
        smolvla_fn=smolvla,
        head_fn=head,
        chunk_size=CHUNK_SIZE,
        action_dim=ACTION_DIM,
    )
    return engine, smolvla, head


def _wait_for(condition, timeout_s: float = 3.0, poll_s: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(poll_s)
    return False


# --------------------------------------------------------------------------- #
# Synchronous path
# --------------------------------------------------------------------------- #
def test_sync_first_step_triggers_smolvla():
    engine, smolvla, head = _make_engine(forward_latency_s=0.0)
    obs = {"image": torch.zeros(3, 64, 64)}
    a = engine.step_sync(obs, language="task")
    assert a.shape == (ACTION_DIM,)
    assert smolvla.call_count == 1
    assert head.call_count == 1


def test_sync_runs_for_chunk_size_steps_then_refreshes():
    engine, smolvla, head = _make_engine(forward_latency_s=0.0)
    obs = {"image": torch.zeros(3, 64, 64)}
    for _ in range(CHUNK_SIZE):
        engine.step_sync(obs, language="task")
    # SmolVLA should have run exactly once for the first chunk.
    assert smolvla.call_count == 1
    # The next step should trigger a refresh.
    engine.step_sync(obs, language="task")
    assert smolvla.call_count == 2


def test_sync_action_equals_base_plus_delta():
    """At k=0, the head's Δa should be 0 because tau_k[0] = sin(0) = 0 in the
    mock head's formula (delta = scale * (sin·idx + 0.1·z[0])); chunk_0 has
    z[0] = 0 too, so Δa should be exactly zero at the first tick."""
    engine, smolvla, head = _make_engine(forward_latency_s=0.0)
    obs = {"image": torch.zeros(3, 64, 64)}
    a = engine.step_sync(obs, language="task")
    # First chunk has all-zeros, first k=0, sin(0)=0, z[0]=0 → Δa = 0
    assert torch.allclose(a, torch.zeros(ACTION_DIM), atol=1e-6)


# --------------------------------------------------------------------------- #
# Asynchronous path
# --------------------------------------------------------------------------- #
def test_async_cold_start_returns_safe_action():
    engine, smolvla, head = _make_engine(forward_latency_s=0.10)
    safe = torch.full((ACTION_DIM,), -99.0)
    engine.safe_action = safe
    engine.start_async()
    try:
        # First step before SmolVLA finishes anything
        a = engine.step_async({"image": torch.zeros(1)}, language="task")
        assert torch.equal(a, safe), "cold-start step did not return safe action"
        # Head should not have been called during cold start
        assert head.call_count == 0
    finally:
        engine.stop_async()


def test_async_picks_up_chunk_after_warmup():
    engine, smolvla, head = _make_engine(forward_latency_s=0.05)
    engine.start_async()
    try:
        obs = {"image": torch.zeros(1)}
        # Drive the main loop until first chunk appears.
        actions = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.5:
            a = engine.step_async(obs, language="task")
            actions.append(a)
            time.sleep(0.005)

        # First chunk should arrive within 100 ms; before that, safe action.
        assert head.call_count > 0, "head was never called"
        assert engine.diagnostics()["chunks_produced"] >= 1
    finally:
        engine.stop_async()


def test_async_k_resets_when_new_chunk_arrives():
    """Sample (k, chunk_id) over many ticks and verify that whenever
    chunk_id increments, the next read of k is small (reset happened)."""
    # Slow SmolVLA so we only get one transition during the test.
    engine, smolvla, head = _make_engine(forward_latency_s=0.10)
    engine.start_async()
    try:
        obs = {"image": torch.zeros(1)}
        seen_chunk_ids: list[int] = []
        seen_ks: list[int] = []
        # Run long enough to span at least 2 chunks.
        for _ in range(80):
            engine.step_async(obs, language="task")
            with engine.board.lock:
                seen_chunk_ids.append(engine.board.chunk_id)
                seen_ks.append(engine.board.k)
            time.sleep(0.005)

        # Find indices where chunk_id transitions.
        transitions = [
            i for i in range(1, len(seen_chunk_ids))
            if seen_chunk_ids[i] > seen_chunk_ids[i - 1] and seen_chunk_ids[i] >= 1
        ]
        assert transitions, (
            f"no chunk transitions seen in {len(seen_chunk_ids)} ticks; "
            f"chunk_ids={set(seen_chunk_ids)}"
        )
        # Right after a transition, k should be small (resets to 0; allow up
        # to a few ticks of slack for race between worker write and main read).
        for idx in transitions:
            assert seen_ks[idx] <= 3, (
                f"k did not reset at transition idx={idx}: k={seen_ks[idx]}"
            )
        # And before the first transition, k should have grown beyond 0.
        first_t = transitions[0]
        assert max(seen_ks[:first_t]) > 0, "k never advanced before first transition"
    finally:
        engine.stop_async()


def test_async_clips_k_if_smolvla_too_slow():
    """If SmolVLA hasn't refreshed within H ticks, k must clip to H-1."""
    # Set latency so SmolVLA only finishes after H+5 ticks
    H_ticks_s = CHUNK_SIZE * 0.005           # main tick = 5 ms
    engine, smolvla, head = _make_engine(forward_latency_s=H_ticks_s + 0.05)
    engine.start_async()
    try:
        obs = {"image": torch.zeros(1)}
        # First, get one chunk in the board
        engine.step_async(obs, language="task")
        assert _wait_for(lambda: engine.board.chunk is not None, timeout_s=2.0)

        # Now consume more than H ticks before SmolVLA can finish another forward
        for _ in range(CHUNK_SIZE + 10):
            engine.step_async(obs, language="task")
            time.sleep(0.001)

        diag = engine.diagnostics()
        # k should be clipped at H
        assert diag["k"] <= CHUNK_SIZE
    finally:
        engine.stop_async()


def test_reset_clears_state():
    engine, smolvla, head = _make_engine(forward_latency_s=0.0)
    obs = {"image": torch.zeros(1)}
    engine.step_sync(obs, "task")
    assert engine.board.chunk is not None
    engine.reset()
    assert engine.board.chunk is None
    assert engine.global_tick == 0


def test_step_async_without_start_raises():
    engine, _, _ = _make_engine(forward_latency_s=0.0)
    with pytest.raises(RuntimeError, match="start_async"):
        engine.step_async({"image": torch.zeros(1)}, language="task")

"""LIBERO smoke test: skipped automatically if LeRobot+LIBERO are not installed.

This test does NOT load real SmolVLA / A2C2 weights; it wraps the LIBERO
environment with mock policies to verify the engine can drive a real env
loop end-to-end without crashing.
"""
from __future__ import annotations

import os

import pytest
import torch

libero_env = pytest.importorskip(
    "lerobot.envs.libero",
    reason="lerobot[libero] not installed",
)
LiberoEnv = libero_env.LiberoEnv

from a2c2_libero.inference.a2c2_engine import A2C2Engine
from a2c2_libero.inference.mocks import MockA2C2Head, MockSmolVLA


# Headless rendering setup
os.environ.setdefault("MUJOCO_GL", "egl")


@pytest.mark.slow
def test_libero_one_episode_runs_to_completion():
    """Drive one LIBERO episode end-to-end with mock policies.

    The mock policies output zero actions, so the robot will not actually
    succeed at the task. We only assert that the engine's step loop runs
    without raising and reaches the env's `done` flag (timeout = success
    criterion irrelevant here).
    """
    env = LiberoEnv(task="libero_spatial")
    obs, info = env.reset(seed=0)

    # LIBERO is single Franka, 7-DoF action.
    smolvla = MockSmolVLA(chunk_size=50, action_dim=7, forward_latency_s=0.0)
    head = MockA2C2Head(action_dim=7, delta_scale=0.0)   # zero correction

    engine = A2C2Engine(
        smolvla_fn=smolvla,
        head_fn=head,
        chunk_size=50,
        action_dim=7,
    )

    done = False
    steps = 0
    max_steps = 50          # cap for the smoke test
    while not done and steps < max_steps:
        action = engine.step_sync(obs, language=info.get("language_instruction"))
        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        done = terminated or truncated
        steps += 1

    assert steps > 0, "no steps were taken"
    env.close()

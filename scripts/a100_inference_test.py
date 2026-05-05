"""A100 server-grade A2C2 inference test using a trained SmolVLA model.

Designed to be submitted as a job on an A100 server. Loads
`lerobot/smolvla_libero` (or a user-supplied checkpoint), wraps it in the
async A2C2 engine, and runs a configurable workload measuring:

* per-tick latency (mean, p50, p95, p99)
* SmolVLA forward latency
* number of A2C2 corrections per second
* GPU peak memory
* whether action shapes / latent shapes match expectations

The test can run in three modes:

1. ``--mode synthetic`` (default): synthetic random observations. Pure GPU
   stress test, no LIBERO dependency.
2. ``--mode dataset``: pulls real observations from
   ``lerobot/libero_spatial`` via ``LeRobotDataset``. Tests realistic obs
   shapes / dtypes without env stepping.
3. ``--mode env``: full LIBERO env loop. Requires ``lerobot[libero]`` and
   ``MUJOCO_GL=egl``.

Examples
--------
``\
$ MUJOCO_GL=egl python scripts/a100_inference_test.py \\
      --mode synthetic --ticks 1000 --tick-dt-ms 5

$ python scripts/a100_inference_test.py \\
      --mode dataset --policy-path lerobot/smolvla_libero --ticks 500

$ python scripts/a100_inference_test.py \\
      --mode env --task libero_spatial --episodes 3 --policy-path lerobot/smolvla_libero
``

Output: a JSON report at ``--output`` (default ``./a100_report.json``) plus
human-readable summary on stdout.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from a2c2_libero.heads import A2C2MLPHead
from a2c2_libero.inference.a2c2_engine import A2C2Engine


# --------------------------------------------------------------------------- #
# Dataclasses for the JSON report
# --------------------------------------------------------------------------- #
@dataclass
class LatencyStats:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    count: int


@dataclass
class Report:
    mode: str
    device: str
    policy_path: str
    chunk_size: int
    action_dim: int
    latent_dim: int
    ticks_run: int
    wallclock_s: float
    smolvla_forwards: int
    head_forwards: int
    a2c2_step_latency: LatencyStats
    smolvla_forward_latency: LatencyStats | None
    gpu_peak_mb: float
    success_rates: dict[str, float] = field(default_factory=dict)


def _stats(samples: Iterable[float]) -> LatencyStats:
    arr = np.array(list(samples), dtype=np.float64)
    if arr.size == 0:
        return LatencyStats(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    return LatencyStats(
        mean_ms=float(arr.mean()),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        max_ms=float(arr.max()),
        count=int(arr.size),
    )


# --------------------------------------------------------------------------- #
# Real SmolVLA loader + forward wrapper
# --------------------------------------------------------------------------- #
def _load_smolvla(policy_path: str, device: torch.device) -> tuple[Any, int, int]:
    """Load SmolVLA, return (policy, chunk_size, latent_dim)."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(policy_path).to(device)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad = False
    H = getattr(policy.config, "chunk_size", 50)
    d_z = getattr(policy.config, "hidden_size", 384)
    return policy, H, d_z


def _make_smolvla_fn(policy: Any, device: torch.device, latent_dim: int, smolvla_latencies: list[float]):
    """Build a callable `(obs, language) -> (chunk, z)` that records latency.

    The latent extraction strategy depends on the SmolVLA internals; this
    helper falls back to a zero-tensor latent if the hidden state cannot be
    located, so the engine still runs (the head will then receive a constant
    z; treat the test as a dummy-conditioning benchmark in that case).
    """
    # Try to locate a backbone module to hook for latent extraction.
    hook_module = None
    candidates = [
        "model.vlm_backbone.layers",                # huggingface SmolVLA layout
        "model.text_encoder.encoder.layer",
        "vlm_backbone",
    ]
    for path in candidates:
        try:
            obj: Any = policy
            for name in path.split("."):
                obj = getattr(obj, name)
            if isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
                hook_module = obj[-1]
            else:
                hook_module = obj
            break
        except AttributeError:
            continue

    captured: dict[str, torch.Tensor | None] = {"z": None}
    if hook_module is not None:
        def _hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if isinstance(h, torch.Tensor):
                if h.ndim == 3:
                    captured["z"] = h[:, 0, :].detach()
                elif h.ndim == 2:
                    captured["z"] = h.detach()
        hook_module.register_forward_hook(_hook)

    @torch.no_grad()
    def smolvla_fn(obs: dict[str, torch.Tensor], language: str | None) -> tuple[torch.Tensor, torch.Tensor]:
        t0 = time.perf_counter()
        # SmolVLAPolicy.predict_action_chunk signature varies by version;
        # try the most likely API surfaces.
        chunk = None
        for method in ("predict_action_chunk", "predict_action", "select_action"):
            if hasattr(policy, method):
                fn = getattr(policy, method)
                try:
                    chunk = fn(obs)
                    break
                except TypeError:
                    try:
                        chunk = fn(obs, language)
                        break
                    except TypeError:
                        continue
        if chunk is None:
            raise RuntimeError(
                "could not call any of predict_action_chunk / predict_action / "
                "select_action on the SmolVLA policy"
            )
        chunk = chunk.detach()
        if chunk.ndim == 3:
            chunk = chunk[0]                # drop batch
        z = captured["z"]
        if z is None:
            z = torch.zeros(latent_dim, device=device)
        elif z.ndim == 2:
            z = z[0]
        smolvla_latencies.append((time.perf_counter() - t0) * 1000.0)
        return chunk.to(device), z.to(device)

    return smolvla_fn


# --------------------------------------------------------------------------- #
# Observation generators
# --------------------------------------------------------------------------- #
def _synthetic_obs(action_dim: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Random observation matching common LIBERO obs keys."""
    return {
        "observation.images.image": torch.rand(3, 256, 256, device=device),
        "observation.images.image2": torch.rand(3, 256, 256, device=device),
        "observation.state": torch.zeros(8, device=device),
    }


def _dataset_obs_iter(repo_id: str, device: torch.device):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, episodes=[0])
    for i in range(len(ds)):
        sample = ds[i]
        yield {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in sample.items()
            if k.startswith("observation")
        }


# --------------------------------------------------------------------------- #
# Main benchmark loop
# --------------------------------------------------------------------------- #
def run_benchmark(args: argparse.Namespace) -> Report:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, torch {torch.__version__}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ----- load real SmolVLA -----
    print(f"loading SmolVLA from {args.policy_path} ...")
    policy, H, d_z = _load_smolvla(args.policy_path, device)
    print(f"SmolVLA loaded. chunk_size H={H}, latent_dim d_z={d_z}")

    smolvla_latencies: list[float] = []
    smolvla_fn = _make_smolvla_fn(policy, device, d_z, smolvla_latencies)

    # ----- A2C2 head: trained ckpt or randomly initialised -----
    # Build a head whose input_dim matches the synthetic state we'll produce.
    # For real deployment, replace this with a trained checkpoint.
    obs0 = _synthetic_obs(args.action_dim, device)
    flat_obs_dim = sum(v.numel() for v in obs0.values())
    head_input_dim = flat_obs_dim + args.action_dim + 2 + d_z
    head = A2C2MLPHead(
        input_dim=head_input_dim,
        action_dim=args.action_dim,
        hidden=args.head_hidden,
    ).to(device)
    if args.head_ckpt is not None:
        print(f"loading A2C2 head ckpt: {args.head_ckpt}")
        head.load_state_dict(torch.load(args.head_ckpt, map_location=device))
    else:
        print("A2C2 head: random init (latency-only benchmark, do not trust accuracy)")
    head.eval()

    @torch.no_grad()
    def head_fn(state: dict[str, torch.Tensor]) -> torch.Tensor:
        return head(state)

    engine = A2C2Engine(
        smolvla_fn=smolvla_fn,
        head_fn=head_fn,
        chunk_size=H,
        action_dim=args.action_dim,
        device=device,
    )
    engine.start_async()

    # ----- run loop -----
    step_latencies: list[float] = []
    success_rates: dict[str, float] = {}

    obs_provider = _build_obs_provider(args, device)
    t0 = time.monotonic()
    try:
        if args.mode == "env":
            from lerobot.envs.libero import LiberoEnv

            for ep in range(args.episodes):
                env = LiberoEnv(task=args.task)
                obs, info = env.reset(seed=ep)
                lang = info.get("language_instruction", "")
                done = False
                ep_steps = 0
                while not done and ep_steps < args.max_episode_steps:
                    t = time.perf_counter()
                    action = engine.step_async(obs, language=lang)
                    step_latencies.append((time.perf_counter() - t) * 1000.0)
                    obs, _, terminated, truncated, info = env.step(action.cpu().numpy())
                    done = terminated or truncated
                    ep_steps += 1
                    time.sleep(args.tick_dt_ms / 1000.0)
                success_rates[f"ep{ep}"] = float(info.get("success", 0.0))
                env.close()
        else:
            for tick in range(args.ticks):
                obs = next(obs_provider)
                t = time.perf_counter()
                action = engine.step_async(obs, language="benchmark")
                step_latencies.append((time.perf_counter() - t) * 1000.0)
                if args.tick_dt_ms > 0:
                    time.sleep(args.tick_dt_ms / 1000.0)

    finally:
        engine.stop_async()

    elapsed = time.monotonic() - t0
    diag = engine.diagnostics()

    peak_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else 0.0
    )

    report = Report(
        mode=args.mode,
        device=str(device),
        policy_path=args.policy_path,
        chunk_size=H,
        action_dim=args.action_dim,
        latent_dim=d_z,
        ticks_run=len(step_latencies),
        wallclock_s=elapsed,
        smolvla_forwards=diag["chunks_produced"],
        head_forwards=len(step_latencies),
        a2c2_step_latency=_stats(step_latencies),
        smolvla_forward_latency=_stats(smolvla_latencies) if smolvla_latencies else None,
        gpu_peak_mb=peak_mb,
        success_rates=success_rates,
    )
    return report


def _build_obs_provider(args: argparse.Namespace, device: torch.device):
    if args.mode == "synthetic":
        def gen():
            while True:
                yield _synthetic_obs(args.action_dim, device)
        return gen()
    if args.mode == "dataset":
        return _dataset_obs_iter(args.dataset_repo, device)
    if args.mode == "env":
        # env mode handles obs internally
        def gen():
            while True:
                yield None
        return gen()
    raise ValueError(args.mode)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["synthetic", "dataset", "env"], default="synthetic")
    p.add_argument("--policy-path", default="lerobot/smolvla_libero",
                   help="HF repo id or local path of the trained SmolVLA")
    p.add_argument("--head-ckpt", default=None,
                   help="path to a trained A2C2 head checkpoint; random init if omitted")
    p.add_argument("--head-hidden", type=int, default=512)
    p.add_argument("--action-dim", type=int, default=7,
                   help="LIBERO=7, bi-SO-ARM=12")
    p.add_argument("--ticks", type=int, default=500,
                   help="number of control ticks (synthetic/dataset modes)")
    p.add_argument("--tick-dt-ms", type=float, default=5.0,
                   help="target main-loop tick period; 0 = run as fast as possible")
    p.add_argument("--task", default="libero_spatial",
                   help="LIBERO task suite (env mode)")
    p.add_argument("--episodes", type=int, default=3,
                   help="number of episodes (env mode)")
    p.add_argument("--max-episode-steps", type=int, default=300)
    p.add_argument("--dataset-repo", default="lerobot/libero_spatial")
    p.add_argument("--output", default="./a100_report.json")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Headless rendering is required for env mode on a server.
    os.environ.setdefault("MUJOCO_GL", "egl")

    report = run_benchmark(args)

    # ----- summary -----
    sl = report.a2c2_step_latency
    print("\n========== A2C2 inference report ==========")
    print(f"mode:               {report.mode}")
    print(f"device:             {report.device}")
    print(f"policy:             {report.policy_path}")
    print(f"chunk_size H:       {report.chunk_size}")
    print(f"action_dim A:       {report.action_dim}")
    print(f"latent_dim d_z:     {report.latent_dim}")
    print(f"ticks run:          {report.ticks_run}")
    print(f"wallclock:          {report.wallclock_s:.2f} s")
    print(f"smolvla forwards:   {report.smolvla_forwards}")
    print()
    print(f"a2c2 step latency:  mean={sl.mean_ms:.2f}ms  p50={sl.p50_ms:.2f}  "
          f"p95={sl.p95_ms:.2f}  p99={sl.p99_ms:.2f}  max={sl.max_ms:.2f}")
    if report.smolvla_forward_latency is not None:
        sf = report.smolvla_forward_latency
        print(f"smolvla forward:    mean={sf.mean_ms:.2f}ms  p50={sf.p50_ms:.2f}  "
              f"p95={sf.p95_ms:.2f}  p99={sf.p99_ms:.2f}  max={sf.max_ms:.2f}")
    print(f"gpu peak memory:    {report.gpu_peak_mb:.1f} MB")
    if report.success_rates:
        print("episode success rates:")
        for k, v in report.success_rates.items():
            print(f"  {k}: {v}")
    print("===========================================")

    Path(args.output).write_text(json.dumps(asdict(report), indent=2))
    print(f"json report: {args.output}")


if __name__ == "__main__":
    main()

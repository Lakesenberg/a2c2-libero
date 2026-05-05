"""Standalone demo: run async A2C2 inference with mock policies and print
the per-tick trace. No GPU required, no LIBERO required.

Useful for understanding the async timing or debugging the engine before
plugging in a real SmolVLA / A2C2 head.

Usage:
    python scripts/run_inference_demo.py [--sync] [--ticks 200]
"""
from __future__ import annotations

import argparse
import time

import torch

from a2c2_libero.inference.a2c2_engine import A2C2Engine
from a2c2_libero.inference.mocks import MockA2C2Head, MockSmolVLA


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=200)
    p.add_argument("--tick-dt-ms", type=float, default=5.0)
    p.add_argument("--smolvla-latency-ms", type=float, default=100.0)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--sync", action="store_true", help="use synchronous step")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    smolvla = MockSmolVLA(
        chunk_size=args.chunk_size,
        action_dim=7,
        latent_dim=384,
        forward_latency_s=args.smolvla_latency_ms / 1000,
    )
    head = MockA2C2Head(action_dim=7, delta_scale=0.01)

    engine = A2C2Engine(
        smolvla_fn=smolvla,
        head_fn=head,
        chunk_size=args.chunk_size,
        action_dim=7,
    )

    if not args.sync:
        engine.start_async()

    obs = {"image": torch.zeros(3, 64, 64), "state": torch.zeros(8)}
    try:
        t0 = time.monotonic()
        for tick in range(args.ticks):
            if args.sync:
                a = engine.step_sync(obs, language="pick up the cup")
            else:
                a = engine.step_async(obs, language="pick up the cup")

            if not args.quiet and tick % 10 == 0:
                d = engine.diagnostics()
                print(
                    f"tick={tick:4d}  chunk_id={d['chunk_id']:2d}  k={d['k']:3d}  "
                    f"chunks_produced={d['chunks_produced']:2d}  "
                    f"||a||={a.norm().item():.4f}"
                )

            time.sleep(args.tick_dt_ms / 1000)

        elapsed = time.monotonic() - t0
        print(
            f"\ndone: {args.ticks} ticks in {elapsed:.2f} s "
            f"(target {args.ticks * args.tick_dt_ms / 1000:.2f} s)"
        )
        print(f"smolvla forwards: {smolvla.call_count}")
        print(f"head forwards:    {head.call_count}")
    finally:
        if not args.sync:
            engine.stop_async()


if __name__ == "__main__":
    main()

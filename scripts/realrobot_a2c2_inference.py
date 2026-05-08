"""Real-robot A2C2 inference on bi-SO-ARM (or any LeRobot robot).

Bypasses lerobot-rollout / inference factory. Loads SmolVLA + the residual
transformer A2C2 head directly, drives the robot at the highest tick rate
it supports, and records to a LeRobot dataset.

The script feeds the residual head the same fields used at training time
in ``src/lerobot/scripts/train_residual_transformer.py``:

* ``action``             - single-step base action to execute this tick
* ``base_action_chunk``  - the FULL chunk produced by SmolVLA at chunk start
* ``time_feature``       - sin/cos of the chunk index (k / H)
* ``vlm_hidden``         - cached VLM latent at chunk start
* ``observation.*``      - current observation

Without ``base_action_chunk`` and ``time_feature`` the residual transformer
falls back to base-action-only mode and refinement quality drops; see
upstream README at https://github.com/k1000dai/a2c2-libero .

Usage:
    python scripts/realrobot_a2c2_inference.py \
        --smolvla-path Lakesenberg/smolvla_libero \
        --head-ckpt   outputs/a2c2_libero_10/checkpoint.pt \
        --robot-id    my_so_follower \
        --episodes    20 \
        --task        "pick the cup" \
        --dataset-repo Lakesenberg/realrobot_a2c2_eval

Press 'q' (in the cv2 window) to abort current episode.
Ctrl-C from terminal stops everything safely.
"""
from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import numpy as np
import torch

from a2c2_libero.heads import A2C2MLPHead
from a2c2_libero.inference.a2c2_engine import A2C2Engine
from a2c2_libero.inference.utils import LatentHook


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smolvla-path", required=True,
                   help="HF repo or local path of trained SmolVLA")
    p.add_argument("--head-ckpt", required=True,
                   help="local path to trained A2C2 head .pt")
    p.add_argument("--head-input-dim", type=int, default=None,
                   help="inferred from first build_state if omitted")
    p.add_argument("--robot-type", default="bi_so_follower")
    p.add_argument("--robot-id", required=True)
    p.add_argument("--cameras-config", default=None,
                   help="JSON string for cameras, or use robot defaults")
    p.add_argument("--action-dim", type=int, default=12,
                   help="bi-SO-ARM = 12 (6+6); single arm = 6 or 7")
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-episode-steps", type=int, default=600)
    p.add_argument("--tick-dt-ms", type=float, default=5.0)
    p.add_argument("--task", default="pick up the object")
    p.add_argument("--dataset-repo", default=None,
                   help="if given, record episodes to this LeRobot dataset")
    p.add_argument("--home-on-start", action="store_true")
    p.add_argument("--no-record", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- #
def _import_robot_api():
    """Locate (make_robot_from_config, RobotConfig) across lerobot layouts."""
    candidates = [
        ("lerobot.robots", "make_robot_from_config", "lerobot.robots.configs", "RobotConfig"),
        ("lerobot.robots", "make_robot_from_config", "lerobot.robots.config", "RobotConfig"),
        ("lerobot.robots.utils", "make_robot_from_config", "lerobot.robots.configs", "RobotConfig"),
        ("lerobot.common.robots", "make_robot_from_config", "lerobot.common.robots.configs", "RobotConfig"),
        ("lerobot.common.robots", "make_robot_from_config", "lerobot.common.robots.config", "RobotConfig"),
        ("lerobot.robots", "make_robot", "lerobot.robots.configs", "RobotConfig"),
        ("lerobot.common.robots", "make_robot", "lerobot.common.robots.config", "RobotConfig"),
    ]
    last_error = None
    for builder_mod, builder_name, cfg_mod, cfg_name in candidates:
        try:
            mb = __import__(builder_mod, fromlist=[builder_name])
            mc = __import__(cfg_mod, fromlist=[cfg_name])
            return getattr(mb, builder_name), getattr(mc, cfg_name)
        except (ImportError, AttributeError) as e:
            last_error = e
    raise ImportError(
        "Could not locate lerobot robot factory. Tried:\n  "
        + "\n  ".join(f"{m}.{n} + {c}.{r}" for m, n, c, r in candidates)
        + f"\nLast error: {last_error}"
    )


def _build_robot_direct(args):
    """Import the robot class + config directly from `lerobot.robots.<robot_type>`.

    Matches the lerobot 0.5+ layout where each robot ships its own
    subpackage exporting a robot class plus a config dataclass. Detection
    is case-insensitive to handle both ``SOFollower`` (acronym style) and
    ``SoFollower`` (Pascal style).
    """
    import importlib

    sub = args.robot_type
    try:
        mod = importlib.import_module(f"lerobot.robots.{sub}")
    except ImportError as e:
        raise ImportError(
            f"lerobot.robots.{sub} not importable. Make sure the robot "
            f"package is installed and `--robot-type` matches the "
            f"submodule name."
        ) from e

    # Build a normalized form of the robot-type string for matching.
    norm = sub.replace("_", "").lower()      # e.g. "sofollower"
    robot_cls = None
    config_cls = None
    candidates_by_kind: dict[str, list[type]] = {"robot": [], "config": []}
    for name in dir(mod):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name)
        if not isinstance(obj, type):
            continue
        nname = name.replace("_", "").lower()
        if "config" in nname and norm in nname:
            candidates_by_kind["config"].append(obj)
        elif norm in nname and "config" not in nname:
            candidates_by_kind["robot"].append(obj)

    # Prefer the most specific match (longest class name) to avoid e.g.
    # picking up a base class.
    if candidates_by_kind["config"]:
        config_cls = max(candidates_by_kind["config"], key=lambda c: len(c.__name__))
    if candidates_by_kind["robot"]:
        robot_cls = max(candidates_by_kind["robot"], key=lambda c: len(c.__name__))

    if robot_cls is None or config_cls is None:
        raise ImportError(
            f"Could not auto-detect robot class / config in lerobot.robots.{sub}. "
            f"Exports: {[n for n in dir(mod) if not n.startswith('_')]}"
        )

    cfg = config_cls(id=args.robot_id)
    if args.cameras_config:
        import json
        cfg.cameras = json.loads(args.cameras_config)
    print(f"[build_robot] direct import: {robot_cls.__name__}({config_cls.__name__})")
    return robot_cls(cfg)


def build_robot(args):
    """Construct a LeRobot robot wrapper, layout-agnostic.

    Strategy:
      1. Try direct per-robot import first (lerobot.robots.<robot_type>).
         This is the most robust path on lerobot 0.5+ where the abstract
         RobotConfig no longer accepts `type=` as a kwarg (it's a draccus
         discriminator field on the union of subclasses).
      2. Fall back to the registry-style ``make_robot_from_config`` only
         if direct import fails.
    """
    # 1. Direct path — works on 0.5.1, 0.4.x and the fork.
    try:
        return _build_robot_direct(args)
    except ImportError as direct_err:
        print(f"[build_robot] direct import failed ({direct_err}); "
              "trying registry path.")

    # 2. Registry fallback (legacy / draccus-aware).
    make_robot_from_config, RobotConfig = _import_robot_api()
    cfg = None
    # 2a. draccus-style decode (lerobot 0.5+).
    try:
        import draccus
        cfg_dict = {"type": args.robot_type, "id": args.robot_id}
        cfg = draccus.decode(cfg_dict, RobotConfig)
    except Exception:
        pass
    # 2b. classmethod from_kwargs (older fork).
    if cfg is None and hasattr(RobotConfig, "from_kwargs"):
        try:
            cfg = RobotConfig.from_kwargs(type=args.robot_type, id=args.robot_id)
        except Exception:
            pass
    # 2c. Last-ditch: positional / kwargs construction (legacy).
    if cfg is None:
        try:
            cfg = RobotConfig(type=args.robot_type, id=args.robot_id)
        except TypeError:
            cfg = RobotConfig(id=args.robot_id)

    if args.cameras_config:
        import json
        cfg.cameras = json.loads(args.cameras_config)
    return make_robot_from_config(cfg)


def _import_smolvla():
    """Locate SmolVLAPolicy across lerobot layouts."""
    for mod in ("lerobot.policies.smolvla.modeling_smolvla",
                "lerobot.common.policies.smolvla.modeling_smolvla"):
        try:
            m = __import__(mod, fromlist=["SmolVLAPolicy"])
            return m.SmolVLAPolicy
        except (ImportError, AttributeError):
            continue
    raise ImportError(
        "Could not locate SmolVLAPolicy. Tried "
        "lerobot.policies.smolvla.modeling_smolvla and "
        "lerobot.common.policies.smolvla.modeling_smolvla."
    )


def load_smolvla(path: str, device: torch.device):
    SmolVLAPolicy = _import_smolvla()
    p = SmolVLAPolicy.from_pretrained(path).to(device).eval()
    for prm in p.parameters():
        prm.requires_grad = False
    return p


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- robot -----
    robot = build_robot(args)
    print(f"[robot] {args.robot_type} ({args.robot_id})")
    robot.connect()
    if not robot.is_calibrated:
        raise RuntimeError(
            f"robot {args.robot_id} not calibrated. Run: "
            f"lerobot-calibrate --robot.type={args.robot_type} --robot.id={args.robot_id}"
        )
    if args.home_on_start:
        print("[robot] homing ...")
        robot.home()

    # ----- SmolVLA + latent hook -----
    print(f"[smolvla] loading {args.smolvla_path}")
    smolvla = load_smolvla(args.smolvla_path, device)
    # Hook depending on backbone layout; adjust for your SmolVLA version.
    backbone_last = (
        getattr(getattr(smolvla, "model", smolvla), "vlm_backbone", None)
    )
    if backbone_last is None:
        raise RuntimeError(
            "could not locate SmolVLA backbone for latent hook; please patch "
            "scripts/realrobot_a2c2_inference.py to point at your version"
        )
    if hasattr(backbone_last, "layers") and len(backbone_last.layers) > 0:
        hook = LatentHook(backbone_last.layers[-1])
    else:
        hook = LatentHook(backbone_last)

    @torch.no_grad()
    def smolvla_fn(obs, language):
        chunk = smolvla.predict_action_chunk(obs)
        z = hook.latest
        if z is None:
            raise RuntimeError("LatentHook produced no latent; check hook target")
        return chunk[0].to(device), z[0].to(device)

    # ----- A2C2 head -----
    print(f"[a2c2 ] loading {args.head_ckpt}")
    head_state = torch.load(args.head_ckpt, map_location=device)
    if isinstance(head_state, dict) and "input_dim" in head_state:
        in_dim = head_state["input_dim"]
        head_state = head_state["state_dict"]
    elif args.head_input_dim is not None:
        in_dim = args.head_input_dim
    else:
        # Fallback: probe from a synthetic state once
        from a2c2_libero.inference.utils import build_state, flatten_state_for_mlp
        sample = robot.get_observation()
        s = build_state(
            obs={k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in sample.items()
                 if isinstance(v, (np.ndarray, torch.Tensor))},
            a_base=torch.zeros(args.action_dim, device=device),
            tau_k=torch.zeros(2, device=device),
            z=torch.zeros(384, device=device),
        )
        in_dim = flatten_state_for_mlp(s).shape[-1]
        print(f"[a2c2 ] inferred head input_dim = {in_dim}")

    head = A2C2MLPHead(input_dim=in_dim, action_dim=args.action_dim).to(device).eval()
    head.load_state_dict(head_state)

    @torch.no_grad()
    def head_fn(state):
        return head(state)

    # ----- A2C2 engine -----
    engine = A2C2Engine(
        smolvla_fn=smolvla_fn,
        head_fn=head_fn,
        chunk_size=args.chunk_size,
        action_dim=args.action_dim,
        device=device,
    )
    engine.start_async()

    # ----- recording -----
    dataset = None
    if args.dataset_repo and not args.no_record:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset.create(
            repo_id=args.dataset_repo,
            features=robot.observation_features | robot.action_features
                     | {"intervention": {"dtype": "bool", "shape": (1,)}},
            fps=int(1000 / args.tick_dt_ms),
        )
        print(f"[record] new dataset: {args.dataset_repo}")

    # ----- safety: ctrl-c safe stop -----
    stop_signal = {"flag": False}
    def _on_sigint(signum, frame):
        print("\n[abort] ctrl-c, stopping after current episode")
        stop_signal["flag"] = True
    signal.signal(signal.SIGINT, _on_sigint)

    # ====================================================================== #
    # main loop
    # ====================================================================== #
    successes = 0
    try:
        for ep in range(args.episodes):
            print(f"\n========== episode {ep+1}/{args.episodes} ==========")
            engine.reset(); engine.start_async()
            input("place objects, press ENTER to start (or ctrl-c to abort)...")

            ep_start = time.monotonic()
            tick_dt = args.tick_dt_ms / 1000.0
            for tick in range(args.max_episode_steps):
                t0 = time.perf_counter()
                obs = robot.get_observation()
                obs_t = {k: torch.as_tensor(v, device=device)
                         for k, v in obs.items()
                         if isinstance(v, (np.ndarray, torch.Tensor))}
                action = engine.step_async(obs_t, language=args.task)
                action_np = action.cpu().numpy().astype(np.float32)
                robot.send_action(action_np)

                if dataset is not None:
                    frame = {**obs, "action": action_np,
                             "intervention": np.array([False])}
                    dataset.add_frame(frame, task=args.task)

                # pace to target tick rate
                elapsed = time.perf_counter() - t0
                if elapsed < tick_dt:
                    time.sleep(tick_dt - elapsed)

                if stop_signal["flag"]:
                    break

            ep_dur = time.monotonic() - ep_start
            ans = input(f"episode took {ep_dur:.1f}s. success? [y/N] ").strip().lower()
            ok = ans == "y"
            successes += int(ok)
            print(f"[ep {ep+1}] success={ok}  ({successes}/{ep+1})")

            if dataset is not None:
                dataset.save_episode()

            if stop_signal["flag"]:
                break

    finally:
        engine.stop_async()
        try:
            robot.home()
        except Exception:
            pass
        robot.disconnect()
        print(f"\n=== final: {successes}/{ep+1 if 'ep' in dir() else 0} episodes succeeded ===")


if __name__ == "__main__":
    main()

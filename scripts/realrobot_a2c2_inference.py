"""Real-robot A2C2 inference following the lerobot + k1000dai/a2c2-libero
canonical pattern.

Important: this module deliberately does NOT use
``from __future__ import annotations`` so that forward-referenced
dataclass field types resolve to real classes at runtime — draccus needs
the concrete `RobotConfig` to walk the discriminated union.


CLI structure mirrors `lerobot-record` / `lerobot-rollout`: the top-level
@parser.wrap()'d dataclass embeds a `robot: RobotConfig` field, so all
`--robot.*` (including `--robot.cameras='{...}'`) works automatically via
draccus, identical to upstream lerobot. No ad-hoc port / camera flags.

Inference loop matches `eval_libero/evaluation_libero.py` (lines 230-310):

    base_policy.predict_action_chunk(observation)         -> (1, T, A)
    vlm_hidden  = base_policy.vlm_hidden                  # cached on policy
    observation["action"]            = base_chunk[:, 0]   # current step
    observation["base_action_chunk"] = base_chunk         # full chunk
    observation["time_feature"]      = sin/cos of (k mod T)
    observation["vlm_hidden"]        = vlm_hidden[k]
    delta = residual_policy.predict_action_chunk(observation)[:, 0]
    a_exec = base_chunk[:, 0] + delta

Example usage (single SO-100 + one camera + one wrist camera):

    python scripts/realrobot_a2c2_inference.py \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM0 \\
        --robot.id=my_arm \\
        --robot.cameras='{image: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist_image: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}' \\
        --base-policy-path=outputs/smolvla_v21 \\
        --residual-policy-path=outputs/a2c2_head_v21 \\
        --task='pick up the cup' \\
        --episodes=3 \\
        --no-record

Bi-SO-ARM (dual arm): swap `--robot.type=bi_so_follower` and supply
`--robot.left_arm_port=/dev/ttyACM0 --robot.right_arm_port=/dev/ttyACM1`
plus the corresponding cameras.
"""
# NOTE: do NOT add `from __future__ import annotations` here.
import math
import signal
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Resolve lerobot at module import time so the dataclass annotation below
# can reference the real RobotConfig class (not a string).
#
# IMPORTANT: also force-import every robot subpackage so each robot's
# `@RobotConfig.register_subclass("...")` decorator runs and the
# discriminator names (so100_follower / so101_follower / bi_so_follower
# / koch_follower / ...) become valid `--robot.type=` choices. Without
# this you'd see "invalid choice: 'so101_follower'" because the subpackage
# directory exists on disk but was never imported, so its decorator
# didn't register the type with draccus.
# --------------------------------------------------------------------------- #
import lerobot.robots as _lr_robots
import pkgutil as _pkgutil
for _finder, _name, _ in _pkgutil.iter_modules(_lr_robots.__path__):
    try:
        __import__(f"lerobot.robots.{_name}")
    except Exception:
        # Subpackages with optional deps (e.g. realsense, zmq) may fail to
        # import; that's fine — only the robot you actually use needs to
        # load successfully.
        pass

from lerobot.robots import RobotConfig, make_robot_from_config


def _home_robot(robot, verbose=True):
    """Best-effort 'go to home pose' across lerobot API versions.

    lerobot 0.5+ dropped the `home()` method on the per-robot classes
    in favor of the more general `lerobot-calibrate` flow + sending an
    explicit zero-pose action. Try a few common method names; if none
    exist, print a warning and let the user pre-position the robot.
    """
    candidates = ("home", "go_home", "move_to_home_position", "reset",
                  "return_to_home", "move_home")
    for name in candidates:
        fn = getattr(robot, name, None)
        if callable(fn):
            try:
                fn()
                if verbose:
                    print(f"[robot] homed via {name}()")
                return True
            except Exception as e:
                if verbose:
                    print(f"[robot] {name}() raised {type(e).__name__}: {e}; "
                          f"trying next candidate.")
    if verbose:
        print("[robot] no home() / go_home() / move_to_home_position() etc. "
              "exposed by this robot class; please pre-position the arm "
              "manually before pressing ENTER.")
    return False


def _resolve_ckpt_path(path, kind):
    """Locate a HuggingFace-style checkpoint directory.

    Accepts a local path or HF repo_id. For local paths, also auto-descend
    into a ``pretrained_model/`` subdirectory if it exists (lerobot saves
    checkpoints to ``<output_dir>/pretrained_model/`` by default).

    Raises a clear FileNotFoundError listing what was tried, instead of
    letting HuggingFace fall through to its `Repo id must be in the form
    'name/repo'` opaque error.
    """
    import os
    candidates = [path]
    # If the user pointed at the parent dir, auto-descend
    if not path.endswith("pretrained_model"):
        candidates.append(os.path.join(path, "pretrained_model"))
    # Or `outputs/<run>/checkpoints/<step>/pretrained_model`
    candidates.append(os.path.join(path, "checkpoints", "last", "pretrained_model"))

    expected_files = ("config.json",)
    for c in candidates:
        if os.path.isdir(c) and all(os.path.exists(os.path.join(c, f))
                                    for f in expected_files):
            return c

    # Maybe it's a HF Hub repo_id (must be 'namespace/name'-shaped)
    if "/" in path and not os.path.isabs(path) and not os.path.exists(path):
        # let HF resolve it remotely
        return path

    raise FileNotFoundError(
        f"[{kind}] could not find a valid checkpoint directory.\n"
        f"  --{kind.replace('_', '-')}={path!r}\n"
        f"  tried: {candidates}\n"
        f"  Expected a directory that contains at least 'config.json' "
        f"(and 'model.safetensors' / 'pytorch_model.bin' for weights).\n"
        f"  Common causes:\n"
        f"    - typo in the path\n"
        f"    - you pointed at the parent of pretrained_model/ "
        f"(this script auto-descends but only one level)\n"
        f"    - ckpt not actually scp'd to the 4090 yet — run\n"
        f"        ls -la {path}\n"
        f"      and confirm there's a config.json + model.safetensors"
    )


def _load_smolvla(path, device):
    resolved = _resolve_ckpt_path(path, "base_policy_path")
    print(f"[smolvla] resolved ckpt dir: {resolved}")
    for mod in (
        "lerobot.policies.smolvla.modeling_smolvla",
        "lerobot.common.policies.smolvla.modeling_smolvla",
    ):
        try:
            SmolVLAPolicy = __import__(mod, fromlist=["SmolVLAPolicy"]).SmolVLAPolicy
            return SmolVLAPolicy.from_pretrained(resolved).to(device).eval()
        except (ImportError, AttributeError):
            continue
    raise ImportError("Could not locate SmolVLAPolicy")


def _load_residual_policy(path, device):
    resolved = _resolve_ckpt_path(path, "residual_policy_path")
    print(f"[a2c2 ] resolved ckpt dir: {resolved}")
    for mod in (
        "lerobot.policies.residual_transformer.modeling_residual_transformer",
        "lerobot.common.policies.residual_transformer.modeling_residual_transformer",
    ):
        try:
            cls = __import__(mod, fromlist=["ResidualTransformerPolicy"])
            return cls.ResidualTransformerPolicy.from_pretrained(resolved).to(device).eval()
        except (ImportError, AttributeError):
            continue
    raise ImportError(
        "ResidualTransformerPolicy not found. The fork that adds this "
        "policy must be importable as lerobot.policies.residual_transformer "
        "(or lerobot.common.policies.residual_transformer)."
    )


# Note: parser-wrap resolution is inlined in `main()` below — both
# `lerobot.configs.parser.wrap` and a direct `draccus.parse` fallback
# are tried in order.


# --------------------------------------------------------------------------- #
# Top-level config (draccus dataclass)
# --------------------------------------------------------------------------- #
@dataclass
class A2C2InferenceConfig:
    # Required fields (no default) — must come first per dataclass rules.
    # `--robot.type=so100_follower --robot.port=/dev/ttyACM0
    # --robot.cameras='{...}' --robot.id=...` all flow through this field
    # via draccus, exactly like lerobot-record / lerobot-rollout.
    robot: RobotConfig
    base_policy_path: str

    # Optional fields (with defaults).
    residual_policy_path: Optional[str] = None
    chunk_size: int = 50
    action_dim: int = 6
    task: str = "do the task"
    episodes: int = 3
    max_episode_steps: int = 600
    tick_dt_ms: float = 5.0
    home_on_start: bool = True
    no_record: bool = True
    dataset_repo: Optional[str] = None
    device: str = "cuda"


# --------------------------------------------------------------------------- #
# A2C2 inference engine
# --------------------------------------------------------------------------- #
class A2C2Inferencer:
    """Per-tick A2C2 inference matching evaluation_libero.py:230-310.

    Notes:
      - SmolVLA stashes its VLM hidden state on the policy/instance after
        `predict_action_chunk(...)`. We read it back from
        `policy.vlm_hidden` (or `policy.model.vlm_hidden`) instead of using
        a forward hook — this is the exact approach upstream uses.
      - Chunks are predicted at the start of every chunk window (every
        `chunk_size` ticks). Within a chunk we only run the residual head.
    """

    def __init__(
        self,
        base_policy,
        residual_policy,
        chunk_size,
        action_dim,
        device,
    ):
        self.base = base_policy
        self.residual = residual_policy
        self.H = chunk_size
        self.A = action_dim
        self.device = device
        self.reset()

    def reset(self):
        self._chunk = None       # (T, A) on device
        self._vlm_hidden = None  # (T, ...) on device
        self._k = 0

    @torch.no_grad()
    def step(self, observation: dict, task: str) -> torch.Tensor:
        """Run one control tick. Returns (action_dim,) tensor on CPU."""
        if self._chunk is None or self._k >= self.H:
            self._refresh_chunk(observation, task)

        k = self._k
        base_step = self._chunk[k]                  # (A,)
        observation = dict(observation)             # don't mutate caller's
        observation["task"] = task
        observation["action"] = base_step.unsqueeze(0)                # (1, A)
        observation["base_action_chunk"] = self._chunk.unsqueeze(0)   # (1, T, A)

        # Sin/cos chunk-index encoding (line 285-289 of evaluation_libero.py)
        phase = 2.0 * math.pi * (k % self.H) / max(self.H - 1, 1)
        observation["time_feature"] = torch.tensor(
            [[math.sin(phase), math.cos(phase)]],
            dtype=torch.float32, device=self.device,
        )

        if self._vlm_hidden is not None:
            entry = self._vlm_hidden[k] if self._vlm_hidden.ndim >= 2 else self._vlm_hidden
            observation["vlm_hidden"] = entry.unsqueeze(0).to(self.device)

        if self.residual is None:
            a_exec = base_step
        else:
            corrected_chunk = self.residual.predict_action_chunk(observation)
            corrected_chunk = corrected_chunk.squeeze(0).cpu()        # (T, A)
            # Residual head outputs the corrected action directly (matches
            # a2c2-libero evaluation: `updated_action = ...[0]`); no need
            # to add it back to base_step.
            a_exec = corrected_chunk[0].to(self.device)

        self._k += 1
        return a_exec.cpu()

    def _refresh_chunk(self, observation: dict, task: str) -> None:
        observation = dict(observation)
        observation["task"] = task
        chunk = self.base.predict_action_chunk(observation)            # (1, T, A)
        self._chunk = chunk.squeeze(0).to(self.device)

        # Pull the cached VLM hidden state (k1000dai pattern, line 244-247).
        vh = getattr(self.base, "vlm_hidden", None)
        if vh is None and hasattr(self.base, "model"):
            vh = getattr(self.base.model, "vlm_hidden", None)
        if vh is not None:
            self._vlm_hidden = vh.detach()
        self._k = 0


# --------------------------------------------------------------------------- #
# Episode loop
# --------------------------------------------------------------------------- #
def run_episode(robot, inferencer: A2C2Inferencer, cfg: A2C2InferenceConfig,
                episode_idx: int) -> dict:
    """Drive one episode end-to-end. Returns metrics dict."""
    inferencer.reset()
    obs = robot.get_observation()
    obs_t = {k: torch.as_tensor(v).to(cfg.device) for k, v in obs.items()
             if isinstance(v, (np.ndarray, torch.Tensor))}

    tick_dt = cfg.tick_dt_ms / 1000.0
    latencies: list[float] = []
    for tick in range(cfg.max_episode_steps):
        t0 = time.perf_counter()
        action = inferencer.step(obs_t, cfg.task)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        action_np = action.cpu().numpy().astype(np.float32)
        robot.send_action(action_np)

        if tick_dt > 0:
            sleep_for = tick_dt - (time.perf_counter() - t0)
            if sleep_for > 0:
                time.sleep(sleep_for)

        # Refresh observation (re-read sensors)
        obs = robot.get_observation()
        obs_t = {k: torch.as_tensor(v).to(cfg.device) for k, v in obs.items()
                 if isinstance(v, (np.ndarray, torch.Tensor))}

    succ_str = input(
        f"[ep {episode_idx + 1}/{cfg.episodes}] success? [y/N] "
    ).strip().lower()
    success = succ_str.startswith("y")
    return {
        "episode": episode_idx,
        "success": success,
        "ticks": cfg.max_episode_steps,
        "tick_latency_ms": {
            "mean": float(np.mean(latencies)) if latencies else 0.0,
            "p99":  float(np.percentile(latencies, 99)) if latencies else 0.0,
        },
    }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def _main(cfg):
    if not cfg.base_policy_path:
        raise SystemExit("--base-policy-path is required")
    if cfg.robot is None:
        raise SystemExit(
            "--robot.* is required (e.g. --robot.type=so100_follower "
            "--robot.port=/dev/ttyACM0)"
        )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # --- robot ---
    print(f"[robot] type={getattr(cfg.robot, 'type', '?')}, "
          f"id={getattr(cfg.robot, 'id', '?')}")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    if not robot.is_calibrated:
        raise RuntimeError(
            f"Robot {cfg.robot.id!r} is not calibrated. Run "
            f"`lerobot-calibrate --robot.type={cfg.robot.type} "
            f"--robot.id={cfg.robot.id}` first."
        )
    if cfg.home_on_start:
        _home_robot(robot)

    # --- base policy (SmolVLA) ---
    print(f"[smolvla] loading {cfg.base_policy_path}")
    base = _load_smolvla(cfg.base_policy_path, device)
    for p in base.parameters():
        p.requires_grad = False

    # --- residual head ---
    residual = None
    if cfg.residual_policy_path:
        print(f"[a2c2 ] loading {cfg.residual_policy_path}")
        residual = _load_residual_policy(cfg.residual_policy_path, device)
        for p in residual.parameters():
            p.requires_grad = False

    inferencer = A2C2Inferencer(
        base_policy=base,
        residual_policy=residual,
        chunk_size=cfg.chunk_size,
        action_dim=cfg.action_dim,
        device=device,
    )

    # --- ctrl-c safety ---
    stop_signal = {"flag": False}
    def _on_sigint(*_):
        print("\n[abort] ctrl-c, finishing current episode then stopping.")
        stop_signal["flag"] = True
    signal.signal(signal.SIGINT, _on_sigint)

    # --- episode loop ---
    metrics = []
    try:
        for ep in range(cfg.episodes):
            print(f"\n========== episode {ep + 1}/{cfg.episodes} ==========")
            input("place objects, press ENTER to start...")
            result = run_episode(robot, inferencer, cfg, ep)
            metrics.append(result)
            print(f"  success={result['success']} "
                  f"latency p99={result['tick_latency_ms']['p99']:.2f} ms")
            if stop_signal["flag"]:
                break
    finally:
        try:
            _home_robot(robot, verbose=False)
        except Exception:
            pass
        robot.disconnect()

    n_succ = sum(int(m["success"]) for m in metrics)
    print(f"\n=== final: {n_succ}/{len(metrics)} episodes succeeded ===")


def main():
    """Entry point.

    Tries lerobot's `parser.wrap` first (matches lerobot-record /
    lerobot-rollout exactly); falls back to a plain `draccus.parse` if
    the wrapper isn't available in this lerobot version. Both paths end
    up calling `_main(cfg)` with a fully-populated A2C2InferenceConfig.
    """
    # Path 1: lerobot wrapper (preferred)
    try:
        from lerobot.configs.parser import wrap as parser_wrap

        @parser_wrap()
        def _wrapped(cfg: A2C2InferenceConfig):
            _main(cfg)
        _wrapped()
        return
    except ImportError:
        pass
    try:
        from lerobot.common.configs.parser import wrap as parser_wrap

        @parser_wrap()
        def _wrapped(cfg: A2C2InferenceConfig):
            _main(cfg)
        _wrapped()
        return
    except ImportError:
        pass

    # Path 2: direct draccus.parse (no wrapper)
    import draccus
    cfg = draccus.parse(config_class=A2C2InferenceConfig)
    _main(cfg)


if __name__ == "__main__":
    main()

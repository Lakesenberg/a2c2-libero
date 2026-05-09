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
import os
import signal
import time
from dataclasses import dataclass
from typing import Optional

# Force HuggingFace to operate offline by default — the inference machine
# (4090) typically can't reach huggingface.co directly, only an internal
# mirror or local cache. transformers/hub will still happily load
# anything already in ~/.cache/huggingface/hub. Override with
# `HF_HUB_OFFLINE=0` if you want online access for some reason.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# If a mirror is set globally it'll still be used when HF_HUB_OFFLINE=0.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Workaround for PyTorch's "Cannot copy out of meta tensor" bug.
#
# Some lerobot fork policies (notably SmolVLAPolicy) load weights via
# transformers' low_cpu_mem_usage=True path, which initializes parameters
# on the `meta` device, then later calls `policy.to(config.device)` to
# move to GPU. With recent torch (2.4+), `.to()` raises
# `NotImplementedError: Cannot copy out of meta tensor; no data!`
# because the meta tensors weren't actually populated by the state dict
# load (a known load_state_dict bug with meta-initialized models —
# warnings show as 'copying from a non-meta parameter ... no-op').
#
# Monkey-patch torch.nn.Module.to so that whenever it would crash on a
# meta tensor, it transparently falls back to `to_empty(device=...)`,
# which allocates real storage. The subsequent state-dict load then
# actually populates the params. Idempotent — only triggers on meta.
# --------------------------------------------------------------------------- #
_orig_module_to = torch.nn.Module.to


def _patched_module_to(self, *args, **kwargs):
    try:
        return _orig_module_to(self, *args, **kwargs)
    except NotImplementedError as e:
        if "meta tensor" not in str(e):
            raise
        # Fall back to to_empty for any module containing meta params.
        device = kwargs.get("device")
        if device is None and args:
            device = args[0]
        if device is None:
            raise
        print(f"[torch-patch] {type(self).__name__}.to() hit meta tensor; "
              f"falling back to to_empty(device={device}).")
        return self.to_empty(device=device)


torch.nn.Module.to = _patched_module_to


# --------------------------------------------------------------------------- #
# Resolve lerobot at module import time.
#
# We need RobotConfig + the SPECIFIC robot subpackage the user selected
# imported so the discriminator name appears in draccus's choices.
# Importing ALL robot subpackages is risky: some forks ship broken
# default factories on robots like `stretch3` that crash draccus's
# default-value introspection (e.g. `RealSenseCameraConfig(name=...)`
# where `name` is not a valid kwarg in the current fork).
#
# So: pre-scan sys.argv for `--robot.type=X` and only import that one.
# Fall back to importing the common SO/koch/openarm subpackages if the
# user didn't pass --robot.type=... (e.g. when they pass --help).
# --------------------------------------------------------------------------- #
import importlib as _importlib
import sys as _sys

_SAFE_ROBOTS = (
    "so_follower", "bi_so_follower",
    "koch_follower",
    "openarm_follower", "bi_openarm_follower",
    "lekiwi", "omx_follower",
)


def _import_robot_for_cli():
    # Look for --robot.type=<name> in argv.
    target = None
    for i, a in enumerate(_sys.argv):
        if a == "--robot.type" and i + 1 < len(_sys.argv):
            target = _sys.argv[i + 1]
            break
        if a.startswith("--robot.type="):
            target = a.split("=", 1)[1]
            break

    if target:
        try:
            _importlib.import_module(f"lerobot.robots.{target}")
            return [target]
        except Exception as e:
            print(f"[bootstrap] failed to import lerobot.robots.{target}: {e}",
                  file=_sys.stderr)
            return []

    # No --robot.type= given (e.g. --help). Best-effort import the safe
    # set of common robots, skipping any that crash on import.
    imported = []
    for name in _SAFE_ROBOTS:
        try:
            _importlib.import_module(f"lerobot.robots.{name}")
            imported.append(name)
        except Exception:
            pass
    return imported


_imported_robots = _import_robot_for_cli()


# Same problem for camera configs — `--robot.cameras='{front: {type:
# opencv, ...}}'` requires lerobot.cameras.opencv to have been imported
# so its `@CameraConfig.register_subclass("opencv")` ran. The package
# `__init__.py` doesn't auto-import its subpackages either. Skip
# RealSense / ZMQ to avoid optional deps blowing up; the user can opt
# into them by passing --robot.cameras='{...type: realsense...}' (would
# then need to add the import here).
_SAFE_CAMERAS = ("opencv",)
for _name in _SAFE_CAMERAS:
    try:
        _importlib.import_module(f"lerobot.cameras.{_name}")
    except Exception as e:
        print(f"[bootstrap] failed to import lerobot.cameras.{_name}: {e}",
              file=_sys.stderr)


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


def _override_dataset_in_config(ckpt_dir, dataset_repo_id, dataset_root):
    """If the policy's config.json points at a dataset that isn't reachable
    from this machine, rewrite it before from_pretrained() reads it.

    Also auto-fixes hard-coded absolute paths to HuggingFace VLM caches
    (e.g. /root/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-...)
    that were baked in during training on a different machine.

    Returns True if config.json was rewritten.
    """
    import json
    import os
    import re

    cfg_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(cfg_path):
        return False
    with open(cfg_path) as f:
        cfg = json.load(f)

    changed = False

    # 1. Override dataset_repo_id / dataset_root if user requested.
    if dataset_repo_id is not None:
        for k in ("dataset_repo_id", "dataset_id", "repo_id"):
            if k in cfg and cfg[k] != dataset_repo_id:
                print(f"[ckpt-fix] {cfg_path}: {k}: {cfg[k]!r} -> {dataset_repo_id!r}")
                cfg[k] = dataset_repo_id
                changed = True
    if dataset_root is not None and "dataset_root" in cfg and cfg["dataset_root"] != dataset_root:
        cfg["dataset_root"] = dataset_root
        changed = True

    # 2. Auto-rewrite baked-in HF cache absolute paths.
    # Matches: '/root/.cache/huggingface/hub/models--<owner>--<name>/snapshots/<hash>'
    # Replaces with: '<owner>/<name>'
    cache_pattern = re.compile(
        r"^/.*?/\.cache/huggingface/hub/models--([^/]+)--([^/]+)/snapshots/[a-f0-9]+/?.*$"
    )

    def _walk(d, path=""):
        nonlocal changed
        if isinstance(d, dict):
            for k, v in list(d.items()):
                if isinstance(v, str):
                    m = cache_pattern.match(v)
                    if m:
                        repo_id = f"{m.group(1)}/{m.group(2)}"
                        print(f"[ckpt-fix] {cfg_path}: {path}.{k}: "
                              f"absolute HF cache path -> {repo_id!r}")
                        d[k] = repo_id
                        changed = True
                elif isinstance(v, (dict, list)):
                    _walk(v, f"{path}.{k}")
        elif isinstance(d, list):
            for i, v in enumerate(d):
                _walk(v, f"{path}[{i}]")

    _walk(cfg)

    if changed:
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
    return changed


def _load_smolvla(path, device, dataset_repo_id=None, dataset_root=None):
    resolved = _resolve_ckpt_path(path, "base_policy_path")
    print(f"[smolvla] resolved ckpt dir: {resolved}")
    # Always run config patcher: fixes baked-in absolute HF cache paths
    # left over from training on a different machine. dataset_repo_id /
    # dataset_root only kick in when the user explicitly passes them.
    _override_dataset_in_config(resolved, dataset_repo_id, dataset_root)
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


def _load_residual_policy(path, device, dataset_repo_id=None, dataset_root=None):
    resolved = _resolve_ckpt_path(path, "residual_policy_path")
    print(f"[a2c2 ] resolved ckpt dir: {resolved}")
    # Always run config patcher: fixes baked-in absolute HF cache paths
    # left over from training on a different machine. dataset_repo_id /
    # dataset_root only kick in when the user explicitly passes them.
    _override_dataset_in_config(resolved, dataset_repo_id, dataset_root)
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

    # If provided, the policy's normalization stats are pulled from this
    # dataset (rather than from the ckpt directory or the dataset name
    # baked into the ckpt's config.json). Use this when the 4090 doesn't
    # have the original training dataset cached and the policy's
    # auto-resolution fails with "Repo id must be in the form ...".
    base_dataset_repo_id: Optional[str] = None
    residual_dataset_repo_id: Optional[str] = None
    base_dataset_root: Optional[str] = None
    residual_dataset_root: Optional[str] = None

    # Bypass for draccus's `dict[str, CameraConfig]` parsing on lerobot
    # forks where the discriminated-union choice gets confused (draccus
    # tries the wrong subclass first and chokes on `name=` kwarg).
    # Pass a flat JSON: '{"laptop": {"index": 4, "width": 640, "height":
    # 480, "fps": 30}, "wrist": {"index": 6}}' — every entry is built as
    # an OpenCVCameraConfig, even if the physical device is a RealSense
    # used through its OpenCV interface (matches data-collection setup).
    # When set, this overrides any cameras passed via --robot.cameras.
    cameras_json: Optional[str] = None

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
            vh = self._vlm_hidden
            # SmolVLA may stash vlm_hidden as either:
            #   - (D,)            single token broadcast across all k
            #   - (T, D)          per-chunk-step token (T == chunk_size)
            #   - (1, D) / (1,T,D) batched variants
            # Resolve to a single (D,) entry for this k.
            if vh.dim() == 1:
                entry = vh
            elif vh.dim() == 2:
                if vh.shape[0] == 1:                       # (1, D)
                    entry = vh[0]
                elif vh.shape[0] >= self.H or vh.shape[0] > k:
                    entry = vh[k]                          # (T, D)
                else:
                    entry = vh[-1]                         # fallback last
            elif vh.dim() == 3:
                # (B, T, D) or (1, T, D)
                t_axis = 1
                if vh.shape[t_axis] == 1:
                    entry = vh[0, 0]
                elif vh.shape[t_axis] > k:
                    entry = vh[0, k]
                else:
                    entry = vh[0, -1]
            else:
                entry = vh.reshape(-1, vh.shape[-1])[
                    min(k, vh.reshape(-1, vh.shape[-1]).shape[0] - 1)]
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
# Observation preprocessing — convert raw robot output to SmolVLA layout
# --------------------------------------------------------------------------- #
def _get_robot_motor_order(robot):
    """Best-effort lookup of the robot's motor names in canonical order.

    For SO-100/SO-101 follower, this is the physical chain order:
    shoulder_pan → shoulder_lift → elbow_flex → wrist_flex →
    wrist_roll → gripper. Sorting alphabetically would mangle the
    state vector, so we read the actual motor list from the robot
    instance.
    """
    candidates = []
    # lerobot 0.5+: robot.bus.motors is an OrderedDict
    bus = getattr(robot, "bus", None)
    if bus is not None:
        motors = getattr(bus, "motors", None)
        if hasattr(motors, "keys"):
            candidates.extend(motors.keys())
    if not candidates:
        # Try robot.motors directly
        m = getattr(robot, "motors", None)
        if hasattr(m, "keys"):
            candidates.extend(m.keys())
    if not candidates:
        # Action features as a last resort
        af = getattr(robot, "action_features", None)
        if hasattr(af, "keys"):
            for k in af.keys():
                if isinstance(k, str) and k.endswith(".pos"):
                    candidates.append(k.removesuffix(".pos"))
    return list(candidates)


def _preprocess_observation(obs, device, motor_order=None):
    """Convert raw lerobot robot.get_observation() output to the layout
    SmolVLA / residual_transformer expects.

    Real-robot observations from lerobot 0.5+ have:
      - Per-motor scalars: `shoulder_pan.pos`, `shoulder_lift.pos`,
        ..., `gripper.pos`     (each is a Python float or 0-d tensor)
      - Cameras: `front`, `hand` (HWC uint8 numpy/tensor)

    SmolVLA's prepare_state / prepare_images expect:
      - `observation.state`             — torch.float32, shape (1, N)
      - `observation.images.<name>`     — torch.float32, shape (1, 3, H, W) in [0, 1]

    This helper:
      1. Aggregates every key ending in `.pos` (or `.position`) into a
         single sorted vector and labels it `observation.state`.
      2. Converts image HWC uint8 → BCHW float [0, 1] and renames bare
         camera keys to `observation.images.<name>`.
      3. Forwards `task` and other non-tensor metadata unchanged.
    """
    out = {}
    state_parts = {}      # name -> scalar value (for sorted aggregation)
    for k, v in obs.items():
        # Per-motor position scalars → collect for `observation.state`
        if k.endswith(".pos") or k.endswith(".position"):
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            if isinstance(v, torch.Tensor):
                state_parts[k] = float(v.detach().reshape(-1)[0].cpu().item())
            else:
                state_parts[k] = float(v)
            continue

        if not isinstance(v, (np.ndarray, torch.Tensor)):
            out[k] = v
            continue

        # numpy → torch
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        v = v.to(device)

        # Detect images by *shape signature* (HWC with 3 channels, or
        # CHW / BCHW with 3 channels) in addition to key heuristics.
        # This catches bare camera-name keys like `front` / `hand` /
        # `wrist` that don't carry "image" in their name.
        def _looks_like_image(t):
            if t.dim() == 3:
                if t.shape[-1] in (1, 3, 4) and t.shape[-1] != t.shape[0]:
                    return True   # HWC
                if t.shape[0] in (1, 3, 4):
                    return True   # CHW
            if t.dim() == 4 and t.shape[1] in (1, 3, 4):
                return True       # BCHW
            return False

        is_image = (
            "image" in k.lower()
            or k.startswith("observation.image")
            or _looks_like_image(v)
        )
        if is_image:
            # HWC → CHW
            if v.dim() == 3 and v.shape[-1] in (1, 3, 4):
                v = v.permute(2, 0, 1).contiguous()
            # CHW → BCHW
            if v.dim() == 3:
                v = v.unsqueeze(0)
            # uint8 [0,255] → float [0,1]
            if v.dtype == torch.uint8:
                v = v.float() / 255.0
            elif v.dtype != torch.float32:
                v = v.float()
            # Rewrap bare camera names under observation.images.<name>
            if not k.startswith("observation."):
                k = f"observation.images.{k}"
            out[k] = v
            continue

        # Already-aggregated state passthrough (shape correction)
        if k.endswith("state") or k == "observation.state":
            if v.dim() == 1:
                v = v.unsqueeze(0)
            if v.dtype != torch.float32:
                v = v.float()
            out["observation.state"] = v
            continue

        # Anything else: cast tensors to float, pass through
        if v.dtype != torch.float32 and v.dtype != torch.long:
            v = v.float()
        out[k] = v

    # Build observation.state from collected `.pos` scalars
    if state_parts and "observation.state" not in out:
        if motor_order:
            # Use robot's canonical motor order (physical chain order),
            # not alphabetical — must match training.
            ordered_keys = []
            for name in motor_order:
                for suffix in (".pos", ".position"):
                    k = name + suffix
                    if k in state_parts:
                        ordered_keys.append(k)
                        break
            # Append any leftover keys at the end (in alphabetical order
            # for stability).
            leftover = sorted(set(state_parts.keys()) - set(ordered_keys))
            ordered_keys.extend(leftover)
        else:
            ordered_keys = sorted(state_parts.keys())
        vec = torch.tensor([state_parts[n] for n in ordered_keys],
                           dtype=torch.float32, device=device).unsqueeze(0)
        out["observation.state"] = vec
        if "_state_keys" not in out:
            out["_state_keys"] = ordered_keys      # debug aid

    return out


def _action_array_to_dict(action_np, motor_order):
    """Convert a 1-D action array into the {name.pos: float} dict that
    lerobot 0.5+ robot.send_action() expects.
    """
    if motor_order is None or len(motor_order) == 0:
        raise RuntimeError(
            "Cannot dispatch action: robot motor order unknown. "
            "Pass --robot.* correctly so robot.bus.motors is populated."
        )
    flat = np.asarray(action_np).reshape(-1)
    if len(flat) != len(motor_order):
        raise RuntimeError(
            f"Action dim mismatch: model output has {len(flat)} values but "
            f"robot has {len(motor_order)} motors ({motor_order}). "
            f"Check --action-dim and your residual head config."
        )
    return {f"{name}.pos": float(v) for name, v in zip(motor_order, flat)}


# --------------------------------------------------------------------------- #
# Episode loop
# --------------------------------------------------------------------------- #
def run_episode(robot, inferencer: A2C2Inferencer, cfg: A2C2InferenceConfig,
                episode_idx: int) -> dict:
    """Drive one episode end-to-end. Returns metrics dict."""
    inferencer.reset()
    motor_order = _get_robot_motor_order(robot)
    if episode_idx == 0 and motor_order:
        print(f"[robot] motor order: {motor_order}")

    obs = robot.get_observation()
    obs_t = _preprocess_observation(obs, cfg.device, motor_order=motor_order)

    if episode_idx == 0:
        # Print the resolved observation layout once so users can verify
        # the keys / shapes match what SmolVLA was trained on.
        print("[obs ] preprocessed layout:")
        for k, v in obs_t.items():
            if k == "_state_keys":
                print(f"  state aggregated from: {v}")
            elif isinstance(v, torch.Tensor):
                print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v).__name__} = {str(v)[:60]}")

    tick_dt = cfg.tick_dt_ms / 1000.0
    latencies: list[float] = []
    for tick in range(cfg.max_episode_steps):
        t0 = time.perf_counter()
        action = inferencer.step(obs_t, cfg.task)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        action_np = action.cpu().numpy().astype(np.float32)
        action_dict = _action_array_to_dict(action_np, motor_order)
        robot.send_action(action_dict)

        if tick_dt > 0:
            sleep_for = tick_dt - (time.perf_counter() - t0)
            if sleep_for > 0:
                time.sleep(sleep_for)

        # Refresh observation (re-read sensors)
        obs = robot.get_observation()
        obs_t = _preprocess_observation(obs, cfg.device, motor_order=motor_order)

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

    # Build OpenCV cameras from --cameras-json (bypasses draccus union
    # parsing — useful on lerobot forks where the discriminator chokes
    # on `name` kwarg or picks the wrong subclass).
    if cfg.cameras_json:
        import json
        try:
            cams_spec = json.loads(cfg.cameras_json)
        except json.JSONDecodeError as e:
            raise SystemExit(
                f"--cameras-json is not valid JSON: {e}\n"
                f"  raw: {cfg.cameras_json!r}\n"
                f"  example: --cameras-json='{{\"laptop\": {{\"index\": 4, "
                f"\"width\": 640, \"height\": 480, \"fps\": 30}}, \"wrist\": "
                f"{{\"index\": 6}}}}'"
            )
        # Locate OpenCVCameraConfig — different layouts in different forks.
        OpenCVCameraConfig = None
        for mod in (
            "lerobot.cameras.opencv.configuration_opencv",
            "lerobot.cameras.opencv",
            "lerobot.common.cameras.opencv.configuration_opencv",
        ):
            try:
                m = __import__(mod, fromlist=["OpenCVCameraConfig"])
                OpenCVCameraConfig = getattr(m, "OpenCVCameraConfig", None) \
                                     or getattr(m, "OpenCVConfig", None)
                if OpenCVCameraConfig is not None:
                    break
            except (ImportError, AttributeError):
                continue
        if OpenCVCameraConfig is None:
            raise ImportError("Could not locate OpenCVCameraConfig in lerobot")

        cams = {}
        for cam_name, spec in cams_spec.items():
            kw = {}
            # Robust against either `index` or `index_or_path` field name.
            idx_value = spec.get("index", spec.get("index_or_path", 0))
            kw["index_or_path"] = idx_value
            for k in ("width", "height", "fps"):
                if k in spec:
                    kw[k] = spec[k]
            cams[cam_name] = OpenCVCameraConfig(**kw)
            print(f"[cameras] {cam_name}: {OpenCVCameraConfig.__name__}({kw})")
        cfg.robot.cameras = cams

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
    base = _load_smolvla(
        cfg.base_policy_path, device,
        dataset_repo_id=cfg.base_dataset_repo_id,
        dataset_root=cfg.base_dataset_root,
    )
    for p in base.parameters():
        p.requires_grad = False

    # --- residual head ---
    residual = None
    if cfg.residual_policy_path:
        print(f"[a2c2 ] loading {cfg.residual_policy_path}")
        residual = _load_residual_policy(
            cfg.residual_policy_path, device,
            dataset_repo_id=cfg.residual_dataset_repo_id,
            dataset_root=cfg.residual_dataset_root,
        )
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

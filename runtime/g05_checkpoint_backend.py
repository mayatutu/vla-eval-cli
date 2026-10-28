"""Load a G05 checkpoint through its native processor and inference pipeline.

This family backend accepts the existing drawer wire observation and returns
DROID absolute joint targets. Importing this module does not load model code.
"""
from __future__ import annotations

import importlib.util
from contextlib import contextmanager
import hashlib
import math
import os
from pathlib import Path
import sys
from threading import Lock

import numpy as np

from drawer_bridge import observation_to_batch


def _weights_path(checkpoint, profile):
    relative = Path(profile.get("g05_weights_file", "checkpoints/model_state_dict.pt"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("g05_weights_file must stay within the selected checkpoint")
    weights = (checkpoint / relative).resolve()
    if checkpoint.resolve() not in weights.parents:
        raise ValueError("G05 weights resolve outside the selected checkpoint")
    return weights


def validate_runtime_profile(profile):
    """Require explicit execution settings without claiming a training frequency."""
    steps = profile.get("action_steps")
    if type(steps) is not int or not 1 <= steps <= 32:
        raise ValueError("G05 action_steps must be in 1..32")
    hz = profile.get("control_hz")
    if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not math.isfinite(hz) or hz <= 0:
        raise ValueError("G05 control_hz needs an explicit positive value")
    if not profile.get("control_hz_evidence"):
        raise ValueError("G05 control_hz needs its source or an explicit test-condition note")
    if profile.get("g05_gripper_mode") not in ("droid", "inverted"):
        raise ValueError("G05 gripper convention must be explicitly selected")
    if not profile.get("g05_gripper_evidence"):
        raise ValueError("G05 gripper conversion evidence is required")
    if profile.get("g05_embodiment") not in ("Droid_Franka", "galaxea_r1lite"):
        raise ValueError("unsupported G05 checkpoint embodiment")


def validate_native_config(cfg, profile):
    """Keep the checkpoint's own 20D/27D merger and normalization pipeline."""
    arch, data = cfg["model"]["model_arch"], cfg["data"]
    if (arch.get("_target_") != "g05.models.g05.g05_policy_qwen35.G05PolicyQwen35"
            or arch.get("continuous_action") is not True
            or arch.get("return_continuous_action") is not True):
        raise ValueError("G05 requires the reviewed Qwen35 continuous action head")
    embodiment = profile["g05_embodiment"]
    expected_dim = 20 if embodiment == "Droid_Franka" else 27
    if arch.get("action_dim") != expected_dim or data.get("action_size") != 32:
        raise ValueError("checkpoint action width/horizon differs from its reviewed layout")
    processor = data["processors"][embodiment]
    shape_meta = processor["shape_meta"]
    for key in ("action", "state"):
        shapes = {item["key"]: item["raw_shape"] for item in shape_meta[key]}
        if shapes != {"right_arm": 7, "right_gripper": 1}:
            raise ValueError("checkpoint must declare seven joints and one gripper")
    transforms = processor.get("action_state_transforms", [])
    if (len(transforms) != 1 or transforms[0].get("_target_") !=
            "g05.data_processor.transforms.relative_action.RelativeJointTransform"
            or list(transforms[0].get("keys", [])) != ["right_arm"]):
        raise ValueError("unsupported G05 action transform; native decoding must restore absolute joints")
    images = shape_meta["images"]
    real = [item for item in images if not item.get("dummy", False)]
    dummy = [item for item in images if item.get("dummy", False)]
    if ({item.get("camera_type") for item in real} != {"exterior", "wrist_left"}
            or len(real) != 2 or len(dummy) != 1 or dummy[0].get("camera_type") != "wrist_right"):
        raise ValueError("G05 checkpoint requires left exterior/wrist plus a dummy right wrist")
    for item in dummy:
        shape = list(item.get("shape", []))
        if len(shape) != 3 or shape[0] != 3 or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError("invalid checkpoint dummy camera shape")
    return images


def _profile_paths(profile):
    checkpoint = Path(profile["checkpoint"]).expanduser().resolve()
    processor = Path(profile["processor"]).expanduser().resolve()
    repo = Path(profile["g05_repo"]).expanduser().resolve()
    overlay = Path(profile["g05_overlay"]).expanduser().resolve()
    if (overlay / "src" / "g05").is_dir():
        overlay = overlay / "src"
    required = (
        checkpoint / ".hydra" / "config.yaml",
        _weights_path(checkpoint, profile),
        checkpoint / "dataset_stats.json",
        checkpoint / "action_tokenizer.pt",
        repo / "scripts" / "serve_policy.py",
        overlay / "g05" / "models" / "g05" / "inferencer.py",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"G05 required local file is missing: {path}")
    for name, digest in profile.get("asset_sha256", {}).items():
        path = _weights_path(checkpoint, {"g05_weights_file": name})
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"G05 checkpoint configuration/statistics changed: {name}")
    if not processor.is_dir() or not (repo / "src").is_dir():
        raise FileNotFoundError("G05 processor and repository src must exist locally")
    return checkpoint, processor, repo, overlay


def _load_native(repo, overlay):
    # These paths supply the already deployed continuous-head correction. Do not
    # import the base inferencer first and silently lose the correction.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for path in (repo / "src", overlay):
        sys.path.insert(0, str(path))
    module_name = "_g05_checkpoint_native_server"
    spec = importlib.util.spec_from_file_location(
        module_name, repo / "scripts" / "serve_policy.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the existing G05 native server")
    native = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = native
    spec.loader.exec_module(native)
    inferencer_module = sys.modules["g05.models.g05.inferencer"]
    if Path(inferencer_module.__file__).resolve() != (
        overlay / "g05" / "models" / "g05" / "inferencer.py"
    ).resolve():
        raise RuntimeError("G05 inferencer did not load from the configured overlay")
    return native


@contextmanager
def _preserve_native_tokenizer_aliases(loader):
    """Keep OmegaConf.update from replacing aliased dictionaries with ckpt_dir only."""
    from omegaconf import DictConfig, OmegaConf

    namespace = loader.__globals__
    original = namespace["_apply_action_tokenizer_sidecar"]

    def preserve_then_apply(cfg, run_dir):
        # The saved exports alias model.tokenizer to the root tokenizer and
        # AT_CONFIG to its vq_config. Native sidecar updates nested keys with
        # merge=False, which otherwise discards the rest of either alias.
        for parent, key in ((cfg.model, "tokenizer"),
                            (cfg.model.model_arch, "AT_CONFIG")):
            if key in parent and OmegaConf.is_interpolation(parent, key):
                value = parent[key]
                if not isinstance(value, DictConfig):
                    raise ValueError(f"G05 tokenizer alias {key} must resolve to a dictionary")
                parent[key] = OmegaConf.to_container(value, resolve=False)
        return original(cfg, run_dir)

    namespace["_apply_action_tokenizer_sidecar"] = preserve_then_apply
    try:
        yield
    finally:
        namespace["_apply_action_tokenizer_sidecar"] = original


def _load_config(native, checkpoint, processor, profile):
    weights = _weights_path(checkpoint, profile)
    run_dir = Path(native.find_run_dir(str(weights))).resolve()
    if run_dir != checkpoint:
        raise ValueError("G05 native loader resolved a different checkpoint run directory")
    # The new weights, their tokenizer and their statistics stay together. Only
    # the generic, existing local Hugging Face processor assets are reused.
    overrides = [
        f"eval_embodiment={profile['g05_embodiment']}",
        f"model.model_arch.hf_processor_path={processor.as_posix()}",
        f"model.model_arch.AT_CONFIG.ckpt_dir={(checkpoint / 'action_tokenizer.pt').as_posix()}",
    ]
    with _preserve_native_tokenizer_aliases(native.load_config_from_run_dir):
        cfg = native.load_config_from_run_dir(checkpoint, str(weights), overrides)
    from omegaconf import OmegaConf

    OmegaConf.update(cfg, "ckpt_path", str(weights), force_add=True)
    OmegaConf.update(cfg, "run_dir", str(checkpoint), force_add=True)
    OmegaConf.update(
        cfg, "datastatics_path", str(checkpoint / "dataset_stats.json"), force_add=True
    )
    # Native load resolves interpolations before applying overrides; patch the
    # already-resolved processor path so no training-host path survives.
    OmegaConf.update(cfg, "model.processor.tokenizer_params.pretrained_model_name_or_path",
                     str(processor), force_add=True)
    if "embodiment_datasets" in cfg.data:
        native.filter_embodiment(cfg, profile["g05_embodiment"])
    return cfg


@contextmanager
def _native_config_context(repo, profile):
    """Resolve native relative YAML paths without modifying the environment on disk."""
    original_cwd = Path.cwd()
    keys = ("RC_DATA", "EMBODIMENT", "TASK")
    original = {key: os.environ.get(key) for key in keys}
    # These paths only occur in unused training dataset_groups. Setting the
    # explicitly selected checkpoint root avoids depending on the training host.
    values = {"RC_DATA": profile["checkpoint"], "EMBODIMENT": profile["g05_embodiment"],
              "TASK": profile.get("model_id", "inference")}
    try:
        os.chdir(repo)
        for key, value in values.items():
            os.environ[key] = str(value)
        yield
    finally:
        os.chdir(original_cwd)
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def require_exact_state_dict(model, state_dict):
    """Reject missing, extra and mismatched tensors before any permissive loader."""
    expected = model.state_dict()
    if not expected or not isinstance(state_dict, dict):
        raise RuntimeError("G05 checkpoint/model state must be a nonempty tensor mapping")
    missing, unexpected = set(expected) - set(state_dict), set(state_dict) - set(expected)
    mismatched = [key for key in set(expected) & set(state_dict)
                  if tuple(expected[key].shape) != tuple(state_dict[key].shape)]
    if missing or unexpected or mismatched:
        raise RuntimeError("G05 exact checkpoint load rejected: "
                           f"missing={sorted(missing)[:12]}, unexpected={sorted(unexpected)[:12]}, "
                           f"shape_mismatch={sorted(mismatched)[:12]}")
    return len(expected)


@contextmanager
def strict_native_weights(checkpoint_utils):
    """Use native load/dtype/tied-weight logic, but forbid its partial-load path."""
    original = checkpoint_utils.load_state_dict_safely

    def verified(model, state_dict, extra_prefixes=None, load_config=None, return_info=False):
        count = require_exact_state_dict(model, state_dict)
        config = dict(load_config or {})
        if (config.get("key_prefix_to_remove") or config.get("ignore_key_prefixes")
                or config.get("token_row_remap")):
            raise RuntimeError("G05 checkpoint requires an unreviewed key/token remapping")
        config.update(partial_load_keys=[], ignore_key_prefixes=[], auto_token_row_remap=False)
        info = original(model, state_dict, extra_prefixes=[], load_config=config, return_info=True)
        for key in ("truly_missing", "partial_loaded_keys", "mismatched_keys",
                    "unexpected_keys", "extra_loaded_keys"):
            if info.get(key):
                raise RuntimeError(f"G05 native checkpoint loader reported {key}: {info[key]}")
        if info.get("loaded_count") != count:
            raise RuntimeError("G05 native loader did not load every expected tensor")
        return info if return_info else info["model"]

    checkpoint_utils.load_state_dict_safely = verified
    try:
        yield
    finally:
        checkpoint_utils.load_state_dict_safely = original


def wire_to_native(payload, control_hz, *, embodiment, image_meta, gripper_mode):
    """Use the checkpoint's exact image keys and explicitly reviewed gripper map."""
    batch = observation_to_batch(payload)
    state = batch["observation.state"]
    if gripper_mode not in ("droid", "inverted"):
        raise ValueError("G05 gripper convention must be explicit")
    sources = {"exterior": batch["observation.images.left_image"],
               "wrist_left": batch["observation.images.wrist"]}
    images = {}
    for item in image_meta:
        if item.get("dummy", False):
            images[item["key"]] = np.zeros(tuple(item["shape"]), dtype=np.uint8)
        else:
            images[item["key"]] = sources[item["camera_type"]].copy()
    return {
        "images": images,
        "state": {"right_arm": state[:7].copy(),
                  "right_gripper": 1.0 - state[7:] if gripper_mode == "inverted" else state[7:].copy()},
        "task": batch["task"],
        "frequency": control_hz,
        "embodiment_type": embodiment,
    }


def _chunk_array(value, width, name):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if (array.ndim != 2 or array.shape[1] != width or array.shape[0] < 1
            or array.dtype.kind not in "fi" or not np.isfinite(array).all()):
        raise ValueError(f"G05 {name} must be a finite (1,H,{width}) or (H,{width}) chunk")
    array = array.astype(np.float32, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"G05 {name} cannot be represented as float32")
    return array


def native_to_droid(action, current_gripper, action_steps, *, gripper_mode):
    """Convert only native output semantics; native postprocess owns unnormalizing."""
    if gripper_mode not in ("droid", "inverted"):
        raise ValueError("G05 gripper convention must be explicit")
    absent = set(action.get("_absent_keys", ()))
    arm_key = "right_arm" if "right_arm" in action else "joint_position"
    if arm_key not in action or arm_key in absent:
        raise ValueError("G05 output is missing the main arm action")
    joints = _chunk_array(action[arm_key], 7, arm_key)
    if len(joints) < action_steps:
        raise ValueError("G05 output chunk is shorter than configured action_steps")
    grip_key = next(
        (key for key in ("right_gripper", "gripper") if key in action and key not in absent),
        None,
    )
    if grip_key is None:
        grip = np.asarray(current_gripper, dtype=np.float32)
        if grip.shape != (1,) or not np.isfinite(grip).all():
            raise ValueError("current DROID gripper must contain one finite value")
        grippers = np.broadcast_to(grip, (action_steps, 1)).copy()
    else:
        grippers = _chunk_array(action[grip_key], 1, grip_key)
        if len(grippers) != len(joints):
            raise ValueError("G05 arm and gripper chunk lengths disagree")
        grippers = grippers[:action_steps]
        if gripper_mode == "inverted":
            grippers = 1.0 - grippers
    # Retain the existing adapter's gripper bounds, but the selected profile owns
    # its direction. Absolute arm targets are not clipped or integrated.
    return np.concatenate(
        (joints[:action_steps], np.clip(grippers, 0.0, 1.0)), axis=1
    ).astype(np.float32, copy=False)


class G05Backend:
    def __init__(self, native, policy, processor, *, action_steps, control_hz, device,
                 embodiment, image_meta, gripper_mode):
        self.native = native
        self.processor = processor
        self.action_steps = action_steps
        self.control_hz = control_hz
        self.embodiment = embodiment
        self.image_meta = image_meta
        self.gripper_mode = gripper_mode
        self.inferencer = native.PolicyInferencer(policy, processor, device=device)
        self._lock = Lock()

    def __call__(self, payload):
        raw_obs = wire_to_native(payload, self.control_hz, embodiment=self.embodiment,
                                image_meta=self.image_meta, gripper_mode=self.gripper_mode)
        current_gripper = np.asarray(payload["state"]["gripper_position"])[0, 0]
        with self._lock:
            self.native._parse_task_and_plan(raw_obs)
            obs_dict = self.native.build_obs_dict(raw_obs, self.processor)
            actions = self.inferencer.infer([obs_dict])
            if not isinstance(actions, (list, tuple)) or len(actions) != 1:
                raise ValueError("G05 native inference must return one postprocessed sample")
            return native_to_droid(actions[0], current_gripper, self.action_steps,
                                   gripper_mode=self.gripper_mode)


def build_backend(profile):
    """Load one checkpoint; no server, Client, robot or camera is started here."""
    validate_runtime_profile(profile)
    steps, hz = profile["action_steps"], profile["control_hz"]
    checkpoint, processor_path, repo, overlay = _profile_paths(profile)
    native = _load_native(repo, overlay)
    device = profile.get("device", "cuda:0")
    from g05.utils.checkpoint import checkpoint_utils
    with _native_config_context(repo, profile):
        cfg = _load_config(native, checkpoint, processor_path, profile)
        image_meta = validate_native_config(cfg, profile)
        if steps > int(cfg.data.action_size):
            raise ValueError("G05 action_steps exceeds the checkpoint prediction horizon")
        with strict_native_weights(checkpoint_utils):
            policy, processor = native.setup(cfg, device=device)
    return G05Backend(
        native, policy, processor, action_steps=steps, control_hz=hz, device=device,
        embodiment=profile["g05_embodiment"], image_meta=image_meta,
        gripper_mode=profile["g05_gripper_mode"],
    )

"""LeRobot PI05 checkpoint backend for absolute DROID joint targets.

Importing this module does not load ML packages, weights, or devices. The
checkpoint owns its preprocessing and action statistics; old OpenPI DROID
velocity normalization is deliberately not used here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

import numpy as np

from drawer_bridge import observation_to_batch


ACTION_NAMES = [f"joint_target_{index}" for index in range(7)] + ["gripper_target"]
IMAGE_KEYS = ("observation.images.left_image", "observation.images.wrist")


def validate_checkpoint_config(config, profile):
    """Reject a different action convention before importing/loading the model."""
    if profile.get("action_space") != "joint_position":
        raise ValueError("PI05 checkpoint backend requires absolute joint_position")
    steps = profile.get("action_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("action_steps must be a positive integer")
    expected = {
        "type": "pi05", "n_obs_steps": 1, "chunk_size": steps,
        "n_action_steps": steps, "use_relative_actions": False,
        "action_feature_names": ACTION_NAMES,
    }
    for name, value in expected.items():
        if name not in config or config[name] != value:
            raise ValueError(f"unsupported PI05 checkpoint {name}: expected {value!r}")
    if config["use_relative_actions"] is not False:
        raise ValueError("relative action decoding has not been validated for this backend")
    inputs = config.get("input_features", {})
    for name, shape in (("observation.state", [8]), ("observation.cartesian_position", [6])):
        if inputs.get(name, {}).get("shape") != shape:
            raise ValueError(f"unsupported checkpoint feature {name}; expected {shape}")
    if config.get("output_features", {}).get("action", {}).get("shape") != [8]:
        raise ValueError("checkpoint action must have eight DROID values")
    actual_images = {name for name in inputs if name.startswith("observation.images.")}
    if actual_images != {*IMAGE_KEYS, "observation.images.empty_camera_0"}:
        raise ValueError("checkpoint must declare the two reviewed views and one masked camera")
    for name in actual_images:
        shape = inputs[name].get("shape", [])
        if len(shape) != 3 or shape[0] != 3 or min(shape[1:]) <= 0:
            raise ValueError(f"invalid RGB image shape for {name}")


def validate_processor_documents(preprocessor, postprocessor):
    """Saved pipeline settings must agree with the absolute checkpoint config."""
    for document, registry_name in (
        (preprocessor, "relative_actions_processor"),
        (postprocessor, "absolute_actions_processor"),
    ):
        matches = [step for step in document.get("steps", [])
                   if step.get("registry_name") == registry_name]
        if len(matches) != 1 or matches[0].get("config", {}).get("enabled") is not False:
            raise ValueError(f"saved {registry_name} must occur once and be disabled")


def observation_to_numpy_batch(payload):
    """Convert RGB to native PI05 [0,1] CHW and keep the real Cartesian pose."""
    batch = observation_to_batch(payload)
    pose = np.asarray(payload["state"]["cartesian_position"])
    if (pose.shape != (1, 1, 6) or pose.dtype.kind not in "fi"
            or not np.isfinite(pose).all()):
        raise ValueError("cartesian_position must be finite numeric (1,1,6)")
    batch["observation.cartesian_position"] = pose[0, 0].astype(np.float32, copy=True)
    if not np.isfinite(batch["observation.cartesian_position"]).all():
        raise ValueError("Cartesian pose cannot be represented as float32")
    for name in IMAGE_KEYS:
        batch[name] = batch[name].astype(np.float32) / np.float32(255)
    # Native PI05 supplies the missing-image mask; a fabricated third frame
    # would incorrectly turn that mask on.
    return batch


def decoded_to_actions(value, steps):
    """Preserve all decoded physical values, rejecting malformed outputs."""
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value)
    if value.shape != (1, steps, 8) or value.dtype.kind != "f" or not np.isfinite(value).all():
        raise ValueError(f"decoded PI05 actions must be finite float (1,{steps},8)")
    result = value[0].astype(np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("decoded actions cannot be represented as float32")
    return result


def load_weights_strict(policy, checkpoint, config, *, load_file=None):
    """Never use PI05Policy.from_pretrained's exception-to-random-model fallback."""
    if load_file is None:
        from safetensors.torch import load_file
    original = load_file(str(Path(checkpoint) / "model.safetensors"), device="cpu")
    fixed = policy._fix_pytorch_state_dict_keys(original, config)
    normalized = {}
    for name, tensor in fixed.items():
        target = name if name.startswith("model.") else f"model.{name}"
        if target in normalized:
            raise ValueError(f"checkpoint key collision after native conversion: {target}")
        normalized[target] = tensor
    # strict=True checks shapes, missing tensors, and unexpected tensors. Load
    # exceptions propagate and prevent the server from reporting readiness.
    policy.load_state_dict(normalized, strict=True)


class Pi05Backend:
    def __init__(self, preprocess, predict, postprocess, tensorize, steps):
        self.preprocess = preprocess
        self.predict = predict
        self.postprocess = postprocess
        self.tensorize = tensorize
        self.steps = steps
        self.lock = Lock()

    def __call__(self, payload):
        batch = observation_to_numpy_batch(payload)
        with self.lock:
            prepared = self.preprocess(self.tensorize(batch))
            raw = self.predict(prepared)
            decoded = self.postprocess(raw)
            return decoded_to_actions(decoded, self.steps)


def build_backend(profile):
    """Explicitly load one local checkpoint and its own native processor pair."""
    checkpoint = Path(profile["checkpoint"])
    tokenizer = Path(profile["tokenizer"])
    if not checkpoint.is_dir() or not tokenizer.is_dir():
        raise ValueError("checkpoint and tokenizer must be existing local directories")
    raw_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    validate_checkpoint_config(raw_config, profile)
    for name in ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        if not (checkpoint / name).is_file():
            raise ValueError(f"incomplete checkpoint: missing {name}")
    validate_processor_documents(
        json.loads((checkpoint / "policy_preprocessor.json").read_text(encoding="utf-8")),
        json.loads((checkpoint / "policy_postprocessor.json").read_text(encoding="utf-8")),
    )

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    import torch
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy
    # Register this checkpoint's native state-tokenizer processor before loading
    # saved JSON pipelines. Do not build a replacement pipeline from old stats.
    from lerobot.policies.pi05 import processor_pi05  # noqa: F401

    torch.set_num_threads(1)
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = "cpu"
    config.compile_model = False
    with torch.device("cpu"):
        policy = PI05Policy(config)
    load_weights_strict(policy, checkpoint, config)
    policy.requires_grad_(False)
    policy.to("cuda").eval()
    config.device = "cuda"
    preprocess, postprocess = make_pre_post_processors(
        config, pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(tokenizer)},
            "device_processor": {"device": "cuda"},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    def tensorize(batch):
        # The checkpoint's to_batch step adds the leading batch dimension and
        # converts task text to a list. Adding a batch here would double it.
        return {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                for key, value in batch.items()}

    def predict(batch):
        with torch.inference_mode():
            return policy.predict_action_chunk(batch)

    return Pi05Backend(preprocess, predict, postprocess, tensorize, profile["action_steps"])

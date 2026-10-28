"""Native MolmoAct2 checkpoint inference; no changes to checkpoint or environment.

The reviewed DROID export owns its normalization, image processor and absolute
action decoding. Importing this module does not import Torch or touch devices.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import numpy as np

from drawer_bridge import observation_to_batch


IMAGE_KEYS = ["observation.images.left_image", "observation.images.right_image",
              "observation.images.wrist"]
ACTION_NAMES = [f"joint_target_{i}" for i in range(7)] + ["gripper_target"]
STATE_NAMES = [f"joint_{i}" for i in range(7)] + ["gripper_position"]


def validate_checkpoint_documents(config, statistics, profile):
    """Confirm native horizon, three views, absolute semantics and own 8D stats."""
    if profile.get("action_space") != "joint_position" or profile.get("action_dim") != 8:
        raise ValueError("MolmoAct2 requires absolute joint_position with eight values")
    steps = profile.get("action_steps")
    if type(steps) is not int or steps < 1:
        raise ValueError("action_steps must be a positive integer")
    for key, expected in (("model_type", "molmoact2"), ("action_mode", "continuous"),
                          ("add_action_expert", True), ("n_obs_steps", 1),
                          ("norm_stats_filename", "norm_stats.json")):
        if config.get(key) != expected:
            raise ValueError(f"unsupported MolmoAct2 {key}: expected {expected!r}")
    if (type(config.get("max_action_horizon")) is not int
            or config["max_action_horizon"] < steps):
        raise ValueError("checkpoint does not support the selected action horizon")
    if statistics.get("format") != "molmoact2_norm_stats.v1":
        raise ValueError("unsupported MolmoAct2 normalization format")
    if statistics.get("norm_mode") != "q01_q99":
        raise ValueError("only the reviewed checkpoint quantile normalization is supported")
    tag = profile.get("norm_tag")
    metadata = statistics.get("metadata_by_tag", {}).get(tag)
    if not isinstance(metadata, dict):
        raise ValueError("selected norm_tag is absent from checkpoint statistics")
    for key, expected in (("control_mode", "absolute joint pose"),
                          ("camera_keys", IMAGE_KEYS),
                          ("state_keys", ["observation.state"]),
                          ("action_horizon", steps), ("n_action_steps", steps),
                          ("normalize_gripper", False)):
        if metadata.get(key) != expected:
            raise ValueError(f"unsupported MolmoAct2 normalization metadata {key}")
    # Native get_action_dim uses the real statistic length. action_dim in this
    # export's metadata is the padded expert width (32), not the robot width.
    for name, names in (("action_stats", ACTION_NAMES), ("state_stats", STATE_NAMES)):
        values = metadata.get(name, {})
        if values.get("names") != names or values.get("mask") != [True] * 7 + [False]:
            raise ValueError(f"unsupported {name} joint/gripper mapping")
        for key in ("min", "max", "q01", "q99", "mean", "std"):
            value = np.asarray(values.get(key, []))
            if value.shape != (8,) or value.dtype.kind not in "fi" or not np.isfinite(value).all():
                raise ValueError(f"{name}.{key} must contain eight finite values")
    if type(profile.get("num_steps")) is not int or profile["num_steps"] < 1:
        raise ValueError("num_steps must be a positive denoising iteration count")
    return metadata


def observation_to_inputs(payload):
    """Keep left/right/wrist RGB order and unmodified physical state values."""
    batch = observation_to_batch(payload)
    right = np.asarray(payload["video"]["exterior_image_2_left"])
    left = batch["observation.images.left_image"].transpose(1, 2, 0)
    wrist = batch["observation.images.wrist"].transpose(1, 2, 0)
    if right.dtype != np.uint8 or right.shape != (1, 1, *left.shape):
        raise ValueError("right exterior must be uint8 RGB (1,1,H,W,3), matching the other views")
    return [left.copy(), right[0, 0].copy(), wrist.copy()], batch["observation.state"], batch["task"]


def decoded_to_actions(value, steps):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value)
    if value.shape != (1, steps, 8) or value.dtype.kind != "f" or not np.isfinite(value).all():
        raise ValueError(f"decoded MolmoAct2 actions must be finite float (1,{steps},8)")
    result = value[0].astype(np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("decoded actions cannot be represented as float32")
    return result


def check_loading_info(info):
    """Fail closed if Transformers would continue with uninitialized weights."""
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(key):
            raise RuntimeError(f"MolmoAct2 checkpoint load has {key}: {info[key]}")


class MolmoBackend:
    def __init__(self, predict, processor, to_image, profile, cache=None):
        self.predict = predict
        self.processor = processor
        self.to_image = to_image
        self.profile = profile
        self.cache = cache
        self.lock = Lock()

    def __call__(self, payload):
        images, state, task = observation_to_inputs(payload)
        with self.lock:
            output = self.predict(
                processor=self.processor, images=[self.to_image(image) for image in images],
                task=task, state=state, norm_tag=self.profile["norm_tag"],
                inference_action_mode="continuous", enable_depth_reasoning=False,
                num_steps=self.profile["num_steps"], n_action_steps=self.profile["action_steps"],
                normalize_language=True, enable_cuda_graph=False, return_dict=True,
            )
        return decoded_to_actions(output.actions, self.profile["action_steps"])


def build_backend(profile):
    """Load the unmodified local export with its native processor and decoder."""
    checkpoint = Path(profile["checkpoint"])
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    statistics = json.loads((checkpoint / "norm_stats.json").read_text(encoding="utf-8"))
    validate_checkpoint_documents(config, statistics, profile)
    # Remote-code loading copies modules into this private temporary directory.
    # It never rewrites downloaded code or the production HF modules cache.
    cache = TemporaryDirectory(prefix="vla-molmo-modules-")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
                      HF_MODULES_CACHE=cache.name)
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor

        torch.set_num_threads(1)
        processor = AutoProcessor.from_pretrained(
            str(checkpoint), trust_remote_code=True, local_files_only=True,
            extra_special_tokens={},
        )
        model, loading_info = AutoModelForImageTextToText.from_pretrained(
            str(checkpoint), trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, output_loading_info=True,
        )
        check_loading_info(loading_info)
        model.requires_grad_(False)
        model.to("cuda").eval()

        def predict(**kwargs):
            with torch.inference_mode():
                return model.predict_action(**kwargs)

        return MolmoBackend(predict, processor, Image.fromarray, profile, cache)
    except BaseException:
        cache.cleanup()
        raise

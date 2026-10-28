"""Task-checkpoint LingBot adapter using the checkpoint's native transforms.

Importing this file opens no sockets, loads no weights and imports no ML stack.
The returned action contract is absolute joints, even though LingBot's learned
arm targets are relative to the observation supplied for that prediction.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import sys
from threading import Lock
from types import SimpleNamespace

import numpy as np

from drawer_bridge import observation_to_batch


IMAGE_MAPPING = (
    ("observation.images.camera_top", "observation.images.left_image"),
    ("observation.images.camera_wrist_left", "observation.images.right_image"),
    ("observation.images.camera_wrist_right", "observation.images.wrist"),
)


def validate_documents(config, training, robot, stats, profile):
    """Accept the reviewed DROID joint convention; reject semantic changes."""
    expected = {"model_type": "lingbotvla", "architectures": ["LingbotVlaV2Policy"],
                "n_obs_steps": 1, "max_state_dim": 55, "max_action_dim": 55,
                "action_dim": 55, "chunk_size": 50, "n_action_steps": 50,
                "vlm_family": "qwen3_vl", "use_qwen3_chat_template": True,
                "return_image_grid_thw": True}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"unsupported LingBot checkpoint {key}: expected {value!r}")
    if profile.get("action_space") != "joint_position" or profile.get("action_steps") != 50:
        raise ValueError("LingBot DROID requires 50 decoded absolute-joint actions")
    data = training["data"]
    if data.get("cameras") != [key.removeprefix("observation.images.") for key, _ in IMAGE_MAPPING]:
        raise ValueError("LingBot camera slots must preserve left, right, wrist order")
    if data.get("img_size") != 256 or data.get("state_norm_type") != "none":
        raise ValueError("unreviewed LingBot image/state preprocessing settings")
    norm_type = {key: value for item in data["norm_type"]
                 for key, value in ast.literal_eval(item).items()}
    for name in ("arm.position", "effector.position"):
        if norm_type.get(name) != "meanstd":
            raise ValueError("LingBot checkpoint requires its saved meanstd transform")
    for category, prefix in (("states", "observation.state"), ("actions", "action")):
        entries = robot.get(category, [])
        if len(entries) != 2:
            raise ValueError("LingBot robot mapping requires arm and gripper only")
        for entry, part, start, end in zip(entries, ("arm", "effector"), (0, 7), (7, 8)):
            target = f"{prefix}.{part}.position"
            if set(entry) != {target}:
                raise ValueError("LingBot robot feature order differs from DROID")
            mapping = entry[target]
            if mapping.get("origin_keys") != [{prefix: {"start": start, "end": end}}]:
                raise ValueError("LingBot joint/gripper source slices differ from DROID")
            if category == "actions":
                if mapping.get("subtract_state") is not (part == "arm"):
                    raise ValueError("LingBot requires relative arm and absolute gripper")
                if part == "arm" and mapping.get("relative_type") != "joint":
                    raise ValueError("LingBot joint deltas must not use Cartesian decoding")
    images = robot.get("images", [])
    if images != [{target: {"origin_keys": source}} for target, source in IMAGE_MAPPING]:
        raise ValueError("LingBot checkpoint requires three distinct physical views")
    values = stats.get("norm_stats", {})
    for prefix in ("action", "observation.state"):
        for part, size in (("arm", 7), ("effector", 1)):
            feature = f"{prefix}.{part}.position"
            for field in ("mean", "std"):
                value = np.asarray(values.get(feature, {}).get(field))
                allowed = ((size,), (50, size)) if prefix == "action" else ((size,),)
                if (value.shape not in allowed or value.dtype.kind not in "fi"
                        or not np.isfinite(value).all()
                        or (field == "std" and (value < 0).any())):
                    raise ValueError(f"invalid checkpoint normalization {feature}.{field}")


def observation_to_numpy_batch(payload):
    batch = observation_to_batch(payload)
    right = np.asarray(payload["video"].get("exterior_image_2_left"))
    left = np.asarray(payload["video"]["exterior_image_1_left"])
    if right.dtype != np.uint8 or right.shape != left.shape:
        raise ValueError("LingBot requires a real right exterior RGB frame matching the other views")
    batch["observation.images.right_image"] = right[0, 0].transpose(2, 0, 1).copy()
    return batch


def decoded_to_actions(value, steps=50):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value)
    if value.shape != (steps, 8) or value.dtype.kind != "f" or not np.isfinite(value).all():
        raise ValueError(f"LingBot native decoder must return finite float ({steps},8)")
    result = value.astype(np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("LingBot actions cannot be represented as float32")
    return result


class LingbotBackend:
    """Keep the exact prepared observation with its relative-action decoder."""
    def __init__(self, transform, predict, tensorize):
        self.transform = transform
        self.predict = predict
        self.tensorize = tensorize
        self.lock = Lock()

    def __call__(self, payload):
        batch = observation_to_numpy_batch(payload)
        with self.lock:
            prepared = self.transform.apply(self.tensorize(batch), policy_eval=True)
            raw = self.predict(prepared)
            # The native unapply performs its own unnormalization and exactly
            # one addition of this observation's joints. The gripper is absolute.
            decoded = self.transform.unapply({**prepared, "actions": raw})["action"]
            return decoded_to_actions(decoded)


def checkpoint_shards(checkpoint):
    checkpoint = Path(checkpoint)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("LingBot checkpoint has no weight map")
    names = sorted(set(weight_map.values()))
    if any(not isinstance(name, str) or Path(name).name != name
           or not name.endswith(".safetensors") for name in names):
        raise ValueError("LingBot shard index must contain local safetensors filenames")
    for name in names:
        if not (checkpoint / name).is_file():
            raise ValueError(f"missing LingBot weight shard: {name}")
    return weight_map, [checkpoint / name for name in names]


def load_weights_strict(model, checkpoint, *, safe_open, torch):
    """Verify every indexed tensor/shape before assigning BF16 CUDA weights."""
    weight_map, shards = checkpoint_shards(checkpoint)
    expected = model.state_dict()
    actual = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as source:
            for key in source.keys():
                if key in actual or weight_map.get(key) != shard.name:
                    raise ValueError(f"duplicate/unindexed LingBot tensor: {key}")
                actual[key] = tuple(source.get_slice(key).get_shape())
    if set(actual) != set(weight_map):
        raise ValueError("LingBot shard keys do not match the saved index")
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = [key for key in expected.keys() & actual.keys()
                  if tuple(expected[key].shape) != actual[key]]
    if missing or unexpected or mismatched:
        raise ValueError(f"LingBot strict shape check failed: missing={missing[:8]}, "
                         f"unexpected={unexpected[:8]}, mismatched={mismatched[:8]}")
    weights = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as source:
            for key in source.keys():
                tensor = source.get_tensor(key)
                weights[key] = tensor.to(device="cuda", dtype=(torch.bfloat16
                                        if tensor.is_floating_point() else tensor.dtype))
    model.load_state_dict(weights, strict=True, assign=True)
    del weights
    if any(tensor.is_meta for tensor in model.parameters()):
        raise ValueError("LingBot strict load left meta parameters")


def build_backend(profile):
    """Load the selected package using existing LingBot/Qwen dependencies."""
    checkpoint = Path(profile["checkpoint"])
    repo = Path(profile["lingbot_repo"])
    vlm_config = Path(profile["lingbot_vlm_config"])
    robot_path = checkpoint / profile["robot_config"]
    if not repo.is_dir() or not (vlm_config / "config.json").is_file():
        raise ValueError("LingBot requires the existing repository and local Qwen3-VL assets")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    sys.path.insert(0, str(repo))
    import yaml
    config_doc = json.loads((checkpoint / "config.json").read_text())
    training = yaml.safe_load((checkpoint / "lingbotvla_cli.yaml").read_text())
    robot_doc = yaml.safe_load(robot_path.read_text())
    stats_path = checkpoint / "norm_stats.json"
    stats = json.loads(stats_path.read_text())
    validate_documents(config_doc, training, robot_doc, stats, profile)
    checkpoint_shards(checkpoint)

    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from torchvision.transforms.v2 import Resize
    from transformers import AutoConfig, AutoProcessor
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
    from lingbotvla.data.vla_data.utils import FeatureTransform

    torch.set_num_threads(1)
    apply_lingbot_qwen3_vl_patch()
    config = LingbotVLAV2Config(**config_doc)
    for key, value in config_doc.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    config.tokenizer_path = str(vlm_config)
    # Same evaluation settings as the repository's deployment policy; weights
    # and feature semantics stay owned by the selected checkpoint package.
    config.attention_implementation = "eager"
    config.use_compile = False
    config.use_cache = True
    base = AutoConfig.from_pretrained(str(vlm_config), local_files_only=True)
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "rms_norm_eps",
                "rope_theta", "vocab_size", "max_position_embeddings",
                "hidden_act", "tie_word_embeddings"):
        if hasattr(base.text_config, key):
            setattr(config, key, getattr(base.text_config, key))
    config.vision_config = base.vision_config
    processor = AutoProcessor.from_pretrained(str(checkpoint), padding_side="right",
                                             local_files_only=True, trust_remote_code=False)
    transform = FeatureTransform(str(robot_path), SimpleNamespace(**training["data"]),
                                 config, processor, chunk_size=50,
                                 norm_stats_path=str(stats_path), image_augment=False,
                                 use_future_image=False)
    # Parameters are assigned from the checkpoint; non-persistent buffers such
    # as rotary frequencies must retain their normally initialized CPU values.
    with init_empty_weights(include_buffers=False):
        model = LingbotVlaV2Policy(config, eval=True)
    load_weights_strict(model, checkpoint, safe_open=safe_open, torch=torch)
    model = model.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    model.model._use_compile_predict_velocity = False
    model.model._compiled_predict_velocity = None
    resize = Resize((training["data"]["img_size"], training["data"]["img_size"]))

    def tensorize(batch):
        result = {}
        for key, value in batch.items():
            if key.startswith("observation.images."):
                # The native deployment resizes float32 RGB in 0..255 before
                # FeatureTransform and the checkpoint image processor rescale it.
                result[key] = resize(torch.from_numpy(value).to(torch.float32))
            else:
                result[key] = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        return result

    def predict(prepared):
        # Match the native BF16 deployment's state cast and float32 decode path.
        prepared["state"] = prepared["state"].to(torch.bfloat16).to(torch.float32)
        if tuple(prepared["img_masks"].shape) != (3,) or not prepared["img_masks"].all():
            raise ValueError("LingBot requires all three physical camera masks")
        with torch.inference_mode():
            raw = model.sample_actions(
                images=prepared["images"].unsqueeze(0).to(device="cuda", dtype=torch.bfloat16),
                img_masks=prepared["img_masks"].unsqueeze(0).to("cuda"),
                lang_tokens=prepared["lang_tokens"].unsqueeze(0).to("cuda"),
                lang_masks=prepared["lang_masks"].unsqueeze(0).to("cuda"),
                state=prepared["state"].unsqueeze(0).to(device="cuda", dtype=torch.bfloat16),
                image_grid_thw=prepared["image_grid_thw"].unsqueeze(0).to(device="cuda", dtype=torch.long),
            )
        if tuple(raw.shape) != (1, 50, 55) or not torch.isfinite(raw).all():
            raise ValueError("LingBot must return finite normalized (1,50,55) predictions")
        return raw[0].float().cpu()

    return LingbotBackend(transform, predict, tensorize)

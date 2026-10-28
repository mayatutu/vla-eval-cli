"""LeRobot MolmoAct2 task checkpoints with their saved processor pipelines.

The native HF export loader is a separate format. This module loads the full
LeRobot state dictionary once, without loading the base model's weights first.
Importing the module does not import ML packages or access devices.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from threading import Lock

import numpy as np

from drawer_bridge import observation_to_batch


IMAGE_KEYS = ["observation.images.left_image", "observation.images.right_image",
              "observation.images.wrist"]
ACTION_NAMES = [f"joint_target_{index}" for index in range(7)] + ["gripper_target"]
STATE_NAMES = [f"joint_{index}" for index in range(7)] + ["gripper_position"]
PRE_STEPS = ["rename_observations_processor", "to_batch_processor",
             "molmoact2_state_frame_transform", "molmoact2_masked_normalizer",
             "molmoact2_clamp_normalized", "molmoact2_pack_inputs", "device_processor"]
POST_STEPS = ["molmoact2_clamp_action", "molmoact2_masked_unnormalizer",
              "molmoact2_action_frame_transform", "device_processor"]


def validate_checkpoint_documents(config, preprocessor, postprocessor, profile):
    """Validate absolute-joint semantics and use the saved normalization state."""
    steps = profile.get("action_steps")
    if (profile.get("action_space") != "joint_position" or profile.get("action_dim") != 8
            or type(steps) is not int or steps <= 0):
        raise ValueError("LeRobot MolmoAct2 requires absolute 8D joint targets")
    expected = {
        "type": "molmoact2", "n_obs_steps": 1, "chunk_size": steps,
        "n_action_steps": steps, "action_mode": "continuous",
        "inference_action_mode": "continuous", "norm_tag": None,
        "control_mode": "absolute joint pose", "image_keys": IMAGE_KEYS,
        "normalize_gripper": False, "expected_max_action_dim": 32,
        "joint_signs": None, "joint_offsets": None, "rtc_config": None,
        "enable_lora_vlm": False, "enable_lora_action_expert": False,
        "model_dtype": "bfloat16",
        "normalization_mapping": {"ACTION": "QUANTILES", "STATE": "QUANTILES", "VISUAL": "IDENTITY"},
    }
    for name, value in expected.items():
        if name not in config or config[name] != value:
            raise ValueError(f"unsupported LeRobot MolmoAct2 {name}: expected {value!r}")
    inference_steps = config.get("num_inference_steps")
    if inference_steps is not None and (type(inference_steps) is not int or inference_steps <= 0):
        raise ValueError("invalid checkpoint inference step setting")
    inputs = config.get("input_features", {})
    if set(inputs) != {*IMAGE_KEYS, "observation.state", "observation.cartesian_position"}:
        raise ValueError("LeRobot MolmoAct2 requires three real views, state and Cartesian pose")
    for name, shape in (("observation.state", [8]), ("observation.cartesian_position", [6])):
        if inputs[name].get("shape") != shape:
            raise ValueError(f"unsupported checkpoint feature {name}")
    for name in IMAGE_KEYS:
        shape = inputs[name].get("shape", [])
        if len(shape) != 3 or shape[0] != 3 or min(shape[1:]) <= 0:
            raise ValueError("checkpoint image feature must contain three RGB channels")
    if config.get("output_features", {}).get("action", {}).get("shape") != [8]:
        raise ValueError("checkpoint must output eight robot action values")
    if config.get("dataset_feature_names") != {"action": ACTION_NAMES, "observation.state": STATE_NAMES}:
        raise ValueError("unreviewed LeRobot MolmoAct2 joint/gripper order")
    for document, names in ((preprocessor, PRE_STEPS), (postprocessor, POST_STEPS)):
        if [step.get("registry_name") for step in document.get("steps", [])] != names:
            raise ValueError("checkpoint processor pipeline differs from the reviewed native sequence")
    pre = {step["registry_name"]: step for step in preprocessor["steps"]}
    post = {step["registry_name"]: step for step in postprocessor["steps"]}
    for name, step in (("state", pre["molmoact2_state_frame_transform"]),
                       ("action", post["molmoact2_action_frame_transform"])):
        if step.get("config") != {"joint_signs": None, "joint_offsets": None}:
            raise ValueError(f"unreviewed {name} frame transformation")
    if pre["rename_observations_processor"].get("config") != {"rename_map": {}}:
        raise ValueError("checkpoint observation renaming differs from the selected views")
    pack = pre["molmoact2_pack_inputs"].get("config", {})
    for name in ("checkpoint_path", "checkpoint_revision", "checkpoint_force_download",
                 "action_mode", "image_keys", "setup_type", "control_mode", "normalize_language",
                 "add_setup_tokens", "add_control_tokens", "num_state_tokens", "max_sequence_length"):
        if pack.get(name) != config.get(name):
            raise ValueError(f"saved MolmoAct2 pack_inputs disagrees with policy {name}")
    for name, value in (("chunk_size", steps), ("max_action_dim", 32),
                        ("env_action_dim", 8), ("allow_image_key_fallback", False)):
        if pack.get(name) != value:
            raise ValueError(f"unreviewed MolmoAct2 input packing {name}")
    for document, name in ((pre, "molmoact2_masked_normalizer"),
                           (post, "molmoact2_masked_unnormalizer")):
        step = document[name]
        filename = step.get("state_file")
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError("normalization must use the checkpoint's local saved state")
        if step.get("config", {}).get("norm_map") != config["normalization_mapping"]:
            raise ValueError("saved MolmoAct2 normalization differs from the checkpoint")


def observation_to_numpy_batch(payload):
    batch = observation_to_batch(payload)
    left = np.asarray(payload["video"]["exterior_image_1_left"])
    right = np.asarray(payload["video"].get("exterior_image_2_left"))
    if right.dtype != np.uint8 or right.shape != left.shape:
        raise ValueError("LeRobot MolmoAct2 requires a real right exterior RGB frame")
    batch["observation.images.right_image"] = right[0, 0].transpose(2, 0, 1).copy()
    pose = np.asarray(payload["state"].get("cartesian_position"))
    if pose.shape != (1, 1, 6) or pose.dtype.kind not in "fi" or not np.isfinite(pose).all():
        raise ValueError("Cartesian pose must be finite numeric (1,1,6)")
    batch["observation.cartesian_position"] = pose[0, 0].astype(np.float32, copy=True)
    if not np.isfinite(batch["observation.cartesian_position"]).all():
        raise ValueError("Cartesian pose cannot be represented as float32")
    # The native pack_inputs accepts uint8 CHW and owns RGB conversion/resizing.
    # No external division by 255 is applied before its saved processor pipeline.
    return batch


def decoded_to_actions(value, steps):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value)
    if value.shape != (1, steps, 8) or value.dtype.kind != "f" or not np.isfinite(value).all():
        raise ValueError(f"LeRobot MolmoAct2 decoded actions must be finite float (1,{steps},8)")
    result = value[0].astype(np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("decoded actions cannot be represented as float32")
    return result


def verify_base_assets(base_assets, profile):
    """Pin the reviewed offline architecture and prompt/image assets."""
    required = {"config.json", "processor_config.json", "tokenizer_config.json",
                "tokenizer.json", "chat_template.jinja"}
    hashes = profile.get("base_asset_sha256", {})
    if not required <= hashes.keys():
        raise ValueError("offline MolmoAct2 base assets require reviewed hashes")
    for filename, digest in hashes.items():
        if Path(filename).name != filename:
            raise ValueError("base assets must be direct local files")
        path = base_assets / filename
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"missing or changed offline MolmoAct2 base asset: {filename}")


def restore_empty_runtime_buffers(policy, torch):
    """Restore only native empty RoPE caches excluded from the state dictionary."""
    for module_name, module in policy.named_modules():
        for name in module._non_persistent_buffers_set:
            value = module._buffers[name]
            if value is None or not value.is_meta:
                continue
            if name not in {"_pos_sin_cache", "_pos_cos_cache"} or value.numel() != 0:
                raise ValueError(f"unreviewed meta runtime buffer: {module_name}.{name}")
            module.register_buffer(name, torch.empty_like(value, device="cpu"), persistent=False)
        # Native RoPE keeps this ordinary alias outside named_buffers. Strict
        # assign replaces inv_freq; preserve the native alias to the loaded tensor.
        original = getattr(module, "original_inv_freq", None)
        if original is not None and original.is_meta:
            loaded = getattr(module, "inv_freq", None)
            if loaded is None or loaded.is_meta:
                raise ValueError("RoPE inv_freq was not loaded from the target checkpoint")
            module.original_inv_freq = loaded


def configure_checkpoint_horizon(hf_config, action_steps):
    """Match both native configuration levels to the saved LeRobot chunk size."""
    if type(action_steps) is not int or action_steps <= 0 or hf_config.action_expert_config is None:
        raise ValueError("a positive checkpoint horizon and action expert are required")
    hf_config.max_action_horizon = action_steps
    hf_config.action_expert_config.max_action_horizon = action_steps


def load_policy_weights_strict(policy, weight_file, *, safe_open, torch):
    """Validate the complete LeRobot key space before reading any large tensor."""
    expected = policy.state_dict()
    with safe_open(str(weight_file), framework="pt", device="cpu") as source:
        actual = {name: tuple(source.get_slice(name).get_shape()) for name in source.keys()}
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        mismatched = [name for name in expected.keys() & actual.keys()
                      if tuple(expected[name].shape) != actual[name]]
        if missing or unexpected or mismatched:
            raise ValueError(f"LeRobot MolmoAct2 strict weight mismatch: missing={missing[:8]}, "
                             f"unexpected={unexpected[:8]}, shape={mismatched[:8]}")
        weights = {}
        for name in actual:
            tensor = source.get_tensor(name)
            weights[name] = tensor.to(device="cuda", dtype=(torch.bfloat16
                                    if tensor.is_floating_point() else tensor.dtype))
    policy.load_state_dict(weights, strict=True, assign=True)
    del weights
    restore_empty_runtime_buffers(policy, torch)
    if any(value.is_meta for value in policy.parameters()):
        raise ValueError("LeRobot MolmoAct2 still has unloaded meta parameters")
    if any(value.is_meta for value in policy.buffers()):
        raise ValueError("LeRobot MolmoAct2 has uninitialized meta buffers")


class MolmoLeRobotBackend:
    def __init__(self, preprocess, predict, postprocess, tensorize, steps):
        self.preprocess, self.predict, self.postprocess = preprocess, predict, postprocess
        self.tensorize, self.steps = tensorize, steps
        self.lock = Lock()

    def __call__(self, payload):
        batch = observation_to_numpy_batch(payload)
        with self.lock:
            prepared = self.preprocess(self.tensorize(batch))
            raw = self.predict(prepared)
            return decoded_to_actions(self.postprocess(raw), self.steps)


def build_backend(profile):
    checkpoint, base_assets = Path(profile["checkpoint"]), Path(profile["base_assets"])
    verify_base_assets(base_assets, profile)
    config_doc = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    pre_doc = json.loads((checkpoint / "policy_preprocessor.json").read_text(encoding="utf-8"))
    post_doc = json.loads((checkpoint / "policy_postprocessor.json").read_text(encoding="utf-8"))
    validate_checkpoint_documents(config_doc, pre_doc, post_doc, profile)
    for document in (pre_doc, post_doc):
        for step in document["steps"]:
            if "state_file" in step and not (checkpoint / step["state_file"]).is_file():
                raise ValueError("missing saved processor state: " + step["state_file"])
    weight_file = checkpoint / "model.safetensors"
    if not weight_file.is_file() or list(checkpoint.glob("*.incomplete")):
        raise ValueError("LeRobot MolmoAct2 checkpoint download is incomplete")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")

    import torch
    from safetensors import safe_open
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
    from lerobot.policies.molmoact2 import modeling_molmoact2 as native
    from lerobot.policies.molmoact2 import processor_molmoact2  # noqa: F401

    torch.set_num_threads(1)
    config = MolmoAct2Config.from_pretrained(checkpoint, local_files_only=True)
    config.pretrained_path = str(checkpoint)
    config.checkpoint_path = str(base_assets)
    config.checkpoint_revision = None
    config.checkpoint_force_download = False
    config.device = "cpu"

    class LocalCheckpointPolicy(native.MolmoAct2Policy):
        def _load_hf_model(self):
            # The native constructor otherwise loads all base weights first.
            # This standalone subclass changes construction only: all model math,
            # action generation and saved preprocessing remain native LeRobot.
            hf_config = native.HFMolmoAct2Config.from_pretrained(str(base_assets), local_files_only=True)
            if int(getattr(hf_config, "max_action_dim", -1)) != self.config.expected_max_action_dim:
                raise ValueError("offline MolmoAct2 base action dimension differs from policy")
            native._validate_checkpoint_action_mode(
                self.config, str(hf_config.action_mode),
                has_action_expert=bool(getattr(hf_config, "add_action_expert", False)))
            # Native HF config construction normally synchronizes these values.
            # Loading base assets first leaves their 15-step expert limit behind
            # unless the saved LeRobot horizon is applied before constructing it.
            configure_checkpoint_horizon(hf_config, int(self.config.chunk_size))
            # Torch's native meta device avoids another package and allocating
            # the base model's full CPU weights. Target tensors are strict-assigned.
            with torch.device("meta"):
                self.model = native.MolmoAct2ForConditionalGeneration(hf_config)
            self._override_loaded_max_action_horizon(int(self.config.chunk_size))

    policy = LocalCheckpointPolicy(config)
    load_policy_weights_strict(policy, weight_file, safe_open=safe_open, torch=torch)
    policy.requires_grad_(False)
    policy.to(device="cuda", dtype=torch.bfloat16).eval()
    config.device = "cuda"
    preprocess, postprocess = make_pre_post_processors(
        config, pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "molmoact2_pack_inputs": {"checkpoint_path": str(base_assets),
                                     "checkpoint_revision": None, "checkpoint_force_download": False},
            "device_processor": {"device": "cuda"},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    def tensorize(batch):
        return {name: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                for name, value in batch.items()}

    def predict(batch):
        with torch.inference_mode():
            # None remains None: the native checkpoint's inference-step default
            # is distinct from its num_flow_timesteps training setting.
            return policy.predict_action_chunk(batch)

    return MolmoLeRobotBackend(preprocess, predict, postprocess, tensorize, profile["action_steps"])

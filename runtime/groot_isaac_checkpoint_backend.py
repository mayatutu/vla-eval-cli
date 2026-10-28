"""Task checkpoints exported by Isaac-GR00T; native processing and decoding.

Importing this module does not load Torch, the model, or a GPU. It does not use
the LeRobot GR00T drawer adapter: these checkpoints have a different format.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
from threading import Lock
from types import SimpleNamespace

import numpy as np

from drawer_bridge import observation_to_batch


COSMOS_ID = "nvidia/Cosmos-Reason2-2B"
EMBODIMENT = "new_embodiment"
ACTION_STEPS = 16
SOURCE_FILES = {
    "policy/policy.py", "policy/gr00t_policy.py", "data/types.py",
    "model/gr00t_n1d7/gr00t_n1d7.py",
    "model/gr00t_n1d7/processing_gr00t_n1d7.py",
    "model/modules/qwen3_backbone.py", "configs/model/gr00t_n1d7.py",
}


def dependency_inventory():
    result = {"python": sys.executable, "modules": {}, "packages": {}, "gpu_initialized": False}
    for name in ("gr00t", "torch", "transformers", "safetensors"):
        spec = importlib.util.find_spec(name)
        result["modules"][name] = None if spec is None else {
            "origin": spec.origin, "search_locations": list(spec.submodule_search_locations or [])}
    for name in ("gr00t", "torch", "transformers", "safetensors"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    return result


def _asset(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Isaac checkpoint assets must be local relative paths")
    path = root / relative
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Missing or escaped Isaac checkpoint asset: " + name)
    return path


def validate_configuration(config, processor):
    if (config.get("model_type") != "Gr00tN1d7"
            or config.get("architectures") != ["Gr00tN1d7"]
            or config.get("model_name") != COSMOS_ID
            or config.get("action_horizon") != 40
            or config.get("max_action_dim") != 132 or config.get("max_state_dim") != 132):
        raise ValueError("Isaac-GR00T model configuration differs from reviewed task exports")
    settings = processor.get("processor_kwargs", {})
    expected = {"model_name": COSMOS_ID, "use_percentiles": True, "use_relative_action": True,
                "use_mean_std": False, "clip_outliers": True, "exclude_state": False,
                "max_action_horizon": 40, "max_action_dim": 132, "max_state_dim": 132}
    if (processor.get("processor_class") != "Gr00tN1d7Processor"
            or any(settings.get(k) != v for k, v in expected.items())):
        raise ValueError("Isaac-GR00T native processing settings differ from reviewed task exports")
    modalities = settings.get("modality_configs", {}).get(EMBODIMENT, {})
    for kind, keys, indices in (
        ("video", ["left", "wrist"], [0]), ("state", ["arm", "gripper"], [0]),
        ("action", ["arm", "gripper"], list(range(ACTION_STEPS))),
        ("language", ["annotation.human.task_description"], [0]),
    ):
        spec = modalities.get(kind, {})
        if spec.get("modality_keys") != keys or spec.get("delta_indices") != indices:
            raise ValueError("Unexpected Isaac-GR00T task modalities/horizon: " + kind)
        if spec.get("sin_cos_embedding_keys") or spec.get("mean_std_embedding_keys"):
            raise ValueError("Unreviewed state/action embedding")
    action_configs = modalities["action"].get("action_configs")
    expected_configs = [{"rep": rep, "type": "NON_EEF", "format": "DEFAULT", "state_key": None}
                        for rep in ("RELATIVE", "ABSOLUTE")]
    if action_configs != expected_configs:
        raise ValueError("Arm must be relative joint position; gripper must be absolute")


def validate_statistics(statistics):
    selected = statistics.get(EMBODIMENT, {})
    for kind, key, shape in (
        ("state", "arm", (7,)), ("state", "gripper", (1,)),
        ("action", "arm", (7,)), ("action", "gripper", (1,)),
        ("relative_action", "arm", (ACTION_STEPS, 7)),
    ):
        values = selected.get(kind, {}).get(key, {})
        arrays = {}
        for name in ("mean", "std", "min", "max", "q01", "q99"):
            value = np.asarray(values.get(name))
            if value.shape != shape or value.dtype.kind not in "fi" or not np.isfinite(value).all():
                raise ValueError(f"Invalid Isaac {kind}.{key}.{name}; expected {shape}")
            arrays[name] = value
        if ((arrays["std"] < 0).any() or (arrays["min"] > arrays["max"]).any()
                or (arrays["q01"] > arrays["q99"]).any()):
            raise ValueError("Invalid Isaac-GR00T statistics range")


def inspect_checkpoint(profile):
    if (profile.get("backend") != "groot_isaac" or profile.get("action_space") != "joint_position"
            or profile.get("action_dim") != 8 or profile.get("action_steps") != ACTION_STEPS):
        raise ValueError("Isaac task adapter requires decoded absolute joint positions, 16 x 8")
    checkpoint = Path(profile["checkpoint"])
    if not checkpoint.is_dir() or next(checkpoint.rglob("*.incomplete"), None):
        raise ValueError("Isaac checkpoint is absent or incomplete")
    required = {"config.json", "processor/processor_config.json", "processor/statistics.json",
                "processor/embodiment_id.json", "model.safetensors.index.json"}
    digests = profile.get("asset_sha256", {})
    if not required.issubset(digests):
        raise ValueError("Reviewed Isaac model/processor/statistics/index hashes are required")
    for name, digest in digests.items():
        if hashlib.sha256(_asset(checkpoint, name).read_bytes()).hexdigest() != digest:
            raise ValueError("Isaac checkpoint asset changed: " + name)
    read = lambda name: json.loads(_asset(checkpoint, name).read_text())
    validate_configuration(read("config.json"), read("processor/processor_config.json"))
    validate_statistics(read("processor/statistics.json"))
    mapping = read("processor/embodiment_id.json")
    if type(mapping.get(EMBODIMENT)) is not int or not 0 <= mapping[EMBODIMENT] < 32:
        raise ValueError("Missing valid task embodiment/projector mapping")
    index = read("model.safetensors.index.json")
    if not isinstance(index.get("weight_map"), dict) or not index["weight_map"]:
        raise ValueError("Empty Isaac weight shard index")
    for name in set(index["weight_map"].values()):
        if not name.endswith(".safetensors") or _asset(checkpoint, name).stat().st_size <= 8:
            raise ValueError("Missing/incomplete Isaac weight shard: " + name)
    return {"checkpoint": str(checkpoint), "configuration_hashes_verified": True,
            "weights_loaded": False, "full_weight_hash_verified": False,
            "native_runtime_verified": False, "native_action_steps": ACTION_STEPS}


def verify_runtime_sources(profile):
    root = Path(profile["isaac_source_root"])
    hashes = profile.get("native_source_sha256", {})
    if not SOURCE_FILES.issubset(hashes):
        raise ValueError("Reviewed Isaac-GR00T runtime source hashes are required")
    for name, digest in hashes.items():
        if hashlib.sha256(_asset(root, name).read_bytes()).hexdigest() != digest:
            raise ValueError("Isaac-GR00T native runtime source changed: " + name)
    return root


def checked_loading_result(result, label, *, require_materialized=True):
    if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[1], dict):
        raise RuntimeError(label + " did not return weight-loading diagnostics")
    model, info = result
    failures = {key: info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if info.get(key)}
    if failures:
        raise RuntimeError(label + " weights are not an exact load: " + str(failures))
    if require_materialized:
        remaining = [name for name, parameter in model.named_parameters() if parameter.is_meta]
        if remaining:
            raise RuntimeError(label + " has unresolved meta parameters: " + str(remaining[:8]))
    return model


@contextmanager
def local_native_loaders(policy, backbone, processing, checkpoint, cosmos, dtype):
    """Temporarily redirect loading inside this isolated process; never edit sources.

    Keep config.model_name as the Cosmos identifier because native backbone
    selection depends on that name. Resolve only the loader's filesystem input.
    """
    original_model = policy.AutoModel
    original_processor = policy.AutoProcessor
    original_backbone = backbone.Qwen3VLForConditionalGeneration
    original_build_processor = processing.build_processor
    checkpoint, cosmos = Path(checkpoint).resolve(), Path(cosmos).resolve()
    if not cosmos.is_dir() or not (cosmos / "config.json").is_file():
        raise FileNotFoundError("Local Cosmos snapshot is required: " + str(cosmos))

    def load_model(path, **kwargs):
        if Path(path).resolve() != checkpoint:
            raise ValueError("Unexpected task checkpoint path")
        kwargs.update(local_files_only=True, output_loading_info=True, torch_dtype=dtype)
        kwargs.pop("ignore_mismatched_sizes", None)
        return checked_loading_result(original_model.from_pretrained(path, **kwargs), "Isaac task")

    def load_backbone(name, **kwargs):
        if name != COSMOS_ID:
            raise ValueError("Unreviewed Isaac backbone: " + str(name))
        kwargs.update(local_files_only=True, output_loading_info=True)
        kwargs.setdefault("torch_dtype", dtype)
        kwargs.pop("ignore_mismatched_sizes", None)
        # The outer task loader constructs this nested backbone under its meta
        # context, then materializes it from the task checkpoint. Only the outer
        # completed model may be required to have real parameters at this point.
        return checked_loading_result(original_backbone.from_pretrained(str(cosmos), **kwargs),
                                      "Cosmos", require_materialized=False)

    def build_processor(name, transformers_loading_kwargs):
        if name != COSMOS_ID:
            raise ValueError("Unreviewed Isaac processor backbone: " + str(name))
        kwargs = dict(transformers_loading_kwargs)
        kwargs.update(local_files_only=True, trust_remote_code=False)
        return original_build_processor(str(cosmos), kwargs)

    def load_processor(path, **kwargs):
        if Path(path).resolve() != checkpoint / "processor":
            raise ValueError("Unexpected Isaac task processor path")
        kwargs.update(local_files_only=True, transformers_loading_kwargs={
            "local_files_only": True, "trust_remote_code": False})
        return original_processor.from_pretrained(path, **kwargs)

    policy.AutoModel = SimpleNamespace(from_pretrained=load_model)
    policy.AutoProcessor = SimpleNamespace(from_pretrained=load_processor)
    backbone.Qwen3VLForConditionalGeneration = SimpleNamespace(from_pretrained=load_backbone)
    processing.build_processor = build_processor
    try:
        yield
    finally:
        policy.AutoModel = original_model
        policy.AutoProcessor = original_processor
        backbone.Qwen3VLForConditionalGeneration = original_backbone
        processing.build_processor = original_build_processor


def observation_to_inputs(payload):
    batch = observation_to_batch(payload)
    return {
        "video": {"left": batch["observation.images.left_image"].transpose(1, 2, 0)[None, None].copy(),
                  "wrist": batch["observation.images.wrist"].transpose(1, 2, 0)[None, None].copy()},
        "state": {"arm": batch["observation.state"][None, None, :7].copy(),
                  "gripper": batch["observation.state"][None, None, 7:].copy()},
        "language": {"annotation.human.task_description": [[batch["task"]]]},
    }


def decoded_to_actions(response):
    if not isinstance(response, tuple) or len(response) != 2 or not isinstance(response[0], dict):
        raise ValueError("Native Isaac policy must return (decoded actions, info)")
    parts = []
    for key, dim in (("arm", 7), ("gripper", 1)):
        value = np.asarray(response[0].get(key))
        if (value.shape != (1, ACTION_STEPS, dim) or value.dtype.kind != "f"
                or not np.isfinite(value).all()):
            raise ValueError(f"Native Isaac {key} must be finite float (1,16,{dim})")
        parts.append(value[0])
    with np.errstate(over="ignore", invalid="ignore"):
        actions = np.concatenate(parts, axis=-1).astype(np.float32, copy=True)
    if not np.isfinite(actions).all():
        raise ValueError("Isaac actions cannot be represented as float32")
    # Native decode_action has already unnormalized and added the current joints.
    # Preserve gripper, joint order and all 16 decoded commands without rebasing.
    return actions


class IsaacGr00tBackend:
    def __init__(self, policy):
        self.policy = policy
        self.lock = Lock()

    def __call__(self, payload):
        observation = observation_to_inputs(payload)
        with self.lock:
            response = self.policy.get_action(observation)
        return decoded_to_actions(response)


def build_backend(profile):
    inspect_checkpoint(profile)
    root = verify_runtime_sources(profile)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    sys.path.insert(0, str(root.parent))
    import torch
    import gr00t
    if Path(gr00t.__file__).resolve().parent != root.resolve():
        raise RuntimeError("Imported GR00T package is not the reviewed native source")
    from gr00t.policy import gr00t_policy
    from gr00t.model.modules import qwen3_backbone
    from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7
    if not torch.cuda.is_available():
        raise RuntimeError("Isaac-GR00T requires CUDA; no silent CPU fallback")
    with local_native_loaders(gr00t_policy, qwen3_backbone, processing_gr00t_n1d7,
                              profile["checkpoint"], profile["cosmos_snapshot"], torch.bfloat16):
        policy = gr00t_policy.Gr00tPolicy(embodiment_tag=EMBODIMENT,
                                        model_path=profile["checkpoint"], device="cuda:0", strict=True)
    return IsaacGr00tBackend(policy)


if __name__ == "__main__":
    print(json.dumps(dependency_inventory(), indent=2))

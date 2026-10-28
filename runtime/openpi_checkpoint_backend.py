"""OpenPI task-checkpoint boundary; no legacy DROID velocity conversions.

These checkpoints are Orbax/JAX exports, not the LeRobot PI05 safetensors format.
Importing this module does not import JAX, OpenPI, or initialize a device.
"""
from __future__ import annotations

import hashlib
import dataclasses
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
from threading import Lock
from urllib.parse import urlparse

import numpy as np

from drawer_bridge import observation_to_batch


def dependency_inventory():
    """Locate installed packages without importing the ML stack or using CUDA."""
    result = {"python": sys.executable, "modules": {}, "packages": {}}
    for name in ("openpi", "jax", "jaxlib", "flax", "orbax", "sentencepiece"):
        spec = importlib.util.find_spec(name)
        result["modules"][name] = None if spec is None else {
            "origin": spec.origin,
            "search_locations": list(spec.submodule_search_locations or []),
        }
    for name in ("openpi", "jax", "jaxlib", "flax", "orbax-checkpoint", "sentencepiece"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    result["gpu_initialized"] = False
    return result


def _asset(checkpoint, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Checkpoint assets must be local relative paths")
    path = checkpoint / relative
    if not path.is_file():
        raise ValueError("Missing checkpoint asset: " + name)
    if not path.resolve().is_relative_to(checkpoint.resolve()):
        raise ValueError("Checkpoint asset resolves outside the selected directory")
    return path


def validate_statistics(document):
    """Keep the package's own joint-delta/gripper statistics, including constants."""
    stats = document.get("norm_stats", {})
    for kind in ("state", "actions"):
        values = {}
        for key in ("mean", "std", "q01", "q99"):
            value = np.asarray(stats.get(kind, {}).get(key))
            if value.shape != (8,) or value.dtype.kind not in "fi" or not np.isfinite(value).all():
                raise ValueError(f"OpenPI {kind}.{key} must contain eight finite values")
            values[key] = value
        if (values["std"] < 0).any() or (values["q01"] > values["q99"]).any():
            raise ValueError("Invalid OpenPI normalization ranges")
    # A checkpoint can have a constant gripper. Zero standard deviation and equal
    # quantile endpoints are valid and remain owned by the native normalizer.


def inspect_checkpoint(profile):
    """Validate small assets only; the Orbax tensor tree still needs native restore."""
    if profile.get("backend") != "openpi" or profile.get("action_space") != "joint_position":
        raise ValueError("OpenPI task checkpoints require decoded joint_position")
    if profile.get("action_dim") != 8:
        raise ValueError("DROID task checkpoints require seven joints and one gripper")
    checkpoint = Path(profile["checkpoint"])
    if not checkpoint.is_dir() or not (checkpoint / "params").is_dir():
        raise ValueError("OpenPI checkpoint must contain its local params directory")
    if next(checkpoint.rglob("*.incomplete"), None):
        raise ValueError("OpenPI checkpoint download is incomplete")
    task = profile.get("dataset_id")
    if (not isinstance(task, str) or not task.strip() or Path(task).is_absolute()
            or '..' in Path(task).parts or '\\' in task):
        raise ValueError("dataset_id must be the relative checkpoint asset directory")
    required = {"train_config.py", f"assets/{task}/norm_stats.json"}
    digests = profile.get("asset_sha256", {})
    if not required.issubset(digests):
        raise ValueError("Reviewed task configuration/statistics hashes are required")
    for name, digest in digests.items():
        if hashlib.sha256(_asset(checkpoint, name).read_bytes()).hexdigest() != digest:
            raise ValueError("OpenPI checkpoint asset changed: " + name)
    validate_statistics(json.loads(_asset(checkpoint, f"assets/{task}/norm_stats.json").read_text()))
    return {"checkpoint": str(checkpoint), "dataset_id": task,
            "configuration_hashes_verified": True, "weights_loaded": False,
            "full_weight_hash_verified": False, "native_runtime_verified": False}


def local_asset_resolver(tokenizer):
    """Replace OpenPI's downloading hook inside this independent process only."""
    tokenizer = Path(tokenizer)
    if not tokenizer.is_file():
        raise ValueError("A local Paligemma tokenizer.model is required")
    def resolve(url, **_kwargs):
        if str(url) == "gs://big_vision/paligemma_tokenizer.model":
            path = tokenizer
        elif Path(str(url)).is_absolute():
            path = Path(url)
        elif urlparse(str(url)).scheme:
            raise FileNotFoundError("Online OpenPI asset access is disabled: " + str(url))
        else:
            path = Path(url)
        if not path.exists():
            raise FileNotFoundError(path)
        return path.resolve()
    return resolve


def verify_runtime_sources(profile):
    if not profile.get("openpi_source_root"):
        raise ValueError("Reviewed OpenPI runtime source root is required")
    root = Path(profile["openpi_source_root"])
    required = {"models/pi0_config.py", "models/model.py", "policies/policy_config.py",
                "policies/policy.py", "policies/droid_policy.py", "training/config.py", "transforms.py"}
    hashes = profile.get("native_source_sha256", {})
    if set(hashes) != required:
        raise ValueError("Reviewed OpenPI runtime source hashes are required")
    for name, digest in hashes.items():
        if hashlib.sha256(_asset(root, name).read_bytes()).hexdigest() != digest:
            raise ValueError("OpenPI native runtime source changed: " + name)
    return root


def strict_model_config(config, native_class):
    """Use native shape/tree validation without silently dropping extra weights."""
    class StrictPi0Config(native_class):
        def load(self, params, *, remove_extra_params=False):
            if remove_extra_params:
                raise ValueError("Extra Orbax parameters must not be silently discarded")
            return super().load(params, remove_extra_params=False)
    return StrictPi0Config(**{field.name: getattr(config, field.name)
                            for field in dataclasses.fields(config) if field.init})


def validate_native_pipeline(config, data, module, profile, transforms, droid_policy):
    """Check runtime defaults and the single relative-to-absolute transform pair."""
    if (config.model.pi05 is not True or config.model.action_horizon != 50
            or profile.get("action_steps") != 50 or config.model.action_dim != 32
            or config.data.repo_id != profile["dataset_id"]):
        raise ValueError("OpenPI task runtime horizon/model/dataset differs from the reviewed export")
    if data.asset_id != profile["dataset_id"] or data.use_quantile_norm is not True:
        raise ValueError("OpenPI task must use its own quantile normalization assets")
    inputs, outputs = data.data_transforms.inputs, data.data_transforms.outputs
    input_types = (module._SplitDroidState, droid_policy.DroidInputs, transforms.DeltaActions)
    output_types = (transforms.AbsoluteActions, droid_policy.DroidOutputs)
    if (len(inputs) != 3 or len(outputs) != 2
            or any(not isinstance(value, expected) for value, expected in zip(inputs, input_types))
            or any(not isinstance(value, expected) for value, expected in zip(outputs, output_types))):
        raise ValueError("Unexpected OpenPI DROID relative-action transform order")
    expected_mask = np.array([True] * 7 + [False])
    if not np.array_equal(inputs[-1].mask, expected_mask) or not np.array_equal(outputs[0].mask, expected_mask):
        raise ValueError("Only the seven arm joints may use relative actions")
    if inputs[1].model_type != config.model.model_type or data.model_transforms.outputs:
        raise ValueError("Unexpected OpenPI model-specific output transform")


def observation_to_inputs(payload):
    """Input to the task's _SplitDroidState then native DroidInputs pipeline."""
    batch = observation_to_batch(payload)
    return {
        "observation/exterior_image_1_left": batch["observation.images.left_image"].transpose(1, 2, 0).copy(),
        "observation/wrist_image_left": batch["observation.images.wrist"].transpose(1, 2, 0).copy(),
        "state": batch["observation.state"].copy(),
        "prompt": batch["task"],
    }


def decoded_to_actions(response, steps):
    """Accept final native AbsoluteActions/DroidOutputs without a second rebase."""
    if type(steps) is not int or steps < 1:
        raise ValueError("Native OpenPI action horizon must be verified before use")
    if not isinstance(response, dict) or "actions" not in response:
        raise ValueError("Native OpenPI response must contain actions")
    value = np.asarray(response["actions"])
    if value.shape != (steps, 8) or value.dtype.kind != "f" or not np.isfinite(value).all():
        raise ValueError(f"Native OpenPI actions must be finite float ({steps},8)")
    with np.errstate(over="ignore", invalid="ignore"):
        result = value.astype(np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("OpenPI actions cannot be represented as float32")
    return result


class OpenPiBackend:
    def __init__(self, infer, steps):
        if type(steps) is not int or steps < 1:
            raise ValueError("A verified native action horizon is required")
        self.infer = infer
        self.steps = steps
        self.lock = Lock()

    def __call__(self, payload):
        inputs = observation_to_inputs(payload)
        with self.lock:
            response = self.infer(inputs)
        return decoded_to_actions(response, self.steps)


def build_backend(profile):
    """Load the reviewed task CONFIG through native OpenPI's complete policy chain."""
    inspect_checkpoint(profile)
    source = verify_runtime_sources(profile)
    resolver = local_asset_resolver(profile["tokenizer"])
    checkpoint = Path(profile["checkpoint"])
    # Force the reviewed Orbax route; do not accidentally load a second format.
    if (checkpoint / "model.safetensors").exists():
        raise ValueError("Ambiguous OpenPI checkpoint also contains PyTorch weights")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", XLA_PYTHON_CLIENT_PREALLOCATE="false",
                      JAX_PLATFORMS="cuda")
    sys.path.insert(0, str(source.parent))
    from openpi.shared import download
    download.maybe_download = resolver
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.policies import droid_policy, policy_config
    import jax

    # The checkpoint-owned Python config was hash checked before executing it.
    name = "_vla_openpi_task_" + profile["dataset_id"]
    spec = importlib.util.spec_from_file_location(name, checkpoint / "train_config.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolves annotated fields via this entry.
    spec.loader.exec_module(module)
    config = module.CONFIG
    assets = dataclasses.replace(config.data.assets, assets_dir=str(checkpoint / "assets"),
                                 asset_id=profile["dataset_id"])
    config = dataclasses.replace(config, data=dataclasses.replace(config.data, assets=assets),
                                 model=strict_model_config(config.model, Pi0Config))
    data = config.data.create(config.assets_dirs, config.model)
    validate_native_pipeline(config, data, module, profile, transforms, droid_policy)
    devices = jax.devices("gpu")  # No silent fallback to CPU execution.
    if not devices:
        raise RuntimeError("OpenPI JAX did not expose a GPU device")
    with jax.default_device(devices[0]):
        policy = policy_config.create_trained_policy(config, checkpoint)
    return OpenPiBackend(policy.infer, config.model.action_horizon)


if __name__ == "__main__":
    print(json.dumps(dependency_inventory(), indent=2))

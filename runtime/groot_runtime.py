"""Strict streamed LeRobot GR00T loading, extracted from the deployed runtime."""
import json
import os
from pathlib import Path
import struct

def read_header(path):
    with Path(path).open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError("truncated safetensors length")
        length = struct.unpack("<Q", length_bytes)[0]
        if not 2 <= length <= 8 * 1024 * 1024:
            raise ValueError("invalid safetensors header size")
        return json.loads(stream.read(length))


def strict_layout(model, header):
    """Require all state, except metadata-declared aliases tied in this actual model."""
    state = model.state_dict(keep_vars=True)
    actual = set(header) - {"__metadata__"}
    unexpected = actual - set(state)
    if unexpected:
        raise ValueError(f"unexpected checkpoint keys: {sorted(unexpected)}")
    aliases = header.get("__metadata__", {})
    missing = set(state) - actual
    for name in missing:
        target = aliases.get(name)
        if target not in actual or state[name] is not state[target]:
            raise ValueError(f"missing state or unverified weight tie: {name}")
    for name in actual:
        if list(state[name].shape) != header[name]["shape"]:
            raise ValueError(f"checkpoint shape mismatch: {name}")
        if header[name]["dtype"] != "F32" or str(state[name].dtype) != "torch.float32":
            raise ValueError(f"checkpoint/runtime dtype mismatch: {name}")
    return state, sorted(actual), sorted(missing)


def load_strict_streamed(model, path):
    """Copy one tensor at a time into a normally initialized CPU model.

    This retains real nonpersistent buffers and avoids a second complete weight
    dictionary in RAM. It deliberately does not load weights into a meta model.
    """
    import torch
    from safetensors import safe_open

    header = read_header(path)
    state, names, aliases = strict_layout(model, header)
    if any(t.device.type != "cpu" for t in state.values()):
        raise ValueError("streamed loading requires a real CPU model")
    with safe_open(str(path), framework="pt", device="cpu") as archive, torch.no_grad():
        for name in names:
            tensor = archive.get_tensor(name)
            state[name].copy_(tensor)
            del tensor
    return {"loaded_tensor_count": len(names), "verified_tied_aliases": aliases}


def offline_environment():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
                      NO_ALBUMENTATIONS_UPDATE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


def build_policy(model_path, architecture_path, *, meta=False):
    import torch
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from lerobot.policies.groot.groot_n1_7 import GR00TN17, GR00TN17Config, _tie_unused_qwen_lm_head

    torch.set_num_threads(1)
    config = GrootConfig.from_pretrained(model_path, local_files_only=True, cli_overrides=["--device=cpu"])
    architecture = json.loads(Path(architecture_path).read_text())

    class LocalCheckpointPolicy(GrootPolicy):
        def _create_groot_model(self):
            values = dict(architecture)
            for name in ("tune_llm", "tune_visual", "tune_projector", "tune_diffusion_model",
                         "tune_vlln", "tune_top_llm_layers", "use_flash_attention"):
                values[name] = getattr(self.config, name)
            for name in ("num_inference_timesteps", "rtc_ramp_rate"):
                if getattr(self.config, name) is not None:
                    values[name] = getattr(self.config, name)
            # Cosmos architecture is built from LeRobot's local static config.
            # No base weights or remote code are loaded by this constructor.
            model = GR00TN17(GR00TN17Config(**values), load_backbone_weights=False,
                            transformers_loading_kwargs={"local_files_only": True})
            _tie_unused_qwen_lm_head(model.backbone.model)
            if self.config.model_params_fp32:
                self._cast_model_parameters_to_fp32(model)
            return model

    with torch.device("meta" if meta else "cpu"):
        policy = LocalCheckpointPolicy(config)
    policy.requires_grad_(False)
    policy.eval()
    return policy, config


def build_processors(config, model_path, cosmos_path, device):
    from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors_from_pretrained

    if not cosmos_path.is_dir():
        raise ValueError("local Cosmos tokenizer/processor snapshot is missing")
    return make_groot_pre_post_processors_from_pretrained(
        config, str(model_path),
        preprocessor_overrides={
            "groot_n1_7_vlm_encode_v1": {"model_name": str(cosmos_path), "device": device},
            "device_processor": {"device": device},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )


def tensorize(batch):
    import numpy as np
    import torch
    return {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            for key, value in batch.items()}

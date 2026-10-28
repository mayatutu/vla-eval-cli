"""LeRobot GR00T adapter using the retained strict streaming loader."""
from pathlib import Path


def build_backend(profile):
    if profile['action_steps'] != 40 or profile['action_dim'] != 8:
        raise ValueError('This LeRobot GR00T loader requires 40 eight-dimensional actions')
    import groot_runtime as runtime
    from model_catalog import inspect_checkpoint
    inspect_checkpoint(profile)
    model_path = Path(profile['checkpoint'])
    architecture = Path(profile['groot_architecture'])
    cosmos = Path(profile['cosmos_snapshot'])
    if not architecture.is_file():
        raise FileNotFoundError(architecture)
    runtime.offline_environment()
    import numpy as np
    import torch
    from drawer_bridge import ChunkBridge
    free, _ = torch.cuda.mem_get_info()
    if free < 17 * 1024**3:
        raise RuntimeError('Less than 17 GiB GPU memory free; defer')
    policy, config = runtime.build_policy(model_path, architecture)
    runtime.load_strict_streamed(policy, model_path / 'model.safetensors')
    policy.to('cuda:0')
    pre, post = runtime.build_processors(config, model_path, cosmos, 'cuda:0')
    bridge = ChunkBridge(pre, policy.predict_action_chunk, post, runtime.tensorize)

    def predict(observation):
        with torch.inference_mode():
            response = bridge.get_action(observation)['action']
        return np.concatenate([response['joint_position'], response['gripper_position']], axis=-1)[0]

    return predict

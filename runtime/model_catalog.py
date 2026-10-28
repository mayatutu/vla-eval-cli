"""Configuration for task-local model tests; importing this never loads a model."""
import copy
import hashlib
import json
from pathlib import Path


BACKENDS = {'pi05', 'g05', 'lingbot', 'molmoact2', 'openpi',
            'molmoact2_lerobot', 'groot_lerobot', 'groot_isaac'}
DEFAULT_CATALOG = Path(__file__).resolve().parent.parent / 'config/models.local.json'


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate configuration key: {key}')
        result[key] = value
    return result


def read_catalog(path=DEFAULT_CATALOG):
    catalog = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_unique)
    if catalog.get('version') != 1 or not isinstance(catalog.get('models'), dict):
        raise ValueError('Expected catalog version 1 with models mapping')
    return catalog


def select_profile(catalog, model_id, *, control_hz=None):
    entry = catalog['models'][model_id]
    backend = entry['backend']
    if backend not in BACKENDS:
        raise ValueError(f'Unsupported model family: {backend}')
    profile = copy.deepcopy(catalog.get('defaults', {}))
    profile.update(copy.deepcopy(catalog.get('backends', {}).get(backend, {})))
    profile.update(copy.deepcopy(entry))
    profile['model_id'] = model_id
    if control_hz is not None:
        profile['control_hz'] = control_hz
    for field in ('checkpoint', 'revision', 'python', 'instruction'):
        if not isinstance(profile.get(field), str) or not profile[field].strip():
            raise ValueError(f'{model_id}: missing {field}')
    if profile.get('action_space') != 'joint_position' or profile.get('action_dim') != 8:
        raise ValueError('These backends require decoded absolute seven-joint + gripper targets')
    steps = profile.get('action_steps')
    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ValueError('action_steps must be a positive integer <= 1000')
    hz = profile.get('control_hz')
    if hz is not None and (type(hz) is not int or not 1 <= hz <= 100):
        raise ValueError('control_hz must be null or an integer in 1..100')
    cameras = profile.get('cameras', {})
    required_cameras = {'external_image', 'wrist_image'}
    if backend in {'lingbot', 'molmoact2', 'molmoact2_lerobot'}:
        required_cameras.add('external_right_image')
    if (set(cameras) != required_cameras
            or len(set(cameras.values())) != len(required_cameras)):
        raise ValueError('Distinct assignments for every checkpoint camera are required')
    return profile


def validate_runtime_profile(profile):
    """Reject unfilled templates before launching a service or robot."""
    import ipaddress
    from urllib.parse import urlsplit
    from evaluation_limits import action_seconds_for_profile
    def check(value, key='profile'):
        if isinstance(value, str) and ('<' in value or '>' in value):
            raise ValueError(f'Fill the placeholder at {key}: {value}')
        if isinstance(value, dict):
            for k, v in value.items():
                check(v, key + '.' + k)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                check(v, f'{key}[{i}]')
    check(profile)
    action_seconds_for_profile(profile)
    ipaddress.ip_address(profile['nuc_ip'])
    parts = urlsplit(profile['bind'])
    if parts.scheme != 'tcp' or not parts.hostname or not parts.port:
        raise ValueError('bind must be tcp://HOST:PORT reachable from both computers')
    if profile.get('control_hz') is None:
        raise ValueError('Set control_hz from the model training/control specification')
    cameras = profile.get('recording_cameras', {})
    if (set(cameras) != {'external_image', 'wrist_image', 'external_right_image'}
            or any(not isinstance(v, str) or not v.isdigit() for v in cameras.values())
            or len(set(cameras.values())) != 3):
        raise ValueError('recording_cameras requires three distinct numeric camera IDs')
    if any(cameras.get(k) != v for k, v in profile['cameras'].items()):
        raise ValueError('Model input camera IDs must match recording_cameras')
    from pathlib import PurePosixPath
    for key in ('python', 'checkpoint', 'camera_config'):
        if not PurePosixPath(profile[key]).is_absolute():
            raise ValueError(f'{key} must be an absolute Linux path')
    return profile


def contract(profile):
    fields = ('model_id', 'backend', 'revision', 'action_space', 'action_dim',
              'action_steps', 'control_hz')
    result = {key: profile[key] for key in fields}
    # Include model paths, processors and camera assignments so two different
    # configurations cannot silently share the same human-readable model id.
    encoded = json.dumps(profile, sort_keys=True, separators=(',', ':')).encode()
    result['profile_sha256'] = hashlib.sha256(encoded).hexdigest()
    return result


def inspect_checkpoint(profile):
    """Small-file/size inspection only. No tensor deserialization or GPU calls."""
    folder = Path(profile['checkpoint'])
    if profile['backend'] == 'openpi':
        from openpi_checkpoint_backend import inspect_checkpoint as inspect_openpi
        return {'contract': contract(profile), **inspect_openpi(profile)}
    if profile['backend'] == 'groot_isaac':
        from groot_isaac_checkpoint_backend import inspect_checkpoint as inspect_isaac
        return {'contract': contract(profile), **inspect_isaac(profile)}
    if profile['backend'] in {'lingbot', 'molmoact2'}:
        return inspect_sharded_checkpoint(profile, folder)
    required = ([profile.get('g05_weights_file', 'checkpoints/model_state_dict.pt'), '.hydra/config.yaml',
                 'dataset_stats.json', 'action_tokenizer.pt'] if profile['backend'] == 'g05'
                else ['model.safetensors', 'config.json', 'policy_preprocessor.json',
                      'policy_postprocessor.json'])
    for name in required:
        if not (folder / name).is_file() or (folder / name).stat().st_size == 0:
            raise ValueError(f'Missing/incomplete checkpoint file: {folder / name}')
    partial = next(folder.rglob('*.incomplete'), None)
    if partial:
        raise ValueError(f'Download is incomplete: {partial}')
    for name, digest in profile.get('asset_sha256', {}).items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Checkpoint configuration changed: ' + name)
    mirror = folder / 'mirror-source.json'
    if mirror.exists():
        source = json.loads(mirror.read_text())
        if source['revision'] != profile['revision']:
            raise ValueError('Checkpoint source revision differs from selected profile')
        for spec in source['files']:
            path = folder / spec['path']
            # ModelScope may regenerate Git transfer metadata. Its size does
            # not describe model content; require the file to exist. Keep all
            # other source sizes, including README, strict.
            if not path.is_file() or (spec['path'] != '.gitattributes'
                                      and path.stat().st_size != spec['size']):
                raise ValueError(f'Checkpoint file size mismatch: {path}')
    if profile['backend'] != 'g05':
        config = json.loads((folder / 'config.json').read_text())
        expected_type = {'groot_lerobot': 'groot', 'molmoact2_lerobot': 'molmoact2'}.get(
            profile['backend'], profile['backend'])
        if config.get('type') != expected_type:
            raise ValueError('Checkpoint model type differs from selected backend')
        if config['output_features']['action']['shape'] != [8]:
            raise ValueError('Checkpoint does not declare an eight-dimensional action')
        if profile['action_steps'] > config['chunk_size']:
            raise ValueError('Configured action_steps exceeds checkpoint chunk_size')
        for filename in ('policy_preprocessor.json', 'policy_postprocessor.json'):
            steps = json.loads((folder / filename).read_text())['steps']
            for step in steps:
                if 'state_file' in step and not (folder / step['state_file']).is_file():
                    raise ValueError(f'Missing processor state: {step["state_file"]}')
    return {'contract': contract(profile), 'required_files': required,
            'weights_loaded': False, 'full_weight_hash_verified': False}


def inspect_sharded_checkpoint(profile, folder):
    """Check the complete tensor index and file bounds without reading weights."""
    import struct
    required = ['config.json', 'norm_stats.json', 'model.safetensors.index.json',
                'tokenizer.json', 'tokenizer_config.json']
    if profile['backend'] == 'lingbot':
        required += ['lingbotvla_cli.yaml', profile['robot_config'], 'preprocessor_config.json']
    else:
        required += ['processor_config.json', 'modeling_molmoact2.py',
                     'configuration_molmoact2.py', 'processing_molmoact2.py']
    for name in required:
        if not (folder / name).is_file() or (folder / name).stat().st_size == 0:
            raise ValueError(f'Missing checkpoint asset: {name}')
    if next(folder.rglob('*.incomplete'), None):
        raise ValueError('Download is incomplete')
    for name, digest in profile.get('asset_sha256', {}).items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Checkpoint configuration changed: {name}')
    config = json.loads((folder / 'config.json').read_text())
    expected_type = 'lingbotvla' if profile['backend'] == 'lingbot' else 'molmoact2'
    if config.get('model_type') != expected_type:
        raise ValueError('Checkpoint architecture differs from selected backend')
    index = json.loads((folder / 'model.safetensors.index.json').read_text())
    mapping = index['weight_map']
    if not mapping:
        raise ValueError('Empty checkpoint weight index')
    actual = {}
    for name in sorted(set(mapping.values())):
        if Path(name).name != name:
            raise ValueError('Unsafe checkpoint shard path')
        path = folder / name
        with path.open('rb') as stream:
            header_size = struct.unpack('<Q', stream.read(8))[0]
            if not 0 < header_size <= 16 * 1024**2:
                raise ValueError('Invalid checkpoint header size')
            header = json.loads(stream.read(header_size))
        data_size = path.stat().st_size - 8 - header_size
        for key, tensor in header.items():
            if key == '__metadata__':
                continue
            start, end = tensor['data_offsets']
            if not 0 <= start <= end <= data_size or key in actual:
                raise ValueError('Truncated/duplicate checkpoint tensor: ' + key)
            actual[key] = name
    if actual != mapping:
        raise ValueError('Checkpoint shards do not match the complete weight index')
    return {'contract': contract(profile), 'required_files': required,
            'checkpoint_tensors': len(actual), 'weights_loaded': False,
            'full_weight_hash_verified': False}

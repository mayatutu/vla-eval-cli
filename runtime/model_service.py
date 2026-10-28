"""One independent transport for model families; no production server changes."""
import json
from pathlib import Path
import sys
import time

from model_catalog import contract, inspect_checkpoint


def configure_transport_path():
    """Expose only the task-local transport built for this Python version."""
    major, minor = sys.version_info[:2]
    transport = Path(__file__).resolve().parent / f'transport-py{major}{minor}'
    if transport.is_dir() and str(transport) not in sys.path:
        sys.path.insert(0, str(transport))


def infer_saved(profile, observation, output, instruction):
    """Run three saved-observation predictions without opening any network port."""
    import numpy as np
    from drawer_bridge import saved_observation_to_payload
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    inspected = inspect_checkpoint(profile)
    with np.load(observation, allow_pickle=False) as saved:
        if 'external_cam' in saved:
            payload = saved_observation_to_payload(observation, instruction)
            pose = saved['cartesian_position'].copy()
        else:
            # Existing station replay: camera names are fixed by its provenance.
            payload = {
                'video': {'exterior_image_1_left': saved['left_image'][None, None].copy(),
                          'wrist_image_left': saved['head_image'][None, None].copy()},
                'state': {'joint_position': saved['joint_position'][None, None].copy(),
                          'gripper_position': saved['gripper_position'][None, None].copy()},
                'language': {'annotation.language.language_instruction': [[instruction]]}}
            pose = saved['cartesian_position'].copy()
        if 'external_right_image' in profile['cameras']:
            right_key = 'external_right_cam' if 'external_cam' in saved else 'right_image'
            if right_key not in saved:
                raise ValueError('This model requires a recorded right exterior image')
            payload['video']['exterior_image_2_left'] = saved[right_key][None, None].copy()
    payload['state']['cartesian_position'] = pose[None, None]
    predict = build_backend(profile)
    responses, seconds = [], []
    request = {'endpoint': 'get_action', 'contract': contract(profile),
               'data': {'observation': payload, 'options': None}}
    for _ in range(3):
        start = time.monotonic()
        responses.append(dispatch(request, profile, predict)['actions'])
        seconds.append(time.monotonic() - start)
    report = {'contract': contract(profile), 'inference_seconds': seconds,
              'action_shapes': [list(actions.shape) for actions in responses],
              'all_finite': True, 'robot_actions_sent': 0,
              'hardware_tested': False, 'weights_loaded': True,
              'full_weight_hash_verified': inspected['full_weight_hash_verified']}
    output.mkdir(parents=True)
    np.savez(output / 'actions.npz', **{f'prediction_{i}': value for i, value in enumerate(responses)})
    (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    return report


def build_backend(profile):
    if profile['backend'] == 'g05':
        from g05_checkpoint_backend import build_backend as build
        return build(profile)
    if profile['backend'] == 'pi05':
        from pi05_checkpoint_backend import build_backend as build
        return build(profile)
    if profile['backend'] == 'lingbot':
        from lingbot_checkpoint_backend import build_backend as build
        return build(profile)
    if profile['backend'] == 'molmoact2':
        from molmo_checkpoint_backend import build_backend as build
        return build(profile)
    if profile['backend'] == 'openpi':
        from openpi_checkpoint_backend import build_backend as build
        return build(profile)
    if profile['backend'] == 'molmoact2_lerobot':
        from molmo_lerobot_checkpoint_backend import build_backend as build
        return build(profile)
    if profile['backend'] == 'groot_lerobot':
        from groot_lerobot_checkpoint_backend import build_backend as build
        return build(profile)
    if profile['backend'] == 'groot_isaac':
        from groot_isaac_checkpoint_backend import build_backend as build
        return build(profile)
    raise ValueError(f'Unsupported model family: {profile["backend"]}')


def dispatch(request, profile, predict):
    import numpy as np
    expected = contract(profile)
    if not isinstance(request, dict):
        raise ValueError('Request must be a mapping')
    if request.get('endpoint') == 'metadata':
        return {'contract': expected}
    if request.get('endpoint') != 'get_action' or request.get('contract') != expected:
        raise ValueError('Requested model/configuration does not match loaded service')
    data = request.get('data')
    if not isinstance(data, dict) or data.get('options') not in (None, {}):
        raise ValueError('Unexpected inference options')
    actions = np.asarray(predict(data['observation']), dtype=np.float32)
    if actions.shape != (profile['action_steps'], 8) or not np.isfinite(actions).all():
        raise ValueError('Decoded action chunk has an invalid shape or nonfinite value')
    return {'contract': expected, 'actions': actions}


def serve(profile, bind, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    inspect_checkpoint(profile)
    # Transport helpers ship alongside this service in runtime/.
    configure_transport_path()
    import zmq
    from franka_vla_control.numpy_msgpack import packb, unpackb
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.MAXMSGSIZE, 32 * 1024**2)
    socket.setsockopt(zmq.SNDTIMEO, 5000)
    try:
        # Refuse an occupied port before allocating any model weights.
        socket.bind(bind)
        predict = build_backend(profile)
        output.mkdir(parents=True)
        (output / 'profile.json').write_text(json.dumps(profile, indent=2) + '\n')
        with (output / 'requests.jsonl').open('x') as log:
            print(json.dumps({'event': 'ready', 'address': bind, **contract(profile)}), flush=True)
            while True:
                if not socket.poll(1000):
                    continue
                started = time.monotonic()
                try:
                    response = dispatch(unpackb(socket.recv()), profile, predict)
                    event = 'served'
                except Exception as exc:
                    response = {'error': type(exc).__name__ + ': ' + str(exc),
                                'contract': contract(profile)}
                    event = 'request_failed'
                socket.send(packb(response))
                record = {'event': event, 'model_id': profile['model_id'],
                          'seconds': time.monotonic() - started}
                if event == 'request_failed':
                    record['error'] = response['error']
                log.write(json.dumps(record) + '\n')
                log.flush()
                print(json.dumps(record), flush=True)
    finally:
        socket.close(linger=0)
        context.term()

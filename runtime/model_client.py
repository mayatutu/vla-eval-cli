"""One task-local Client for configured absolute-joint model checkpoints.

Importing this module opens neither a socket nor a robot.  The production
Client is reused only after the selected model contract has been checked.
"""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np


def _dependencies(profile):
    from franka_vla_control import contracts, policy_runner
    from franka_vla_control.droid_molmo import CameraRoles
    from franka_vla_control.numpy_msgpack import packb, unpackb
    return SimpleNamespace(contracts=contracts, runner=policy_runner,
                           CameraRoles=CameraRoles, packb=packb, unpackb=unpackb)


def _make_robot(profile, action_space):
    from colosseum_client.robots.franka import DroidRobot
    # Configure the retained DROID client before importing its runtime.
    import os
    for key, value in {
        'VLA_NUC_IP': profile['nuc_ip'],
        'VLA_WRIST_CAMERA': profile['recording_cameras']['wrist_image'],
        'VLA_LEFT_CAMERA': profile['recording_cameras']['external_image'],
        'VLA_RIGHT_CAMERA': profile['recording_cameras']['external_right_image'],
    }.items():
        os.environ[key] = str(value)
    return DroidRobot(dict(profile['cameras']), action_space=action_space,
                      image_size=(640, 360), camera_backend='uvc',
                      camera_config=profile['camera_config'])



def _contract(profile):
    from model_catalog import contract
    return contract(profile)


class _ZmqTransport:
    """Bounded async request/reply transport, confined to one event loop."""
    def __init__(self, url, dependencies):
        import zmq
        import zmq.asyncio
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 240000)
        self.socket.setsockopt(zmq.SNDTIMEO, 30000)
        self.socket.setsockopt(zmq.MAXMSGSIZE, 32 * 1024**2)
        self.socket.connect(url)
        self.packb, self.unpackb = dependencies.packb, dependencies.unpackb

    async def request(self, request):
        await self.socket.send(self.packb(request))
        return self.unpackb(await self.socket.recv())

    def close(self):
        self.socket.close(linger=0)
        self.context.term()


def _check_contract(response, expected):
    if not isinstance(response, Mapping):
        raise ValueError('Model service response must be a mapping')
    if 'error' in response:
        raise RuntimeError('Model service error: ' + str(response['error']))
    if response.get('contract') != expected:
        raise ValueError('Model service contract mismatch; refusing robot execution')


def _actions(response, expected):
    _check_contract(response, expected)
    value = np.asarray(response.get('actions'))
    shape = (expected['action_steps'], expected['action_dim'])
    if value.shape != shape or value.dtype.kind not in 'fi' or not np.isfinite(value).all():
        raise ValueError(f'Model actions must be finite numeric {shape}')
    with np.errstate(over='ignore', invalid='ignore'):
        value = value.astype(np.float32, copy=True)
    if not np.isfinite(value).all():
        raise ValueError('Model actions cannot be represented as float32')
    return value


def _payload(dependencies, observation, instruction, roles, *, right_required=False):
    # "groot" selects the existing wire schema here, never the model identity
    # or action semantics.  Both are checked separately in the contract.
    payload = dependencies.runner.build_policy_observation('groot', observation, instruction, roles)
    pose = np.asarray(observation.cartesian_position, dtype=np.float32)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError('Cartesian observation must be finite shape (6,)')
    payload['state']['cartesian_position'] = pose[None, None, :].copy()
    if right_required:
        right = np.asarray(observation.images['external_right_image'])
        if right.dtype != np.uint8 or right.ndim != 3 or right.shape[-1] != 3:
            raise ValueError('Right exterior must be an RGB uint8 image')
        payload['video']['exterior_image_2_left'] = right[None, None].copy()
    return payload


def _load_three_view_shadow(path):
    with np.load(path, allow_pickle=False) as saved:
        if 'external_cam' in saved:
            state = saved['state'].copy()
            images = {'external_image': saved['external_cam'].copy(),
                      'external_right_image': saved['external_right_cam'].copy(),
                      'wrist_image': saved['wrist_cam'].copy()}
        else:
            state = np.concatenate((saved['joint_position'], saved['gripper_position']))
            images = {'external_image': saved['left_image'].copy(),
                      'external_right_image': saved['right_image'].copy(),
                      'wrist_image': saved['head_image'].copy()}
        pose = saved['cartesian_position'].copy()
    if state.shape != (8,) or pose.shape != (6,):
        raise ValueError('Saved state/Cartesian pose must be (8,)/(6,)')
    return SimpleNamespace(images=images, joints=state[:7], gripper=state[7:],
                           cartesian_position=pose)


def _new_path(path, label):
    if not path:
        raise ValueError(label + ' is required')
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    return path


async def _run(profile, *, url, instruction, steps, control_hz, execution_log,
               shadow_observation, shadow_report):
    expected = _contract(profile)
    if expected['action_space'] != 'joint_position' or expected['action_dim'] != 8:
        raise ValueError('This Client requires decoded absolute 8D joint targets')
    if type(expected['action_steps']) is not int or expected['action_steps'] <= 0:
        raise ValueError('action_steps must be a positive integer')
    if not isinstance(url, str) or not url.startswith('tcp://'):
        raise ValueError('The unified model service requires a tcp:// URL')
    instruction = instruction or profile.get('instruction')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('A nonempty instruction is required')
    is_shadow = shadow_observation is not None
    if is_shadow:
        output = _new_path(shadow_report, 'shadow_report')
        if execution_log is not None:
            raise ValueError('Shadow mode must not use a live execution log')
    else:
        if type(steps) is not int or steps <= 0:
            raise ValueError('steps must be a positive integer')
        control_hz = profile.get('control_hz') if control_hz is None else control_hz
        if (isinstance(control_hz, bool) or not isinstance(control_hz, (int, float))
                or not math.isfinite(control_hz) or control_hz <= 0):
            raise ValueError('A verified positive control_hz must be configured or supplied')
        output = _new_path(execution_log, 'execution_log')
        if shadow_report is not None:
            raise ValueError('Live mode must not use a shadow report')
    dependencies = _dependencies(profile)
    runner, contracts = dependencies.runner, dependencies.contracts
    roles = dependencies.CameraRoles('external_image', 'wrist_image')
    transport = _ZmqTransport(url, dependencies)
    robot = None
    try:
        _check_contract(await transport.request({'endpoint': 'metadata'}), expected)

        async def infer(observation, task):
            started = time.monotonic()
            reply = await transport.request({
                'endpoint': 'get_action', 'contract': expected,
                'data': {'observation': _payload(dependencies, observation, task, roles,
                          right_required='external_right_image' in profile['cameras']),
                         'options': None},
            })
            actions = _actions(reply, expected)
            return runner.PolicyResult(profile['model_id'], actions, reply,
                                       (time.monotonic() - started) * 1000)

        if is_shadow:
            if 'external_right_image' in profile['cameras']:
                observation = _load_three_view_shadow(shadow_observation)
            else:
                observation, _ = runner.load_policy_shadow_observation(shadow_observation)
            result = await infer(observation, instruction)
            report = {**result.report(), 'contract': expected, 'robot_actions_sent': 0}
            with output.open('x', encoding='utf-8') as stream:
                json.dump(report, stream, indent=2)
                stream.write('\n')
            return report

        spec = runner.ModelSpec(profile['model_id'], url, contracts.ControlSpec(
            contracts.ActionSpace.JOINT_POSITION, 8,
            'decoded absolute joint targets with DROID gripper position',
            expected['action_steps']))
        with output.open('x', encoding='utf-8') as stream:
            def audit(event):
                stream.write(json.dumps(event, separators=(',', ':')) + '\n')
                stream.flush()
            audit({'event': 'model_selected', 'model': profile['model_id'],
                   'contract': expected, 'instruction': instruction,
                   'control_hz': control_hz, 'max_control_steps': steps})
            robot = _make_robot(profile, spec.control.action_space.value)
            await runner.run_live_policy(infer, spec, robot, instruction,
                                         control_hz=control_hz, max_control_steps=steps,
                                         audit_callback=audit)
        return {'model': profile['model_id'], 'contract': expected,
                'execution_log': str(output), 'max_control_steps': steps}
    finally:
        try:
            if robot is not None:
                robot.close()
        finally:
            transport.close()


def run_client(profile, *, url, instruction, steps, control_hz, execution_log,
               shadow_observation=None, shadow_report=None):
    """Run an explicitly selected shadow check or user-authorized Local Client."""
    return asyncio.run(_run(profile, url=url, instruction=instruction, steps=steps,
                            control_hz=control_hz, execution_log=execution_log,
                            shadow_observation=shadow_observation,
                            shadow_report=shadow_report))

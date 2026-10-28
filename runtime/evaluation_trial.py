"""Isolated trial recording around the already validated model Client.

Importing this module never connects to a model, camera or robot. Live execution
is only performed by an explicit call to :func:`run_trial`.
"""
from __future__ import annotations

import asyncio
from contextlib import suppress
import ipaddress
import json
import math
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np

from evaluation_limits import action_seconds_for_profile


def recording_cameras(profile):
    cameras = profile.get('recording_cameras', {})
    required = {'wrist_image', 'external_image', 'external_right_image'}
    if (set(cameras) != required or any(not isinstance(v, str) or not v for v in cameras.values())
            or len(set(cameras.values())) != 3):
        raise ValueError('recording_cameras must assign three distinct camera IDs')
    return dict(cameras)


RECORDING_NAMES = {
    'wrist_image': 'head_image',
    'external_image': 'left_image',
    'external_right_image': 'right_image',
}
NORMAL_ENDINGS = {'step_limit', 'early_finish'}
EXPORT_ELIGIBLE_ENDINGS = NORMAL_ENDINGS | {'time_limit'}
MAX_ACTION_SECONDS = 30.0
RECORDING_FPS = 30
CONTINUOUS_SCHEMA = 'continuous-state-video-v1'
ACTION_SEMANTICS = (
    'recorded action is the selected absolute joint/gripper command passed to '
    'DroidRobot.execute; it is not measured execution. The existing adapter '
    'thresholds the last (gripper) value to 1 when >0.5, otherwise 0; joint '
    'targets are unchanged. Measured pre-action state is recorded separately.'
)


class _StopTrial(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class _ActionTimer:
    """Count wall time from the first model command, excluding initialization."""
    def __init__(self, max_action_seconds=MAX_ACTION_SECONDS):
        self.max_action_seconds = max_action_seconds
        self.first_action_at = None
        self.elapsed = 0.0

    def check(self):
        if (self.first_action_at is not None
                and time.monotonic() - self.first_action_at >= self.max_action_seconds):
            raise _StopTrial('time_limit')

    def before_command(self):
        now = time.monotonic()
        if self.first_action_at is None:
            self.first_action_at = now
        elif now - self.first_action_at >= self.max_action_seconds:
            raise _StopTrial('time_limit')
        return now

    def stop(self):
        if self.first_action_at is not None:
            self.elapsed = max(0.0, time.monotonic() - self.first_action_at)


def _check_stop(stop_requested, action_timer=None):
    reason = stop_requested()
    if reason is not None:
        if reason not in {'early_finish', 'paused'}:
            raise ValueError('stop_requested must return None, early_finish or paused')
        if reason == 'paused':
            raise _StopTrial(reason)
    if action_timer is not None:
        action_timer.check()
    if reason is not None:
        raise _StopTrial(reason)


def _json_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _load_client():
    # The launcher selects the existing suite path; no production file is edited.
    import model_client
    return model_client


def _recorder_class():
    from colosseum_client.recording import TrialRecorder
    return TrialRecorder


def _export_function():
    from colosseum_client.lerobot_export import export_trial
    return export_trial


class _ContinuousSampling:
    """Own separate camera-cache readers and one read-only state connection."""
    def __init__(self, robot, path):
        from evaluation_video import ContinuousCameraRecorder
        from evaluation_state import StateRecorder
        # Reuse the actual station endpoint, never construct another RobotEnv or
        # ServerInterface: their constructors can launch/reset the controller.
        address = str(ipaddress.ip_address(robot.environment._robot.ip_address))
        if ':' in address:
            address = '[' + address + ']'
        self.video = ContinuousCameraRecorder(robot, path, {
            RECORDING_NAMES[name]: serial for name, serial in robot.cameras.items()})
        self.state = StateRecorder(Path(path) / 'state', endpoint=f'tcp://{address}:4242',
                                   hz=RECORDING_FPS, rpc_timeout=1.0)
        self.started = False

    def start(self):
        if self.started:
            return
        self.started = True
        self.video.start()
        self.state.start()
        self.video.check_ready(timeout=2.0)
        self.state.wait_ready(timeout=2.0)
        self.check_error()

    def check_error(self):
        if self.started:
            self.video.check_error()
            self.state.check_error()

    def stop_capture(self, *, through=None):
        # Always try both joins, even when a camera or state RPC has failed.
        errors = []
        if self.started:
            try:
                if through is not None:
                    # Wait only for observation coverage after commands have
                    # stopped. UVC processing can lag its driver timestamp.
                    # The exported horizon still ends at the action stop time.
                    deadline = time.monotonic() + 1.0
                    while True:
                        self.check_error()
                        cameras = self.video.status()['cameras'].values()
                        latest_state = self.state.status()['last_sample_monotonic']
                        if (latest_state is not None and latest_state >= through
                                and all(row['last_driver_time'] is not None
                                        and row['last_driver_time'] >= through for row in cameras)):
                            break
                        if time.monotonic() >= deadline:
                            raise RuntimeError('Continuous samples did not cover the final action time')
                        time.sleep(0.005)
            except BaseException as error:
                errors.append(str(error))
            for stop in (self.video.stop_capture, self.state.stop):
                try:
                    stop()
                except BaseException as error:
                    errors.append(str(error))
        if errors:
            raise RuntimeError('; '.join(errors))

    def finish(self):
        if not self.started:
            return {'complete': False, 'started': False}
        self.video.finish()
        self.check_error()
        return {'complete': True, 'started': True, 'video': self.video.status(),
                'state': self.state.status()}


def _continuous_sampling(robot, path):
    return _ContinuousSampling(robot, path)


def _validated_recording_observation(observation):
    """Freeze an actual pre-action sample, before execute can change any buffers."""
    images = {}
    for source, destination in RECORDING_NAMES.items():
        pixels = np.asarray(observation.images[source])
        if (pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3
                or not pixels.size):
            raise ValueError(f'{source} must contain a nonempty RGB uint8 image')
        images[destination] = pixels.copy()
    vectors = {}
    for key, size in [('joints', 7), ('gripper', 1), ('cartesian_position', 6)]:
        value = np.asarray(getattr(observation, key), dtype=np.float32)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f'Recorded {key} must be finite shape ({size},)')
        vectors[key] = value.copy()
    return SimpleNamespace(images=images, **vectors)


def _recording_state_observation(robot):
    """Read a fresh pre-action state without resizing/saving three images twice."""
    from evaluation_state import validate_state
    state = validate_state(robot.environment.get_state())
    return SimpleNamespace(images={}, **{
        name: np.asarray(state[name], dtype=np.float32)
        for name in ('joints', 'gripper', 'cartesian_position')})


class _RecordingRobot:
    """Keep native chunk inference and capture real state before every command."""
    def __init__(self, robot, recorder, stop_requested, audit, action_timer, sampling=None):
        self.robot = robot
        self.recorder = recorder
        self.stop_requested = stop_requested
        self.audit = audit
        self.action_timer = action_timer
        self.action_dim = robot.action_dim
        self.steps_sent = 0
        self.sampling = sampling

    def get_observation(self):
        _check_stop(self.stop_requested, self.action_timer)
        if self.sampling is not None:
            self.sampling.check_error()
        return self.robot.get_observation()

    def execute(self, action):
        _check_stop(self.stop_requested, self.action_timer)
        command = np.asarray(action, dtype=np.float32).copy()
        if command.shape != (8,) or not np.isfinite(command).all():
            raise ValueError('Selected command must be finite shape (8,)')
        if self.sampling is not None:
            # First inference may take minutes. Start recording only once a
            # validated model command is available, before its first send.
            self.sampling.start()
            self.sampling.check_error()
            _check_stop(self.stop_requested, self.action_timer)
        # Inference can take seconds. The old chunk-boundary observation is not
        # reused as the measured state for the first or subsequent actions.
        observation = (_recording_state_observation(self.robot) if self.sampling is not None
                       else _validated_recording_observation(self.robot.get_observation()))
        captured_at = time.monotonic()
        _check_stop(self.stop_requested, self.action_timer)
        adapter_target = command.copy()
        adapter_target[-1] = float(command[-1] > 0.5)
        step = self.steps_sent
        self.audit({'event': 'execute_attempt', 'control_step': step,
                    'captured_at_monotonic': captured_at,
                    'selected_command': command.tolist(),
                    'expected_adapter_command': adapter_target.tolist(),
                    'expected_adapter_command_semantics': 'existing DroidRobot gripper threshold; not measured execution'})
        execute_command = command.copy()
        # Capture/audit may block. Check again at the actual send boundary,
        # starting the clock only after the first action's preparation finishes.
        _check_stop(self.stop_requested, self.action_timer)
        sent_at = self.action_timer.before_command()
        try:
            self.robot.execute(execute_command)
        except BaseException as exc:
            self.audit({'event': 'execute_failed', 'control_step': step,
                        'error': type(exc).__name__ + ': ' + str(exc),
                        'physical_outcome': 'unknown; execute did not return successfully'})
            raise
        self.steps_sent += 1
        self.audit({'event': 'execute_succeeded', 'control_step': step,
                    'sent_at_monotonic': sent_at,
                    'returned_at_monotonic': time.monotonic(),
                    'meaning': 'adapter call returned; not proof of target attainment'})
        self.recorder.add(observation, command, captured_at=captured_at)
        # A blocking adapter call may return after the deadline. Preserve its
        # real observation/action pair, then stop even if it was the final step.
        _check_stop(self.stop_requested, self.action_timer)


async def _request_cancellable(transport, payload, stop_requested, action_timer=None,
                               check_recording=None):
    """Interrupt waiting for inference without stopping the shared model service."""
    _check_stop(stop_requested, action_timer)
    request = asyncio.create_task(transport.request(payload))
    try:
        while True:
            _check_stop(stop_requested, action_timer)
            if check_recording is not None:
                check_recording()
            ready, _ = await asyncio.wait({request}, timeout=0.05)
            _check_stop(stop_requested, action_timer)
            if ready:
                return await request
    finally:
        if not request.done():
            request.cancel()
            with suppress(asyncio.CancelledError):
                await request
        elif not request.cancelled():
            # Retrieve a raced failure without overriding an operator stop.
            request.exception()


def _disk_preflight(output, steps, max_action_seconds=MAX_ACTION_SECONDS):
    # Existing _make_robot uses three 640x360 RGB views. Keep room for the
    # uncompressed PNGs AND a conservative equally large export staging copy.
    # Keep original action-paired raw samples AND independent 30 Hz images.
    # Startup/stop margin is included, with an equally large export workspace.
    image_samples = steps + math.ceil((max_action_seconds + 5) * RECORDING_FPS)
    raw_bytes = image_samples * (3 * 640 * 360 * 3 + 8192)
    required = 2 * raw_bytes + 256 * 1024**2
    available = shutil.disk_usage(output).free
    if available < required:
        raise OSError(f'Insufficient recording disk space: need {required} bytes, have {available}')
    return {'required_bytes': required, 'available_bytes': available,
            'basis': f'original action samples plus {max_action_seconds + 5:g}s of three independent 30Hz RGB streams, export staging and 256 MiB margin'}


def _validate_arguments(profile, url, steps, instruction, output):
    if type(steps) is not int or steps < 1:
        raise ValueError('steps must be a positive integer')
    hz = profile.get('control_hz')
    if (isinstance(hz, bool) or not isinstance(hz, (float, int))
            or not math.isfinite(hz) or hz <= 0 or int(hz) != hz):
        raise ValueError('Configured control_hz must be a positive integral FPS')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('instruction must be nonempty')
    if not isinstance(url, str) or not url.startswith('tcp://'):
        raise ValueError('Model service URL must start with tcp://')
    cameras = profile.get('cameras', {})
    if not {'external_image', 'wrist_image'} <= set(cameras):
        raise ValueError('Model camera assignments are missing')
    if any(recording_cameras(profile).get(name) != serial for name, serial in cameras.items()):
        raise ValueError('Model cameras differ from this station recording camera mapping')
    if not output.is_dir():
        raise FileNotFoundError('Parent must create a unique attempt directory: ' + str(output))
    for name in ('actions.jsonl', 'recording', 'trial-result.json', 'export-context.json'):
        if (output / name).exists():
            raise FileExistsError(output / name)


async def _run_trial(profile, *, url, steps, instruction, output, stop_requested):
    output = Path(output)
    instruction = instruction or profile.get('instruction')
    _validate_arguments(profile, url, steps, instruction, output)
    client = _load_client()
    expected = client._contract(profile)
    if (expected['action_space'] != 'joint_position' or expected['action_dim'] != 8
            or type(expected['action_steps']) is not int or expected['action_steps'] < 1):
        raise ValueError('Selected contract must contain absolute 8D joint action chunks')
    dependencies = client._dependencies(profile)
    runner, contracts = dependencies.runner, dependencies.contracts
    roles = dependencies.CameraRoles('external_image', 'wrist_image')
    transport = robot = recorder = wrapped = sampling = None
    sampling_status = None
    reason, errors = 'error', []
    recording = output / 'recording'
    started_at = time.time()
    max_action_seconds = action_seconds_for_profile(profile)
    action_timer = _ActionTimer(max_action_seconds)
    with (output / 'actions.jsonl').open('x', encoding='utf-8') as stream:
        def audit(event):
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()

        try:
            _check_stop(stop_requested)
            disk = _disk_preflight(output, steps, max_action_seconds)
            transport = client._ZmqTransport(url, dependencies)
            metadata = await _request_cancellable(transport, {'endpoint': 'metadata'}, stop_requested)
            client._check_contract(metadata, expected)
            _check_stop(stop_requested)
            policy_record = {
                'model': profile['model_id'], 'contract': expected,
                'instruction': instruction, 'control_hz': profile['control_hz'],
                'max_control_steps': steps, 'model_cameras': dict(profile['cameras']),
                'max_action_seconds': max_action_seconds,
                'requires_manual_outcome': True,
                'action_time_semantics': 'wall time from immediately before the first model command; later inference/capture included, initialization excluded',
                'recording_cameras': recording_cameras(profile), 'recording_names': RECORDING_NAMES,
                'disk_preflight': disk, 'recording_queue_size': 32,
                'action_semantics': ACTION_SEMANTICS,
                'observation_semantics': 'one newly captured real pre-action sample per successfully returned execute call',
                'recording_schema': CONTINUOUS_SCHEMA, 'recording_fps': RECORDING_FPS,
                'training_sampling': 'independent real state RPC and camera sequences; export uses nearest recorded sources, marks reuse, and reports timing diagnostics',
            }
            spec = runner.ModelSpec(profile['model_id'], url, contracts.ControlSpec(
                contracts.ActionSpace.JOINT_POSITION, 8, ACTION_SEMANTICS, expected['action_steps']))
            # Add the third view only to robot capture, never to the contract or
            # to the original profile passed to model payload construction.
            capture_profile = {**profile, 'cameras': recording_cameras(profile)}
            _check_stop(stop_requested)
            robot = client._make_robot(capture_profile, spec.control.action_space.value)
            sampling = _continuous_sampling(robot, recording)
            action_cameras = [] if sampling is not None else list(RECORDING_NAMES.values())
            recorder = _recorder_class()(recording, action_cameras, queue_size=32)
            policy_record['action_recording_images'] = sampling is None
            _json_new(recording / 'policy.json', policy_record)
            wrapped = _RecordingRobot(robot, recorder, stop_requested, audit, action_timer, sampling)

            async def infer(observation, task):
                started = time.monotonic()
                reply = await _request_cancellable(transport, {
                    'endpoint': 'get_action', 'contract': expected,
                    'data': {'observation': client._payload(
                        dependencies, observation, task, roles,
                        right_required='external_right_image' in profile['cameras']), 'options': None},
                }, stop_requested, action_timer,
                    sampling.check_error if sampling is not None else None)
                return runner.PolicyResult(profile['model_id'], client._actions(reply, expected),
                                           reply, (time.monotonic() - started) * 1000)

            await runner.run_live_policy(infer, spec, wrapped, instruction,
                                         control_hz=profile['control_hz'], max_control_steps=steps,
                                         audit_callback=audit)
            _check_stop(stop_requested, action_timer)
            reason = 'step_limit'
        except _StopTrial as exc:
            reason = exc.reason
        except (KeyboardInterrupt, asyncio.CancelledError):
            reason = 'paused'
        except Exception as exc:
            errors.append(type(exc).__name__ + ': ' + str(exc))
        finally:
            action_timer.stop()
            # No reader may retain access to cameras after RobotEnv.close().
            if sampling is not None:
                try:
                    through = (action_timer.first_action_at + action_timer.elapsed
                               if action_timer.first_action_at is not None
                               and reason in EXPORT_ELIGIBLE_ENDINGS else None)
                    sampling.stop_capture(through=through)
                except BaseException as exc:
                    errors.append('continuous sampling stop: ' + type(exc).__name__ + ': ' + str(exc))
            # Release the robot before waiting for PNG I/O. Attempt every close
            # even when an earlier cleanup fails.
            for label, resource in [('robot', robot), ('transport', transport)]:
                if resource is not None:
                    try:
                        resource.close()
                    except BaseException as exc:
                        errors.append(label + ' close: ' + type(exc).__name__ + ': ' + str(exc))
            if sampling is not None:
                try:
                    sampling_status = sampling.finish()
                except BaseException as exc:
                    errors.append('continuous sampling close: ' + type(exc).__name__ + ': ' + str(exc))
            if recorder is not None:
                try:
                    recorder.close()
                except BaseException as exc:
                    errors.append('recording close: ' + type(exc).__name__ + ': ' + str(exc))
    sent = 0 if wrapped is None else wrapped.steps_sent
    complete = False
    if recorder is not None and not errors:
        try:
            status = json.loads((recording / 'recording-status.json').read_text(encoding='utf-8'))
            rows = (recording / 'frames.jsonl').read_text(encoding='utf-8').splitlines() if sent else []
            complete = (status.get('complete') is True and status.get('frames') == sent
                        and len(rows) == sent and sent > 0)
            if reason in EXPORT_ELIGIBLE_ENDINGS and not complete:
                errors.append('Recording is empty or its persisted frame count differs from returned actions')
        except Exception as exc:
            errors.append('recording verification: ' + type(exc).__name__ + ': ' + str(exc))
    if errors:
        reason, complete = 'error', False
    result = {'end_reason': reason, 'recording_complete': complete, 'steps_sent': sent,
              'recording_path': str(recording), 'error': '; '.join(errors) or None,
              'model': profile['model_id'], 'contract': expected, 'instruction': instruction,
              'max_steps': steps, 'control_hz': profile['control_hz'],
              'max_action_seconds': max_action_seconds,
              'first_action_at_monotonic': action_timer.first_action_at,
              'action_elapsed_seconds': action_timer.elapsed,
              'requires_manual_outcome': True,
              'outcome': ('awaiting_review' if reason in EXPORT_ELIGIBLE_ENDINGS else
                          'interrupted' if reason == 'paused' else 'error'),
              'started_at_unix': started_at, 'ended_at_unix': time.time()}
    if sampling is not None:
        result.update(recording_schema=CONTINUOUS_SCHEMA, recording_fps=RECORDING_FPS,
                      continuous_sampling=sampling_status)
    _json_new(output / 'trial-result.json', result)
    return result


def run_trial(profile, *, url, steps, instruction, output: Path, stop_requested):
    """Execute one explicitly authorized live attempt and retain its raw records."""
    return asyncio.run(_run_trial(profile, url=url, steps=steps, instruction=instruction,
                                  output=output, stop_requested=stop_requested))


def export_recording(profile, *, attempt_path: Path, metadata: dict) -> Path:
    """Export complete scored records; new attempts require a manual outcome."""
    attempt_path = Path(attempt_path)
    result = json.loads((attempt_path / 'trial-result.json').read_text(encoding='utf-8'))
    if result['end_reason'] not in EXPORT_ELIGIBLE_ENDINGS or not result['recording_complete']:
        raise ValueError('Only normally completed or time-limited complete recordings may be exported')
    score = metadata.get('score')
    if (isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score) or not 0 <= score <= 100):
        raise ValueError('Submit a score from 0 to 100 before export')
    manual = metadata.get('outcome_source') == 'manual'
    if ((manual and metadata.get('outcome') not in ('success', 'failed'))
            or (result.get('requires_manual_outcome') is True and not manual)):
        raise ValueError('Submit an explicit manual success/failed outcome before export')
    client = _load_client()
    expected = client._contract(profile)
    if result['contract'] != expected:
        raise ValueError('Selected profile differs from the recorded contract')
    recording = attempt_path / 'recording'
    policy = json.loads((recording / 'policy.json').read_text(encoding='utf-8'))
    if policy['contract'] != expected:
        raise ValueError('Recording policy differs from the selected profile')
    context = {**metadata, 'model': profile['model_id'], 'contract': expected,
               'end_reason': result['end_reason'], 'steps_sent': result['steps_sent'],
               'trial_started_at_unix': result['started_at_unix'],
               'trial_ended_at_unix': result['ended_at_unix'],
               'recording_cameras': recording_cameras(profile), 'recording_names': RECORDING_NAMES,
               'selected_command_semantics': ACTION_SEMANTICS}
    # Do not change the frozen context of a pre-timer recording on export retry.
    for key in ('max_action_seconds', 'first_action_at_monotonic', 'action_elapsed_seconds',
                'outcome', 'requires_manual_outcome', 'recording_schema', 'recording_fps'):
        if key in result:
            context[key] = result[key]
    if manual:
        context.update(outcome=metadata['outcome'], outcome_source='manual',
                       success=metadata['outcome'] == 'success')
    elif result['end_reason'] == 'time_limit':
        # Historical timeout exports used this rule. Keep their frozen context
        # unchanged on retry, while all newly recorded trials require review.
        context.update(success=False, outcome='failed')
    # Reject changed scores/notes on retry: the underlying exporter signature
    # must be stable, including after it has deleted committed source PNGs.
    context = json.loads(json.dumps(context, allow_nan=False))
    frozen = attempt_path / 'export-context.json'
    if frozen.exists():
        if json.loads(frozen.read_text(encoding='utf-8')) != context:
            raise ValueError('Export metadata changed; retain the original score/context for retry')
    else:
        _json_new(frozen, context)
    client._dependencies(profile)
    if result.get('recording_schema') == CONTINUOUS_SCHEMA:
        from evaluation_dense_export import export_dense
        start = result['first_action_at_monotonic']
        return export_dense(recording, list(RECORDING_NAMES.values()),
                            fps=result['recording_fps'], robot_type='franka',
                            task=result['instruction'], metadata=context,
                            start_monotonic=start,
                            stop_monotonic=start + result['action_elapsed_seconds'])
    return _export_function()(recording, list(RECORDING_NAMES.values()),
                              fps=int(result['control_hz']), robot_type='franka',
                              task=result['instruction'], metadata=context)

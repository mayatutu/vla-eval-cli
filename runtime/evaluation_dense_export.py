"""Export actual state RPC replies/camera frames on a uniform LeRobot clock.

No hardware is accessed here. Sparse action recordings are retained verbatim.
Every output state/image comes from an actual recorded RPC row/camera frame.
Nearest-time resampling may reuse a source; those rows are explicitly marked
and original timestamps retained. No state or image values are interpolated.
Repeated or regressing robot timestamps are preserved as diagnostic evidence.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


CAMERAS = ('head_image', 'left_image', 'right_image')
SCHEMA = 'continuous-state-video-v1'


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')


def _digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]
    if not rows or [row.get('frame_index') for row in rows] != list(range(len(rows))):
        raise ValueError(f'{path}: source frame indices must be nonempty and contiguous')
    return rows


def _number(value, description):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{description} must be a finite number')
    return float(value)


def _unique_nearest(rows, grid, *, key, tolerance=None, label, reference_times=None):
    """Choose nearest actual samples in order, permitting explicit source reuse.

    A missing sample does not invalidate an otherwise complete recording.
    Reused source indices remain visible to callers. For camera images,
    the midpoint of the grid and selected state time balances both distances.
    ``tolerance`` remains accepted for callers but is not a rejection limit.
    """
    times = [_number(row.get(key), f'{label} {key}') for row in rows]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError(f'{label}: source timestamps must be strictly increasing')
    if not rows:
        raise ValueError(f'{label}: source samples must be nonempty')
    if reference_times is not None and len(reference_times) != len(grid):
        raise ValueError('Camera state references must match the output row count')
    chosen, previous = [], -1
    for index, stamp in enumerate(grid):
        target = stamp if reference_times is None else (stamp + reference_times[index]) / 2
        low = max(0, previous)
        high = len(rows)
        insertion = bisect_left(times, target, low, high)
        candidates = {max(low, min(high - 1, insertion)),
                      max(low, min(high - 1, insertion - 1))}
        selected = min(candidates, key=lambda item: (abs(times[item] - target), item))
        chosen.append(selected)
        previous = selected
    return chosen


def _cadence_metrics(times, *, scale=1.0):
    """Report cadence without deciding whether timing is acceptable for training.

    Integer NUC nanoseconds are subtracted before conversion to seconds.
    """
    intervals = [(right - left) * scale for left, right in zip(times, times[1:])]
    if not intervals:
        return {'records': len(times), 'mean_hz': None, 'p95_interval_s': None,
                'max_interval_s': None, 'cadence_valid': False, 'reason': 'fewer_than_two_records'}
    repeats = sum(value == 0 for value in intervals)
    regressions = sum(value < 0 for value in intervals)
    if repeats or regressions:
        return {'records': len(times), 'mean_hz': None, 'p95_interval_s': None,
                'max_interval_s': None, 'cadence_valid': False,
                'reason': 'source_timestamps_not_strictly_increasing',
                'repeated_timestamp_count': repeats, 'regressed_timestamp_count': regressions}
    ordered = sorted(intervals)
    position = 0.95 * (len(ordered) - 1)
    low = int(position)
    percentile = ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (position - low)
    return {'records': len(times), 'mean_hz': len(intervals) / sum(intervals),
            'p95_interval_s': percentile, 'max_interval_s': max(intervals),
            'cadence_valid': True, 'reason': None}


def _commands(path):
    attempts, result = {}, []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('event') == 'execute_attempt':
            step = row['control_step']
            if step in attempts:
                raise ValueError('Duplicate command attempt index')
            attempts[step] = row
        elif row.get('event') == 'execute_succeeded':
            step = row['control_step']
            if step != len(result) or step not in attempts:
                raise ValueError('Successful commands must have contiguous matching attempts')
            command = attempts[step]['selected_command']
            if len(command) != 8:
                raise ValueError('Expected eight selected command values')
            command = [_number(value, 'command') for value in command]
            stamp = _number(row.get('sent_at_monotonic'), 'actual command send timestamp')
            if result and stamp <= result[-1]['timestamp']:
                raise ValueError('Command send timestamps must strictly increase')
            result.append({'index': step, 'timestamp': stamp, 'command': command})
    if not result:
        raise ValueError('No successful commands with actual send timestamps')
    return result


def _source_image(continuous, row, camera):
    relative = Path(row['file'])
    if relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 2:
        raise ValueError('Unsafe continuous image path')
    if relative.parts[0] != camera or relative.suffix.lower() != '.png':
        raise ValueError('Continuous image path does not match its camera')
    resolved = (continuous / relative).resolve()
    if continuous.resolve() not in resolved.parents:
        raise ValueError('Continuous image must stay inside its recording')
    return resolved


def _validate_vector(row, key, size):
    values = row.get(key)
    if not isinstance(values, list) or len(values) != size:
        raise ValueError(f'State {key} must have {size} real values')
    return [_number(value, f'state {key}') for value in values]


def _probe(video, count, fps):
    output = subprocess.run(['ffprobe', '-v', 'error', '-count_frames',
        '-select_streams', 'v:0', '-show_entries',
        'stream=nb_read_frames,avg_frame_rate,duration', '-of', 'json', str(video)],
        check=True, capture_output=True, text=True)
    stream = json.loads(output.stdout)['streams'][0]
    numerator, denominator = stream['avg_frame_rate'].split('/')
    if (int(stream['nb_read_frames']) != count
            or abs(float(numerator) / float(denominator) - fps) > 1e-6
            or abs(float(stream['duration']) - count / fps) > 1 / fps / 2):
        raise ValueError(f'Encoded video timing/frame count mismatch: {video}')
    return stream


def _cleanup(recording, manifest):
    """Delete only hash-matching images belonging to this committed export."""
    if manifest.get('retain_source_pngs'):
        return
    for relative, expected in manifest['source_png_sha256'].items():
        parts = Path(relative).parts
        if parts and parts[0] == 'continuous':
            parts = parts[1:]
        if len(parts) != 2 or parts[0] not in CAMERAS or not parts[1].endswith('.png'):
            raise ValueError('Refusing cleanup outside recorded source images')
        source = (recording / relative).resolve()
        if recording.resolve() not in source.parents or source.suffix.lower() != '.png':
            raise ValueError('Refusing cleanup outside recorded source images')
        if source.exists():
            if _digest(source) != expected:
                raise ValueError(f'Source image changed; not deleting {source}')
            source.unlink()


def _action_image_hashes(recording, cameras):
    """Validate numeric action records; enumerate images unless explicitly off.

    Only the boolean policy flag authorizes numeric-only action recording.
    Missing legacy PNGs are never treated as an implicit format migration.
    Continuous camera samples remain the dense dataset's sole image source.
    """
    policy_path = recording / 'policy.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8')) if policy_path.exists() else {}
    numeric_only = policy.get('action_recording_images') is False
    path = recording / 'frames.jsonl'
    if not path.exists():
        raise ValueError('Original numeric action recording is missing')
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]
    if (not rows or any(type(row.get('frame')) is not int for row in rows)
            or [row.get('frame') for row in rows] != list(range(len(rows)))):
        raise ValueError('Original action recording frame indices are not contiguous')
    status = json.loads((recording / 'recording-status.json').read_text(encoding='utf-8'))
    if not status.get('complete') or status.get('frames') != len(rows):
        raise ValueError('Original numeric action recording is incomplete or count differs')
    previous_seconds = None
    action_count = 0
    for index, row in enumerate(rows):
        seconds = _number(row.get('seconds'), 'original action capture time')
        if seconds < 0 or previous_seconds is not None and seconds <= previous_seconds:
            raise ValueError('Original action capture times must be nonnegative and strictly increasing')
        previous_seconds = seconds
        for key, length in [('joints', 7), ('gripper', 1), ('cartesian_position', 6)]:
            _validate_vector(row, key, length)
        if 'action' not in row:
            raise ValueError('Original action record is missing its action field')
        if row.get('action') is None:
            if index != len(rows) - 1:
                raise ValueError('Only the final original observation may have no action')
        else:
            _validate_vector(row, 'action', 8)
            action_count += 1
    if not action_count:
        raise ValueError('Original recording contains no numeric action samples')
    if numeric_only:
        return {}
    hashes = {}
    for row in rows:
        for camera in cameras:
            relative = f"{camera}/{row['frame']:06d}.png"
            source = (recording / relative).resolve()
            if (recording / camera).resolve() not in source.parents:
                raise ValueError('Original action image leaves its camera directory')
            hashes[relative] = _digest(source)
    return hashes


def _validated_streams(recording, cameras, *, fps, start_monotonic, stop_monotonic):
    recording = Path(recording).resolve()
    cameras = list(cameras)
    if type(fps) is not int or fps <= 0 or set(cameras) != set(CAMERAS) or len(cameras) != 3:
        raise ValueError('A positive integer FPS and all three unique cameras are required')
    start = _number(start_monotonic, 'start')
    stop = _number(stop_monotonic, 'stop')
    if stop <= start:
        raise ValueError('stop must be after start')
    count = max(1, math.ceil((stop - start) * fps - 1e-8))
    grid = [start + index / fps for index in range(count)]
    continuous, state = recording / 'continuous', recording / 'state'
    camera_status = json.loads((continuous / 'continuous-status.json').read_text(encoding='utf-8'))
    state_status = json.loads((state / 'status.json').read_text(encoding='utf-8'))
    if not camera_status.get('complete') or camera_status.get('error'):
        raise ValueError('Continuous camera recording did not finish completely')
    if not state_status.get('complete') or state_status.get('error'):
        raise ValueError('Independent state recording did not finish completely')
    states = _rows(state / 'frames.jsonl')
    if state_status.get('frames') != len(states):
        raise ValueError('State recording count differs from status')
    robot_timestamps = [sample.get('robot_source_timestamp_ns') for sample in states]
    if any(type(stamp) is not int or not 0 <= stamp <= 2**63 - 1 for stamp in robot_timestamps):
        raise ValueError('State robot_source_timestamp_ns must be exact nonnegative int64 nanoseconds')
    source_changes = ['initial'] + [
        'advanced' if right > left else 'repeated' if right == left else 'regressed'
        for left, right in zip(robot_timestamps, robot_timestamps[1:])]
    repeated_source_indices = [index for index, change in enumerate(source_changes) if change == 'repeated']
    regressed_source_indices = [index for index, change in enumerate(source_changes) if change == 'regressed']
    tolerance = 1 / fps
    state_indices = _unique_nearest(states, grid, key='captured_at_monotonic',
                                    tolerance=tolerance, label='state')
    for index, sample in enumerate(states):
        duration = _number(sample.get('rpc_duration_seconds'), 'state RPC duration')
        if duration < 0:
            raise ValueError(f'State sample {index} has negative RPC duration')
        requested = _number(sample.get('request_started_monotonic'), 'state request time')
        received = _number(sample.get('response_received_monotonic'), 'state response time')
        if (received < requested or abs(received - requested - duration) > 1e-6
                or abs(sample['captured_at_monotonic'] - (requested + received) / 2) > 1e-6):
            raise ValueError(f'State sample {index} has inconsistent RPC midpoint/duration')
        for key, length in [('joints', 7), ('gripper', 1), ('cartesian_position', 6)]:
            _validate_vector(sample, key, length)
    image_rows, image_indices, source_png_hashes = {}, {}, {}
    for camera in cameras:
        rows = _rows(continuous / camera / 'frames.jsonl')
        if camera_status['cameras'][camera]['frames'] != len(rows):
            raise ValueError(f'{camera}: frame count differs from camera status')
        for row in rows:
            if row.get('camera') != camera or (int(row.get('flags', 0)) & 0xE000) != 0x2000:
                raise ValueError(f'{camera}: missing verified V4L2 monotonic timestamp')
            if row.get('captured_at_monotonic') != row.get('driver_time'):
                raise ValueError(f'{camera}: capture timestamp must equal driver timestamp')
            path = _source_image(continuous, row, camera)
            source_png_hashes[str(path.relative_to(recording)).replace('\\', '/')] = _digest(path)
        if any(b['sequence'] <= a['sequence'] for a, b in zip(rows, rows[1:])):
            raise ValueError(f'{camera}: repeated or regressing physical frame sequence')
        image_rows[camera] = rows
        image_indices[camera] = _unique_nearest(rows, grid, key='driver_time',
            label=camera, reference_times=[states[index]['captured_at_monotonic'] for index in state_indices])

    state_image_skews = {
        camera: max(abs(states[state_index]['captured_at_monotonic']
                        - image_rows[camera][image_index]['driver_time'])
                    for state_index, image_index in zip(state_indices, image_indices[camera]))
        for camera in cameras}

    def metrics(rows, indices, key):
        stamps = [row[key] for row in rows]
        intervals = [right - left for left, right in zip(stamps, stamps[1:])]
        errors = [rows[source_index][key] - stamp for source_index, stamp in zip(indices, grid)]
        return {'source_frames': len(rows), 'selected_frames': len(indices),
                'unique_selected_frames': len(set(indices)),
                'reused_output_frames': len(indices) - len(set(indices)),
                'observed_hz': (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else None,
                'max_interval_seconds': max(intervals, default=0.0),
                'max_abs_alignment_error_seconds': max(map(abs, errors)),
                'first_source_index': indices[0], 'last_source_index': indices[-1]}

    slow_rpc = [{'source_index': index, 'selected': index in state_indices,
                 'rpc_duration_seconds': sample['rpc_duration_seconds'],
                 'request_started_monotonic': sample['request_started_monotonic'],
                 'response_received_monotonic': sample['response_received_monotonic'],
                 'midpoint_halfwidth_seconds': sample['rpc_duration_seconds'] / 2}
                for index, sample in enumerate(states) if sample['rpc_duration_seconds'] > tolerance]
    warnings = [{'code': 'state_rpc_longer_than_nominal_period', 'samples': slow_rpc}] if slow_rpc else []
    if repeated_source_indices or regressed_source_indices:
        warnings.append({'code': 'robot_source_timestamp_not_advanced',
                         'repeated_source_indices': repeated_source_indices,
                         'regressed_source_indices': regressed_source_indices,
                         'meaning': 'Actual RPC replies retained; source timestamps do not prove a fresh robot measurement.'})
    state_metrics = metrics(states, state_indices, 'captured_at_monotonic')
    if state_metrics['reused_output_frames']:
        warnings.append({'code': 'state_source_reused',
                         'count': state_metrics['reused_output_frames'],
                         'meaning': 'Nearest recorded RPC row reused on the nominal grid; no new measurement or interpolation.'})
    if state_metrics['max_abs_alignment_error_seconds'] > tolerance:
        warnings.append({'code': 'state_grid_time_offset',
                         'max_seconds': state_metrics['max_abs_alignment_error_seconds']})
    for camera in cameras:
        image_metrics = metrics(image_rows[camera], image_indices[camera], 'driver_time')
        if image_metrics['reused_output_frames']:
            warnings.append({'code': 'camera_source_reused', 'camera': camera,
                             'count': image_metrics['reused_output_frames'],
                             'meaning': 'Nearest recorded camera frame reused on the nominal grid; no new exposure or interpolation.'})
        if image_metrics['max_abs_alignment_error_seconds'] > tolerance or state_image_skews[camera] > tolerance:
            warnings.append({'code': 'camera_time_offset', 'camera': camera,
                'max_grid_offset_seconds': image_metrics['max_abs_alignment_error_seconds'],
                'max_state_image_skew_seconds': state_image_skews[camera]})
        gaps = sum(row.get('sequence_gap', 0) for row in image_rows[camera])
        if gaps:
            warnings.append({'code': 'camera_source_sequence_gaps', 'camera': camera, 'count': gaps})
    summary = {'passed': True, 'structural_validation_passed': True,
        'timing_assessment': 'warning' if warnings else 'nominal',
        'timing_policy': {'name': 'preserve-real-samples-diagnostic-timing-v1',
            'hard_timing_limits': False, 'nominal_period_seconds': tolerance,
            'meaning': 'Cadence, latency and alignment are reported, not export rejection rules. No claim of precise hardware synchronization or automatic training suitability.'},
        'warnings': warnings, 'schema_version': 1, 'recording_schema': SCHEMA,
        'fps': fps, 'start_monotonic': start, 'stop_monotonic': stop,
        'output_frames': count, 'nominal_period_seconds': tolerance,
        'max_state_image_timestamp_skew_seconds': max(state_image_skews.values()),
        'state_image_timestamp_semantics': 'host state RPC midpoint estimate versus V4L2 monotonic capture estimate; not precise hardware synchronization',
        'no_reused_rpc_rows_or_camera_frames': len(set(state_indices)) == count and all(len(set(indices)) == count for indices in image_indices.values()),
        'resampling_policy': 'nearest actual source in nondecreasing source order; source reuse explicitly marked; no interpolation',
        'state_sample_semantics': 'Each output row references an actual recorded RPC reply. Reused source rows are marked by state_rpc_is_new=0; distinct RPC replies do not imply fresh robot measurements.',
        'state': {**state_metrics,
                  'host_cadence': _cadence_metrics([row['captured_at_monotonic'] for row in states]),
                  'robot_source_cadence': _cadence_metrics(robot_timestamps, scale=1e-9),
                  'repeated_robot_source_timestamp_count': len(repeated_source_indices),
                  'regressed_robot_source_timestamp_count': len(regressed_source_indices),
                  'unique_robot_source_timestamps': len(set(robot_timestamps)),
                  'slow_rpc_count': len(slow_rpc),
                  'robot_source_timestamp_semantics': 'original NUC robot sample timestamp; exact int64 nanoseconds; distinct from host alignment clock',
                  'first_robot_source_timestamp_ns': robot_timestamps[state_indices[0]],
                  'last_robot_source_timestamp_ns': robot_timestamps[state_indices[-1]],
                  'max_selected_rpc_duration_seconds': max(states[index]['rpc_duration_seconds'] for index in state_indices)},
        'cameras': {camera: {**metrics(image_rows[camera], image_indices[camera], 'driver_time'),
                            'cadence': _cadence_metrics([row['driver_time'] for row in image_rows[camera]]),
                            'max_state_image_timestamp_skew_seconds': state_image_skews[camera],
                            'source_sequence_gaps': sum(row.get('sequence_gap', 0) for row in image_rows[camera])}
                    for camera in cameras}}
    return {'summary': summary, 'grid': grid, 'states': states, 'state_indices': state_indices,
            'robot_source_timestamp_changes': source_changes,
            'image_rows': image_rows, 'image_indices': image_indices,
            'source_png_hashes': source_png_hashes,
            'camera_status': camera_status, 'state_status': state_status}


def validate_streams(recording, cameras, *, fps, start_monotonic, stop_monotonic):
    """Read-only real-stream gate shared by the no-action probe and exporter.

    Requires no actions.jsonl, robot, GPU, FFmpeg or training-dataset export.
    Raises ValueError/FileNotFoundError on malformed/incomplete or empty sources.
    Timing differences and source reuse produce diagnostic warnings only.
    """
    return _validated_streams(recording, cameras, fps=fps,
        start_monotonic=start_monotonic, stop_monotonic=stop_monotonic)['summary']


def export_dense(recording, cameras, *, fps, robot_type, task, metadata,
                 start_monotonic, stop_monotonic, cleanup_sources=True):
    """Create a new dense v3 dataset, or verify/reuse the identical committed one.

    State/image source indices may repeat; no observation values are synthesized.
    Source timing offsets and RPC midpoint uncertainty are preserved and reported,
    without rejecting a structurally complete recording for timing jitter.
    The action is the last successfully sent command at/before the grid tick.
    """
    recording = Path(recording).resolve()
    cameras = list(cameras)
    if type(fps) is not int or fps <= 0 or set(cameras) != set(CAMERAS) or len(cameras) != 3:
        raise ValueError('A positive integer FPS and all three unique cameras are required')
    start = _number(start_monotonic, 'start')
    stop = _number(stop_monotonic, 'stop')
    if stop <= start:
        raise ValueError('stop must be after start')
    count = max(1, math.ceil((stop - start) * fps - 1e-8))
    grid = [start + index / fps for index in range(count)]
    continuous, state = recording / 'continuous', recording / 'state'
    source_paths = [continuous / 'continuous-status.json', state / 'status.json',
                    state / 'frames.jsonl', recording.parent / 'actions.jsonl',
                    *[continuous / camera / 'frames.jsonl' for camera in cameras]]
    for optional in ('frames.jsonl', 'recording-status.json', 'policy.json'):
        if (recording / optional).exists():
            source_paths.append(recording / optional)
    source_hashes = {str(path.relative_to(recording.parent)).replace('\\', '/'): _digest(path)
                     for path in source_paths}
    context = {**metadata, 'robot_type': robot_type, 'task': task, 'fps': fps,
               'cameras': cameras, 'recording_schema': SCHEMA,
               'policy': json.loads((recording / 'policy.json').read_text(encoding='utf-8')) if (recording / 'policy.json').exists() else None,
               'start_monotonic': start, 'stop_monotonic': stop,
               'source_files_sha256': source_hashes}
    signature = hashlib.sha256(json.dumps(context, sort_keys=True,
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    destination = recording / 'lerobot'
    lock = recording / '.dense-export.lock'
    with lock.open('x', encoding='utf-8'):
        pass
    staging = None
    try:
        if destination.exists():
            marker = json.loads((destination / 'meta/colosseum.json').read_text(encoding='utf-8'))
            if marker.get('source_sha256') != signature:
                raise ValueError('Existing LeRobot dataset differs; refusing to overwrite it')
            for relative, expected in marker['files'].items():
                if _digest(destination / relative) != expected:
                    raise ValueError('Existing LeRobot dataset is incomplete or modified')
            manifest = json.loads((destination / 'meta/alignment.json').read_text(encoding='utf-8'))
            if cleanup_sources:
                _cleanup(recording, manifest)
            return destination
        validated = _validated_streams(recording, cameras, fps=fps,
            start_monotonic=start, stop_monotonic=stop)
        states, state_indices = validated['states'], validated['state_indices']
        image_rows, image_indices = validated['image_rows'], validated['image_indices']
        source_png_hashes = validated['source_png_hashes']
        source_png_hashes.update(_action_image_hashes(recording, cameras))
        camera_status, state_status = validated['camera_status'], validated['state_status']
        tolerance = 1 / fps
        commands = _commands(recording.parent / 'actions.jsonl')
        command_times = [command['timestamp'] for command in commands]
        command_indices = [bisect_right(command_times, stamp + 1e-9) - 1 for stamp in grid]
        if min(command_indices) < 0:
            raise ValueError('No actually sent command at the first grid tick')
        staging = Path(tempfile.mkdtemp(prefix='.dense-', dir=recording))
        proxy = staging / 'source'
        proxy.mkdir()
        aligned, alignment = [], []
        for index, stamp in enumerate(grid):
            state_index, command_index = state_indices[index], command_indices[index]
            state_rpc_is_new = index == 0 or state_index != state_indices[index - 1]
            sample, command = states[state_index], commands[command_index]
            duration = _number(sample.get('rpc_duration_seconds'), 'state RPC duration')
            aligned.append({'frame': index, 'seconds': index / fps,
                'joints': _validate_vector(sample, 'joints', 7),
                'gripper': _validate_vector(sample, 'gripper', 1),
                'cartesian_position': _validate_vector(sample, 'cartesian_position', 6),
                'action': command['command']})
            mapping = {'frame_index': index, 'timestamp': index / fps,
                'target_monotonic': stamp, 'state_index': state_index,
                'state_rpc_is_new': state_rpc_is_new,
                'state_source_reused': not state_rpc_is_new,
                'state_source_timestamp': sample['captured_at_monotonic'],
                'robot_source_timestamp_ns': sample['robot_source_timestamp_ns'],
                'robot_source_timestamp_change': validated['robot_source_timestamp_changes'][state_index],
                'state_rpc_duration_seconds': duration,
                'state_request_started_timestamp': sample['request_started_monotonic'],
                'state_response_received_timestamp': sample['response_received_monotonic'],
                'state_rpc_halfwidth_seconds': duration / 2,
                'state_timing_warning': (not state_rpc_is_new or duration > tolerance or abs(sample['captured_at_monotonic'] - stamp) > tolerance
                                         or validated['robot_source_timestamp_changes'][state_index] in ('repeated', 'regressed')),
                'state_alignment_error_seconds': sample['captured_at_monotonic'] - stamp,
                'action_command_index': command_index, 'action_source_timestamp': command['timestamp'],
                'action_is_new': index == 0 or command_index != command_indices[index - 1],
                'action_age_seconds': stamp - command['timestamp'], 'cameras': {}}
            for camera in cameras:
                row = image_rows[camera][image_indices[camera][index]]
                image_is_new = index == 0 or image_indices[camera][index] != image_indices[camera][index - 1]
                source = _source_image(continuous, row, camera)
                target = proxy / camera / f'{index:06d}.png'
                target.parent.mkdir(exist_ok=True)
                # Same-filesystem hardlinks add no copy to the recording hot path.
                os.link(source, target)
                mapping['cameras'][camera] = {'source_index': row['frame_index'],
                    'source_timestamp': row['driver_time'], 'sequence': row['sequence'],
                    'alignment_error_seconds': row['driver_time'] - stamp,
                    'state_image_skew_seconds': row['driver_time'] - sample['captured_at_monotonic'],
                    'timing_warning': not image_is_new or abs(row['driver_time'] - stamp) > tolerance or abs(row['driver_time'] - sample['captured_at_monotonic']) > tolerance,
                    'is_new': image_is_new, 'source_reused': not image_is_new, 'source_file': row['file']}
            alignment.append(mapping)
        (proxy / 'frames.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in aligned), encoding='utf-8')
        _json(proxy / 'recording-status.json', {'complete': True, 'frames': count})
        if (recording / 'policy.json').exists():
            shutil.copyfile(recording / 'policy.json', proxy / 'policy.json')
        from evaluation_lerobot_export import export_trial, _stats
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        dataset = export_trial(proxy, cameras, fps=fps, robot_type=robot_type,
                               task=task, metadata=context)
        data_path = dataset / 'data/chunk-000/file-000.parquet'
        table = pq.read_table(data_path)
        info = json.loads((dataset / 'meta/info.json').read_text(encoding='utf-8'))
        stats = json.loads((dataset / 'meta/stats.json').read_text(encoding='utf-8'))
        scalars = {'observation.capture_timestamp': [item['state_source_timestamp'] for item in alignment],
            'observation.state_source_timestamp': [item['state_source_timestamp'] for item in alignment],
            'observation.state_source_index': state_indices,
            'observation.robot_source_timestamp_ns': [item['robot_source_timestamp_ns'] for item in alignment],
            'observation.state_is_new': [int(item['state_rpc_is_new'] and item['robot_source_timestamp_change'] in ('initial', 'advanced')) for item in alignment],
            'observation.state_rpc_is_new': [int(item['state_rpc_is_new']) for item in alignment],
            'observation.state_source_reused': [int(item['state_source_reused']) for item in alignment],
            'observation.robot_source_timestamp_repeated': [int(item['robot_source_timestamp_change'] == 'repeated') for item in alignment],
            'observation.robot_source_timestamp_regressed': [int(item['robot_source_timestamp_change'] == 'regressed') for item in alignment],
            'observation.state_alignment_error_seconds': [item['state_alignment_error_seconds'] for item in alignment],
            'observation.state_rpc_duration_seconds': [item['state_rpc_duration_seconds'] for item in alignment],
            'observation.state_request_started_timestamp': [item['state_request_started_timestamp'] for item in alignment],
            'observation.state_response_received_timestamp': [item['state_response_received_timestamp'] for item in alignment],
            'observation.state_rpc_halfwidth_seconds': [item['state_rpc_halfwidth_seconds'] for item in alignment],
            'observation.state_timing_warning': [int(item['state_timing_warning']) for item in alignment],
            'action_command_index': command_indices,
            'action_source_timestamp': [item['action_source_timestamp'] for item in alignment],
            'action_is_new': [int(item['action_is_new']) for item in alignment],
            'action_age_seconds': [item['action_age_seconds'] for item in alignment]}
        for camera in cameras:
            for field in ('source_index', 'source_timestamp', 'alignment_error_seconds', 'state_image_skew_seconds', 'sequence'):
                scalars[f'observation.image_{field}.{camera}'] = [item['cameras'][camera][field] for item in alignment]
            scalars[f'observation.image_is_new.{camera}'] = [int(item['cameras'][camera]['is_new']) for item in alignment]
            scalars[f'observation.image_source_reused.{camera}'] = [int(item['cameras'][camera]['source_reused']) for item in alignment]
            scalars[f'observation.image_timing_warning.{camera}'] = [int(item['cameras'][camera]['timing_warning']) for item in alignment]
        for name, values in scalars.items():
            integer = (name.endswith(('_index', '_is_new', '_source_reused', '_timestamp_ns', '_timing_warning', '_timestamp_repeated', '_timestamp_regressed')) or '.image_source_index.' in name
                       or '.image_sequence.' in name or '.image_is_new.' in name or '.image_timing_warning.' in name or '.image_source_reused.' in name)
            values = np.asarray(values, dtype=np.int64 if integer else np.float64)
            column = pa.array(values)
            position = table.schema.get_field_index(name)
            table = table.set_column(position, name, column) if position >= 0 else table.append_column(name, column)
            info['features'][name] = {'dtype': str(values.dtype), 'shape': [1], 'names': None}
            stats[name] = _stats(values)
        pq.write_table(table, data_path)
        episode_path = dataset / 'meta/episodes/chunk-000/file-000.parquet'
        episode = pq.read_table(episode_path).to_pylist()[0]
        for name in scalars:
            for key, value in stats[name].items():
                episode[f'stats/{name}/{key}'] = value
        pq.write_table(pa.Table.from_pylist([episode]), episode_path)
        _json(dataset / 'meta/info.json', info)
        _json(dataset / 'meta/stats.json', stats)
        probes = {camera: _probe(dataset / f'videos/observation.images.{camera}/chunk-000/file-000.mp4', count, fps)
                  for camera in cameras}
        manifest = {'schema_version': 1, 'recording_schema': SCHEMA,
            'fps': fps, 'output_frames': count, 'action_duration_seconds': stop - start,
            'encoded_duration_seconds': count / fps, 'nominal_period_seconds': tolerance,
            'state_timestamp_semantics': 'host RPC request/response midpoint estimate; not exact robot sensor timestamp',
            'image_timestamp_semantics': 'V4L2 CLOCK_MONOTONIC driver capture timestamp',
            'action_semantics': 'most recent successfully sent selected absolute joint/gripper command; holding a target is not a new command or measured execution',
            'no_reused_rpc_rows_or_camera_frames': validated['summary']['no_reused_rpc_rows_or_camera_frames'], 'source_png_sha256': source_png_hashes,
            'resampling_policy': validated['summary']['resampling_policy'],
            'state_sample_semantics': validated['summary']['state_sample_semantics'],
            'state_is_new_semantics': 'Distinct selected RPC row with initial source timestamp or timestamp advanced versus the previous recorded RPC reply; not a guarantee of fresh sensor data. state_rpc_is_new identifies a source index different from the previous exported row.',
            'image_is_new_semantics': 'Camera source index differs from the previous exported row; reused frames preserve their original sequence and capture timestamp.',
            'source_reused_semantics': 'The same recorded RPC row or camera frame as the previous output row; no new measurement/exposure. First output row is never marked reused.',
            'robot_source_timestamp_change_semantics': 'Compared against the previous recorded RPC reply, not the previous exported row; raw int64 source values are unchanged.',
            'source_signature_semantics': 'SHA256 of canonical sorted JSON export context, including original source-file digests; not the legacy sparse-recording signature',
            'source_files_sha256': source_hashes, 'video_probes': probes,
            'quality': validated['summary'],
            # Timing warnings remain in the dataset; only verified commits clean sources.
            'retain_source_pngs': False,
            'camera_source_status': camera_status, 'state_source_status': state_status,
            'frames': alignment}
        _json(dataset / 'meta/alignment.json', manifest)
        files = {str(path.relative_to(dataset)).replace('\\', '/'): _digest(path)
                 for path in dataset.rglob('*') if path.is_file() and path != dataset / 'meta/colosseum.json'}
        _json(dataset / 'meta/colosseum.json', {**context, 'source_sha256': signature,
            'files': files, 'timestamp_semantics': 'uniform elapsed time grid referencing actual recorded RPC rows/camera frames; reused sources are marked and retain original timestamps; robot source timestamps may repeat or regress',
            'capture_timestamp_semantics': 'absolute host CLOCK_MONOTONIC RPC midpoint estimate',
            'action_semantics': manifest['action_semantics']})
        dataset.rename(destination)
        if cleanup_sources:
            _cleanup(recording, manifest)
        return destination
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        lock.unlink(missing_ok=True)

"""Independent, read-only robot state recording for the evaluation recorder.

Import and construction are inert. Only ``start`` creates a dedicated worker,
which owns its ZeroRPC client for its entire lifetime. It never uses the action
client or ServerInterface (whose constructor can launch robot controllers).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import threading
import time
from collections.abc import Mapping


class _StateTransport:
    def __init__(self, endpoint, timeout):
        # Import, context, socket, calls and close all belong to the worker.
        import zerorpc
        self.context = zerorpc.Context()
        self.client = zerorpc.Client(context=self.context, timeout=timeout, heartbeat=20)
        try:
            self.client.connect(endpoint)
        except BaseException:
            self.close()
            raise

    def get_robot_state(self):
        return self.client.get_robot_state()

    def close(self):
        try:
            self.client.close()
        finally:
            self.context.term()


def _vector(value, size, name):
    if size == 1 and isinstance(value, (int, float)) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f'{name} must contain {size} numbers')
    if any(isinstance(item, bool) or not isinstance(item, (int, float))
           or not math.isfinite(item) for item in value):
        raise ValueError(f'{name} must contain finite numbers')
    return [float(item) for item in value]


def validate_state(reply):
    """Validate DROID's (state_dict, timestamp_dict), preserving NUC timestamps."""
    if not isinstance(reply, (list, tuple)) or len(reply) != 2:
        raise ValueError('get_robot_state must return state and timestamps')
    state, stamps = reply
    if not isinstance(state, Mapping) or not isinstance(stamps, Mapping):
        raise ValueError('State and timestamps must be mappings')
    # Remote values are JSON-compatible MessagePack scalars/lists, not numpy.
    source_timestamps = json.loads(json.dumps(dict(stamps), allow_nan=False))
    seconds = source_timestamps.get('robot_timestamp_seconds')
    nanos = source_timestamps.get('robot_timestamp_nanos')
    if type(seconds) is not int or seconds < 0 or type(nanos) is not int or not 0 <= nanos < 1_000_000_000:
        raise ValueError('NUC robot_timestamp_seconds/nanos must be nonnegative integers with nanos < 1000000000')
    source_ns = seconds * 1_000_000_000 + nanos
    if source_ns > 2**63 - 1:
        raise ValueError('NUC robot timestamp must fit signed int64 nanoseconds')
    return {
        'joints': _vector(state.get('joint_positions'), 7, 'joint_positions'),
        'gripper': _vector(state.get('gripper_position'), 1, 'gripper_position'),
        'cartesian_position': _vector(state.get('cartesian_position'), 6, 'cartesian_position'),
        'source_timestamps': source_timestamps,
        'robot_source_timestamp_ns': source_ns,
    }


class StateRecorder:
    """Persist actual state RPC replies; never synthesize missing samples.

    ``path`` is the new ``recording/state`` directory. ``complete`` means all
    structurally valid replies were written and the worker closed normally;
    it does not assert that every RPC returned a new NUC sample or that the
    requested rate was achieved. Original source timestamps and timing
    diagnostics are retained alongside each actual RPC reply.
    """
    def __init__(self, path, endpoint, *, hz=30, rpc_timeout=1.0,
                 transport_factory=None, clock=time.monotonic):
        if not isinstance(endpoint, str) or not endpoint.startswith('tcp://'):
            raise ValueError('A tcp:// state endpoint is required')
        if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not math.isfinite(hz) or not 0 < hz <= 30:
            raise ValueError('State sampling rate must be positive and at most 30 Hz')
        if (isinstance(rpc_timeout, bool) or not isinstance(rpc_timeout, (int, float))
                or not math.isfinite(rpc_timeout) or not 0 < rpc_timeout <= 2):
            raise ValueError('RPC timeout must be positive and at most 2 seconds')
        self.path = Path(path)
        self.endpoint = endpoint
        self.hz = float(hz)
        self.rpc_timeout = float(rpc_timeout)
        self.factory = transport_factory or _StateTransport
        self.clock = clock
        self.ready = threading.Event()
        self._stop = threading.Event()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._error = None
        self._frames = 0
        self._first = self._last = None
        self._last_source_ns = None
        self._source_repeats = self._source_regressions = 0
        self._source_repeat_run = self._max_source_repeat_run = 0
        self._max_interval = self._max_rpc = 0.0
        self._slow_rpcs = 0
        self._skipped_slots = 0
        self._started = None

    def start(self):
        if self._thread is not None:
            raise RuntimeError('State recorder has already started')
        self.path.mkdir(parents=True, exist_ok=False)
        self._started = self.clock()
        self._thread = threading.Thread(target=self._run, name='evaluation-state', daemon=True)
        self._thread.start()
        return self

    def wait_ready(self, timeout=2.0):
        if self._thread is None:
            raise RuntimeError('State recorder has not started')
        deadline = time.monotonic() + timeout
        while not self.ready.is_set():
            if self._done.is_set():
                raise RuntimeError('State recorder failed before readiness: ' + str(self._error))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('No validated robot state arrived before readiness timeout')
            self.ready.wait(min(remaining, 0.02))
        if self._error is not None:
            raise RuntimeError('State recorder failed: ' + self._error)
        return True

    def _set_error(self, exc):
        with self._lock:
            if self._error is None:
                self._error = type(exc).__name__ + ': ' + str(exc)

    def check_error(self):
        """Surface asynchronous read failures to the trial before another action."""
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError('State recording failed: ' + error)

    def _run(self):
        transport = None
        period = 1.0 / self.hz
        try:
            transport = self.factory(self.endpoint, self.rpc_timeout)
            with (self.path / 'frames.jsonl').open('x', encoding='utf-8') as stream:
                next_start = self.clock()
                while not self._stop.is_set():
                    if self._stop.wait(max(0.0, next_start - self.clock())):
                        break
                    started = self.clock()
                    reply = transport.get_robot_state()
                    received = self.clock()
                    # A reply arriving after stop is outside the requested range.
                    if self._stop.is_set():
                        break
                    if not math.isfinite(started) or not math.isfinite(received) or received < started:
                        raise ValueError('Non-monotonic RPC timing')
                    row = validate_state(reply)
                    source_ns = row['robot_source_timestamp_ns']
                    stamp = (started + received) / 2.0
                    row.update(frame_index=self._frames,
                               captured_at_monotonic=stamp,
                               request_started_monotonic=started,
                               response_received_monotonic=received,
                               rpc_duration_seconds=received - started,
                               timestamp_semantics='host RPC interval midpoint estimate; original NUC timestamps preserved')
                    if self._last is not None and stamp <= self._last:
                        raise ValueError('State sample host timestamps did not increase')
                    stream.write(json.dumps(row, allow_nan=False, ensure_ascii=False) + '\n')
                    stream.flush()
                    with self._lock:
                        if self._first is None:
                            self._first = stamp
                        if self._last is not None:
                            self._max_interval = max(self._max_interval, stamp - self._last)
                        self._last = stamp
                        # The normal Client records each RPC reply without
                        # requiring a new NUC timestamp. Preserve that behavior:
                        # source-clock repeats/regressions are diagnostics only.
                        repeated = self._last_source_ns is not None and source_ns == self._last_source_ns
                        self._source_repeats += int(repeated)
                        self._source_regressions += int(self._last_source_ns is not None
                                                        and source_ns < self._last_source_ns)
                        self._source_repeat_run = self._source_repeat_run + 1 if repeated else 0
                        self._max_source_repeat_run = max(self._max_source_repeat_run, self._source_repeat_run)
                        self._last_source_ns = source_ns
                        self._frames += 1
                        self._max_rpc = max(self._max_rpc, received - started)
                        self._slow_rpcs += int(received - started > period)
                    self.ready.set()
                    # Keep the schedule anchored, avoiding accumulated timer
                    # drift. Missed slots are logged and skipped, never replayed.
                    next_start += period
                    now = self.clock()
                    if next_start <= now:
                        missed = int((now - next_start) // period) + 1
                        next_start += missed * period
                        with self._lock:
                            self._skipped_slots += missed
        except BaseException as exc:
            self._set_error(exc)
        finally:
            if transport is not None:
                try:
                    transport.close()
                except BaseException as exc:
                    self._set_error(exc)
            self._done.set()
            try:
                self._write_status()
            except BaseException as exc:
                self._set_error(exc)

    def status(self):
        with self._lock:
            duration = (self._last - self._first
                        if self._last is not None and self._first is not None else 0.0)
            return {
                'complete': self._done.is_set() and self._error is None and self._frames > 0,
                'frames': self._frames,
                'error': self._error,
                'requested_hz': self.hz,
                'observed_hz': (self._frames - 1) / duration if duration > 0 else None,
                'first_sample_monotonic': self._first,
                'last_sample_monotonic': self._last,
                'last_robot_source_timestamp_ns': self._last_source_ns,
                'source_timestamp_repeated_count': self._source_repeats,
                'source_timestamp_regression_count': self._source_regressions,
                # Counts equal-to-previous replies, excluding the first reply
                # in each run (three identical replies therefore report two).
                'source_timestamp_max_consecutive_repeats': self._max_source_repeat_run,
                'max_interval_seconds': self._max_interval,
                'max_rpc_duration_seconds': self._max_rpc,
                'slow_rpc_count': self._slow_rpcs,
                'skipped_schedule_slots': self._skipped_slots,
                'rpc_timeout_seconds': self.rpc_timeout,
                'ready': self.ready.is_set(),
                'sampling_semantics': 'actual read-only RPC replies; NUC samples may repeat; no interpolation, synthetic replies, or catch-up sampling',
                'timestamp_semantics': 'host RPC midpoint estimate; source_timestamps retain original NUC clock and meaning',
                'rate_validation_required': True,
            }

    def _write_status(self):
        # Unique temporary filename also permits a bounded-stop failure report
        # when a transport violates its own timeout contract.
        temporary = self.path / ('status.' + str(threading.get_ident()) + '.tmp')
        temporary.write_text(json.dumps(self.status(), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(self.path / 'status.json')

    def stop(self):
        self._stop.set()
        if self._thread is None:
            return self.status()
        self._thread.join(timeout=self.rpc_timeout + 1.0)
        if self._thread.is_alive():
            self._set_error(TimeoutError('State worker did not close within its bounded shutdown window'))
        self._write_status()
        return self.status()

    close = stop

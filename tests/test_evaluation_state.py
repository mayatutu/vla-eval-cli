"""Read-only state recorder tests with in-process fake transports only."""
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation_state import StateRecorder, _StateTransport, validate_state


def reply(index=1):
    return ({'joint_positions': [index / 100] * 7,
             'gripper_position': 0.3, 'cartesian_position': [0.1] * 6},
            {'robot_timestamp_seconds': 1700000000, 'robot_timestamp_nanos': index,
             'nested': {'gripper': index}})


class FakeTransport:
    def __init__(self, *, delay=0.0, fail=None, response=None):
        self.delay, self.fail, self.response = delay, fail, response
        self.calls = []
        self.closed_on = None

    def get_robot_state(self):
        self.calls.append((time.monotonic(), threading.get_ident()))
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise self.fail
        if callable(self.response):
            return self.response(len(self.calls))
        return self.response if self.response is not None else reply(len(self.calls))

    def close(self):
        self.closed_on = threading.get_ident()


class StateRecorderTests(unittest.TestCase):
    def setUp(self):
        clock_patch = patch.object(time, 'monotonic', time.perf_counter)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'state'

    def recorder(self, transport, **options):
        self.constructed = []
        def factory(endpoint, timeout):
            self.constructed.append((endpoint, timeout, threading.get_ident()))
            return transport
        options.setdefault('clock', time.perf_counter)
        recorder = StateRecorder(self.path, 'tcp://127.0.0.1:4242',
                                 transport_factory=factory, **options)
        self.addCleanup(recorder.stop)
        return recorder

    def test_construction_inert_and_worker_owns_transport(self):
        transport = FakeTransport()
        recorder = self.recorder(transport)
        self.assertEqual(self.constructed, [])
        self.assertFalse(self.path.exists())
        recorder.start()
        self.assertTrue(recorder.wait_ready())
        time.sleep(0.09)
        status = recorder.stop()
        worker = self.constructed[0][2]
        self.assertNotEqual(worker, threading.get_ident())
        self.assertTrue(all(thread == worker for _, thread in transport.calls))
        self.assertEqual(transport.closed_on, worker)
        self.assertTrue(status['complete'])
        self.assertGreaterEqual(status['frames'], 2)
        self.assertTrue(status['rate_validation_required'])
        rows = [json.loads(row) for row in (self.path / 'frames.jsonl').read_text().splitlines()]
        self.assertEqual([row['frame_index'] for row in rows], list(range(len(rows))))
        self.assertEqual(rows[0]['source_timestamps'], reply()[1])
        self.assertEqual(rows[0]['robot_source_timestamp_ns'], 1700000000000000001)
        self.assertEqual(rows[0]['gripper'], [0.3])
        for row in rows:
            self.assertLessEqual(row['request_started_monotonic'], row['captured_at_monotonic'])
            self.assertLessEqual(row['captured_at_monotonic'], row['response_received_monotonic'])
        self.assertEqual(json.loads((self.path / 'status.json').read_text()), status)

    def test_schedule_is_bounded_and_does_not_claim_slow_samples_are_30hz(self):
        transport = FakeTransport(delay=0.05)
        recorder = self.recorder(transport).start()
        recorder.wait_ready()
        time.sleep(0.13)
        status = recorder.stop()
        intervals = [b[0] - a[0] for a, b in zip(transport.calls, transport.calls[1:])]
        self.assertTrue(all(interval >= 0.04 for interval in intervals))
        self.assertGreater(status['slow_rpc_count'], 0)
        self.assertLess(status['observed_hz'], 25)

    def test_fast_transport_respects_anchored_sampling_rate(self):
        transport = FakeTransport()
        recorder = self.recorder(transport).start()
        recorder.wait_ready()
        time.sleep(0.13)
        recorder.stop()
        intervals = [b[0] - a[0] for a, b in zip(transport.calls, transport.calls[1:])]
        self.assertGreaterEqual(len(intervals), 2)
        # A delayed Windows scheduler wake can shorten the following interval
        # on a fixed schedule. The actual sample count, not every inter-call
        # gap, is bounded by the elapsed 30 Hz sampling slots.
        elapsed = transport.calls[-1][0] - transport.calls[0][0]
        self.assertLessEqual(len(transport.calls), math.ceil(elapsed * 30) + 1)

    def test_invalid_response_fails_without_fabricating_rows(self):
        transport = FakeTransport(response=({'joint_positions': [0] * 6}, {}))
        recorder = self.recorder(transport).start()
        with self.assertRaises(RuntimeError):
            recorder.wait_ready()
        status = recorder.stop()
        self.assertFalse(status['complete'])
        self.assertEqual(status['frames'], 0)
        self.assertEqual((self.path / 'frames.jsonl').read_text(), '')
        self.assertIsNotNone(transport.closed_on)

    def test_rpc_failure_closes_and_reports_original_failure(self):
        recorder = self.recorder(FakeTransport(fail=TimeoutError('read timed out'))).start()
        with self.assertRaisesRegex(RuntimeError, 'read timed out'):
            recorder.wait_ready()
        status = recorder.stop()
        self.assertIn('read timed out', status['error'])
        self.assertFalse(status['complete'])

    def test_existing_output_is_never_overwritten(self):
        self.path.mkdir()
        marker = self.path / 'frames.jsonl'
        marker.write_text('original')
        recorder = StateRecorder(self.path, 'tcp://127.0.0.1:4242', transport_factory=lambda *_: self.fail())
        with self.assertRaises(FileExistsError):
            recorder.start()
        self.assertEqual(marker.read_text(), 'original')

    def test_stop_during_rpc_discards_late_reply_and_does_not_close_from_main(self):
        transport = FakeTransport(delay=0.08)
        recorder = self.recorder(transport).start()
        time.sleep(0.02)
        status = recorder.stop()
        self.assertFalse(status['complete'])
        self.assertEqual(status['frames'], 0)
        self.assertEqual(transport.closed_on, self.constructed[0][2])

    def test_shutdown_is_bounded_if_transport_violates_timeout_contract(self):
        release = threading.Event()
        entered = threading.Event()
        class StuckTransport(FakeTransport):
            def get_robot_state(self):
                entered.set()
                release.wait(3)
                return reply()
        transport = StuckTransport()
        recorder = self.recorder(transport, rpc_timeout=0.01).start()
        self.assertTrue(entered.wait(1))
        started = time.monotonic()
        try:
            status = recorder.stop()
            self.assertLess(time.monotonic() - started, 1.5)
            self.assertFalse(status['complete'])
            self.assertIn('bounded shutdown', status['error'])
        finally:
            release.set()
            recorder._thread.join(1)
        self.assertFalse(recorder.stop()['complete'])

    def test_validate_all_vectors_and_source_timestamps(self):
        for key, value in [('joint_positions', [0] * 6), ('gripper_position', [float('nan')]),
                           ('cartesian_position', [0] * 5), ('gripper_position', True)]:
            state, stamps = reply()
            state[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_state((state, stamps))
        with self.assertRaises(ValueError):
            validate_state((reply()[0], {'bad': float('nan')}))

    def test_source_timestamp_requires_exact_integer_fields_and_int64_range(self):
        for seconds, nanos in [(None, 1), (1700000000.0, 1), (True, 1),
                               (-1, 1), (1, -1), (1, 1_000_000_000),
                               (1, 0.5), (2**63, 0)]:
            with self.subTest(seconds=seconds, nanos=nanos), self.assertRaises(ValueError):
                validate_state((reply()[0], {'robot_timestamp_seconds': seconds,
                                             'robot_timestamp_nanos': nanos}))
        sample = validate_state((reply()[0], {'robot_timestamp_seconds': 1700000000,
                                              'robot_timestamp_nanos': 999999999}))
        self.assertEqual(sample['robot_source_timestamp_ns'], 1700000000999999999)
        self.assertIs(type(sample['robot_source_timestamp_ns']), int)

    def wait_for_frames(self, recorder, count):
        deadline = time.perf_counter() + 2
        while recorder.status()['frames'] < count and time.perf_counter() < deadline:
            recorder.check_error()
            time.sleep(0.005)
        self.assertGreaterEqual(recorder.status()['frames'], count)

    def test_repeated_nuc_sensor_timestamp_preserves_each_real_rpc_reply(self):
        recorder = self.recorder(FakeTransport(response=reply())).start()
        recorder.wait_ready()
        self.wait_for_frames(recorder, 8)
        recorder.check_error()
        status = recorder.stop()
        self.assertTrue(status['complete'])
        self.assertIsNone(status['error'])
        rows = [json.loads(line) for line in (self.path / 'frames.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), status['frames'])
        self.assertEqual(status['source_timestamp_repeated_count'], len(rows) - 1)
        self.assertEqual(status['source_timestamp_max_consecutive_repeats'], len(rows) - 1)
        self.assertEqual(status['source_timestamp_regression_count'], 0)
        self.assertIn('NUC samples may repeat', status['sampling_semantics'])
        for row in rows:
            self.assertEqual(row['source_timestamps'], reply()[1])
            self.assertEqual(row['robot_source_timestamp_ns'], 1700000000000000001)
            self.assertEqual(row['joints'], [0.01] * 7)
        self.assertTrue(all(b['captured_at_monotonic'] > a['captured_at_monotonic']
                            for a, b in zip(rows, rows[1:])))

    def test_source_clock_repeats_and_regressions_are_diagnostic_without_dropping_replies(self):
        source_indices = [10, 10, 10, 11, 9, 9, 12, 12, 13]
        def response(call):
            state, stamps = reply(call)
            stamps['robot_timestamp_nanos'] = (source_indices[call - 1]
                                               if call <= len(source_indices) else 13 + call)
            return state, stamps
        recorder = self.recorder(FakeTransport(response=response)).start()
        recorder.wait_ready()
        self.wait_for_frames(recorder, len(source_indices))
        recorder.check_error()
        status = recorder.stop()
        self.assertTrue(status['complete'])
        self.assertIsNone(status['error'])
        self.assertEqual(status['source_timestamp_repeated_count'], 4)
        self.assertEqual(status['source_timestamp_regression_count'], 1)
        self.assertEqual(status['source_timestamp_max_consecutive_repeats'], 2)
        rows = [json.loads(line) for line in (self.path / 'frames.jsonl').read_text().splitlines()]
        for index, row in enumerate(rows, start=1):
            original = response(index)
            self.assertEqual(row['frame_index'], index - 1)
            self.assertEqual(row['source_timestamps'], original[1])
            self.assertEqual(row['robot_source_timestamp_ns'],
                             1700000000000000000 + original[1]['robot_timestamp_nanos'])
            self.assertEqual(row['joints'], original[0]['joint_positions'])

    def test_nonmonotonic_host_rpc_timing_still_fails(self):
        ticks = iter([0.0, 0.0, 0.0, 2.0, 1.0])
        recorder = self.recorder(FakeTransport(), clock=lambda: next(ticks)).start()
        with self.assertRaisesRegex(RuntimeError, 'Non-monotonic RPC timing'):
            recorder.wait_ready()
        status = recorder.stop()
        self.assertFalse(status['complete'])
        self.assertEqual(status['frames'], 0)

    def test_nonincreasing_host_midpoints_still_fail(self):
        # First RPC midpoint is 2.0; second is 1.0, despite valid individual
        # intervals and a normally increasing NUC source timestamp.
        ticks = iter([0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0, 1.0])
        recorder = self.recorder(FakeTransport(), clock=lambda: next(ticks)).start()
        self.assertTrue(recorder._done.wait(1))
        status = recorder.stop()
        self.assertFalse(status['complete'])
        self.assertEqual(status['frames'], 1)
        self.assertIn('State sample host timestamps did not increase', status['error'])

    def test_transport_only_connects_reads_and_closes_in_own_context(self):
        calls = []
        class Context:
            def __init__(self): calls.append('context')
            def term(self): calls.append('term')
        class Client:
            def __init__(self, **kwargs):
                calls.append(('client', kwargs['timeout'], kwargs['heartbeat'], isinstance(kwargs['context'], Context)))
            def connect(self, endpoint): calls.append(('connect', endpoint))
            def get_robot_state(self): calls.append('get_robot_state'); return reply()
            def close(self): calls.append('close')
        with patch.dict(sys.modules, zerorpc=SimpleNamespace(Context=Context, Client=Client)):
            transport = _StateTransport('tcp://127.0.0.1:4242', 0.5)
            self.assertEqual(transport.get_robot_state(), reply())
            transport.close()
        self.assertEqual(calls, ['context', ('client', 0.5, 20, True),
                                 ('connect', 'tcp://127.0.0.1:4242'), 'get_robot_state', 'close', 'term'])


if __name__ == '__main__':
    unittest.main()

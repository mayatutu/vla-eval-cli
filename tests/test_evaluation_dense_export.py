"""Offline real FFmpeg/Parquet validation of continuous-time dataset exports."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

import evaluation_dense_export as dense


class DenseExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.attempt = Path(self.directory.name)
        self.recording = self.attempt / 'recording'
        self.recording.mkdir()
        self.start, self.stop = 100.0, 100.15
        self.grid = [self.start + index / 30 for index in range(5)]
        self.stamps = [stamp + offset for stamp, offset in
                       zip(self.grid, [-0.004, 0.003, -0.002, 0.004, -0.001])]
        self.states = [{'frame_index': index, 'captured_at_monotonic': stamp,
                        'robot_source_timestamp_ns': 1791943200000000001 + index * 33333333,
                        'rpc_duration_seconds': 0.002,
                        'request_started_monotonic': stamp - 0.001,
                        'response_received_monotonic': stamp + 0.001,
                        'joints': [index / 10] * 7, 'gripper': [0.4],
                        'cartesian_position': [index / 100] * 6}
                       for index, stamp in enumerate(self.stamps)]
        self.write_states()
        self.images = {}
        statuses = {}
        for camera in dense.CAMERAS:
            directory = self.recording / 'continuous' / camera
            directory.mkdir(parents=True)
            rows = []
            for index, stamp in enumerate(self.stamps):
                pixels = np.full((32, 48, 3), index * 40, dtype=np.uint8)
                Image.fromarray(pixels).save(directory / f'{index:06d}.png')
                rows.append({'frame_index': index, 'file': f'{camera}/{index:06d}.png',
                             'camera': camera, 'serial': camera,
                             'sequence': 100 + index, 'flags': 0x2000,
                             'captured_at_monotonic': stamp, 'driver_time': stamp,
                             'received_at_monotonic': stamp + 0.001,
                             'processed_at_monotonic': stamp + 0.002,
                             'sampled_at_monotonic': stamp + 0.003,
                             'sequence_gap': 0})
            self.images[camera] = rows
            self.write_images(camera)
            statuses[camera] = {'frames': len(rows), 'sequence_gaps': 0}
        self.write_json(self.recording / 'continuous/continuous-status.json',
                        {'schema_version': 1, 'complete': True, 'error': None,
                         'cameras': statuses})
        actions = []
        for index, stamp in enumerate([self.start, self.start + 0.075]):
            actions.extend([{'event': 'execute_attempt', 'control_step': index,
                             'selected_command': [index + 0.2] * 8},
                            {'event': 'execute_succeeded', 'control_step': index,
                             'sent_at_monotonic': stamp,
                             'returned_at_monotonic': stamp + 0.002}])
        self.write_rows(self.attempt / 'actions.jsonl', actions)
        self.write_rows(self.recording / 'frames.jsonl', [{'frame': 0, 'seconds': 0.0,
            'joints': [0.1] * 7, 'gripper': [0.4], 'cartesian_position': [0.2] * 6,
            'action': [0.2] * 8}])
        self.write_json(self.recording / 'recording-status.json', {'complete': True, 'frames': 1})
        for camera in dense.CAMERAS:
            directory = self.recording / camera
            directory.mkdir()
            Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8)).save(directory / '000000.png')
        self.original_actions = (self.attempt / 'actions.jsonl').read_bytes()
        self.original_sparse = (self.recording / 'frames.jsonl').read_bytes()

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    @staticmethod
    def write_rows(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

    def write_states(self):
        self.write_rows(self.recording / 'state/frames.jsonl', self.states)
        self.write_json(self.recording / 'state/status.json',
                        {'complete': True, 'error': None, 'frames': len(self.states)})

    def write_images(self, camera):
        self.write_rows(self.recording / 'continuous' / camera / 'frames.jsonl', self.images[camera])

    def export(self, **kwargs):
        return dense.export_dense(self.recording, dense.CAMERAS, fps=30,
            robot_type='franka', task='test', metadata={'score': 2},
            start_monotonic=self.start, stop_monotonic=self.stop, **kwargs)

    def assert_raw_preserved(self):
        self.assertEqual(len(list((self.recording / 'continuous').rglob('*.png'))), 15)
        self.assertEqual(len(list(self.recording.rglob('*.png'))), 18)
        self.assertEqual((self.attempt / 'actions.jsonl').read_bytes(), self.original_actions)
        self.assertEqual((self.recording / 'frames.jsonl').read_bytes(), self.original_sparse)
        self.assertFalse((self.recording / '.dense-export.lock').exists())
        self.assertEqual(list(self.recording.glob('.dense-*')), [])

    def test_irregular_actual_samples_produce_real_duration_and_distinct_frames(self):
        output = self.export(cleanup_sources=False)
        table = pq.read_table(output / 'data/chunk-000/file-000.parquet').to_pydict()
        np.testing.assert_allclose(table['timestamp'], np.arange(5) / 30)
        np.testing.assert_allclose(table['observation.state_source_timestamp'], self.stamps)
        self.assertEqual(table['observation.state_source_index'], list(range(5)))
        self.assertEqual(table['observation.robot_source_timestamp_ns'],
                         [sample['robot_source_timestamp_ns'] for sample in self.states])
        raw_table = pq.read_table(output / 'data/chunk-000/file-000.parquet')
        self.assertEqual(str(raw_table.schema.field('observation.robot_source_timestamp_ns').type), 'int64')
        self.assertEqual(table['action_command_index'], [0, 0, 0, 1, 1])
        self.assertEqual(table['action_is_new'], [1, 0, 0, 1, 0])
        np.testing.assert_allclose(table['action'][3], [1.2] * 8)
        self.assertEqual(table['observation.state_is_new'], [1] * 5)
        manifest = json.loads((output / 'meta/alignment.json').read_text())
        self.assertEqual(manifest['output_frames'], 5)
        self.assertAlmostEqual(manifest['encoded_duration_seconds'], 5 / 30)
        for camera in dense.CAMERAS:
            self.assertEqual(table[f'observation.image_source_index.{camera}'], list(range(5)))
            video = output / f'videos/observation.images.{camera}/chunk-000/file-000.mp4'
            result = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video),
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True)
            decoded = np.frombuffer(result.stdout, np.uint8).reshape(5, 32, 48, 3)
            np.testing.assert_allclose(decoded.mean(axis=(1, 2, 3)), np.arange(5) * 40, atol=3)
        self.assert_raw_preserved()

    def test_retry_after_cleanup_reuses_verified_identical_dataset(self):
        output = self.export()
        self.assertEqual(list(self.recording.rglob('*.png')), [])
        before = {str(path.relative_to(output)): dense._digest(path)
                  for path in output.rglob('*') if path.is_file()}
        self.assertEqual(self.export(), output)
        after = {str(path.relative_to(output)): dense._digest(path)
                 for path in output.rglob('*') if path.is_file()}
        self.assertEqual(before, after)
        self.assertEqual((self.recording / 'frames.jsonl').read_bytes(), self.original_sparse)

    def test_missing_actual_state_is_marked_reused_in_export(self):
        self.states.pop()
        self.write_states()
        output = self.export()
        table = pq.read_table(output / 'data/chunk-000/file-000.parquet').to_pydict()
        self.assertEqual(len(table['timestamp']), 5)
        self.assertTrue(any(table['observation.state_source_reused']))

    def test_state_gap_rejects_and_keeps_sources(self):
        self.states[2]['captured_at_monotonic'] += 0.1
        self.write_states()
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            self.export()
        self.assert_raw_preserved()

    def test_monotonic_misalignment_is_reported(self):
        for sample in self.states:
            sample['captured_at_monotonic'] += 0.05
            sample['request_started_monotonic'] += 0.05
            sample['response_received_monotonic'] += 0.05
        self.write_states()
        result = dense.validate_streams(self.recording, dense.CAMERAS, fps=30,
                                       start_monotonic=self.start, stop_monotonic=self.stop)
        self.assertTrue(result['passed'])
        self.assertGreater(result['state']['max_abs_alignment_error_seconds'], 0.04)
        self.assert_raw_preserved()

    def test_camera_gap_rejects_instead_of_copying_frame(self):
        camera = dense.CAMERAS[0]
        self.images[camera][2]['sequence'] = self.images[camera][1]['sequence']
        self.write_images(camera)
        with self.assertRaisesRegex(ValueError, 'physical frame sequence'):
            self.export()
        self.assert_raw_preserved()

    def test_camera_clock_must_be_verified(self):
        camera = dense.CAMERAS[0]
        self.images[camera][0]['flags'] = 0
        self.write_images(camera)
        with self.assertRaisesRegex(ValueError, 'monotonic timestamp'):
            self.export()
        self.assert_raw_preserved()

    def test_slow_rpc_duration_is_preserved_in_export(self):
        self.states[1]['rpc_duration_seconds'] = 0.2
        self.states[1]['request_started_monotonic'] = self.states[1]['captured_at_monotonic'] - 0.1
        self.states[1]['response_received_monotonic'] = self.states[1]['captured_at_monotonic'] + 0.1
        self.write_states()
        output = self.export()
        table = pq.read_table(output / 'data/chunk-000/file-000.parquet').to_pydict()
        self.assertGreaterEqual(max(table['observation.state_rpc_duration_seconds']), 0.2)

    def test_no_command_before_tick_rejects_future_action(self):
        path = self.attempt / 'actions.jsonl'
        events = [json.loads(line) for line in path.read_text().splitlines()]
        events[1]['sent_at_monotonic'] += 0.01
        self.write_rows(path, events)
        with self.assertRaisesRegex(ValueError, 'No actually sent command'):
            self.export()
        self.assertFalse((self.recording / 'lerobot').exists())
        self.assertEqual(len(list((self.recording / 'continuous').rglob('*.png'))), 15)

    def test_probe_failure_preserves_raw_and_retry_is_possible(self):
        with patch.object(dense, '_probe', side_effect=ValueError('simulated bad video')):
            with self.assertRaisesRegex(ValueError, 'simulated bad video'):
                self.export()
        self.assertFalse((self.recording / 'lerobot').exists())
        self.assert_raw_preserved()
        self.assertTrue(self.export().is_dir())

    def test_changed_context_does_not_overwrite_existing_dataset(self):
        self.export()
        self.stop += 0.01
        with self.assertRaisesRegex(ValueError, 'differs; refusing'):
            self.export()

    def test_modified_output_is_detected_before_cleanup(self):
        output = self.export(cleanup_sources=False)
        with (output / 'meta/stats.json').open('a') as stream:
            stream.write(' ')
        with self.assertRaisesRegex(ValueError, 'incomplete or modified'):
            self.export()
        self.assert_raw_preserved()

    def test_incomplete_capture_never_publishes(self):
        path = self.recording / 'continuous/continuous-status.json'
        status = json.loads(path.read_text())
        status['complete'] = False
        self.write_json(path, status)
        with self.assertRaisesRegex(ValueError, 'did not finish completely'):
            self.export()
        self.assert_raw_preserved()

    def test_export_lock_prevents_concurrent_writers(self):
        lock = self.recording / '.dense-export.lock'
        lock.write_text('other writer')
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(lock.read_text(), 'other writer')

    def test_nearest_matching_can_explicitly_reuse_the_nearest_source(self):
        rows = [{'time': stamp} for stamp in [-0.03, 0.01, 0.07]]
        actual = dense._unique_nearest(rows, [0.0, 1 / 30], key='time',
                                       tolerance=1 / 30, label='test')
        self.assertEqual(actual, [1, 1])

    def test_readonly_gate_needs_no_action_log_or_dataset_export(self):
        (self.attempt / 'actions.jsonl').unlink()
        before = {str(path.relative_to(self.recording)): dense._digest(path)
                  for path in self.recording.rglob('*') if path.is_file()}
        result = dense.validate_streams(self.recording, dense.CAMERAS, fps=30,
                                       start_monotonic=self.start, stop_monotonic=self.stop)
        self.assertTrue(result['passed'])
        self.assertEqual(result['state']['unique_selected_frames'], 5)
        self.assertLess(result['state']['max_abs_alignment_error_seconds'], 0.01)
        for camera in dense.CAMERAS:
            self.assertEqual(result['cameras'][camera]['unique_selected_frames'], 5)
        after = {str(path.relative_to(self.recording)): dense._digest(path)
                 for path in self.recording.rglob('*') if path.is_file()}
        self.assertEqual(before, after)

    def test_state_camera_skew_remains_visible_in_metrics(self):
        camera = dense.CAMERAS[0]
        self.images[camera][0]['driver_time'] = self.start + 0.032
        self.images[camera][0]['captured_at_monotonic'] = self.start + 0.032
        self.write_images(camera)
        result = dense.validate_streams(self.recording, dense.CAMERAS, fps=30,
                                       start_monotonic=self.start, stop_monotonic=self.stop)
        self.assertGreater(result['max_state_image_timestamp_skew_seconds'], 0.03)
        self.assert_raw_preserved()

    def test_claimed_state_capture_time_must_match_observed_rpc_interval(self):
        self.states[1]['request_started_monotonic'] -= 0.01
        self.write_states()
        with self.assertRaisesRegex(ValueError, 'inconsistent RPC midpoint'):
            self.export()
        self.assert_raw_preserved()

    def test_unlisted_png_is_not_deleted(self):
        unrelated = self.recording / 'head_image/unlisted.png'
        Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8)).save(unrelated)
        expected = dense._digest(unrelated)
        self.export()
        self.assertEqual(list(self.recording.rglob('*.png')), [unrelated])
        self.assertEqual(dense._digest(unrelated), expected)

    def test_duplicate_robot_timestamp_is_preserved_and_reported(self):
        self.states[2]['robot_source_timestamp_ns'] = self.states[1]['robot_source_timestamp_ns']
        self.write_states()
        output = self.export()
        table = pq.read_table(output / 'data/chunk-000/file-000.parquet').to_pydict()
        stamps = table['observation.robot_source_timestamp_ns']
        self.assertEqual(stamps[1], stamps[2])
        metrics = dense._cadence_metrics(stamps, scale=1e-9)
        self.assertFalse(metrics['cadence_valid'])
        self.assertEqual(metrics['repeated_timestamp_count'], 1)

    def test_float_robot_nanoseconds_are_rejected_instead_of_losing_precision(self):
        self.states[0]['robot_source_timestamp_ns'] = float(self.states[0]['robot_source_timestamp_ns'])
        self.write_states()
        with self.assertRaisesRegex(ValueError, 'exact nonnegative int64'):
            self.export()
        self.assert_raw_preserved()

    def remove_legacy_action_images(self):
        for camera in dense.CAMERAS:
            (self.recording / camera / '000000.png').unlink()

    def test_explicit_numeric_only_actions_export_and_retry_without_legacy_pngs(self):
        self.remove_legacy_action_images()
        self.write_json(self.recording / 'policy.json', {'action_recording_images': False})
        output = self.export()
        self.assertEqual(self.export(), output)
        manifest = json.loads((output / 'meta/alignment.json').read_text())
        self.assertEqual(len(manifest['source_png_sha256']), 15)
        self.assertTrue(all(path.startswith('continuous/') for path in manifest['source_png_sha256']))
        self.assertEqual(list(self.recording.rglob('*.png')), [])
        self.assertEqual((self.recording / 'frames.jsonl').read_bytes(), self.original_sparse)
        self.assertEqual((self.attempt / 'actions.jsonl').read_bytes(), self.original_actions)

    def test_legacy_missing_pngs_still_refuse_export(self):
        self.remove_legacy_action_images()
        with self.assertRaises(FileNotFoundError):
            self.export()
        self.assertFalse((self.recording / 'lerobot').exists())
        self.assertEqual(len(list(self.recording.rglob('*.png'))), 15)

    def test_numeric_only_policy_must_be_literal_boolean_false(self):
        self.remove_legacy_action_images()
        for invalid in [True, 0, 'false', None]:
            with self.subTest(flag=invalid):
                self.write_json(self.recording / 'policy.json', {'action_recording_images': invalid})
                with self.assertRaises(FileNotFoundError):
                    self.export()
        self.assertEqual(len(list(self.recording.rglob('*.png'))), 15)

    def test_numeric_only_mode_still_requires_full_state_and_action_log(self):
        self.remove_legacy_action_images()
        self.write_json(self.recording / 'policy.json', {'action_recording_images': False})
        path = self.recording / 'frames.jsonl'
        original_row = json.loads(path.read_text())
        for key in ['joints', 'gripper', 'cartesian_position', 'action']:
            with self.subTest(missing=key):
                row = dict(original_row)
                row.pop(key)
                self.write_rows(path, [row])
                with self.assertRaises(ValueError):
                    self.export()
                self.assertFalse((self.recording / 'lerobot').exists())
                self.assertEqual(len(list(self.recording.rglob('*.png'))), 15)
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'numeric action recording is missing'):
            self.export()


if __name__ == '__main__':
    unittest.main()

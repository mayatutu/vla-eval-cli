"""Real-thread offline tests of cache-only continuous camera recording."""
from __future__ import annotations

import json
import io
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import zlib

import numpy as np
from PIL import Image

from evaluation_video import ContinuousCameraRecorder, ContinuousRecordingError, _write_rgb_png


class PngEncoderTests(unittest.TestCase):
    @staticmethod
    def encode(rgb):
        output = io.BytesIO()
        _write_rgb_png(output, rgb)
        return output.getvalue()

    def test_standard_chunks_crc_no_filter_and_exact_pixels(self):
        rng = np.random.default_rng(19)
        for pixels in (np.zeros((1, 1, 3), np.uint8),
                       np.full((7, 13, 3), (7, 91, 251), np.uint8),
                       rng.integers(0, 256, (360, 640, 3), dtype=np.uint8),
                       rng.integers(0, 256, (18, 26, 3), dtype=np.uint8)[::2, ::2, ::-1]):
            with self.subTest(shape=pixels.shape, contiguous=pixels.flags.c_contiguous):
                encoded = self.encode(pixels)
                self.assertEqual(encoded[:8], b'\x89PNG\r\n\x1a\n')
                offset, chunks = 8, []
                while offset < len(encoded):
                    length = struct.unpack('>I', encoded[offset:offset + 4])[0]
                    kind = encoded[offset + 4:offset + 8]
                    data = encoded[offset + 8:offset + 8 + length]
                    crc = struct.unpack('>I', encoded[offset + 8 + length:offset + 12 + length])[0]
                    self.assertEqual(crc, zlib.crc32(data, zlib.crc32(kind)) & 0xffffffff)
                    chunks.append((kind, data))
                    offset += 12 + length
                self.assertEqual(offset, len(encoded))
                self.assertEqual([kind for kind, _ in chunks], [b'IHDR', b'IDAT', b'IEND'])
                self.assertEqual(struct.unpack('>IIBBBBB', chunks[0][1]),
                                 (pixels.shape[1], pixels.shape[0], 8, 2, 0, 0, 0))
                scanlines = np.frombuffer(zlib.decompress(chunks[1][1]), np.uint8).reshape(pixels.shape[0], -1)
                self.assertTrue(np.all(scanlines[:, 0] == 0))
                np.testing.assert_array_equal(scanlines[:, 1:].reshape(pixels.shape), pixels)
                with Image.open(io.BytesIO(encoded)) as image:
                    image.load()
                    np.testing.assert_array_equal(np.asarray(image), pixels)

    def test_opencv_decodes_same_rgb_pixels(self):
        import cv2
        pixels = np.random.default_rng(20).integers(0, 256, (360, 640, 3), dtype=np.uint8)
        decoded = cv2.imdecode(np.frombuffer(self.encode(pixels), np.uint8), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(decoded[..., ::-1], pixels)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required for external decoder check')
    def test_ffmpeg_decodes_same_rgb_pixels(self):
        pixels = np.random.default_rng(21).integers(0, 256, (360, 640, 3), dtype=np.uint8)
        result = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error',
                                 '-f', 'image2pipe', '-i', 'pipe:0', '-frames:v', '1',
                                 '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
                                input=self.encode(pixels), capture_output=True, check=True)
        np.testing.assert_array_equal(np.frombuffer(result.stdout, np.uint8).reshape(pixels.shape), pixels)

    def test_invalid_png_inputs_fail_explicitly(self):
        for pixels in (np.zeros((2, 2, 4), np.uint8), np.zeros((2, 2, 3), np.float32),
                       np.zeros((0, 2, 3), np.uint8)):
            with self.subTest(shape=pixels.shape, dtype=pixels.dtype):
                with self.assertRaises(ValueError):
                    self.encode(pixels)


class FakeCamera:
    """Match the existing UvcCamera lock/latest/error publication contract."""
    def __init__(self, *, automatic=False, period=1 / 30):
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.closed = False
        self._stop = threading.Event()
        self._worker = None
        self._period = period
        self.publish(1)
        if automatic:
            self._worker = threading.Thread(target=self._produce, daemon=True)
            self._worker.start()

    def publish(self, sequence, *, flags=0x2000, stamp=None):
        if stamp is None:
            # Python 3.10 on Windows can return the same monotonic clock tick
            # for successive manual publications. Real UVC frames have distinct
            # driver timestamps. Wait for a real host-clock advance so the fake
            # remains both strictly increasing and never future-dated.
            with self.lock:
                previous = self.latest
            now = time.monotonic()
            if previous is not None and sequence != previous[1]['sequence']:
                deadline = time.perf_counter() + 0.2
                while now <= previous[1]['driver_time']:
                    if time.perf_counter() >= deadline:
                        raise RuntimeError('Test host monotonic clock did not advance')
                    time.sleep(0.001)
                    now = time.monotonic()
        else:
            now = stamp
        pixels = np.full((4, 6, 3), [sequence % 256, 80, 160], dtype=np.uint8)
        event = {'sequence': sequence, 'flags': flags, 'driver_time': now,
                 'received': now, 'processed': now}
        with self.lock:
            self.latest = (pixels, event)

    def _produce(self):
        sequence = 1
        while not self._stop.wait(self._period):
            sequence += 1
            self.publish(sequence)

    def close(self):
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=1)
        with self.lock:
            self.closed = True
            self.error = RuntimeError('Camera closed')


class FakeRobot:
    def __init__(self, cameras, prepare=None):
        self.environment = SimpleNamespace(camera_reader=SimpleNamespace(camera_dict=cameras))
        self._prepare_image = prepare or self._prepare

    @staticmethod
    def _prepare(pixels):
        return np.ascontiguousarray(pixels[..., ::-1])

    def get_observation(self):
        raise AssertionError('Continuous recorder must never perform robot RPC')

    def execute(self, _action):
        raise AssertionError('Continuous recorder must never command robot')

    def close(self):
        for camera in self.environment.camera_reader.camera_dict.values():
            camera.close()


class ContinuousVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cameras = []
        self.recorders = []

    def tearDown(self):
        for recorder in self.recorders:
            try:
                recorder.stop_capture()
                recorder.finish()
            except ContinuousRecordingError:
                pass
            self.assertFalse(any(worker.is_alive() for worker in recorder._samplers + recorder._writers))
        for camera in self.cameras:
            camera.close()
        self.temp.cleanup()

    def make(self, *, count=1, automatic=False, prepare=None, **kwargs):
        cameras = {str(index): FakeCamera(automatic=automatic) for index in range(count)}
        self.cameras.extend(cameras.values())
        robot = FakeRobot(cameras, prepare=prepare)
        recorder = ContinuousCameraRecorder(robot, self.root, {f'view{key}': key for key in cameras}, **kwargs)
        self.recorders.append(recorder)
        return recorder, robot

    @staticmethod
    def rows(recorder, name='view0'):
        return [json.loads(line) for line in (recorder.root / name / 'frames.jsonl').read_text().splitlines()]

    def wait_frames(self, recorder, minimum, timeout=1):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            recorder.check_error()
            if all(value['frames'] >= minimum for value in recorder.status()['cameras'].values()):
                return
            time.sleep(0.005)
        self.fail('Timed out waiting for real recorder threads')

    def wait_error(self, recorder, text, timeout=1):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                recorder.check_error()
            except ContinuousRecordingError as error:
                self.assertIn(text, str(error))
                return
            time.sleep(0.005)
        self.fail('Expected recording failure was not reported')

    def test_all_three_views_continue_during_inference_and_keep_real_times(self):
        recorder, robot = self.make(count=3, automatic=True)
        recorder.start()
        recorder.check_ready()
        # No action/RPC call: this interval represents waiting for inference.
        self.wait_frames(recorder, 5)
        recorder.stop_capture()
        robot.close()
        status = recorder.finish()
        self.assertTrue(status['recorded_complete'])
        for name in status['cameras']:
            rows = self.rows(recorder, name)
            self.assertGreaterEqual(len(rows), 5)
            self.assertEqual(len({row['sequence'] for row in rows}), len(rows))
            self.assertTrue(all(b['driver_time'] > a['driver_time'] for a, b in zip(rows, rows[1:])))
            self.assertGreater(rows[-1]['driver_time'] - rows[0]['driver_time'], 0.1)
            for row in rows:
                self.assertEqual(row['captured_at_monotonic'], row['driver_time'])
                self.assertGreaterEqual(row['sampled_at_monotonic'], row['driver_time'])
                self.assertEqual(row['flags'] & 0xE000, 0x2000)
                with Image.open(recorder.root / row['file']) as image:
                    self.assertEqual(image.getpixel((0, 0)), (160, 80, row['sequence'] % 256))
        self.assertEqual(json.loads((recorder.root / 'continuous-status.json').read_text()), status)

    def test_duplicate_cached_sequences_do_not_create_duplicate_frames(self):
        recorder, _ = self.make()
        recorder.start()
        recorder.check_ready()
        time.sleep(0.075)
        recorder.stop_capture()
        recorder.finish()
        self.assertEqual(len(self.rows(recorder)), 1)

    def test_skipped_sequence_is_explicit_and_never_claimed_smooth(self):
        recorder, _ = self.make()
        recorder.start()
        recorder.check_ready()
        self.cameras[0].publish(4)
        self.wait_frames(recorder, 2)
        recorder.stop_capture()
        result = recorder.finish()
        self.assertTrue(result['recorded_complete'])
        self.assertFalse(result['smooth_complete'])
        self.assertEqual(result['cameras']['view0']['sequence_gaps'], 2)
        self.assertEqual(self.rows(recorder)[1]['sequence_gap'], 2)

    def test_slow_writer_reports_queue_full_without_silent_loss(self):
        release = threading.Event()

        def slow_prepare(pixels):
            release.wait(timeout=1)
            return FakeRobot._prepare(pixels)

        recorder, _ = self.make(automatic=True, prepare=slow_prepare, queue_size=1)
        recorder.start()
        try:
            self.wait_error(recorder, 'queue is full')
        finally:
            release.set()
        recorder.stop_capture()
        with self.assertRaisesRegex(ContinuousRecordingError, 'queue is full'):
            recorder.finish()
        self.assertFalse(recorder.status()['recorded_complete'])

    def test_camera_failure_is_reported_and_remaining_writes_drained(self):
        recorder, _ = self.make()
        recorder.start()
        recorder.check_ready()
        with self.cameras[0].lock:
            self.cameras[0].error = RuntimeError('USB disconnected')
        self.wait_error(recorder, 'USB disconnected')
        with self.assertRaisesRegex(ContinuousRecordingError, 'USB disconnected'):
            recorder.finish()
        self.assertEqual(recorder.status()['cameras']['view0']['unsaved_frames'], 0)

    def test_frozen_camera_fails_instead_of_repeating_images(self):
        recorder, _ = self.make()
        recorder.start()
        recorder.check_ready()
        self.wait_error(recorder, 'stale')
        with self.assertRaisesRegex(ContinuousRecordingError, 'stale'):
            recorder.finish()
        self.assertEqual(len(self.rows(recorder)), 1)

    def test_writer_failure_is_reported(self):
        def broken_prepare(_pixels):
            raise OSError('No space left on device')
        recorder, _ = self.make(prepare=broken_prepare)
        recorder.start()
        with self.assertRaisesRegex(ContinuousRecordingError, 'No space left'):
            recorder.check_ready()
        with self.assertRaisesRegex(ContinuousRecordingError, 'No space left'):
            recorder.finish()
        self.assertFalse(recorder.status()['recorded_complete'])

    def test_non_monotonic_driver_clock_rejected(self):
        recorder, _ = self.make()
        self.cameras[0].publish(1, flags=0)
        recorder.start()
        with self.assertRaisesRegex(ContinuousRecordingError, 'CLOCK_MONOTONIC'):
            recorder.check_ready()
        with self.assertRaisesRegex(ContinuousRecordingError, 'CLOCK_MONOTONIC'):
            recorder.finish()

    def test_readers_stop_before_camera_close_and_finish_is_idempotent(self):
        recorder, robot = self.make(automatic=True)
        recorder.start()
        recorder.check_ready()
        recorder.stop_capture()
        self.assertFalse(any(worker.is_alive() for worker in recorder._samplers))
        robot.close()
        result = recorder.finish()
        self.assertTrue(result['recorded_complete'])
        self.assertEqual(result, recorder.finish())

    def test_refuses_existing_recording(self):
        recorder, _ = self.make()
        recorder.root.mkdir()
        with self.assertRaises(FileExistsError):
            recorder.start()
        # No threads were started, so there is nothing to drain.
        self.recorders.remove(recorder)


if __name__ == '__main__':
    unittest.main()

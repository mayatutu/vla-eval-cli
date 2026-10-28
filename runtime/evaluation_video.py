"""Record existing UVC camera caches independently of model/action progress.

This module never opens cameras or calls a robot RPC. The owner must stop the
snapshot threads before closing the existing robot/cameras; PNG writers can be
drained afterwards. Driver timestamps are host monotonic estimates, not evidence
of hardware exposure synchronization between the three cameras.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import queue
import struct
import threading
import time
import zlib

import numpy as np


def _write_rgb_png(stream, rgb):
    """Write exact RGB8 PNG pixels without adaptive scanline filtering.

    Pillow's compress_level=0 still evaluates PNG filters for every scanline.
    The installed OpenCV 4.11 does not expose its PNG filter option. This narrow
    RGB8 encoder uses only standard PNG chunks and zlib's uncompressed stream;
    it changes neither pixel values nor the existing image preparation path.
    """
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError('PNG input must be nonempty RGB uint8')
    height, width, _ = rgb.shape
    scanlines = np.empty((height, width * 3 + 1), dtype=np.uint8)
    scanlines[:, 0] = 0  # PNG filter method None, one byte before each row.
    scanlines[:, 1:] = rgb.reshape(height, width * 3)

    def chunk(kind, data):
        stream.write(struct.pack('>I', len(data)))
        stream.write(kind)
        stream.write(data)
        stream.write(struct.pack('>I', zlib.crc32(data, zlib.crc32(kind)) & 0xffffffff))

    stream.write(b'\x89PNG\r\n\x1a\n')
    chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    chunk(b'IDAT', zlib.compress(memoryview(scanlines), level=0))
    chunk(b'IEND', b'')


class ContinuousRecordingError(RuntimeError):
    """Continuous recording is incomplete; it must not be exported as complete."""


class ContinuousCameraRecorder:
    """Capture each new cached camera sequence, including during inference.

    ``cameras`` maps recording names (e.g. ``head_image``) to camera serials.
    Queues retain immutable-by-contract BGR buffers from UvcCamera.latest.
    UvcCamera creates each buffer with cv2.remap and replaces latest; it never
    mutates a published buffer. Conversion/resizing and PNG writing happen on
    dedicated writers and never on the action thread.
    """

    def __init__(self, robot, recording_dir, cameras, *, queue_size=24,
                 poll_hz=90.0, stop_timeout=3.0, writer_timeout=30.0):
        if not cameras or any(not isinstance(name, str) or not name
                              or Path(name).name != name or name in {'.', '..'}
                              or '/' in name or '\\' in name
                              for name in cameras):
            raise ValueError('cameras must map safe recording names to serials')
        if type(queue_size) is not int or queue_size < 1:
            raise ValueError('queue_size must be a positive integer')
        if not math.isfinite(poll_hz) or poll_hz <= 0:
            raise ValueError('poll_hz must be positive and finite')
        if min(stop_timeout, writer_timeout) <= 0:
            raise ValueError('join timeouts must be positive')
        self.root = Path(recording_dir) / 'continuous'
        self._prepare_image = robot._prepare_image
        available = robot.environment.camera_reader.camera_dict
        self._cameras = {name: available[str(serial)] for name, serial in cameras.items()}
        for camera in self._cameras.values():
            if not all(hasattr(camera, key) for key in ('lock', 'latest', 'error')):
                raise TypeError('Continuous recording requires the existing UvcCamera cache contract')
        self._serials = {name: str(serial) for name, serial in cameras.items()}
        self._queues = {name: queue.Queue(maxsize=queue_size) for name in cameras}
        self._done = {name: threading.Event() for name in cameras}
        self._ready = {name: threading.Event() for name in cameras}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._period = 1.0 / poll_hz
        self._stop_timeout = stop_timeout
        self._writer_timeout = writer_timeout
        self._samplers = []
        self._writers = []
        self._error = None
        self._started = False
        self._finished = False
        self._started_at = None
        self._stopped_at = None
        self._stats = {
            name: {'serial': self._serials[name], 'frames': 0, 'enqueued_frames': 0,
                   'sequence_gaps': 0, 'first_driver_time': None, 'last_driver_time': None,
                   'max_interval_s': 0.0, 'max_queue_depth_observed': 0,
                   'max_prepare_seconds': 0.0, 'max_png_write_seconds': 0.0,
                   'total_prepare_seconds': 0.0, 'total_png_write_seconds': 0.0}
            for name in cameras
        }

    def _fail(self, error):
        with self._lock:
            if self._error is None:
                self._error = error
        self._stop.set()

    def check_error(self):
        with self._lock:
            error = self._error
        if error is not None:
            raise ContinuousRecordingError(str(error)) from error

    def start(self):
        if self._started:
            raise RuntimeError('Continuous recorder already started')
        # Refuse to overwrite any prior data, even an empty previous directory.
        self.root.mkdir(parents=True, exist_ok=False)
        self._started = True
        self._started_at = time.monotonic()
        try:
            for name in self._cameras:
                (self.root / name).mkdir()
                writer = threading.Thread(target=self._write, args=(name,),
                                          name='eval-video-write-' + name, daemon=True)
                sampler = threading.Thread(target=self._sample, args=(name,),
                                           name='eval-video-read-' + name, daemon=True)
                writer.start()
                self._writers.append(writer)
                sampler.start()
                self._samplers.append(sampler)
        except BaseException as error:
            self._fail(error)
            # A writer whose sampler could not start still needs a completion
            # signal. Existing samplers see _stop and do not enqueue afterwards.
            for name in self._cameras:
                self._done[name].set()
            self.stop_capture()
            self.finish()
            raise

    def check_ready(self, timeout=2.0):
        """Wait until every camera has saved one frame, or report its failure."""
        if not self._started:
            raise RuntimeError('Continuous recorder has not started')
        deadline = time.monotonic() + timeout
        while not all(event.is_set() for event in self._ready.values()):
            self.check_error()
            if self._stop.is_set() or time.monotonic() >= deadline:
                error = ContinuousRecordingError('Continuous cameras did not become ready')
                self._fail(error)
                self.check_error()
            self._stop.wait(min(0.01, max(0.0, deadline - time.monotonic())))
        self.check_error()

    @staticmethod
    def _event(event):
        sequence = event['sequence']
        if isinstance(sequence, bool) or not isinstance(sequence, (int, np.integer)):
            raise ValueError('Camera sequence must be an integer')
        sequence = int(sequence)
        if not 0 <= sequence < 2**32:
            raise ValueError('Camera sequence must be an unsigned 32-bit value')
        flags = int(event['flags'])
        if flags & 0xE000 != 0x2000 or flags & 0x40:
            raise ValueError('Camera driver frame is invalid or timestamp is not CLOCK_MONOTONIC')
        values = {key: float(event[key]) for key in ('driver_time', 'received', 'processed')}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError('Camera timestamps must be finite')
        return sequence, flags, values

    def _sample(self, name):
        previous_sequence = previous_time = None
        index = 0
        try:
            while not self._stop.is_set():
                camera = self._cameras[name]
                # The production UvcCamera producer only swaps this tuple after
                # creating a fresh cv2.remap result. Retaining the array prevents
                # freeing it; no lock is held for conversion or writing.
                with camera.lock:
                    if camera.error is not None:
                        raise ContinuousRecordingError(f'Camera failed: {camera.error}')
                    cached = camera.latest
                    if cached is None:
                        raise ContinuousRecordingError('Camera has no cached frame')
                    # Bind age to the selected cache while publication is locked.
                    sampled_at = time.monotonic()
                    pixels, event = cached
                    event = dict(event)
                if self._stop.is_set():
                    break
                sequence, flags, stamps = self._event(event)
                driver_time = stamps['driver_time']
                age = sampled_at - driver_time
                if not 0 <= age <= 0.2:
                    reason = 'stale' if age > 0.2 else 'from the future'
                    raise ContinuousRecordingError(
                        f'Camera cached frame is {reason}: age_s={age:.6f}, '
                        f'sequence={sequence}, '
                        f'driver_to_received_s={stamps["received"] - driver_time:.6f}, '
                        f'received_to_processed_s={stamps["processed"] - stamps["received"]:.6f}, '
                        f'processed_to_sampled_s={sampled_at - stamps["processed"]:.6f}, '
                        'allowed_age_s=[0.000000, 0.200000]')
                sequence_gap = 0
                if previous_sequence is not None:
                    advance = (sequence - previous_sequence) % 2**32
                    if advance == 0:
                        if driver_time != previous_time:
                            raise ValueError('Camera timestamp changed without a new sequence')
                        self._stop.wait(self._period)
                        continue
                    if advance > 100000 or driver_time <= previous_time:
                        raise ValueError('Non-increasing camera sequence or driver timestamp')
                    sequence_gap = advance - 1
                pixels = np.asarray(pixels)
                if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
                    raise ValueError('Camera snapshot must be a BGR uint8 image')
                row = {
                    'frame_index': index, 'file': f'{name}/{index:06d}.png',
                    'camera': name, 'serial': self._serials[name], 'sequence': sequence,
                    'flags': flags,
                    'captured_at_monotonic': driver_time, 'driver_time': driver_time,
                    'received_at_monotonic': stamps['received'],
                    'processed_at_monotonic': stamps['processed'],
                    'sampled_at_monotonic': sampled_at, 'sequence_gap': sequence_gap,
                }
                try:
                    self._queues[name].put_nowait((pixels, row))
                except queue.Full as error:
                    raise ContinuousRecordingError(f'{name}: PNG write queue is full; recording incomplete') from error
                with self._lock:
                    stats = self._stats[name]
                    stats['enqueued_frames'] += 1
                    stats['max_queue_depth_observed'] = max(stats['max_queue_depth_observed'],
                                                            self._queues[name].qsize())
                    stats['sequence_gaps'] += sequence_gap
                    if stats['first_driver_time'] is None:
                        stats['first_driver_time'] = driver_time
                    if previous_time is not None:
                        stats['max_interval_s'] = max(stats['max_interval_s'], driver_time - previous_time)
                    stats['last_driver_time'] = driver_time
                previous_sequence, previous_time = sequence, driver_time
                index += 1
                self._stop.wait(self._period)
        except BaseException as error:
            self._fail(ContinuousRecordingError(f'{name}: {error}'))
        finally:
            self._done[name].set()

    def _write(self, name):
        work = self._queues[name]
        try:
            with (self.root / name / 'frames.jsonl').open('x', encoding='utf-8') as stream:
                while not self._done[name].is_set() or not work.empty():
                    try:
                        pixels, row = work.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    try:
                        prepare_started = time.perf_counter()
                        rgb = np.asarray(self._prepare_image(pixels))
                        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
                            raise ValueError('Prepared camera image must be nonempty RGB uint8')
                        png_started = time.perf_counter()
                        with (self.root / row['file']).open('xb') as image_file:
                            _write_rgb_png(image_file, rgb)
                        png_finished = time.perf_counter()
                        stream.write(json.dumps(row, allow_nan=False) + '\n')
                        stream.flush()
                        with self._lock:
                            stats = self._stats[name]
                            stats['frames'] += 1
                            preparation = png_started - prepare_started
                            writing = png_finished - png_started
                            stats['max_prepare_seconds'] = max(stats['max_prepare_seconds'], preparation)
                            stats['max_png_write_seconds'] = max(stats['max_png_write_seconds'], writing)
                            stats['total_prepare_seconds'] += preparation
                            stats['total_png_write_seconds'] += writing
                        self._ready[name].set()
                    finally:
                        work.task_done()
        except BaseException as error:
            self._fail(ContinuousRecordingError(f'{name}: PNG writer failed: {error}'))

    def stop_capture(self):
        """Stop/join cache readers before the owner closes robot/cameras."""
        self._stop.set()
        deadline = time.monotonic() + self._stop_timeout
        for worker in self._samplers:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
        alive = [worker.name for worker in self._samplers if worker.is_alive()]
        if alive:
            error = ContinuousRecordingError('Camera cache readers did not stop: ' + ', '.join(alive))
            self._fail(error)
            raise error
        if self._started and self._stopped_at is None:
            self._stopped_at = time.monotonic()

    def status(self):
        with self._lock:
            cameras = {name: dict(stats) for name, stats in self._stats.items()}
            error = None if self._error is None else str(self._error)
        for name, stats in cameras.items():
            stats['unsaved_frames'] = stats['enqueued_frames'] - stats['frames']
        return {
            'schema_version': 1, 'complete': self._finished and error is None,
            'recorded_complete': self._finished and error is None,
            'smooth_complete': self._finished and error is None and
                               all(stats['sequence_gaps'] == 0 for stats in cameras.values()),
            'error': error, 'started_at_monotonic': self._started_at,
            'stopped_at_monotonic': self._stopped_at, 'cameras': cameras,
            'png_encoding': 'RGB8 PNG; filter None on every row; zlib level 0; original pixels preserved',
            'timestamp_semantics': 'driver_time is host monotonic estimated capture time; cameras are not hardware synchronized',
        }

    def finish(self):
        """Drain writers after stop_capture (safe to call after robot.close)."""
        if self._finished:
            self.check_error()
            return self.status()
        if not self._started:
            raise RuntimeError('Continuous recorder has not started')
        self.stop_capture()
        deadline = time.monotonic() + self._writer_timeout
        for worker in self._writers:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
        alive = [worker.name for worker in self._writers if worker.is_alive()]
        if alive:
            self._fail(ContinuousRecordingError('PNG writers did not stop: ' + ', '.join(alive)))
        with self._lock:
            incomplete = [name for name, stats in self._stats.items()
                          if stats['frames'] == 0 or stats['frames'] != stats['enqueued_frames']]
        if incomplete:
            self._fail(ContinuousRecordingError('Incomplete continuous streams: ' + ', '.join(incomplete)))
        self._finished = True
        result = self.status()
        with (self.root / 'continuous-status.json').open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        self.check_error()
        return result

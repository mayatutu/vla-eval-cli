"""Per-trial multi-camera recording with capture timing and restartable encoding."""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image


class TrialRecorder:
    def __init__(self, path: Path, cameras: list[str], *, queue_size: int = 256):
        if shutil.which('ffmpeg') is None:
            raise RuntimeError('Install ffmpeg before starting a trial')
        self.path = path
        self.cameras = cameras
        path.mkdir(parents=True, exist_ok=True)
        self.frames = 0
        self.started = time.monotonic()
        if queue_size < 1:
            raise ValueError('Recording queue_size must be positive')
        self._queue = queue.Queue(maxsize=queue_size)
        self._error = None
        self._closed = False
        self._worker = threading.Thread(target=self._write_loop, name='colosseum-recording')
        self._worker.start()

    def _check_error(self):
        if self._error is not None:
            raise RuntimeError('Recording failed; trial video is incomplete') from self._error

    def add(self, observation, action=None, *, captured_at=None):
        """Copy buffers and enqueue without waiting for compression or disk I/O."""
        if self._closed:
            raise RuntimeError('Recorder is closed')
        self._check_error()
        stamp = (time.monotonic() if captured_at is None else captured_at) - self.started
        images = {}
        for camera in self.cameras:
            if camera not in observation.images:
                raise ValueError(f'Missing required camera: {camera}')
            images[camera] = np.array(observation.images[camera], dtype=np.uint8, copy=True)
        record = {'frame': self.frames, 'seconds': stamp,
                  'joints': np.asarray(observation.joints).tolist(),
                  'gripper': np.asarray(observation.gripper).tolist(),
                  'cartesian_position': np.asarray(observation.cartesian_position).tolist(),
                  'action': None if action is None else np.asarray(action).tolist()}
        try:
            self._queue.put_nowait((images, record))
        except queue.Full as exc:
            self._error = RuntimeError('Recording queue full: disk writer cannot keep up')
            raise self._error from exc
        self.frames += 1

    def _write_frame(self, images, record):
        for camera, pixels in images.items():
            folder = self.path / camera
            folder.mkdir(exist_ok=True)
            Image.fromarray(pixels).save(folder / f"{record['frame']:06d}.png", compress_level=0)
        with (self.path / 'frames.jsonl').open('a') as output:
            output.write(json.dumps(record) + '\n')

    def _write_loop(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if self._error is None:
                    self._write_frame(*item)
            except Exception as exc:
                self._error = exc
            finally:
                self._queue.task_done()

    def close(self):
        """Drain accepted frames before encoding. Call off the asyncio event loop."""
        if not self._closed:
            self._closed = True
            self._queue.put(None)
            self._worker.join()
            (self.path / 'recording-status.json').write_text(json.dumps({
                'complete': self._error is None, 'frames': self.frames}) + '\n')
        self._check_error()

    @staticmethod
    def encode(path: Path, cameras: list[str]) -> dict[str, Path]:
        status = path / 'recording-status.json'
        if status.exists() and not json.loads(status.read_text())['complete']:
            raise RuntimeError('Cannot encode an incomplete recording')
        records = [json.loads(line) for line in (path / 'frames.jsonl').read_text().splitlines()]
        if not records:
            raise ValueError('No recorded frames')
        videos = {}
        for camera in cameras:
            folder = path / camera
            lines = []
            for index, row in enumerate(records):
                duration = records[index+1]['seconds'] - row['seconds'] if index+1 < len(records) else 1/15
                lines.extend([f"file '{row['frame']:06d}.png'", f'duration {max(duration, .001):.6f}'])
            lines.append(f"file '{records[-1]['frame']:06d}.png'")
            (folder / 'frames.txt').write_text('\n'.join(lines) + '\n')
            temp = folder / 'video.pending.mp4'
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat',
                '-safe', '1', '-i', 'frames.txt', '-vsync', 'vfr', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', temp.name],
                cwd=folder, check=True)
            final = folder / 'video.mp4'
            temp.replace(final)
            videos[camera] = final
        return videos

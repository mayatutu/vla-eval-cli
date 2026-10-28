"""CPU-only ZED stereo capture. Importing this module never opens hardware."""
from __future__ import annotations

import ctypes as C
import errno
import hashlib
import json
import os
from pathlib import Path
import select
import threading
import time

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass


# Linux x86_64 UAPI. Fixed-width timeval avoids Windows c_long layout in tests.
class Timeval(C.Structure):
    _fields_ = [('sec', C.c_int64), ('usec', C.c_int64)]


class Timecode(C.Structure):
    _fields_ = [('type', C.c_uint32), ('flags', C.c_uint32),
                ('frames', C.c_uint8), ('seconds', C.c_uint8),
                ('minutes', C.c_uint8), ('hours', C.c_uint8), ('userbits', C.c_uint8 * 4)]


class BufferMemory(C.Union):
    _fields_ = [('offset', C.c_uint32), ('userptr', C.c_uint64)]


class Buffer(C.Structure):
    _fields_ = [('index', C.c_uint32), ('type', C.c_uint32), ('bytesused', C.c_uint32),
                ('flags', C.c_uint32), ('field', C.c_uint32), ('timestamp', Timeval),
                ('timecode', Timecode), ('sequence', C.c_uint32), ('memory', C.c_uint32),
                ('m', BufferMemory), ('length', C.c_uint32), ('reserved2', C.c_uint32),
                ('request_fd', C.c_int32)]


class PixelFormat(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in
                ('width', 'height', 'pixelformat', 'field', 'bytesperline', 'sizeimage',
                 'colorspace', 'priv', 'flags', 'ycbcr_enc', 'quantization', 'xfer_func')]


class FormatUnion(C.Union):
    _fields_ = [('pix', PixelFormat), ('raw', C.c_uint8 * 200), ('align', C.c_uint64)]


class Format(C.Structure):
    _fields_ = [('type', C.c_uint32), ('fmt', FormatUnion)]


class Fraction(C.Structure):
    _fields_ = [('numerator', C.c_uint32), ('denominator', C.c_uint32)]


class CaptureParm(C.Structure):
    _fields_ = [('capability', C.c_uint32), ('capturemode', C.c_uint32),
                ('timeperframe', Fraction), ('extendedmode', C.c_uint32),
                ('readbuffers', C.c_uint32), ('reserved', C.c_uint32 * 4)]


class ParmUnion(C.Union):
    _fields_ = [('capture', CaptureParm), ('raw', C.c_uint8 * 200)]


class StreamParm(C.Structure):
    _fields_ = [('type', C.c_uint32), ('parm', ParmUnion)]


class RequestBuffers(C.Structure):
    _fields_ = [('count', C.c_uint32), ('type', C.c_uint32), ('memory', C.c_uint32),
                ('capabilities', C.c_uint32), ('flags', C.c_uint8), ('reserved', C.c_uint8 * 3)]


def request(number, structure, direction=3):
    return (direction << 30) | (C.sizeof(structure) << 16) | (ord('V') << 8) | number


def ioctl(fd, number, value, direction=3):
    import fcntl
    data = bytearray(C.string_at(C.addressof(value), C.sizeof(value)))
    fcntl.ioctl(fd, request(number, type(value), direction), data, True)
    C.memmove(C.addressof(value), bytes(data), len(data))


def load_config(path):
    import yaml
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get('version') != 1:
        raise CameraError('Expected version: 1 camera configuration')
    cameras = config.get('cameras', [])
    if not isinstance(cameras, list) or not cameras:
        raise CameraError('No cameras configured')
    ids, devices = set(), set()
    for original in cameras:
        row = dict(original)
        serial = str(row['serial'])
        device = str(row['device'])
        if not serial.isdigit() or serial in ids or device in devices:
            raise CameraError('Camera IDs and USB paths must be unique')
        if not device.startswith('/dev/v4l/by-path/') or not device.endswith('video-index0'):
            raise CameraError('Use a fixed /dev/v4l/by-path capture endpoint')
        if row.get('identity_confirmed') is not True:
            raise CameraError('Confirm camera identity before loading calibration: ' + serial)
        calibration = (path.parent / row['calibration']).resolve()
        if not calibration.is_file() or hashlib.sha256(calibration.read_bytes()).hexdigest() != row['calibration_sha256']:
            raise CameraError('Missing or changed calibration for ' + serial)
        row.update(serial=serial, device=device, calibration=str(calibration))
        ids.add(serial); devices.add(device)
        yield row


class Rectifier:
    """Maps are generated and checked against SDK/SVO references offline."""
    def __init__(self, path, serial):
        with np.load(path, allow_pickle=False) as data:
            if str(data['serial'].item()) != str(serial):
                raise CameraError('Calibration belongs to another camera')
            if str(data['pixel_order'].item()) != 'BGR':
                raise CameraError('Expected BGR calibration contract')
            self.map_x = data['map_x'].copy()
            self.map_y = data['map_y'].copy()
            self.K = data['K'].copy()
        if self.map_x.shape != (720, 1280) or self.map_y.shape != (720, 1280):
            raise CameraError('Calibration must be for 1280x720 left images')
        if self.map_x.dtype != np.float32 or self.map_y.dtype != np.float32:
            raise CameraError('Calibration maps must be float32')
        if self.K.shape != (3, 3) or not all(np.isfinite(x).all() for x in [self.K, self.map_x, self.map_y]):
            raise CameraError('Invalid calibration values')

    def apply(self, stereo_yuyv):
        if stereo_yuyv.shape != (720, 2560, 2) or stereo_yuyv.dtype != np.uint8:
            raise CameraError('Expected 2560x720 YUYV stereo frame')
        # Select complete YUYV pixel pairs from the left half before conversion.
        left = np.ascontiguousarray(stereo_yuyv[:, :1280])
        bgr = cv2.cvtColor(left, cv2.COLOR_YUV2BGR_YUY2)
        return cv2.remap(bgr, self.map_x, self.map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)


def validate_config(path):
    rows = list(load_config(path))
    for row in rows:
        Rectifier(row['calibration'], row['serial'])
    return rows


class NativeStream:
    WIDTH, HEIGHT, BYTES = 2560, 720, 2560 * 720 * 2

    def __init__(self, device):
        self.device, self.fd = device, None
        self.buffers, self.streaming = [], False
        self.old_format = self.old_parm = None

    def open(self):
        import mmap
        import platform
        if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'amd64'):
            raise CameraError('This V4L2 UAPI implementation targets Linux x86_64')
        resolved = Path(self.device).resolve(strict=True)
        if resolved.parent != Path('/dev') or not resolved.name.startswith('video'):
            raise CameraError('Unexpected capture device')
        self.fd = os.open(self.device, os.O_RDWR | os.O_NONBLOCK)
        try:
            self.old_format = Format(type=1); ioctl(self.fd, 4, self.old_format)
            self.old_parm = StreamParm(type=1); ioctl(self.fd, 21, self.old_parm)
            fmt = Format(type=1)
            fmt.fmt.pix = PixelFormat(width=self.WIDTH, height=self.HEIGHT,
                pixelformat=int.from_bytes(b'YUYV', 'little'), field=1)
            ioctl(self.fd, 5, fmt)
            p = fmt.fmt.pix
            if (p.width, p.height, p.pixelformat, p.bytesperline, p.sizeimage) != (
                    self.WIDTH, self.HEIGHT, int.from_bytes(b'YUYV', 'little'), self.WIDTH*2, self.BYTES):
                raise CameraError('Driver did not accept the exact YUYV layout')
            parm = StreamParm(type=1)
            parm.parm.capture.timeperframe = Fraction(1, 30)
            ioctl(self.fd, 22, parm)
            f = parm.parm.capture.timeperframe
            if not f.numerator or abs(f.denominator/f.numerator - 30) > .01:
                raise CameraError('Driver did not accept 30 FPS')
            buffers = RequestBuffers(count=4, type=1, memory=1)
            ioctl(self.fd, 8, buffers)
            if buffers.count < 2 or buffers.count > 8:
                raise CameraError('Unexpected V4L2 buffer count')
            for index in range(buffers.count):
                buf = Buffer(index=index, type=1, memory=1)
                ioctl(self.fd, 9, buf)
                if buf.length < self.BYTES:
                    raise CameraError('Short V4L2 mmap allocation')
                self.buffers.append(mmap.mmap(self.fd, buf.length, mmap.MAP_SHARED,
                    mmap.PROT_READ | mmap.PROT_WRITE, offset=buf.m.offset))
                ioctl(self.fd, 15, buf)
            ioctl(self.fd, 18, C.c_uint32(1), direction=1)
            self.streaming = True
        except BaseException:
            self.close()
            raise

    def read(self, timeout=.1):
        if not select.select([self.fd], [], [], timeout)[0]:
            return None
        buf = Buffer(type=1, memory=1)
        try:
            ioctl(self.fd, 17, buf)
        except OSError as error:
            if error.errno == errno.EAGAIN:
                return None
            raise
        try:
            if buf.index >= len(self.buffers):
                raise CameraError('Invalid dequeued buffer index')
            received = time.monotonic()
            stamp = buf.timestamp.sec + buf.timestamp.usec / 1e6
            raw = bytes(self.buffers[buf.index][:buf.bytesused])
            event = dict(sequence=int(buf.sequence), bytesused=int(buf.bytesused),
                         flags=int(buf.flags), driver_time=stamp, received=received)
            # CLOCK_MONOTONIC is required to validate age without inventing sensor time.
            if buf.bytesused == 0:
                return None
            if (buf.flags & 0xE000) != 0x2000:
                raise CameraError('Driver timestamp is not CLOCK_MONOTONIC')
            if buf.flags & 0x40 or buf.bytesused != self.BYTES:
                error = CameraError('V4L2 delivered an error or incomplete frame')
                error.raw, error.event = raw, event
                raise error
            if stamp <= 0 or stamp > received + .01:
                raise CameraError('Invalid driver timestamp')
            return np.frombuffer(raw, dtype=np.uint8).reshape(self.HEIGHT, self.WIDTH, 2), event
        finally:
            ioctl(self.fd, 15, buf)

    def close(self):
        if self.fd is None:
            return
        errors = []
        try:
            if self.streaming:
                try: ioctl(self.fd, 19, C.c_uint32(1), direction=1)
                except OSError as e: errors.append(str(e))
            for buffer in self.buffers:
                buffer.close()
            self.buffers.clear()
            try: ioctl(self.fd, 8, RequestBuffers(count=0, type=1, memory=1))
            except OSError as e: errors.append(str(e))
            for number, original in [(5, self.old_format), (22, self.old_parm)]:
                if original is not None:
                    try: ioctl(self.fd, number, original)
                    except OSError as e: errors.append(str(e))
        finally:
            os.close(self.fd)
            self.fd, self.streaming = None, False
        if errors:
            raise CameraError('Capture settings restoration failed: ' + '; '.join(errors))


class UvcCamera:
    def __init__(self, config, *, stream_factory=NativeStream, clock=time.monotonic):
        self.serial_number = str(config['serial'])
        self.config = config
        self.rectifier = Rectifier(config['calibration'], self.serial_number)
        self.stream = stream_factory(config['device'])
        self.clock = clock
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.worker = None
        self.stats = dict(frames=0, sequence_gaps=0, max_interval_s=0.0)

    def start(self):
        if self.worker is not None:
            raise CameraError('Camera already started')
        self.stream.open()
        self.worker = threading.Thread(target=self._capture, name='uvc-'+self.serial_number, daemon=True)
        self.worker.start()

    def _capture(self):
        previous = None
        try:
            while not self.stop.is_set():
                value = self.stream.read()
                if value is None:
                    continue
                raw, event = value
                pixels = self.rectifier.apply(raw)
                event['processed'] = self.clock()
                if previous is not None:
                    gap = (event['sequence']-previous['sequence']) % 2**32
                    if gap == 0 or gap > 100000:
                        raise CameraError('Non-increasing camera sequence')
                    self.stats['sequence_gaps'] += gap-1
                    interval = event['driver_time']-previous['driver_time']
                    if interval <= 0:
                        raise CameraError('Non-increasing camera timestamp')
                    self.stats['max_interval_s'] = max(self.stats['max_interval_s'], interval)
                previous = event
                with self.lock:
                    self.latest = pixels, event
                    self.stats['frames'] += 1
        except BaseException as error:
            with self.lock:
                self.error = error

    def snapshot(self):
        with self.lock:
            if self.error is not None:
                raise CameraError('Camera '+self.serial_number+' failed: '+str(self.error)) from self.error
            if self.latest is None:
                raise CameraError('Camera '+self.serial_number+' has no frame')
            pixels, event = self.latest
            age = self.clock() - event['driver_time']
            if not 0 <= age <= .2:
                raise CameraError('Camera '+self.serial_number+' frame is stale: %.3fs' % age)
            return pixels.copy(), dict(event)

    def get_intrinsics(self):
        return {self.serial_number+'_left': {'cameraMatrix': self.rectifier.K.copy(), 'distCoeffs': np.zeros(5)}}

    def close(self):
        self.stop.set()
        if self.worker is not None:
            self.worker.join(timeout=2)
            if self.worker.is_alive():
                raise CameraError('Capture worker did not stop')
        self.stream.close()


class UvcMultiCameraWrapper:
    def __init__(self, config_path):
        self.camera_dict = {}
        try:
            for config in validate_config(config_path):
                camera = UvcCamera(config)
                self.camera_dict[camera.serial_number] = camera
            for camera in self.camera_dict.values():
                camera.start()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if all(c.latest is not None or c.error is not None for c in self.camera_dict.values()):
                    break
                time.sleep(.01)
            self.read_cameras()
        except BaseException:
            self.disable_cameras()
            raise

    def read_cameras(self):
        images, stamps = {}, {}
        for serial, camera in self.camera_dict.items():
            pixels, event = camera.snapshot()
            images[serial+'_left'] = pixels
            # These are monotonic driver/host times, not hardware exposure timestamps.
            stamps[serial+'_frame_received'] = int(event['received']*1000)
            stamps[serial+'_estimated_capture'] = int(event['driver_time']*1000)
            stamps[serial+'_sequence'] = event['sequence']
        return {'image': images}, stamps

    def get_camera(self, camera_id):
        return self.camera_dict[camera_id]

    def disable_cameras(self):
        errors = []
        for camera in self.camera_dict.values():
            try: camera.close()
            except Exception as error: errors.append(str(error))
        if errors:
            raise CameraError('; '.join(errors))

    def start_recording(self, *_args, **_kwargs):
        raise CameraError('CPU backend does not support SDK SVO recording; use Client recording')

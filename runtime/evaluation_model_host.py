"""Owned model lifecycle over SSH stdio. Importing this module does nothing.

This is an explicitly launched AI-host helper, not a daemon or an installer.
It never stops a process it did not create, never changes existing suite files,
and never opens another listening port. EOF terminates its own model child.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import uuid


DEFAULT_SUITE = str(Path(__file__).resolve().parent)
DEFAULT_OUTPUT = str(Path.home() / 'vla-eval-runs/models')
KILL_SIGNAL = getattr(signal, 'SIGKILL', 9)  # Host runs on Linux; tests also run on Windows.
BUSY_PORTS = re.compile(r':(?:8000|91[0-2][0-9])\b')
MODEL_MARKERS = ('molmoact2_local_server.py', 'pi05_lerobot_local_server.py',
                 'lingbot_server.py', 'drawer_service.py', 'g05_server.py')


class BusyError(RuntimeError):
    """A guard is busy/uncertain; no unrelated process may be changed."""


def _command(argv):
    return subprocess.run(argv, check=True, text=True, capture_output=True,
                          timeout=10).stdout


def _process_identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    return {'start_ticks': fields[19], 'group': int(fields[2]),
            'session': int(fields[3])}


def _web_idle(state):
    return (isinstance(state, dict) and
            ((state.get('state') == 'idle' and state.get('running') is False) or
             (state.get('state') in ('done', 'done_feishu_error')
              and state.get('result_submitted') is True)))


def _scan_processes(owned_pid=None):
    """Read visible GPU owners; never identify an owner by a historical PID."""
    graphics, unreadable = [], 0
    for folder in Path('/proc').glob('[0-9]*'):
        if not folder.is_dir() or int(folder.name) == owned_pid:
            continue
        try:
            argv = [p.decode(errors='replace') for p in (folder / 'cmdline').read_bytes().split(b'\0') if p]
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            unreadable += 1
            continue
        if not argv:
            continue
        text = ' '.join(argv)
        if (any(marker in text for marker in MODEL_MARKERS) or
                ('model_test.py' in text and any(mode in argv for mode in ('infer', 'serve')))):
            raise BusyError('Another model process is running: PID ' + folder.name)
        try:
            fds = list((folder / 'fd').iterdir())
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            unreadable += 1
            continue
        gpu = False
        for fd in fds:
            try:
                gpu = gpu or os.readlink(fd).startswith('/dev/nvidia')
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        if gpu:
            # These are display applications, not a blanket allowlist of Python.
            name = Path(argv[0]).name.lower()
            if name not in {'xorg', 'xwayland', 'awesun', 'webkitwebprocess', 'webkitnetworkprocess',
                            'webkitgpuprocess', 'gnome-shell'}:
                raise BusyError('Unrecognized visible GPU owner: PID ' + folder.name + ' ' + name)
            graphics.append(int(folder.name))
    return {'visible_graphics_pids': graphics, 'unreadable_process_directories': unreadable}


def require_idle(owned_pid=None, *, bind='tcp://127.0.0.1:9117'):
    """Fail closed on connections and visible compute; a low GPU % proves nothing.

    The existing AI has an NVML driver/library mismatch. When NVML is unavailable
    the evidence explicitly remains the readable /proc GPU-owner scan used by
    the reviewed suite; this is never described as a machine-wide NVML check.
    """
    from urllib.parse import urlsplit
    port = urlsplit(bind).port
    if port is None:
        raise ValueError('A TCP port is required')
    if BUSY_PORTS.search(_command(['ss', '-Htn', 'state', 'established'])):
        raise BusyError('Existing Policy/model connection; close the trial before loading/stopping')
    listeners = _command(['ss', '-Hltnp'])
    for line in listeners.splitlines():
        if re.search(r':' + str(port) + r'\b', line) and (owned_pid is None or f'pid={owned_pid},' not in line):
            raise BusyError(f'Port {port} is occupied by an unowned service; leave it running')
    evidence = _scan_processes(owned_pid)
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(memory['MemAvailable'].split()[0]) * 1024
    if owned_pid is None and available < 18 * 1024**3:
        raise BusyError('Less than 18 GiB host memory available')
    evidence.update(host_memory_available=available,
                    gpu_scope='readable /proc device descriptors', nvml_available=False)
    try:
        compute = _command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    except (OSError, subprocess.SubprocessError):
        # This branch is reported to the operator, not silently called full GPU validation.
        pass
    else:
        for line in compute.splitlines():
            if line.strip() and (not line.strip().isdigit() or int(line.strip()) != owned_pid):
                raise BusyError('Another/unknown GPU compute owner: ' + line.strip())
        evidence.update(nvml_available=True, gpu_scope='NVML compute PIDs and readable /proc')
        if owned_pid is None:
            free = _command(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'])
            rows = free.splitlines()
            if not rows or not rows[0].strip().isdigit() or int(rows[0].strip()) < 17 * 1024:
                raise BusyError('GPU0 free memory is below 17 GiB or unknown')
    return evidence


def _catalog_module(root):
    spec = importlib.util.spec_from_file_location('_evaluation_suite_catalog', root / 'model_catalog.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelHost:
    def __init__(self, suite_root=DEFAULT_SUITE, output_root=DEFAULT_OUTPUT, *,
                 guard=require_idle, popen=subprocess.Popen, identity=_process_identity,
                 signal_group=os.killpg if hasattr(os, 'killpg') else None,
                 disconnected=None, ready_timeout=600, idle_seconds=30,
                 catalog_module=None, catalog_path=None):
        self.suite_root = Path(suite_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.catalog_path = Path(catalog_path) if catalog_path else self.suite_root / 'models.json'
        self.bind = 'tcp://127.0.0.1:9117'
        self.catalog = catalog_module or _catalog_module(self.suite_root)
        self.guard = (lambda pid: require_idle(pid, bind=self.bind)) if guard is require_idle else guard
        self.popen = popen
        self.identity, self.signal_group = identity, signal_group
        self.disconnected = disconnected or threading.Event()
        self.ready_timeout, self.idle_seconds = ready_timeout, idle_seconds
        self.child = None
        self.child_identity = None
        self.record = {'state': 'empty'}
        self.log_stream = None

    def _check_connected(self, seconds=0):
        if self.disconnected.wait(seconds):
            raise ConnectionError('SSH input closed; cancelling owned model')

    def _assert_owned(self):
        if self.child is None or self.child.poll() is not None:
            return False
        try:
            actual = self.identity(self.child.pid)
        except (FileNotFoundError, ProcessLookupError):
            if self.child.poll() is not None:
                return False
            raise
        if actual != self.child_identity or actual['group'] != self.child.pid or actual['session'] != self.child.pid:
            raise RuntimeError('Owned child identity changed; refusing to signal')
        return True

    def _signal_owned(self, number):
        if not self._assert_owned():
            return False
        try:
            self.signal_group(self.child.pid, number)
        except ProcessLookupError:
            # The process may exit between identity verification and killpg.
            if self.child.poll() is None:
                raise
        return True

    def status(self):
        result = dict(self.record)
        if self.child is not None:
            code = self.child.poll()
            result.update(pid=self.child.pid, returncode=code)
            if code is not None:
                result['state'] = 'exited'
        return result

    def _stop_owned(self):
        if self._signal_owned(signal.SIGTERM):
            try:
                self.child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._signal_owned(KILL_SIGNAL)
                self.child.wait(timeout=10)
        if self.log_stream:
            self.log_stream.close()
        previous = self.record.get('model_id')
        self.child, self.child_identity, self.log_stream = None, None, None
        self.record = {'state': 'empty', 'previous_model_id': previous}
        return dict(self.record)

    def stop(self):
        if self.child is not None and self.child.poll() is None:
            self.guard(self.child.pid)
        return self._stop_owned()

    def close(self):
        # A lost SSH owner cannot leave its test model running indefinitely.
        # This never broadens ownership, even if the ordinary idle guard fails.
        return self._stop_owned()

    def _wait_ready(self, expected, log_path):
        deadline, pending = time.monotonic() + self.ready_timeout, ''
        with log_path.open(encoding='utf-8', errors='replace') as log:
            while time.monotonic() < deadline:
                self._check_connected()
                code = self.child.poll()
                if code is not None:
                    raise RuntimeError(f'Model exited ({code}); inspect {log_path}')
                pending += log.read(1024 * 1024)
                lines = pending.split('\n')
                pending = lines.pop()
                if len(pending) > 1024 * 1024:
                    pending = ''
                for line in lines:
                    try:
                        event = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(event, dict) or event.get('event') != 'ready':
                        continue
                    actual = {key: event.get(key) for key in expected}
                    if actual != expected or event.get('address') != self.bind:
                        raise RuntimeError(f'Ready event contract mismatch; inspect {log_path}')
                    return
                self._check_connected(0.1)
        raise TimeoutError(f'Model readiness exceeded {self.ready_timeout}s; inspect {log_path}')

    def load(self, model_id, expected_contract=None):
        if not isinstance(model_id, str):
            raise ValueError('model_id must be a catalog key')
        catalog = self.catalog.read_catalog(self.catalog_path)
        profile = self.catalog.select_profile(catalog, model_id)
        expected = self.catalog.contract(profile)
        if expected_contract is not None and expected_contract != expected:
            raise ValueError('Local/AI model contracts differ; no model was changed')
        from urllib.parse import urlsplit
        address = urlsplit(profile['bind'])
        if address.scheme != 'tcp' or not address.hostname or not address.port:
            raise ValueError('bind must be tcp://HOST:PORT')
        self.bind = profile['bind']
        if (self._assert_owned() and self.record.get('state') == 'ready' and
                self.record.get('contract') == expected):
            return dict(self.record, reused=True)
        self._check_connected()
        owned = self.child.pid if self._assert_owned() else None
        first = self.guard(owned)
        self._check_connected(self.idle_seconds)
        second = self.guard(owned)
        self._check_connected()
        self._stop_owned()
        # Check released resources and port before allocation of another model.
        free = self.guard(None)
        self._check_connected()
        run = self.output_root / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:12])
        run.mkdir(parents=True, exist_ok=False)
        output, log_path = run / 'service', run / 'model.log'
        command = [sys.executable, '-B', '-u', str(self.suite_root / 'model_test.py'),
                   'serve', '--catalog', str(self.catalog_path), '--model', model_id,
                   '--bind', self.bind, '--output', str(output)]
        self.record = {'state': 'loading', 'model_id': model_id, 'contract': expected,
                       'bind': self.bind, 'output': str(output), 'log': str(log_path),
                       'command': command, 'guards': [first, second, free], 'reused': False}
        (run / 'launch.json').write_text(json.dumps(self.record, indent=2) + '\n', encoding='utf-8')
        self.log_stream = log_path.open('x', encoding='utf-8')
        try:
            self.child = self.popen(command, cwd=self.suite_root, stdin=subprocess.DEVNULL,
                                    stdout=self.log_stream, stderr=subprocess.STDOUT,
                                    start_new_session=True, close_fds=True)
            self.child_identity = self.identity(self.child.pid)
            self._wait_ready(expected, log_path)
            self.record['state'] = 'ready'
            self.record['pid'] = self.child.pid
            return dict(self.record)
        except BaseException:
            self._stop_owned()
            raise


def serve_stdio(host, input_stream=sys.stdin, output_stream=sys.stdout):
    """A reader observes EOF even while load waits, so disconnect cancels loading."""
    requests = queue.Queue()

    def reader():
        try:
            while True:
                line = input_stream.readline(65538)
                if not line:
                    break
                if len(line) > 65536 or not line.endswith('\n'):
                    requests.put(ValueError('Request exceeds 64 KiB or is unterminated'))
                    break
                requests.put(line)
        finally:
            host.disconnected.set()

    threading.Thread(target=reader, daemon=True).start()
    try:
        while not host.disconnected.is_set():
            try:
                line = requests.get(timeout=0.2)
            except queue.Empty:
                continue
            request_id = None
            try:
                if isinstance(line, Exception):
                    raise line
                request = json.loads(line)
                if not isinstance(request, dict) or type(request.get('id')) is not int:
                    raise ValueError('A request needs an integer id')
                request_id = request['id']
                op = request.get('op')
                if op == 'load':
                    result = host.load(request.get('model_id'), request.get('expected_contract'))
                elif op == 'status':
                    result = host.status()
                elif op == 'stop':
                    result = host.stop()
                else:
                    raise ValueError('Unknown operation')
                response = {'id': request_id, 'ok': True, 'result': result}
            except Exception as exc:
                response = {'id': request_id, 'ok': False, 'error': type(exc).__name__ + ': ' + str(exc)}
            output_stream.write(json.dumps(response, ensure_ascii=False) + '\n')
            output_stream.flush()
    finally:
        host.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-root', default=DEFAULT_SUITE)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT)
    parser.add_argument('--catalog', type=Path, required=True)
    args = parser.parse_args(argv)
    host = ModelHost(args.suite_root, args.output_root, catalog_path=args.catalog)
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, lambda *_: host.disconnected.set())
    serve_stdio(host)


if __name__ == '__main__':
    main()

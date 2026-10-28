"""Lazy SSH JSON-lines controller. No passwords, remote upload, or import effects."""
from __future__ import annotations

import json
from pathlib import PurePosixPath
import queue
import re
import shlex
import subprocess
import threading


class RemoteModels:
    def __init__(self, suite_root, host_script, *,
                 host, catalog, output_root=None, timeout=780,
                 batch_mode=False, popen=subprocess.Popen):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+', host):
            raise ValueError('host must be user@hostname without shell options')
        for value in (suite_root, host_script, catalog, output_root):
            if value is not None and (not PurePosixPath(str(value)).is_absolute() or
                                      '\n' in str(value) or '\0' in str(value)):
                raise ValueError('Remote paths must be absolute single-line paths')
        if timeout <= 0:
            raise ValueError('timeout must be positive')
        remote = ['/usr/bin/python3', '-B', '-u', str(host_script), '--suite-root', str(suite_root), '--catalog', str(catalog)]
        if output_root is not None:
            remote += ['--output-root', str(output_root)]
        self.command = ['ssh', '-T', '-o', 'StrictHostKeyChecking=yes',
                        '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=10',
                        '-o', 'ServerAliveCountMax=3', '-o', 'NumberOfPasswordPrompts=1',
                        '-o', 'BatchMode=' + ('yes' if batch_mode else 'no'),
                        host, shlex.join(remote)]
        self.timeout, self.popen = timeout, popen
        self.process = None
        self.responses = queue.Queue()
        self.next_id = 0
        self.lock = threading.Lock()

    def _connect(self):
        if self.process is not None:
            if self.process.poll() is not None:
                raise ConnectionError('SSH controller exited; create a new evaluation connection')
            return
        # Inherit stderr: SSH uses the operator's terminal for authentication;
        # neither this class nor the remote script ever receives the password.
        self.process = self.popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=None, text=True, encoding='utf-8', bufsize=1)
        process, responses = self.process, self.responses

        def read():
            try:
                while True:
                    line = process.stdout.readline(65538)
                    if not line:
                        break
                    if len(line) > 65536 or not line.endswith('\n'):
                        responses.put(ValueError('Oversized/unterminated SSH response'))
                        return
                    responses.put(line)
            finally:
                responses.put(ConnectionError('SSH closed; its owned model is being cleaned up'))

        threading.Thread(target=read, daemon=True).start()

    def _request(self, op, **arguments):
        with self.lock:
            self._connect()
            self.next_id += 1
            request = {'id': self.next_id, 'op': op, **arguments}
            try:
                self.process.stdin.write(json.dumps(request) + '\n')
                self.process.stdin.flush()
                response = self.responses.get(timeout=self.timeout)
                if isinstance(response, Exception):
                    raise response
                response = json.loads(response)
                if not isinstance(response, dict) or response.get('id') != self.next_id:
                    raise ValueError('SSH response id mismatch')
                if response.get('ok') is not True:
                    raise RuntimeError(str(response.get('error', 'Remote model operation failed')))
                if not isinstance(response.get('result'), dict):
                    raise ValueError('SSH response result must be a mapping')
                return response['result']
            except queue.Empty as exc:
                self._disconnect()
                raise TimeoutError(f'Model operation exceeded {self.timeout}s; SSH input closed for cleanup') from exc
            except (OSError, ConnectionError, ValueError):
                self._disconnect()
                raise

    def load(self, model_id, expected_contract=None):
        result = self._request('load', model_id=model_id, expected_contract=expected_contract)
        if result.get('state') != 'ready' or result.get('model_id') != model_id:
            self.close()
            raise RuntimeError('Remote host did not confirm the selected model ready')
        if expected_contract is not None and result.get('contract') != expected_contract:
            self.close()
            raise RuntimeError('Ready model contract differs from local selection')
        return result

    def status(self):
        if self.process is None:
            return {'state': 'disconnected'}
        return self._request('status')

    def stop(self):
        if self.process is None:
            return {'state': 'empty'}
        return self._request('stop')

    def _disconnect(self):
        process = self.process
        if process is None:
            return
        try:
            process.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            process.wait(timeout=25)
        except subprocess.TimeoutExpired:
            # Terminate only our local SSH child. Remote EOF/HUP handlers own cleanup.
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        process.stdout.close()
        self.process = None
        self.responses = queue.Queue()

    def close(self):
        with self.lock:
            self._disconnect()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

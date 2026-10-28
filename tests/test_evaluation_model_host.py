"""Local fake lifecycle tests: no SSH, GPU, camera, robot, or real model process."""
import io
import json
from pathlib import Path
import signal
import subprocess
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import evaluation_model_host as host_module


def profile(model_id='pi05-drawer'):
    return {'model_id': model_id, 'bind': 'tcp://127.0.0.1:9117', 'backend': 'pi05'}


def contract(value):
    return {'model_id': value['model_id'], 'profile_sha256': 'exact-' + value['model_id']}


def select(_, model_id):
    if model_id not in {'pi05-drawer', 'g05-drawer'}:
        raise KeyError(model_id)
    return profile(model_id)


class HostTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.guard = Mock(return_value={'idle': True})
        self.signals = Mock()
        self.child = Mock(pid=12345)
        self.child.poll.return_value = None
        self.child.wait.return_value = -15
        self.catalog = SimpleNamespace(read_catalog=Mock(return_value={}),
                                       select_profile=select, contract=contract)
        self.identity = Mock(return_value={'start_ticks': '789', 'group': 12345, 'session': 12345})
        self.launches = []

        def launch(command, **kwargs):
            self.launches.append((command, kwargs))
            selected = command[command.index('--model') + 1]
            kwargs['stdout'].write('loader messages\n' + json.dumps(
                {'event': 'ready', 'address': 'tcp://127.0.0.1:9117', **contract(profile(selected))}) + '\n')
            kwargs['stdout'].flush()
            return self.child

        self.popen = Mock(side_effect=launch)
        self.host = host_module.ModelHost(self.root / 'suite', self.root / 'runs',
                                         guard=self.guard, popen=self.popen, identity=self.identity,
                                         signal_group=self.signals, idle_seconds=0,
                                         catalog_module=self.catalog)
        self.addCleanup(self.host.close)

    def test_construct_and_status_have_no_launch_or_guard_side_effects(self):
        self.assertEqual(self.host.status(), {'state': 'empty'})
        self.guard.assert_not_called()
        self.popen.assert_not_called()
        self.assertFalse((self.root / 'runs').exists())

    def test_load_uses_existing_launcher_and_exact_ready_contract(self):
        result = self.host.load('pi05-drawer', contract(profile()))
        self.assertEqual(result['state'], 'ready')
        self.assertEqual(result['contract'], contract(profile()))
        self.assertEqual(self.guard.call_count, 3)
        args, options = self.launches[0]
        self.assertEqual(args[3:5], [str((self.root / 'suite/model_test.py').resolve()), 'serve'])
        self.assertNotIn('--in-runtime', args)
        self.assertTrue(options['start_new_session'])
        self.assertFalse(Path(result['output']).exists())
        self.assertTrue(Path(result['log']).exists())

    def test_same_contract_keeps_resident_child(self):
        first = self.host.load('pi05-drawer')
        second = self.host.load('pi05-drawer')
        self.assertTrue(second['reused'])
        self.assertEqual(first['pid'], second['pid'])
        self.popen.assert_called_once()
        self.signals.assert_not_called()

    def test_unknown_model_or_contract_mismatch_never_touches_service(self):
        with self.assertRaises(KeyError):
            self.host.load('shell ; anything')
        with self.assertRaisesRegex(ValueError, 'contracts differ'):
            self.host.load('pi05-drawer', {'model_id': 'wrong'})
        self.popen.assert_not_called()
        self.guard.assert_not_called()

    def test_busy_guard_or_second_check_leaves_previous_model_running(self):
        self.host.load('pi05-drawer')
        self.guard.reset_mock(side_effect=True)
        self.guard.side_effect = [{'idle': True}, host_module.BusyError('normal client active')]
        with self.assertRaisesRegex(host_module.BusyError, 'normal client'):
            self.host.load('g05-drawer')
        self.assertEqual(self.host.status()['model_id'], 'pi05-drawer')
        self.signals.assert_not_called()
        self.assertEqual(len(self.launches), 1)

    def test_switch_stops_only_owned_group_and_uses_a_fresh_output(self):
        first = self.host.load('pi05-drawer')
        second = self.host.load('g05-drawer')
        self.assertNotEqual(first['output'], second['output'])
        self.signals.assert_called_once_with(12345, signal.SIGTERM)
        self.assertEqual(len(self.launches), 2)

    def test_changed_process_identity_blocks_signal(self):
        self.host.load('pi05-drawer')
        self.identity.return_value = {'start_ticks': 'changed', 'group': 12345, 'session': 12345}
        with self.assertRaisesRegex(RuntimeError, 'identity changed'):
            self.host.close()
        self.signals.assert_not_called()
        self.identity.return_value = self.host.child_identity

    def test_busy_explicit_stop_does_not_signal_but_disconnect_cleans_own_child(self):
        self.host.load('pi05-drawer')
        self.guard.side_effect = host_module.BusyError('connection is established')
        with self.assertRaises(host_module.BusyError):
            self.host.stop()
        self.signals.assert_not_called()
        self.host.close()
        self.signals.assert_called_once_with(12345, signal.SIGTERM)

    def test_stop_escalates_only_after_own_child_does_not_exit(self):
        self.host.load('pi05-drawer')
        self.child.wait.side_effect = [subprocess.TimeoutExpired('owned model', 10), -9]
        self.host.close()
        self.assertEqual([call.args for call in self.signals.call_args_list],
                         [(12345, signal.SIGTERM), (12345, host_module.KILL_SIGNAL)])

    def test_ready_timeout_cleans_up_own_model_and_preserves_log(self):
        self.host.ready_timeout = 0
        with self.assertRaisesRegex(TimeoutError, 'inspect'):
            self.host.load('pi05-drawer')
        self.signals.assert_called_once_with(12345, signal.SIGTERM)
        self.assertEqual(self.host.status()['state'], 'empty')
        self.assertEqual(len(list((self.root / 'runs').glob('*/model.log'))), 1)

    def test_mismatched_ready_event_never_announces_ready(self):
        def wrong(command, **kwargs):
            kwargs['stdout'].write(json.dumps({'event': 'ready', 'address': 'tcp://127.0.0.1:9117',
                                               'model_id': 'wrong'}) + '\n')
            kwargs['stdout'].flush()
            return self.child
        self.popen.side_effect = wrong
        with self.assertRaisesRegex(RuntimeError, 'contract mismatch'):
            self.host.load('pi05-drawer')
        self.signals.assert_called_once_with(12345, signal.SIGTERM)

    def test_disconnected_before_load_never_starts_model(self):
        self.host.disconnected.set()
        with self.assertRaises(ConnectionError):
            self.host.load('pi05-drawer')
        self.popen.assert_not_called()

    def test_disconnect_during_loading_cleans_owned_child(self):
        def launch_then_disconnect(command, **kwargs):
            self.host.disconnected.set()
            return self.child
        self.popen.side_effect = launch_then_disconnect
        with self.assertRaises(ConnectionError):
            self.host.load('pi05-drawer')
        self.signals.assert_called_once_with(12345, signal.SIGTERM)
        self.assertEqual(self.host.status()['state'], 'empty')

    def test_exited_child_is_reported_and_never_signalled(self):
        self.host.load('pi05-drawer')
        self.child.poll.return_value = 5
        self.assertEqual(self.host.status()['state'], 'exited')
        self.host.close()
        self.signals.assert_not_called()

    def test_child_exiting_between_identity_and_signal_is_cleaned_up(self):
        self.host.load('pi05-drawer')
        self.child.poll.side_effect = [None, 0]
        self.signals.side_effect = ProcessLookupError('already exited')
        self.host.close()
        self.assertIsNone(self.host.child)
        self.assertIsNone(self.host.log_stream)


class GuardTests(unittest.TestCase):
    def call_guard(self, state=None, connection='', listener='', compute='', memory=None):
        output = [connection, listener, compute, '24576\n']
        with patch.object(host_module, '_command', side_effect=output), \
                patch.object(host_module, '_scan_processes', return_value={}), \
                patch.object(host_module.Path, 'read_text', return_value=memory or 'MemAvailable: 40000000 kB\n'):
            return host_module.require_idle()




    def test_established_policy_or_model_connection_blocks_even_with_free_gpu(self):
        for port in (8000, 9105, 9117, 9129):
            with self.subTest(port=port), self.assertRaisesRegex(host_module.BusyError, 'connection'):
                self.call_guard(connection=f'0 0 172.16.0.4:{port} 172.16.0.3:45123')

    def test_occupied_9117_cannot_be_taken_over(self):
        with self.assertRaisesRegex(host_module.BusyError, 'unowned'):
            self.call_guard(listener='LISTEN 0 1 0.0.0.0:9117 *:* users:(("python",pid=88,fd=1))')

    def test_any_nvml_compute_pid_blocks(self):
        with self.assertRaisesRegex(host_module.BusyError, 'GPU compute'):
            self.call_guard(compute='450\n')

    def test_low_memory_blocks(self):
        with self.assertRaisesRegex(host_module.BusyError, '18 GiB'):
            self.call_guard(memory='MemAvailable: 1000 kB\n')

    def test_clear_guard_reports_actual_scope(self):
        result = self.call_guard()
        self.assertTrue(result['nvml_available'])
        self.assertIn('NVML', result['gpu_scope'])


class ProtocolTests(unittest.TestCase):
    def test_unknown_operation_is_rejected_then_eof_cleans_owned_service(self):
        request = json.dumps({'id': 3, 'op': 'kill_everything'}) + '\n'
        replied = threading.Event()

        class Input:
            first = True

            def readline(self, _limit):
                if self.first:
                    self.first = False
                    return request
                replied.wait(1)
                return ''

        class Output(io.StringIO):
            def flush(self):
                replied.set()

        host = Mock(disconnected=threading.Event())
        output = Output()
        host_module.serve_stdio(host, Input(), output)
        response = json.loads(output.getvalue())
        self.assertEqual(response['id'], 3)
        self.assertFalse(response['ok'])
        self.assertIn('Unknown operation', response['error'])
        host.load.assert_not_called()
        host.stop.assert_not_called()
        host.close.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()

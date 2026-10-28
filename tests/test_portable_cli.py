"""Portable selection, session, configuration and lifecycle regression tests."""
import copy
import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import evaluate
from evaluation_session import Session, SessionError, SessionLocked
from evaluation_limits import action_seconds_for_profile
from model_catalog import read_catalog, select_profile, contract, validate_runtime_profile
from evaluation_remote import RemoteModels

ROOT = Path(__file__).resolve().parents[1]


def profile():
    return select_profile(read_catalog(ROOT / 'config/models.example.json'), 'pi05')


def test_arbitrary_task_and_explicit_reversed_order(tmp_path):
    with Session.create(tmp_path / 's', 'a new task', ['model-b', 'model-a']) as session:
        assert session.state['model_ids'] == ['model-b', 'model-a']
        assert session.state['task'] == 'a new task'
    with Session.open(tmp_path / 's') as session:
        assert session.state['model_ids'] == ['model-b', 'model-a']


@pytest.mark.parametrize('ids', [None, [], ['x', 'x'], ['', 'x']])
def test_model_selection_must_be_explicit_and_unique(tmp_path, ids):
    with pytest.raises(SessionError):
        Session.create(tmp_path / 's', 'custom', ids)


def test_export_failure_never_counts_and_successful_retry_counts_once(tmp_path):
    with Session.create(tmp_path / 's', 'custom', ['one'], repeats=1, steps=1) as session:
        attempt = session.begin_attempt()
        session.finish_attempt(attempt['id'], {'end_reason': 'step_limit', 'steps_sent': 1,
                                               'recording_complete': True})
        session.score_attempt(attempt['id'], 0, outcome='failed')
        assert session.state['progress']['one']['completed'] == 0
        session.mark_export(attempt['id'], 'failed', error='disk full')
        assert session.state['progress']['one']['completed'] == 0
        session.mark_export(attempt['id'], 'succeeded', path='dataset')
        session.mark_export(attempt['id'], 'succeeded', path='dataset')
        assert session.state['progress']['one']['completed'] == 1
        assert session.state['attempts'][0]['success'] is False


def test_crash_recovery_marks_attempt_interrupted_and_preserves_order(tmp_path):
    with Session.create(tmp_path / 's', 'custom', ['second', 'first']) as session:
        session.begin_attempt()
    with Session.open(tmp_path / 's') as session:
        assert session.state['attempts'][0]['end_reason'] == 'interrupted'
        assert session.state['progress']['second']['completed'] == 0
        assert session.state['model_ids'] == ['second', 'first']
        with pytest.raises(SessionLocked):
            Session.open(tmp_path / 's')


def test_resume_rejects_changed_profile_without_mutating_session(tmp_path):
    catalog = read_catalog(ROOT / 'config/models.example.json')
    profiles = [select_profile(catalog, 'pi05')]
    with Session.create(tmp_path / 's', 'any task', ['pi05']) as session:
        saved = session.state
    (tmp_path / 's/profiles.json').write_text(json.dumps({'profiles': profiles}), encoding='utf-8')
    before = (tmp_path / 's/session.json').read_bytes()
    catalog['defaults']['instruction'] = 'a different instruction'
    with pytest.raises(RuntimeError, match='配置'):
        evaluate.resume_profiles(catalog, tmp_path / 's', saved)
    assert before == (tmp_path / 's/session.json').read_bytes()


def test_task_name_never_changes_duration():
    assert action_seconds_for_profile({'model_id': 'anything', 'max_action_seconds': 45}) == 45
    assert action_seconds_for_profile({'model_id': 'a-different-name', 'max_action_seconds': 45}) == 45
    for value in [0, -1, True, float('nan'), float('inf')]:
        with pytest.raises(ValueError):
            action_seconds_for_profile({'max_action_seconds': value})


def test_plan_and_list_do_not_connect_or_import_device_runtime(monkeypatch, capsys):
    import socket
    monkeypatch.setattr(socket, 'create_connection', Mock(side_effect=AssertionError('network forbidden')))
    evaluate.main(['--catalog', str(ROOT / 'config/models.example.json'), '--task', 'custom_task',
                   '--models', 'groot', 'pi05', '--plan'])
    result = json.loads(capsys.readouterr().out)
    assert result['started'] is False
    assert list(result['target_counts']) == ['groot', 'pi05']


def test_unfilled_template_refused_before_live_operations():
    with pytest.raises(ValueError, match='placeholder'):
        validate_runtime_profile(profile())


def test_profile_hash_changes_with_camera_weights_and_instruction():
    original = profile()
    for key in ('checkpoint', 'instruction', 'camera_config'):
        modified = copy.deepcopy(original)
        modified[key] = 'changed'
        assert contract(original) != contract(modified)


def test_remote_uses_explicit_paths_and_does_not_connect_on_construction():
    popen = Mock(side_effect=AssertionError('must stay disconnected'))
    remote = RemoteModels('/opt/eval/runtime', '/opt/eval/runtime/evaluation_model_host.py',
                          host='operator@model-host', catalog='/opt/eval/config/models.local.json', popen=popen)
    assert '--catalog' in remote.command[-1]
    assert '/opt/eval/config/models.local.json' in remote.command[-1]
    assert remote.status() == {'state': 'disconnected'}
    remote.close()
    popen.assert_not_called()


def test_interactive_load_never_starts_robot(tmp_path):
    p = profile()
    remote = Mock()
    remote.load.return_value = {'state': 'ready', 'contract': contract(p)}
    trial = Mock(side_effect=AssertionError('load must not move robot'))
    with Session.create(tmp_path / 's', 'custom', ['pi05']) as session:
        controller = evaluate.Evaluation(session, [p], remote, trial=trial, guard=Mock(), emit=lambda _: None)
        try:
            evaluate.interactive(controller, input_stream=io.StringIO('load\nquit\n'))
            trial.assert_not_called()
            remote.load.assert_called_once()
        finally:
            controller.close()

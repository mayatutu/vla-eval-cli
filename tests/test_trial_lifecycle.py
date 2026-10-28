"""Run the real recorder and action runner against in-memory robot/model doubles."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
import evaluation_trial as trial
import model_client
from model_catalog import contract


@pytest.mark.parametrize('ending', ['step_limit', 'early_finish', 'paused', 'error'])
def test_trial_records_and_closes_every_resource(tmp_path, monkeypatch, ending):
    p = dict(model_id='arbitrary', backend='pi05', revision='fixture', action_space='joint_position',
             action_dim=8, action_steps=2, control_hz=30, max_action_seconds=30,
             instruction='fixture instruction', cameras={'external_image':'1','wrist_image':'2'},
             recording_cameras={'external_image':'1','wrist_image':'2','external_right_image':'3'})
    class Robot:
        action_dim = 8
        steps = 0
        closed = False
        def get_observation(self):
            return SimpleNamespace(images={k:np.zeros((12,16,3),np.uint8) for k in p['recording_cameras']},
                joints=np.zeros(7,np.float32), gripper=np.zeros(1,np.float32), cartesian_position=np.zeros(6,np.float32))
        def execute(self, action):
            self.steps += 1
            if ending == 'error':
                raise RuntimeError('simulated device failure')
        def close(self):
            self.closed = True
    class Transport:
        closed = False
        async def request(self, request):
            return {'contract':contract(p), 'actions':np.zeros((2,8),np.float32)}
        def close(self):
            self.closed = True
    robot, transport = Robot(), Transport()
    monkeypatch.setattr(model_client, '_make_robot', lambda *_: robot)
    monkeypatch.setattr(model_client, '_ZmqTransport', lambda *_: transport)
    monkeypatch.setattr(trial, '_continuous_sampling', lambda *_: None)
    monkeypatch.setattr(trial, '_disk_preflight', lambda *_: {})
    stop = lambda: ending if robot.steps and ending in ('early_finish','paused') else None
    result = trial.run_trial(p, url='tcp://127.0.0.1:9999', steps=2, instruction=p['instruction'],
                             output=tmp_path, stop_requested=stop)
    assert result['end_reason'] == ending
    assert robot.closed and transport.closed
    assert json.loads((tmp_path/'trial-result.json').read_text())['end_reason'] == ending
    if ending == 'step_limit':
        assert result['recording_complete'] and result['steps_sent'] == 2
    elif ending == 'error':
        assert not result['recording_complete']
        assert 'simulated device failure' in result['error']

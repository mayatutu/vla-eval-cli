"""Input schemas for the deployed checkpoints, independent of model libraries."""
from collections.abc import Mapping
import numpy as np
from scipy.spatial.transform import Rotation

def _vector(value, shape, name):
    arr=np.asarray(value)
    if arr.shape!=shape or not np.issubdtype(arr.dtype,np.number) or not np.isfinite(arr).all():
        raise ValueError(f'{name}: expected finite numeric {shape}')
    return arr

def _image(value, layout, name):
    arr=np.asarray(value)
    if arr.dtype!=np.uint8 or any(n<=0 for n in arr.shape):
        raise ValueError(f'{name}: expected nonempty uint8 RGB image')
    valid=(arr.ndim==3 and arr.shape[-1]==3) if layout=='HWC' else (
        arr.ndim==3 and arr.shape[0]==3) if layout=='CHW' else (
        arr.ndim==5 and arr.shape[:2]==(1,1) and arr.shape[-1]==3)
    if not valid:raise ValueError(f'{name}: invalid {layout} image shape')

def _rot6d(value):
    rows=np.asarray(value).reshape(-1,6)
    if np.any(np.linalg.norm(np.cross(rows[:,:3],rows[:,3:]),axis=-1)<1e-6):
        raise ValueError('rotation has collinear or zero axes')

def validate_observation(model, payload):
    """Validate full observations; cache-only G05 messages use a separate path."""
    if not isinstance(payload,Mapping):raise ValueError('observation must be a mapping')
    try:
        if model=='groot':
            state=payload['state']
            for name,dim in [('eef_9d',9),('joint_position',7),('gripper_position',1)]:
                _vector(state[name],(1,1,dim),name)
            _rot6d(state['eef_9d'][...,3:])
            for name in ['exterior_image_1_left','wrist_image_left']:_image(payload['video'][name],'BTHWC',name)
            prompt=payload['language']['annotation.language.language_instruction'][0][0]
        elif model in ('pi05', 'pi05_lerobot'):
            _vector(payload['observation/joint_position'],(7,),'joints')
            _vector(payload['observation/gripper_position'],(1,),'gripper')
            for name in ['observation/exterior_image_1_left','observation/wrist_image_left']:_image(payload[name],'HWC',name)
            prompt=payload['prompt']
        elif model=='g05':
            _vector(payload['state']['right_arm'],(7,),'right_arm')
            _vector(payload['state']['right_gripper'],(1,),'right_gripper')
            for name in ['exterior_image','wrist_image','dummy_wrist_right']:_image(payload['images'][name],'CHW',name)
            if payload['frequency']!=15 or payload['embodiment_type']!='Droid_Franka':
                raise ValueError('G05 deployment frequency/embodiment mismatch')
            prompt=payload['task']
        elif model=='lap':
            obs=payload['observation'];state=_vector(obs['state'],(10,),'LAP state')
            _rot6d(state[3:9])
            if 'euler' in obs:
                euler=_vector(obs['euler'],(3,),'Euler xyz')
                matrix=Rotation.from_euler('xyz',euler).as_matrix()
                columns=np.r_[matrix[:,0],matrix[:,1]]
                if not np.allclose(state[3:9],columns,atol=1e-5):
                    raise ValueError('LAP Euler and rot6d column representations disagree')
            for name in ['base_0_rgb','left_wrist_0_rgb']:_image(obs[name],'HWC',name)
            prompt=payload['prompt']
        else:raise ValueError(f'unknown checkpoint schema: {model}')
        if not isinstance(prompt,str) or not prompt.strip():raise ValueError('nonempty task prompt required')
    except (KeyError,IndexError,TypeError) as exc:
        raise ValueError(f'{model}: missing or malformed observation field') from exc

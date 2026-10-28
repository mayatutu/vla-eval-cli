"""Source-backed offline action conversions for individual VLA models.

These functions deliberately return data only.  They do not import DROID or
hold a robot execution handle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ConvertedAction:
    model: str
    raw_joint_position: np.ndarray
    joint_position_target: np.ndarray
    raw_gripper_position: np.ndarray


def g05_action_to_droid_joint_target(
    policy_action: Mapping[str, object], current_gripper_position: np.ndarray
) -> np.ndarray:
    """Apply the G05 DROID client's documented joint/gripper conversion."""
    arm_key = "right_arm" if "right_arm" in policy_action else "joint_position"
    if arm_key not in policy_action:
        raise ValueError("G05 output missing main arm action")
    arm = np.asarray(policy_action[arm_key], dtype=np.float32).reshape(-1)
    if arm.shape != (7,) or not np.isfinite(arm).all():
        raise ValueError("G05 main arm action must contain seven finite values")
    grip_key = next((key for key in ("right_gripper", "gripper") if key in policy_action), None)
    if grip_key is None:
        gripper = np.asarray(current_gripper_position, dtype=np.float32).reshape(-1)
    else:
        raw = np.asarray(policy_action[grip_key], dtype=np.float32).reshape(-1)
        gripper = 1.0 - raw
    if gripper.shape != (1,) or not np.isfinite(gripper).all():
        raise ValueError("G05 gripper action must contain one finite value")
    # G05 occasionally emits a slight overshoot. DROID execution already
    # thresholds gripper to 0/1; aborting an Open Track trial is worse than
    # clipping the inverted command into the legal range.
    gripper = np.clip(gripper, 0.0, 1.0)
    return np.concatenate((arm, gripper)).astype(np.float32)


def lap_relative_cartesian_to_target(
    policy_actions: np.ndarray,
    current_xyz: np.ndarray,
    current_euler_xyz: np.ndarray,
    *,
    use_quaternion: bool = False,
) -> np.ndarray:
    """Apply LAP's official absolute-pose postprocessing.

    LAP predicts relative xyz + Euler XYZ + gripper actions. Its Franka runner
    converts the composed Euler orientation to a quaternion before passing the
    command to ``RobotEnv(action_space="cartesian_position")``.
    """
    actions = np.asarray(policy_actions, dtype=np.float32)
    xyz = np.asarray(current_xyz, dtype=np.float32).reshape(-1)
    euler = np.asarray(current_euler_xyz, dtype=np.float32).reshape(-1)
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("LAP action chunk must be finite with shape (N, 7)")
    if xyz.shape != (3,) or euler.shape != (3,) or not np.isfinite(xyz).all() or not np.isfinite(euler).all():
        raise ValueError("LAP current Cartesian state must contain finite xyz and Euler XYZ")
    converted = actions.copy()
    converted[:, :3] += xyz
    converted[:, 3:6] = (
        Rotation.from_euler("xyz", euler)
        * Rotation.from_euler("xyz", actions[:, 3:6])
    ).as_euler("xyz")
    converted[:, 6] = 1.0 - converted[:, 6]
    if use_quaternion:
        quaternion = Rotation.from_euler("xyz", converted[:, 3:6]).as_quat().astype(np.float32)
        return np.concatenate((converted[:, :3], quaternion, converted[:, 6:7]), axis=1)
    return converted


def groot_server_joint_to_target(
    policy_action: Mapping[str, object]
) -> ConvertedAction:
    """Validate the decoded GR00T server's absolute joint targets.

    Gr00tPolicy calls processor.decode_action with the observation states.
    That decoder already undoes normalization and relative action encoding.
    This boundary accepts server output, never raw model action_pred tensors.
    """
    try:
        joints = np.asarray(policy_action["joint_position"], dtype=np.float32)
        gripper = np.asarray(policy_action["gripper_position"], dtype=np.float32)
    except KeyError as exc:
        raise ValueError(f"GR00T output missing {exc.args[0]}") from exc
    if joints.ndim == 3 and joints.shape[0] == 1:
        joints = joints[0]
    if gripper.ndim == 3 and gripper.shape[0] == 1:
        gripper = gripper[0]
    if joints.ndim != 2 or joints.shape[0] == 0 or joints.shape[1] != 7 or gripper.shape != (joints.shape[0], 1):
        raise ValueError("invalid GR00T DROID action tensor shape")
    if not np.isfinite(joints).all() or not np.isfinite(gripper).all():
        raise ValueError("GR00T action must be finite")
    return ConvertedAction("groot", joints.copy(), joints.copy(), gripper.copy())

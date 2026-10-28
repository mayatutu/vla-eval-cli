from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np


class ActionSpace(str, Enum):
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    CARTESIAN_POSITION = "cartesian_position"
    CARTESIAN_VELOCITY = "cartesian_velocity"


@dataclass(frozen=True)
class ControlSpec:
    action_space: ActionSpace
    action_dim: int
    semantic: str
    max_chunk: int = 1

    @staticmethod
    def droid_dimension(action_space: ActionSpace) -> int:
        return 8 if "joint" in action_space.value else 7

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "ControlSpec":
        try:
            action_space = ActionSpace(metadata["action_space"])
            action_dim = int(metadata["action_dim"])
            semantic = str(metadata["semantic"])
            max_chunk = int(metadata.get("max_chunk", 1))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid adapter control metadata") from exc
        if not semantic:
            raise ValueError("adapter metadata must declare action semantics")
        if max_chunk <= 0:
            raise ValueError("adapter metadata max_chunk must be positive")
        return cls(action_space, action_dim, semantic, max_chunk)


def validate_action_chunk(actions: np.ndarray, spec: ControlSpec) -> np.ndarray:
    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    if chunk.ndim != 2:
        raise ValueError(f"actions must have shape (N, D), got {chunk.shape}")
    if chunk.shape[0] == 0:
        raise ValueError("actions must not be empty")
    if chunk.shape[0] > spec.max_chunk:
        raise ValueError(f"action chunk length {chunk.shape[0]} exceeds {spec.max_chunk}")
    expected = ControlSpec.droid_dimension(spec.action_space)
    allowed_dimensions = {expected}
    if spec.action_space is ActionSpace.CARTESIAN_POSITION:
        # DROID accepts xyz + Euler XYZ + gripper (7) and, for Franka's LAP
        # adapter, xyz + quaternion XYZW + gripper (8).
        allowed_dimensions = {7, 8}
    if spec.action_dim not in allowed_dimensions or chunk.shape[1] != spec.action_dim:
        raise ValueError(
            f"{spec.action_space.value} requires one of {sorted(allowed_dimensions)} values, "
            f"got spec={spec.action_dim}, chunk={chunk.shape[1]}"
        )
    if not np.all(np.isfinite(chunk)):
        raise ValueError("actions must contain only finite values")
    if spec.action_space is ActionSpace.JOINT_VELOCITY and np.any(np.abs(chunk[:, :-1]) > 1.0):
        raise ValueError("joint velocity actions must be in [-1, 1]")
    if spec.action_space is ActionSpace.CARTESIAN_VELOCITY and np.any(np.abs(chunk[:, :-1]) > 1.0):
        raise ValueError("cartesian velocity actions must be in [-1, 1]")
    return chunk


def validate_adapter_response(response: Mapping[str, Any], metadata: Mapping[str, Any]) -> np.ndarray:
    if "actions" not in response:
        raise ValueError("adapter response is missing actions")
    return validate_action_chunk(np.asarray(response["actions"]), ControlSpec.from_metadata(metadata))

"""Local MolmoAct2 adapter that preserves the existing DROID execution boundary.

The shadow helpers in this module never import DROID.  The live helper imports
the already-deployed Colosseum DROID adapter only after an explicit caller
chooses to run it.  Consequently, the only code path that can reach hardware
is the existing ``DroidRobot.execute() -> RobotEnv.step()`` path.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from .contracts import validate_action_chunk
from .molmoact2 import MolmoAct2Client, MolmoAct2Request, control_spec


class ObservationLike(Protocol):
    images: Mapping[str, np.ndarray]
    joints: np.ndarray
    gripper: np.ndarray


class DroidLike(Protocol):
    def get_observation(self) -> ObservationLike: ...

    def execute(self, action: np.ndarray) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class CameraRoles:
    """Names from the existing DROID camera map assigned to Molmo roles."""

    external_image: str
    wrist_image: str

    def __post_init__(self) -> None:
        if not self.external_image or not self.wrist_image:
            raise ValueError("external_image and wrist_image must be non-empty camera names")
        if self.external_image == self.wrist_image:
            raise ValueError("external_image and wrist_image must be different cameras")


@dataclass(frozen=True)
class ShadowResult:
    actions: np.ndarray
    model_dt_ms: float

    def report(self) -> dict[str, Any]:
        return {
            "mode": "shadow",
            "action_shape": list(self.actions.shape),
            "action_finite": bool(np.all(np.isfinite(self.actions))),
            "joint_min": float(np.min(self.actions[:, :-1])),
            "joint_max": float(np.max(self.actions[:, :-1])),
            "gripper_min": float(np.min(self.actions[:, -1])),
            "gripper_max": float(np.max(self.actions[:, -1])),
            "model_dt_ms": self.model_dt_ms,
        }


def molmo_request(
    observation: ObservationLike,
    instruction: str,
    camera_roles: CameraRoles,
    *,
    num_steps: int = 10,
) -> MolmoAct2Request:
    """Map DROID's already-RGB observation into Molmo's published wire schema."""
    try:
        external_cam = observation.images[camera_roles.external_image]
        wrist_cam = observation.images[camera_roles.wrist_image]
    except KeyError as exc:
        raise ValueError(f"required camera {exc.args[0]!r} is absent from DROID observation") from exc
    joints = np.asarray(observation.joints, dtype=np.float32).reshape(-1)
    gripper = np.asarray(observation.gripper, dtype=np.float32).reshape(-1)
    state = np.concatenate((joints, gripper))
    return MolmoAct2Request(external_cam, wrist_cam, instruction, state, num_steps=num_steps)


def infer_shadow(
    client: MolmoAct2Client,
    observation: ObservationLike,
    instruction: str,
    camera_roles: CameraRoles,
    *,
    num_steps: int = 10,
) -> ShadowResult:
    """Run local model inference without DROID import, camera access, or execution."""
    actions, model_dt_ms = client.act(molmo_request(observation, instruction, camera_roles, num_steps=num_steps))
    # Retain the model-specific maximum horizon before the legacy client checks it.
    return ShadowResult(validate_action_chunk(actions, control_spec()), model_dt_ms)


def load_shadow_observation(path: str | Path) -> tuple[ObservationLike, str | None]:
    """Load a recorded or synthetic observation; object deserialization is forbidden."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"external_cam", "wrist_cam", "state"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"shadow observation is missing: {', '.join(sorted(missing))}")
        external_cam = np.asarray(archive["external_cam"])
        wrist_cam = np.asarray(archive["wrist_cam"])
        state = np.asarray(archive["state"], dtype=np.float32).reshape(-1)
        if state.shape != (8,):
            raise ValueError("shadow observation state must have shape (8,)")
        instruction = None
        if "instruction" in archive.files:
            value = np.asarray(archive["instruction"])
            if value.ndim != 0:
                raise ValueError("shadow observation instruction must be a scalar")
            instruction = str(value.item())

    @dataclass(frozen=True)
    class _RecordedObservation:
        images: Mapping[str, np.ndarray]
        joints: np.ndarray
        gripper: np.ndarray

    return _RecordedObservation(
        {"external_image": external_cam, "wrist_image": wrist_cam}, state[:7], state[7:]
    ), instruction


def write_shadow_report(path: str | Path, result: ShadowResult) -> None:
    Path(path).write_text(json.dumps(result.report(), indent=2) + "\n", encoding="utf-8")


LegacyActionValidator = Callable[[np.ndarray, int], np.ndarray]
ActionAuditCallback = Callable[[Mapping[str, Any]], None]


def legacy_colosseum_validator(
    actions: np.ndarray, control_step: int, expected_dim: int = 8
) -> np.ndarray:
    """Apply the unmodified Colosseum action-plan validation to local actions.

    Imports stay inside this function so importing shadow tooling cannot load
    DROID or create RobotEnv.
    """
    from colosseum_client import colosseum_pb2 as pb
    from colosseum_client.robot_runner import action_chunk
    from colosseum_client.tensors import tensor_from_numpy

    plan = pb.ActionPlan(
        start_step=control_step,
        valid_until_step=control_step + len(actions) - 1,
        actions=tensor_from_numpy(np.asarray(actions, dtype=np.float32)),
    )
    return action_chunk(plan, control_step=control_step, expected_dim=expected_dim)


async def run_live(
    client: MolmoAct2Client,
    robot: DroidLike,
    instruction: str,
    camera_roles: CameraRoles,
    *,
    control_hz: int = 15,
    max_control_steps: int = 1,
    num_steps: int = 10,
    legacy_validator: LegacyActionValidator = legacy_colosseum_validator,
    audit_callback: ActionAuditCallback | None = None,
) -> None:
    """Execute a bounded local-policy test through the existing DROID adapter.

    The caller owns construction of ``robot``.  In production that construction
    can reset the Franka, so this function must be reached only after the
    operator's immediate physical confirmation.
    """
    if control_hz <= 0:
        raise ValueError("control_hz must be positive")
    if max_control_steps <= 0:
        raise ValueError("max_control_steps must be positive")
    period = 1.0 / control_hz
    control_step = 0
    while control_step < max_control_steps:
        current = robot.get_observation()
        result = infer_shadow(client, current, instruction, camera_roles, num_steps=num_steps)
        # This preserves the legacy checks that normally consume Router output.
        actions = legacy_validator(result.actions, control_step)
        state = np.concatenate(
            (
                np.asarray(current.joints, dtype=np.float32).reshape(-1),
                np.asarray(current.gripper, dtype=np.float32).reshape(-1),
            )
        )
        for action in actions:
            if control_step >= max_control_steps:
                return
            started = time.monotonic()
            command = np.asarray(action, dtype=np.float32).reshape(-1)
            executed_command = command.copy()
            executed_command[-1] = 1.0 if executed_command[-1] > 0.5 else 0.0
            if audit_callback is not None:
                audit_callback(
                    {
                        "event": "action_planned",
                        "control_step": control_step,
                        "model_dt_ms": result.model_dt_ms,
                        "observation_state": state.tolist(),
                        "model_action": command.tolist(),
                        "joint_delta": (command[:-1] - state[:-1]).tolist(),
                        "droid_action": executed_command.tolist(),
                    }
                )
            robot.execute(action)
            if audit_callback is not None:
                audit_callback({"event": "execute_returned", "control_step": control_step})
            control_step += 1
            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                await asyncio.sleep(remaining)


def make_existing_droid_robot(
    cameras: Mapping[str, str],
    image_size: tuple[int, int] = (512, 288),
    *,
    action_space: str = "joint_position",
) -> DroidLike:
    """Create the original adapter only for the explicitly selected live mode."""
    from colosseum_client.droid_robot import DroidRobot

    return DroidRobot(cameras, action_space=action_space, image_size=image_size)

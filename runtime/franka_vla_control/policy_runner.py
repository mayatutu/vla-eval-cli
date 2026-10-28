"""Local VLA adapters that retain the deployed DROID execution boundary."""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from .adapters import (
    g05_action_to_droid_joint_target,
    groot_server_joint_to_target,
    lap_relative_cartesian_to_target,
)
from .contracts import ActionSpace, ControlSpec, validate_action_chunk
from .droid_molmo import CameraRoles, legacy_colosseum_validator
from .local_transport import ensure_local_transport

ensure_local_transport()

from .numpy_msgpack import packb as groot_packb, unpackb as groot_unpackb
from .openpi_msgpack import packb as websocket_packb, unpackb as websocket_unpackb
from .observations import validate_observation
from .pi05_lerobot_contract import validate_server_metadata as validate_pi05_lerobot_metadata


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    control: ControlSpec


MODEL_SPECS = {
    "pi05": ModelSpec("pi05", "ws://127.0.0.1:9102", ControlSpec(ActionSpace.JOINT_VELOCITY, 8, "DROID normalized joint velocity with gripper position", 15)),
    "pi05_lerobot": ModelSpec("pi05_lerobot", "ws://127.0.0.1:9112", ControlSpec(ActionSpace.JOINT_VELOCITY, 8, "DROID normalized joint velocity with gripper position from the audited LeRobot conversion", 15)),
    "groot": ModelSpec("groot", "tcp://127.0.0.1:9103", ControlSpec(ActionSpace.JOINT_POSITION, 8, "decoded absolute DROID joint target with gripper position", 40)),
    "g05": ModelSpec("g05", "ws://127.0.0.1:9104", ControlSpec(ActionSpace.JOINT_POSITION, 8, "absolute DROID joint target with inverted gripper", 1)),
    "lap": ModelSpec("lap", "ws://127.0.0.1:9105", ControlSpec(ActionSpace.CARTESIAN_POSITION, 7, "absolute Cartesian Euler target with gripper position", 16)),
}


class ObservationLike(Protocol):
    images: Mapping[str, np.ndarray]
    joints: np.ndarray
    gripper: np.ndarray
    cartesian_position: np.ndarray


class DroidLike(ObservationLike, Protocol):
    action_dim: int

    def get_observation(self) -> ObservationLike: ...

    def execute(self, action: np.ndarray) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class PolicyResult:
    model: str
    actions: np.ndarray
    raw_response: Mapping[str, Any]
    model_dt_ms: float

    def report(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "action_shape": list(self.actions.shape),
            "action_finite": bool(np.isfinite(self.actions).all()),
            "action_min": float(np.min(self.actions)),
            "action_max": float(np.max(self.actions)),
            "model_dt_ms": self.model_dt_ms,
        }


def _images(observation: ObservationLike, roles: CameraRoles) -> tuple[np.ndarray, np.ndarray]:
    try:
        external = np.asarray(observation.images[roles.external_image], dtype=np.uint8)
        wrist = np.asarray(observation.images[roles.wrist_image], dtype=np.uint8)
    except KeyError as exc:
        raise ValueError(f"required camera {exc.args[0]!r} is absent from DROID observation") from exc
    for name, image in (("external", external), ("wrist", wrist)):
        if image.ndim != 3 or image.shape[-1] != 3 or image.size == 0:
            raise ValueError(f"{name} image must be nonempty HWC RGB")
    return external, wrist


def _pose(observation: ObservationLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose = np.asarray(observation.cartesian_position, dtype=np.float32).reshape(-1)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("DROID cartesian_position must be finite xyz + Euler XYZ with shape (6,)")
    matrix = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    rot6d = np.concatenate((matrix[:, 0], matrix[:, 1])).astype(np.float32)
    return pose[:3], pose[3:], rot6d


def build_policy_observation(
    model: str, observation: ObservationLike, instruction: str, roles: CameraRoles
) -> dict[str, Any]:
    """Convert one already-RGB DROID observation to a published model schema."""
    if model not in MODEL_SPECS:
        raise ValueError(f"unknown model {model!r}")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be nonempty")
    external, wrist = _images(observation, roles)
    joints = np.asarray(observation.joints, dtype=np.float32).reshape(-1)
    gripper = np.asarray(observation.gripper, dtype=np.float32).reshape(-1)
    if joints.shape != (7,) or gripper.shape != (1,) or not np.isfinite(np.r_[joints, gripper]).all():
        raise ValueError("DROID observation must contain seven finite joints and one finite gripper position")
    if model in {"pi05", "pi05_lerobot"}:
        payload: dict[str, Any] = {
            "observation/exterior_image_1_left": external,
            "observation/wrist_image_left": wrist,
            "observation/joint_position": joints,
            "observation/gripper_position": gripper,
            "prompt": instruction,
        }
    elif model == "groot":
        xyz, _, rot6d = _pose(observation)
        payload = {
            "video": {
                "exterior_image_1_left": external[None, None, ...],
                "wrist_image_left": wrist[None, None, ...],
            },
            "state": {
                "eef_9d": np.concatenate((xyz, rot6d))[None, None, :],
                "joint_position": joints[None, None, :],
                "gripper_position": gripper[None, None, :],
            },
            "language": {"annotation.language.language_instruction": [[instruction]]},
        }
    elif model == "g05":
        payload = {
            "images": {
                "exterior_image": np.ascontiguousarray(external.transpose(2, 0, 1)),
                "wrist_image": np.ascontiguousarray(wrist.transpose(2, 0, 1)),
                "dummy_wrist_right": np.ascontiguousarray(wrist.transpose(2, 0, 1)),
            },
            "state": {"right_arm": joints, "right_gripper": 1.0 - gripper},
            "task": instruction,
            "frequency": 15,
            "embodiment_type": "Droid_Franka",
        }
    else:
        xyz, euler, rot6d = _pose(observation)
        cartesian = np.concatenate((xyz, rot6d)).astype(np.float32)
        payload = {
            "observation": {
                "base_0_rgb": external,
                "left_wrist_0_rgb": wrist,
                "cartesian_position": cartesian,
                "gripper_position": gripper,
                "joint_position": joints,
                "state": np.concatenate((cartesian, gripper)).astype(np.float32),
                "euler": euler,
            },
            "prompt": instruction,
        }
    validate_observation(model, payload)
    return payload


async def _websocket_request(url: str, payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    import websockets

    async with websockets.connect(url, compression=None, max_size=None, open_timeout=30, close_timeout=5) as socket:
        greeting = websocket_unpackb(await asyncio.wait_for(socket.recv(), timeout=30))
        if not isinstance(greeting, Mapping):
            raise ValueError("policy server greeting must be a mapping")
        await socket.send(websocket_packb(dict(payload)))
        response = websocket_unpackb(await asyncio.wait_for(socket.recv(), timeout=240))
        if not isinstance(response, Mapping):
            raise ValueError("policy server response must be a mapping")
        return dict(greeting), dict(response)


def _groot_action_mapping(response: Any) -> Mapping[str, Any]:
    if isinstance(response, Mapping) and "action" in response:
        response = response["action"]
    elif isinstance(response, (list, tuple)):
        if not response:
            raise ValueError("GR00T response action list is empty")
        response = response[0]
    if not isinstance(response, Mapping):
        raise ValueError("GR00T response must contain an action mapping")
    return dict(response)


def _groot_request(url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 240000)
    socket.setsockopt(zmq.SNDTIMEO, 30000)
    try:
        socket.connect(url)
        socket.send(groot_packb({"endpoint": "get_action", "data": {"observation": dict(payload), "options": None}}))
        response = groot_unpackb(socket.recv())
    finally:
        socket.close(linger=0)
        context.term()
    return _groot_action_mapping(response)


def convert_policy_response(model: str, raw: Mapping[str, Any], observation: ObservationLike) -> np.ndarray:
    """Apply only published model-to-DROID semantics, then validate the contract."""
    spec = MODEL_SPECS[model].control
    if model in {"pi05", "pi05_lerobot"}:
        actions = np.asarray(raw.get("actions"), dtype=np.float32)
    elif model == "groot":
        converted = groot_server_joint_to_target(raw)
        actions = np.concatenate((converted.joint_position_target, converted.raw_gripper_position), axis=-1)
    elif model == "g05":
        response_action = raw.get("action")
        if not isinstance(response_action, Mapping):
            raise ValueError("G05 response is missing action mapping")
        actions = g05_action_to_droid_joint_target(response_action, np.asarray(observation.gripper))[None, :]
    else:
        xyz, euler, _ = _pose(observation)
        actions = lap_relative_cartesian_to_target(np.asarray(raw.get("actions")), xyz, euler)
    return validate_action_chunk(actions, spec)


async def infer_policy_async(
    model: str, observation: ObservationLike, instruction: str, roles: CameraRoles, *, url: str | None = None
) -> PolicyResult:
    """Infer inside an existing event loop without nesting ``asyncio.run``."""
    spec = MODEL_SPECS.get(model)
    if spec is None:
        raise ValueError(f"unknown model {model!r}")
    payload = build_policy_observation(model, observation, instruction, roles)
    started = time.monotonic()
    if model == "groot":
        raw = await asyncio.to_thread(_groot_request, url or spec.url, payload)
    else:
        metadata, raw = await _websocket_request(url or spec.url, payload)
        if model == "pi05" and metadata.get("checkpoint") != "official-openpi-pi05_droid":
            raise ValueError("pi05 server is not the approved official OpenPI checkpoint")
        if model == "pi05_lerobot":
            validate_pi05_lerobot_metadata(metadata)
    actions = convert_policy_response(model, raw, observation)
    return PolicyResult(model, actions, raw, (time.monotonic() - started) * 1000)


def infer_policy(
    model: str, observation: ObservationLike, instruction: str, roles: CameraRoles, *, url: str | None = None
) -> PolicyResult:
    """Synchronous inference entry point used by the shadow-only CLI path."""
    return asyncio.run(infer_policy_async(model, observation, instruction, roles, url=url))


def infer_shadow_policy(
    model: str, observation: ObservationLike, instruction: str, roles: CameraRoles, *, url: str | None = None
) -> PolicyResult:
    """Model inference only; this function has no DROID import or robot handle."""
    return infer_policy(model, observation, instruction, roles, url=url)


def load_policy_shadow_observation(path: str | Path) -> tuple[ObservationLike, str | None]:
    """Load a DROID-shaped, non-pickle NPZ fixture for offline policy shadow tests."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"external_cam", "wrist_cam", "state", "cartesian_position"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"shadow observation is missing: {', '.join(sorted(missing))}")
        state = np.asarray(archive["state"], dtype=np.float32).reshape(-1)
        cartesian = np.asarray(archive["cartesian_position"], dtype=np.float32).reshape(-1)
        if state.shape != (8,) or cartesian.shape != (6,):
            raise ValueError("shadow state must be (8,) and cartesian_position must be (6,)")
        instruction = str(np.asarray(archive["instruction"]).item()) if "instruction" in archive.files else None
        external = np.asarray(archive["external_cam"])
        wrist = np.asarray(archive["wrist_cam"])

    @dataclass(frozen=True)
    class _RecordedObservation:
        images: Mapping[str, np.ndarray]
        joints: np.ndarray
        gripper: np.ndarray
        cartesian_position: np.ndarray

    return _RecordedObservation(
        {"external_image": external, "wrist_image": wrist},
        state[:7], state[7:], cartesian,
    ), instruction


PolicyInference = Callable[[ObservationLike, str], PolicyResult | Awaitable[PolicyResult]]
ActionAuditCallback = Callable[[Mapping[str, Any]], None]


async def run_live_policy(
    inference: PolicyInference,
    spec: ModelSpec,
    robot: DroidLike,
    instruction: str,
    *,
    control_hz: int = 15,
    max_control_steps: int = 1,
    audit_callback: ActionAuditCallback | None = None,
) -> None:
    """Bounded execution exclusively through the already-deployed DROID adapter."""
    if control_hz <= 0 or max_control_steps <= 0:
        raise ValueError("control_hz and max_control_steps must be positive")
    if robot.action_dim != spec.control.action_dim:
        raise ValueError("DROID action dimension does not match selected model")
    period = 1.0 / control_hz
    control_step = 0
    while control_step < max_control_steps:
        observation = robot.get_observation()
        result_or_awaitable = inference(observation, instruction)
        result = await result_or_awaitable if inspect.isawaitable(result_or_awaitable) else result_or_awaitable
        actions = legacy_colosseum_validator(result.actions, control_step, expected_dim=spec.control.action_dim)
        for action in actions:
            if control_step >= max_control_steps:
                return
            started = time.monotonic()
            command = np.asarray(action, dtype=np.float32).reshape(-1)
            if audit_callback is not None:
                event: dict[str, Any] = {
                    "event": "action_planned",
                    "model": spec.name,
                    "action_space": spec.control.action_space.value,
                    "control_step": control_step,
                    "model_dt_ms": result.model_dt_ms,
                    "model_action": command.tolist(),
                    # ``execute`` receives this exact vector.  Do not record a
                    # thresholded gripper preview as though it had been sent.
                    "droid_action": command.tolist(),
                }
                if spec.control.action_space is ActionSpace.JOINT_POSITION:
                    event["joint_position_target"] = command[:-1].tolist()
                elif spec.control.action_space is ActionSpace.JOINT_VELOCITY:
                    event["normalized_joint_velocity"] = command[:-1].tolist()
                else:
                    event["cartesian_position_target"] = command[:-1].tolist()
                audit_callback(event)
            robot.execute(command)
            if audit_callback is not None:
                audit_callback({"event": "execute_returned", "model": spec.name, "control_step": control_step})
            control_step += 1
            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                await asyncio.sleep(remaining)

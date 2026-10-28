from __future__ import annotations

from typing import Any, Mapping, Protocol

import numpy as np
from PIL import Image

from ..robot_interface import Robot, RobotObservation


class RobotEnvironment(Protocol):
    def get_observation(self) -> Mapping[str, Any]: ...

    def step(self, action: np.ndarray) -> Any: ...

    def reset(self) -> None: ...


class DroidRobot(Robot):
    """Small adapter around DROID/R2D2's RobotEnv."""

    joint_count = 7
    has_gripper = True
    action_dim = 8
    action_space_name = 'joint_position'

    def __init__(
        self,
        cameras: Mapping[str, str],
        *,
        action_space: str = "joint_position",
        image_size: tuple[int, int] = (512, 288),
        environment: RobotEnvironment | None = None,
        camera_backend: str = "zed",
        camera_config: str | None = None,
        do_reset: bool = True,
    ) -> None:
        if action_space not in {"joint_position", "joint_velocity", "cartesian_position"}:
            raise ValueError(f"unsupported action space: {action_space}")
        if environment is None:
            try:
                from droid.robot_env import RobotEnv
            except ModuleNotFoundError:
                try:
                    from r2d2.robot_env import RobotEnv
                except ModuleNotFoundError as exc:
                    raise RuntimeError("DROID or R2D2 must be installed on the robot computer") from exc
            camera_options = {}
            if camera_backend != "zed" or camera_config is not None:
                camera_options = dict(camera_backend=camera_backend, camera_config=camera_config)
            environment = RobotEnv(action_space=action_space, gripper_action_space="position", do_reset=do_reset, **camera_options)
        self.environment = environment
        self.cameras = dict(cameras)
        self.action_space = action_space
        self.action_space_name = action_space
        self.action_dim = 8 if action_space.startswith("joint") else 7
        self.image_size = image_size

    def get_observation(self) -> RobotObservation:
        raw = self.environment.get_observation()
        raw_images = raw.get("image")
        robot_state = raw.get("robot_state")
        if not isinstance(raw_images, Mapping) or not isinstance(robot_state, Mapping):
            raise ValueError("RobotEnv observation must contain image and robot_state mappings")

        images: dict[str, np.ndarray] = {}
        for image_name, camera_id in self.cameras.items():
            source = next(
                (value for key, value in raw_images.items() if camera_id in str(key) and "left" in str(key)),
                None,
            )
            if source is None:
                raise ValueError(f"camera {image_name} with ID {camera_id} was not found")
            images[image_name] = self._prepare_image(source)

        joints = self._vector(robot_state, "joint_positions")
        gripper = self._vector(robot_state, "gripper_position")
        cartesian = self._vector(robot_state, "cartesian_position")
        if joints.size != 7:
            raise ValueError(f"DROID must report 7 joint positions, got {joints.size}")
        if gripper.size != 1:
            raise ValueError("DROID must report one gripper position")
        return RobotObservation(images, joints, gripper, cartesian)

    def execute(self, action: np.ndarray) -> None:
        command = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        if command.size != self.action_dim:
            raise ValueError(
                f"DROID {self.action_space} action must have {self.action_dim} values, got {command.size}"
            )
        if not np.all(np.isfinite(command)):
            raise ValueError("action must contain finite values")
        command[-1] = 1.0 if command[-1] > 0.5 else 0.0
        if self.action_space == "joint_velocity":
            command[:-1] = np.clip(command[:-1], -1.0, 1.0)
        self.environment.step(command)

    def reset(self) -> None:
        """Use the existing DROID gripper-open and joint reset sequence."""
        self.environment.reset()

    def close(self) -> None:
        close = getattr(self.environment, "close", None)
        if callable(close):
            close()

    def _prepare_image(self, value: Any) -> np.ndarray:
        image = np.asarray(value)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"camera image must have shape (height, width, >=3), got {image.shape}")
        rgb = np.ascontiguousarray(image[..., :3][..., ::-1], dtype=np.uint8)
        width, height = self.image_size
        return np.asarray(Image.fromarray(rgb).resize((width, height), resample=Image.Resampling.LANCZOS))

    @staticmethod
    def _vector(value: Any, name: str) -> np.ndarray:
        if isinstance(value, Mapping):
            value = value.get(name)
        if value is None:
            raise ValueError(f"RobotEnv robot_state is missing {name}")
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain finite values")
        return vector

"""Shared robot contract, independent of any hardware SDK or model runtime."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class RobotObservation:
    """RGB images and state vectors in the adapter/model's agreed ordering."""

    images: Mapping[str, np.ndarray]
    joints: np.ndarray
    gripper: np.ndarray
    cartesian_position: np.ndarray


class Robot(ABC):
    """Base class for hardware drivers and the synthetic test robot.

    Implement all three lifecycle methods before instantiation. Concrete drivers
    also supply the hardware metadata below, either as class or instance fields.
    """
    joint_count: int
    has_gripper: bool
    action_dim: int
    action_space_name: str

    @abstractmethod
    def get_observation(self) -> RobotObservation:
        """Read RGB images and state in the agreed model-facing format."""
        raise NotImplementedError

    @abstractmethod
    def execute(self, action: np.ndarray) -> None:
        """Validate and execute one action using the hardware SDK."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Safely stop/hold and release resources; allow repeated cleanup."""
        raise NotImplementedError

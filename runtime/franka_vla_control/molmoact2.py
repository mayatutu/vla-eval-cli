"""Offline client for the official MolmoAct2-DROID ``/act`` protocol.

This module deliberately has no dependency on DROID.  It only validates and
transports observations and actions between a local policy server and the
offline action-contract layer.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from .contracts import ActionSpace, ControlSpec, validate_adapter_response


MOLMOACT2_DROID_METADATA = {
    "action_space": ActionSpace.JOINT_POSITION.value,
    "action_dim": 8,
    "semantic": "absolute_joint_position",
    # The checkpoint config declares max_action_horizon=15. `num_steps` is
    # the flow-matching denoising count, not the returned action horizon.
    "max_chunk": 15,
}


class MolmoAct2ProtocolError(ValueError):
    """Raised for an invalid MolmoAct2 wire payload or server response."""


@dataclass(frozen=True)
class MolmoAct2Request:
    """One validated observation accepted by the official DROID server."""

    external_cam: np.ndarray
    wrist_cam: np.ndarray
    instruction: str
    state: np.ndarray
    num_steps: int = 10

    def __post_init__(self) -> None:
        for name, image in (("external_cam", self.external_cam), ("wrist_cam", self.wrist_cam)):
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
                raise MolmoAct2ProtocolError(f"{name} must be a uint8 numpy array")
            if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] <= 0 or image.shape[1] <= 0:
                raise MolmoAct2ProtocolError(f"{name} must have non-empty HxWx3 shape, got {image.shape}")
        state = np.asarray(self.state, dtype=np.float32).reshape(-1)
        if state.shape != (8,) or not np.all(np.isfinite(state)):
            raise MolmoAct2ProtocolError("state must contain exactly eight finite float values")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise MolmoAct2ProtocolError("instruction must be a non-empty string")
        if not isinstance(self.num_steps, int) or not 1 <= self.num_steps <= 10:
            raise MolmoAct2ProtocolError("num_steps must be an integer in [1, 10]")
        object.__setattr__(self, "state", state)

    def payload(self) -> dict[str, Any]:
        return {
            "external_cam": _encode_numpy(self.external_cam),
            "wrist_cam": _encode_numpy(self.wrist_cam),
            "instruction": self.instruction,
            "state": _encode_numpy(self.state),
            "num_steps": self.num_steps,
        }


def _encode_numpy(value: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(value)
    return {
        "__numpy__": base64.b64encode(contiguous.tobytes()).decode("ascii"),
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
    }


def _decode_numpy(value: Any) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value)
    if not isinstance(value, Mapping) or set(value) != {"__numpy__", "dtype", "shape"}:
        raise MolmoAct2ProtocolError("expected a json_numpy array")
    dtype = np.dtype(value["dtype"])
    shape = value["shape"]
    if not isinstance(shape, list) or not shape or any(not isinstance(size, int) or size <= 0 for size in shape):
        raise MolmoAct2ProtocolError("json_numpy array has an invalid shape")
    count = int(np.prod(shape))
    if count > 1_000_000:
        raise MolmoAct2ProtocolError("json_numpy array is too large")
    try:
        raw = base64.b64decode(value["__numpy__"], validate=True)
    except (TypeError, ValueError) as exc:
        raise MolmoAct2ProtocolError("json_numpy array has invalid base64 data") from exc
    if len(raw) != count * dtype.itemsize:
        raise MolmoAct2ProtocolError("json_numpy array byte length does not match its shape")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def parse_act_response(body: bytes) -> tuple[np.ndarray, float]:
    """Decode and validate a successful official ``POST /act`` response."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MolmoAct2ProtocolError("server returned invalid JSON") from exc
    if not isinstance(payload, Mapping) or "actions" not in payload or "dt_ms" not in payload:
        raise MolmoAct2ProtocolError("server response must contain actions and dt_ms")
    try:
        dt_ms = float(payload["dt_ms"])
    except (TypeError, ValueError) as exc:
        raise MolmoAct2ProtocolError("server response dt_ms must be numeric") from exc
    if not np.isfinite(dt_ms) or dt_ms < 0:
        raise MolmoAct2ProtocolError("server response dt_ms must be finite and non-negative")
    actions = validate_adapter_response(
        {"actions": _decode_numpy(payload["actions"])}, MOLMOACT2_DROID_METADATA
    )
    return actions, dt_ms


class MolmoAct2Client:
    """HTTP client for a locally hosted official MolmoAct2-DROID server."""

    def __init__(self, base_url: str = "http://127.0.0.1:9101", timeout_s: float = 30.0) -> None:
        if not base_url.startswith("http://127.0.0.1:") and not base_url.startswith("http://localhost:"):
            raise ValueError("MolmoAct2 server must be bound to localhost")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._act_url = f"{base_url.rstrip('/')}/act"
        self._timeout_s = timeout_s

    def act(self, observation: MolmoAct2Request) -> tuple[np.ndarray, float]:
        request = Request(
            self._act_url,
            data=json.dumps(observation.payload(), separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_s) as response:
                return parse_act_response(response.read())
        except HTTPError as exc:
            raise MolmoAct2ProtocolError(f"MolmoAct2 server returned HTTP {exc.code}") from exc
        except URLError as exc:
            raise MolmoAct2ProtocolError("could not reach local MolmoAct2 server") from exc


def control_spec() -> ControlSpec:
    """Return the declared DROID action contract for this checkpoint."""
    return ControlSpec.from_metadata(MOLMOACT2_DROID_METADATA)

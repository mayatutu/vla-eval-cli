"""Wire-compatible NumPy MessagePack codec for OpenPI and G05 WebSocket services."""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np


def _encode(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported NumPy dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {b"__ndarray__": True, b"data": value.tobytes(), b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"cannot MessagePack value of type {type(value).__name__}")


def _decode(value: dict[Any, Any]) -> Any:
    marker = b"__ndarray__" if b"__ndarray__" in value else "__ndarray__" if "__ndarray__" in value else None
    if marker is not None:
        data_key = b"data" if isinstance(marker, bytes) else "data"
        dtype_key = b"dtype" if isinstance(marker, bytes) else "dtype"
        shape_key = b"shape" if isinstance(marker, bytes) else "shape"
        return np.ndarray(buffer=value[data_key], dtype=np.dtype(value[dtype_key]), shape=tuple(value[shape_key])).copy()
    scalar_marker = b"__npgeneric__" if b"__npgeneric__" in value else "__npgeneric__" if "__npgeneric__" in value else None
    if scalar_marker is not None:
        data_key = b"data" if isinstance(scalar_marker, bytes) else "data"
        dtype_key = b"dtype" if isinstance(scalar_marker, bytes) else "dtype"
        return np.dtype(value[dtype_key]).type(value[data_key])
    return value


def packb(value: Any) -> bytes:
    return msgpack.packb(value, default=_encode)


def unpackb(value: bytes) -> Any:
    return msgpack.unpackb(value, object_hook=_decode, raw=False, strict_map_key=False)

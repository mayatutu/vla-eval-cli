"""Small, non-pickle NumPy/MessagePack codec used by local policy services."""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError("object arrays are not permitted in policy messages")
        return {b"nd": True, b"type": array.dtype.str, b"shape": list(array.shape), b"data": array.tobytes()}
    if isinstance(value, np.generic):
        scalar = np.asarray(value)
        return {b"nd": False, b"type": scalar.dtype.str, b"data": scalar.tobytes()}
    raise TypeError(f"cannot MessagePack value of type {type(value).__name__}")


def _decode(value: dict[Any, Any]) -> Any:
    marker = value.get(b"nd")
    if marker is None:
        return value
    dtype = np.dtype(value[b"type"])
    data = value[b"data"]
    if marker is True:
        return np.ndarray(buffer=data, dtype=dtype, shape=tuple(value[b"shape"])).copy()
    return np.frombuffer(data, dtype=dtype, count=1)[0]


def packb(value: Any) -> bytes:
    return msgpack.packb(value, default=_encode, use_bin_type=True)


def unpackb(value: bytes) -> Any:
    return msgpack.unpackb(value, raw=False, object_hook=_decode, strict_map_key=False)

"""Load the two existing local wire-protocol packages without an installation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_TRANSPORT_SITE = None


def _load_package(name: str) -> None:
    raise RuntimeError(f'Missing {name}; install the transport dependencies in this Python environment')


def ensure_local_transport() -> None:
    """Make only MessagePack and WebSocket packages visible to Python 3.12.

    Normal environments already import these packages.  On the model machine,
    π0.5 uses Python 3.12 while these pure-Python transport packages already
    exist in the Python 3.11 LAP environment.  Loading only their package roots
    avoids leaking incompatible compiled 3.11 dependencies into LeRobot.
    """
    for name in ("msgpack", "websockets"):
        if name not in sys.modules and importlib.util.find_spec(name) is None:
            _load_package(name)

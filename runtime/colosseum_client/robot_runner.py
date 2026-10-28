"""Retained Colosseum action validation; no network runner."""
import numpy as np
from . import colosseum_pb2 as pb
from .client import ProtocolError
from .tensors import tensor_to_numpy

def action_chunk(plan: pb.ActionPlan, *, control_step: int, expected_dim: int = 8) -> np.ndarray:
    actions = tensor_to_numpy(plan.actions)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != expected_dim:
        raise ProtocolError(f"policy action chunk must have shape (horizon, {expected_dim}), got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ProtocolError("policy action chunk contains non-finite values")
    if plan.start_step != control_step:
        raise ProtocolError(
            f"policy action starts at step {plan.start_step}, expected {control_step}"
        )
    if plan.valid_until_step < control_step + len(actions) - 1:
        raise ProtocolError("policy action chunk expires before its final action")
    return np.asarray(actions, dtype=np.float32)

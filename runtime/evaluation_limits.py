"""Explicit per-profile limits; never infer behavior from a task identifier."""
import math

def action_seconds_for_profile(profile):
    value = profile.get('max_action_seconds', 30.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('max_action_seconds must be finite and positive')
    return float(value)

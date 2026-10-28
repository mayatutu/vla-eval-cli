"""Candidate bridge for the drawer checkpoint; no devices, sockets or ML imports."""
from collections.abc import Mapping
from threading import Lock

import numpy as np


REPO = "tsangb34/groot-n17-droid-open-drawer-5ep"
REVISION = "9d3784bde9ccc901206377970c75f2fa38062e98"
HORIZON = 40


def saved_observation_to_payload(path, instruction=None):
    """Read the existing shadow NPZ format without importing any Client modules."""
    with np.load(path, allow_pickle=False) as saved:
        state = np.asarray(saved["state"])
        pose = np.asarray(saved["cartesian_position"])
        if state.shape != (8,) or pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError("saved state/Cartesian position must have shape (8,)/(6,)")
        if instruction is None and "instruction" in saved:
            instruction = str(saved["instruction"].item())
        return {
            "video": {
                "exterior_image_1_left": saved["external_cam"][None, None],
                "wrist_image_left": saved["wrist_cam"][None, None],
            },
            "state": {"joint_position": state[None, None, :7],
                      "gripper_position": state[None, None, 7:]},
            "language": {"annotation.language.language_instruction": [[instruction]]},
        }


def observation_to_batch(observation):
    """Keep RGB, joint order and gripper convention; rename the two model views."""
    if not isinstance(observation, Mapping):
        raise ValueError("observation must be a mapping")
    video = observation["video"]
    state = observation["state"]
    batch = {}
    sizes = []
    for source, destination in (
        ("exterior_image_1_left", "left_image"),
        ("wrist_image_left", "wrist"),
    ):
        image = np.asarray(video[source])
        if (image.dtype != np.uint8 or image.ndim != 5
                or image.shape[:2] != (1, 1) or image.shape[-1] != 3
                or min(image.shape[2:4]) < 230):
            raise ValueError(f"{source}: expected uint8 RGB (1,1,H,W,3), H/W >= 230")
        sizes.append(image.shape)
        # The checkpoint processor owns cropping, resize, rescale and normalization.
        batch[f"observation.images.{destination}"] = image[0, 0].transpose(2, 0, 1).copy()
    if sizes[0] != sizes[1]:
        raise ValueError("both checkpoint views must have the same image dimensions")
    vectors = []
    for key, size in (("joint_position", 7), ("gripper_position", 1)):
        value = np.asarray(state[key])
        if (value.dtype.kind not in "fi" or value.shape != (1, 1, size)
                or not np.isfinite(value).all()):
            raise ValueError(f"{key}: expected finite numeric (1,1,{size})")
        vectors.append(value[0, 0].astype(np.float32))
    batch["observation.state"] = np.concatenate(vectors)
    if not np.isfinite(batch["observation.state"]).all():
        raise ValueError("state cannot be represented as float32")
    language = observation["language"]["annotation.language.language_instruction"]
    if (not isinstance(language, (list, tuple)) or len(language) != 1
            or not isinstance(language[0], (list, tuple)) or len(language[0]) != 1
            or not isinstance(language[0][0], str) or not language[0][0].strip()):
        raise ValueError("expected one nonempty instruction in shape (1,1)")
    batch["task"] = language[0][0]
    # The pinned pack step consumes observation.state only. It does not read the
    # Cartesian feature declared in config.json; do not invent Euler angles here.
    return batch


def decoded_to_response(actions):
    """Accept an entire decoded absolute chunk; never clip, invert or rebase it."""
    if hasattr(actions, "detach"):
        actions = actions.detach().float().cpu().numpy()
    actions = np.asarray(actions)
    if (actions.shape != (1, HORIZON, 8) or actions.dtype.kind != "f"
            or not np.isfinite(actions).all()):
        raise ValueError("decoded action must be finite float (1,40,8)")
    actions = actions.astype(np.float32, copy=True)
    if not np.isfinite(actions).all():
        raise ValueError("decoded actions cannot be represented as float32")
    return {
        "action": {
            "joint_position": actions[..., :7].copy(),
            "gripper_position": actions[..., 7:].copy(),
        },
        "info": {"repo": REPO, "revision": REVISION, "absolute_actions": True},
    }


class ChunkBridge:
    """One prediction and immediate whole-chunk decode share one state cache."""

    def __init__(self, preprocess, predict_chunk, postprocess, tensorize):
        self.preprocess = preprocess
        self.predict_chunk = predict_chunk
        self.postprocess = postprocess
        self.tensorize = tensorize
        self._lock = Lock()

    def get_action(self, observation):
        batch = observation_to_batch(observation)
        with self._lock:
            prepared = self.preprocess(self.tensorize(batch))
            raw = self.predict_chunk(prepared)
            decoded = self.postprocess(raw)
            return decoded_to_response(decoded)

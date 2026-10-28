"""Offline transport/decode-boundary tests; no ML stack, sockets, or devices."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from lingbot_checkpoint_backend import (
    IMAGE_MAPPING, LingbotBackend, checkpoint_shards, decoded_to_actions,
    load_weights_strict, observation_to_numpy_batch, validate_documents,
)


def observation():
    return {
        "video": {name: np.full((1, 1, 240, 320, 3), value, dtype=np.uint8)
                  for name, value in (("exterior_image_1_left", 11),
                                      ("exterior_image_2_left", 22),
                                      ("wrist_image_left", 33))},
        "state": {"joint_position": np.arange(7, dtype=np.float32)[None, None],
                  "gripper_position": np.array([[[.8]]], dtype=np.float32)},
        "language": {"annotation.language.language_instruction": [["close laptop"]]},
    }


def documents():
    root = Path(__file__).with_name("checkpoint-inspection-20261013") / "lingbot"
    config = json.loads((root / "config.json").read_text())
    stats = json.loads((root / "norm_stats.json").read_text())
    training = {"data": {"cameras": [key.removeprefix("observation.images.") for key, _ in IMAGE_MAPPING],
                         "img_size": 256, "state_norm_type": "none",
                         "norm_type": ["{'arm.position': 'meanstd'}", "{'effector.position': 'meanstd'}"]}}
    robot = {"images": [{target: {"origin_keys": source}} for target, source in IMAGE_MAPPING]}
    for category, prefix in (("states", "observation.state"), ("actions", "action")):
        robot[category] = []
        for part, start, end in (("arm", 0, 7), ("effector", 7, 8)):
            value = {"origin_keys": [{prefix: {"start": start, "end": end}}]}
            if category == "actions":
                value["subtract_state"] = part == "arm"
                if part == "arm":
                    value["relative_type"] = "joint"
            robot[category].append({f"{prefix}.{part}.position": value})
    return config, training, robot, stats


class InputTests(unittest.TestCase):
    def test_three_real_views_keep_their_roles_and_rgb_values(self):
        source = observation()
        batch = observation_to_numpy_batch(source)
        for name, value in (("left_image", 11), ("right_image", 22), ("wrist", 33)):
            image = batch["observation.images." + name]
            self.assertEqual(image.shape, (3, 240, 320))
            self.assertEqual(image.dtype, np.uint8)
            self.assertTrue(np.all(image == value))
        source["video"]["exterior_image_2_left"][...] = 0
        self.assertEqual(batch["observation.images.right_image"][0, 0, 0], 22)

    def test_missing_or_invalid_right_view_refuses_inference(self):
        for value in (None, np.zeros((240, 320, 3)), np.zeros((1, 1, 240, 320, 3))):
            source = observation()
            source["video"]["exterior_image_2_left"] = value
            with self.subTest(value=type(value)), self.assertRaises(ValueError):
                observation_to_numpy_batch(source)



class InferenceTests(unittest.TestCase):
    def test_native_decoder_receives_same_observation_without_second_rebase(self):
        prepared = {"state": np.arange(55, dtype=np.float32), "action_joint_mask": "mask"}
        raw = np.arange(50 * 55, dtype=np.float32).reshape(50, 55)
        decoded = np.linspace(-3, 4, 400, dtype=np.float32).reshape(50, 8)
        test = self

        class Transform:
            def apply(self, batch, *, policy_eval):
                test.assertTrue(policy_eval)
                test.assertIn("observation.images.right_image", batch)
                return prepared

            def unapply(self, batch):
                test.assertIs(batch["state"], prepared["state"])
                test.assertIs(batch["actions"], raw)
                test.assertEqual(batch["action_joint_mask"], "mask")
                return {"action": decoded}

        backend = LingbotBackend(Transform(), lambda batch: raw, lambda batch: batch)
        result = backend(observation())
        np.testing.assert_array_equal(result, decoded)
        self.assertEqual(result.shape, (50, 8))
        self.assertNotIn("actions", prepared)
        result[0, 0] = 123
        self.assertEqual(decoded[0, 0], -3)

    def test_bad_decoded_chunk_refuses_execution(self):
        for value in (np.zeros((25, 8)), np.zeros((1, 50, 8)),
                      np.zeros((50, 55)), np.full((50, 8), np.nan)):
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                decoded_to_actions(value)


class WeightTests(unittest.TestCase):
    def test_missing_shard_and_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            index = folder / "model.safetensors.index.json"
            for shard in ("model-00001.safetensors", "../outside.safetensors"):
                index.write_text(json.dumps({"weight_map": {"tensor": shard}}))
                with self.subTest(shard=shard), self.assertRaises(ValueError):
                    checkpoint_shards(folder)

    def test_shape_error_prevents_any_tensor_loading(self):
        class Header:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def keys(self):
                return ["tensor"]
            def get_slice(self, key):
                return SimpleNamespace(get_shape=lambda: [3])
            def get_tensor(self, key):
                raise AssertionError("weights must not load after a shape mismatch")
        model = SimpleNamespace(state_dict=lambda: {"tensor": SimpleNamespace(shape=(2,))})
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "a.safetensors").write_bytes(b"fixture")
            (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"tensor": "a.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "shape check"):
                load_weights_strict(model, folder, safe_open=lambda *args, **kwargs: Header(), torch=None)


if __name__ == "__main__":
    unittest.main()

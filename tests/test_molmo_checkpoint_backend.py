"""Contract checks without Torch/model weights/GPU/remote writes or devices."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from molmo_checkpoint_backend import (
    MolmoBackend, check_loading_info, decoded_to_actions, observation_to_inputs,
    validate_checkpoint_documents,
)
from test_pi05_checkpoint_backend import observation


SNAPSHOT = Path(__file__).with_name("checkpoint-inspection-20261013") / "molmoact2"
PROFILE = {"action_space": "joint_position", "action_dim": 8, "action_steps": 30,
           "norm_tag": "droid_rc", "num_steps": 10}


def three_views():
    value = observation()
    value["video"]["exterior_image_2_left"] = np.full((1, 1, 360, 640, 3), 123, np.uint8)
    return value




class MolmoInputOutputTests(unittest.TestCase):
    def test_three_real_rgb_views_order_no_rescale_and_state_preserved(self):
        payload = three_views()
        images, state, task = observation_to_inputs(payload)
        self.assertEqual([int(image[0, 0, 0]) for image in images], [255, 123, 17])
        self.assertTrue(all(image.dtype == np.uint8 for image in images))
        np.testing.assert_array_equal(state[:7], np.arange(7))
        self.assertEqual(state[7], np.float32(.7))
        self.assertEqual(task, "open drawer")
        payload["video"]["exterior_image_2_left"][...] = 0
        self.assertEqual(images[1][0, 0, 0], 123)

    def test_absent_or_invalid_third_view_rejected_not_fabricated(self):
        with self.assertRaises(KeyError):
            observation_to_inputs(observation())
        for image in (np.zeros((1, 1, 360, 640, 3)), np.zeros((360, 640, 3), np.uint8)):
            payload = three_views()
            payload["video"]["exterior_image_2_left"] = image
            with self.assertRaises(ValueError):
                observation_to_inputs(payload)

    def test_native_call_receives_checkpoint_tag_and_complete_chunk(self):
        decoded = np.linspace(-3, 4, 240, dtype=np.float32).reshape(1, 30, 8)
        observed = {}

        def predict(**kwargs):
            observed.update(kwargs)
            return SimpleNamespace(actions=decoded)

        backend = MolmoBackend(predict, "processor", lambda value: value, PROFILE)
        result = backend(three_views())
        self.assertEqual(observed["norm_tag"], "droid_rc")
        self.assertEqual(observed["num_steps"], 10)
        self.assertEqual(observed["n_action_steps"], 30)
        self.assertEqual(observed["inference_action_mode"], "continuous")
        self.assertFalse(observed["enable_depth_reasoning"])
        self.assertFalse(observed["enable_cuda_graph"])
        self.assertEqual(len(observed["images"]), 3)
        np.testing.assert_array_equal(result, decoded[0])
        result[0, 0] = 9
        self.assertEqual(decoded[0, 0, 0], -3)

    def test_bad_native_output_rejected(self):
        for value in (np.zeros((1, 15, 8)), np.zeros((1, 30, 32)),
                      np.zeros((1, 30, 8), np.int32), np.full((1, 30, 8), np.nan)):
            with self.assertRaises(ValueError):
                decoded_to_actions(value, 30)

    def test_missing_weights_are_never_accepted_as_random_initialization(self):
        check_loading_info({"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []})
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                check_loading_info({key: ["bad tensor"]})


if __name__ == "__main__":
    unittest.main()

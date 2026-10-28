"""CPU-only contract checks; no LeRobot import, model load, GPU, or devices."""
import unittest

import numpy as np

from pi05_checkpoint_backend import (
    ACTION_NAMES, IMAGE_KEYS, Pi05Backend, decoded_to_actions,
    load_weights_strict, observation_to_numpy_batch, validate_checkpoint_config,
    validate_processor_documents,
)


def observation():
    return {
        "video": {
            "exterior_image_1_left": np.full((1, 1, 360, 640, 3), 255, dtype=np.uint8),
            "wrist_image_left": np.full((1, 1, 360, 640, 3), 17, dtype=np.uint8),
        },
        "state": {
            "joint_position": np.arange(7, dtype=np.float32)[None, None],
            "gripper_position": np.array([[[.7]]], dtype=np.float32),
            "cartesian_position": np.arange(6, dtype=np.float32)[None, None],
        },
        "language": {"annotation.language.language_instruction": [["open drawer"]]},
    }


def config():
    return {
        "type": "pi05", "n_obs_steps": 1, "chunk_size": 50, "n_action_steps": 50,
        "use_relative_actions": False, "action_feature_names": list(ACTION_NAMES),
        "input_features": {
            "observation.state": {"shape": [8]},
            "observation.cartesian_position": {"shape": [6]},
            **{name: {"shape": [3, 360, 640]} for name in IMAGE_KEYS},
            "observation.images.empty_camera_0": {"shape": [3, 224, 224]},
        },
        "output_features": {"action": {"shape": [8]}},
    }


PROFILE = {"action_space": "joint_position", "action_steps": 50}


class InputTests(unittest.TestCase):
    def test_rgb_scaled_once_without_fabricated_camera(self):
        source = observation()
        source["video"]["exterior_image_1_left"][0, 0, 0, 0] = [255, 0, 127]
        batch = observation_to_numpy_batch(source)
        self.assertEqual(batch[IMAGE_KEYS[0]].shape, (3, 360, 640))
        self.assertEqual(batch[IMAGE_KEYS[0]].dtype, np.float32)
        np.testing.assert_allclose(batch[IMAGE_KEYS[0]][:, 0, 0], [1, 0, 127 / 255])
        self.assertAlmostEqual(float(batch[IMAGE_KEYS[1]][0, 0, 0]), 17 / 255)
        self.assertNotIn("observation.images.empty_camera_0", batch)
        self.assertEqual(batch["task"], "open drawer")

    def test_state_pose_and_gripper_unchanged_and_not_aliased(self):
        source = observation()
        batch = observation_to_numpy_batch(source)
        np.testing.assert_array_equal(batch["observation.state"][:7], np.arange(7))
        self.assertEqual(batch["observation.state"][7], np.float32(.7))
        np.testing.assert_array_equal(batch["observation.cartesian_position"], np.arange(6))
        source["state"]["cartesian_position"][...] = 9
        source["video"]["exterior_image_1_left"][...] = 0
        self.assertEqual(batch["observation.cartesian_position"][0], 0)
        self.assertEqual(batch[IMAGE_KEYS[0]][0, 0, 0], 1)

    def test_bad_pose_and_non_rgb_input_rejected(self):
        for bad in (np.zeros(6), np.zeros((1, 1, 7)), np.full((1, 1, 6), np.nan)):
            source = observation()
            source["state"]["cartesian_position"] = bad
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                observation_to_numpy_batch(source)
        source = observation()
        source["video"]["wrist_image_left"] = source["video"]["wrist_image_left"].astype(float)
        with self.assertRaises(ValueError):
            observation_to_numpy_batch(source)


class ConfigurationTests(unittest.TestCase):
    def test_expected_checkpoint_is_accepted(self):
        validate_checkpoint_config(config(), PROFILE)

    def test_velocity_relative_unknown_mapping_and_horizon_rejected(self):
        for name, value in (("type", "pi0"), ("use_relative_actions", True),
                            ("chunk_size", 15), ("n_action_steps", 8),
                            ("action_feature_names", ["joint_velocity"] * 8)):
            changed = config()
            changed[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_checkpoint_config(changed, PROFILE)
        with self.assertRaises(ValueError):
            validate_checkpoint_config(config(), {**PROFILE, "action_space": "joint_velocity"})

    def test_missing_pose_wrong_action_dimension_or_real_third_view_rejected(self):
        for change in ("pose", "action", "view"):
            changed = config()
            if change == "pose":
                del changed["input_features"]["observation.cartesian_position"]
            elif change == "action":
                changed["output_features"]["action"]["shape"] = [32]
            else:
                changed["input_features"]["observation.images.right_wrist"] = {"shape": [3, 360, 640]}
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_checkpoint_config(changed, PROFILE)

    def test_saved_pipeline_relative_flags_must_match_absolute_config(self):
        def document(name, enabled):
            return {"steps": [{"registry_name": name, "config": {"enabled": enabled}}]}

        pre = document("relative_actions_processor", False)
        post = document("absolute_actions_processor", False)
        validate_processor_documents(pre, post)
        for bad_pre, bad_post in (
            (document("relative_actions_processor", True), post),
            (pre, document("absolute_actions_processor", True)),
            ({"steps": []}, post), (pre, {"steps": post["steps"] * 2}),
        ):
            with self.subTest(pre=bad_pre, post=bad_post), self.assertRaises(ValueError):
                validate_processor_documents(bad_pre, bad_post)


class InferenceTests(unittest.TestCase):
    def test_full_postprocessed_chunk_preserved_without_clipping_or_rebasing(self):
        raw = np.zeros((1, 50, 8), dtype=np.float32)
        decoded = np.linspace(-3, 4, 400, dtype=np.float32).reshape(1, 50, 8)
        events = []

        def pre(batch):
            events.append("pre")
            self.assertEqual(batch[IMAGE_KEYS[0]].max(), 1)
            return {"prepared": True}

        def predict(batch):
            events.append("predict")
            self.assertEqual(batch, {"prepared": True})
            return raw

        def post(value):
            events.append("post")
            self.assertIs(value, raw)
            return decoded

        backend = Pi05Backend(pre, predict, post, lambda value: value, 50)
        result = backend(observation())
        self.assertEqual(events, ["pre", "predict", "post"])
        np.testing.assert_array_equal(result, decoded[0])
        result[0, 0] = 0
        self.assertEqual(decoded[0, 0, 0], -3)

    def test_bad_shape_nonfinite_and_integer_predictions_rejected(self):
        for value in (np.zeros((50, 8)), np.zeros((1, 40, 8)),
                      np.zeros((1, 50, 32)), np.ones((1, 50, 8), dtype=np.int32),
                      np.full((1, 50, 8), np.nan)):
            with self.subTest(shape=value.shape, dtype=value.dtype), self.assertRaises(ValueError):
                decoded_to_actions(value, 50)

    def test_inference_error_propagates(self):
        def fail(value):
            raise RuntimeError("inference failed")
        backend = Pi05Backend(lambda x: x, fail, lambda x: x, lambda x: x, 50)
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            backend(observation())


class WeightLoadingTests(unittest.TestCase):
    def test_native_key_fix_prefix_and_strictness(self):
        events = []

        class Policy:
            def _fix_pytorch_state_dict_keys(self, state, cfg):
                events.append("fix")
                self.asserted_config = cfg
                return {"already": state["original"], "model.other": 2}

            def load_state_dict(self, state, *, strict):
                events.append("load")
                self.loaded = state
                self.strict = strict

        policy = Policy()

        def loader(path, *, device):
            self.assertTrue(path.endswith("model.safetensors"))
            self.assertEqual(device, "cpu")
            return {"original": 1}

        load_weights_strict(policy, "/checkpoint", "config", load_file=loader)
        self.assertEqual(events, ["fix", "load"])
        self.assertTrue(policy.strict)
        self.assertEqual(policy.loaded, {"model.already": 1, "model.other": 2})

    def test_read_and_strict_state_errors_are_not_swallowed(self):
        def failed_read(*args, **kwargs):
            raise OSError("weight file truncated")

        with self.assertRaisesRegex(OSError, "truncated"):
            load_weights_strict(object(), "/checkpoint", None, load_file=failed_read)

        class Policy:
            def _fix_pytorch_state_dict_keys(self, state, cfg):
                return state

            def load_state_dict(self, state, *, strict):
                raise RuntimeError("missing tensor")

        with self.assertRaisesRegex(RuntimeError, "missing tensor"):
            load_weights_strict(Policy(), "/checkpoint", None, load_file=lambda *a, **k: {"a": 1})

    def test_ambiguous_prefix_collision_rejected(self):
        class Policy:
            def _fix_pytorch_state_dict_keys(self, state, cfg):
                return state

        with self.assertRaisesRegex(ValueError, "collision"):
            load_weights_strict(Policy(), "/checkpoint", None,
                                load_file=lambda *a, **k: {"a": 1, "model.a": 2})


if __name__ == "__main__":
    unittest.main()

"""CPU tests using actual saved configs and extracted native preprocessing code."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from molmo_lerobot_checkpoint_backend import (
    IMAGE_KEYS, MolmoLeRobotBackend, decoded_to_actions, load_policy_weights_strict,
    observation_to_numpy_batch, validate_checkpoint_documents, verify_base_assets,
    restore_empty_runtime_buffers,
    configure_checkpoint_horizon,
)


HERE = Path(__file__).parent
SOURCE = HERE / "suite-native-sources/lerobot_molmo/policies/molmoact2"
PROFILE = {"action_space": "joint_position", "action_dim": 8, "action_steps": 30}


def documents():
    suite = json.loads((HERE / "model-suite-inspection-20261013.json").read_text(encoding="utf-8"))
    docs = suite["models"]["molmoact2-droid-open-drawer-5ep"]["documents"]
    return tuple(json.loads(docs[name]["text"]) for name in
                 ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"))


def native_functions(filename, names):
    """Run the captured native NumPy helpers unchanged, without importing Torch."""
    tree = ast.parse((SOURCE / filename).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError("native function snapshot is incomplete")
    for node in nodes:
        node.decorator_list = []
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + nodes, type_ignores=[])
    scope = {"np": np}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE / filename), "exec"), scope)
    return scope


def observation():
    return {
        "video": {name: np.full((1, 1, 240, 320, 3), value, dtype=np.uint8)
                  for name, value in (("exterior_image_1_left", 255),
                                      ("exterior_image_2_left", 127),
                                      ("wrist_image_left", 17))},
        "state": {"joint_position": np.arange(7, dtype=np.float32)[None, None],
                  "gripper_position": np.array([[[.7]]], dtype=np.float32),
                  "cartesian_position": np.arange(6, dtype=np.float32)[None, None]},
        "language": {"annotation.language.language_instruction": [["open drawer"]]},
    }




class InputOutputTests(unittest.TestCase):
    def test_pose_and_joint_gripper_values_are_preserved_and_copied(self):
        source = observation()
        result = observation_to_numpy_batch(source)
        np.testing.assert_array_equal(result["observation.state"], [0, 1, 2, 3, 4, 5, 6, np.float32(.7)])
        np.testing.assert_array_equal(result["observation.cartesian_position"], np.arange(6))
        source["state"]["cartesian_position"][...] = 123
        self.assertEqual(result["observation.cartesian_position"][0], 0)

    def test_no_fabricated_camera_or_cartesian_pose(self):
        for field in ("right", "pose"):
            source = observation()
            if field == "right":
                del source["video"]["exterior_image_2_left"]
            else:
                del source["state"]["cartesian_position"]
            with self.subTest(field=field), self.assertRaises(ValueError):
                observation_to_numpy_batch(source)

    def test_native_postprocessor_output_is_not_rebased_or_clipped_again(self):
        raw = np.zeros((1, 30, 8), dtype=np.float32)
        decoded = np.linspace(-3, 4, 240, dtype=np.float32).reshape(1, 30, 8)
        events = []
        def pre(batch):
            events.append("pre")
            return batch
        def predict(batch):
            events.append("predict")
            return raw
        def post(value):
            events.append("post")
            self.assertIs(value, raw)
            return decoded
        backend = MolmoLeRobotBackend(pre, predict, post, lambda value: value, 30)
        np.testing.assert_array_equal(backend(observation()), decoded[0])
        self.assertEqual(events, ["pre", "predict", "post"])
        for value in (np.zeros((30, 8)), np.zeros((1, 30, 32)), np.full((1, 30, 8), np.nan)):
            with self.assertRaises(ValueError):
                decoded_to_actions(value, 30)


class StrictWeightTests(unittest.TestCase):


    def test_restore_empty_caches_keeps_persistent_checkpoint_value(self):
        loaded = SimpleNamespace(is_meta=False)
        empty = SimpleNamespace(is_meta=True, numel=lambda: 0)
        module = SimpleNamespace(_non_persistent_buffers_set={"_pos_cos_cache", "_pos_sin_cache"},
                                 _buffers={"_pos_cos_cache": empty, "_pos_sin_cache": empty},
                                 original_inv_freq=SimpleNamespace(is_meta=True), inv_freq=loaded)
        def register(name, value, *, persistent):
            self.assertFalse(persistent)
            module._buffers[name] = value
        module.register_buffer = register
        def empty_like(value, *, device):
            self.assertIs(value, empty)
            self.assertEqual(device, "cpu")
            return SimpleNamespace(is_meta=False, numel=lambda: 0)
        restore_empty_runtime_buffers(SimpleNamespace(named_modules=lambda: [("rotary", module)]),
                                      SimpleNamespace(empty_like=empty_like))
        self.assertIs(module.inv_freq, loaded)
        self.assertIs(module.original_inv_freq, loaded)
        self.assertTrue(all(not value.is_meta for value in module._buffers.values()))

    def test_unknown_or_nonempty_runtime_buffer_is_not_fabricated(self):
        for name, count in (("unknown_cache", 0), ("_pos_sin_cache", 4)):
            value = SimpleNamespace(is_meta=True, numel=lambda: count)
            module = SimpleNamespace(_non_persistent_buffers_set={name}, _buffers={name: value})
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unreviewed meta"):
                restore_empty_runtime_buffers(SimpleNamespace(named_modules=lambda: [("rotary", module)]), None)

    def test_wrong_prefix_or_shape_fails_before_loading_tensors(self):
        class Header:
            def __init__(self, key, shape):
                self.key, self.shape = key, shape
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def keys(self):
                return [self.key]
            def get_slice(self, key):
                return SimpleNamespace(get_shape=lambda: self.shape)
            def get_tensor(self, key):
                raise AssertionError("no tensor should load after a metadata mismatch")
        policy = SimpleNamespace(state_dict=lambda: {"model.weight": SimpleNamespace(shape=(2,))})
        for key, shape in (("weight", [2]), ("model.weight", [3])):
            with self.subTest(key=key, shape=shape), self.assertRaisesRegex(ValueError, "strict weight mismatch"):
                load_policy_weights_strict(policy, "unused.safetensors",
                                           safe_open=lambda *args, **kwargs: Header(key, shape), torch=None)

    def test_strict_assignment_errors_are_not_swallowed(self):
        class Tensor:
            dtype = "float32"
            def is_floating_point(self):
                return True
            def to(self, **kwargs):
                return self
        class Header:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def keys(self):
                return ["model.weight"]
            def get_slice(self, key):
                return SimpleNamespace(get_shape=lambda: [2])
            def get_tensor(self, key):
                return Tensor()
        class Policy:
            def state_dict(self):
                return {"model.weight": SimpleNamespace(shape=(2,))}
            def load_state_dict(self, values, *, strict, assign):
                if not strict or not assign:
                    raise AssertionError("strict assignment is required")
                raise RuntimeError("incompatible checkpoint tensor")
        with self.assertRaisesRegex(RuntimeError, "incompatible checkpoint"):
            load_policy_weights_strict(Policy(), "unused.safetensors",
                                       safe_open=lambda *args, **kwargs: Header(),
                                       torch=SimpleNamespace(bfloat16="bfloat16"))


if __name__ == "__main__":
    unittest.main()

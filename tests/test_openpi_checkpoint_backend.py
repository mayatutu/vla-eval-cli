"""CPU boundary checks; deliberately not presented as native OpenPI acceptance."""
import copy
import ast
import dataclasses
from enum import Enum
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from openpi_checkpoint_backend import (
    OpenPiBackend, build_backend, decoded_to_actions, inspect_checkpoint,
    local_asset_resolver, observation_to_inputs, strict_model_config,
    validate_statistics, verify_runtime_sources,
)


def observation():
    return {
        "video": {"exterior_image_1_left": np.full((1, 1, 240, 320, 3), 255, np.uint8),
                  "wrist_image_left": np.full((1, 1, 240, 320, 3), 19, np.uint8)},
        "state": {"joint_position": np.arange(7, dtype=np.float32)[None, None],
                  "gripper_position": np.array([[[.4]]], np.float32)},
        "language": {"annotation.language.language_instruction": [["close laptop"]]},
    }


def statistics():
    return {"norm_stats": {kind: {"mean": [0.] * 8, "std": [1.] * 7 + [0.],
                                  "q01": [-1.] * 7 + [0.], "q99": [1.] * 7 + [0.]}
                           for kind in ("state", "actions")}}


class BoundaryTests(unittest.TestCase):
    def test_input_keeps_8d_state_for_split_transform_and_hwc_rgb(self):
        source = observation()
        source["video"]["exterior_image_1_left"][0, 0, 0, 0] = [255, 0, 127]
        result = observation_to_inputs(source)
        self.assertEqual(set(result), {"observation/exterior_image_1_left",
                                      "observation/wrist_image_left", "state", "prompt"})
        self.assertEqual(result["state"].shape, (8,))
        np.testing.assert_array_equal(result["state"][:7], np.arange(7))
        self.assertEqual(result["state"][7], np.float32(.4))
        np.testing.assert_array_equal(result["observation/exterior_image_1_left"][0, 0], [255, 0, 127])
        self.assertEqual(result["observation/wrist_image_left"].dtype, np.uint8)
        self.assertEqual(result["observation/wrist_image_left"].shape, (240, 320, 3))
        self.assertEqual(result["prompt"], "close laptop")

    def test_native_absolute_output_is_not_rebased_inverted_or_clipped(self):
        actions = np.linspace(-4, 4, 24).reshape(3, 8).astype(np.float32)
        captured = []
        def infer(inputs):
            captured.append(inputs)
            return {"actions": actions}
        result = OpenPiBackend(infer, 3)(observation())
        np.testing.assert_array_equal(result, actions)
        self.assertEqual(len(captured), 1)
        result[0, 0] = 99
        self.assertEqual(actions[0, 0], -4)

    def test_native_errors_propagate(self):
        def failed(_):
            raise RuntimeError("Orbax native inference failed")
        with self.assertRaisesRegex(RuntimeError, "Orbax"):
            OpenPiBackend(failed, 3)(observation())

    def test_bad_output_unknown_horizon_and_nonfinite_values_rejected(self):
        for actions in (np.zeros((1, 3, 8)), np.zeros((3, 32)),
                        np.zeros((3, 8), dtype=np.int32), np.full((3, 8), np.nan),
                        np.full((3, 8), 1e100)):
            with self.subTest(shape=actions.shape), self.assertRaises(ValueError):
                decoded_to_actions({"actions": actions}, 3)
        for steps in (None, 0, True):
            with self.assertRaises(ValueError):
                OpenPiBackend(lambda _: None, steps)

    def test_constant_gripper_statistics_are_allowed_but_invalid_ranges_are_not(self):
        validate_statistics(statistics())
        for kind, field, value in (("actions", "std", [-1.] * 8),
                                   ("state", "q01", [9.] * 8),
                                   ("actions", "mean", [0.] * 32)):
            changed = statistics()
            changed["norm_stats"][kind][field] = value
            with self.assertRaises(ValueError):
                validate_statistics(changed)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "params").mkdir()
        (self.root / "assets/CloseLaptop_joint_v21").mkdir(parents=True)
        contents = {"train_config.py": b"CONFIG = reviewed_task_config\n",
                    "assets/CloseLaptop_joint_v21/norm_stats.json": json.dumps(statistics()).encode()}
        for name, data in contents.items():
            (self.root / name).write_bytes(data)
        self.profile = {"backend": "openpi", "action_space": "joint_position", "action_dim": 8,
                        "checkpoint": str(self.root), "dataset_id": "CloseLaptop_joint_v21",
                        "asset_sha256": {name: hashlib.sha256(data).hexdigest()
                                         for name, data in contents.items()}}

    def test_asset_inspection_does_not_claim_weight_or_runtime_acceptance(self):
        result = inspect_checkpoint(self.profile)
        self.assertTrue(result["configuration_hashes_verified"])
        self.assertFalse(result["weights_loaded"])
        self.assertFalse(result["full_weight_hash_verified"])
        self.assertFalse(result["native_runtime_verified"])

    def test_changed_config_missing_stats_and_old_velocity_profiles_are_rejected(self):
        changed = copy.deepcopy(self.profile)
        changed["action_space"] = "joint_velocity"
        with self.assertRaises(ValueError):
            inspect_checkpoint(changed)
        (self.root / "train_config.py").write_text("CONFIG = unreviewed_config\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            inspect_checkpoint(self.profile)

    def test_incomplete_download_and_asset_escape_are_rejected(self):
        partial = self.root / "params/weights.incomplete"
        partial.touch()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            inspect_checkpoint(self.profile)
        partial.unlink()
        self.profile["asset_sha256"]["../outside"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "relative paths"):
            inspect_checkpoint(self.profile)

    def test_unverified_runtime_fails_before_attempting_native_import_or_gpu(self):
        with self.assertRaisesRegex(ValueError, "runtime source root"):
            build_backend(self.profile)

    def test_offline_asset_resolver_allows_only_local_files_and_pinned_tokenizer(self):
        tokenizer = self.root / "tokenizer.model"
        tokenizer.write_bytes(b"fixture")
        resolve = local_asset_resolver(tokenizer)
        self.assertEqual(resolve("gs://big_vision/paligemma_tokenizer.model"), tokenizer.resolve())
        self.assertEqual(resolve(str(self.root)), self.root.resolve())
        for url in ("gs://other/tokenizer.model", "https://example.com/model"):
            with self.assertRaisesRegex(FileNotFoundError, "disabled"):
                resolve(url)

    def test_native_model_config_requests_full_tree_validation(self):
        calls = []
        @dataclasses.dataclass(frozen=True)
        class NativeConfig:
            action_horizon: int = 50
            def load(self, params, *, remove_extra_params=True):
                calls.append((params, remove_extra_params))
                return "native-model"
        config = strict_model_config(NativeConfig(), NativeConfig)
        self.assertEqual(config.load({"params": 1}), "native-model")
        self.assertEqual(calls, [({"params": 1}, False)])
        with self.assertRaises(ValueError):
            config.load({}, remove_extra_params=True)


def native_definitions(path, names, namespace):
    """Execute only reviewed native CPU transforms, never their ML imports."""
    source = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in source.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
                        names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)




if __name__ == "__main__":
    unittest.main()

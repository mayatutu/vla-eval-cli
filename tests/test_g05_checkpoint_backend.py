"""Semantic CPU checks; no model imports, GPU, network or devices."""
import json
import importlib.util
from contextlib import nullcontext
import os
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import g05_checkpoint_backend as backend


def wire():
    return {"video": {
        "exterior_image_1_left": np.full((1, 1, 240, 320, 3), 17, np.uint8),
        "wrist_image_left": np.full((1, 1, 240, 320, 3), 41, np.uint8)},
        "state": {"joint_position": np.arange(7, dtype=np.float32)[None, None],
                  "gripper_position": np.array([[[.25]]], np.float32)},
        "language": {"annotation.language.language_instruction": [["open drawer"]]}}


def config(embodiment="Droid_Franka"):
    is_drawer = embodiment == "Droid_Franka"
    names = ("exterior", "wrist_left", "wrist_right") if is_drawer else ("head_rgb", "left_wrist_rgb", "dummy_wrist_right")
    images = [{"key": names[0], "camera_type": "exterior"},
              {"key": names[1], "camera_type": "wrist_left"},
              {"key": names[2], "camera_type": "wrist_right", "dummy": True,
               "shape": [3, 224, 224] if is_drawer else [3, 256, 160]}]
    parts = [{"key": "right_arm", "raw_shape": 7}, {"key": "right_gripper", "raw_shape": 1}]
    return {"model": {"model_arch": {
        "_target_": "g05.models.g05.g05_policy_qwen35.G05PolicyQwen35",
        "continuous_action": True, "return_continuous_action": True,
        "action_dim": 20 if is_drawer else 27}}, "data": {"action_size": 32, "processors": {
            embodiment: {"shape_meta": {"action": parts, "state": parts, "images": images},
                         "action_state_transforms": [{"_target_":
                             "g05.data_processor.transforms.relative_action.RelativeJointTransform",
                             "keys": ["right_arm"]}]}}}}


def verified_profile(embodiment="Droid_Franka", mode="droid"):
    # Fixture values are explicit test choices, not proof of checkpoint semantics.
    return {"action_steps": 8, "control_hz": 30, "control_hz_evidence": "test fixture",
            "g05_gripper_mode": mode, "g05_gripper_evidence": "test fixture",
            "g05_embodiment": embodiment}


class G05MappingTests(unittest.TestCase):
    def test_each_native_layout_receives_its_exact_names_and_dummy_shape(self):
        for embodiment in ("Droid_Franka", "galaxea_r1lite"):
            images = backend.validate_native_config(config(embodiment), verified_profile(embodiment))
            result = backend.wire_to_native(wire(), 30, embodiment=embodiment,
                                           image_meta=images, gripper_mode="droid")
            self.assertEqual(set(result["images"]), {item["key"] for item in images})
            np.testing.assert_array_equal(result["images"][images[0]["key"]], 17)
            np.testing.assert_array_equal(result["images"][images[1]["key"]], 41)
            dummy = result["images"][images[2]["key"]]
            self.assertEqual(dummy.shape, tuple(images[2]["shape"]))
            self.assertEqual(dummy.max(), 0)
            np.testing.assert_array_equal(result["state"]["right_gripper"], [.25])
            self.assertEqual(result["embodiment_type"], embodiment)

    def test_gripper_direction_is_explicit_and_symmetric(self):
        images = config()["data"]["processors"]["Droid_Franka"]["shape_meta"]["images"]
        for mode, expected in (("droid", .25), ("inverted", .75)):
            raw = backend.wire_to_native(wire(), 30, embodiment="Droid_Franka",
                                        image_meta=images, gripper_mode=mode)
            self.assertEqual(raw["state"]["right_gripper"].item(), expected)
            result = backend.native_to_droid({"right_arm": np.zeros((1, 8, 7)),
                                              "right_gripper": np.full((1, 8, 1), expected)},
                                             [.25], 8, gripper_mode=mode)
            np.testing.assert_array_equal(result[:, 7], .25)
        with self.assertRaisesRegex(ValueError, "explicit"):
            backend.wire_to_native(wire(), 30, embodiment="Droid_Franka", image_meta=images, gripper_mode=None)

    def test_preserves_absolute_joint_values_and_selects_requested_prefix(self):
        joints = np.arange(224, dtype=np.float32).reshape(1, 32, 7)
        output = backend.native_to_droid({"right_arm": joints, "right_gripper": np.zeros((1, 32, 1))},
                                         [.5], 8, gripper_mode="droid")
        np.testing.assert_array_equal(output[:, :7], joints[0, :8])
        np.testing.assert_array_equal(output[:, 7], 0)

    def test_absent_gripper_holds_current_but_missing_arm_fails(self):
        action = {"right_arm": np.zeros((1, 8, 7)), "_absent_keys": {"right_gripper"}}
        np.testing.assert_allclose(backend.native_to_droid(action, [.3], 8, gripper_mode="droid")[:, 7], .3)
        action["_absent_keys"] = {"right_arm"}
        with self.assertRaisesRegex(ValueError, "main arm"):
            backend.native_to_droid(action, [.3], 8, gripper_mode="droid")

    def test_malformed_or_short_outputs_fail(self):
        for action in ({"right_arm": np.zeros((1, 1, 7))},
                       {"right_arm": np.full((1, 8, 7), np.nan)},
                       {"right_arm": np.zeros((1, 8, 7)), "right_gripper": np.zeros((1, 7, 1))}):
            with self.assertRaises(ValueError):
                backend.native_to_droid(action, [.3], 8, gripper_mode="droid")

    def test_native_pipeline_decodes_once_without_manual_joint_rebase(self):
        calls = []
        class Inferencer:
            def __init__(self, *args, **kwargs): pass
            def infer(self, samples):
                calls.append(samples)
                return [{"right_arm": np.ones((1, 32, 7)), "right_gripper": np.full((1, 32, 1), .8)}]
        native = SimpleNamespace(PolicyInferencer=Inferencer, _parse_task_and_plan=lambda raw: None,
                                 build_obs_dict=lambda raw, processor: raw)
        images = backend.validate_native_config(config(), verified_profile())
        predictor = backend.G05Backend(native, "policy", "processor", action_steps=8, control_hz=30,
                                       device="cuda:0", embodiment="Droid_Franka", image_meta=images,
                                       gripper_mode="droid")
        result = predictor(wire())
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(result[:, :7], 1)
        np.testing.assert_allclose(result[:, 7], .8)


class G05ConfigurationTests(unittest.TestCase):
    def test_native_loader_gets_path_and_selected_layout_and_resolved_processor(self):
        checkpoint=Path("selected-checkpoint").resolve(); processor=Path("selected-processor").resolve()
        calls=[]; updates={}; cfg=SimpleNamespace(data={})
        def load(run_dir, weights, overrides):
            self.assertIsInstance(run_dir,Path)
            calls.append((run_dir,weights,overrides));return cfg
        native=SimpleNamespace(find_run_dir=lambda _:checkpoint,load_config_from_run_dir=load)
        omega=SimpleNamespace(OmegaConf=SimpleNamespace(update=lambda cfg,key,value,**kw:updates.update({key:value})))
        profile={**verified_profile("galaxea_r1lite"),"g05_weights_file":"model.pt"}
        with patch.dict(sys.modules,{"omegaconf":omega}), \
                patch.object(backend, "_preserve_native_tokenizer_aliases", return_value=nullcontext()):
            self.assertIs(backend._load_config(native,checkpoint,processor,profile),cfg)
        self.assertEqual(calls[0][1],str(checkpoint/"model.pt"))
        self.assertIn("eval_embodiment=galaxea_r1lite",calls[0][2])
        self.assertEqual(updates["model.processor.tokenizer_params.pretrained_model_name_or_path"],str(processor))
        self.assertEqual(updates["datastatics_path"],str(checkpoint/"dataset_stats.json"))

    def test_unknown_gripper_or_frequency_fails_before_model_import(self):
        for key in ("control_hz", "control_hz_evidence", "g05_gripper_mode", "g05_gripper_evidence"):
            profile = verified_profile(); profile[key] = None
            with self.subTest(key=key), patch.object(backend, "_profile_paths", side_effect=AssertionError("no model reads")):
                with self.assertRaises(ValueError): backend.build_backend(profile)

    def test_wrong_internal_width_head_relative_transform_or_views_rejected(self):
        changes = []
        cfg = config(); cfg["model"]["model_arch"]["action_dim"] = 27; changes.append(cfg)
        cfg = config(); cfg["model"]["model_arch"]["return_continuous_action"] = False; changes.append(cfg)
        cfg = config(); cfg["data"]["processors"]["Droid_Franka"]["action_state_transforms"] = []; changes.append(cfg)
        cfg = config(); cfg["data"]["processors"]["Droid_Franka"]["shape_meta"]["images"][2]["dummy"] = False; changes.append(cfg)
        for cfg in changes:
            with self.assertRaises(ValueError): backend.validate_native_config(cfg, verified_profile())

    def test_two_weight_layouts_require_checkpoint_own_stats_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); checkpoint=root/"ckpt"; processor=root/"processor"; repo=root/"repo"; overlay=root/"overlay"
            for name in (".hydra/config.yaml", "model.pt", "checkpoints/model_state_dict.pt", "dataset_stats.json", "action_tokenizer.pt"):
                path=checkpoint/name; path.parent.mkdir(parents=True,exist_ok=True);path.write_text("fixture")
            for path in (repo/"scripts/serve_policy.py",overlay/"src/g05/models/g05/inferencer.py"):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text("fixture")
            processor.mkdir();(repo/"src").mkdir()
            profile=dict(checkpoint=str(checkpoint),processor=str(processor),g05_repo=str(repo),g05_overlay=str(overlay))
            for name in ("model.pt", "checkpoints/model_state_dict.pt"):
                self.assertEqual(backend._profile_paths({**profile,"g05_weights_file":name})[0],checkpoint.resolve())
            with self.assertRaisesRegex(ValueError,"changed"):
                backend._profile_paths({**profile,"asset_sha256":{"dataset_stats.json":"not-the-actual-sha"}})
            with self.assertRaises(ValueError):backend._weights_path(checkpoint,{"g05_weights_file":"../other.pt"})
            (checkpoint/"dataset_stats.json").unlink()
            with self.assertRaises(FileNotFoundError):backend._profile_paths(profile)

    def test_native_config_resolution_restores_cwd_and_environment_even_on_error(self):
        original=os.getcwd()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{"TASK":"previous"}):
            with self.assertRaises(RuntimeError):
                with backend._native_config_context(Path(tmp),{**verified_profile(),"checkpoint":"/fixture"}):
                    self.assertEqual(Path.cwd(),Path(tmp));self.assertEqual(os.environ["RC_DATA"],"/fixture")
                    raise RuntimeError("test")
            self.assertEqual(os.getcwd(),original);self.assertEqual(os.environ["TASK"],"previous")





class G05StrictLoadTests(unittest.TestCase):
    def test_missing_extra_and_shape_changes_rejected_before_native_loader(self):
        model=SimpleNamespace(state_dict=lambda:{"weight":np.zeros((2,3))})
        for state in ({},{"other":np.zeros((2,3))},{"weight":np.zeros((2,4))}):
            calls=[]
            module=SimpleNamespace(load_state_dict_safely=lambda *a,**k:calls.append(1))
            with backend.strict_native_weights(module),self.assertRaises(RuntimeError):
                module.load_state_dict_safely(model,state)
            self.assertEqual(calls,[])

    def test_native_dtype_tie_logic_reused_but_partial_loading_disabled(self):
        model=SimpleNamespace(state_dict=lambda:{"weight":np.zeros((2,3))})
        calls=[]
        def original(*args,**kwargs):
            calls.append(kwargs)
            return {"model":model,"loaded_count":1,"truly_missing":[],"partial_loaded_keys":[],
                    "mismatched_keys":[],"unexpected_keys":[],"extra_loaded_keys":[]}
        module=SimpleNamespace(load_state_dict_safely=original)
        with backend.strict_native_weights(module):
            self.assertIs(module.load_state_dict_safely(model,{"weight":np.zeros((2,3))}),model)
        self.assertIs(module.load_state_dict_safely,original)
        self.assertEqual(calls[0]["load_config"]["partial_load_keys"],[])
        self.assertEqual(calls[0]["extra_prefixes"],[])
        self.assertTrue(calls[0]["return_info"])

    def test_native_diagnostics_or_incomplete_counts_are_not_swallowed(self):
        model=SimpleNamespace(state_dict=lambda:{"weight":np.zeros((2,3))})
        for info in ({"loaded_count":0},{"loaded_count":1,"partial_loaded_keys":["weight"]}):
            original=lambda *a,**k:info
            module=SimpleNamespace(load_state_dict_safely=original)
            with self.assertRaises(RuntimeError),backend.strict_native_weights(module):
                module.load_state_dict_safely(model,{"weight":np.zeros((2,3))})
            self.assertIs(module.load_state_dict_safely,original)

    def test_unreviewed_key_or_token_remapping_rejected(self):
        model=SimpleNamespace(state_dict=lambda:{"weight":np.zeros((2,3))})
        module=SimpleNamespace(load_state_dict_safely=lambda *a,**k:self.fail("do not load"))
        for setting in ("key_prefix_to_remove","ignore_key_prefixes","token_row_remap"):
            with backend.strict_native_weights(module),self.assertRaises(RuntimeError):
                module.load_state_dict_safely(model,{"weight":np.zeros((2,3))},load_config={setting:"unknown"})


if __name__ == "__main__": unittest.main()

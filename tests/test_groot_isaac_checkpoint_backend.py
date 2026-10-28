"""Offline boundary checks, not native model/GPU acceptance."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from groot_isaac_checkpoint_backend import (
    ACTION_STEPS, COSMOS_ID, EMBODIMENT, IsaacGr00tBackend, checked_loading_result,
    decoded_to_actions, inspect_checkpoint, local_native_loaders, observation_to_inputs,
    validate_configuration, validate_statistics, verify_runtime_sources,
)

ROOT = Path(__file__).parent


def observation():
    return {
        "video": {"exterior_image_1_left": np.full((1, 1, 240, 320, 3), 73, np.uint8),
                  "wrist_image_left": np.full((1, 1, 240, 320, 3), 19, np.uint8)},
        "state": {"joint_position": np.arange(7, dtype=np.float32)[None, None],
                  "gripper_position": np.array([[[.4]]], np.float32)},
        "language": {"annotation.language.language_instruction": [["close laptop"]]},
    }


def response():
    return ({"arm": np.arange(112, dtype=np.float32).reshape(1, 16, 7),
             "gripper": np.linspace(-.1, 1.1, 16, dtype=np.float32).reshape(1, 16, 1)}, {})


def native_method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    fn.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


class WireTests(unittest.TestCase):
    def test_input_renames_without_changing_rgb_state_or_instruction(self):
        source = observation()
        source["video"]["exterior_image_1_left"][0, 0, 0, 0] = [255, 0, 127]
        result = observation_to_inputs(source)
        np.testing.assert_array_equal(result["video"]["left"], source["video"]["exterior_image_1_left"])
        np.testing.assert_array_equal(result["video"]["wrist"], source["video"]["wrist_image_left"])
        np.testing.assert_array_equal(result["state"]["arm"], source["state"]["joint_position"])
        np.testing.assert_array_equal(result["state"]["gripper"], source["state"]["gripper_position"])
        self.assertEqual(result["state"]["arm"].dtype, np.float32)
        self.assertEqual(result["language"], {"annotation.human.task_description": [["close laptop"]]})

    def test_decoded_output_has_no_second_rebase_clipping_or_gripper_inversion(self):
        original = response()
        policy = SimpleNamespace(get_action=lambda obs: original)
        result = IsaacGr00tBackend(policy)(observation())
        np.testing.assert_array_equal(result, np.concatenate(list(original[0].values()), axis=-1)[0])
        self.assertEqual(result.shape, (16, 8))
        result[0, 0] = -99
        self.assertEqual(original[0]["arm"][0, 0, 0], 0)

    def test_malformed_or_nonfinite_output_is_rejected(self):
        for result in ({}, (response()[0],), ({"arm": np.zeros((1, 40, 7)), "gripper": np.zeros((1,16,1))}, {}),
                       ({"arm": np.zeros((1,16,7)), "gripper": np.full((1,16,1), np.nan)}, {}),
                       ({"arm": np.zeros((1,16,7)), "gripper": np.full((1,16,1), 1e100)}, {})):
            with self.subTest(result=str(result)[:30]), self.assertRaises(ValueError):
                decoded_to_actions(result)






class LoaderTests(unittest.TestCase):
    def test_missing_mismatched_unexpected_or_meta_weights_fail(self):
        model = SimpleNamespace(named_parameters=lambda: [("ok", SimpleNamespace(is_meta=False))])
        self.assertIs(checked_loading_result((model, {}), "test"), model)
        for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            with self.assertRaisesRegex(RuntimeError, "exact load"):
                checked_loading_result((model, {field:["bad"]}), "test")
        model.named_parameters = lambda: [("bad", SimpleNamespace(is_meta=True))]
        with self.assertRaisesRegex(RuntimeError, "meta"):
            checked_loading_result((model, {}), "test")

    def test_loader_forces_local_assets_and_restores_symbols_after_failure(self):
        calls = []
        model = SimpleNamespace(named_parameters=lambda: [])
        def model_factory(path, **kwargs):
            calls.append((str(path), kwargs))
            return model, {}
        def processor_factory(path, **kwargs):
            calls.append((str(path), kwargs))
            return "processor"
        def original_build(path, kwargs):
            calls.append((str(path), kwargs))
            return "cosmos_processor"
        model_class = SimpleNamespace(from_pretrained=model_factory)
        processor_class = SimpleNamespace(from_pretrained=processor_factory)
        policy = SimpleNamespace(AutoModel=model_class, AutoProcessor=processor_class)
        backbone = SimpleNamespace(Qwen3VLForConditionalGeneration=model_class)
        processing = SimpleNamespace(build_processor=original_build)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, cosmos = Path(temporary)/"task", Path(temporary)/"cosmos"
            checkpoint.mkdir(); cosmos.mkdir(); (cosmos/"config.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with local_native_loaders(policy, backbone, processing, checkpoint, cosmos, "bf16"):
                    policy.AutoModel.from_pretrained(checkpoint)
                    backbone.Qwen3VLForConditionalGeneration.from_pretrained(COSMOS_ID)
                    processing.build_processor(COSMOS_ID, {})
                    policy.AutoProcessor.from_pretrained(checkpoint/"processor")
                    with self.assertRaises(ValueError):
                        processing.build_processor("unexpected/remote-model", {})
                    raise RuntimeError("intentional")
            self.assertIs(policy.AutoModel, model_class)
            self.assertIs(policy.AutoProcessor, processor_class)
            self.assertIs(backbone.Qwen3VLForConditionalGeneration, model_class)
            self.assertIs(processing.build_processor, original_build)
            self.assertTrue(all(kwargs["local_files_only"] for _, kwargs in calls))
            self.assertEqual(calls[1][0], str(cosmos.resolve()))
            self.assertEqual(calls[2][0], str(cosmos.resolve()))
            self.assertTrue(calls[0][1]["output_loading_info"])
            self.assertTrue(calls[1][1]["output_loading_info"])

    def test_nested_meta_construction_is_allowed_but_task_must_materialize(self):
        parameter = SimpleNamespace(is_meta=True)
        model = SimpleNamespace(named_parameters=lambda: [("backbone.weight", parameter)])
        state = {"materialize": True, "loading_info": {}}
        backbone = SimpleNamespace(Qwen3VLForConditionalGeneration=SimpleNamespace(
            from_pretrained=lambda path, **kwargs: (model, state["loading_info"])))
        def task_factory(path, **kwargs):
            parameter.is_meta = True
            loaded = backbone.Qwen3VLForConditionalGeneration.from_pretrained(COSMOS_ID)
            self.assertIs(loaded, model)
            self.assertTrue(parameter.is_meta)
            parameter.is_meta = not state["materialize"]
            return model, {}
        policy = SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=task_factory),
                                 AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **k: None))
        processing = SimpleNamespace(build_processor=lambda *a, **k: None)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, cosmos = Path(temporary)/"task", Path(temporary)/"cosmos"
            checkpoint.mkdir(); cosmos.mkdir(); (cosmos/"config.json").write_text("{}")
            with local_native_loaders(policy, backbone, processing, checkpoint, cosmos, "bf16"):
                self.assertIs(policy.AutoModel.from_pretrained(checkpoint), model)
                self.assertFalse(parameter.is_meta)
                state["materialize"] = False
                with self.assertRaisesRegex(RuntimeError, "Isaac task has unresolved meta"):
                    policy.AutoModel.from_pretrained(checkpoint)
                for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
                    state["loading_info"] = {field: ["bad"]}
                    with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "Cosmos weights are not an exact load"):
                        policy.AutoModel.from_pretrained(checkpoint)


if __name__ == "__main__":
    unittest.main()

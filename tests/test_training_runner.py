"""CPU-only checks of data, truncation, and recovery boundaries; no model imports."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("training_run", ROOT / "training/run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class TrainingBoundaryTests(unittest.TestCase):
    def test_invalid_model_json_remains_invalid(self):
        for raw in ('{"decision":[],"skill":null}', '{"decision":"continue","skill":{}}',
                    '```json\n{"decision":"pause","skill":null}\n```',
                    '{"decision":"cancel","skill":null,"explanation":"x"}'):
            self.assertIsNone(runner.parse_prediction(raw))
        self.assertEqual(runner.parse_prediction('{"decision":"replace_task","skill":"craft"}'),
                         {"decision": "replace_task", "skill": "craft"})

    def test_mask_requires_complete_target_and_excludes_prompt(self):
        runner.assert_mask([1, 2], [3, 4], [1, 2, 3, 4], [-100, -100, 3, 4])
        for ids, labels in (([1, 2, 3], [-100, -100, 3]),
                            ([1, 2, 3, 4], [1, 2, 3, 4]),
                            ([1, 2, 3, 4], [-100, -100, -100, -100])):
            with self.assertRaises(ValueError):
                runner.assert_mask([1, 2], [3, 4], ids, labels)

    def test_checkpoint_requires_optimizer_scheduler_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            checkpoint = directory / "checkpoint-25"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.json").write_text("{}")
            with self.assertRaises(ValueError):
                runner.latest_checkpoint(directory)
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
                (checkpoint / name).write_bytes(b"fixture")
            self.assertEqual(runner.latest_checkpoint(directory), checkpoint)

    def test_prediction_resume_checks_order_and_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.jsonl"
            path.write_text(json.dumps({"id": "a", "raw_output": "bad", "prediction": None}) + "\n")
            self.assertEqual(len(runner.read_completed_predictions(path, ["a", "b"])), 1)
            with self.assertRaises(ValueError):
                runner.read_completed_predictions(path, ["b", "a"])

    def test_expired_runtime_stops_without_compute(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(TimeoutError):
                runner.ensure_budget(Path(folder), "2000-01-01T00:00:00Z", 7)

    def test_runtime_budget_is_shared_and_never_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner.ensure_budget(root, None, 7)
            budget_path = root / "runs/runtime_budget.json"
            initial = budget_path.read_bytes()
            runner.ensure_budget(root, None, 7)
            self.assertEqual(initial, budget_path.read_bytes())
            expired = runner.read_json(budget_path)
            expired["started_at"] = "2000-01-01T00:00:00+00:00"
            runner.write_json(budget_path, expired)
            with self.assertRaises(TimeoutError):
                runner.ensure_budget(root, None, 7)

    def test_training_invokes_yaml_with_identical_json_receipt(self):
        import yaml
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            name = "qwen2.5-7b"
            runner.write_json(root / "results" / f"{name}_base.meta.json", {"status": "complete"})
            report = {"models": {name: {"revision": "a" * 40}}, "data_sha256": {"test": "fixture"}}
            args = SimpleNamespace(max_length=512, epochs=2, hard_deadline_utc=None, max_gpu_hours=7)
            calls = []
            def fake_logged(command, project, run_dir, received_args):
                calls.append(command)
                config_path = Path(command[-1])
                self.assertEqual(config_path.suffix, ".yaml")
                self.assertEqual(yaml.safe_load(config_path.read_text()),
                                 runner.read_json(run_dir / "resolved_config.json"))
                adapter = run_dir / "adapter"
                adapter.mkdir()
                for filename in ("adapter_model.safetensors", "adapter_config.json", "trainer_state.json"):
                    (adapter / filename).write_bytes(b"fixture")
            with mock.patch.object(runner, "require_preflight", return_value=report), \
                 mock.patch.object(runner, "compute_dtype", return_value="bf16"), \
                 mock.patch.object(runner, "environment", return_value={}), \
                 mock.patch.object(runner, "run_logged", side_effect=fake_logged), \
                 mock.patch.dict("sys.modules", {"torch": SimpleNamespace(float16="fp16", bfloat16="bf16")}):
                runner.train(root, name, args)
            self.assertEqual(calls[0][:2], ["llamafactory-cli", "train"])

    def test_failed_process_receipt_is_preserved_on_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            previous = {"utc": "fixture", "status": "failed", "raw_exit_code": 17}
            runner.write_json(root / "last_train_process.json", previous)
            args = SimpleNamespace(hard_deadline_utc=None, max_gpu_hours=7)
            with mock.patch.object(runner, "gpu_snapshot", return_value="CPU fixture"), \
                 self.assertRaises(runner.subprocess.CalledProcessError):
                runner.run_logged([runner.sys.executable, "-c", "raise SystemExit(3)"], root, root, args)
            archived = list(root.glob("train_process.failed.*.json"))
            self.assertEqual(len(archived), 1)
            self.assertEqual(runner.read_json(archived[0]), previous)
            self.assertEqual(runner.read_json(root / "last_train_process.json")["raw_exit_code"], 3)


if __name__ == "__main__":
    unittest.main()

"""Temporary CPU fixtures only: no model weights, inference, or publication."""
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_release", ROOT / "publication/prepare_release.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
NAME = "qwen2.5-7b"
REVISION = "a" * 40


def fixture(root):
    """Construct internally consistent synthetic evidence solely for boundary tests."""
    for directory in ("data", "evaluation", "results", f"runs/{NAME}/adapter"):
        (root / directory).mkdir(parents=True)
    for relative in ("data/test.json", "data/DATASET_CARD.md", "evaluation/evaluate.py", "evaluation/README.md"):
        shutil.copyfile(ROOT / relative, root / relative)
    run = root / "runs" / NAME
    adapter = run / "adapter"
    (adapter / "adapter_model.safetensors").write_bytes(b"CPU TEST FIXTURE: NOT MODEL WEIGHTS")
    (adapter / "optimizer.pt").write_bytes(b"must not be published")
    adapter_hash = release.digest(adapter / "adapter_model.safetensors")
    hashes = {"train": "fixture", "val": "fixture", "test": release.digest(root / "data/test.json")}
    release.write_json(adapter / "adapter_config.json", {"base_model_name_or_path": release.MODELS[NAME], "peft_type": "LORA"})
    release.write_json(run / "training_complete.json", {"status": "complete", "adapter_sha256": adapter_hash,
                       "utc": "fixture", "selection": "validation loss", "environment": {"gpu": "fixture GPU,private UUID"}})
    release.write_json(root / "runs/preflight.json", {"passed": True, "data_sha256": hashes,
                       "llamafactory_commit": "fixture", "models": {NAME: {"model_id": release.MODELS[NAME],
                       "revision": REVISION, "lengths": {"test": {"count": 300}}}}})
    config = {"model_name_or_path": release.MODELS[NAME], "model_revision": REVISION, "template": "qwen",
              "finetuning_type": "lora", "quantization_method": "bnb", "quantization_bit": 4,
              "quantization_type": "nf4", "seed": 42, "lora_rank": 16, "lora_alpha": 32,
              "lora_dropout": .05, "learning_rate": .0001, "num_train_epochs": 2, "cutoff_len": 512,
              "output_dir": "/content/private/run", "dataset_dir": "/Users/private/data"}
    release.write_json(run / "training_identity.json", {"config": config, "data_sha256": hashes})
    shared = release.evaluator(root)
    gold = shared.load_gold(root / "data/test.json")
    predictions = [{"id": row["id"], "raw_output": row["output"], "latency_seconds": .25} for row in gold]
    metrics = shared.evaluate(gold, predictions)
    for kind in ("base", "adapter"):
        path = root / "results" / f"{NAME}_{kind}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
        identity = {"model": NAME, "kind": kind, "revision": REVISION, "test_sha256": hashes["test"],
                    "do_sample": False, "max_new_tokens": 96, "enable_thinking": False}
        if kind == "adapter":
            identity["adapter_sha256"] = adapter_hash
        release.write_json(path.with_suffix(".meta.json"), {"status": "complete", "count": 300, "identity": identity})
        release.write_json(path.with_name(path.stem + "_metrics.json"), metrics)


class ReleaseBoundaries(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_allowlist_and_idempotence(self):
        output = release.prepare(self.root, NAME)
        self.assertFalse((output / "optimizer.pt").exists())
        self.assertNotIn("/content/private", (output / "provenance.json").read_text())
        self.assertNotIn("private UUID", (output / "provenance.json").read_text())
        self.assertEqual(release.read_json(output / "adapter_config.json")["revision"], REVISION)
        card = (output / "README.md").read_text()
        self.assertIn("not a Minecraft player", card)
        self.assertIn("single-seed experiment", card)
        example = card.split("```python\n", 1)[1].split("```", 1)[0]
        compile(example, "model-card-example", "exec")
        self.assertEqual(release.prepare(self.root, NAME), output)

    def test_modified_adapter_rejected(self):
        (self.root / "runs" / NAME / "adapter/adapter_model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Adapter hash"):
            release.prepare(self.root, NAME)

    def test_incomplete_predictions_rejected(self):
        path = self.root / "results" / f"{NAME}_adapter.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "300 locked test IDs"):
            release.prepare(self.root, NAME)

    def test_stale_metrics_rejected(self):
        path = self.root / "results" / f"{NAME}_adapter_metrics.json"
        metrics = release.read_json(path)
        metrics["joint_accuracy"] = 0
        release.write_json(path, metrics)
        with self.assertRaisesRegex(ValueError, "metrics do not match"):
            release.prepare(self.root, NAME)

    def test_mlx_metadata_rejected(self):
        path = self.root / "runs" / NAME / "training_identity.json"
        identity = release.read_json(path)
        identity["config"]["quantization_method"] = "mlx"
        release.write_json(path, identity)
        with self.assertRaisesRegex(ValueError, "package MLX adapters separately"):
            release.prepare(self.root, NAME)

    def test_missing_completion_rejected(self):
        (self.root / "runs" / NAME / "training_complete.json").unlink()
        with self.assertRaises(FileNotFoundError):
            release.prepare(self.root, NAME)

    def test_optional_decision_diagnostic_preserves_primary_score(self):
        shutil.copyfile(ROOT / "evaluation/diagnostics.py", self.root / "evaluation/diagnostics.py")
        shared = release.evaluator(self.root)
        gold = shared.load_gold(self.root / "data/test.json")
        for kind in ("base", "adapter"):
            path = self.root / "results" / f"{NAME}_{kind}.jsonl"
            if kind == "base":
                predictions = shared.load_predictions(path)
                for row in predictions:
                    parsed = json.loads(row["raw_output"])
                    if parsed["decision"] != "replace_task":
                        parsed["skill"] = "craft"
                    row["raw_output"] = json.dumps(parsed)
                path.write_text("".join(json.dumps(row) + "\n" for row in predictions))
                release.write_json(path.with_name(path.stem + "_metrics.json"), shared.evaluate(gold, predictions))
            subprocess.run([sys.executable, str(self.root / "evaluation/diagnostics.py"), "--gold",
                            str(self.root / "data/test.json"), "--predictions", str(path), "--output",
                            str(path.with_name(path.stem + "_diagnostics.json"))], check=True, capture_output=True)
        output = release.prepare(self.root, NAME)
        self.assertEqual(release.read_json(output / "base_metrics.json")["decision_accuracy"], .2)
        self.assertEqual(release.read_json(output / "base_diagnostics.json")["decision_accuracy"]["value"], 1.0)
        card = (output / "README.md").read_text()
        self.assertIn("| Strict decision accuracy | 20.00% | 100.00% |", card)
        self.assertIn("| Decision-only accuracy | 100.00% | 100.00% |", card)
        self.assertIn("Secondary post hoc diagnostics", card)


if __name__ == "__main__":
    unittest.main()

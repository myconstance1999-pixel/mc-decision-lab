"""Temporary byte fixtures; no model execution or publication."""
import importlib.util
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path


SPEC = importlib.util.spec_from_file_location("sanitize_packages", Path(__file__).with_name("sanitize_packages.py"))
sanitizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sanitizer)


class PackagePrivacy(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for relative, archive in sanitizer.PACKAGE_ARCHIVES.items():
            folder = self.root / relative
            folder.mkdir(parents=True)
            for name in ("adapter.safetensors", "predictions.jsonl", "metrics.json", "DATASET_CARD.md"):
                (folder / name).write_bytes(b"unchanged scientific fixture\n")
            if relative == sanitizer.PEFT_PACKAGE:
                sanitizer.write_json(folder / "provenance.json", {
                    "completed_at": "private wall-clock fixture",
                    "packages": {"torch": "fixture-build", "unrelated-host-tool": "fixture"},
                    "training_seconds": 123.5, "gpu_model": "fixture GPU",
                })
                sanitizer.write_json(folder / "RELEASE_MANIFEST.json", {
                    path.name: sanitizer.digest(path) for path in sorted(folder.iterdir())
                })
            (self.root / "artifacts").mkdir(exist_ok=True)
            with zipfile.ZipFile(self.root / "artifacts" / archive, "w") as output:
                for path in sorted(folder.iterdir()):
                    output.writestr(zipfile.ZipInfo(path.name, (2020, 2, 3, 4, 5, 6)), path.read_bytes())
        sanitizer.write_json(self.root / "artifacts/RELEASE_CHECKSUMS.json", {"original": "fixture"})

    def test_redaction_preservation_backup_and_deterministic_archives(self):
        folder = self.root / sanitizer.PEFT_PACKAGE
        original_provenance = (folder / "provenance.json").read_bytes()
        original_zip = (self.root / "artifacts/mc-decision-qwen2.5-7b-peft.zip").read_bytes()
        receipt = sanitizer.sanitize_release(self.root)
        self.assertEqual(receipt["protected_files_unchanged"], 8)
        metadata = json.loads((folder / "provenance.json").read_text())
        self.assertNotIn("completed_at", metadata)
        self.assertEqual(metadata["packages"], {"torch": "fixture-build"})
        self.assertEqual(metadata["training_seconds"], 123.5)
        self.assertEqual(metadata["gpu_model"], "fixture GPU")
        self.assertEqual(sanitizer.sanitize_provenance(metadata), metadata)
        backup = self.root / "artifacts/privacy-backup/packages"
        self.assertEqual((backup / sanitizer.PEFT_PACKAGE / "provenance.json").read_bytes(), original_provenance)
        self.assertEqual((backup / "artifacts/mc-decision-qwen2.5-7b-peft.zip").read_bytes(), original_zip)
        for relative, filename in sanitizer.PACKAGE_ARCHIVES.items():
            with zipfile.ZipFile(self.root / "artifacts" / filename) as archive:
                self.assertEqual(archive.comment, b"")
                for entry in archive.infolist():
                    self.assertEqual(entry.date_time, sanitizer.ZIP_TIMESTAMP)
                    self.assertEqual(entry.extra, b"")
                    self.assertEqual(entry.comment, b"")
                    self.assertEqual((entry.external_attr >> 16) & 0o777, 0o644)
                    self.assertEqual(archive.read(entry), (self.root / relative / entry.filename).read_bytes())
            for path in (self.root / relative).iterdir():
                os.utime(path, (1234567890, 1234567890))
                path.chmod(0o600)
        repeated = sanitizer.sanitize_release(self.root)
        self.assertEqual(repeated["changed_public_files"], [])
        self.assertEqual(repeated["archives"], receipt["archives"])
        self.assertEqual((backup / sanitizer.PEFT_PACKAGE / "provenance.json").read_bytes(), original_provenance)

    def test_manifest_mismatch_preserves_public_metadata(self):
        folder = self.root / sanitizer.PEFT_PACKAGE
        before = (folder / "provenance.json").read_bytes()
        (folder / "predictions.jsonl").write_bytes(b"unexpected mutation")
        with self.assertRaisesRegex(ValueError, "does not match"):
            sanitizer.sanitize_release(self.root)
        self.assertEqual((folder / "provenance.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

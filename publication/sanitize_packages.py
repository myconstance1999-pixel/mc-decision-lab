"""Remove incidental public metadata and rebuild reproducible release archives.

Only public PEFT provenance and its hash manifest are edited. Model weights,
predictions, metrics, tokenizer files, and the MLX package remain unchanged.
ZIP entry dates are fixed compatibility values, not experiment or release times.
This module never uploads files or executes model code.
"""
import argparse
import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path


ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_METADATA = {
    "entry_timestamp": list(ZIP_TIMESTAMP),
    "meaning": "Normalized ZIP compatibility value; not an experiment or release time.",
    "entry_permissions": "regular file, 0644",
}
# Declared training dependencies plus selected model/tokenizer/runtime packages.
# This intentionally is not a complete inventory of the host environment.
TRAINING_PACKAGES = frozenset({
    "accelerate", "bitsandbytes", "datasets", "huggingface_hub", "llamafactory",
    "numpy", "peft", "protobuf", "PyYAML", "safetensors", "sentencepiece",
    "tokenizers", "torch", "torchaudio", "torchdata", "torchvision",
    "transformers", "triton", "trl",
})
PACKAGE_ARCHIVES = {
    "runs_mlx/release-qwen3-1.7b-mlx": "mc-decision-qwen3-1.7b-mlx.zip",
    "artifacts/hf/qwen2.5-7b": "mc-decision-qwen2.5-7b-peft.zip",
}
PEFT_PACKAGE = "artifacts/hf/qwen2.5-7b"


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def package_files(package):
    """Public release packages are flat; never follow a link into private files."""
    files = sorted(Path(package).iterdir())
    if not files or any(path.is_symlink() or not path.is_file() for path in files):
        raise ValueError("Release package must contain only regular, non-symlink files")
    return files


def sanitize_provenance(provenance):
    """Return public metadata without changing the original runtime receipt."""
    result = dict(provenance)
    result.pop("completed_at", None)
    result["packages"] = {
        name: version for name, version in sorted(provenance.get("packages", {}).items())
        if name in TRAINING_PACKAGES
    }
    result["privacy_transformations"] = {
        "omitted_fields": ["completed_at"],
        "package_inventory": (
            "Only direct training and selected model/tokenizer/runtime dependencies are published; "
            "unrelated installed packages are omitted. This is not a complete environment lock."
        ),
        "scientific_artifacts": "Model weights, raw predictions, metrics, and dataset bytes are unchanged.",
        "archive_metadata": ZIP_METADATA,
    }
    return result


def sanitize_peft_package(package):
    package = Path(package)
    manifest_path = package / "RELEASE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = {path.name: digest(path) for path in package_files(package) if path != manifest_path}
    if actual != manifest:
        raise ValueError("Public package does not match RELEASE_MANIFEST.json; preserve and investigate")
    provenance_path = package / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    write_json(provenance_path, sanitize_provenance(provenance))
    actual["provenance.json"] = digest(provenance_path)
    write_json(manifest_path, actual)


def write_normalized_zip(package, archive):
    """Archive exact package bytes with stable ordering, dates, and permissions."""
    files = package_files(package)
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".normalized-zip-", dir=archive.parent) as temporary:
        stage = Path(temporary) / archive.name
        with zipfile.ZipFile(stage, "w") as output:
            for path in files:
                info = zipfile.ZipInfo(path.name, date_time=ZIP_TIMESTAMP)
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                output.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        stage.replace(archive)
    return {"sha256": digest(archive), "bytes": archive.stat().st_size, "zip_metadata": ZIP_METADATA}


def package_hashes(root):
    return {
        path.relative_to(root).as_posix(): digest(path)
        for relative in PACKAGE_ARCHIVES for path in package_files(root / relative)
    }


def backup_originals(root, hashes):
    """Keep first originals; repeated sanitation must never overwrite the backup."""
    backup = root / "artifacts/privacy-backup/packages"
    backup.mkdir(parents=True, exist_ok=True)
    originals = [
        PEFT_PACKAGE + "/provenance.json",
        PEFT_PACKAGE + "/RELEASE_MANIFEST.json",
        "artifacts/RELEASE_CHECKSUMS.json",
        *("artifacts/" + name for name in PACKAGE_ARCHIVES.values()),
    ]
    for relative in originals:
        source, destination = root / relative, backup / relative
        if source.exists() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    baseline = backup / "original-package-hashes.json"
    if not baseline.exists():
        write_json(baseline, hashes)
    return backup


def sanitize_release(root):
    root = Path(root).resolve()
    before = package_hashes(root)
    backup = backup_originals(root, before)
    sanitize_peft_package(root / PEFT_PACKAGE)
    after = package_hashes(root)
    mutable = {PEFT_PACKAGE + "/provenance.json", PEFT_PACKAGE + "/RELEASE_MANIFEST.json"}
    changed = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    if not set(changed) <= mutable:
        raise ValueError("Unexpected scientific artifact change; do not publish")
    original = json.loads((backup / "original-package-hashes.json").read_text(encoding="utf-8"))
    protected = {name: sha for name, sha in after.items() if name not in mutable}
    if any(original.get(name) != sha for name, sha in protected.items()):
        raise ValueError("Scientific artifact differs from preserved original; do not publish")
    archives = {
        name: write_normalized_zip(root / relative, root / "artifacts" / name)
        for relative, name in PACKAGE_ARCHIVES.items()
    }
    write_json(root / "artifacts/RELEASE_CHECKSUMS.json", archives)
    receipt = {
        "changed_public_files": changed,
        "protected_files_unchanged": len(protected),
        "protected_sha256": protected,
        "archives": archives,
    }
    write_json(backup / "sanitization-validation.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    receipt = sanitize_release(args.root)
    print(json.dumps({key: value for key, value in receipt.items() if key != "protected_sha256"}, indent=2))


if __name__ == "__main__":
    main()

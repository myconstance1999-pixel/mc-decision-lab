#!/usr/bin/env python3
"""Run one visible stage at a time; heavyweight imports occur only on GPU stages.

Example: python training/run.py --model qwen3-1.7b --stage baseline
No code in this module installs dependencies or publishes artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

LLAMAFACTORY_COMMIT = "47d407d2aabbd52e11d4f34c7a06b9de3e2f0a77"
MODELS = ("qwen3-1.7b", "qwen2.5-7b")
DECISIONS = {"continue", "pause", "cancel", "replace_task", "clarify"}
SKILLS = {"follow_player", "return_home", "collect", "craft", "store_items"}
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_config(name):
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}")
    return read_json(CONFIG_DIR / f"{name}.json")


def record_id(record, index, split):
    return str(record.get("id", f"{split}-{index:04d}"))


def load_data(root):
    datasets, hashes = {}, {}
    for split in ("train", "val", "test"):
        path = root / "data" / f"{split}.json"
        rows = read_json(path)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{path} must contain a nonempty JSON array")
        ids = set()
        for index, row in enumerate(rows):
            for field in ("system", "instruction", "input", "output"):
                if not isinstance(row.get(field), str):
                    raise ValueError(f"{split}[{index}] lacks string {field}")
            answer = parse_prediction(row["output"])
            if answer is None:
                raise ValueError(f"Invalid target JSON in {split}[{index}]")
            rid = record_id(row, index, split)
            if rid in ids:
                raise ValueError(f"Duplicate ID in {split}: {rid}")
            ids.add(rid)
        datasets[split], hashes[split] = rows, sha256(path)
    if len(datasets["test"]) != 300:
        raise ValueError("The locked evaluation must contain exactly 300 test examples")
    return datasets, hashes


def parse_prediction(raw):
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != {"decision", "skill"}:
        return None
    if not isinstance(value["decision"], str) or value["decision"] not in DECISIONS:
        return None
    if value["skill"] is not None and (not isinstance(value["skill"], str) or value["skill"] not in SKILLS):
        return None
    return value


def prompt_messages(record, answer=""):
    user = record["instruction"]
    if record["input"]:
        user += "\n" + record["input"]
    return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]


def assert_mask(source_ids, target_ids, input_ids, labels):
    if not target_ids or input_ids != source_ids + target_ids:
        raise ValueError("Truncation or template drift detected")
    if labels != [-100] * len(source_ids) + target_ids:
        raise ValueError("Prompt/answer loss mask is incorrect")


def command_output(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=20).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {type(exc).__name__}"


def gpu_snapshot():
    return command_output([
        "nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw,driver_version",
        "--format=csv,noheader,nounits",
    ])


def environment():
    return {
        "utc": utc_now(), "python": sys.version, "platform": platform.platform(),
        "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
                                if d.metadata.get("Name"))),
        "gpu": gpu_snapshot(), "llamafactory_commit": LLAMAFACTORY_COMMIT,
    }


def verify_llamafactory():
    dist = importlib.metadata.distribution("llamafactory")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    actual = direct.get("vcs_info", {}).get("commit_id")
    if actual != LLAMAFACTORY_COMMIT:
        raise RuntimeError(f"LlamaFactory commit is {actual!r}; install the pinned Git URL from the notebook")


def get_tokenizer_template(name, revision):
    from transformers import AutoTokenizer
    from llamafactory.hparams.data_args import DataArguments
    from llamafactory.data.template import get_template_and_fix_tokenizer
    cfg = model_config(name)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], revision=revision, trust_remote_code=False)
    data_args = DataArguments(template=cfg["template"], enable_thinking=False,
                              train_on_prompt=False, mask_history=True, packing=False)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    return tokenizer, template, data_args


def preflight(root, max_length):
    """Download tokenizer files only, check both formats, pin both base revisions."""
    verify_llamafactory()
    from huggingface_hub import HfApi
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    datasets, hashes = load_data(root)
    destination = root / "runs" / "preflight.json"
    if destination.exists():
        previous = read_json(destination)
        if previous.get("passed") and (previous["data_sha256"] != hashes or previous["max_length"] != max_length):
            raise ValueError("Existing run has different data or cutoff; use a new project run folder")
        if previous.get("passed"):
            print("Reusing the successful both-tokenizer preflight", flush=True)
            return previous
    else:
        previous = None
    report = {"utc": utc_now(), "max_length": max_length, "data_sha256": hashes,
              "llamafactory_commit": LLAMAFACTORY_COMMIT, "passed": False, "models": {}}
    errors = []
    for name in MODELS:
        model_id = model_config(name)["model_name_or_path"]
        revision = (previous or {}).get("models", {}).get(name, {}).get("revision") or HfApi().model_info(model_id).sha
        tokenizer, template, data_args = get_tokenizer_template(name, revision)
        data_args.cutoff_len = max_length
        processor = SupervisedDatasetProcessor(template=template, tokenizer=tokenizer,
                                              processor=None, data_args=data_args)
        counts = {}
        for split, rows in datasets.items():
            lengths = []
            for index, row in enumerate(rows):
                messages = prompt_messages(row, row["output"])
                source, target = template.encode_multiturn(tokenizer, messages, system=row["system"])[0]
                length = len(source) + len(target)
                lengths.append(length)
                if length > max_length:
                    errors.append(f"{name}/{split}/{record_id(row,index,split)}: {length}>{max_length}")
                    continue
                actual_ids, labels = processor._encode_data_example(
                    prompt=messages[:1], response=messages[1:], system=row["system"],
                    tools=None, images=[], videos=[], audios=[])
                assert_mask(source, target, actual_ids, labels)
                decoded_target = tokenizer.decode(target, skip_special_tokens=True).strip()
                if decoded_target != row["output"]:
                    raise ValueError(f"Answer/EOS mismatch for {name}/{split}/{index}")
                if name == "qwen3-1.7b" and not tokenizer.decode(source).endswith("<think>\n\n</think>\n\n"):
                    raise ValueError("Qwen3 nonthinking prefix is missing from masked prompt")
            ordered = sorted(lengths)
            counts[split] = {"count": len(rows), "max_tokens": max(ordered),
                             "p95_tokens": ordered[min(len(ordered)-1, int(len(ordered)*.95))]}
        report["models"][name] = {"model_id": model_id, "revision": revision, "lengths": counts,
                                  "mask_check": "all in-range examples"}
    report["overflow_count"] = len(errors)
    report["overflow_examples"] = errors[:20]
    report["passed"] = not errors
    write_json(destination, report)
    if errors:
        write_json(destination.with_name(f"preflight.failed.{time.time_ns()}.json"), report)
        raise ValueError(f"{len(errors)} overlong examples; no weights loaded. " + "; ".join(errors[:5]))
    print(json.dumps(report, indent=2), flush=True)
    return report


def require_preflight(root, max_length):
    path = root / "runs" / "preflight.json"
    if not path.exists():
        return preflight(root, max_length)
    report = read_json(path)
    _, hashes = load_data(root)
    if not report.get("passed") or report["data_sha256"] != hashes or report["max_length"] != max_length:
        raise ValueError("Both-tokenizer preflight is missing, failed, or no longer matches the data")
    verify_llamafactory()
    return report


def ensure_budget(root, hard_deadline, max_gpu_hours):
    path = root / "runs" / "runtime_budget.json"
    if not path.exists():
        write_json(path, {"started_at": utc_now(), "hard_deadline_utc": hard_deadline,
                          "max_runtime_hours": max_gpu_hours,
                          "note": "Conservative elapsed runtime cap; compute units require Colab UI monitoring."})
    budget = read_json(path)
    start = datetime.fromisoformat(budget["started_at"]).timestamp()
    deadlines = [start + float(budget["max_runtime_hours"]) * 3600]
    if budget.get("hard_deadline_utc"):
        deadlines.append(datetime.fromisoformat(budget["hard_deadline_utc"].replace("Z", "+00:00")).timestamp())
    if hard_deadline:
        deadlines.append(datetime.fromisoformat(hard_deadline.replace("Z", "+00:00")).timestamp())
    remaining = min(deadlines) - time.time()
    if remaining <= 0:
        raise TimeoutError("The authorized runtime/deadline budget has expired")
    return remaining


def compute_dtype():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA Colab GPU is required; this runner never trains on the Mac/CPU")
    # Native BF16 is supported on Ampere and newer, not T4.
    bf16 = torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()
    return torch.bfloat16 if bf16 else torch.float16


def metrics(root, predictions):
    evaluator = root / "evaluation" / "evaluate.py"
    if not evaluator.is_file():
        raise FileNotFoundError("The shared evaluator evaluation/evaluate.py is required")
    output = predictions.with_name(predictions.stem + "_metrics.json")
    subprocess.run([sys.executable, str(evaluator), "--gold", str(root / "data/test.json"),
                    "--predictions", str(predictions), "--output", str(output)], check=True, cwd=root)


def read_completed_predictions(path, expected_ids):
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        records.append(json.loads(line))
    if [str(r["id"]) for r in records] != expected_ids[:len(records)] or len(records) > len(expected_ids):
        raise ValueError("Prediction IDs do not match the locked test order")
    return records


def evaluate(root, name, kind, args):
    report = require_preflight(root, args.max_length)
    rows = load_data(root)[0]["test"]
    revision = report["models"][name]["revision"]
    predictions = root / "results" / f"{name}_{kind}.jsonl"
    meta_path = predictions.with_suffix(".meta.json")
    adapter_dir = root / "runs" / name / "adapter"
    stamp = {"model": name, "kind": kind, "revision": revision,
             "test_sha256": report["data_sha256"]["test"], "max_new_tokens": 96,
             "do_sample": False, "enable_thinking": False}
    if kind == "adapter":
        if not (root / "runs" / name / "training_complete.json").exists():
            raise ValueError("Adapter evaluation requires completed training")
        stamp["adapter_sha256"] = sha256(adapter_dir / "adapter_model.safetensors")
    if meta_path.exists() and read_json(meta_path)["identity"] != stamp:
        raise ValueError("Existing predictions belong to a different model, data, or generation setup")
    if predictions.exists() and not meta_path.exists():
        raise ValueError("Unidentified existing predictions; preserve them and choose a fresh run folder")
    expected_ids = [record_id(r, i, "test") for i, r in enumerate(rows)]
    completed = read_completed_predictions(predictions, expected_ids)
    if len(completed) == len(rows):
        metrics(root, predictions)
        print(f"Reused all {len(rows)} completed {kind} predictions", flush=True)
        return
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    dtype = compute_dtype()
    tokenizer, template, _ = get_tokenizer_template(name, revision)
    write_json(meta_path, {"identity": stamp, "status": "running", "environment": environment()})
    ensure_budget(root, args.hard_deadline_utc, args.max_gpu_hours)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_config(name)["model_name_or_path"], revision=revision, trust_remote_code=False,
        torch_dtype=dtype, quantization_config=quantization, device_map={"": 0}, attn_implementation="sdpa")
    if kind == "adapter":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    predictions.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with predictions.open("a", encoding="utf-8") as handle:
        for index in range(len(completed), len(rows)):
            ensure_budget(root, args.hard_deadline_utc, args.max_gpu_hours)
            row = rows[index]
            prompt_ids, _ = template.encode_oneturn(tokenizer, prompt_messages(row), system=row["system"])
            inputs = torch.tensor([prompt_ids], device="cuda")
            torch.cuda.synchronize()
            sample_started = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                        max_new_tokens=96, do_sample=False, num_beams=1,
                                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                        use_cache=True)
            torch.cuda.synchronize()
            latency = time.perf_counter() - sample_started
            raw = tokenizer.decode(output[0, len(prompt_ids):], skip_special_tokens=True).strip()
            item = {"id": expected_ids[index], "raw_output": raw, "prediction": parse_prediction(raw),
                    "latency_seconds": latency, "generated_tokens": int(output.shape[-1] - len(prompt_ids))}
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            if (index + 1) % 10 == 0:
                print(f"{name} {kind}: {index+1}/{len(rows)}", flush=True)
    write_json(meta_path, {"identity": stamp, "status": "complete", "utc": utc_now(),
                          "elapsed_seconds_this_invocation": time.monotonic() - started,
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                          "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                          "environment": environment(), "count": len(rows)})
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    metrics(root, predictions)


def latest_checkpoint(adapter_dir):
    candidates = []
    for path in Path(adapter_dir).glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit() and (path / "trainer_state.json").is_file():
            candidates.append((int(suffix), path))
    if not candidates:
        return None
    path = max(candidates)[1]
    for required in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        if not (path / required).is_file():
            raise ValueError(f"Checkpoint cannot exactly resume: missing {path / required}")
    return path


def train(root, name, args):
    report = require_preflight(root, args.max_length)
    ensure_budget(root, args.hard_deadline_utc, args.max_gpu_hours)
    baseline_meta = root / "results" / f"{name}_base.meta.json"
    if not baseline_meta.exists() or read_json(baseline_meta).get("status") != "complete":
        raise ValueError("Run the locked base-model evaluation before training")
    dtype = compute_dtype()
    import torch
    run_dir = root / "runs" / name
    adapter_dir = run_dir / "adapter"
    completion = run_dir / "training_complete.json"
    if completion.exists():
        print(f"{name}: training already completed; preserving artifacts", flush=True)
        return
    cfg = read_json(CONFIG_DIR / "common.json")
    cfg.update(model_config(name))
    cfg.update(model_revision=report["models"][name]["revision"], cutoff_len=args.max_length,
               output_dir=str(adapter_dir), dataset_dir=str(run_dir / "dataset"),
               num_train_epochs=args.epochs, fp16=dtype == torch.float16, bf16=dtype == torch.bfloat16)
    dataset_info = {}
    for split in ("train", "val"):
        dataset_info[f"mc_{split}"] = {"file_name": str(root / "data" / f"{split}.json"),
                                     "columns": {"prompt": "instruction", "query": "input",
                                                 "response": "output", "system": "system"}}
    write_json(run_dir / "dataset" / "dataset_info.json", dataset_info)
    identity = {"config": cfg, "data_sha256": report["data_sha256"],
                "llamafactory_commit": LLAMAFACTORY_COMMIT}
    manifest_path = run_dir / "training_identity.json"
    if manifest_path.exists() and read_json(manifest_path) != identity:
        raise ValueError("Training settings/data changed since this run began; refusing to overwrite")
    write_json(manifest_path, identity)
    checkpoint = latest_checkpoint(adapter_dir)
    if checkpoint:
        cfg["resume_from_checkpoint"] = str(checkpoint)
    elif adapter_dir.exists() and any(adapter_dir.iterdir()):
        raise ValueError("Nonempty output has no restorable checkpoint; preserve it and use a fresh run folder")
    write_json(run_dir / "resolved_config.json", cfg)
    # The pinned LlamaFactory JSON CLI loader passes a Path to json.load.
    # Its YAML loader uses OmegaConf.load and accepts the same configuration.
    import yaml
    config_path = run_dir / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_json(run_dir / "environment_before.json", environment())
    run_logged(["llamafactory-cli", "train", str(config_path)], root, run_dir, args)
    for expected in ("adapter_model.safetensors", "adapter_config.json", "trainer_state.json"):
        if not (adapter_dir / expected).is_file():
            raise RuntimeError(f"Training returned without required artifact: {expected}")
    write_json(completion, {"utc": utc_now(), "status": "complete", "adapter_sha256": sha256(adapter_dir / "adapter_model.safetensors"),
                            "selection": "lowest validation loss at checkpoint evaluations; test unused for selection",
                            "environment": environment()})


def run_logged(command, root, run_dir, args):
    previous_receipt = run_dir / "last_train_process.json"
    if previous_receipt.exists() and read_json(previous_receipt).get("status") != "complete":
        archive = run_dir / f"train_process.failed.{time.time_ns()}.json"
        archive.write_bytes(previous_receipt.read_bytes())
    stop = threading.Event()
    def sample_gpu():
        with (run_dir / "gpu_telemetry.jsonl").open("a", encoding="utf-8") as handle:
            while not stop.is_set():
                handle.write(json.dumps({"utc": utc_now(), "gpu": gpu_snapshot()}) + "\n")
                handle.flush()
                stop.wait(30)
    monitor = threading.Thread(target=sample_gpu, daemon=True)
    monitor.start()
    process = None
    started = time.monotonic()
    status = "interrupted"
    try:
        with (run_dir / "train.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, start_new_session=True)
            def forward():
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
            forwarding = threading.Thread(target=forward, daemon=True)
            forwarding.start()
            while process.poll() is None:
                ensure_budget(root, args.hard_deadline_utc, args.max_gpu_hours)
                time.sleep(1)
            forwarding.join(timeout=10)
            status = "complete" if process.returncode == 0 else "failed"
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
        stop.set()
        monitor.join(timeout=25)
        if process is not None and process.stdout is not None:
            process.stdout.close()
        write_json(run_dir / "last_train_process.json", {"utc": utc_now(), "status": status,
                   "raw_exit_code": process.returncode if process else None,
                   "elapsed_seconds": time.monotonic() - started, "command": command})


def package(root, name):
    run_dir = root / "runs" / name
    if not (run_dir / "training_complete.json").exists():
        raise ValueError("No completed training to package; partial checkpoints remain under runs/")
    for kind in ("base", "adapter"):
        predictions = root / "results" / f"{name}_{kind}.jsonl"
        if (not predictions.exists() or not predictions.with_suffix(".meta.json").exists()
                or read_json(predictions.with_suffix(".meta.json")).get("status") != "complete"
                or not predictions.with_name(predictions.stem + "_metrics.json").exists()):
            raise ValueError(f"Missing completed {kind} predictions")
    archive = root / "artifacts" / f"{name}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    sources = list(run_dir.rglob("*")) + list((root / "results").glob(f"{name}_*"))
    sources += [root / "runs" / "preflight.json", root / "runs" / "runtime_budget.json"]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for path in sorted(set(sources)):
            if path.is_file() and not any(p.startswith("checkpoint-") for p in path.parts):
                target.write(path, path.relative_to(root))
    write_json(archive.with_suffix(".sha256.json"), {"file": archive.name, "sha256": sha256(archive),
               "note": "Adapter, logs, predictions, metadata. Base weights, caches and optimizer checkpoints excluded."})
    print(f"Artifact ready: {archive}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--stage", choices=("preflight", "baseline", "train", "adapter", "package", "all"), required=True)
    parser.add_argument("--max-length", type=int, choices=(512, 1024), default=512)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-gpu-hours", type=float, default=7.0)
    parser.add_argument("--hard-deadline-utc", default=None)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if not 0 < args.epochs <= 2 or not 0 < args.max_gpu_hours <= 7:
        parser.error("Epochs must be <= 2 and runtime hours <= 7")
    stages = ("preflight", "baseline", "train", "adapter", "package") if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"Starting {args.model}: {stage}", flush=True)
        if stage == "preflight":
            preflight(args.root, args.max_length)
        elif stage == "baseline":
            evaluate(args.root, args.model, "base", args)
        elif stage == "train":
            train(args.root, args.model, args)
        elif stage == "adapter":
            evaluate(args.root, args.model, "adapter", args)
        else:
            package(args.root, args.model)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate completed local evidence and prepare an HF folder; never upload or train."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_PRIVACY_SPEC = importlib.util.spec_from_file_location(
    "mc_release_privacy", Path(__file__).with_name("sanitize_packages.py"))
_PRIVACY = importlib.util.module_from_spec(_PRIVACY_SPEC)
_PRIVACY_SPEC.loader.exec_module(_PRIVACY)

MODELS = {"qwen3-1.7b": "Qwen/Qwen3-1.7B", "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct"}
PROJECT = "https://github.com/myconstance1999-pixel/mc-decision-lab"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model", "chat_template.jinja")
PUBLIC_CONFIG_KEYS = ("model_name_or_path", "model_revision", "template", "enable_thinking", "seed", "data_seed",
                      "lora_rank", "lora_alpha", "lora_dropout", "lora_target", "quantization_bit",
                      "quantization_method", "quantization_type", "double_quantization", "cutoff_len",
                      "learning_rate", "num_train_epochs", "per_device_train_batch_size",
                      "gradient_accumulation_steps", "lr_scheduler_type", "warmup_steps", "optim",
                      "fp16", "bf16", "load_best_model_at_end", "metric_for_best_model")
PRIVATE_PATH = re.compile(r"/(?:Users|home|content|tmp|root)/|[A-Za-z]:\\(?:Users|Temp)\\")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def evaluator(root):
    spec = importlib.util.spec_from_file_location("mc_release_evaluator", root / "evaluation/evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(root, name):
    run = root / "runs" / name
    adapter = run / "adapter"
    complete = read_json(run / "training_complete.json")
    check(complete.get("status") == "complete", "Training is not complete")
    adapter_hash = digest(adapter / "adapter_model.safetensors")
    check(adapter_hash == complete.get("adapter_sha256"), "Adapter hash differs from the completed training receipt")
    check((adapter / "adapter_model.safetensors").stat().st_size > 0, "Empty adapter file")
    preflight = read_json(root / "runs/preflight.json")
    check(preflight.get("passed") is True, "Preflight did not pass")
    selected = preflight["models"][name]
    revision = selected["revision"]
    check(selected["model_id"] == MODELS[name], "Wrong base model in preflight")
    check(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision), "Base revision is not a pinned commit")
    check(digest(root / "data/test.json") == preflight["data_sha256"]["test"], "Frozen test hash mismatch")
    identity = read_json(run / "training_identity.json")
    config = identity["config"]
    check(config.get("finetuning_type") == "lora" and config.get("quantization_method") == "bnb"
          and config.get("quantization_bit") == 4 and config.get("quantization_type") == "nf4",
          "This preparer accepts only the recorded BNB/NF4 PEFT run; package MLX adapters separately")
    check(identity["data_sha256"] == preflight["data_sha256"], "Training data differs from preflight")
    check(config["model_name_or_path"] == MODELS[name] and config["model_revision"] == revision,
          "Training base identity differs from preflight")
    expected_template = "qwen3" if name == "qwen3-1.7b" else "qwen"
    check(config["template"] == expected_template, "Unexpected training template")
    if name == "qwen3-1.7b":
        check(config.get("enable_thinking") is False, "Qwen3 training did not disable thinking")
    adapter_config = read_json(adapter / "adapter_config.json")
    check(adapter_config.get("peft_type") == "LORA", "Not a PEFT LoRA adapter configuration")
    check(adapter_config.get("base_model_name_or_path") == MODELS[name], "Adapter configuration names a different base")
    adapter_config["revision"] = revision
    shared = evaluator(root)
    gold = shared.load_gold(root / "data/test.json")
    check(len(gold) == 300, "Expected exactly 300 locked test examples")
    gold_ids = {row["id"] for row in gold}
    measured, prediction_files = {}, {}
    for kind in ("base", "adapter"):
        path = root / "results" / f"{name}_{kind}.jsonl"
        metadata = read_json(path.with_suffix(".meta.json"))
        check(metadata.get("status") == "complete" and metadata.get("count") == 300, f"{kind} evaluation is not complete")
        stamp = metadata["identity"]
        check(stamp.get("model") == name and stamp.get("kind") == kind and stamp.get("revision") == revision,
              f"{kind} evaluation model/revision mismatch")
        check(stamp.get("test_sha256") == preflight["data_sha256"]["test"], f"{kind} evaluation test mismatch")
        check(stamp.get("do_sample") is False and stamp.get("max_new_tokens") == 96
              and stamp.get("enable_thinking") is False, f"Unexpected {kind} decoding protocol")
        if kind == "adapter":
            check(stamp.get("adapter_sha256") == adapter_hash, "Evaluated adapter differs from packaged adapter")
        predictions = shared.load_predictions(path)
        check(len(predictions) == 300 and all(isinstance(r.get("id"), str) for r in predictions)
              and {r.get("id") for r in predictions} == gold_ids,
              f"{kind} must have exactly the 300 locked test IDs")
        check(all(isinstance(row.get("raw_output"), str) for row in predictions), f"Missing {kind} raw outputs")
        actual = shared.evaluate(gold, predictions)
        stored = read_json(path.with_name(path.stem + "_metrics.json"))
        check(all(stored.get(key) == value for key, value in actual.items()), f"Stored {kind} metrics do not match raw predictions")
        measured[kind], prediction_files[kind] = actual, path
    provenance = {
        "model": name, "base_model": MODELS[name], "base_revision": revision,
        "adapter_sha256": adapter_hash,
        "llamafactory_commit": preflight["llamafactory_commit"],
        "data_sha256": preflight["data_sha256"], "tokenizer_preflight": selected["lengths"],
        "training": {key: config[key] for key in PUBLIC_CONFIG_KEYS if key in config},
        "selection": complete.get("selection"),
        "decoding": {"do_sample": False, "max_new_tokens": 96, "enable_thinking": False},
        "packages": complete.get("environment", {}).get("packages", {}),
        "gpu_model": complete.get("environment", {}).get("gpu", "unrecorded").split(",")[0],
        "predictions_sha256": {kind: digest(path) for kind, path in prediction_files.items()},
        "evaluation_source_sha256": digest(root / "evaluation/evaluate.py"),
        "scope": "One seed; authored synthetic Chinese text-state decision task; offline evaluation only.",
    }
    provenance = _PRIVACY.sanitize_provenance(provenance)
    return adapter, adapter_config, gold, measured, prediction_files, provenance


def optional_diagnostics(root, name, stage, predictions):
    """Include only available, reproducible secondary evidence; primary release stays independent."""
    included = {}
    for kind in ("base", "adapter"):
        source = root / "results" / f"{name}_{kind}_diagnostics.json"
        if not source.is_file():
            continue
        target = stage / f"{kind}_diagnostics.json"
        try:
            subprocess.run([sys.executable, str(root / "evaluation/diagnostics.py"),
                            "--gold", str(root / "data/test.json"), "--predictions", str(predictions[kind]),
                            "--output", str(target)], check=True, capture_output=True, text=True)
            check(read_json(source) == read_json(target), f"Stored {kind} diagnostics do not match raw predictions")
            included[kind] = read_json(target)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            target.unlink(missing_ok=True)
            print(f"Optional {kind} diagnostics omitted: {type(exc).__name__}; primary release is unchanged.", file=sys.stderr)
    return included


def model_card(name, gold, metrics, provenance, diagnostics=None):
    base = MODELS[name]
    config = provenance["training"]
    rows = []
    for label, key in (("Strict decision accuracy", "decision_accuracy"), ("Joint decision + skill accuracy", "joint_accuracy"),
                       ("Schema validity", "strict_json_validity"), ("Replacement skill accuracy", "skill_accuracy")):
        rows.append(f"| {label} | {metrics['base'][key]:.2%} | {metrics['adapter'][key]:.2%} |")
    rows.append(f"| Decision macro-F1 | {metrics['base']['macro_f1']:.4f} | {metrics['adapter']['macro_f1']:.4f} |")
    for subset in ("controlled", "challenge"):
        left, right = metrics["base"]["subsets"][subset], metrics["adapter"]["subsets"][subset]
        rows.append(f"| {subset.title()} joint accuracy (n={right['n']}) | {left['joint_accuracy']:.2%} | {right['joint_accuracy']:.2%} |")
    secondary = "Secondary decision-only diagnostics were not included in this release."
    if diagnostics:
        diagnostic_rows = []
        for label, key in (("Raw JSON syntax validity", "raw_json_syntax_validity"),
                           ("Decision-only accuracy", "decision_accuracy"),
                           ("Extracted joint accuracy", "joint_accuracy")):
            values = [f"{diagnostics[kind][key]['value']:.2%}" if kind in diagnostics else "Not available"
                      for kind in ("base", "adapter")]
            diagnostic_rows.append(f"| {label} | {values[0]} | {values[1]} |")
        files = ", ".join(f"`{kind}_diagnostics.json`" for kind in diagnostics)
        secondary = ("### Secondary post hoc diagnostics\n\n"
                     "These exploratory diagnostics were added after observing schema failures. They do not replace "
                     "the frozen primary scores above. Decision-only accuracy scores the stated decision independently "
                     "of skill semantics and may extract exactly one unambiguous JSON object from surrounding text. "
                     "No null-skill inference, label repair, or gold-guided extraction is performed. Raw JSON syntax "
                     "validity checks syntax alone.\n\n| Diagnostic | Base model | Adapter |\n|---|---:|---:|\n"
                     + "\n".join(diagnostic_rows) + f"\n\nComplete diagnostic artifacts: {files}.")
    example = {key: gold[0][key] for key in ("system", "instruction", "input")}
    return f'''---
license: apache-2.0
library_name: peft
base_model: {base}
base_model_relation: adapter
pipeline_tag: text-generation
language:
  - zh
tags:
  - lora
  - qlora
  - minecraft
  - synthetic-data
  - intent-classification
---

# MC Decision Lab · {name}

This LoRA adapter maps a compact Minecraft-themed **text state and player instruction** to a task-control decision and optional replacement skill. It is an offline intent-classification prototype. It is **not a Minecraft player**, does not see the game, and cannot execute or verify actions. No research originality, real-player acceptance, or gameplay performance is claimed.

Base: [`{base}`](https://huggingface.co/{base}), revision `{provenance['base_revision']}`. Adapter SHA256: `{provenance['adapter_sha256']}`. This repository distributes adapter weights, not the base model. License: Apache-2.0.

## Output contract

Return exactly `{{"decision":"continue","skill":null}}`, with the decision selected from `continue`, `pause`, `cancel`, `replace_task`, or `clarify`. Only `replace_task` has a non-null skill: `follow_player`, `return_home`, `collect`, `craft`, or `store_items`. Object and quantity arguments are outside this schema.

## Measured offline evaluation

Both columns use the same frozen 300 examples, four-bit NF4 loading, greedy decoding, and a 96-token output cap. Qwen3 thinking is disabled. The primary strict decision accuracy counts an invalid output schema as incorrect even when the decision field itself matches: for example, a non-null skill on a nonreplacement decision invalidates the whole response. Schema validity requires valid JSON and the full output contract; it is not JSON syntax validity alone. Replacement skill accuracy includes every gold replacement case. These are measured saved results, recomputed from the included raw predictions during release preparation.

| Metric | Base model | Adapter |
|---|---:|---:|
{chr(10).join(rows)}

Full metrics, denominators, confusion matrices, cancellation errors, and latency summaries are in `base_metrics.json` and `adapter_metrics.json`. Latency is specific to the recorded run and is not a portable serving benchmark. Raw responses are in `base_predictions.jsonl` and `adapter_predictions.jsonl`.

{secondary}

## Data, training, and limits

The corpus has 1,000 training records, 160 validation records, and 300 locked test records. It is authored synthetic Chinese task data, not human annotation or Minecraft trajectories. The training records contain 250 utterance seeds in 25 correlated story families, each rendered in four state snapshots. Validation covers four families. The test combines 200 controlled variants and 100 separately agent-authored challenge cases. Family splits precede expansion; this does not eliminate every semantic similarity across splits.

This is a **single-seed experiment** (seed `{config.get('seed', 'unrecorded')}`). Training uses QLoRA rank `{config.get('lora_rank')}`, alpha `{config.get('lora_alpha')}`, dropout `{config.get('lora_dropout')}`, learning rate `{config.get('learning_rate')}`, and an epoch cap of `{config.get('num_train_epochs')}`. Selection uses validation loss, not the locked test. Training and inference use `{config.get('template')}`; Qwen3 uses `enable_thinking: false`. The actual sequence limit is `{config.get('cutoff_len')}` tokens, checked without truncation using both model tokenizers. Exact settings, source revision, package versions, and hashes are in `provenance.json`.

Some clarification labels implement a conservative authored policy rather than a uniquely correct interpretation of human intent. There is no independent human annotation, inter-annotator agreement, multi-seed stability study, Minecraft integration, or real-world acceptance test. Performance here measures this finite authored policy; it does not establish general player-intent understanding or reliable action execution. No positive adaptation effect is assumed: inspect both measured columns and challenge results.

See the included `DATASET_CARD.md` and `EVALUATION.md`, and the [project sources]({PROJECT}).

## Load and run

Use the project's [bounded training dependencies]({PROJECT}/blob/main/training/requirements-colab.txt). Run this example from the downloaded adapter repository directory. It uses a public input example without the gold answer, the pinned base revision, and the same nonthinking/greedy protocol. The example has not been separately benchmarked.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_id = {base!r}
revision = {provenance['base_revision']!r}
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
tokenizer = AutoTokenizer.from_pretrained(base_id, revision=revision)
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
base = AutoModelForCausalLM.from_pretrained(base_id, revision=revision,
    torch_dtype=dtype, quantization_config=quant, device_map={{"": 0}}, attn_implementation="sdpa")
model = PeftModel.from_pretrained(base, ".").eval()
record = {example!r}
messages = [{{"role": "system", "content": record["system"]}},
            {{"role": "user", "content": record["instruction"] + "\\n" + record["input"]}}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
    enable_thinking=False, return_tensors="pt").to("cuda")
with torch.inference_mode():
    output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
        max_new_tokens=96, do_sample=False, num_beams=1,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        pad_token_id=tokenizer.pad_token_id)
print(tokenizer.decode(output[0, inputs.shape[-1]:], skip_special_tokens=True))
```

For full experiment reproduction, use the [stage runner]({PROJECT}/blob/main/training/README.md). Public upload folders exclude base weights, caches, optimizer checkpoints, and private runtime paths.
'''


def prepare(root, name):
    adapter, adapter_config, gold, measured, predictions, provenance = validate(root, name)
    destination = root / "artifacts" / "hf" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        shutil.copyfile(adapter / "adapter_model.safetensors", stage / "adapter_model.safetensors")
        write_json(stage / "adapter_config.json", adapter_config)
        for filename in TOKENIZER_FILES:
            if (adapter / filename).is_file():
                shutil.copyfile(adapter / filename, stage / filename)
        for kind in ("base", "adapter"):
            shutil.copyfile(predictions[kind], stage / f"{kind}_predictions.jsonl")
            write_json(stage / f"{kind}_metrics.json", measured[kind])
        diagnostics = optional_diagnostics(root, name, stage, predictions)
        if diagnostics:
            provenance["secondary_diagnostics"] = {
                "source_sha256": digest(root / "evaluation/diagnostics.py"),
                "artifacts_sha256": {kind: digest(stage / f"{kind}_diagnostics.json") for kind in diagnostics},
                "role": "exploratory post hoc; frozen primary metrics unchanged",
            }
        write_json(stage / "provenance.json", provenance)
        shutil.copyfile(root / "data/DATASET_CARD.md", stage / "DATASET_CARD.md")
        shutil.copyfile(root / "evaluation/README.md", stage / "EVALUATION.md")
        (stage / "README.md").write_text(model_card(name, gold, measured, provenance, diagnostics), encoding="utf-8")
        for path in stage.iterdir():
            if path.name not in ("tokenizer.json", "vocab.json", "added_tokens.json") and path.suffix in (".json", ".jsonl", ".md", ".jinja", ".txt"):
                check(not PRIVATE_PATH.search(path.read_text(encoding="utf-8")),
                      f"Private runtime path in {path.name}; refuse publication rather than altering evidence")
        manifest = {path.name: digest(path) for path in sorted(stage.iterdir())}
        write_json(stage / "RELEASE_MANIFEST.json", manifest)
        if destination.exists():
            actual = {p.name: digest(p) for p in destination.iterdir() if p.is_file() and p.name != "RELEASE_MANIFEST.json"}
            check(actual == manifest and read_json(destination / "RELEASE_MANIFEST.json") == manifest,
                  "An existing upload directory differs; preserve it and select a fresh run folder")
        else:
            # Copy only the validated flat allowlist; no checkpoint or cache discovery.
            shutil.copytree(stage, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    args = parser.parse_args()
    destination = prepare(args.root.resolve(), args.model)
    print(f"Validated local upload directory: {destination}\nNo upload or publication was performed.")


if __name__ == "__main__":
    main()

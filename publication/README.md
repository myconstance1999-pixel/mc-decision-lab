# Prepare completed adapters for Hugging Face

From the project root, after restoring actual completed run and result artifacts:

```sh
python publication/prepare_release.py --model qwen2.5-7b
```

The standard-library script creates `artifacts/hf/<model>/` only after checking the completed-training receipt, adapter SHA256, pinned base revision, frozen test hash, decoding metadata, and exactly 300 unique prediction IDs for both base and adapter. It recomputes metrics with the shared evaluator and rejects stored metrics that disagree with raw predictions. It accepts poor or negative results; it never substitutes example scores.

The upload directory contains the adapter, configuration pinned to its base revision, tokenizer files if present, a model card with actual metrics, predictions, dataset/evaluation descriptions, sanitized provenance, and file checksums. It does not copy optimizer states, model caches, base weights, raw runtime configuration, or runtime-private paths. If copied evidence contains a private runtime path, preparation fails rather than rewriting the evidence. An existing identical directory is reusable; differing outputs are preserved and rejected.

Cards label primary decision accuracy as **Strict decision accuracy** and output compliance as **Schema validity**. An invalid schema makes the primary decision incorrect even if its decision field matches; JSON syntax validity alone is a separate diagnostic. Existing primary metric keys and values are unchanged.

When `results/<model>_{base,adapter}_diagnostics.json` is available, the preparer recomputes it with `evaluation/diagnostics.py` before including the artifact and a separately labeled post hoc diagnostic table. Missing or unreproducible secondary diagnostics are omitted without blocking primary publication. Diagnostics never replace the frozen primary scores.

Nothing is installed, downloaded, trained, authenticated, or uploaded by this script. It requires the existing project `evaluation/evaluate.py` and frozen `data/test.json`; these are read-only. The release is an Apache-2.0 adapter for a single-seed synthetic offline intent task, not a Minecraft player or a claim of original research.

The current 1.7B fallback uses MLX and is packaged by its execution worker. This preparer rejects MLX metadata; it does not convert MLX weights or claim PEFT compatibility. The `qwen3-1.7b` option is valid only for an independently completed run of the original BNB/NF4 PEFT recipe with the required evidence.

CPU fixture checks:

```sh
python -m unittest discover -s publication -p 'test_*.py' -v
```

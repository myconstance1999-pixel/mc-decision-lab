# Native MLX lane: Qwen3-1.7B

This lane completed the authorized 1.7B experiment on Apple M5 Pro with 24 GiB unified memory. The separate Colab lane owns Qwen2.5-7B. This is a native MLX adapter, not a PEFT adapter or a model-size-only controlled comparison.

## Execution and outcome

- Isolated environment: project `.venv-mlx`, Python 3.12.13; MLX-LM 0.32.2 and MLX 0.32.4. Exact installed dependency pins are in `requirements-lock.txt`. No global Python changes.
- Public official base: `Qwen/Qwen3-1.7B`, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Downloaded with `token=False`; no paid API or credential was used.
- Local conversion: affine 4-bit, group size 64, bfloat16 source. The base download, converted weights and environment remain local and ignored.
- Full run: two epochs, 2,000 single-record microsteps, accumulation 16, 125 optimizer updates. LoRA rank 16, alpha 32 represented by native scale 2, dropout 0.05, learning rate 1e-4. All 28 transformer layers' attention and MLP projections were adapted.
- Baseline completed before training. The selected native adapter was freshly reloaded for one 300-case post-training test. Raw process exit was 0.
- Training took 556.5 seconds within a three-hour execution cap. Native allocator peak was 2.429 GB, not total Mac memory. Wall-clock run times are omitted from public evidence.

| Metric | Base | Adapter |
|---|---:|---:|
| Primary strict decision / joint accuracy | 1.00% | 91.00% |
| Primary strict decision macro-F1 | 0.0190 | 0.9145 |
| Schema validity | 1.00% | 99.67% |
| Secondary raw JSON syntax validity | 100.00% | 100.00% |
| Secondary decision-only accuracy | 33.00% | 91.00% |

The base's low primary score came mainly from non-null skills on nonreplacement decisions. The actual applied prompt and token boundary were inspected; no decoding defect was found. Secondary diagnostics were added after observing that anomaly and are explicitly post hoc; raw outputs and frozen primary metrics were preserved. No test-driven prompt, hyperparameter, or dataset change was made.

Adapter joint accuracy was 100% for the 200 controlled variants and 73% for the 100 challenge cases. The corpus remains correlated, agent-authored synthetic data with an authored ambiguity convention. These measurements do not establish live Minecraft competence or research novelty.

The selected full-validation checkpoint was reported at microstep 1199, with validation loss 0.04824473. The installed trainer evaluates before loop step 1200 and reports `it - 1`; updates had occurred at 16 through 1184, so the selected weights contain 74 completed optimizer updates. The entire run completed all 125 updates; the selected earlier checkpoint was used for evaluation and release.

## Source and masking checks

Primary upstream sources:

- [Official MLX-LM 0.32.2 release](https://pypi.org/project/mlx-lm/0.32.2/)
- [Pinned MLX-LM source](https://github.com/ml-explore/mlx-lm/tree/93e6d4a6447b3080861985947f0820e76fcb957d)
- [Official pinned Qwen base](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e)

`run.py` uses official MLX conversion, loading, LoRA conversion and trainer APIs. It supplies a strict assistant-only loss mask: target positions are at least the prompt offset and strictly less than the original sequence length. This excludes the padded target that the installed default loss's inclusive upper bound would include. It also clips gradients before native AdamW updates.

The real tokenizer checked every record: prompt prefix equals full-template prefix, all target tokens are preserved, length stays under 512, and masking excludes all prompt and padded positions. Maximum complete sequence: 240 tokens. Training/evaluation use the same native chat template with `enable_thinking=False`.

Backend differences from Colab include affine quantization instead of NF4, native AdamW instead of paged 8-bit AdamW, native Hugging Face templates instead of LlamaFactory rendering, and the documented validation timing. The target includes terminal chat EOS/newline tokens. The optional `generated_tokens` prediction field counts retokenized decoded output and is not an exact decoder-emission counter or a speed benchmark.

## Files and reproduction boundary

- `run.py`: bounded run, enforcing a caller-selected elapsed duration and frozen data hashes; refuses to overwrite prediction files or an existing adapter.
- `finalize_release.py`: copies completed evidence, adds the model card, sanitizes only the public config copies, and recomputes the release manifest.
- `requirements.txt` and `requirements-lock.txt`: isolated package specification and exact environment.
- Local ignored `runs_mlx/`: original/downloaded weights, converted base, unmodified raw outputs, event log, intermediate/final/best native adapters, metrics, receipts, and public-ready `release-qwen3-1.7b-mlx/`.

The public release contains the exact native adapter, base identity, conversion metadata, both raw prediction files, primary and secondary metrics, family bootstrap, training events, dataset card/manifest, and a runnable official-API loading example. Public config paths are relative; original local configs remain unchanged. No upload is performed by this lane. Publication remains with the controller.

For a newly authorized experiment, `run.py --stage STAGE --max-runtime-hours 3` starts an elapsed duration cap for that invocation; a shorter positive cap may be supplied. The maximum remains three hours. Existing output protections still prevent accidental reuse of completed runs. Loading instructions in the released model card are independent of the training budget.

# MC Decision Lab

A small-model adaptation experiment for Minecraft task intentions. Given a current task, environment, recent dialogue, and a player's latest instruction, predict a structured `{decision, skill}` for an external executor.

**Status: both adapters are published.** Qwen3-1.7B training and evaluation are complete; its [native MLX adapter package](https://huggingface.co/oopbofficial/mc-decision-qwen3-1.7b-lora) is published. All 28 package files matched the frozen local release by Git blob or LFS hashes; see the [verification receipt](evidence/hf-qwen3-1.7b-verification.json). The [Qwen2.5-7B-Instruct PEFT adapter](https://huggingface.co/oopbofficial/mc-decision-qwen2.5-7b-lora) is published; all 20 package files matched public Git blob or LFS hashes in its [verification receipt](evidence/hf-qwen2.5-7b-verification.json). All data are synthetic, including separately agent-authored contextual challenges. Live Minecraft behavior has not been validated.

| Qwen3-1.7B · 300 held-out examples | Base | Adapter |
|---|---:|---:|
| Strict decision accuracy | 1.00% | 91.00% |
| Joint decision + skill accuracy | 1.00% | 91.00% |
| Schema validity | 1.00% | 99.67% |
| Decision-only accuracy (post hoc) | 33.00% | 91.00% |
| JSON syntax validity (post hoc) | 100.00% | 100.00% |

See the [1.7B behavior notes and negative examples](evidence/behavior-1.7b.md) for concrete remaining failures.

The strict gain combines schema adaptation and improved decision choice: all 300 baseline outputs were valid JSON, but 297 violated the null-skill rule for nonreplacement decisions. The selected adapter satisfies the full schema on 299/300 examples. Adapter joint accuracy is 100% on 200 controlled variants and 73% on 100 contextual challenges; the full result is their count-weighted 91%.

| Qwen2.5-7B-Instruct · 300 held-out examples | Base | Adapter |
|---|---:|---:|
| Strict decision accuracy | 60.33% | 91.00% |
| Joint decision + skill accuracy | 60.33% | 91.00% |
| Schema validity | 77.33% | 100.00% |
| Decision-only accuracy (post hoc) | 72.00% | 91.00% |
| JSON syntax validity (post hoc) | 88.67% | 100.00% |

The restored 7B run records joint accuracy of 65% → 98% on 200 controlled variants and 51% → 77% on 100 contextual challenges. Its [primary metrics](results/qwen2.5-7b_adapter_metrics.json) and [post hoc diagnostics](results/qwen2.5-7b_adapter_diagnostics.json) remain separate. The [published PEFT adapter](https://huggingface.co/oopbofficial/mc-decision-qwen2.5-7b-lora), [evaluation report](https://huggingface.co/oopbofficial/mc-decision-qwen2.5-7b-lora/blob/main/EVALUATION.md), and [raw adapter predictions](https://huggingface.co/oopbofficial/mc-decision-qwen2.5-7b-lora/resolve/main/adapter_predictions.jsonl) are available.

The [1.7B model card](https://huggingface.co/oopbofficial/mc-decision-qwen3-1.7b-lora/blob/main/README.md) documents the package; [raw adapter predictions](https://huggingface.co/oopbofficial/mc-decision-qwen3-1.7b-lora/resolve/main/qwen3-1.7b_adapter.jsonl) are available.

The 1.7B release is **native MLX-LM LoRA**, using MLX affine 4-bit quantization on Apple silicon; PEFT / Transformers compatibility is unverified. The 7B pipeline uses LlamaFactory / Transformers, NF4, and PEFT LoRA on Colab L4. Quantization, optimizer, chat templates, and validation timing differ, so this is not a controlled model-size comparison. Matching 91% aggregate adapter scores do not establish equivalent ability or superiority of the smaller model. Within the 1.7B run, baseline and adapter share the converted base, prompt, 300 test IDs, and decoding settings.

The fixed keyword reference scored 200/300 (66.7%) strict decision accuracy, 196/300 (65.3%) joint accuracy, and macro F1 0.6889201671810368. See its [evaluation artifact](evaluation/keyword_metrics.json).

## Use the project

- Training: [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb)
- Data: [`data/`](data/)
- Training and evaluation code: [`training/`](training/), [`training_mlx/`](training_mlx/), [`evaluation/`](evaluation/)
- Results and evidence: [`results/`](results/), [`evidence/`](evidence/)
- Public page: [`site/`](site/), a buildless static site intended for GitHub Pages.

The model chooses `continue`, `pause`, `cancel`, `replace_task`, or `clarify`. Only a replacement includes a skill: `follow_player`, `return_home`, `collect`, `craft`, or `store_items`. Game actions, cancellation, safety checks, and completion feedback remain responsibilities of the external executor.

## Preview the page

From the project root:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory site
```

Open `http://127.0.0.1:8765`. Serve over HTTP so the browser can read `site/results.json`. No build, package install, external font, or backend is required. Stop the preview with Ctrl-C.

Before publishing, inspect desktop (1440 px) and mobile (375 px), all four example tabs, keyboard arrow navigation, both result selectors, and external links. Screenshots should capture the hero, an example, and the results section. The examples are authored illustrations, not recorded model output.

## Publish measured results

`site/results.json` is the only result source for the page. Keep absent measurements `null`; never substitute zero. Schema version is `1`, with `models.qwen3_1_7b` and `models.qwen2_5_7b`:

- `status`: `pending`, `running`, `completed`, or `failed`.
- `baseline` / `adapter`: `decision_accuracy`, `joint_accuracy`, `macro_f1`, `json_validity`, and `skill_accuracy`; each a value in `[0, 1]` or `null`.
- `diagnostics` (optional): `baseline` / `adapter`, each with `decision_accuracy` and `json_syntax_validity` values in `[0, 1]` or `null`. These render in a separate exploratory table for the same selected model.
- `sample_count`: evaluated held-out sample count, or `null`.
- `adapter_url`, `report_url`, `predictions_url`: artifact links, or `null`. Download links appear only when supplied.
- `error`: a short public failure explanation, or `null`.
- The page uses a release label; public result data omit update timestamps.

The optional top-level `keyword_baseline` object stores the separately labeled deterministic reference results (`sample_count`, `decision_accuracy`, `decision_correct`, `joint_accuracy`, `joint_correct`, `macro_f1`, `source_url`). Preserve it when updating model results. These are rule-based measurements, not model outputs.

Metric keys remain fixed: `decision_accuracy` is displayed as **Strict decision accuracy**, with an invalid decision/skill schema counted as incorrect. `json_validity` is displayed as **Schema validity**: it measures the valid decision/skill contract, not syntactic JSON parsing alone. Schema compliance and intended-decision choice differ. The separate **Exploratory diagnostics** table shows decision choice ignoring skill-schema errors and JSON syntax validity. These post hoc diagnostics use the full 300-example denominator with no inferred labels or null repairs, and do not replace the primary metrics. Missing diagnostics display Pending.

The model table shows percentages and percentage-point changes. A failed run can retain measured baseline values; missing adapter measurements remain pending. Per-model status is independent. Malformed or unavailable result data produces a visible pending state.

## Source acknowledgments

The 7B training pipeline uses [LlamaFactory at `47d407d2aabbd52e11d4f34c7a06b9de3e2f0a77`](https://github.com/hiyouga/LlamaFactory/tree/47d407d2aabbd52e11d4f34c7a06b9de3e2f0a77). Agent context is based on [dsh-mc-agent at `f081451960ceebcd7649a3eddbfdccdeba8d4cf6`](https://github.com/blueziii/dsh-mc-agent/tree/f081451960ceebcd7649a3eddbfdccdeba8d4cf6). The completed 1.7B run uses MLX-LM 0.32.2 and MLX 0.32.4. See [inspected source locations](evidence/sources.md). Base models are Qwen3-1.7B and Qwen2.5-7B-Instruct. Adapters do not include base-model weights.

Project: [GitHub](https://github.com/myconstance1999-pixel/mc-decision-lab) · [Releases](https://github.com/myconstance1999-pixel/mc-decision-lab/releases)

This independent project is not affiliated with Mojang or Microsoft.

## Public evidence privacy

Public copies omit personal paths, exact wall-clock timestamps, local timezone information, account balances, and billing details. Training durations, hardware, frozen data, predictions, measured metrics, and upstream scientific revisions remain unchanged. Links to this project’s published packages follow the current public revision. Original evidence is retained privately; public verification receipts identify their verification scope.

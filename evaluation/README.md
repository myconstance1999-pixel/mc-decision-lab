# Evaluation protocol

The evaluator is standard-library-only and reads frozen Alpaca JSON gold plus prediction JSONL. It never changes data or trains a model.

```sh
python evaluation/evaluate.py --gold data/test.json --predictions results/model_base.jsonl --output results/model_base_metrics.json
```

Each prediction line uses an existing gold `id`, its exact `raw_output` string, and optional `latency` (seconds; `latency_seconds` also accepted). An optional parsed `prediction` object is ignored when raw output is present. Prediction-only input is supported for semantic metrics but cannot establish raw-format compliance: coverage and strict-format failures remain explicit. Never put the gold decision, family ID, or expected skill into the model prompt.

Unknown and duplicate prediction IDs are fatal. Missing predictions stay in every relevant denominator as invalid/incorrect. Output order is immaterial. The exact two-key JSON schema is strict: no code fences, trailing prose, duplicate or extra keys, unknown labels, wrong skill types, or non-null skills for nonreplacement decisions. `json_validity` aliases `strict_json_validity`; it measures valid JSON **and output-schema compliance**, not just syntax.

- Decision accuracy and macro-F1 require a schema-valid decision. Macro-F1 is the unweighted mean over all five fixed decisions; invalid outputs add false negatives.
- Joint accuracy requires both decision and skill to match.
- Skill accuracy uses **all gold replacement examples**, including invalid outputs and wrong decisions; its numerator/denominator are exported.
- False-cancel rate is predicted cancel on gold non-cancel / all gold non-cancel.
- Missed-cancel rate is any non-cancel or invalid/missing output on gold cancel / all gold cancel.
- Confusion matrices include an explicit invalid column. Latency summaries report available sample count and do not impute missing values.
- Both the full test and its controlled/challenge subsets are exported. The full score weights 200 controlled examples and 100 challenges.

Optional paired family bootstrap:

```sh
python evaluation/evaluate.py --gold data/test.json --predictions results/model_base.jsonl --compare results/model_adapter.jsonl --output results/model_comparison.json --bootstrap 2000 --seed 20261029
```

The bootstrap resamples whole families with replacement and includes all of each sampled family's cases. It reports comparison-minus-primary differences and percentile 95% intervals for decision accuracy, joint accuracy, and macro-F1. These intervals reflect uncertainty across this small set of authored families, not across human players or Minecraft worlds. Family sizes differ (40 controlled, 5 challenge); example-weighted statistics are recomputed for each resample. No significance or novelty threshold is implied.

## Fixed keyword baseline

```sh
python evaluation/keyword_baseline.py --gold data/test.json --output evaluation/keyword_predictions.jsonl
python evaluation/evaluate.py --gold data/test.json --predictions evaluation/keyword_predictions.jsonl --output evaluation/keyword_metrics.json
```

This deliberately simple rule uses only the latest utterance. Its precedence and vocabulary were written before its measurement and were not tuned against test failures. It cannot resolve context, quotations, negation reliably, or most implicit instructions. The stored run is a measured **rule baseline**, not a pretrained-model or fine-tuned-model result.

Observed frozen-test baseline: 300 predictions; decision accuracy 66.67%; joint accuracy 65.33%; decision macro-F1 0.6889; strict JSON/schema validity 100%; replacement skill accuracy 28/60 = 46.67%. False cancellation is 10/240 = 4.17%; missed cancellation is 15/60 = 25.00%. Joint accuracy is 74% on the 200 controlled variants and 48% on the 100 challenge records. No latency claim is made for this run. Raw predictions and complete denominator-bearing metrics are saved alongside this file.

Boundary checks:

```sh
python -m unittest discover -s tests -p 'test_data*.py' -v
```

## Secondary decision-only diagnostics

`diagnostics.py` was added after observing non-null skills on nonreplacement decisions in the 1.7B baseline. It is explicitly a post hoc diagnostic, not a change to the frozen primary evaluator. It reports raw JSON syntax separately from schema validity and scores the stated decision independently of skill correctness. Joint correctness still requires the stated skill to match; it never infers or repairs a null value.

```sh
python evaluation/diagnostics.py --gold data/test.json --predictions results/model_base.jsonl --output results/model_base_diagnostics.json
python evaluation/diagnostics.py --gold data/test.json --predictions results/model_adapter.jsonl --output results/model_adapter_diagnostics.json
```

Exactly one unambiguous JSON object may be extracted from surrounding prose or fences. Multiple objects, duplicate keys, malformed outer objects, and missing records remain failures. Both base and adapter must receive the same diagnostic. Interpret decision-only gains separately from strict schema-compliance gains.

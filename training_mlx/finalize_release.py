"""Package completed native MLX evidence; no inference, upload, or PEFT conversion."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs_mlx';DEST=RUN/'release-qwen3-1.7b-mlx'

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main():
    receipt=read(RUN/'training_receipt.json')
    if not receipt.get('completed'):raise RuntimeError('Incomplete training')
    base=read(RUN/'qwen3-1.7b_base_metrics.json');adapt=read(RUN/'qwen3-1.7b_adapter_metrics.json')
    bd=read(RUN/'qwen3-1.7b_base_diagnostics.json');ad=read(RUN/'qwen3-1.7b_adapter_diagnostics.json')
    for kind in ('base','adapter'):
        if not read(RUN/f'{kind}_receipt.json').get('completed'):raise RuntimeError('Incomplete evaluation')
        for suffix in ('jsonl','diagnostics.json'):
            filename=f'qwen3-1.7b_{kind}'+('.jsonl' if suffix=='jsonl' else '_diagnostics.json')
            shutil.copy2(RUN/filename,DEST/filename)
    for name in ('training_events.jsonl','generation_boundary_check.json'):
        shutil.copy2(RUN/name,DEST/name)
    shutil.copy2(ROOT/'data/DATASET_CARD.md',DEST/'DATASET_CARD.md')
    shutil.copy2(ROOT/'data/manifest.json',DEST/'data_manifest.json')
    example=read(ROOT/'data/train.json')[0]
    save(DEST/'example_input.json',{key:example[key] for key in ('system','instruction','input')})
    shutil.copy2(RUN/'official-base/LICENSE',DEST/'LICENSE')
    # Public configs retain load semantics without exposing local workspace paths.
    for filename in ('adapter_config.json','training_config.json'):
        config=read(DEST/filename);config['model']='mlx-qwen3-1.7b-4bit';save(DEST/filename,config)
    save(DEST/'release_transformations.json',{
        'preserved_originals':['runs_mlx/adapter/adapter_config.json','runs_mlx/training_config.json'],
        'public_config_change':{'field':'model','from':'local absolute workspace path (omitted for privacy)',
                               'to':'mlx-qwen3-1.7b-4bit'},
        'weights_changed':False,
        'adapter_copy_verified_sha256':sha(DEST/'adapters.safetensors'),
        'note':'Only public config copies were sanitized; original local run configurations and receipts remain intact.'})
    identity=read(RUN/'base_identity.json')
    files=['training_mlx/run.py','training_mlx/finalize_release.py','training_mlx/requirements-lock.txt',
           'evaluation/evaluate.py','evaluation/diagnostics.py','data/generate.py']
    save(DEST/'source_identity.json',{'repository':'https://github.com/myconstance1999-pixel/mc-decision-lab',
         'sha256':{name:sha(ROOT/name) for name in files}})
    save(DEST/'selection_details.json',{'validation_callback_micro_step':receipt['best_micro_step'],
         'completed_optimizer_updates_at_selection':receipt['best_micro_step']//16,
         'applied_training_examples_at_selection':(receipt['best_micro_step']//16)*16,
         'full_run_micro_steps':2000,'full_run_optimizer_updates':125,
         'source':'mlx-lm 0.32.2 mlx_lm/tuner/trainer.py',
         'source_order_evidence':'Loop iteration it=1200 evaluates before step(batch,...,it%16==0). The callback reports iteration=it-1=1199. Prior updates occurred at 16,32,...,1184 (74 updates); update75 at1200 occurs after validation. Thus15 pending microstep gradients do not enter the selected weights.',
         'note':'MLX validates before the indicated upcoming microstep; pending accumulated gradients are not part of saved weights. The selected checkpoint is earlier than the final two-epoch checkpoint.'})
    def pct(x):return f'{100*x:.2f}%'
    card=f'''---
language:
- zh
- en
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
base_model: Qwen/Qwen3-1.7B
base_model_relation: adapter
tags:
- mlx-lm
- lora
- synthetic-data
- task-control
---

# MC Decision Qwen3-1.7B — native MLX LoRA adapter

This is a **native MLX-LM adapter**, trained and evaluated on Apple M5 Pro / 24 GiB unified memory. It is not a PEFT adapter, merged full model, or verified Minecraft controller. The repository contains adapter weights and reproducible experiment evidence; it does not contain the base model weights. No PEFT/Hugging Face Transformers loading compatibility is claimed.

The narrow task maps a concise Minecraft-like task state and Chinese player utterance to `{{"decision": ..., "skill": ...}}`. Decisions are `continue`, `pause`, `cancel`, `replace_task`, and `clarify`; replacement skills are `follow_player`, `return_home`, `collect`, `craft`, and `store_items`. Nonreplacement decisions require a null skill.

## Base, training, and checkpoint identity

- Official base: [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), exact revision `{identity['revision']}`.
- Local conversion: MLX affine 4-bit quantization, group size 64, original floating weights in bfloat16. Quantization metadata plus its overhead averaged 4.501 bits/weight. This is **not bitsandbytes NF4**.
- Software: MLX-LM 0.32.2, MLX 0.32.4, Python 3.12.13; complete dependencies are in `requirements-lock.txt`.
- LoRA: rank 16, alpha 32 represented by native `scale=2`, dropout 0.05; all 28 transformer layers, attention q/k/v/o and MLP gate/up/down projections; {receipt['trainable_parameters']:,} trainable parameters.
- Training run: 1,000 records, two epochs, 2,000 microsteps, batch size 1, accumulation 16, 125 optimizer updates, seed 42. AdamW learning rate 1e-4, ten-update warmup then linear decay, gradient norm cap 1.0, gradient checkpointing.
- Selection: lowest full validation loss across 160 held-out records. Selected callback microstep {receipt['best_micro_step']}, corresponding to **{receipt['best_micro_step']//16} completed optimizer updates**; validation loss {receipt['best_validation_loss']:.8f}. The final two-epoch weights were retained locally; the released weights are the selected earlier checkpoint.
- Native allocator-reported peak memory: {receipt['peak_memory_bytes']/1e9:.3f} GB; this is not total Mac process/OS memory. Training took {receipt['seconds']:.1f} seconds. No general throughput benchmark is claimed.
- Selected adapter SHA256: `{receipt['adapter_sha256']}`.

The target mask includes only assistant completion tokens and terminal chat tokens. Prompt and padding positions are excluded explicitly. All 1,460 records passed actual-tokenizer prefix, complete-token preservation, and mask checks; the maximum complete sequence was 240 tokens under a 512-token cap. Non-thinking mode is fixed for training and evaluation. An explicit mask corrects the installed trainer's inclusive upper boundary that would otherwise include a padded target position.

The baseline used the same locally converted quantized base, prompt, 300 IDs, greedy decoding, and 96-new-token limit before any training. The selected adapter was freshly loaded and evaluated once on that same test set. Neither prompts nor hyperparameters were changed using test predictions.

## Frozen primary measurements

Primary strict decision accuracy requires the entire response to satisfy the output schema; a wrong non-null skill can therefore make its decision count incorrect even if the decision string itself is right. Schema validity here is stricter than JSON syntax validity.

| Metric | Base | Adapter |
|---|---:|---:|
| Strict decision accuracy | {pct(base['decision_accuracy'])} | {pct(adapt['decision_accuracy'])} |
| Joint decision + skill accuracy | {pct(base['joint_accuracy'])} | {pct(adapt['joint_accuracy'])} |
| Strict decision macro-F1 | {base['macro_f1']:.4f} | {adapt['macro_f1']:.4f} |
| Schema validity | {pct(base['strict_json_validity'])} | {pct(adapt['strict_json_validity'])} |
| Skill accuracy on all 60 gold replacements | {pct(base['skill_accuracy'])} | {pct(adapt['skill_accuracy'])} |
| False cancellation / 240 gold non-cancel | 0 / 240 | 0 / 240 |
| Missed cancellation / 60 gold cancel | 60 / 60 | 6 / 60 |

## Secondary post hoc diagnostics

A diagnostic was added after observing the baseline's non-null skills on nonreplacement decisions. It preserves raw outputs, extracts at most one unambiguous JSON object, and never repairs a label or infers a null value. The same diagnostic is applied to both models.

| Diagnostic | Base | Adapter |
|---|---:|---:|
| Raw JSON syntax validity | {pct(bd['raw_json_syntax_validity']['value'])} | {pct(ad['raw_json_syntax_validity']['value'])} |
| Decision-only accuracy independent of skill | {pct(bd['decision_accuracy']['value'])} | {pct(ad['decision_accuracy']['value'])} |
| Extracted joint accuracy without repairs | {pct(bd['joint_accuracy']['value'])} | {pct(ad['joint_accuracy']['value'])} |

The 1% to 91% primary increase combines schema adaptation and decision improvement. The separate 33% to 91% decision-only result avoids representing the entire strict increase as improved intent understanding. All 300 baseline outputs were JSON, but 297 violated the nonreplacement/null skill rule. All 300 adapter outputs were JSON; 299 satisfied the full schema.

## Generalization and limitations

The test contains 200 controlled snapshot variants from five held-out narrative families and 100 separately agent-authored contextual challenges from twenty additional families. Adapter joint accuracy is **100% on controlled variants** and **73% on challenges**; these are materially different difficulty subsets. The full score is 91%, weighted by their 200/100 record counts. The fixed keyword baseline scored 65.33% full joint accuracy and 48% challenge joint accuracy.

All data are controlled, agent-authored synthetic examples, not human gameplay traces. The 1,000 training records are four state variants of 250 authored utterance seeds across 25 families; they are correlated, not 1,000 independent situations. Validation contains only four families. Some clarification targets encode a deliberately conservative ambiguity policy and are not uniquely correct human intent labels. See `DATASET_CARD.md`.

The paired family bootstrap in `comparison.json` reports a primary joint-accuracy difference of +0.90 with a 95% percentile interval approximately [0.785, 0.955]. It reflects these authored families and the strict schema-sensitive metric, not human-player population uncertainty or proof of novelty.

The two-model project also uses a Colab/Transformers pipeline for a different model. Backend differences include MLX affine 4-bit versus NF4, native AdamW versus paged 8-bit AdamW, native Hugging Face chat rendering versus LlamaFactory templates, and validation timing. This is an application experiment; it is not a controlled model-size-only comparison, a live Minecraft acceptance test, or a new-method claim. No action execution, security enforcement, or runtime interruption safety is provided by this adapter.

## Loading this adapter

Use the pinned MLX-LM environment on compatible Apple silicon. Download this adapter repository first, then create the matching base conversion from the official pinned weights. No credential is needed for the public base. The following uses official Python APIs and does not upload anything:

```python
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from mlx_lm.convert import convert
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

original = snapshot_download(
    "Qwen/Qwen3-1.7B", revision="{identity['revision']}",
    token=False, local_dir="official-base",
    allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"],
)
convert(original, "mlx-qwen3-1.7b-4bit", quantize=True,
        q_bits=4, q_group_size=64, q_mode="affine", dtype="bfloat16",
        trust_remote_code=False)
model, tokenizer = load("mlx-qwen3-1.7b-4bit", adapter_path=".")
row = json.loads(Path("example_input.json").read_text())
messages = [
    {{"role": "system", "content": row["system"]}},
    {{"role": "user", "content": row["instruction"] + "\\n" + row["input"]}},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=96,
               sampler=make_sampler(temp=0), verbose=False))
```

[Source project and frozen data](https://github.com/myconstance1999-pixel/mc-decision-lab) contain `training_mlx/run.py`, the primary evaluator, secondary diagnostics, dataset generator/card, and tests. `source_identity.json` identifies the exact source file hashes used for this release. No third-party inference/serving integration has been verified.
'''
    (DEST/'README.md').write_text(card)
    save(DEST/'file_hashes.json',{p.name:sha(p) for p in DEST.iterdir() if p.is_file() and p.name!='file_hashes.json'})
    print(json.dumps({'release':str(DEST),'files':len(list(DEST.iterdir())),
                      'adapter_bytes':(DEST/'adapters.safetensors').stat().st_size,'adapter_sha256':sha(DEST/'adapters.safetensors')}))

if __name__=='__main__':main()

# MC Decision Lab synthetic corpus v1.0

This is an agent-authored, controlled synthetic instruction dataset for learning a narrow task-control output policy. It is not a human-collected dataset, Minecraft trajectory dataset, accepted real-game controller, or novel research contribution. The authoring agent did not inspect model predictions while writing or freezing this corpus. There are no external generation/API calls in the generator.

## Scope and schema

Each Alpaca record contains `system`, `instruction`, `input`, `output`, `id`, and `family`. The system and instruction are constant across every record; neither contains the example's answer. `input` is a compact JSON string with `current_task` (`status`, `goal`, `progress`), `environment`, `recent_dialogue`, and `latest_utterance`. Chinese utterances are authored task data. `output` is an exact two-key JSON object serialized as a string. IDs and families are metadata and must not be presented to the model.

| Decision | Meaning | Skill |
|---|---|---|
| `continue` | Keep the current task, or resume it when explicitly requested | `null` |
| `pause` | Temporarily suspend the task with its continuation retained | `null` |
| `cancel` | Abandon the current task without an immediate replacement | `null` |
| `replace_task` | Start a specified replacement task | One supported skill |
| `clarify` | The requested action, reference, condition, or supported target cannot be resolved | `null` |

Replacement skills: `follow_player`, `return_home`, `collect`, `craft`, `store_items`. Object/quantity arguments are intentionally outside the output schema. An explicit replacement takes precedence over the cancellation of the old task: “cancel mining and return home” is `replace_task` / `return_home`.

## Authored labeling rules

- Interpret the latest utterance in the supplied state and dialogue. Clear requests phrased as questions still count as instructions; descriptive questions about commands do not issue those commands.
- Quoted signs, tutorial examples, role-play lines, item names, reported chat, and tool text do not control the assistant unless the player explicitly adopts them as instructions.
- Negation and corrections change scope: “do not stop” continues; “not cancel, just wait” pauses; “not a pause, abandon it” cancels.
- Stopping speech, narration, or counting leaves the physical task running. An explicit request to stop both channels changes the task according to temporary/permanent intent.
- Temporary waiting is `pause`, not `cancel`. A permanent abandonment is `cancel`. Continuing an explicitly paused task is `continue`.
- Use known environment values for conditional instructions. Unknown values needed to choose an action require `clarify`.
- Use uniquely identified references in prior dialogue. Multiple unselected alternatives and unspecified new targets require `clarify`.
- Unsupported tasks (for example teleportation) require `clarify`; the taxonomy is deliberately limited.
- Canceling another agent, a future appointment, a narrative event, or a different project does not cancel the current task.

**Ambiguity caveat:** Some `clarify` examples deliberately use a conservative authored reference policy. In particular, an unspecified “that” may be labeled `clarify` even when a human could reasonably infer the sole current task. Several deliberately conflicting or underspecified instructions are artificial stress cases. These labels are policy targets, not proof of a uniquely correct human-intent interpretation. No independent human annotation or inter-annotator agreement has been collected. Results measure adherence to this authored policy; inspecting examples or reporting disagreements is necessary before generalizing to real players.

## Construction, splits, and effective diversity

| Split | Records | Narrative families | Authored utterance seeds / challenge cases | Per decision |
|---|---:|---:|---:|---:|
| Train | 1,000 | 25 | 250 utterance seeds | 200 |
| Validation | 160 | 4 | 40 utterance seeds | 32 |
| Test controlled subset | 200 | 5 | 50 utterance seeds | 40 |
| Test challenge subset | 100 | 20 | 100 separately authored cases | 20 |

The generator assigns narrative families to splits before creating variations. Every controlled family has two authored utterances per decision. Each is then rendered in four progress/time snapshots; these snapshots preserve the intended label. They are legitimate state variations, **not four independent semantic samples**. The 1,000 training records therefore contain only 250 distinct latest-utterance seeds and 25 correlated story families. Validation is especially small at four families.

The 100 challenge records were separately authored by the agent; they were not produced by the snapshot/paraphrase generator. They contain contextual minimal sets: for example, identical latest text yields five different decisions depending on the bell count and prior mapping. They remain agent-authored synthetic examples, not independent human ground truth. “Independently authored” distinguishes this construction from generated variants, not an independent annotator or a second author.

The family registry gives each family one split and identifies challenge origins. Assertions reject overlapping family identifiers, duplicate IDs, and exact duplicate rendered inputs across all splits. Narrative premises and references are manually grouped, while decision mechanisms such as negation and temporary stopping intentionally recur. These checks establish partition integrity and exact deduplication; they do not mathematically establish absence of all semantic similarity or a clean real-world distribution shift. The fixed schema and policy create strong common structure.

`manifest.json` records record counts, class balance, maximum input length, and SHA256 for each frozen split. Maximum input is 185 characters (JSON field names included); system, instruction, template and target tokens are additional. Only the actual model tokenizer can establish the 512-token training limit. The training runner must reject truncation rather than silently remove target information.

## Frozen test and reproduction

`test.json` was frozen before model training or model-output inspection. Its SHA256 is `dcce2afe36f5de36434b0210484d24b49b49754bff10ac0f5defc496d981d6b2`. Test examples must not be used to choose prompts, checkpoints, hyperparameters, or patches. Validation loss is available for checkpoint selection. If later corpus corrections are necessary, version a new dataset and disclose the change; do not silently replace this test set.

From the project root:

```sh
python data/generate.py
python -m unittest discover -s tests -p 'test_data*.py' -v
```

Generation is deterministic and uses only the Python standard library. It refuses to overwrite an existing split whose bytes differ. Tests verify frozen hashes, class counts, metadata separation, cross-split family/input rejection, and a genuine identical-utterance contextual minimal set. No tokenizer, model, GPU, installation, or paid service is invoked by these checks.

## Interpretation limits

A Chinese-capable pretrained model may already solve much of this task. The useful comparison is fixed-rule baseline, untrained base model, and adapted model using the same frozen inputs and decoding protocol. Report challenge and controlled subsets separately, strict output validity, decision versus joint correctness, cancellation errors, and replacement skill errors. A win on this compact authored task does not demonstrate open-ended Minecraft competence, safe action execution, robust player intent understanding, or research novelty. No Minecraft runtime integration or real-player acceptance is included.

# Qwen3-1.7B: observed behavior beyond the headline score

This is a retrospective analysis of the existing frozen 300-case test and saved predictions. No model was run, no output was repaired, and no data, prompt, checkpoint, or package was changed. The three cases were selected after evaluation to illustrate a decision correction, a remaining contextual error, and a keyword-baseline win. They are explanatory examples, not a representative random sample; the full-set counts below constrain their interpretation.

## What the complete saved results show

The primary evaluator requires a schema-valid output before crediting its decision. The base produced valid JSON syntax in all 300 cases, but 297 outputs incorrectly supplied a non-null skill for a nonreplacement decision. Consequently, its strict decision and joint accuracy were only 3/300. The adapter's strict decision and joint accuracy were 273/300; its schema validity was 299/300.

The secondary decision-only diagnostic separates that schema effect: correct decisions increased from **99/300 (33%) to 273/300 (91%)**. Exact paired decision counts, read from the same raw outputs, are:

| Base decision | Adapter decision | Cases |
|---|---|---:|
| Correct | Correct | 93 |
| Incorrect | Correct | 180 |
| Correct | Incorrect | 6 |
| Incorrect | Incorrect | 21 |

Thus 180 decisions were corrected, but six previously correct decisions regressed. This supports improvement on this authored test beyond merely replacing an invalid skill with null; it does not show uniformly improved behavior.

The adapter was jointly correct on all 200 controlled variants but only **73/100 challenges**. All 27 remaining errors were challenges. The saved confusion matrix has a specific pattern:

- **19/27 errors predict `clarify` when another decision is labeled correct:** five continue, three pause, five cancel, and six replacement cases.
- **7/27 errors predict `continue`:** one pause, one cancel, two replacement, and three clarification cases.
- **1/27 error emits the unsupported decision `stop`.** This is valid JSON syntax but invalid schema.
- Of 76 total `clarify` predictions, 57 match the gold decision and 19 are excess clarification. The evidence supports a tendency to ask for clarification too readily on some known contextual mappings; it does not establish an internal reasoning mechanism.

Challenge decision/joint correctness by gold class is continue 14/20, pause 16/20, cancel 14/20, replace_task 12/20, and clarify 17/20. All correctly selected replacement decisions also have the correct skill in this run; the replacement errors are mainly wrong decision choices, not wrong skills after a correct replacement choice.

The fixed keyword baseline was jointly correct on 196/300 cases, compared with 273/300 for the adapter. Their paired outcomes are 190 correct for both, 83 adapter-only successes, **six keyword-only successes**, and 21 errors for both. The third case below is one of those six keyword-only successes.

## Case 1 — a real decision correction, not only schema repair
**Exact test ID:** `test.challenge_approval_scope.pause`

- Current task: `整理花圃`; status `running`; progress `25%`.
- Environment: 花圃安全
- Recent dialogue: 玩家：我提议等下雨结束再继续。
- Latest utterance: **同意，就这么办。**

Gold policy target:

```json
{"decision":"pause","skill":null}
```

Base raw output:

```json
{
  "decision": "continue",
  "skill": "return_home"
}
```

Adapter raw output:

```json
{"decision":"pause","skill":null}
```

Fixed keyword raw output:

```json
{"decision":"continue","skill":null}
```

The player accepts the immediately preceding proposal to wait until the rain ends. Under the frozen rubric, this is a temporary suspension, so `pause` is the target. The base says `continue`; changing its skill to null alone would still leave its decision wrong. The adapter changes the decision to `pause`, while the latest-utterance-only keyword rule falls back to `continue`.

This is evidence of a useful contextual decision correction on this case. It is not proof that the adapter reliably resolves all acknowledgements: it gets four of the five records in this approval family right, but still fails the family's unresolved-choice clarification case.

## Case 2 — a known contextual mapping still fails

**Exact test ID:** `test.challenge_coded_bell.pause`

- Current task: `巡视菜园`; status `running`; progress `25%`.
- Environment: 铃响两次
- Recent dialogue: 玩家：一响继续，二响暂停，三响取消。
- Latest utterance: **按铃声约定处理。**

Gold policy target:

```json
{"decision":"pause","skill":null}
```

Base raw output:

```json
{
  "decision": "continue",
  "skill": "follow_player"
}
```

Adapter raw output:

```json
{"decision":"clarify","skill":null}
```

Fixed keyword raw output:

```json
{"decision":"continue","skill":null}
```

The supplied dialogue explicitly maps two bell rings to pausing, and the environment says the bell rang twice. No new meaning must be invented. The base ignores that mapping and continues; the adapter asks for clarification even though the relevant mapping is supplied. Both remain wrong under the authored rubric.

The failure is broader than this single selected example: the adapter outputs `clarify` for all five bell-family records. It therefore gets only the genuinely undefined five-ring case right, for 1/5 family accuracy. A successful undefined-signal clarification example from that family would conceal this weakness if reported on its own. This is a concrete limit on transferring a supplied symbolic cue-to-action rule.

## Case 3 — the fixed keyword rule beats the adapter

**Exact test ID:** `test.challenge_question_vs_request.replace_task`

- Current task: `烘干海带`; status `running`; progress `25%`.
- Environment: 燃料足够
- Recent dialogue: 玩家：现在需要木炭。
- Latest utterance: **能改为收集木头吗？**

Gold policy target:

```json
{"decision":"replace_task","skill":"collect"}
```

Base raw output:

```json
{
  "decision": "continue",
  "skill": "collect"
}
```

Adapter raw output:

```json
{"decision":"clarify","skill":null}
```

Fixed keyword raw output:

```json
{"decision":"replace_task","skill":"collect"}
```

The player asks to switch from drying kelp to collecting wood. The request is politely phrased as a question, but its replacement action and object are explicit. The rubric treats such a request as `replace_task` with skill `collect`. The keyword baseline recognizes the replacement/collection wording and matches both fields. The adapter returns `clarify`; the base remains on `continue` despite naming the collection skill.

This is an actual keyword-only success, not a hypothetical comparison. It does not make the keyword method generally superior: it wins only six paired cases, while the adapter uniquely wins 83. Here, however, the simple lexical rule handles an explicit instruction that the adapted model unnecessarily declines to resolve.

## Interpretation boundary

These results concern a five-decision output policy on controlled, agent-authored synthetic data. They do not establish safe execution, interrupt responsiveness, or player satisfaction inside Minecraft. The 200 controlled test records are correlated snapshot variants, and the conservative clarification convention is an authored target rather than independently adjudicated human intent. The challenge failures show that a strong aggregate score can coexist with systematic over-clarification and weak symbolic-rule use.

The secondary decision-only diagnostic was added after observing the baseline's schema failures. It is useful for explaining this result, but remains post hoc. The frozen primary metrics, raw outputs, and unsuccessful cases are retained; no inference or training was repeated for this note.

## Evidence identity

All case IDs, Chinese state text, target outputs, and raw outputs above were copied from the following existing files. Paths below identify source evidence in the project; ignored local raw outputs are also included by filename in the native MLX release package.

- `data/test.json` — SHA256 `dcce2afe36f5de36434b0210484d24b49b49754bff10ac0f5defc496d981d6b2`
- `runs_mlx/qwen3-1.7b_base.jsonl` — SHA256 `579d48e0cd7be56ddcf450f30e51c2221e37dedbf02560d1a46d35afa28cd7eb`
- `runs_mlx/qwen3-1.7b_adapter.jsonl` — SHA256 `dd9fe8f928cfe595df9648a88575313550efc2fa6513e2088749a09c029da4b5`
- `evaluation/keyword_predictions.jsonl` — SHA256 `4cf87ec65c3179f5468e7d9494f513662acd736b8b41307167620af72f96026d`

Confusion patterns and subset counts were checked against `runs_mlx/qwen3-1.7b_adapter_metrics.json` and `runs_mlx/qwen3-1.7b_base_diagnostics.json`; paired transitions were counted directly from matching frozen test IDs.

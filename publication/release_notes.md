Two trained adapters for a narrow Minecraft-style task-intention decision: continue, pause, cancel, replace a task, or ask for clarification. Both use the same frozen synthetic data split and are compared against their own unadapted base.

| Model / format | Strict decision accuracy, base → adapter | Decision-only diagnostic, base → adapter | Adapter challenge subset |
|---|---:|---:|---:|
| Qwen3-1.7B / native MLX | 1.00% → 91.00% | 33.00% → 91.00% | 73/100 |
| Qwen2.5-7B-Instruct / PEFT | 60.33% → 91.00% | 72.00% → 91.00% | 77/100 |

Each full test contains 300 cases. Strict scores require a valid decision/skill contract; decision-only diagnostics are explicitly post hoc. The 1.7B base produced valid JSON but usually violated the null-skill rule. Matching final scores do not establish equal model ability: models, quantizers, optimizers, templates, and hardware differ. These are single-seed results on correlated, agent-authored synthetic data, with no live Minecraft acceptance test.

- [Project homepage](https://myconstance1999-pixel.github.io/mc-decision-lab/)
- [1.7B MLX model card and weights](https://huggingface.co/oopbofficial/mc-decision-qwen3-1.7b-lora)
- [7B PEFT model card and weights](https://huggingface.co/oopbofficial/mc-decision-qwen2.5-7b-lora)
- [Concrete corrections and failures](https://github.com/myconstance1999-pixel/mc-decision-lab/blob/main/evidence/behavior-1.7b.md)

The release archives contain adapter weights, loading instructions, raw predictions, metrics, dataset documentation, and provenance. Base model weights are not included. The native MLX adapter requires compatible Apple silicon and MLX-LM; the PEFT example uses CUDA and bitsandbytes. Neither adapter is a complete game-playing or voice-companion system.

Privacy revision: exact experiment dates and unrelated environment inventory are omitted from public provenance. Archive entry timestamps are normalized packaging metadata, not experiment times. Model weights, raw predictions, dataset records, and measured scores are unchanged. Use the current release checksums for these repackaged downloads.

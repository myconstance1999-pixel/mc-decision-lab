# Implementation sources

These are source inspections, not independently reproduced upstream results.

- LlamaFactory: commit `47d407d2aabbd52e11d4f34c7a06b9de3e2f0a77`. Reuse its QLoRA training engine, masked SFT preprocessing, templates, and checkpoint support. Qwen3-1.7B uses `qwen3` plus `enable_thinking: false`; Qwen2.5-7B-Instruct uses `qwen`. The current upstream Qwen3 example targets a different variant and must not be copied unchanged.
- https://github.com/hiyouga/LlamaFactory/blob/47d407d2aabbd52e11d4f34c7a06b9de3e2f0a77/src/llamafactory/data/template.py
- https://github.com/hiyouga/LlamaFactory/blob/47d407d2aabbd52e11d4f34c7a06b9de3e2f0a77/src/llamafactory/data/processor/supervised.py
- dsh-mc-agent: commit `f081451960ceebcd7649a3eddbfdccdeba8d4cf6`. Its task ledger and executor separate decision, action receipts, and task-epoch checks. This motivates preserving task status and uncertain execution outcomes in the input contract. No upstream game code is executed or bundled here.
- https://github.com/blueziii/dsh-mc-agent/blob/f081451960ceebcd7649a3eddbfdccdeba8d4cf6/src/task-exec.mjs
- https://github.com/blueziii/dsh-mc-agent/blob/f081451960ceebcd7649a3eddbfdccdeba8d4cf6/tools/check-task-epoch.mjs
- https://huggingface.co/Qwen/Qwen3-1.7B
- https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

This project is an application experiment, not a claim of a novel planning algorithm. Offline synthetic decision accuracy is not Minecraft gameplay success, cancellation correctness, memory quality, or companion quality.

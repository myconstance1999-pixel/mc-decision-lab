# Training entry points

GPU execution is unrun in the checked-in notebook. The scripts execute only when invoked; importing `training.run` does not download models, install packages, or start compute.

Use the Colab notebook in `notebooks/train_colab.ipynb`, with a single GPU. It fetches the public project and installs the inspected LlamaFactory commit `47d407d2aabbd52e11d4f34c7a06b9de3e2f0a77` inside the temporary Colab runtime. The direct dependency file is bounded by that commit; the complete resolved environment is recorded for each run. Colab's existing matching CUDA PyTorch packages are preserved.

Each stage is separately executable from the project root:

```sh
python training/run.py --model qwen3-1.7b --stage preflight
python training/run.py --model qwen3-1.7b --stage baseline
python training/run.py --model qwen3-1.7b --stage train
python training/run.py --model qwen3-1.7b --stage adapter
python training/run.py --model qwen3-1.7b --stage package
```

Repeat with `--model qwen2.5-7b` only after completing and downloading the first model. `--stage all` invokes those stages sequentially. The task notebook also supplies the authorized UTC deadline. `--max-gpu-hours` defaults to a conservative seven-hour elapsed-runtime limit; it cannot measure Colab compute-unit consumption. Monitor the 40-CU task limit in Colab itself. No new quota, GPU, or account is created by the runner.

The preflight downloads tokenizer metadata for both exact base revisions, validates all split lengths and supervised token masks, and refuses truncation. A failed length-only preflight may be rerun after fixing the data or selecting `--max-length 1024`; the failure receipt is retained. Once preflight passes, data and cutoff are immutable for the run. A successful 512-token preflight is required before using 512-token training. Input is an Alpaca JSON array at `data/train.json`, `data/val.json`, and `data/test.json`. The locked test has 300 records. No history messages or reasoning traces are needed.

Qwen3 uses `template: qwen3` and `enable_thinking: false`, including its masked empty-think prefix. Qwen2.5 uses `template: qwen`. Training and inference use the same pinned LlamaFactory templates. Both models use NF4 QLoRA, rank 16, alpha 32, dropout .05, all linear targets, learning rate .0001, effective batch 16, and at most two epochs. T4 uses FP16; L4/A100 uses BF16. Base and adapter evaluation use the same four-bit loading, greedy decoding, and 96-token output cap.

Outputs:

- `results/<model>_{base,adapter}.jsonl`: raw response, parsed response, stable ID, generation latency in seconds, and token count. Invalid outputs remain in the denominator.
- `results/<model>_{base,adapter}_metrics.json`: the shared evaluator's metrics.
- `runs/preflight.json`: exact model revisions, dataset hashes, length and loss-mask checks.
- `runs/<model>/`: configuration, dependency versions, GPU clocks/memory samples, original process exit status, logs, and adapter.
- `artifacts/<model>.zip`: adapter, logs, predictions, and metadata. Base weights, caches, and optimizer checkpoints are excluded.

An interrupted training run resumes the most recent complete `checkpoint-*` with optimizer, scheduler, and RNG state. Configuration and data changes are rejected. An interrupted evaluation resumes only a matching ordered test prefix. Do not delete the existing runtime outputs or use `overwrite_output_dir`. Runtime-local checkpoints disappear if Colab deletes the runtime; download the optional resume ZIP before ending the session. A publication adapter ZIP is not a full training-resume archive.

The notebook's normal `files.download` cells download artifacts without Drive mounting or new authentication. Publication and homepage deployment are outside this runner.

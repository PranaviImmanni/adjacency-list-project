# MLP Follow-up Experiments: Statistical Pruning, Recovery, Protected Output Layer, LoRA, INT8

These five experiments extend the MLP weight-pruning study (`mlp_weight_pruning_plan.md`) with the professor's
statistical pruning rule and four ways to recover or replace accuracy. The original sweep's results
(`results/mlp/cuda1`) are reused as baselines and are not rerun: the original dense model, and global
magnitude pruning without fine-tuning at 50-99%.

| # | Experiment | Question |
|---|---|---|
| 1 | Statistical pruning: keep weights outside mean ± k·std | Does the professor's rule preserve accuracy better than magnitude pruning at the same retained count? |
| 2 | Pruning + fine-tuning, masks fixed | Can accuracy recover without increasing storage? |
| 3 | Protect the output layer: fc3 dense, fc1 + fc2 pruned to the same stored bytes | Does preserving the classifier improve accuracy at comparable storage? |
| 4 | Pruning + LoRA: frozen pruned weights, rank-4 adapters | Is the accuracy recovery worth the added memory and latency? |
| 5 | INT8 without pruning | Can fewer bits give a better trade-off than removing connections? |

Code: `src/mlp/variants.py` (what each model is), `src/mlp/recovery.py` (fine-tuning, LoRA),
`src/mlp/quantization.py` (INT8), `src/mlp/weight_pruning.py` (statistical rule, exact-count magnitude),
`src/mlp/followup_reporting.py`. Settings: `configs/mlp_followup.json`, which inherits every setting of the
original experiment (split, preprocessing, batch sizes, timing, FP32, TF32 off) and may not override any of them.

## Fixed controls (all experiments)

- The same three trained checkpoints (`checkpoints/fashion_mlp_seed{42,43,44}_cuda1.pt`), data split and
  standardization. The inputs are rebuilt with the checkpoint's saved normalization constants, so they are
  bit-identical to the training run on any machine.
- Selection uses validation data only; test accuracy is reported for the selected configuration.
- Measured for every deployed model at batch 1, 32 and 256: accuracy, stored weight bytes, resident GPU
  memory, peak GPU memory and latency (20 warm-ups, 50 timed passes, 3 repetitions, one process per run).
- Every deployed format is checked against a dense implementation of the same modified model, over all
  15,000 evaluation images and again on each benchmark batch (GCN tolerance `allclose(atol=1e-5, rtol=1e-4)`,
  prediction agreement, margin condition).
- Seed 42 first; seeds 43 and 44 after its results are checked.
- Scope is bounded: no cross-products (no statistical threshold x fine-tuning x LoRA x INT8).

## 1. Statistical pruning (run first, on its own)

- For each weight matrix separately (fc1, fc2, fc3; biases excluded): mean and population standard deviation
  (ddof = 0) of the original weights, in float64. Keep w when |w − mean| > k · std (strict).
- k = 0.5, 1, 1.5, 2. Retained values and signs are unchanged (not binary edges).
- Recorded per variant: mean, std, threshold, retained count and actual sparsity per layer and overall.
- **Matched baseline:** for each k and seed, global magnitude pruning that keeps exactly the same total number
  of weights (`prune_to_count`). Same retained count gives the same CSR bytes, so the comparison isolates the
  selection rule from the amount removed.
- No fine-tuning. Both stored as CSR.

In the CPU pilot (seed 42), the rule pruned every layer at nearly the same rate (fc3 included), while global
magnitude prunes the small output layer much less. At the same count, global magnitude scored higher at
every k.

## 2. Pruning + fine-tuning

- Start from the original sweep's magnitude 80% and 90% masks (the evaluator checks the mask hashes match).
- Train every weight and bias with the masks fixed. Removed weights get zero gradients before each step (so
  Adam's moments for them stay zero) and are re-masked after it, so they stay exactly zero and the CSR bytes
  do not change.
- **Budget (shared with experiments 3 and 4):** Adam, batch 128, 10 epochs for each learning rate in
  {1e-3, 3e-4, 1e-4}. The batch order is seeded with the model seed.
- **Selection:** the (learning rate, epoch ≥ 1) with the best validation accuracy, ties going to the lower
  validation loss. Epoch 0 (pruning only) is recorded but cannot be selected.
- Compared with the pruning-only models from the original sweep.

## 3. Protect the output layer

- fc3 is unpruned and stored dense. fc1 and fc2 are pruned by global magnitude, with the retained count
  chosen so the stored weight bytes (CSR fc1 + fc2, plus dense fc3) match experiment 2's all-CSR model at
  80% and 90%.
- CSR costs 12 B per retained weight plus 8 B per row pointer, so the match is exact: 0 B mismatch for
  every seed. The mismatch is still computed and reported.
- Evaluated before and after the same fine-tuning procedure as experiment 2.

## 4. Pruning + LoRA

- Start from the same magnitude 80% / 90% models. Pruned weights and all biases are frozen.
- Rank-4 adapters on fc1, fc2 and fc3: W x + (alpha / rank) B A x, with alpha = 4, so the scale is 1.
  A is initialized like `nn.Linear` (seeded); B starts at zero, so training starts exactly from the pruned
  model.
- Same budget and selection as experiment 2.
- At inference the adapters stay separate, as a low-rank product next to the CSR weights; they are never
  merged into W. Adapter bytes: 6,376 parameters = 25,504 B, reported next to the weight bytes.

## 5. INT8 without pruning

Two real INT8 paths on the GPU, both from the original FP32 model:

| Format | Stored | Compute |
|---|---|---|
| INT8 weight-only | int8 weights + one FP32 scale per output row (symmetric, max abs / 127) | dequantize on the device each pass, then an FP32 matmul |
| INT8 W8A8 | the same int8 weights, zero-padded to multiples of 8 (300 → 304, 100 → 104, 10 → 16) | activations quantized per sample at run time, int8 × int8 → int32 with `torch._int_mm` (cuBLASLt), rescaled to FP32 |

- Stored bytes include the scales (quantization metadata) and the padding.
- **GPU support check:** before running, the W8A8 path probes `torch._int_mm` on the device. It finds the
  smallest batch the device accepts (smaller batches are zero-padded) and checks the result against an
  exact int64 product. If the kernel is unsupported, the run fails; there is no CPU or simulated fallback.
- **References for the equivalence check:** an FP32 dense model holding the dequantized weights
  (weight-only), and the same activation quantization with an exact float64 integer matmul (W8A8).

## Running on the lab GPU

```bash
cd /data/adjli/adjacency-list-project && source .venv/bin/activate
git fetch origin && git checkout mlp-followup-experiments
pytest -q

# Experiment 1, seed 42; check results/mlp_followup/cuda1/evaluate, then seeds 43 and 44
python scripts/run_mlp_followup.py --experiments statistical --stage all --seeds 42 --device cuda:1
python scripts/run_mlp_followup.py --experiments statistical --stage all --seeds 43 44 --device cuda:1

# Experiments 2-5, seed 42 first, then 43 and 44
python scripts/run_mlp_followup.py --experiments finetune protect_fc3 lora int8 --stage all --seeds 42 --device cuda:1
python scripts/run_mlp_followup.py --experiments finetune protect_fc3 lora int8 --stage all --seeds 43 44 --device cuda:1
```

- Stages: `variants` makes the models (and trains on the GPU), `evaluate` measures accuracy and equivalence,
  `benchmark` measures timing and memory, and `all` runs the three in order.
- Finished work is skipped, so rerunning a command resumes it.
- Models go to `checkpoints/variants/cuda1/` (ignored by git); results go to `results/mlp_followup/cuda1/`.
- Report: `python scripts/summarize_mlp_followup.py --run cuda1` writes `results/mlp_followup/summary_cuda1/`.
  It contains the combined table, per-experiment tables, accuracy-vs-memory plots and latency by model.

Debug runs on a CPU (`--device cpu --checkpoint-run cuda1`) go to `results/mlp_followup/cpu/`, which git
ignores. Their timing is not a GPU result.

## Choices made here (not fixed by the request)

| Choice | Setting | Why |
|---|---|---|
| Standard deviation | population (ddof = 0), per matrix, float64 | the statistical definition of a matrix's spread; the difference from ddof = 1 is negligible at these sizes |
| Layers under the statistical rule | fc1, fc2, fc3 | same pruned layers as the original sweep (all three) |
| Recovery learning rates | {1e-3, 3e-4, 1e-4}, chosen on validation | one fixed rate could favor one method; the grid is the same for fine-tuning and LoRA |
| Fine-tuning trains | weights and biases | standard; removed weights stay exactly zero |
| LoRA layers, rank, alpha | fc1-fc3, rank 4, alpha 4 | rank 4 as specified; alpha = rank gives scale 1, and the learning-rate grid covers scaling |
| INT8 scheme | symmetric, per output row; W8A8 with per-sample activations | the common GPU INT8 scheme; per-row scales keep the error small |

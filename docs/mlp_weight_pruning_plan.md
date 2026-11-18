# MLP Weight Pruning: Dense vs Sparse Weight Storage

This is Minh's part of the project plan ("Sparse Storage for Graph and
Neural-Network Weight Matrices", Nov 12, 2026): the trained MLP, pruning
integration, storage-format inference and the accuracy/epsilon evaluation. It
extends the GCN study (`gcn_inference_plan.md`, `pruning_and_error_analysis.md`)
from graph adjacency to neural-network **weight matrices**. The GCN experiments
are finished and are not rerun.

| Plan item | Owner | Where |
|---|---|---|
| Dense MLP baseline, 3 seeds | Minh | sections 1, 6 |
| Pruning heuristics applied at each sparsity | Minh (interface), Pranavi (signed-weight implementation) | section 2 |
| Dense+mask, original dense, pruned dense, COO, CSR models | Minh | section 3 |
| Python adjacency list, bipartite conversion, byte model vs measured bytes | Pranavi | measured per-layer bytes in every result (section 3) |
| Prediction agreement, logit differences, margin condition | Minh | section 4 |
| Timing/energy harness (NVML), synthetic layer sweep (1,024 / 4,096 / 8,192) | Dai | entry points in section 5, `runs.csv` in section 7 |

**Question.** When a trained MLP's weights are pruned, how do accuracy, weight
storage, peak GPU memory and latency change with sparsity, and do dense, dense+mask,
COO and CSR storage of the *same* pruned weights give the same predictions?

Code: `src/mlp/`. Config: `configs/mlp_fashion_mnist.json`. Every setting below
is fixed in that file before any result is looked at.

## 1. Fixed configuration

| | Setting |
|---|---|
| Dataset | Fashion-MNIST: 60,000 train / 10,000 test images, 28 × 28 grayscale, 10 classes |
| Download | the four official IDX files, checked against their MD5s (no torchvision) |
| Split | 55,000 train / 5,000 validation, chosen once with split seed 0 (independent of model seeds); the official test set is only used for reporting |
| Preprocessing | pixels / 255, then standardized with the training split's mean and std (float64), flattened to 784 |
| Architecture | MLP 784 → 300 → 100 → 10, ReLU, biases, no dropout (LeNet-300-100, the standard fully connected pruning benchmark). Layers `fc1`, `fc2`, `fc3`. |
| Training | Adam, lr 1e-3, no weight decay, batch 128, 30 epochs, cross-entropy |
| Selection | best validation accuracy; ties go to the lower validation loss |
| Model seeds | 42, 43, 44 (initialization and batch order) |
| Precision | FP32; on CUDA, TF32 off (`fp32_precision = "ieee"`) |
| GPU | RTX 3090, `cuda:1`, same measurement settings as the GCN runs |

Each checkpoint (`checkpoints/fashion_mlp_seed<S>_<device>.pt`) stores the weights,
architecture, seed, best epoch, validation and test accuracy, the training
history, the configuration, and the split seed, validation-index hash and
normalization constants. Evaluation refuses to run if the data split or
normalization differs from the checkpoint's.

## 2. Pruning interface (shared with Pranavi)

```python
prune_layers(weights, method, sparsity, pruning_seed=None) -> {"masks": {layer: keep-mask}, "summary": {...}}
```

`weights` maps the pruned layers' names to their weight matrices (outputs ×
inputs), in model order. A mask is a CPU bool tensor, True = keep. Pruning only
chooses positions. It never changes a value: retained weights keep their exact
values and signs, removed ones become +0.0, and biases are never pruned.
`src/mlp/weight_pruning.py` is a reference implementation;
`tests/test_weight_pruning.py` is the contract any implementation must pass.

| Method | Rule | Ties |
|---|---|---|
| magnitude (global) | one threshold over **all pruned layers together**: remove the floor(s · N) smallest \|w\| of all N pruned weights, so each layer gets its own sparsity | layer order, then row-major |
| topk_per_neuron | every output neuron (row) of every layer removes its floor(s · in) smallest \|w\|, so every row keeps the same K | column order |
| random (global) | remove the first floor(s · N) positions of `torch.randperm(N)` over all pruned weights, seeded with the pruning seed | n/a |

- **Global percentile** (in the project plan): at a fixed target sparsity, the
  global percentile cutoff selects exactly the global magnitude set, so it is
  the same condition, not a separate one.
- Sparsities: 50, 80, 90, 95, 99%. Counts use exact decimals (`Fraction(str(s))`).
- Removed sets are nested: whatever is removed at 80% is also removed at 90%.
- Random pruning seeds: 101, 102, 103. They choose weights only; they are not model seeds.
- Top-K per neuron rounds per row, so its actual sparsity is slightly below the
  target (e.g. 79.98% at 80%). Every result records requested and actual
  sparsity per layer and overall.
- No fine-tuning after pruning (first round; the plan's optional fine-tuning is
  not included).

**Global magnitude spreads sparsity unevenly** (seed 42, actual sparsity of
fc1 / fc2 / fc3): 50% → 50.7 / 45.2 / 25.6%; 90% → 90.4 / 87.3 / 71.0%;
99% → 99.1 / 98.7 / 90.5%. The small output layer, which has the largest
weights, keeps the most.

**Which layers are pruned: all three (decided).** `fc1`, `fc2` and `fc3` are
all pruned (`pruned_layers` in the config). Reasons:
- every weight matrix is then stored in the format being compared, so no
  "COO" or "CSR" model carries a dense layer, and the storage, memory, latency
  and equivalence numbers describe the whole network in one format;
- the same rule applies to every method, with no special case for the output layer;
- global magnitude already prunes the small output layer less on its own
  (about 70% at a 90% target);
- classic LeNet-300-100 pruning also prunes all three layers.

The cost is lower accuracy at high sparsity than with `fc3` kept dense, measured
before the decision (seed 42, CPU, no fine-tuning; the unpruned model scores 89.1%):

| Pruned | 50% | 80% | 90% | 95% | 99% |
|---|---|---|---|---|---|
| global magnitude, fc1 + fc2 + fc3 (used) | 88.5% | 75.0% | 55.4% | 39.2% | 19.0% |
| global magnitude, fc1 + fc2 (fc3 dense) | 88.4% | 76.4% | 65.5% | 53.8% | 29.4% |
| per-layer magnitude, all three (not used; for comparison) | 87.1% | 64.1% | 45.1% | 28.1% | 10.3% |

`fc3` has only 1,000 of the 266,200 weights, so keeping it dense would barely
change the model's overall sparsity. If that variant is ever wanted, it is one
line in the config (layers left unpruned are stored dense in every format)
followed by rerunning `evaluate` and `benchmark`; training is unaffected.

## 3. Storage formats

| Format | What is stored per pruned layer | Forward pass |
|---|---|---|
| original dense | the trained W (unpruned model, the accuracy reference) | `F.linear(x, W, b)` |
| pruned dense | W with zeros applied | `F.linear(x, W_p, b)` |
| dense+mask | the original W and a bool mask | `F.linear(x, W * mask, b)` every pass, as `torch.nn.utils.prune` does |
| COO | values + int64 row/column indices of the retained entries | `torch.sparse.mm(W, h) + b` |
| CSR | values + int64 column indices + row pointers | `torch.sparse.mm(W, h) + b` |

- COO and CSR are built from the **mask positions**, not from `value != 0`, so a
  retained weight that is exactly 0.0 is still stored. All formats hold exactly
  the same retained set.
- None of the GCN graph steps (symmetrization, self-loops, degree normalization)
  is applied to weight matrices.
- Sparse formats keep activations feature-major (`h' = W h + b`) because
  `torch.sparse.mm` needs the sparse operand first. The function is the same;
  only the floating-point summation order differs.
- **Weight storage bytes** = `numel × element_size` of every stored tensor (values,
  indices, row pointers, mask). Biases (1,640 bytes) are dense in every format
  and reported separately. Per value: dense 4 B for *every* weight, dense+mask
  5 B for every weight, COO 20 B per retained weight, CSR 12 B per retained
  weight + 8 B per row. COO therefore breaks even with dense at 80% sparsity,
  CSR at about 67%.

## 4. Correctness and epsilon

Two comparisons, kept apart as in the GCN pruning study:

| | Compares | Same weights? | Meaning of a difference |
|---|---|---|---|
| **Pruning effect** | pruned model vs original dense model | no | the accuracy cost of pruning (an experimental outcome) |
| **Format error** | dense+mask / COO / CSR vs pruned dense, same device | yes | numerical error of the storage format (must be ≈ 0) |

Both reuse the GCN functions unchanged (`src/gcn/evaluation.py`,
`src/gcn/error_analysis.py`), over all 5,000 validation + 10,000 test images:
- test and validation accuracy, signed accuracy difference in pp;
- prediction agreement;
- mean and max (= epsilon) absolute logit difference, `allclose(atol=1e-5, rtol=1e-4)`;
- the margin condition: if an image's margin (top logit minus second) exceeds
  2 × epsilon, its prediction cannot change. Violations must be 0.

A format is **equivalent** when agreement is 100%, the accuracy difference is
exactly 0 and `allclose` holds. The tolerance is the GCN one and is never loosened.

**Dense+mask is bit-identical to pruned dense** (`bitwise_equal`, epsilon 0):
multiplying by the mask reproduces the pruned weights exactly and the same
kernel runs. So the plan's "prediction agreement against dense with mask" and
the comparison against pruned dense are the same numbers.

**Pilot result (seed 42, CPU):** across all 25 pruned conditions, COO and CSR
agreed on every prediction. The largest epsilon was 1.5e-5 (top-K per neuron
at 50%, the level with the most weights left to sum). It is still inside
`allclose`.

## 5. Harness interface (shared with Dai)

`src/mlp/harness.py`:

```python
variant = load_variant(checkpoint, method, sparsity, pruning_seed, storage_format, device, pruned_layers)
x = make_batch(test_images, batch_size, device)      # first B standardized test images
with torch.inference_mode():
    logits = variant.model(x)
check_resident(variant)                              # sparse formats: no dense W or mask on the device
```

- `variant.metadata` holds what every result records:
  - model seed, pruning seed, method;
  - requested and actual sparsity (per layer and overall);
  - format, storage bytes (per layer, broken down by component);
  - checkpoint identity (path, sha256, best epoch, accuracies).
- `variant.timing_ms` holds pruning, format build and transfer times. These are
  never part of inference latency.
- **GPU residency:** the checkpoint is loaded and pruned on CPU, the requested
  format is built on CPU, and only that format's tensors and the biases are
  moved to the device. The dense `nn.Module` never leaves the CPU.
  `variant.model.tensors()` lists exactly what is resident.
- Batch sizes: 1, 32, 256.
- `time_inference` is the reference timing loop until Dai's harness replaces it:
  20 warm-ups, peak counters reset, then 50 synchronized passes.

## 6. Experiment grid and stages

`scripts/run_mlp_experiments.py --stage <stage> --device cuda:1`:

| Stage | What | Size |
|---|---|---|
| `pilot` | train seed 42; evaluate the original and magnitude 90%; benchmark both in every format and batch size, 1 repetition | 15 benchmark runs |
| `train` | train seeds 42, 43, 44 | 3 processes |
| `evaluate` | every seed × 26 conditions (original + 3 methods × 5 levels, random × 3 pruning seeds) × 4 formats | 3 processes |
| `benchmark` | 3 repetitions × 3 seeds × (original dense + 15 pruned conditions × 4 formats) × 3 batch sizes; random uses pruning seed 101 only (latency depends on how many weights are kept, not which) | 1,647 processes |

- Each benchmark run is its own process, with the repetition as the outer loop.
- Each run writes one JSON:
  - its model seed, pruning seed, method, actual sparsity, format and batch size;
  - storage bytes, resident tensor bytes and CUDA allocated/peak/reserved bytes;
  - all 50 latency samples;
  - a format check of its own batch against pruned dense on the same device.
- **Energy:** recorded as `null` until Dai establishes a reliable method.
- **Resuming:** finished outputs are skipped, so rerunning a stage resumes it.

## 7. Reports

`scripts/summarize_mlp_results.py --results-dir results/mlp/cuda1` writes
`results/mlp/summary_cuda1/`:
- `summary.md` and CSVs: baseline, accuracy, weight storage, latency, peak
  memory, epsilon;
- `runs.csv`: one flat row per benchmark run (seeds, method, actual sparsity,
  format, batch size, bytes, peak memory, latency, energy, format check), for
  the merged results CSV with Dai's harness;
- `accuracy_vs_sparsity.png`, by pruning method;
- `weight_storage_vs_sparsity.png`, by format;
- `peak_memory_vs_sparsity_<method>.png` and `latency_vs_sparsity_<method>.png`,
  by format, one panel per batch size.

All charts show the mean over seeds with min–max whiskers, and the original
dense model as a gray reference line. The sparsity axis is a log scale of the
retained fraction.

## 8. Reproduction (lab GPU)

```bash
source .venv/bin/activate
pytest
python scripts/run_mlp_experiments.py --stage pilot     --device cuda:1
python scripts/run_mlp_experiments.py --stage train     --device cuda:1
python scripts/run_mlp_experiments.py --stage evaluate  --device cuda:1
python scripts/run_mlp_experiments.py --stage benchmark --device cuda:1   # long: run inside tmux
python scripts/summarize_mlp_results.py --results-dir results/mlp/cuda1
```

CPU runs (`--device cpu`) are for development only. They go to
`results/mlp/*cpu*/`, which git ignores.

## 9. Open decisions

1. **Pruned layers: decided.** All three layers are pruned (section 2).
2. **Pruning implementation:** swap in Pranavi's signed-weight heuristics once
   they pass `tests/test_weight_pruning.py`, and rerun `evaluate` and `benchmark`.
3. **Harness and CSV columns:** Dai's harness can call `load_variant` /
   `make_batch` directly, or replace `time_inference` inside
   `scripts/benchmark_mlp.py`. The columns in `runs.csv` are a proposal for the
   agreed results CSV. Energy (NVML) fills `energy_joules`.
4. **Fine-tuning:** optional in the plan, not in this round. If added, it
   creates new checkpoints per (seed, method, sparsity). The format comparison
   stays the same.
5. **Python adjacency list:** Pranavi's bipartite list is for correctness and
   byte accounting only, as in the plan. It is not timed here.

## 10. Exported bundle (for the adjacency-list benchmark)

`exports/fashion_mlp_cuda1.npz` holds everything needed to run the trained
models outside this pipeline. It is written by
`python scripts/export_mlp_bundle.py --run cuda1` and needs only numpy to read:

- **Weights and biases:** `m{S}_{fc1,fc2,fc3}_weight` and `_bias` for each model
  seed S (42, 43, 44). Weights are (out, in), so a layer computes `h @ W.T + b`,
  with ReLU after fc1 and fc2.
- **Test set:** `x_test` (10,000 × 784, standardized exactly as in every run)
  and `y_test`. The benchmark batches are `x_test[:B]` for B = 1, 32, 256.
- **Masks:** `{tag}_{layer}_mask` for all 25 pruning conditions per seed, as bool
  arrays where True = keep. `tag` is the condition name used in `results/`, e.g.
  `m42_magnitude_s90` or `m42_random_p101_s90`. Each mask's hash equals the
  `mask_sha256` in the evaluate and benchmark JSONs.
- **Reference logits:** `m{S}_logits_test` (original model, full test set) and
  `{tag}_logits_batch` (pruned dense, first 256 test images), both computed
  on CPU. Use them to check that another implementation computes the same
  function.
- **Metadata:** `metadata`, a JSON string with the conventions, checkpoint
  hashes, per-condition actual sparsity and test accuracy, and the benchmark
  settings.

```python
import json
import numpy as np

b = np.load("exports/fashion_mlp_cuda1.npz")
meta = json.loads(str(b["metadata"]))
W1 = np.where(b["m42_magnitude_s90_fc1_mask"], b["m42_fc1_weight"], 0)  # pruned fc1
```

Before writing, the export script checks that the data split matches training.
It then reloads the bundle with numpy alone and recomputes every model and
condition from it. Finally it checks that every checkpoint and mask hash
matches the committed results.

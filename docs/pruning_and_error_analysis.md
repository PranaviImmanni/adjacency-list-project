# PubMed, Edge Pruning and Error Analysis

This extends `gcn_inference_plan.md`. The Cora baseline and the synthetic
sweeps are finished and are not rerun. Everything below uses the same model,
normalization, FP32/TF32-off setting and benchmark procedure.

The settings in sections 1–4 were fixed in code (`src/gcn/pruning.py`,
`scripts/run_pruning_experiments.py`) before any pruned-graph test accuracy
was looked at. Test accuracy is reported, never used to choose a method,
fraction, seed or tie rule.

## 1. Datasets and checkpoints

| | Cora | PubMed |
|---|---|---|
| Split | Planetoid public | Planetoid public |
| Nodes | 2,708 | 19,717 |
| Unique undirected edges | 5,278 | 44,324 |
| Normalized nnz (A + I, both directions) | 13,264 | 108,365 |
| Features | 1,433, binary bag-of-words | 500, TF-IDF |
| Classes | 7 paper topics | 3 diabetes categories |
| Train / val / test | 140 / 500 / 1,000 | 60 / 500 / 1,000 |
| Checkpoint | `checkpoints/cora_seed42_gpu.pt` (existing, best epoch 45) | `checkpoints/pubmed_seed42_gpu.pt` (new) |

- Feature and class counts are read from the dataset. Features are
  row-normalized (`NormalizeFeatures`) for training, inference and pruning
  scores alike.
- Both datasets come with exactly these fields: `x`, `y`, `edge_index`,
  `train_mask`, `val_mask` and `test_mask`. Every result file records them
  under `dataset_info.available_fields`. There are no publication years, edge
  weights or edge confidences, and no raw text, so no method uses them.
- Neither dataset has duplicate edges or self-loops in `edge_index`, and
  neither has a node with an all-zero feature vector. The code still handles
  both cases (section 3).
- PubMed checkpoint: the same two-layer GCN (hidden 64, dropout 0.5, Adam,
  lr 0.01, weight decay 5e-4, 200 epochs), seed 42, trained once on the
  original graph in COO. The saved weights come from the epoch with the best
  validation accuracy, with ties broken by validation loss. It is trained on
  the lab GPU like the Cora checkpoint that the GPU baseline used.
- Each checkpoint is frozen: every experiment on that dataset loads the same
  file, and every result records its sha256, model seed and best epoch.
  Nothing is retrained after pruning.

## 2. Unpruned format comparison (PubMed)

This is the same as Cora (`gcn_inference_plan.md` §4–5):
- `compare_gcn_formats.py`: dense/COO/CSR on the GPU, with COO on the same GPU
  as the reference;
- `benchmark_gcn.py`: one process per format and repetition, with 3
  repetitions, 20 warm-ups and 50 timed passes.

The results go to `results/gpu/` next to Cora's. Dense PubMed needs
19,717² × 4 B = 1.56 GB.

## 3. Edge pruning

**Removable edges.** These are the unique undirected pairs (u, v) with u < v,
after duplicates are merged and self-loops dropped (`unique_undirected_edges`).
Removing a pair removes both directions.

**Count.** A removal fraction f removes `floor(f × E)` pairs, where E is the
original unique-pair count. The floor is exact for decimal fractions
(`Fraction(str(f))`). The fractions are 0, 10, 25, 50 and 75%. Every result
records the requested fraction, the actual fraction, and the removed and
retained counts.

**Nested sets.** Each method (and each seed, for random) has one ordering of
the pairs. The first `floor(f × E)` pairs of that ordering are removed, so
the edges removed at 10% are also removed at 25%, 50% and 75%.

**Random.** The ordering is `torch.randperm(E)` from a dedicated
`torch.Generator` seeded with the pruning seed. The pruning seeds are 101, 102
and 103. They choose edges only; they are not model seeds.

**Feature similarity.** This ordering puts the least similar pairs first:
- The score is the cosine similarity of the two endpoints' feature vectors,
  computed in float64 on the same row-normalized features the GCN sees.
  Cosine ignores row scaling, so this equals the cosine of the raw features.
- Only existing edges are scored, in chunks of 4,096. No N × N matrix is ever
  built.
- **Zero vectors:** if either endpoint has an all-zero feature vector, the
  pair scores exactly 0.0 instead of NaN. The features are non-negative, so 0
  is the lowest possible score: such pairs are removed among the first.
- **Ties:** the pairs are sorted by (u, v) and the sort is stable, so equal
  scores are removed in (u, v) order. The result is fully deterministic.
- No labels or split masks are used. The heuristic sees only node features.

**After pruning.** The retained pairs go through the same
`build_normalized_coo`:
1. both directions are restored;
2. exactly one self-loop is added per node;
3. degrees and `D^-1/2 (A + I) D^-1/2` are recomputed from the pruned graph;
4. the result is one canonical COO tensor, and CSR or dense is derived from
   it only when needed.

All nodes, features, labels and masks are kept. A node whose edges are all
removed keeps its self-loop, and is counted under
`isolated_nodes_before_self_loops`.

**Saved for offline analysis:**
- `orderings/<dataset>_<method>[_s<seed>]_order.pt`: the pairs, the
  ordering, and the similarity scores. Any removal set can be rebuilt from it.
- `logits/*.pt`: the logits of every condition, from repetition 0.
- `logits/<dataset>_labels.pt`: the labels and masks.

Later runs check that they reproduce the saved ordering, and every run
records a sha256 of its logits.

## 4. Pruning experiments

For each dataset, `scripts/run_pruning_experiments.py --stage full` runs 17
conditions, all with CSR:

| Condition | Count |
|---|---|
| original graph (the single unpruned baseline, shared by every curve) | 1 |
| random, seeds 101/102/103 × 10/25/50/75% | 12 |
| feature similarity × 10/25/50/75% (deterministic, run once) | 4 |

Each condition runs 3 times, each in its own process (51 runs per dataset).
The repetition is the outer loop.

Each run (`scripts/benchmark_pruning.py`) records:
- validation and test accuracy, and the signed accuracy loss against the
  original graph;
- prediction disagreement on all nodes and on test nodes;
- the mean and max absolute logit change;
- exact adjacency bytes, value and index dtypes, and CUDA peak allocated and
  reserved memory;
- the 50 latency samples and their summary;
- time spent on ordering, selection, normalization, conversion and transfer.

**Measurement rules:**
- Only the target CSR, the features and the model are on the GPU while
  memory and time are measured.
- The peak counters are reset after the warm-up outputs are released and
  before the measured passes. The peak is absolute: it includes the resident
  adjacency, features and parameters.
- `cuda_allocated_before_inference_bytes` is recorded separately.
- Pruning, normalization, conversion and transfer are never inside the
  latency.

After the measurement, the run computes two references outside the timer:
- the original graph in CSR on the same GPU (for the pruning effect);
- the pruned graph in COO on CPU (a format sanity check).

**Note:** the unpruned format benchmarks (`benchmark_gcn.py`, section 2) reset
the peak before warm-up, as in the existing Cora baseline. Warm-up outputs
are freed immediately, so both resets give the same peak. The pruning runs
use the post-warm-up reset that the pruning plan specifies.

**Format verification** (`--stage verify`) runs dense, COO and CSR on three
graphs per dataset:
- the original graph;
- feature similarity at 50%;
- random seed 101 at 50%.

All three formats come from the same normalized COO, with the same checkpoint
and features. COO on the same GPU is the reference. Not every format is
benchmarked at every pruning level.

## 5. Error analysis

### A. Format error: same graph, different representation

- `mean_abs_logit_diff`;
- `epsilon_logit` = the max absolute logit difference;
- `allclose(atol=1e-5, rtol=1e-4)`;
- prediction agreement (all nodes and test nodes);
- the test-accuracy difference in pp, both signed (format − reference) and
  absolute.

The tolerance is the same as for Cora and is never loosened. A failure is
reported as a failure.

### B. Pruning effect: original graph vs pruned graph

Both are CSR, with the same checkpoint and the same device.
- `accuracy_loss_pp = original − pruned`, signed. It is negative when pruning
  improves accuracy.
- Prediction disagreement on all nodes and test nodes.
- The mean and max absolute logit change.

A changed prediction here is the experiment's outcome, not a correctness
failure.

### C. Prediction margins

For node i, with reference logits z and compared logits z':
- `margin_i = z_(1) − z_(2)`, the largest minus the second-largest reference
  logit;
- `epsilon_i = max_c |z'_ic − z_ic|`.

The reported quantities are:
- the fraction of nodes with `margin_i > 2 epsilon_i` (all nodes and test
  nodes);
- the observed disagreement rate;
- the number of disagreements among nodes that meet the condition. This must
  be 0.

These are computed in float64 on the saved outputs. `summarize_pruning_results.py`
recomputes them from the `.pt` files.

### Why these statements hold

1. **Lossless conversion keeps the computation the same.** Dense, COO and CSR
   all hold exactly the same nonzero values of `A_hat`, because every format
   is derived from one COO tensor. So `A_hat (X W)` is mathematically the same
   product in every format.
2. **Floating-point results can still differ.** FP32 addition is not
   associative. cuBLAS (dense) and cuSPARSE (COO, CSR) add the terms of each
   row in different orders and with different blocking, so results differ in
   the last bits. On the GPU this is about 1e-7 (Cora: max 9.5e-7). On CPU it
   was exactly 0. A format error is bounded rounding noise, not a different
   model.
3. **The margin condition is sufficient for a stable prediction.** Let c* be
   the reference argmax. Suppose `margin_i > 2 epsilon_i` and every logit moves
   by at most `epsilon_i`. Then for any c ≠ c*:
   `z'_c* ≥ z_c* − ε > z_c + ε ≥ z'_c`.
   So c* is still the argmax. The condition is not necessary: a node can fail
   it and still keep its class, because the bound assumes the worst-case
   direction for every logit at once.
4. **The accuracy change is at most the disagreement rate.** On a fixed set of
   n evaluated nodes, a node's correctness can change only if its prediction
   changes. So |Δaccuracy| ≤ (#disagreements)/n, and in pp,
   |Δacc_pp| ≤ 100 × the disagreement rate. Equality needs every changed
   prediction to move in the same direction (all right→wrong, or all
   wrong→right). Format comparisons have 0 disagreements, so their accuracy
   difference is exactly 0.
5. **Edge pruning has no general accuracy guarantee.** Pruning changes `A_hat`
   itself, not just the rounding:
   - the removed entries disappear;
   - the degree normalization of every touched node changes;
   - the change spreads through two layers.

   Nothing bounds `epsilon_i` in advance, and neither the margin condition nor
   accuracy can be promised for an arbitrary pruned graph. The measured
   coverage of the margin condition is a description of what happened, not a
   guarantee. Accuracy can even go up (a negative loss), which is why the sign
   is kept.

**Units.** Accuracy differences are in percentage points (pp) of
fraction-of-nodes accuracy. Logit differences (`epsilon_logit`, mean |Δ|) are
raw logit units. The two are never compared directly: only the margin
condition links them.

## 6. Reproduction (lab GPU, `cuda:1`)

```bash
source .venv/bin/activate
pytest

# Cora: pilot, full pruning, format verification (existing checkpoint)
python scripts/run_pruning_experiments.py --dataset cora --stage pilot  --device cuda:1
python scripts/run_pruning_experiments.py --dataset cora --stage full   --device cuda:1
python scripts/run_pruning_experiments.py --dataset cora --stage verify --device cuda:1

# PubMed: train once, unpruned format comparison, then pruning
python scripts/train_gcn.py --dataset pubmed --device cuda:1 --output checkpoints/pubmed_seed42_gpu.pt
python scripts/compare_gcn_formats.py --dataset pubmed --device cuda:1 \
  --checkpoint checkpoints/pubmed_seed42_gpu.pt --output results/gpu/pubmed_format_comparison_cuda1.json
for rep in 0 1 2; do for fmt in dense coo csr; do
  python scripts/benchmark_gcn.py --dataset pubmed --format $fmt --device cuda:1 \
    --checkpoint checkpoints/pubmed_seed42_gpu.pt --repetition $rep \
    --output results/gpu/pubmed_${fmt}_cuda1_rep${rep}.json
done; done
python scripts/run_pruning_experiments.py --dataset pubmed --stage full   --device cuda:1
python scripts/run_pruning_experiments.py --dataset pubmed --stage verify --device cuda:1

# Reports (no GPU needed)
python scripts/summarize_gcn_results.py --results-dir results/gpu --out-dir results/summary_gpu
python scripts/summarize_pruning_results.py
```

The pruning stages resume if they are interrupted (rerun the same command).
The outputs are:
- `results/pruning/<dataset>_cuda1/`: the runs, `orderings/` and `logits/`;
- `results/pruning/verify_<dataset>_cuda1/`;
- `results/pruning/summary/`: `summary.md` and the charts.

# GCN Inference Plan: Dense vs COO vs CSR Adjacency

This plan covers the GCN part of the project (`src/gcn/`, `scripts/`). It
follows the memory and equivalence rules in `memory_eval_spec.md`.

## 1. Research question

Can a GCN's dense N x N adjacency matrix be replaced by a sparse
representation (COO or CSR) without changing its predictions, and how much
adjacency memory and inference latency does that save on a GPU?

Milestone 1 (done on CPU): on Cora, build mathematically equivalent dense,
COO and CSR normalized adjacencies, train one two-layer GCN checkpoint with
COO, and show that this one checkpoint gives equivalent predictions with all
three formats.

## 2. Model

One model class (`src/gcn/model.py`) is used for every format:

```
H = ReLU(A_hat (X W1) + b1)
H = Dropout(H, p=0.5)        # training only
Z = A_hat (H W2) + b2        # class logits
```

- The features are transformed before aggregation (`A_hat (X W)`) for every format, so all formats do the same amount of work.
- Aggregation always goes through `adjacency_multiply` in `src/gcn/adjacency.py`:
  - dense (`torch.strided`) uses `torch.mm`;
  - COO and CSR use `torch.sparse.mm`.
- PyG's `GCNConv` is not used, so the adjacency storage is the only thing that changes between runs.
- The bias is added after aggregation and weights use Glorot init. Both match Kipf & Welling and PyG's `GCNConv`.
- Training: Adam, lr 0.01, weight decay 5e-4, hidden size 64, dropout 0.5, up to 200 epochs, seed 42.
  - The saved weights are from the epoch with the best validation accuracy.
  - If two epochs tie on validation accuracy, the one with lower validation loss wins.
- Training uses COO only. Training time and memory are not measured.

## 3. Canonical adjacency preprocessing

`build_normalized_coo(edge_index, num_nodes)`:

1. Make the graph undirected.
2. Merge duplicate edges. The graph is unweighted, so a duplicate edge still has weight 1.
3. Remove existing self-loops, then add exactly one per node.
4. Index entries as `[target, source]`, so `A @ X` sums source features into the target node's row.
5. Compute `A_hat = D^{-1/2} (A + I) D^{-1/2}`.
6. Store the result as one coalesced FP32 COO tensor. This is the canonical copy.
7. Build CSR (`to_sparse_csr`) and dense (`to_dense`) only from that COO tensor (`convert_adjacency`).

Normalization runs once, so all three formats hold exactly the same values. Any
difference in their outputs comes from the multiplication kernels, not from
the preprocessing.

Cora check: 10,556 directed edges + 2,708 self-loops = 13,264 nonzeros.
Cora already has no duplicate edges and no self-loops.

## 4. Metrics

**Prediction equivalence** (`src/gcn/evaluation.py`; reference = COO):

| Metric | Target |
|---|---|
| Prediction-label agreement (all nodes and test nodes) | 100% |
| Test-accuracy difference (percentage points) | 0 |
| Mean / max absolute logit difference | reported |
| `torch.allclose(atol=1e-5, rtol=1e-4)` | True |

COO is the reference because the other formats are built from it and the
checkpoint was trained with it. Cosine similarity is not used: it is
weaker than max-abs-diff and `allclose`.

**Storage.** `adjacency_storage_bytes` adds up `numel() * element_size()` over the
tensors that actually exist, not a formula:

| Format | What is counted |
|---|---|
| dense | values |
| COO | values + indices |
| CSR | values + column indices + row pointers |

`estimate_dense_bytes` gives the N x N size without allocating it. It reuses
`structural_memory_adj_matrix` from `src/measurement/memory_utils.py`.

**Latency and GPU memory.** `scripts/benchmark_gcn.py` runs one
(dataset, format, device, repetition) per process:

- 20 warm-up runs, then 50 timed forward passes, each surrounded by a device
  synchronize.
- Not timed: dataset loading, normalization, format conversion and the transfer
  to the device. Conversion and transfer are recorded separately.
- On CUDA it records `max_memory_allocated` and `max_memory_reserved`, plus the
  memory already allocated before inference. On any other device these are `null`.

## 5. CPU now, CUDA later

| Device | Status |
|---|---|
| CPU (Apple M1) | Development and correctness. All three formats supported. Latency is labeled `latency_is_debug_only: true` and is **not** a result. |
| `cuda:N` (lab NVIDIA GPU) | Official latency and memory runs. FP32 with TF32 off (`torch.backends.cuda.matmul.fp32_precision = "ieee"`; older PyTorch uses `allow_tf32 = False`), set only when the device is CUDA. |
| MPS | Sparse runs are rejected with an error, because MPS sparse support is incomplete. Not used. |

All scripts take `--device cpu` or `--device cuda:0`. Nothing calls `.cuda()`
directly. Every CUDA-only call is guarded by `device.type == "cuda"`.

Steps on the lab GPU:

1. Install the CUDA build of PyTorch, then `pip install -r requirements.txt`.
2. `pytest` (unit tests), then `pytest -m integration`.
3. Reuse `checkpoints/cora_seed42.pt` (copy it over), or retrain with
   `--device cuda:0`. Checkpoints load on CPU first, so either works.
4. `python scripts/compare_gcn_formats.py --device cuda:0`. This must report all
   formats equivalent before any timing is trusted. GPU results will not be
   bit-identical to each other (cuBLAS vs cuSPARSE), but must stay inside the tolerance.
5. For each format in dense/coo/csr and each repetition in 0/1/2, run
   `scripts/benchmark_gcn.py --device cuda:0 --repetition <r>` in a separate process.
6. Report the median of the per-repetition medians, together with adjacency
   bytes and peak allocated/reserved memory.

## 6. Scope boundaries

In scope now:
- Cora
- FP32
- One two-layer GCN
- Dense, COO and CSR formats
- CPU correctness, plus the CUDA-ready benchmark runner

Not implemented yet. Each has a planned extension point:

| Later work | Extension point |
|---|---|
| GraphSAGE, GAT | a new model in `src/gcn/` (or a sibling package) that reuses `adjacency_multiply` |
| FP16 / BF16 / AMP | `SUPPORTED_DTYPES` in `src/gcn/config.py` |
| Custom CUDA kernels, Tensor Core/MMA, CUTLASS, other formats (CSC/BSR) | `STORAGE_FORMATS` + `adjacency_multiply` |
| Maximum-capacity search (size sweep toward the GPU memory limit) | `scripts/run_synthetic_sweep.py` (see section 8) |
| ogbn-arxiv | a loader in `LOADERS` (`src/gcn/datasets.py`) |
| Automatic format selection | reads the benchmark JSON results |

## 7. How this fits with the existing work

- `src/gcn/` and `scripts/` are new and self-contained. The existing conversion,
  custom multiplication, Strassen, synthetic generator and benchmark code are
  unchanged.
- Reused from `src/measurement/memory_utils.py` because their behavior matches
  exactly:
  - `structural_memory_adj_matrix` (dense size estimate);
  - `prediction_match_rate` (label agreement);
  - `output_closeness(..., "max_abs_diff")`.
- The custom adjacency-list multiplication (`src/operations/multiplication.py`)
  and Strassen are **not** used in the GCN forward pass. The GCN baseline uses
  PyTorch's own dense and sparse kernels, so it stays a clean reference that those
  operations can be checked against later.
- Synthetic graphs come from `src/synthetic/edge_generator.py`, not
  `generate_synthetic_graph`. That generator builds a dense, directed N x N
  matrix, which costs O(N^2) memory and roughly doubles the edge density once
  `build_normalized_coo` symmetrizes it (see section 8).
- Equivalence follows the three checks in `memory_eval_spec.md`: prediction
  match, accuracy parity and output closeness. The tolerance is stricter here
  because the same checkpoint is used for every format.

## 8. Synthetic sweeps

Cora is one point: 2,708 nodes at 0.14% density. The synthetic sweeps change one
variable at a time to show where the formats cross over.

| Sweep | Changes | Fixed | Values | Graph seeds |
|---|---|---|---|---|
| pilot | nothing | everything | 1,000 nodes, average degree 8 | 42 |
| size | number of nodes | average degree 8 | 1k, 5k, 10k, 20k, 40k | 42, 43, 44 |
| density | % of possible edges | 5,000 nodes | 0.1, 1, 5, 10, 20, 30, 35, 40% | 42, 43, 44 |

**Graphs.** Uniform random G(n, m) (`sample_undirected_edges`): exactly m
distinct undirected pairs, drawn without replacement, stored once as
(i, j) with i < j, with no self-loops. Memory is O(m); it never builds all
N(N-1)/2 pairs.
- Average degree d = average neighbors before self-loops, so m = N*d/2.
- Density = m / (N(N-1)/2).
- After `build_normalized_coo(edge_index, N)`, nnz = 2m + N. The benchmark
  checks this, and passing N keeps isolated nodes.

**Model and seeds.** No training: the same two-layer GCN with random Glorot
weights. Input 64, hidden 64 (from the config), output 7. Features are
standard normal. There are three independent, recorded seeds:
- graph seed: varies across official runs;
- feature seed: fixed at 0;
- model seed: fixed at 1.

The graph and features use their own `torch.Generator`. The model is seeded
just before it is built. So changing the graph never changes the features or
the weights.

**Equivalence.** There are no labels, so `compare_logits` checks the logits
against COO on CPU. It requires 100% argmax agreement on all nodes and
`allclose(atol=1e-5, rtol=1e-4)`, the same tolerance as Cora. Failures are
reported, not hidden by loosening the tolerance.

**Measurement.** `scripts/benchmark_synthetic.py` runs one (graph, format,
repetition) per process and regenerates the graph from its seed.
`graph_fingerprint` records a hash of the edges, so it can be checked that
every format saw the same graph. Timing, memory and the JSON fields otherwise
match `benchmark_gcn.py`. `scripts/run_synthetic_sweep.py` runs a whole sweep
and resumes where it stopped.

**Storage crossovers** (adjacency bytes only, FP32 values, int64 indices):
- COO: 20 bytes/nnz, so it matches dense near 20% density.
- CSR: 12 bytes/nnz + 8(N+1), so it matches dense near 33%.

These do not predict latency or total peak memory. The 30% and 35% density
points bracket the CSR crossover.

**Memory limit.** Dense adjacency needs 4N^2 bytes: 6.4 GB at 40k nodes and
22.5 GB (about 21 GiB) at 75k. That is a lower bound on what a run needs:
features, intermediate results, allocator overhead and other processes add to
it. Dense is built on CPU and then copied, so the host also needs 4N^2 bytes
of RAM. Densifying a CUDA tensor would put that cost on the GPU instead. Going
beyond 40k nodes (the capacity search) needs explicit pre-checks against free
GPU memory first.

**Future work.** G(n, m) degrees are nearly uniform. Real graphs have skewed
degrees, which affects how evenly sparse GPU kernels spread their work.
Power-law graphs are a follow-up and are not part of these sweeps.

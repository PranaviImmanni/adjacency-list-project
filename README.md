# Hypergraph multiplication experiments

This project implements Tests 1–3 for incidence-based feature propagation.
Open this extracted folder in VS Code. Python 3.10 or later is recommended;
the included run was performed with Python 3.12 on CPU. See the result metadata
for the exact PyTorch version and machine information.

## 1. Set up Python in VS Code

Install the Microsoft Python extension, then open **Terminal → New Terminal**.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS / Linux / WSL:

```bash
source .venv/bin/activate
```

For a CPU setup on Windows / Linux / WSL:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install psutil
```

For macOS:

```bash
python -m pip install -r requirements.txt
```

In VS Code, use **Python: Select Interpreter** and choose `.venv`.
If PowerShell blocks activation, you can select the interpreter and run
`.venv\Scripts\python.exe hypergraph_tests.py test1` directly.

For a laboratory NVIDIA GPU, install a CUDA-enabled PyTorch build appropriate
for the laboratory driver using the official selector: https://pytorch.org/get-started/locally/
Do not use the CPU wheel above for GPU experiments. This project does not require
CUDA for the initial tests. On Apple M1, run `--device cpu`; MPS is not benchmarked.

## 2. Test 1: correctness and edge cases

```bash
python hypergraph_tests.py test1 --output results/test1_run
```

The worked example uses e1 = {A,B,C}, e2 = {B,D}, X = [1,2,3,4]^T.
For unnormalized unit-weight propagation, expected Y = [6,12,6,6]^T.

Seven graph cases are checked: worked example, isolated node, singleton groups,
weighted overlapping groups, repeated memberships, distinct groups with identical
members, and a mostly isolated graph. Each is checked with one and seven features,
with and without normalization, across six implementations: 168 comparisons.
Invalid memberships and nonpositive/nonfinite weights must be rejected.

The six implementations are:

| Method name | Operation |
| --- | --- |
| dense | Dense incidence factors |
| coo | Sparse COO incidence factors |
| csr | Sparse CSR incidence factors, including materialized CSR transpose |
| membership | Gather / index_add in both directions |
| explicit_dense | Precompute dense propagation operator P |
| explicit_sparse | Precompute sparse COO P using sparse × sparse multiplication |

Unnormalized: Y = H W H^T X.
Normalized: Y = D_v^(-1/2) H W D_e^(-1) H^T D_v^(-1/2) X.
Degrees and weights are vectors; no dense diagonal matrices are allocated.
Zero-degree node inverse factors are zero, giving zero outputs for isolated nodes.
Repeated memberships within a group are deduplicated; separate hyperedge IDs are retained.

Small cases use an FP64 dense reference. The implementation uses FP32 and checks
atol = 1e-5, rtol = 1e-4. Inspect the reported maximum absolute error as well.
Failures are reported, not hidden by increasing tolerances or changing algorithms.

## 3. Test 2: explicit vs. factorized propagation

```bash
python hypergraph_tests.py test2 --output results/test2_run
```

Default configuration: N = M = 256, eight nodes per group, 64 features,
seed 42, one CPU thread, 10 warm-ups, 50 measured calls per method.

To use a different configuration and three seeds:

```bash
python hypergraph_tests.py test2 --n 1000 --m 1000 --k 8 --f 64 --seeds 42 43 44 --output results/test2_1000
```

Each method runs in a fresh subprocess. This avoids retaining other methods'
representations during memory measurements. The same graph, positive weights,
and features are regenerated deterministically for every method.

The factorized methods calculate the operator from right to left without
materializing P. The explicit methods build P once, then time P @ X.
Construction time is separate from propagation time. Estimates for 10 and 100
calls are build_ms + K × median_ms, not independently measured multi-call totals.

## 4. Test 3: scaling sweeps

Small development sweep (six configurations, six methods = 36 benchmark rows):

```bash
python hypergraph_tests.py test3 --output results/test3_small
```

Larger study with node, hyperedge, group-size, and feature sweeps:

```bash
python hypergraph_tests.py test3 --full --seeds 42 43 44 --output results/test3_full
```

The full grid uses node counts 100, 1000, 5000; hyperedge counts 500, 1000, 2000;
group sizes 2, 4, 8, 16, 32; and feature dimensions 16, 64, 128.
Duplicate configurations are run once per seed. The full sweep takes longer.

On NVIDIA GPU:

```bash
python hypergraph_tests.py test3 --device cuda --full --seeds 42 43 44 --budget-mb 1024 --output results/gpu_full
```

`--budget-mb` is a conservative **preflight estimate ceiling**, not a GPU allocator
limit. Unsafe estimates are skipped. It is not a measured maximum-capacity search.
Inspect actual peaks and choose larger configurations using Test 2 when studying
capacity. Report only the largest tested configuration that completes within your
chosen actual memory budget; do not call that the theoretical maximum.

## 5. Run all three initial tests

```bash
python hypergraph_tests.py all --output results/my_initial_run
```

Alternatively, use **Run and Debug** in VS Code and choose one of the three included
launch configurations. CPU is the default. In VS Code, open this project folder
itself so `${workspaceFolder}` points to the directory containing the script.

## 6. Read the results

Each run saves `test1.json/csv`, `test2.json/csv`, or `test3.json/csv` and
`environment.json` in the chosen output directory. Initial results from this
workspace are included in `results/cpu_initial/`.

| Field | Meaning |
| --- | --- |
| passed / max_abs_error | Numerical equivalence to the FP64 reference |
| representation_bytes | Actual unique tensor buffers needed for the representation and normalization; excludes X and output |
| operator_nnz | Nonzero count in explicit P; blank for factorized methods |
| graph_transfer_ms | Time transferring membership and weight inputs; CPU often performs no copy |
| build_ms | Representation construction / normalization; separate from feature propagation |
| median_ms / iqr_ms | Median and interquartile range of measured propagation calls |
| cpu_baseline_rss_bytes | Process memory immediately before measured setup |
| cpu_sampled_peak_rss_bytes | Highest observed process RSS during setup, validation, and timing |
| cpu_observed_rss_increase_bytes | Sampled peak minus baseline; not exact allocator memory |
| cuda_peak_allocated_bytes | Peak live PyTorch CUDA tensor allocations during measured setup, validation, and timing |
| cuda_peak_reserved_bytes | Peak memory reserved by the CUDA caching allocator |
| status | ok, skipped_preflight, or error |

Storage counts shared buffers once. Dense H transpose is a view; COO and CSR
transposes are counted when they own separate buffers. Explicit P only needs P
for propagation, so incidence and degree data are not retained in its storage count.
Setup peaks still include construction intermediates. Membership gathers may
allocate S × F temporary tensors; their cost belongs in peak memory.

CPU RSS includes the Python runtime, imported libraries, original CPU inputs,
and the CPU reference. It is sampled at roughly 1 ms and can miss brief peaks.
The baseline difference can also reflect native allocator reuse. Use storage bytes
for precise structural comparisons; CPU RSS is supporting information, not a
replacement for laboratory CUDA measurements. Process isolation improves
comparability but does not eliminate system noise.

CUDA timing is synchronized and TF32 is disabled for dense matmul. CUDA peaks
include the X tensor and input graph transfers but not the CUDA runtime's memory
outside PyTorch. Graph generation and the FP64 reference are performed on CPU
before the measured setup. This runner measures propagation, not model training
or prediction-label agreement. CUDA code is included but was not exercised in
the supplied CPU-only environment.

Rows with errors preserve the exception in JSON/CSV and make the command exit
unsuccessfully. Unsupported sparse operations are not replaced by dense ones.

## 7. Research interpretation

Start by checking that all outputs pass. Then compare storage bytes and latency
for the same configuration and seed. A lower storage count does not establish a
speedup or unchanged model accuracy. Synthetic hypergraph results alone do not
establish performance on real datasets.

When group size grows with N and M fixed, memberships and average degree also
grow. Interpret that sweep as a workload change, not a pure group-size effect.
Multiple seeds and realistic datasets are needed before drawing general conclusions.

## References

- Assigned structural-products paper: https://arxiv.org/html/2007.01842v4
- Incidence-based HGNN formulation: https://arxiv.org/abs/1809.09401
- PyTorch sparse multiplication: https://docs.pytorch.org/docs/stable/generated/torch.sparse.mm.html
- PyTorch installation selector: https://pytorch.org/get-started/locally/

# Memory-Efficient Machine Learning Using Adjacency Lists

CS 180 Individual Studies — San Jose State University, Fall 2026
Student: Pranavi Immanni
Professor: William Andreopoulos

## Project overview

This project studies whether adjacency lists can replace adjacency
matrices to reduce memory usage in graph-based machine learning, while
keeping model predictions equivalent. See `docs/` for the formal
specification of how memory is measured and how equivalence is checked.

## Structure

- `src/conversion/` — general method for converting any matrix into an
  adjacency matrix, and from there into an adjacency list (and back).
- `src/operations/` — adjacency-list analogs of matrix operations
  (starting with multiplication).
- `src/measurement/` — memory measurement and equivalence-check
  utilities (`memory_utils.py`).
- `src/synthetic/` — synthetic dataset generator, used to validate the
  pipeline before running on real datasets.
- `src/gcn/` — device-portable two-layer GCN that runs on dense, COO, or
  CSR normalized adjacency (see `docs/gcn_inference_plan.md`).
- `scripts/` — GCN entry points: train, compare formats, benchmark.
- `configs/` — experiment configs (`gcn_baseline.json`).
- `tests/` — round-trip correctness tests and equivalence tests.
- `docs/` — formalization write-up and `memory_eval_spec.md`.
- `data/` — downloaded datasets (not committed — see `.gitignore`).
- `reports/` — weekly updates, results, and graphs.
- `checkpoints/`, `results/raw/` — trained models and per-run benchmark
  JSON (not committed).

## How to run things

Setup (Python 3.10+; on the NVIDIA machine install the CUDA build of
PyTorch first):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### GCN: dense vs COO vs CSR

1. Unit tests (offline). The Cora integration test downloads data, so
   it only runs when asked for:

   ```bash
   pytest
   pytest -m integration
   ```

2. Train the Cora GCN on CPU, using COO adjacency. This downloads Cora to
   `data/` on the first run:

   ```bash
   python scripts/train_gcn.py --dataset cora --device cpu --output checkpoints/cora_seed42.pt
   ```

3. Check that the same checkpoint gives equivalent predictions with dense,
   COO and CSR. The script exits non-zero if any format fails:

   ```bash
   python scripts/compare_gcn_formats.py --dataset cora --device cpu --checkpoint checkpoints/cora_seed42.pt
   ```

4. CPU benchmark dry run. Each run covers one dataset, format, device and
   repetition, and writes one JSON file:

   ```bash
   python scripts/benchmark_gcn.py --dataset cora --format csr --device cpu \
     --checkpoint checkpoints/cora_seed42.pt --output results/raw/cora_csr_cpu_rep0.json
   ```

   To see all results as tables and charts, run the summary script. It
   reads every JSON file in `results/raw/` and writes
   `results/summary/summary.md` plus PNG charts. Open `summary.md` with
   the Markdown preview (Cmd+Shift+V in VS Code):

   ```bash
   python scripts/compare_gcn_formats.py --checkpoint checkpoints/cora_seed42.pt \
     --output results/raw/cora_format_comparison_cpu.json
   python scripts/summarize_gcn_results.py
   ```

5. **CPU timings are for debugging only** and are marked
   `latency_is_debug_only: true`. Official latency and GPU-memory results
   will be collected later on the lab NVIDIA GPU by running the same
   commands with `--device cuda:0`. See `docs/gcn_inference_plan.md` §5.

### Existing utilities

```bash
python src/measurement/run_benchmarks.py   # structural memory vs sparsity
python src/operations/multiplication.py    # adjacency-list multiply demo
python src/operations/strassen.py          # Strassen demo
```

# Synthetic dataset generator

Generates small synthetic (optionally weighted) graphs for testing the
conversion pipeline and memory measurement before running on real
datasets.


- `generator.py` — dense N x N matrix (optionally weighted), for the
  conversion and memory tests. Not used for the GCN sweeps: it is
  O(N^2) memory and directed.
- `edge_generator.py` — uniform random undirected G(n, m) edge lists with
  exactly m edges and O(m) memory, for the GCN size and density sweeps
  (`scripts/run_synthetic_sweep.py`, `docs/gcn_inference_plan.md` section 8).

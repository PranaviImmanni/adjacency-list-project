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
- `tests/` — round-trip correctness tests and equivalence tests.
- `docs/` — formalization write-up and `memory_eval_spec.md`.
- `data/` — downloaded datasets (not committed — see `.gitignore`).
- `reports/` — weekly updates, results, and graphs.

## How to run things

(To be filled in as each piece is built — e.g. how to run the
conversion utility, how to run the memory benchmarks, etc.)

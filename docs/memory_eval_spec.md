# Memory Evaluation Spec — CS 180 Individual Study

This spec fixes two things before any conversion or model code is written:
how memory will be measured, and what counts as "the same ML results."
Both apply to every experiment for the rest of the project.

## 1. Memory Measurement Method

Two numbers are reported for every comparison, not one.

### 1a. Structural memory (the representation itself)

This isolates whether the *representation* is smaller, independent of how
it's used.

- **Adjacency matrix:** `N x N x bytes_per_element` — constant regardless
  of how sparse the graph is.
- **Adjacency list (COO-style):** `2 x E x bytes_per_index + E x bytes_per_value`
  — scales with number of edges, not nodes squared.

Computed directly from array sizes (`.nbytes` in NumPy/PyTorch), not
estimated.

### 1b. Empirical memory (during actual use)

Captures what happens during a real forward/backward pass, since a naive
implementation could internally densify and lose the benefit of the
sparse representation.

- `torch.profiler.profile(profile_memory=True)` — detailed breakdown
  during training/inference
- `torch.cuda.max_memory_allocated()` — before/after on GPU
- `tracemalloc` — for pure-Python/CPU data structures

### 1c. Reporting

Memory is reported as a curve across graph size (nodes and/or edges), not
a single point — the adjacency matrix grows O(V^2) and the list grows
O(V+E), so plotting both shows where the crossover happens. This is
stronger evidence than a single "X% less memory" figure at one size.

## 2. Equivalence Criterion ("same ML results")

Checked in three tiers, cheapest to strongest. All three should pass
before a result is reported as "equivalent."

1. **Prediction match** — do the matrix-based and list-based models
   predict the exact same class for every test node? Should be ~100% if
   the operations are implemented correctly. Cheapest sanity check.
2. **Accuracy parity** — is the list-based model's accuracy within a
   small tolerance (0.5%) of the matrix-based model's, averaged across
   multiple random seeds to account for noise? This is the headline
   number for the report.
3. **Output-level closeness** — do raw logits/embeddings match closely
   (max absolute difference, or cosine similarity), not just the final
   argmax? Catches cases where predictions happen to agree but the
   underlying computation drifted.

### Controls

Held fixed across both representations for any single comparison:
- Random seed (weight init)
- Train/test split
- Number of training epochs

## 3. Weighted edges

A plain adjacency list of (row, col) pairs drops any edge weight/value a
matrix cell can hold. Whenever the underlying matrix is weighted (not
just 0/1), the adjacency list must store (row, col, value) triples so no
information is lost in the conversion. Round-trip correctness tests
(convert to list, convert back) must check that the *exact weighted
matrix* is recovered, not just the same connectivity pattern.

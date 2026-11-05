# Formalization: Any Matrix → Adjacency Matrix → Adjacency List

CS 180 Individual Study, Pranavi Immanni
Code: `src/conversion/general_converter.py`  Tests: `tests/test_general_converter.py`

## 1. Goal

Given any real matrix A, define a graph that carries all of A's information,
store that graph as an adjacency list, and recover A exactly from the list.

Conventions: A has shape m × n and entries in a numeric type (float32, float64,
integer). An entry equal to 0 means "no edge". Edge weight = entry value.

## 2. Definitions

**Graph of a matrix, direct case (m = n, A already a graph).**
Nodes V = {0, …, n−1}. There is a directed edge i → j with weight A[i,j]
whenever A[i,j] ≠ 0. The adjacency matrix is A itself.

**Graph of a matrix, bipartite case (any m × n).**
Nodes V = R ∪ C with R = {0, …, m−1} ("row nodes") and C = {m, …, m+n−1}
("column nodes"). For every A[i,j] ≠ 0 there is an edge i → m+j with weight
A[i,j]. Two variants:

- *Directed:* store only i → m+j.
- *Symmetric:* also store m+j → i with the same weight (an undirected graph).

The (m+n) × (m+n) adjacency matrix of the symmetric variant is the block matrix

    B = [ 0    A ]
        [ A^T  0 ]

**Adjacency list.** A map from each node v to the list of pairs
(u, w) with an edge v → u of weight w, plus metadata (mode, shape, dtype,
symmetric flag) needed to invert the construction.

**Choice of mode.** "auto" uses the direct case when m = n and the bipartite
case otherwise. Any square matrix may also be forced into the bipartite case
(useful when A is a feature or rating matrix rather than a graph).

## 3. Conversion procedure

Forward (`matrix_to_graph_list`): scan A for nonzero entries; for each
nonzero (i, j, w) append (j, w) to node i (direct) or (m+j, w) to node i
(bipartite), and additionally (i, w) to node m+j if symmetric.

Inverse (`graph_list_to_matrix`): allocate an m × n zero matrix of the stored
dtype; for each stored edge write its weight back (direct: A[i,j] = w;
bipartite: read only row-node lists, A[i, node−m] = w).

## 4. Correctness

**Claim (lossless round trip).** For every matrix A, inverse(forward(A)) = A,
with identical shape, dtype and values.

*Argument.* Each nonzero entry (i, j, w) produces exactly one forward edge, at a
location determined only by (i, j), so the map from nonzero entries to forward
edges is injective and the inverse reads each one back to the same position
with the same weight. Every other position is never written, so it keeps the
initial value 0, which matches A because those entries were zero. In the
symmetric variant the reverse edges are never read by the inverse, so they do
not affect the result. Shape and dtype are stored in the metadata. ∎

**Caveats.**
- A stored weight must round-trip through a Python number. float32, float64
  and integer values do (float32 widens exactly to double and narrows back).
- Entries equal to 0 are indistinguishable from "no edge". This is lossless for
  the matrix, but a graph that intends an explicit zero-weight edge would lose
  that connection.

**Empirical checks** (`tests/test_general_converter.py`, 19 groups, all pass):
identity, symmetric, all-zero, fully dense, 1×1, tall, wide, row and column
vectors, negative weights, integer and float64 dtypes, forced bipartite mode,
symmetric variant, 90 random cases (5 shapes × 3 sparsities × 3 seeds × 2
variants), bipartite structure (no edges inside the row side or column side,
every forward edge has a matching reverse edge), and rejection of invalid input.

## 5. Spectral note (why the block matrix is the right embedding)

The eigenvalues of B = [[0, A], [A^T, 0]] are ±σ_i, where σ_i are the singular
values of A, plus |m − n| zeros. The symmetric bipartite graph therefore keeps
the spectral information of A, which is why graph methods can be applied to
rectangular data through it. The test suite verifies this numerically.

## 6. Cost

Let nnz be the number of nonzeros of A and use COO-style accounting
(row index + column index + value per stored entry, 4 bytes each).

| Representation | Words stored |
| --- | --- |
| Dense A | m · n |
| List, direct or bipartite directed | 3 · nnz |
| List, bipartite symmetric | 6 · nnz |
| Dense bipartite adjacency (the matrix B) | (m + n)² |

Break-even against dense A: directed list wins when density nnz/(m·n) < 1/3,
symmetric list when density < 1/6. Example from the test run (200 × 500,
95% sparse, float32): dense A is 400,000 bytes, directed list 59,820 bytes
(85.05% saved), symmetric list 119,640 bytes (70.09% saved). Storing B densely
would take 1,960,000 bytes, which is why the bipartite embedding is only
practical in list form.

Conversion time is O(m·n) to scan a dense matrix and O(nnz) to build the lists;
the inverse is O(nnz) plus O(m·n) to allocate the output.

## 7. Limits and next steps

- Only 2-D matrices are covered (no tensors or hypergraph incidence arrays).
- The lists are Python dicts, so real memory use is higher than the COO-style
  byte count above. The byte counts describe the representation, not the
  Python objects.
- The dict-of-lists format is a reference implementation. For GPU work, convert
  to PyTorch COO/CSR (as in the team's GCN experiments).
- Planned: list multiplication for rectangular operands (A is m × n, B is n × p).

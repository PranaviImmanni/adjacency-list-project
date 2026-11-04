"""
General matrix -> graph adjacency list conversion
===================================================
Converts ANY 2-D matrix A (square or rectangular) into an adjacency-list
graph, and back, with no loss of values.

Two modes (see docs/formalization.md for the math):

  "direct"    A is m x m and is already a graph (A[i, j] = weight of edge
              i -> j). Nodes are 0..m-1.

  "bipartite" A is m x n (any shape). Rows become m "row nodes" (ids 0..m-1)
              and columns become n "column nodes" (ids m..m+n-1). Each
              nonzero A[i, j] becomes the edge  i -> m + j  with weight
              A[i, j]. If symmetric=True the reverse edge m + j -> i is also
              stored, giving the undirected bipartite graph whose adjacency
              matrix is  [[0, A], [A^T, 0]].

"auto" picks "direct" for square matrices and "bipartite" otherwise.

Run the tests with:
    python tests/test_general_converter.py
"""

import numpy as np


def matrix_to_graph_list(A, mode="auto", symmetric=False):
    """Convert a 2-D matrix to a graph adjacency list.

    Args:
        A: 2-D array-like (NumPy array or CPU torch tensor).
        mode: "auto", "direct", or "bipartite".
        symmetric: only used in bipartite mode; also store reverse edges.

    Returns:
        dict with keys:
          mode, shape (m, n), dtype, symmetric, num_nodes,
          adj_list: {node: [(neighbor, weight), ...]}
        The extra metadata is what makes exact inversion possible.
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got {A.ndim}-D")
    m, n = A.shape

    if mode == "auto":
        mode = "direct" if m == n else "bipartite"
    if mode not in ("direct", "bipartite"):
        raise ValueError(f"unknown mode: {mode}")
    if mode == "direct" and m != n:
        raise ValueError("direct mode needs a square matrix; use bipartite")

    num_nodes = m if mode == "direct" else m + n
    adj_list = {v: [] for v in range(num_nodes)}
    rows, cols = np.nonzero(A)  # row-major order, so neighbor lists are sorted

    for i, j in zip(rows.tolist(), cols.tolist()):
        w = A[i, j].item()
        if mode == "direct":
            adj_list[i].append((j, w))
        else:
            adj_list[i].append((m + j, w))
            if symmetric:
                adj_list[m + j].append((i, w))

    return {
        "mode": mode,
        "shape": (m, n),
        "dtype": str(A.dtype),
        "symmetric": bool(symmetric and mode == "bipartite"),
        "num_nodes": num_nodes,
        "adj_list": adj_list,
    }


def graph_list_to_matrix(graph):
    """Exact inverse of matrix_to_graph_list. Returns a NumPy array with the
    original shape and dtype."""
    m, n = graph["shape"]
    A = np.zeros((m, n), dtype=np.dtype(graph["dtype"]))
    adj = graph["adj_list"]

    if graph["mode"] == "direct":
        for i, neighbors in adj.items():
            for j, w in neighbors:
                A[i, j] = w
    else:
        # Row nodes 0..m-1 hold the forward edges i -> m + j. Reverse edges
        # (when symmetric) live on column nodes and are redundant.
        for i in range(m):
            for node, w in adj[i]:
                A[i, node - m] = w
    return A


def count_stored_entries(graph):
    """Number of (neighbor, weight) entries actually stored in the list."""
    return sum(len(v) for v in graph["adj_list"].values())


def storage_comparison(graph, bytes_per_value=4, bytes_per_index=4):
    """Compare the dense original matrix against the list, COO-style
    (row index + col index + value per stored entry).

    Also reports the dense bipartite adjacency matrix (size (m+n)^2) that you
    would need if you stored the bipartite graph as a matrix instead.
    """
    m, n = graph["shape"]
    entries = count_stored_entries(graph)
    dense_original = m * n * bytes_per_value
    list_bytes = entries * (2 * bytes_per_index + bytes_per_value)
    out = {
        "stored_entries": entries,
        "dense_original_bytes": dense_original,
        "list_bytes": list_bytes,
        "saved_pct_vs_original": round(
            (dense_original - list_bytes) / dense_original * 100, 2
        ) if dense_original else 0.0,
    }
    if graph["mode"] == "bipartite":
        N = m + n
        out["dense_bipartite_adjacency_bytes"] = N * N * bytes_per_value
    return out
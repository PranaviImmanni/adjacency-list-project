"""
Memory measurement and ML-result equivalence utilities.

Implements the methodology in docs/memory_eval_spec.md:
  1. Structural memory: theoretical byte cost of a representation.
  2. Empirical memory: measured byte cost during actual use.
  3. Equivalence checks: prediction match, accuracy parity, output closeness.

These are meant to be validated on small toy matrices first (see the demo
at the bottom), then reused unchanged on real datasets later.
"""

import tracemalloc
import numpy as np


# ---------------------------------------------------------------------------
# 1a. Structural memory
# ---------------------------------------------------------------------------

def structural_memory_adj_matrix(n_nodes, bytes_per_element=4):
    """Theoretical byte cost of a dense N x N adjacency matrix.

    Args:
        n_nodes: number of nodes (N).
        bytes_per_element: e.g. 4 for float32/int32, 8 for float64/int64,
            1 for a boolean/bit-packed-per-byte matrix.
    """
    return n_nodes * n_nodes * bytes_per_element


def structural_memory_adj_list(n_edges, index_bytes=4, value_bytes=4,
                                store_values=True):
    """Theoretical byte cost of a COO-style adjacency list.

    Each edge stores a (row, col[, value]) triple.

    Args:
        n_edges: number of edges (E).
        index_bytes: bytes per row/col index (e.g. 4 for int32).
        value_bytes: bytes per edge value/weight.
        store_values: set False for an unweighted graph (row, col only).
    """
    cost = 2 * n_edges * index_bytes
    if store_values:
        cost += n_edges * value_bytes
    return cost


def structural_memory_from_array(arr):
    """Exact byte cost of an actual NumPy/PyTorch array, for validation
    against the theoretical formulas above."""
    return arr.nbytes


# ---------------------------------------------------------------------------
# 1b. Empirical memory
# ---------------------------------------------------------------------------

def measure_peak_memory_cpu(func, *args, **kwargs):
    """Run func(*args, **kwargs) and return (result, peak_bytes_allocated)
    using tracemalloc. Use for CPU/pure-Python code paths.
    """
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak


def measure_peak_memory_torch(func, *args, device="cuda", **kwargs):
    """Run func(*args, **kwargs) and return (result, peak_bytes_allocated)
    using torch.cuda's memory stats. Only meaningful on a CUDA device.
    """
    import torch
    torch.cuda.reset_peak_memory_stats(device)
    result = func(*args, **kwargs)
    peak = torch.cuda.max_memory_allocated(device)
    return result, peak


# ---------------------------------------------------------------------------
# 2. Equivalence checks
# ---------------------------------------------------------------------------

def prediction_match_rate(preds_a, preds_b):
    """Fraction of predictions that match exactly between two models."""
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    assert preds_a.shape == preds_b.shape, "prediction arrays must be the same shape"
    return float(np.mean(preds_a == preds_b))


def accuracy_parity(acc_a, acc_b, tolerance=0.005):
    """Check whether two accuracy values are within `tolerance` (default 0.5%).

    Returns (within_tolerance: bool, abs_difference: float).
    """
    diff = abs(acc_a - acc_b)
    return diff <= tolerance, diff


def output_closeness(out_a, out_b, metric="max_abs_diff"):
    """Compare raw logits/embeddings between two models.

    metric:
        "max_abs_diff"     -> largest elementwise absolute difference
        "cosine_similarity" -> average cosine similarity across rows
    """
    out_a = np.asarray(out_a, dtype=np.float64)
    out_b = np.asarray(out_b, dtype=np.float64)
    assert out_a.shape == out_b.shape, "output arrays must be the same shape"

    if metric == "max_abs_diff":
        return float(np.max(np.abs(out_a - out_b)))
    elif metric == "cosine_similarity":
        num = np.sum(out_a * out_b, axis=-1)
        denom = np.linalg.norm(out_a, axis=-1) * np.linalg.norm(out_b, axis=-1)
        denom = np.where(denom == 0, 1e-12, denom)  # avoid div-by-zero
        return float(np.mean(num / denom))
    else:
        raise ValueError(f"unknown metric: {metric}")


# ---------------------------------------------------------------------------
# Demo: validate the measurement pipeline on a small toy matrix
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # A small 5-node sparse adjacency matrix (toy example).
    toy_adj = np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0],
    ], dtype=np.int32)

    n_nodes = toy_adj.shape[0]
    n_edges = int(np.count_nonzero(toy_adj) / 2)  # undirected, each edge counted twice

    print("=== Structural memory (theoretical) ===")
    mat_bytes = structural_memory_adj_matrix(n_nodes, bytes_per_element=4)
    list_bytes = structural_memory_adj_list(n_edges, index_bytes=4, value_bytes=4,
                                             store_values=False)
    print(f"Adjacency matrix ({n_nodes}x{n_nodes}): {mat_bytes} bytes")
    print(f"Adjacency list ({n_edges} edges):       {list_bytes} bytes")

    print("\n=== Structural memory (actual arrays, for validation) ===")
    print(f"toy_adj.nbytes: {structural_memory_from_array(toy_adj)} bytes")
    rows, cols = np.nonzero(toy_adj)
    coo = np.stack([rows, cols], axis=1).astype(np.int32)
    print(f"COO array.nbytes: {structural_memory_from_array(coo)} bytes")

    print("\n=== Empirical memory (tracemalloc) ===")
    def build_dense_copy(m):
        return m.copy()
    _, peak = measure_peak_memory_cpu(build_dense_copy, toy_adj)
    print(f"Peak memory to copy dense matrix: {peak} bytes")

    print("\n=== Equivalence checks (toy example) ===")
    preds_a = np.array([0, 1, 1, 0, 1])
    preds_b = np.array([0, 1, 1, 0, 1])  # identical, should be 100% match
    print(f"Prediction match rate: {prediction_match_rate(preds_a, preds_b):.3f}")

    ok, diff = accuracy_parity(0.812, 0.809, tolerance=0.005)
    print(f"Accuracy parity within tolerance: {ok} (diff={diff:.4f})")

    out_a = np.random.randn(5, 3)
    out_b = out_a + 1e-6  # tiny numerical noise, simulating dense vs. sparse ops
    print(f"Max abs diff in outputs: {output_closeness(out_a, out_b, 'max_abs_diff'):.2e}")
    print(f"Cosine similarity of outputs: {output_closeness(out_a, out_b, 'cosine_similarity'):.6f}")

"""
Tests for src/conversion/general_converter.py

Checks that matrix -> graph list -> matrix recovers the ORIGINAL matrix
exactly (same shape, dtype, and values) for square and rectangular inputs,
in both modes, plus structural checks on the bipartite embedding.

Run from the project root:
    python tests/test_general_converter.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.conversion.general_converter import (
    matrix_to_graph_list,
    graph_list_to_matrix,
    count_stored_entries,
    storage_comparison,
)


def random_sparse(m, n, sparsity, seed, dtype=np.float32):
    rng = np.random.default_rng(seed)
    vals = rng.random((m, n))
    A = np.where(vals > sparsity, vals, 0.0)
    return A.astype(dtype)


def check_round_trip(name, A, **kwargs):
    g = matrix_to_graph_list(A, **kwargs)
    B = graph_list_to_matrix(g)
    ok = (B.shape == A.shape) and (B.dtype == A.dtype) and np.array_equal(A, B)
    print(f"{name:<42} | {str(A.shape):>12} | {g['mode']:>9} | "
          f"{count_stored_entries(g):>7} | {'PASS' if ok else 'FAIL'}")
    return ok


def run_all():
    results = []
    print("=" * 88)
    print(f"{'case':<42} | {'shape':>12} | {'mode':>9} | {'entries':>7} | result")
    print("-" * 88)

    # --- Hand-picked structural cases -------------------------------------
    results.append(check_round_trip("identity 5x5", np.eye(5, dtype=np.float32)))
    sym = np.array([[0, 2, 0], [2, 0, 3], [0, 3, 0]], dtype=np.float32)
    results.append(check_round_trip("symmetric weighted 3x3", sym))
    results.append(check_round_trip("all-zero 4x4", np.zeros((4, 4), dtype=np.float32)))
    results.append(check_round_trip("fully dense 4x4", np.arange(1, 17, dtype=np.float32).reshape(4, 4)))
    results.append(check_round_trip("1x1", np.array([[7.5]], dtype=np.float32)))
    results.append(check_round_trip("tall 6x2", random_sparse(6, 2, 0.4, 1)))
    results.append(check_round_trip("wide 2x6", random_sparse(2, 6, 0.4, 2)))
    results.append(check_round_trip("row vector 1x9", random_sparse(1, 9, 0.3, 3)))
    results.append(check_round_trip("column vector 9x1", random_sparse(9, 1, 0.3, 4)))
    results.append(check_round_trip("negative weights", np.array([[0, -1.5], [2.25, 0], [0, -3]], dtype=np.float32)))
    results.append(check_round_trip("integer matrix", np.array([[0, 3, 0], [4, 0, 0]], dtype=np.int64)))
    results.append(check_round_trip("float64 weights", random_sparse(5, 7, 0.5, 5, np.float64)))

    # --- Forced modes ------------------------------------------------------
    sq = random_sparse(6, 6, 0.6, 6)
    results.append(check_round_trip("square forced bipartite", sq, mode="bipartite"))
    results.append(check_round_trip("square bipartite symmetric", sq, mode="bipartite", symmetric=True))
    results.append(check_round_trip("rect bipartite symmetric", random_sparse(5, 8, 0.7, 7), symmetric=True))

    # --- Random sweep: shapes x sparsities x seeds -------------------------
    sweep_pass = True
    count = 0
    for (m, n) in [(10, 10), (30, 12), (12, 30), (100, 100), (64, 200)]:
        for s in [0.5, 0.8, 0.95]:
            for seed in [42, 7, 123]:
                A = random_sparse(m, n, s, seed)
                for sym_flag in (False, True):
                    g = matrix_to_graph_list(A, symmetric=sym_flag)
                    B = graph_list_to_matrix(g)
                    sweep_pass &= np.array_equal(A, B) and B.dtype == A.dtype
                    count += 1
    print(f"{'random sweep (' + str(count) + ' cases)':<42} | {'various':>12} | {'auto':>9} | {'-':>7} | "
          f"{'PASS' if sweep_pass else 'FAIL'}")
    results.append(sweep_pass)

    # --- Structural checks on the bipartite embedding ----------------------
    A = random_sparse(5, 8, 0.6, 11)
    m, n = A.shape
    g = matrix_to_graph_list(A, symmetric=True)
    nnz = int(np.count_nonzero(A))
    struct_ok = True
    struct_ok &= g["num_nodes"] == m + n
    struct_ok &= count_stored_entries(g) == 2 * nnz            # edge + reverse edge
    # no edges inside the row side or inside the column side
    for v, nbrs in g["adj_list"].items():
        for u, _ in nbrs:
            struct_ok &= (v < m) != (u < m)
    # symmetry: every i -> m+j has a matching m+j -> i with the same weight
    for i in range(m):
        for node, w in g["adj_list"][i]:
            struct_ok &= (i, w) in g["adj_list"][node]
    print(f"{'bipartite structure (no intra-side edges)':<42} | {str(A.shape):>12} | {'bipartite':>9} | "
          f"{count_stored_entries(g):>7} | {'PASS' if struct_ok else 'FAIL'}")
    results.append(struct_ok)

    # --- Spectral sanity check: eigenvalues of [[0,A],[A^T,0]] = +/- singular values
    big = np.zeros((m + n, m + n))
    big[:m, m:] = A
    big[m:, :m] = A.T
    # eigenvalues come in +sigma / -sigma pairs, so the largest min(m, n)
    # signed eigenvalues are exactly the singular values
    eig = np.sort(np.linalg.eigvalsh(big))[::-1][: min(m, n)]
    sv = np.sort(np.linalg.svd(A.astype(np.float64), compute_uv=False))[::-1]
    spec_ok = np.allclose(eig, sv, atol=1e-6)
    print(f"{'eigenvalues of block matrix = singular values':<42} | {str(A.shape):>12} | {'-':>9} | "
          f"{'-':>7} | {'PASS' if spec_ok else 'FAIL'}")
    results.append(spec_ok)

    # --- Error handling ----------------------------------------------------
    err_ok = True
    try:
        matrix_to_graph_list(np.zeros((3, 4)), mode="direct")
        err_ok = False
    except ValueError:
        pass
    try:
        matrix_to_graph_list(np.zeros((2, 2, 2)))
        err_ok = False
    except ValueError:
        pass
    print(f"{'rejects bad inputs':<42} | {'-':>12} | {'-':>9} | {'-':>7} | {'PASS' if err_ok else 'FAIL'}")
    results.append(err_ok)

    print("-" * 88)
    print(f"Overall: {'ALL PASSED' if all(results) else 'SOME FAILED'} ({sum(results)}/{len(results)} groups)")

    # --- Storage example ---------------------------------------------------
    print("\nStorage example, 200 x 500 matrix at 95% sparsity (float32, int32 indices):")
    A = random_sparse(200, 500, 0.95, 99)
    for sym_flag in (False, True):
        g = matrix_to_graph_list(A, symmetric=sym_flag)
        print(f"  symmetric={sym_flag}: {storage_comparison(g)}")

    return all(results)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
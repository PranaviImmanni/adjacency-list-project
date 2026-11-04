"""
Multiplication Reference Test
==============================
Checks that multiplying two adjacency lists directly gives the same result
as dense matrix multiplication, across sizes, sparsities and seeds.

Ground truth is a float64 dense product (A.double() @ B.double()). The list
code accumulates in Python doubles, so it should agree with float64 up to the
rounding of its float32 output matrix. PyTorch's own float32 matmul error is
reported alongside for context.

Pass criterion per case:
    |list - ref64| <= atol + rtol * |ref64|   elementwise
    rtol = 1e-6  (float32 output rounding is about 6e-8 relative)
    atol = 2e-6  (multiply_adj_lists drops entries with |value| <= 1e-6)

Run from the project root:
    python tests/test_multiply.py
"""

import sys
import csv
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.synthetic.generator import generate_synthetic_graph
from src.conversion.converter import matrix_to_adj_list, adj_list_to_matrix
from src.operations.multiply import multiply_adj_lists

NODE_SIZES = [5, 10, 50, 100, 250]
SPARSITIES = [0.5, 0.8, 0.95]
SEED_PAIRS = [(42, 99), (7, 55), (123, 11)]
RTOL = 1e-6
ATOL = 2e-6

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def run_multiply_tests():
    print("\n" + "=" * 100)
    print("  MULTIPLICATION REFERENCE TEST: adjacency-list A x B vs float64 dense A @ B")
    print("=" * 100)
    print(f"{'Nodes':>6} | {'Sparsity':>8} | {'Seeds':>9} | {'List vs fp64':>13} | "
          f"{'Torch fp32 vs fp64':>19} | {'List vs fp32':>13} | {'Result':>6}")
    print("-" * 100)

    rows = []
    all_passed = True

    for n in NODE_SIZES:
        for s in SPARSITIES:
            for seed_a, seed_b in SEED_PAIRS:
                A = generate_synthetic_graph(num_nodes=n, sparsity=s, weighted=True, seed=seed_a)
                B = generate_synthetic_graph(num_nodes=n, sparsity=s, weighted=True, seed=seed_b)

                ref64 = torch.matmul(A.double(), B.double())   # ground truth
                ref32 = torch.matmul(A, B)                      # PyTorch float32

                list_C = multiply_adj_lists(matrix_to_adj_list(A), matrix_to_adj_list(B), num_nodes=n)
                result = adj_list_to_matrix(list_C, num_nodes=n)

                err_list = (result.double() - ref64).abs().max().item()
                err_t32 = (ref32.double() - ref64).abs().max().item()
                err_vs32 = (result - ref32).abs().max().item()
                ok = torch.allclose(result.double(), ref64, rtol=RTOL, atol=ATOL)
                all_passed &= ok

                print(f"{n:>6} | {int(s * 100):>7}% | {seed_a:>4},{seed_b:<4} | {err_list:>13.2e} | "
                      f"{err_t32:>19.2e} | {err_vs32:>13.2e} | {'PASS' if ok else 'FAIL':>6}")

                rows.append({
                    "nodes": n, "sparsity": s, "seed_a": seed_a, "seed_b": seed_b,
                    "list_vs_fp64": f"{err_list:.2e}",
                    "torch_fp32_vs_fp64": f"{err_t32:.2e}",
                    "list_vs_torch_fp32": f"{err_vs32:.2e}",
                    "result": "PASS" if ok else "FAIL",
                })

    print("-" * 100)
    print(f"  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED, check above'} "
          f"({sum(r['result'] == 'PASS' for r in rows)}/{len(rows)} cases)")

    csv_path = REPORTS_DIR / "multiply_reference_test.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved -> {csv_path}\n")
    return all_passed


if __name__ == "__main__":
    ok = run_multiply_tests()
    sys.exit(0 if ok else 1)
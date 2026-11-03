"""
Benchmark: Dense Adjacency Matrix vs. Adjacency List
=====================================================
Produces:
  1. Memory comparison table (console + CSV)
  2. Memory curve plot saved to reports/memory_curve.png
  3. Round-trip correctness summary across sizes and seeds

Run from the project root:
    python tests/benchmark.py
"""

import sys
import csv
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

# Make sure project root is on the path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.synthetic.generator import generate_synthetic_graph
from src.conversion.converter import matrix_to_adj_list, adj_list_to_matrix
from src.measurement.memory_utils import (
    structural_memory_adj_matrix,
    structural_memory_adj_list,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NODE_SIZES   = [50, 100, 250, 500, 1000, 2000]   # graph sizes to benchmark
SPARSITIES   = [0.5, 0.8, 0.95]                   # fraction of zeros
SEEDS        = [42, 7, 123]                        # for correctness checks
BYTES_ELEM   = 4                                   # float32

REPORTS_DIR  = Path(__file__).resolve().parents[1] / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Memory comparison table
# ---------------------------------------------------------------------------

def run_memory_benchmark():
    print("\n" + "=" * 78)
    print("  MEMORY BENCHMARK: Dense Matrix vs. Adjacency List")
    print("=" * 78)
    header = f"{'Nodes':>6} | {'Sparsity':>8} | {'Edges':>8} | {'Dense (KB)':>10} | {'List (KB)':>10} | {'Saved (%)':>9}"
    print(header)
    print("-" * 78)

    rows = []

    for n in NODE_SIZES:
        for s in SPARSITIES:
            matrix = generate_synthetic_graph(num_nodes=n, sparsity=s, weighted=True)
            n_edges = int((matrix != 0).sum().item())

            dense_bytes = structural_memory_adj_matrix(n, bytes_per_element=BYTES_ELEM)
            list_bytes  = structural_memory_adj_list(n_edges, index_bytes=4,
                                                     value_bytes=4, store_values=True)
            dense_kb    = dense_bytes / 1024
            list_kb     = list_bytes  / 1024
            saved_pct   = (dense_bytes - list_bytes) / dense_bytes * 100

            print(f"{n:>6} | {int(s*100):>7}% | {n_edges:>8} | "
                  f"{dense_kb:>10.2f} | {list_kb:>10.2f} | {saved_pct:>8.1f}%")

            rows.append({
                "nodes": n, "sparsity": s, "edges": n_edges,
                "dense_kb": round(dense_kb, 2),
                "list_kb":  round(list_kb,  2),
                "saved_pct": round(saved_pct, 2),
            })

    # Save to CSV
    csv_path = REPORTS_DIR / "memory_benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved -> {csv_path}")

    return rows


# ---------------------------------------------------------------------------
# 2. Memory curve plot
# ---------------------------------------------------------------------------

def plot_memory_curves(rows):
    fig, axes = plt.subplots(1, len(SPARSITIES), figsize=(14, 5), sharey=False)
    fig.suptitle(
        "Memory: Dense Adjacency Matrix vs. Adjacency List\n"
        "(float32, undirected graph, edges stored with weights)",
        fontsize=12, y=1.02
    )

    for ax, s in zip(axes, SPARSITIES):
        subset = [r for r in rows if r["sparsity"] == s]
        nodes   = [r["nodes"]    for r in subset]
        dense   = [r["dense_kb"] for r in subset]
        lst     = [r["list_kb"]  for r in subset]

        ax.plot(nodes, dense, marker="o", label="Dense matrix", color="#e74c3c", linewidth=2)
        ax.plot(nodes, lst,   marker="s", label="Adjacency list", color="#2ecc71", linewidth=2)
        ax.set_title(f"Sparsity {int(s*100)}%")
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("Memory (KB)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Annotate the savings at the largest size
        last = subset[-1]
        ax.annotate(
            f"{last['saved_pct']:.0f}% saved",
            xy=(last["nodes"], last["list_kb"]),
            xytext=(-60, 20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=9, color="#2ecc71"
        )

    plt.tight_layout()
    plot_path = REPORTS_DIR / "memory_curve.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 3. Round-trip correctness summary
# ---------------------------------------------------------------------------

def run_correctness_checks():
    print("\n" + "=" * 78)
    print("  ROUND-TRIP CORRECTNESS: matrix -> adj list -> matrix")
    print("=" * 78)
    print(f"{'Nodes':>6} | {'Sparsity':>8} | {'Seed':>6} | {'Edges':>8} | {'Exact Match':>12}")
    print("-" * 78)

    all_passed = True

    for n in [50, 100, 500, 1000]:
        for s in SPARSITIES:
            for seed in SEEDS:
                matrix       = generate_synthetic_graph(num_nodes=n, sparsity=s,
                                                        weighted=True, seed=seed)
                adj_list     = matrix_to_adj_list(matrix)
                reconstructed = adj_list_to_matrix(adj_list, num_nodes=n)
                match        = torch.allclose(matrix, reconstructed)
                status       = "PASS" if match else "FAIL"
                if not match:
                    all_passed = False

                print(f"{n:>6} | {int(s*100):>7}% | {seed:>6} | "
                      f"{int((matrix != 0).sum()):>8} | {status:>12}")

    print("-" * 78)
    overall = "ALL PASSED" if all_passed else "SOME FAILED — check above"
    print(f"  Overall: {overall}")
    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rows = run_memory_benchmark()
    print("\n  Generating memory curve plot...")
    plot_memory_curves(rows)
    run_correctness_checks()
    print("\nDone. Check the reports/ folder for outputs.\n")
"""
Edge Pruning Heuristics
========================
Decides which edges to keep or remove when converting a dense matrix
to an adjacency list. Rather than removing edges randomly (like sparsity
does in the generator), these heuristics use the actual edge values to
make smarter decisions about what to keep.

Current heuristics:
  1. Weight threshold  — keep edges whose weight is above a cutoff value
  2. Top-K per node   — keep only the K strongest edges per node
  3. Percentile       — keep edges above the Nth percentile of all weights

Run from project root:
    python src/conversion/pruning.py
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Heuristic 1: Weight threshold
# ---------------------------------------------------------------------------

def prune_by_weight_threshold(matrix: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Keep only edges whose weight is strictly above `threshold`.
    Everything at or below the threshold is zeroed out.

    This is the simplest heuristic: strong connections are assumed to
    carry more information, so weak ones are dropped first.

    Args:
        matrix:    dense N x N adjacency matrix (weighted)
        threshold: minimum edge weight to keep (e.g. 0.5 keeps top half)

    Returns:
        pruned dense matrix (same shape, zeros where edges were removed)
    """
    return torch.where(matrix > threshold, matrix, torch.zeros_like(matrix))


# ---------------------------------------------------------------------------
# Heuristic 2: Top-K edges per node
# ---------------------------------------------------------------------------

def prune_top_k_per_node(matrix: torch.Tensor, k: int) -> torch.Tensor:
    """
    For each node (row), keep only its K strongest outgoing edges.
    All other edges for that node are zeroed out.

    Useful when nodes have very different degree distributions — a node
    with 100 connections loses less information dropping 90 of them than
    a node with only 5 connections dropping 4.

    Args:
        matrix: dense N x N adjacency matrix (weighted)
        k:      number of edges to keep per node

    Returns:
        pruned dense matrix
    """
    n = matrix.shape[0]
    pruned = torch.zeros_like(matrix)

    for i in range(n):
        row = matrix[i]
        nonzero_indices = row.nonzero(as_tuple=True)[0]

        if len(nonzero_indices) == 0:
            continue

        # If fewer than k nonzero edges exist, keep all of them
        if len(nonzero_indices) <= k:
            pruned[i] = row
        else:
            # Pick the top-k by weight
            top_k_vals, top_k_idx = torch.topk(row, k)
            pruned[i][top_k_idx] = top_k_vals

    return pruned


# ---------------------------------------------------------------------------
# Heuristic 3: Percentile threshold
# ---------------------------------------------------------------------------

def prune_by_percentile(matrix: torch.Tensor, percentile: float) -> torch.Tensor:
    """
    Keep only edges above the Nth percentile of all nonzero edge weights.

    Unlike a fixed threshold, this adapts to whatever weight scale the
    matrix uses — so a percentile of 80 always keeps the top 20% of
    edges regardless of what the actual weight values are.

    Args:
        matrix:     dense N x N adjacency matrix (weighted)
        percentile: 0-100, edges below this percentile are removed
                    e.g. 80 keeps the top 20% strongest edges

    Returns:
        pruned dense matrix
    """
    nonzero_vals = matrix[matrix != 0]

    if len(nonzero_vals) == 0:
        return matrix.clone()

    cutoff = torch.quantile(nonzero_vals, percentile / 100.0)
    return torch.where(matrix > cutoff, matrix, torch.zeros_like(matrix))


# ---------------------------------------------------------------------------
# Helper: summarize what a pruning removed
# ---------------------------------------------------------------------------

def pruning_summary(original: torch.Tensor, pruned: torch.Tensor) -> dict:
    """
    Returns a dict summarising how much the pruning removed.
    """
    original_edges = int((original != 0).sum().item())
    pruned_edges   = int((pruned   != 0).sum().item())
    removed        = original_edges - pruned_edges
    pct_removed    = removed / original_edges * 100 if original_edges > 0 else 0.0

    return {
        "original_edges": original_edges,
        "pruned_edges":   pruned_edges,
        "edges_removed":  removed,
        "pct_removed":    round(pct_removed, 1),
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.synthetic.generator import generate_synthetic_graph

    matrix = generate_synthetic_graph(num_nodes=8, sparsity=0.3, weighted=True, seed=42)
    print("Original matrix:")
    print(matrix)
    print(f"Total edges: {int((matrix != 0).sum().item())}\n")

    # Heuristic 1: weight threshold
    pruned_thresh = prune_by_weight_threshold(matrix, threshold=0.5)
    s1 = pruning_summary(matrix, pruned_thresh)
    print(f"After weight threshold (> 0.5): {s1}")

    # Heuristic 2: top-2 per node
    pruned_topk = prune_top_k_per_node(matrix, k=2)
    s2 = pruning_summary(matrix, pruned_topk)
    print(f"After top-2 per node:           {s2}")

    # Heuristic 3: 75th percentile
    pruned_pct = prune_by_percentile(matrix, percentile=75)
    s3 = pruning_summary(matrix, pruned_pct)
    print(f"After 75th percentile cutoff:   {s3}")
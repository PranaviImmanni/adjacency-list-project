"""Edge pruning for the frozen-checkpoint experiments (docs/pruning_and_error_analysis.md).

The removable edges are the graph's unique undirected pairs (u, v) with u < v,
self-loops excluded. A method produces one ordering of those pairs, and a
removal fraction f removes the first floor(f * E) pairs of it. Every fraction
uses the same ordering, so the removed sets are nested: whatever is removed at
10% is also removed at 25%, 50% and 75%.

The retained pairs go back through build_normalized_coo, which restores both
directions, adds exactly one self-loop per node, and recomputes the degrees and
the normalization. Nodes, features, labels and split masks are never touched.

Methods:
  none        the original graph (removal fraction must be 0)
  random      a seeded random permutation of the pairs
  similarity  pairs sorted by the cosine similarity of their endpoints' feature
              vectors, least similar first. Scores only the existing edges, in
              chunks, never an N x N matrix. Uses no labels or split masks.
"""

import hashlib
import math
import time
from fractions import Fraction
from pathlib import Path

import torch

METHODS = ("none", "random", "similarity")
REMOVAL_FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75)
RANDOM_SEEDS = (101, 102, 103)
DEFAULT_CHUNK_SIZE = 4096


def unique_undirected_edges(edge_index, num_nodes):
    """Unique undirected pairs as a [2, E] long tensor with u < v, sorted by (u, v).

    Direction and duplicates are ignored and self-loops are dropped, so each
    undirected edge appears exactly once whatever form edge_index is in.
    """
    edge_index = torch.as_tensor(edge_index).to(device="cpu", dtype=torch.long)
    u = torch.minimum(edge_index[0], edge_index[1])
    v = torch.maximum(edge_index[0], edge_index[1])
    keep = u != v
    keys = torch.unique(u[keep] * num_nodes + v[keep])  # sorted, so (u, v) order
    return torch.stack([keys // num_nodes, keys % num_nodes])


def num_to_remove(removal_fraction, num_edges):
    """floor(removal_fraction * num_edges), exact for decimal fractions such as 0.1."""
    fraction = Fraction(str(removal_fraction))
    if not 0 <= fraction <= 1:
        raise ValueError(f"removal fraction must be in [0, 1], got {removal_fraction}")
    return math.floor(fraction * num_edges)


def random_edge_order(num_edges, seed):
    """A seeded random permutation of edge positions. Never touches the global RNG."""
    return torch.randperm(num_edges, generator=torch.Generator().manual_seed(seed))


def cosine_edge_similarity(x, edges, chunk_size=DEFAULT_CHUNK_SIZE):
    """Cosine similarity of the feature vectors at both ends of each edge, in float64.

    Only the given edges are scored, chunk by chunk. An edge with a zero feature
    vector at either end has no defined cosine and scores exactly 0.0.
    """
    x = torch.as_tensor(x).to(device="cpu", dtype=torch.float64)
    norms = x.norm(dim=1, keepdim=True)
    unit = torch.where(norms > 0, x / torch.where(norms > 0, norms, 1.0), 0.0)
    scores = torch.empty(edges.size(1), dtype=torch.float64)
    for start in range(0, edges.size(1), chunk_size):
        u, v = edges[:, start:start + chunk_size]
        scores[start:start + chunk_size] = (unit[u] * unit[v]).sum(dim=1)
    return scores


def similarity_edge_order(x, edges, chunk_size=DEFAULT_CHUNK_SIZE):
    """Edge positions from least to most similar, with the scores.

    The sort is stable and `edges` is sorted by (u, v), so equal scores are
    broken by (u, v) ascending: the result is fully deterministic.
    """
    scores = cosine_edge_similarity(x, edges, chunk_size)
    return torch.sort(scores, stable=True).indices, scores


def edge_order(method, edges, x=None, seed=None, chunk_size=DEFAULT_CHUNK_SIZE):
    """Returns (order, scores or None) for a pruning method. `none` has no ordering."""
    if method == "none":
        return None, None
    if method == "random":
        if seed is None:
            raise ValueError("random pruning needs a pruning seed")
        return random_edge_order(edges.size(1), seed), None
    if method == "similarity":
        if x is None:
            raise ValueError("similarity pruning needs node features")
        return similarity_edge_order(x, edges, chunk_size)
    raise ValueError(f"unknown pruning method {method!r} (supported: {METHODS})")


def retained_mask(order, num_edges, num_removed):
    """Boolean mask over edge positions: False for the first num_removed positions of order."""
    mask = torch.ones(num_edges, dtype=torch.bool)
    if num_removed:
        mask[order[:num_removed]] = False
    return mask


def graph_statistics(edges, num_nodes):
    """Size of an undirected graph given as unique pairs (u < v, no self-loops)."""
    num_edges = edges.size(1)
    degree = torch.bincount(edges.flatten(), minlength=num_nodes)
    return {
        "num_nodes": num_nodes,
        "unique_undirected_edges": num_edges,
        "normalized_nnz": 2 * num_edges + num_nodes,
        "density_excluding_self_loops": num_edges / (num_nodes * (num_nodes - 1) // 2),
        "isolated_nodes_before_self_loops": int((degree == 0).sum()),
    }


def prune_graph(edge_index, num_nodes, method, removal_fraction, x=None, seed=None,
                chunk_size=DEFAULT_CHUNK_SIZE):
    """Remove floor(removal_fraction * E) unique undirected edges with the given method.

    Returns a dict with the retained pairs ([2, E'] with u < v, ready for
    build_normalized_coo), the retained mask over the original pairs, the
    ordering and scores, counts, graph statistics, and timings in ms
    (`order_ms` = scoring/ordering, `selection_ms` = applying the cut).
    """
    if method == "none" and removal_fraction != 0:
        raise ValueError("method 'none' is the original graph; its removal fraction must be 0")
    edges = unique_undirected_edges(edge_index, num_nodes)
    num_edges = edges.size(1)

    start = time.perf_counter()
    order, scores = edge_order(method, edges, x, seed, chunk_size)
    order_ms = (time.perf_counter() - start) * 1e3

    start = time.perf_counter()
    num_removed = num_to_remove(removal_fraction, num_edges)
    mask = retained_mask(order, num_edges, num_removed)
    retained = edges[:, mask]
    selection_ms = (time.perf_counter() - start) * 1e3

    return {
        "edges": edges,
        "order": order,
        "scores": scores,
        "retained_mask": mask,
        "retained_edges": retained,
        "method": method,
        "seed": seed if method == "random" else None,
        "requested_removal_fraction": removal_fraction,
        "original_unique_edges": num_edges,
        "removed_unique_edges": num_removed,
        "retained_unique_edges": num_edges - num_removed,
        "actual_removal_fraction": num_removed / num_edges if num_edges else 0.0,
        "original_graph": graph_statistics(edges, num_nodes),
        "pruned_graph": graph_statistics(retained, num_nodes),
        "order_ms": order_ms,
        "selection_ms": selection_ms,
    }


def tensor_sha256(t):
    """Short hash of a CPU tensor's bytes, to show two runs saw (or produced) the same data."""
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def save_or_check_ordering(path, pruned):
    """Save the edge ordering once; on later runs, check it is unchanged.

    The file holds the unique pairs (int32), the ordering and, for similarity
    pruning, the scores, so any removal set can be rebuilt offline:
    removed = edges[:, order[:floor(f * E)]]. Returns the ordering hash.
    """
    path = Path(path)
    order_hash = tensor_sha256(pruned["order"])
    if path.exists():
        saved = torch.load(path, weights_only=True)
        if not torch.equal(saved["order"], pruned["order"]):
            raise RuntimeError(f"edge ordering differs from the one saved in {path}; pruning is not reproducible")
        return order_hash
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"method": pruned["method"], "seed": pruned["seed"],
                "edges": pruned["edges"].to(torch.int32), "order": pruned["order"],
                "scores": pruned["scores"]}, path)
    return order_hash

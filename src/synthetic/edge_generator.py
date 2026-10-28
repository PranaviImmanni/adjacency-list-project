"""Uniform random undirected graphs as edge lists (Erdős–Rényi G(n, m)).

Used by the synthetic GCN sweeps. Unlike generator.py, nothing here allocates
an N x N matrix or enumerates all N(N-1)/2 possible pairs: memory is O(m).

Each undirected edge is stored once as (i, j) with i < j, without self-loops.
build_normalized_coo symmetrizes and adds self-loops, so the normalized
adjacency has exactly 2m + N nonzeros (pass num_nodes=N so isolated nodes
are kept).
"""

import torch


def num_possible_edges(num_nodes):
    """Undirected pairs without self-loops: N(N-1)/2."""
    return num_nodes * (num_nodes - 1) // 2


def edges_for_avg_degree(num_nodes, avg_degree):
    """m such that the average number of neighbors (before self-loops) is avg_degree: m = N*d/2."""
    return round(num_nodes * avg_degree / 2)


def edges_for_density(num_nodes, density):
    """m such that m / (N(N-1)/2) = density (a fraction, e.g. 0.01 for 1%)."""
    return round(density * num_possible_edges(num_nodes))


def pair_from_index(keys):
    """Map linear indices k in [0, N(N-1)/2) to pairs (i, j) with i < j.

    Lower-triangle order: k = j(j-1)/2 + i. The float sqrt estimate of j is
    corrected in integer arithmetic, so it is exact for any N whose pair
    count fits in int64.
    """
    keys = torch.as_tensor(keys, dtype=torch.long)
    j = ((1 + torch.sqrt(1 + 8 * keys.double())) / 2).floor().long()
    j = torch.where(j * (j - 1) // 2 > keys, j - 1, j)
    j = torch.where((j + 1) * j // 2 <= keys, j + 1, j)
    i = keys - j * (j - 1) // 2
    return i, j


def sample_undirected_edges(num_nodes, num_edges, seed):
    """Sample exactly num_edges distinct undirected pairs uniformly, without replacement.

    Draws random pair indices, keeps the distinct ones, and repeats until
    there are enough; then keeps a random subset of exactly num_edges. Any
    subset chosen uniformly from a set of uniform draws is itself a uniform
    m-subset of all pairs.

    Uses its own torch.Generator, so it never touches the global RNG (model
    initialization does not depend on how many numbers this consumes).

    Returns a [2, num_edges] long tensor in PyG order ([source, target]) with
    source < target, sorted by (target, source).
    """
    total = num_possible_edges(num_nodes)
    if num_edges < 0 or num_edges > total:
        raise ValueError(f"num_edges must be in [0, {total}] for {num_nodes} nodes, got {num_edges}")

    gen = torch.Generator().manual_seed(seed)
    keys = torch.empty(0, dtype=torch.long)
    while keys.numel() < num_edges:
        draws = max(2 * (num_edges - keys.numel()), 64)
        keys = torch.unique(torch.cat([keys, torch.randint(0, total, (draws,), generator=gen)]))
    keys = keys[torch.randperm(keys.numel(), generator=gen)[:num_edges]].sort().values

    i, j = pair_from_index(keys)
    return torch.stack([i, j])

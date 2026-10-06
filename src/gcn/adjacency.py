"""Canonical normalized adjacency and its dense / COO / CSR representations.

Policy (fixed for every format, see docs/gcn_inference_plan.md):
  1. Symmetrize the graph (undirected).
  2. Coalesce duplicate edges. The graph is unweighted, so duplicates
     collapse to a single edge of weight 1.
  3. Remove existing self-loops and add exactly one per node.
  4. Index entries as [target, source] so `A @ X` aggregates source
     features into target rows.
  5. A_hat = D^{-1/2} (A + I) D^{-1/2}.
  6. Store A_hat as one coalesced COO tensor (the canonical copy).
  7. Derive CSR and dense ONLY from that COO tensor, never by
     re-normalizing, so all formats hold bit-identical values.
"""

import torch

from src.measurement.memory_utils import structural_memory_adj_matrix

from .config import torch_dtype

# Extension point: add "csc", "bsr", etc. here and in convert_adjacency /
# adjacency_storage_breakdown / adjacency_multiply.
STORAGE_FORMATS = ("dense", "coo", "csr")


def build_normalized_coo(edge_index, num_nodes, dtype=torch.float32):
    """Build the canonical A_hat as a coalesced COO tensor on CPU.

    Args:
        edge_index: [2, E] integer tensor in PyG order ([source, target]).
        num_nodes: number of nodes N; the result is N x N.
        dtype: value dtype (FP32 for the primary experiment).
    """
    edge_index = torch.as_tensor(edge_index).to(device="cpu", dtype=torch.long)
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
        raise ValueError(f"edge_index has node ids outside [0, {num_nodes})")

    source, target = edge_index

    # 1 + 4. Undirected, with rows = targets and columns = sources.
    row = torch.cat([target, source])
    col = torch.cat([source, target])

    # 3. Exactly one self-loop per node.
    off_diagonal = row != col
    nodes = torch.arange(num_nodes)
    row = torch.cat([row[off_diagonal], nodes])
    col = torch.cat([col[off_diagonal], nodes])

    # 2. Collapse duplicates. Linear keys sorted by torch.unique give
    # row-major order, which is also what coalesce() produces.
    keys = torch.unique(row * num_nodes + col)
    row, col = keys // num_nodes, keys % num_nodes

    # 5. Symmetric normalization. Every degree is >= 1 because of the self-loops.
    deg_inv_sqrt = torch.bincount(row, minlength=num_nodes).to(dtype).pow(-0.5)
    values = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # 6. Canonical coalesced COO.
    coo = torch.sparse_coo_tensor(
        torch.stack([row, col]), values, (num_nodes, num_nodes), check_invariants=True,
    )
    return coo.coalesce()


def convert_adjacency(coo, storage_format):
    """Derive the requested representation from the canonical COO tensor."""
    if coo.layout != torch.sparse_coo or not coo.is_coalesced():
        raise ValueError("convert_adjacency expects the coalesced COO tensor from build_normalized_coo")
    if storage_format == "coo":
        return coo
    if storage_format == "csr":
        return coo.to_sparse_csr()
    if storage_format == "dense":
        return coo.to_dense()
    raise ValueError(f"unknown storage format {storage_format!r} (supported: {STORAGE_FORMATS})")


def layout_name(adj):
    """Storage-format name ("dense", "coo", "csr") of an adjacency tensor."""
    names = {torch.strided: "dense", torch.sparse_coo: "coo", torch.sparse_csr: "csr"}
    if adj.layout not in names:
        raise ValueError(f"unsupported adjacency layout {adj.layout}")
    return names[adj.layout]


def adjacency_multiply(adj, x):
    """A_hat @ x for any supported layout. Shared by every format."""
    if adj.layout == torch.strided:
        return torch.mm(adj, x)
    if adj.layout in (torch.sparse_coo, torch.sparse_csr):
        return torch.sparse.mm(adj, x)
    raise ValueError(f"unsupported adjacency layout {adj.layout}")


def _nbytes(t):
    return t.numel() * t.element_size()


def adjacency_storage_breakdown(adj):
    """Bytes held by each component tensor of an adjacency representation."""
    if adj.layout == torch.strided:
        return {"values": _nbytes(adj)}
    if adj.layout == torch.sparse_coo:
        return {"values": _nbytes(adj._values()), "indices": _nbytes(adj._indices())}
    if adj.layout == torch.sparse_csr:
        return {
            "values": _nbytes(adj.values()),
            "col_indices": _nbytes(adj.col_indices()),
            "crow_indices": _nbytes(adj.crow_indices()),
        }
    raise ValueError(f"unsupported adjacency layout {adj.layout}")


def adjacency_storage_bytes(adj):
    """Exact bytes of all component tensors (numel * element_size)."""
    return sum(adjacency_storage_breakdown(adj).values())


def estimate_dense_bytes(num_nodes, dtype=torch.float32):
    """Bytes a dense N x N adjacency would need, without allocating it."""
    if isinstance(dtype, str):
        dtype = torch_dtype(dtype)
    element_size = torch.empty((), dtype=dtype).element_size()
    return structural_memory_adj_matrix(num_nodes, bytes_per_element=element_size)

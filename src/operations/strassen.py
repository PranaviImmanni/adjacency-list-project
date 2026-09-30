import torch

def strassen_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Recursive Strassen Matrix Multiplication on PyTorch Tensors.
    Base case uses standard matmul for small matrices (N <= 2).
    """
    n = A.shape[0]

    # Base case: fallback to standard multiplication for small sizes
    if n <= 2:
        return torch.matmul(A, B)

    # Make sure matrix size is even (pad if needed in general implementation)
    mid = n // 2

    # Split A into 4 sub-matrices (Tiling concept)
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]

    # Split B into 4 sub-matrices
    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]

    # 7 Strassen Multiplications
    M1 = strassen_matmul(A11 + A22, B11 + B22)
    M2 = strassen_matmul(A21 + A22, B11)
    M3 = strassen_matmul(A11, B12 - B22)
    M4 = strassen_matmul(A22, B21 - B11)
    M5 = strassen_matmul(A11 + A12, B22)
    M6 = strassen_matmul(A21 - A11, B11 + B12)
    M7 = strassen_matmul(A12 - A22, B21 + B22)

    # Reconstruct C matrix
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    # Combine quadrants
    C = torch.zeros((n, n), dtype=A.dtype)
    C[:mid, :mid] = C11
    C[:mid, mid:] = C12
    C[mid:, :mid] = C21
    C[mid:, mid:] = C22

    return C


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.synthetic.generator import generate_synthetic_graph

    # Test on a 4x4 matrix
    size = 4
    A = generate_synthetic_graph(num_nodes=size, sparsity=0.5, seed=42)
    B = generate_synthetic_graph(num_nodes=size, sparsity=0.5, seed=100)

    ground_truth = torch.matmul(A, B)
    strassen_out = strassen_matmul(A, B)

    is_close = torch.allclose(ground_truth, strassen_out, atol=1e-5)
    print("Standard PyTorch Output:\n", ground_truth)
    print("\nStrassen Output:\n", strassen_out)
    print(f"\nStrassen Matches Ground Truth: {is_close}")
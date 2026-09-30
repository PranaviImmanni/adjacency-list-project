import torch

def multiply_adj_lists(adj_list_A, adj_list_B, num_nodes):
    """
    Multiplies two adjacency lists directly (A x B) without fully 
    converting them back into dense 2D matrices.
    
    Formula: C[i, j] = sum_k (A[i, k] * B[k, j])
    """
    result_list = {i: [] for i in range(num_nodes)}
    
    # Pre-index B by source node for fast lookup
    # B_map[k] gives a dict of {target: weight} for node k
    B_map = {k: {target: w for target, w in neighbors} for k, neighbors in adj_list_B.items()}
    
    for i in range(num_nodes):
        row_accumulator = {}
        
        # Look at all outgoing edges from node i in graph A
        for k, weight_A in adj_list_A.get(i, []):
            # Check where node k connects to in graph B
            if k in B_map:
                for j, weight_B in B_map[k].items():
                    # Accumulate matrix multiplication value: A[i,k] * B[k,j]
                    row_accumulator[j] = row_accumulator.get(j, 0.0) + (weight_A * weight_B)
                    
        # Store non-zero results in the resulting adjacency list
        for j, val in row_accumulator.items():
            if abs(val) > 1e-6:
                result_list[i].append((j, val))
                
    return result_list


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    
    from src.synthetic.generator import generate_synthetic_graph
    from src.conversion.converter import matrix_to_adj_list, adj_list_to_matrix

    num_nodes = 5
    
    # 1. Create two synthetic matrices A and B
    dense_A = generate_synthetic_graph(num_nodes=num_nodes, sparsity=0.6, seed=42)
    dense_B = generate_synthetic_graph(num_nodes=num_nodes, sparsity=0.6, seed=100)
    
    # 2. Perform standard PyTorch dense matrix multiplication (Ground Truth)
    ground_truth = torch.matmul(dense_A, dense_B)
    
    # 3. Convert A and B to Adjacency Lists
    list_A = matrix_to_adj_list(dense_A)
    list_B = matrix_to_adj_list(dense_B)
    
    # 4. Multiply directly using adjacency lists
    list_result = multiply_adj_lists(list_A, list_B, num_nodes=num_nodes)
    
    # 5. Convert list result back to matrix to compare
    reconstructed_result = adj_list_to_matrix(list_result, num_nodes=num_nodes)
    
    # 6. Check equivalence
    is_correct = torch.allclose(ground_truth, reconstructed_result, atol=1e-5)
    
    print("Standard Dense Output (A @ B):\n", ground_truth)
    print("\nAdjacency List Output Reconstructed:\n", reconstructed_result)
    print(f"\nOperations Match Ground Truth: {is_correct}")
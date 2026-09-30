import torch

def matrix_to_adj_list(adj_matrix):
    """
    Converts a dense adjacency matrix into an adjacency list representation.
    
    Returns:
    - adj_list: Dict where key = source node (row), value = list of (target_node, weight) tuples.
    """
    adj_list = {}
    num_nodes = adj_matrix.shape[0]
    
    for src in range(num_nodes):
        adj_list[src] = []
        for dst in range(num_nodes):
            weight = adj_matrix[src, dst].item()
            if weight != 0.0:  # Only store non-zero connections
                adj_list[src].append((dst, weight))
                
    return adj_list


def adj_list_to_matrix(adj_list, num_nodes=None):
    """
    Converts an adjacency list back into a dense PyTorch adjacency matrix.
    Reconstructs the original matrix for round-trip verification.
    """
    if num_nodes is None:
        num_nodes = len(adj_list)
        
    reconstructed_matrix = torch.zeros((num_nodes, num_nodes))
    
    for src, neighbors in adj_list.items():
        for dst, weight in neighbors:
            reconstructed_matrix[src, dst] = weight
            
    return reconstructed_matrix


if __name__ == "__main__":
    # Import our generator from the synthetic folder
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.synthetic.generator import generate_synthetic_graph

    # 1. Generate a test matrix
    dense_matrix = generate_synthetic_graph(num_nodes=5, sparsity=0.6, weighted=True)
    print("1. Dense Matrix:\n", dense_matrix)

    # 2. Convert to Adjacency List
    adj_list = matrix_to_adj_list(dense_matrix)
    print("\n2. Adjacency List Representation:")
    for node, neighbors in adj_list.items():
        print(f"  Node {node} -> {neighbors}")

    # 3. Convert back to Matrix
    reconstructed = adj_list_to_matrix(adj_list)

    # 4. Round-Trip Equality Check
    is_equal = torch.allclose(dense_matrix, reconstructed)
    print(f"\n3. Round-Trip Check (Original == Reconstructed): {is_equal}")
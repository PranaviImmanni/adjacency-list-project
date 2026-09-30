import torch

def generate_synthetic_graph(num_nodes=10, sparsity=0.8, weighted=True, seed=42):
    """
    Generates a synthetic adjacency matrix.
    - num_nodes: Number of vertices (N x N matrix)
    - sparsity: Fraction of zeros (e.g., 0.8 means 80% zeros)
    - weighted: If True, edges have random weights; if False, binary 0/1
    """
    torch.manual_seed(seed)
    
    # Generate random matrix
    prob_matrix = torch.rand((num_nodes, num_nodes))
    
    # Apply sparsity mask (zero out values below sparsity threshold)
    adj_matrix = torch.where(prob_matrix > sparsity, prob_matrix, torch.tensor(0.0))
    
    if not weighted:
        adj_matrix = (adj_matrix > 0).float()
        
    return adj_matrix

if __name__ == "__main__":
    # Test print when running this file directly
    sample_matrix = generate_synthetic_graph(num_nodes=5, sparsity=0.6, weighted=True)
    print("Generated 5x5 Synthetic Adjacency Matrix:\n")
    print(sample_matrix)
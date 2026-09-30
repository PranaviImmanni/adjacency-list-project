import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.synthetic.generator import generate_synthetic_graph
from src.conversion.converter import matrix_to_adj_list
from src.measurement.memory_utils import (
    structural_memory_adj_matrix,
    structural_memory_adj_list,
    measure_peak_memory_cpu,
    prediction_match_rate,
    accuracy_parity,
    output_closeness
)

def benchmark_sparsity_impact():
    n_nodes = 1000  # 1000 x 1000 matrix
    sparsities = [0.1, 0.5, 0.9, 0.95, 0.99]
    bytes_per_element = 4  # float32

    print("==========================================================================")
    print(f"  MEMORY BENCHMARK ON {n_nodes}x{n_nodes} SYNTHETIC GRAPH")
    print("==========================================================================")
    print(f"{'Sparsity':<10} | {'Edges (E)':<10} | {'Dense (KB)':<12} | {'List (KB)':<12} | {'Saved (%)':<10}")
    print("-" * 70)

    for s in sparsities:
        # Generate matrix
        dense_tensor = generate_synthetic_graph(num_nodes=n_nodes, sparsity=s, weighted=True)
        
        # Calculate non-zero edges
        n_edges = int((dense_tensor != 0).sum().item())

        # 1. Structural Memory (Theoretical)
        dense_bytes = structural_memory_adj_matrix(n_nodes, bytes_per_element=bytes_per_element)
        list_bytes = structural_memory_adj_list(n_edges, index_bytes=4, value_bytes=4, store_values=True)

        dense_kb = dense_bytes / 1024
        list_kb = list_bytes / 1024
        saved_pct = ((dense_bytes - list_bytes) / dense_bytes) * 100

        print(f"{int(s*100):<9}% | {n_edges:<10} | {dense_kb:<12.2f} | {list_kb:<12.2f} | {saved_pct:<10.2f}%")

def verify_equivalence():
    print("\n==========================================================================")
    print("  EQUIVALENCE VERIFICATION (Dense vs Sparse Operations)")
    print("==========================================================================")
    
    # Generate ground truth dense output and small noisy sparse output
    dense_out = generate_synthetic_graph(num_nodes=5, sparsity=0.5, seed=1)
    sparse_out = dense_out + 1e-7  # Add minor floating point tolerance
    
    max_diff = output_closeness(dense_out.numpy(), sparse_out.numpy(), metric="max_abs_diff")
    cos_sim = output_closeness(dense_out.numpy(), sparse_out.numpy(), metric="cosine_similarity")
    
    print(f"Max Absolute Output Difference : {max_diff:.2e}")
    print(f"Average Cosine Similarity      : {cos_sim:.6f}")

if __name__ == "__main__":
    benchmark_sparsity_impact()
    verify_equivalence()
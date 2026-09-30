# Conversion utilities

General method for converting any matrix into an adjacency matrix, and
from there into an adjacency list (and the inverse, for validation).

Two cases:
1. Square, graph-like matrix -> treated directly as an adjacency matrix.
2. Rectangular / non-graph matrix -> treated as the biadjacency matrix
   of a bipartite graph (rows and columns become two node sets).


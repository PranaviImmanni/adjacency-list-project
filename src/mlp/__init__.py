"""MLP weight-pruning experiment: Fashion-MNIST, one fixed MLP, dense vs sparse weight storage.

See docs/mlp_weight_pruning_plan.md. Modules:
  config          JSON config loading and validation
  data            checksum-verified Fashion-MNIST, fixed split, train-split normalization
  model           the fixed MLP (784-300-100-10) and its checkpoints
  training        dense baseline training with validation-based selection
  weight_pruning  signed-weight pruning interface (masks) and reference implementation
  formats         dense / dense+mask / COO / CSR weight storage and one inference model
  experiment      condition grid, pruning-effect and format-error comparisons
  harness         model loading and inference entry points for the benchmark harness
  reporting       tables and charts
"""

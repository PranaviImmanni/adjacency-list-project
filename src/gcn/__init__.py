"""Device-portable two-layer GCN for comparing dense / COO / CSR adjacency.

See docs/gcn_inference_plan.md. Modules:
  config       JSON config loading and seeding
  datasets     format-agnostic GraphData + dataset loaders (Cora)
  adjacency    canonical normalized COO and its dense / CSR derivations
  model        the single GCN used for every format
  training     train one checkpoint with COO
  evaluation   inference + prediction-equivalence metrics
  measurement  device checks, synchronization, timing, CUDA memory
"""

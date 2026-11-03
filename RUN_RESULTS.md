# Initial CPU results

Tests 1–3 were run in this workspace. GPU experiments have not been run.

- Test 1: 168/168 output comparisons passed, plus invalid-input validation.
- Test 2 (N=M=256, k=8, F=64): 6/6 methods passed.
- Test 3 development sweep: 36/36 benchmark rows passed.
- Additional Test 2 (N=M=1000, k=8, F=64, seeds 42/43/44): 18/18 rows passed.

## Larger configuration: three-seed summary

All values below summarize N=M=1000, k=8, F=64 on CPU with one PyTorch thread. Each seed used 10 warm-ups and 50 measured calls. Latency is the median of the three per-seed medians; build time is also summarized across seeds. Storage is actual unique representation buffers, excluding node features and outputs.

| Method | Representation KiB | Build ms | Median propagation ms | Worst max absolute error |
| --- | ---: | ---: | ---: | ---: |
| dense | 3914.06 | 2.455 | 1.954 | 1.82e-07 |
| coo | 320.31 | 1.954 | 0.591 | 1.58e-07 |
| csr | 335.95 | 1.994 | 0.133 | 1.58e-07 |
| membership | 132.81 | 0.289 | 1.349 | 1.58e-07 |
| explicit_dense | 3906.25 | 21.184 | 0.841 | 2.18e-07 |
| explicit_sparse | 1083.44 | 4.912 | 1.852 | 3.27e-07 |

## Interpretation

Sparse factorized CSR was faster than dense incidence propagation in this measured configuration and used substantially less representation storage. Membership-list storage was smaller still, but propagation was slower than CSR. Explicit sparse P expanded into many node-pair entries and was slower than factorized sparse incidence in this workload.

CSR storage here includes underlying retained index buffers from PyTorch conversions and the separately materialized transpose. It is not the idealized CSR byte formula. Dense incidence retains degree/weight vectors; precomputed P does not need them for inference. Construction peaks and total working memory are separate from these storage counts.

These are initial synthetic CPU results, not final GPU or model-accuracy evidence. The three-seed run is a repeatability check, not a broad statistical study. CPU RSS is sampled process memory and can miss short peaks. No maximum-capacity search was performed. CSV/JSON files contain raw per-seed numbers and environment metadata.

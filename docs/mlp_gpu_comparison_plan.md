# MLP GPU Format Comparison: Adjacency Lists, Low-Rank Factors and Energy

Pranavi's adjacency-list and low-rank factor experiments (`scripts/benchmark_adjacency_list_mlp.py`,
`scripts/benchmark_low_rank_adjacency.py`) ran in NumPy on a laptop CPU, so their timings cannot be compared with
the CUDA results for dense, dense+mask, COO and CSR. This experiment runs all of them on the lab RTX 3090 with the
same harness, and adds GPU energy measured with NVML.

| Part | Models | Formats | Reference |
|---|---|---|---|
| pruned | the original sweep's conditions: original model; global magnitude, top-K per neuron and global random at 50, 80, 90, 95, 99% | dense, dense+mask, COO, CSR, adjacency list | pruned dense |
| low_rank | W ~= B A (truncated SVD) at rank 4, 8, 16, 32, 64; factor rules all, abs-std k = 0, abs-std k = 1 | factor adjacency list, dense factors | dense factors |

Code: `src/mlp/formats.py` (`AdjacencyList`), `src/mlp/low_rank.py`, `src/mlp/energy.py`,
`src/mlp/gpu_comparison.py` (plan, model building, checks against Pranavi's CSVs),
`src/mlp/gpu_comparison_reporting.py`. Settings: `configs/mlp_gpu_comparison.json`, which inherits every setting
of the original experiment and may not override any of them.

## Fixed controls (same as the original CUDA sweep)

- The three trained checkpoints (`checkpoints/fashion_mlp_seed{42,43,44}_cuda1.pt`); inputs rebuilt from each
  checkpoint's saved normalization constants (bit-identical to training).
- Batch sizes 1, 32, 256 (the first B test images); 20 warm-ups, 50 synchronized timed passes, 3 repetitions;
  one process per run; FP32 with TF32 off; peak-memory counters reset after the warm-up.
- Random pruning: accuracy over pruning seeds 101-103, timing and energy with seed 101.
- Every format is compared with its reference over all 15,000 validation + test images and on each benchmark batch
  with the project's criterion (identical predictions and accuracy, `allclose(atol=1e-5, rtol=1e-4)`). The
  evaluation fails if a prediction or the accuracy differs. Each format and its FP32 reference are also compared
  with an exact float64 computation of the same model, to tell rounding from errors when the tolerance is missed:
  in the seed-42 pilot the low-rank factor lists gave identical predictions but differed from dense factors by up to
  1.2e-4 (logits reach 116); they are 1.2e-4 from float64, as plain FP32 summation is, while cuBLAS's dense factors
  are 2.8e-5 from it. They are reported as outside the tolerance, with these numbers.
- The four standard formats are rerun (not reused) so that all six are measured in the same session with the same
  energy method. Their accuracy is checked against the original sweep and their latency compared with it.

## Decisions (made here, not by the original plan)

| Decision | Setting | Why |
|---|---|---|
| Adjacency list on the GPU | each output neuron's incoming edges (input neuron, weight), packed as offsets + neighbor indices + weights; computed edge by edge: gather the source activations, multiply by the weights, sum each neuron's list (`torch.segment_reduce`) | the standard GPU layout of an adjacency list; it uses generic gather / segment-sum kernels, so the comparison with CSR is graph-style processing vs cuSPARSE's sparse-matrix multiply. A row-by-row Python loop (about 1,200 kernel launches per pass) would measure Python overhead, not the format |
| Adjacency list storage | 4 B value + 8 B neighbor index per edge + 8 B per offset | the accounting of Pranavi's CSVs; nothing else is stored per edge (equal to CSR's bytes) |
| Low-rank factors | NumPy float64 SVD, rank capped per layer, factor rules and thresholds exactly as in her script; stored in FP32; no fine-tuning | reproduces her factors: seed-42 accuracy, edges, bytes and reconstruction error match her CSVs exactly |
| Low-rank reference | dense factors (both factors dense FP32, unselected entries 0), also timed | the dense implementation of the same modified model, and the obvious baseline for the factor lists |
| Factor rule abs-std k = 1 | included | covers her `reports/low_rank_adjacency_k1.csv` as well as the k = 0 table in the paper |

## Energy (NVML)

- Package `nvidia-ml-py` (module `pynvml`), installed in the project's `.venv`. No root access needed.
- The NVML device is matched to `cuda:1` by UUID (or PCI address), never by index.
- Per run, after the latency timing: idle power over 1.5 s (model loaded, nothing running), 0.5 s of the workload
  to bring the clocks up, then synchronized inference of the benchmark batch, back to back, for 3 s.
- Method: NVML's cumulative energy counter, read only at the two ends of each window. On the lab RTX 3090,
  polling it every few ms makes it undercount by up to 20x (first pilots); read about once a second it matches
  NVML's power reading within 2% at idle (108 vs 110 W) and under load (347 vs 349 W)
  (`scripts/check_gpu_energy.py`). The counter changes every ~100 ms, so each end can be up to 100 ms stale
  (unbiased, about 1.4% SD on 3 s). Fallback when a GPU has no counter: power sampling every 10 ms.
- Reported: average power (W), energy per batch and per example (J), gross and net of idle power; NVML's power
  reading and instantaneous power right after each window as a cross-check; temperature, clocks, power limit and
  any other process on the GPU. Clocks are not locked. With our CUDA context open, GPU 1 stays in P2 at
  1,710 MHz and draws about 110 W doing nothing, so the net values are the extra power of the computation.
- Another process is resident on GPU 1 in every run: PID 3700, the display server (Xorg) of another account, at 0%
  utilization. NVML measures the whole card; its share is part of the idle power.

## Running on the lab GPU

```bash
cd /data/adjli/adjacency-list-project && source .venv/bin/activate
git fetch origin && git checkout mlp-gpu-comparison
pip install nvidia-ml-py
pytest -q

# Energy readings check (~25 s): counter vs power reading, idle and under load
python scripts/check_gpu_energy.py --device cuda:1

# Pilot (~5 min): original model, magnitude 90% in all five formats, rank 32 low-rank; check before the rest
python scripts/run_mlp_gpu_comparison.py --stage pilot --seeds 42 --device cuda:1

# Full grid inside tmux (2,862 benchmark runs, about 7-8 h; resumable, the pilot's runs are reused)
python scripts/run_mlp_gpu_comparison.py --stage all --seeds 42 43 44 --device cuda:1
```

Results go to `results/mlp_gpu/cuda1/` (evaluate, benchmark). Report: `python scripts/summarize_mlp_gpu_comparison.py
--run cuda1` writes `results/mlp_gpu/summary_cuda1/` (summary.md, pruned.csv, low_rank.csv, runs.csv,
equivalence.csv, pranavi_crosscheck.csv, stability.csv, charts). `--parts pruned` or `--parts low_rank` runs one
part only. CPU debug runs (`--device cpu --checkpoint-run cuda1`) go to `results/mlp_gpu/cpu/`, which git ignores;
energy is not measured off CUDA.

#!/usr/bin/env python3
"""Benchmark low-rank weight factorization with adjacency-list factors.

For each layer W, compute a truncated SVD

    W ~= B A,  B = U_r diag(S_r),  A = V_r^T.

The two factors are stored as row-oriented weighted adjacency lists.  The
benchmark evaluates the resulting MLP on the exported Fashion-MNIST test set
and reports accuracy, reconstruction error, structural storage, conversion
time, and CPU inference latency.  ``all`` keeps every factor entry as an edge;
``abs_std`` keeps entries whose magnitude exceeds mean(abs(F)) + k std(abs(F))
for each factor F.

This is a representation experiment, not LoRA: the original W is replaced by
the low-rank product BA rather than frozen and augmented with an adapter.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np


def forward_dense(weights, biases, x):
    value = x
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        value = value.dot(weight.T) + bias
        if index + 1 < len(weights):
            value = np.maximum(value, 0.0)
    return value


def factorize(weight, rank):
    u, singular, vt = np.linalg.svd(weight, full_matrices=False)
    rank = min(int(rank), len(singular))
    return u[:, :rank] * singular[:rank], vt[:rank, :]


def factor_mask(factor, rule, threshold_r):
    if rule == "all":
        return np.ones(factor.shape, dtype=bool), float("nan")
    if rule != "abs_std":
        raise ValueError("factor edge rule must be 'all' or 'abs_std'")
    magnitudes = np.abs(factor)
    threshold = float(magnitudes.mean() + threshold_r * magnitudes.std())
    return magnitudes > threshold, threshold


def make_adjacency(factor, mask):
    rows = []
    for row in range(factor.shape[0]):
        columns = np.flatnonzero(mask[row]).astype(np.int64, copy=False)
        rows.append((columns, factor[row, columns].astype(np.float32, copy=False)))
    return rows


def adjacency_matmul(rows, x, output_width):
    output = np.zeros((x.shape[0], output_width), dtype=np.float64)
    for row, (columns, values) in enumerate(rows):
        if len(columns):
            output[:, row] = x[:, columns].dot(values.astype(np.float64, copy=False))
    return output


def forward_factors(factor_layers, biases, x):
    value = x
    for index, ((left, right), bias) in enumerate(zip(factor_layers, biases)):
        rank = len(right)
        hidden = adjacency_matmul(right, value, rank)
        value = adjacency_matmul(left, hidden, len(left)) + bias
        if index + 1 < len(factor_layers):
            value = np.maximum(value, 0.0)
    return value


def timed(function, warmups, repeats):
    for _ in range(warmups):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(samples)), float(np.percentile(samples, 95))


def structural_factor_bytes(factor_layers):
    """Values + 64-bit column indices + 64-bit row pointers for both factors."""
    total = 0
    edges = 0
    for left, right in factor_layers:
        for rows in (left, right):
            edges += sum(len(columns) for columns, _ in rows)
            total += sum(len(columns) * (4 + 8) for columns, _ in rows)
            total += (len(rows) + 1) * 8
    return total, edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="exports/fashion_mlp_cuda1.npz")
    parser.add_argument("--output", default="reports/low_rank_adjacency.csv")
    parser.add_argument("--model-seeds", default="42,43,44")
    parser.add_argument("--ranks", default="4,8,16,32")
    parser.add_argument("--factor-edge-rules", default="all,abs_std")
    parser.add_argument("--threshold-r", type=float, default=0.0)
    parser.add_argument("--accuracy-examples", type=int, default=10000)
    parser.add_argument("--timing-examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    bundle = np.load(args.bundle, allow_pickle=False)
    seeds = [int(v) for v in args.model_seeds.split(",") if v.strip()]
    ranks = [int(v) for v in args.ranks.split(",") if v.strip()]
    rules = [v.strip() for v in args.factor_edge_rules.split(",") if v.strip()]
    rows = []

    for seed in seeds:
        prefix = f"m{seed}"
        weights = [bundle[f"{prefix}_fc{i}_weight"].astype(np.float64) for i in (1, 2, 3)]
        biases = [bundle[f"{prefix}_fc{i}_bias"].astype(np.float64) for i in (1, 2, 3)]
        x = bundle["x_test"].astype(np.float64)
        y = bundle["y_test"]
        n_accuracy = min(args.accuracy_examples, len(x))
        n_timing = min(args.timing_examples, len(x))
        original = forward_dense(weights, biases, x[:n_accuracy])
        original_predictions = np.argmax(original, axis=1)
        original_accuracy = float(np.mean(original_predictions == y[:n_accuracy]))
        batches = [x[i:i + args.batch_size] for i in range(0, n_timing, args.batch_size)]

        for rank in ranks:
            base_factors = [factorize(weight, rank) for weight in weights]
            for rule in rules:
                build_start = time.perf_counter()
                factor_layers = []
                thresholds = []
                for left, right in base_factors:
                    left_mask, left_threshold = factor_mask(left, rule, args.threshold_r)
                    right_mask, right_threshold = factor_mask(right, rule, args.threshold_r)
                    factor_layers.append((make_adjacency(left, left_mask),
                                          make_adjacency(right, right_mask)))
                    thresholds.extend((left_threshold, right_threshold))
                conversion_ms = (time.perf_counter() - build_start) * 1000.0

                reconstructed = [left.dot(right) for left, right in base_factors]
                factor_logits = forward_factors(factor_layers, biases, x[:n_accuracy])
                predictions = np.argmax(factor_logits, axis=1)
                factor_call = lambda: [forward_factors(factor_layers, biases, batch) for batch in batches]
                median_ms, p95_ms = timed(factor_call, args.warmups, args.repeats)
                storage_bytes, edges = structural_factor_bytes(factor_layers)
                # Evaluation uses float64 arrays, but the reported storage
                # budget follows the project's FP32 representation convention.
                dense_bytes = sum(weight.size * 4 for weight in weights)
                rows.append({
                    "model_seed": seed,
                    "rank": rank,
                    "factor_edge_rule": rule,
                    "threshold_r": args.threshold_r if rule == "abs_std" else "",
                    "accuracy": float(np.mean(predictions == y[:n_accuracy])),
                    "dense_accuracy": original_accuracy,
                    "accuracy_delta_pp": 100.0 * (float(np.mean(predictions == y[:n_accuracy])) - original_accuracy),
                    "prediction_match_to_dense": float(np.mean(predictions == original_predictions)),
                    "max_logit_error_vs_dense": float(np.max(np.abs(factor_logits - original))),
                    "mean_weight_reconstruction_mse": float(np.mean([
                        np.mean((weight - approx) ** 2) for weight, approx in zip(weights, reconstructed)])),
                    "factor_edges": edges,
                    "factor_sparsity": 1.0 - edges / sum(left.size + right.size for left, right in base_factors),
                    "factor_adjacency_bytes": storage_bytes,
                    "dense_weight_bytes": dense_bytes,
                    "storage_ratio_to_dense": storage_bytes / dense_bytes,
                    "threshold_min": min(thresholds) if rule == "abs_std" else "",
                    "threshold_max": max(thresholds) if rule == "abs_std" else "",
                    "conversion_time_ms": conversion_ms,
                    "cpu_median_ms": median_ms,
                    "cpu_p95_ms": p95_ms,
                })
                print(f"seed={seed} rank={rank} rule={rule} accuracy={rows[-1]['accuracy']:.4f} "
                      f"factor_sparsity={rows[-1]['factor_sparsity']:.3f} latency={median_ms:.2f} ms")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark the weighted adjacency-list MLP path from the exported NumPy bundle.

This intentionally reports CPU dense and CPU adjacency-list timings together.
They must not be compared directly with Minh's CUDA JSON timings.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np


def forward_dense(weights, biases, x):
    value = x
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        value = value.dot(weight.T) + bias
        if i + 1 < len(weights):
            value = np.maximum(value, 0.0)
    return value


def make_adjacency(weight, mask):
    rows, cols = np.nonzero(mask)
    return [(cols[rows == row], weight[row, cols[rows == row]])
            for row in range(weight.shape[0])]


def forward_adjacency(layers, biases, x):
    value = x
    for layer_index, (adjacency, bias) in enumerate(zip(layers, biases)):
        output = np.zeros((value.shape[0], len(adjacency)), dtype=np.float64)
        for row, (columns, weights) in enumerate(adjacency):
            if len(columns):
                output[:, row] = value[:, columns].dot(weights)
        value = output + bias
        if layer_index + 1 < len(layers):
            value = np.maximum(value, 0.0)
    return value


def condition_key(method, sparsity, prune_seed):
    if prune_seed is None:
        return "{}_s{}".format(method, int(round(100 * sparsity)))
    return "{}_p{}_s{}".format(method, prune_seed, int(round(100 * sparsity)))


def timing(function, warmups, repeats):
    for _ in range(warmups):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(samples)), float(np.percentile(samples, 95))


def run(args):
    bundle = np.load(args.bundle, allow_pickle=False)
    prefix = "m{}".format(args.model_seed)
    weights = [bundle["{}_fc{}_weight".format(prefix, i)] for i in (1, 2, 3)]
    biases = [bundle["{}_fc{}_bias".format(prefix, i)].astype(np.float64) for i in (1, 2, 3)]
    x = bundle["x_test"].astype(np.float64)
    y = bundle["y_test"]
    accuracy_count = min(args.accuracy_examples, len(x))
    timing_count = min(args.timing_examples, len(x))
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    sparsities = [float(item) for item in args.sparsities.split(",") if item.strip()]
    rows = []
    for method in methods:
        prune_seeds = [None] if method != "random" else [int(args.prune_seed)]
        for sparsity in sparsities:
            for prune_seed in prune_seeds:
                key = condition_key(method, sparsity, prune_seed)
                masks = [bundle["{}_{}_fc{}_mask".format(prefix, key, i)].astype(bool)
                         for i in (1, 2, 3)]
                masked_weights = [np.where(mask, weight, 0.0) for weight, mask in zip(weights, masks)]
                build_start = time.perf_counter()
                adjacency = [make_adjacency(weight, mask)
                             for weight, mask in zip(weights, masks)]
                build_ms = (time.perf_counter() - build_start) * 1000.0
                x_accuracy = x[:accuracy_count]
                dense_accuracy = forward_dense(masked_weights, biases, x_accuracy)
                adjacency_accuracy = forward_adjacency(adjacency, biases, x_accuracy)
                dense_classes = np.argmax(dense_accuracy, axis=1)
                adjacency_classes = np.argmax(adjacency_accuracy, axis=1)
                nnz = sum(int(mask.sum()) for mask in masks)
                for batch_size in [int(item) for item in args.batch_sizes.split(",") if item.strip()]:
                    x_timing = x[:timing_count]
                    batches = [x_timing[i:i + batch_size]
                               for i in range(0, len(x_timing), batch_size)]
                    dense_call = lambda: [forward_dense(masked_weights, biases, batch) for batch in batches]
                    adjacency_call = lambda: [forward_adjacency(adjacency, biases, batch) for batch in batches]
                    dense_median, dense_p95 = timing(dense_call, args.warmups, args.repeats)
                    list_median, list_p95 = timing(adjacency_call, args.warmups, args.repeats)
                    rows.extend([
                        {"model_seed": args.model_seed, "method": method,
                         "pruning_seed": "" if prune_seed is None else prune_seed,
                         "sparsity": sparsity, "format": "pruned_dense_cpu",
                         "batch_size": batch_size, "accuracy": float(np.mean(dense_classes == y[:accuracy_count])),
                         "prediction_match_rate": 1.0,
                         "max_output_error": 0.0, "median_latency_ms": dense_median,
                         "p95_latency_ms": dense_p95, "nnz": nnz,
                         "conversion_time_ms": build_ms, "energy_joules": ""},
                        {"model_seed": args.model_seed, "method": method,
                         "pruning_seed": "" if prune_seed is None else prune_seed,
                         "sparsity": sparsity, "format": "adjacency_list_cpu",
                         "batch_size": batch_size, "accuracy": float(np.mean(adjacency_classes == y[:accuracy_count])),
                         "prediction_match_rate": float(np.mean(adjacency_classes == dense_classes)),
                         "max_output_error": float(np.max(np.abs(adjacency_accuracy - dense_accuracy))),
                         "median_latency_ms": list_median, "p95_latency_ms": list_p95,
                         "nnz": nnz, "conversion_time_ms": build_ms, "energy_joules": ""},
                    ])
                print("{} {}% nnz={} accuracy={:.4f} match={:.4f} error={:.2e}".format(
                    method, int(sparsity * 100), nnz, np.mean(adjacency_classes == y[:accuracy_count]),
                    np.mean(adjacency_classes == dense_classes), np.max(np.abs(adjacency_accuracy - dense_accuracy))))
    fields = ["model_seed", "method", "pruning_seed", "sparsity", "format", "batch_size",
              "accuracy", "prediction_match_rate", "max_output_error", "median_latency_ms",
              "p95_latency_ms", "nnz", "conversion_time_ms", "energy_joules"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote {} ({} rows)".format(output, len(rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="exports/fashion_mlp_cuda1.npz")
    parser.add_argument("--output", default="reports/adjacency_list_mlp_benchmark.csv")
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--methods", default="magnitude,topk_per_neuron,random")
    parser.add_argument("--sparsities", default="0.5,0.8,0.9,0.95,0.99")
    parser.add_argument("--prune-seed", type=int, default=101)
    parser.add_argument("--batch-sizes", default="1,32,256")
    parser.add_argument("--accuracy-examples", type=int, default=10000)
    parser.add_argument("--timing-examples", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

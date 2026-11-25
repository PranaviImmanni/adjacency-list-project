#!/usr/bin/env python3
"""Evaluate statistical conversion of MLP weights into weighted networks.

For each layer W and parameter r, retain an edge when
    abs(W_ij) > mean(abs(W)) + r * std(abs(W)).

The experiment uses the committed NumPy export, so it does not retrain or
change Minh's results. It reports one row per model seed and r value.
"""

import argparse
import csv
import tracemalloc
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


def make_adjacency(weight, mask):
    rows, cols = np.nonzero(mask)
    return [(cols[rows == row], weight[row, cols[rows == row]])
            for row in range(weight.shape[0])]


def forward_adjacency(layers, biases, x):
    value = x
    for index, (layer, bias) in enumerate(zip(layers, biases)):
        output = np.zeros((value.shape[0], len(layer)), dtype=np.float64)
        for row, (columns, weights) in enumerate(layer):
            if len(columns):
                output[:, row] = value[:, columns].dot(weights)
        value = output + bias
        if index + 1 < len(layers):
            value = np.maximum(value, 0.0)
    return value


def timed(function, warmups, repeats):
    for _ in range(warmups):
        function()
    samples = []
    tracemalloc.start()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            function()
            samples.append((time.perf_counter() - start) * 1000.0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return float(np.median(samples)), float(np.percentile(samples, 95)), int(peak)


def build_statistical_masks(weights, r, edge_policy):
    masks = []
    thresholds = []
    for weight in weights:
        if edge_policy == "both_signs":
            magnitudes = np.abs(weight)
        elif edge_policy == "positive_only":
            magnitudes = weight[weight > 0]
        elif edge_policy == "negative_only":
            magnitudes = np.abs(weight[weight < 0])
        else:
            raise ValueError("unknown edge policy: {}".format(edge_policy))
        if magnitudes.size == 0:
            thresholds.append(float("inf"))
            masks.append(np.zeros(weight.shape, dtype=bool))
            continue
        mean = float(magnitudes.mean())
        std = float(magnitudes.std())
        threshold = mean + r * std
        thresholds.append(threshold)
        if edge_policy == "both_signs":
            masks.append(np.abs(weight) > threshold)
        elif edge_policy == "positive_only":
            masks.append(weight > threshold)
        else:
            masks.append(weight < -threshold)
    return masks, thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="exports/fashion_mlp_cuda1.npz")
    parser.add_argument("--output", default="reports/statistical_graph_conversion.csv")
    parser.add_argument("--model-seeds", default="42,43,44")
    parser.add_argument("--r-values", default="0,0.5,1,1.5,2")
    parser.add_argument("--edge-policies", default="both_signs")
    parser.add_argument("--accuracy-examples", type=int, default=10000)
    parser.add_argument("--timing-examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    bundle = np.load(args.bundle, allow_pickle=False)
    seeds = [int(value) for value in args.model_seeds.split(",") if value.strip()]
    r_values = [float(value) for value in args.r_values.split(",") if value.strip()]
    edge_policies = [value.strip() for value in args.edge_policies.split(",") if value.strip()]
    rows = []

    for seed in seeds:
        prefix = "m{}".format(seed)
        weights = [bundle["{}_fc{}_weight".format(prefix, i)].astype(np.float64)
                   for i in (1, 2, 3)]
        biases = [bundle["{}_fc{}_bias".format(prefix, i)].astype(np.float64)
                  for i in (1, 2, 3)]
        x = bundle["x_test"].astype(np.float64)
        y = bundle["y_test"]
        n_accuracy = min(args.accuracy_examples, len(x))
        n_timing = min(args.timing_examples, len(x))
        original_logits = forward_dense(weights, biases, x[:n_accuracy])
        original_predictions = np.argmax(original_logits, axis=1)
        original_accuracy = float(np.mean(original_predictions == y[:n_accuracy]))

        for edge_policy in edge_policies:
          for r in r_values:
            masks, thresholds = build_statistical_masks(weights, r, edge_policy)
            masked_weights = [np.where(mask, weight, 0.0)
                              for weight, mask in zip(weights, masks)]
            build_start = time.perf_counter()
            adjacency = [make_adjacency(weight, mask)
                         for weight, mask in zip(weights, masks)]
            conversion_ms = (time.perf_counter() - build_start) * 1000.0

            graph_logits = forward_adjacency(adjacency, biases, x[:n_accuracy])
            threshold_logits = forward_dense(masked_weights, biases, x[:n_accuracy])
            graph_predictions = np.argmax(graph_logits, axis=1)
            threshold_predictions = np.argmax(threshold_logits, axis=1)
            nnz = sum(int(mask.sum()) for mask in masks)
            total = sum(mask.size for mask in masks)
            positive = sum(int(((weight > 0) & mask).sum())
                           for weight, mask in zip(weights, masks))
            negative = sum(int(((weight < 0) & mask).sum())
                           for weight, mask in zip(weights, masks))

            batches = [x[i:i + args.batch_size]
                       for i in range(0, n_timing, args.batch_size)]
            graph_call = lambda: [forward_adjacency(adjacency, biases, batch)
                                  for batch in batches]
            dense_call = lambda: [forward_dense(masked_weights, biases, batch)
                                  for batch in batches]
            graph_median, graph_p95, graph_peak = timed(graph_call, args.warmups, args.repeats)
            dense_median, dense_p95, dense_peak = timed(dense_call, args.warmups, args.repeats)
            row = {
                "model_seed": seed, "edge_policy": edge_policy, "r": r,
                "threshold_rule": "layer statistics with signed edge policy",
                "accuracy": float(np.mean(graph_predictions == y[:n_accuracy])),
                "dense_accuracy": original_accuracy,
                "accuracy_delta_pp": 100.0 * (float(np.mean(graph_predictions == y[:n_accuracy])) - original_accuracy),
                "prediction_match_to_dense": float(np.mean(graph_predictions == original_predictions)),
                "graph_vs_threshold_dense_match": float(np.mean(graph_predictions == threshold_predictions)),
                "max_output_error_vs_threshold_dense": float(np.max(np.abs(graph_logits - threshold_logits))),
                "overall_sparsity": 1.0 - float(nnz) / total, "nnz": nnz,
                "positive_edges": positive, "negative_edges": negative,
                "threshold_fc1": thresholds[0], "threshold_fc2": thresholds[1],
                "threshold_fc3": thresholds[2], "dense_cpu_median_ms": dense_median,
                "dense_cpu_p95_ms": dense_p95, "graph_cpu_median_ms": graph_median,
                "graph_cpu_p95_ms": graph_p95, "conversion_time_ms": conversion_ms,
                "dense_peak_python_bytes": dense_peak, "graph_peak_python_bytes": graph_peak,
                "energy_joules": "",
            }
            rows.append(row)
            print("seed={} policy={} r={} sparsity={:.3f} accuracy={:.4f} match={:.4f} graph_ms={:.2f}".format(
                seed, edge_policy, r, row["overall_sparsity"], row["accuracy"],
                row["graph_vs_threshold_dense_match"], row["graph_cpu_median_ms"]))
          

    fields = list(rows[0].keys())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote {} ({} rows)".format(output, len(rows)))


if __name__ == "__main__":
    main()

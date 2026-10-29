"""Check that one checkpoint gives equivalent predictions with dense, COO and CSR.

COO is the reference. Exits with status 1 if any format misses the target
(100% label agreement, identical test accuracy, logits within atol/rtol).

Example:
    python scripts/compare_gcn_formats.py --dataset cora --device cpu \
        --checkpoint checkpoints/cora_seed42.pt

With --prune-method/--prune-fraction (and --prune-seed for random), the
comparison runs on that pruned graph instead (src/gcn/pruning.py). Every
format is still derived from the one normalized COO of that graph.
--logits-file saves all three formats' logits for the offline margin analysis.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gcn.adjacency import (STORAGE_FORMATS, adjacency_storage_bytes, build_normalized_coo,
                               convert_adjacency)
from src.gcn.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config, torch_dtype
from src.gcn.datasets import load_dataset, print_summary
from src.gcn.error_analysis import format_error
from src.gcn.evaluation import compare_to_reference, is_equivalent, run_inference
from src.gcn.measurement import check_format_supported, configure_precision, device_info, resolve_device
from src.gcn.model import checkpoint_identity, load_gcn_checkpoint
from src.gcn.pruning import METHODS, prune_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "cora_seed42.pt"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output", default=None, help="optional JSON file for the metrics")
    parser.add_argument("--prune-method", default="none", choices=METHODS)
    parser.add_argument("--prune-fraction", type=float, default=0.0)
    parser.add_argument("--prune-seed", type=int, default=None, help="random pruning only")
    parser.add_argument("--logits-file", default=None, help="optional .pt file for every format's logits")
    args = parser.parse_args()
    if args.prune_method == "random" and args.prune_seed is None:
        parser.error("--prune-method random needs --prune-seed")

    config = load_config(args.config)
    dtype = torch_dtype(config["dtype"])
    device = resolve_device(args.device)
    for storage_format in STORAGE_FORMATS:
        check_format_supported(device, storage_format)
    configure_precision(device, config["tf32"])

    graph = load_dataset(args.dataset)
    print_summary(graph)
    model, checkpoint = load_gcn_checkpoint(args.checkpoint, device)
    print(f"Checkpoint {args.checkpoint}: best epoch {checkpoint['best_epoch']}, "
          f"saved test acc {checkpoint['test_accuracy']:.4f}")

    pruned = prune_graph(graph.edge_index, graph.num_nodes, args.prune_method, args.prune_fraction,
                         x=graph.x.to(dtype), seed=args.prune_seed)
    if args.prune_method != "none":
        print(f"Pruned graph: {args.prune_method}"
              f"{'' if args.prune_seed is None else f' seed {args.prune_seed}'}, removed "
              f"{pruned['removed_unique_edges']:,} of {pruned['original_unique_edges']:,} undirected edges")
    coo = build_normalized_coo(pruned["retained_edges"], graph.num_nodes, dtype)
    x = graph.x.to(device=device, dtype=dtype)

    logits, storage = {}, {}
    for storage_format in ("coo", "dense", "csr"):  # reference first
        adj = convert_adjacency(coo, storage_format)
        storage[storage_format] = adjacency_storage_bytes(adj)
        logits[storage_format] = run_inference(model, x, adj.to(device)).cpu()

    results = {}
    print(f"\nReference: COO on {device}.  Tolerance: allclose(atol=1e-5, rtol=1e-4)")
    print(f"{'format':<7}{'adj bytes':>12}{'test acc':>10}{'acc diff pp':>13}{'agree all':>11}"
          f"{'agree test':>12}{'mean |dz|':>11}{'max |dz|':>11}{'allclose':>10}{'equivalent':>12}")
    for storage_format in STORAGE_FORMATS:
        m = compare_to_reference(logits["coo"], logits[storage_format], graph.y, graph.test_mask)
        m.update(format_error(logits["coo"], logits[storage_format], graph.test_mask, m))
        m["adjacency_bytes"] = storage[storage_format]
        m["equivalent"] = is_equivalent(m)
        results[storage_format] = m
        print(f"{storage_format:<7}{m['adjacency_bytes']:>12,}{m['test_accuracy']:>10.4f}"
              f"{m['accuracy_difference_pp']:>13.4f}{m['prediction_agreement_all']:>11.4f}"
              f"{m['prediction_agreement_test']:>12.4f}{m['mean_abs_logit_diff']:>11.2e}"
              f"{m['max_abs_logit_diff']:>11.2e}{str(m['allclose']):>10}{str(m['equivalent']):>12}")

    all_equivalent = all(m["equivalent"] for m in results.values())
    print(f"\nAll formats equivalent: {all_equivalent}")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump({"dataset": args.dataset, "checkpoint": args.checkpoint,
                       "checkpoint_identity": checkpoint_identity(args.checkpoint, checkpoint),
                       "reference_format": "coo", **device_info(device),
                       "pruning": {key: pruned[key] for key in (
                           "method", "seed", "requested_removal_fraction", "actual_removal_fraction",
                           "original_unique_edges", "retained_unique_edges", "removed_unique_edges")},
                       "graph": pruned["pruned_graph"],
                       "logits_file": f"logits/{Path(args.logits_file).name}" if args.logits_file else None,
                       "all_equivalent": all_equivalent, "formats": results}, f, indent=2)
        print(f"Wrote {output}")
    if args.logits_file:
        path = Path(args.logits_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"logits": logits, "y": graph.y, "val_mask": graph.val_mask, "test_mask": graph.test_mask}, path)
        print(f"Wrote {path}")

    sys.exit(0 if all_equivalent else 1)


if __name__ == "__main__":
    main()

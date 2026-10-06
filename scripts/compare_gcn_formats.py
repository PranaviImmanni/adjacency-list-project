"""Check that one checkpoint gives equivalent predictions with dense, COO and CSR.

COO is the reference. Exits with status 1 if any format misses the target
(100% label agreement, identical test accuracy, logits within atol/rtol).

Example:
    python scripts/compare_gcn_formats.py --dataset cora --device cpu \
        --checkpoint checkpoints/cora_seed42.pt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gcn.adjacency import (STORAGE_FORMATS, adjacency_storage_bytes, build_normalized_coo,
                               convert_adjacency)
from src.gcn.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config, torch_dtype
from src.gcn.datasets import load_dataset, print_summary
from src.gcn.evaluation import compare_to_reference, is_equivalent, run_inference
from src.gcn.measurement import check_format_supported, configure_precision, device_info, resolve_device
from src.gcn.model import load_gcn_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "cora_seed42.pt"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output", default=None, help="optional JSON file for the metrics")
    args = parser.parse_args()

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

    coo = build_normalized_coo(graph.edge_index, graph.num_nodes, dtype)
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
            json.dump({"dataset": args.dataset, "checkpoint": args.checkpoint, "reference_format": "coo",
                       **device_info(device), "all_equivalent": all_equivalent, "formats": results}, f, indent=2)
        print(f"Wrote {output}")

    sys.exit(0 if all_equivalent else 1)


if __name__ == "__main__":
    main()

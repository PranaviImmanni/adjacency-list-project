"""Accuracy, pruning effect and storage-format epsilon for one trained MLP.

For every condition of one model seed, on one device, over all 5,000
validation + 10,000 test images (in chunks of eval_batch_size):
  1. the original dense model (computed once, the pruning-effect reference);
  2. the pruned model stored dense (zeros applied): its accuracy, and the
     pruning effect against (1);
  3. dense+mask, COO and CSR of the same pruned weights, each compared with
     (2): epsilon, allclose, agreement, accuracy difference and margins.

One JSON per condition. Existing files are skipped, so a rerun resumes.

Example:
    python scripts/evaluate_mlp_pruning.py --checkpoint checkpoints/fashion_mlp_seed42_cuda1.pt \
        --device cuda:1 --output-dir results/mlp/cuda1/evaluate
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# PyTorch prints this notice every time a CSR tensor is created; it is not about our results.
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

from src.gcn.measurement import check_format_supported, configure_precision, device_info, resolve_device
from src.gcn.pruning import tensor_sha256
from src.mlp.config import DEFAULT_CONFIG_PATH, load_config
from src.mlp.data import load_fashion_mnist
from src.mlp.experiment import (ORIGINAL, accuracies, condition_tag, conditions, format_comparison,
                                logits_in_batches, pruning_comparison)
from src.mlp.formats import SPARSE_FORMATS, build_format_model
from src.mlp.model import checkpoint_identity, load_mlp_checkpoint
from src.mlp.weight_pruning import prune_layers

SCHEMA_VERSION = 1
COMPARED_FORMATS = ("dense_mask", "coo", "csr")


def check_normalization(data, checkpoint):
    saved = checkpoint["data"]
    if (saved["split_seed"] != data.split_seed or saved["val_indices_sha256"] != data.val_indices_sha256
            or abs(saved["normalization"]["mean"] - data.mean) > 1e-12
            or abs(saved["normalization"]["std"] - data.std) > 1e-12):
        raise SystemExit("data split or normalization differs from the one the checkpoint was trained with")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--conditions", default="all", choices=["all", "pilot"],
                        help="pilot: the original model and magnitude at 90%%")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-logits", action="store_true", help="also save pruned-dense logits (.pt)")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    for storage_format in SPARSE_FORMATS:
        check_format_supported(device, storage_format)
    precision = configure_precision(device, config["tf32"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_fashion_mnist(config["val_size"], config["split_seed"])
    model, checkpoint = load_mlp_checkpoint(args.checkpoint)
    check_normalization(data, checkpoint)
    identity = checkpoint_identity(args.checkpoint, checkpoint)
    seed = checkpoint["seed"]
    x, y, val_mask, test_mask = data.eval_set()
    batch = config["eval_batch_size"]
    pruned_layers = config["pruned_layers"]

    original = build_format_model(model, {}, "dense").to(device)
    original_logits = logits_in_batches(original, x, batch, device)
    del original

    todo = conditions(config) if args.conditions == "all" else [ORIGINAL, ("magnitude", 0.9, None)]
    common = {"schema_version": SCHEMA_VERSION, "experiment": "mlp_evaluate", "status": "ok",
              "checkpoint": identity, "model_seed": seed, **device_info(device), **precision,
              "dtype": config["dtype"], "eval_batch_size": batch,
              "eval_set": {"val": int(val_mask.sum()), "test": int(test_mask.sum())}}
    failed = []
    for i, (method, sparsity, pruning_seed) in enumerate(todo, 1):
        tag = condition_tag(seed, method, sparsity, pruning_seed)
        output = output_dir / f"{tag}.json"
        if output.exists():
            print(f"[{i}/{len(todo)}] exists, skipping {output.name}")
            continue
        result = {**common, "timestamp": datetime.now(timezone.utc).isoformat(), "method": method,
                  "requested_sparsity": sparsity, "pruning_seed": pruning_seed}

        if method == "none":
            result.update(actual_sparsity=0.0, accuracy=accuracies(original_logits, y, val_mask, test_mask),
                          storage={"dense": build_format_model(model, {}, "dense").storage()},
                          logits_sha256=tensor_sha256(original_logits))
            result["accuracy_matches_checkpoint"] = result["accuracy"]["test"] == checkpoint["test_accuracy"]
        else:
            weights = {name: getattr(model, name).weight.detach() for name in pruned_layers}
            pruned = prune_layers(weights, method, sparsity, pruning_seed)
            masks = pruned["masks"]

            reference = build_format_model(model, masks, "dense")
            storage = {"dense": reference.storage()}
            reference_logits = logits_in_batches(reference.to(device), x, batch, device)

            checks = {}
            for storage_format in COMPARED_FORMATS:
                variant = build_format_model(model, masks, storage_format)
                storage[storage_format] = variant.storage()
                logits = logits_in_batches(variant.to(device), x, batch, device)
                checks[storage_format] = format_comparison(reference_logits, logits, y, test_mask)
                checks[storage_format]["reference"] = f"pruned dense on {device}"

            result.update(actual_sparsity=pruned["summary"]["actual_sparsity"], pruning=pruned["summary"],
                          pruned_layers=pruned_layers,
                          accuracy=accuracies(reference_logits, y, val_mask, test_mask),
                          pruning_effect=pruning_comparison(original_logits, reference_logits, y, val_mask,
                                                            test_mask),
                          format_checks=checks, all_formats_equivalent=all(c["equivalent"] for c in checks.values()),
                          storage=storage, logits_sha256=tensor_sha256(reference_logits))
            if args.save_logits:
                path = output_dir / "logits" / f"{tag}_dense.pt"
                path.parent.mkdir(exist_ok=True)
                torch.save({"logits": reference_logits, "original_logits": original_logits}, path)
            if not result["all_formats_equivalent"]:
                failed.append(tag)

        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        acc = result["accuracy"]["test"]
        extra = "" if method == "none" else (
            f", loss {result['pruning_effect']['test_accuracy_loss_pp']:+.2f} pp, actual sparsity "
            f"{result['actual_sparsity']:.4f}, max format epsilon "
            f"{max(c['epsilon_logit'] for c in result['format_checks'].values()):.1e}, formats equivalent: "
            f"{result['all_formats_equivalent']}")
        print(f"[{i}/{len(todo)}] {tag}: test acc {acc:.4f}{extra}", flush=True)

    print(f"\nDone: {len(failed)} condition(s) where a format was not equivalent to pruned dense.")
    for tag in failed:
        print(f"  NOT EQUIVALENT: {tag}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

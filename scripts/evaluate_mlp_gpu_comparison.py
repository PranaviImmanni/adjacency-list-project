"""Accuracy and equivalence of every model in the GPU format comparison, for one model seed.

Over all 5,000 validation + 10,000 test images (chunks of eval_batch_size), on --device:
  pruned    pruned dense (the reference) and dense+mask, COO, CSR, adjacency list, each
            compared with it; accuracy and the pruning effect against the original model;
            the same condition in the original sweep (results/mlp/<baseline run>) must give
            the same test accuracy; Pranavi's CPU adjacency-list accuracy (seed 42) is compared.
  low_rank  dense factors (the reference) and factor adjacency lists; accuracy and the
            change against the original model; Pranavi's CPU rows (accuracy, edges, bytes,
            reconstruction error) are compared.

One JSON per model in results/mlp_gpu/<run>/evaluate. Existing files are skipped.

Example (lab GPU):
    python scripts/evaluate_mlp_gpu_comparison.py --seed 42 --device cuda:1
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

from src.gcn.config import PROJECT_ROOT, set_seed
from src.gcn.measurement import check_format_supported, configure_precision, device_info, resolve_device
from src.gcn.pruning import tensor_sha256
from src.mlp.config import COMPARISON_CONFIG_PATH, load_comparison_config
from src.mlp.data import load_for_checkpoint
from src.mlp.experiment import accuracies, format_comparison, logits_in_batches, pruning_comparison
from src.mlp.formats import build_format_model
from src.mlp.gpu_comparison import (PARTS, build, cross_check, pilot_plan, plan, pranavi_reference, prepare,
                                    spec_fields)
from src.mlp.model import checkpoint_identity, load_mlp_checkpoint

SCHEMA_VERSION = 1


def original_sweep_check(baseline_dir, tag, test_accuracy, logits_sha256, checkpoint_sha256):
    """The same condition in the original sweep, when it was evaluated from the same checkpoint."""
    path = baseline_dir / f"{tag}.json"
    shown = str(path.relative_to(PROJECT_ROOT))
    if not path.exists():
        return {"found": False, "path": shown, "reason": "no such result"}
    saved = json.loads(path.read_text())
    if saved.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        return {"found": False, "path": shown, "reason": "evaluated from a different checkpoint"}
    return {"found": True, "path": shown, "device": saved.get("device"),
            "test_accuracy_equal": saved["accuracy"]["test"] == test_accuracy,
            "logits_sha256_equal": saved.get("logits_sha256") == logits_sha256}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--checkpoint-run", default=None, help="trained checkpoints (default: the device tag)")
    parser.add_argument("--baseline-run", default=None, help="original sweep results to match (default: device tag)")
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    parser.add_argument("--pilot", action="store_true", help="only the pilot models")
    parser.add_argument("--config", default=str(COMPARISON_CONFIG_PATH))
    args = parser.parse_args()

    config = load_comparison_config(args.config)
    set_seed(0)
    device = resolve_device(args.device)
    for storage_format in ("coo", "csr"):
        check_format_supported(device, storage_format)
    precision = configure_precision(device, config["tf32"])
    run = args.device.replace(":", "")
    output_dir = PROJECT_ROOT / "results" / "mlp_gpu" / run / "evaluate"
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = PROJECT_ROOT / "results" / "mlp" / (args.baseline_run or run) / "evaluate"

    checkpoint_path = PROJECT_ROOT / "checkpoints" / f"fashion_mlp_seed{args.seed}_{args.checkpoint_run or run}.pt"
    model, checkpoint = load_mlp_checkpoint(checkpoint_path)
    identity = checkpoint_identity(checkpoint_path, checkpoint)
    data = load_for_checkpoint(checkpoint)
    x, y, val_mask, test_mask = data.eval_set()
    test_images = int(test_mask.sum())
    batch = config["eval_batch_size"]

    original_logits = logits_in_batches(build_format_model(model, {}, "dense").to(device), x, batch, device)
    specs = pilot_plan(config, args.seed) if args.pilot else plan(config, args.seed, "evaluate", args.parts)
    common = {"schema_version": SCHEMA_VERSION, "experiment": "mlp_gpu_comparison_evaluate", "status": "ok",
              "checkpoint": identity, **device_info(device), **precision, "dtype": config["dtype"],
              "eval_batch_size": batch, "eval_set": {"val": int(val_mask.sum()), "test": test_images}}
    failed = []
    for i, spec in enumerate(specs, 1):
        output = output_dir / f"{spec['tag']}.json"
        if output.exists():
            print(f"[{i}/{len(specs)}] exists, skipping {output.name}")
            continue
        result = {**common, "timestamp": datetime.now(timezone.utc).isoformat(), **spec_fields(spec),
                  "formats": spec["formats"], "reference_format": spec["reference_format"]}
        prepared = prepare(spec, model, config)

        if spec["kind"] == "original":
            storage = build(spec, prepared, model, "dense").storage()
            sha = tensor_sha256(original_logits)
            acc = accuracies(original_logits, y, val_mask, test_mask)
            result.update(accuracy=acc, storage={"dense": storage}, logits_sha256=sha, format_checks={},
                          all_formats_equivalent=True,
                          original_sweep=original_sweep_check(baseline_dir, spec["tag"], acc["test"], sha,
                                                              identity["sha256"]))
        else:
            reference_format = spec["reference_format"]
            reference = build(spec, prepared, model, reference_format)
            storage = {reference_format: reference.storage()}
            reference_logits = logits_in_batches(reference.to(device), x, batch, device)
            del reference
            checks, format_accuracy = {}, {reference_format: accuracies(reference_logits, y, val_mask, test_mask)}
            for storage_format in spec["formats"]:
                if storage_format == reference_format:
                    continue
                variant = build(spec, prepared, model, storage_format)
                storage[storage_format] = variant.storage()
                logits = logits_in_batches(variant.to(device), x, batch, device)
                del variant
                checks[storage_format] = format_comparison(reference_logits, logits, y, test_mask)
                checks[storage_format]["reference"] = f"{reference_format} on {device}"
                format_accuracy[storage_format] = accuracies(logits, y, val_mask, test_mask)
            acc = format_accuracy[reference_format]
            sha = tensor_sha256(reference_logits)
            result.update(accuracy=acc, format_accuracy=format_accuracy, storage=storage, format_checks=checks,
                          all_formats_equivalent=all(c["equivalent"] for c in checks.values()),
                          effect_vs_original=pruning_comparison(original_logits, reference_logits, y, val_mask,
                                                                test_mask),
                          logits_sha256=sha)
            if spec["kind"] == "pruned":
                summary = prepared["summary"]
                edges = sum(int(m.sum()) for m in prepared["masks"].values())
                result.update(actual_sparsity=summary["actual_sparsity"], pruning=summary, edges=edges,
                              original_sweep=original_sweep_check(baseline_dir, spec["tag"], acc["test"], sha,
                                                                  identity["sha256"]))
                list_format = "adjacency_list"
            else:
                summary = prepared["summary"]
                edges = summary["factor_edges"]
                result.update(low_rank=summary, edges=edges, actual_sparsity=None)
                list_format = "factor_adjacency"
            if list_format in format_accuracy:
                result["pranavi_cpu"] = cross_check(
                    pranavi_reference(spec), format_accuracy[list_format]["test"], edges, test_images,
                    stored_bytes=storage[list_format]["weight_bytes"],
                    reconstruction_mse=summary.get("mean_weight_reconstruction_mse"))
            if not result["all_formats_equivalent"]:
                failed.append(spec["tag"])

        output.write_text(json.dumps(result, indent=2))
        extra = ""
        if result["format_checks"]:
            worst = max(c["epsilon_logit"] for c in result["format_checks"].values())
            extra = f", max epsilon {worst:.1e}, formats equivalent: {result['all_formats_equivalent']}"
        cpu = result.get("pranavi_cpu") or {}
        if cpu.get("found"):
            extra += f", vs Pranavi's CPU result {cpu['difference_images']:+d} test images"
        print(f"[{i}/{len(specs)}] {spec['tag']}: test acc {result['accuracy']['test']:.4f}{extra}", flush=True)

    if failed:
        print(f"NOT EQUIVALENT: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Accuracy and equivalence of every follow-up variant of one model seed.

For each variant, over all 5,000 validation + 10,000 test images on --device:
  1. its reference (dense implementation of the same modified model);
  2. each deployment format (CSR, INT8), compared with (1): epsilon, allclose,
     agreement, accuracy difference, margins;
  3. the effect against the original dense model: accuracy change and
     prediction disagreement.
Fine-tuning and LoRA variants also record whether their masks equal the
original sweep's masks for the same seed and sparsity.

One JSON per variant in results/mlp_followup/<run>/evaluate/. Existing files
are skipped, so a rerun resumes.

Example (lab GPU):
    python scripts/evaluate_mlp_variants.py --seed 42 --experiments statistical --device cuda:1
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

from src.gcn.config import PROJECT_ROOT
from src.gcn.measurement import configure_precision, device_info, resolve_device
from src.gcn.pruning import tensor_sha256
from src.mlp.config import FOLLOWUP_CONFIG_PATH, FOLLOWUP_EXPERIMENTS, load_followup_config
from src.mlp.data import load_for_checkpoint
from src.mlp.experiment import accuracies, format_comparison, logits_in_batches, pruning_comparison
from src.mlp.formats import build_format_model
from src.mlp.model import load_mlp_checkpoint
from src.mlp.variants import (REFERENCE_FORMAT, build_variant, check_resident, load_variant_file, s_label,
                              variant_plan, variant_summary)

SCHEMA_VERSION = 1


def original_sweep(variant, sweep_dir):
    """The pruning-only result of the original sweep this variant starts from (finetune, lora)."""
    if variant["kind"] not in ("finetune", "lora"):
        return None
    path = sweep_dir / f"m{variant['model_seed']}_magnitude_{s_label(variant['params']['sparsity'])}.json"
    if not path.exists():
        return {"found": False, "path": str(path)}
    record = json.loads(path.read_text())
    same = all(record["pruning"]["layers"][name]["mask_sha256"] == layer["mask_sha256"]
               for name, layer in variant["pruning"]["layers"].items())
    return {"found": True, "path": str(path.relative_to(PROJECT_ROOT)), "masks_match": same,
            "pruning_only_accuracy": record["accuracy"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--experiments", nargs="+", choices=FOLLOWUP_EXPERIMENTS, default=None)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--checkpoint-run", default=None, help="default: the device tag, e.g. cuda1")
    parser.add_argument("--config", default=str(FOLLOWUP_CONFIG_PATH))
    args = parser.parse_args()

    config = load_followup_config(args.config)
    device = resolve_device(args.device)
    precision = configure_precision(device, config["tf32"])
    run = args.device.replace(":", "")
    checkpoint_run = args.checkpoint_run or run
    variants_dir = PROJECT_ROOT / "checkpoints" / "variants" / run
    output_dir = PROJECT_ROOT / "results" / "mlp_followup" / run / "evaluate"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir = PROJECT_ROOT / "results" / "mlp" / checkpoint_run / "evaluate"

    base_model, checkpoint = load_mlp_checkpoint(
        PROJECT_ROOT / "checkpoints" / f"fashion_mlp_seed{args.seed}_{checkpoint_run}.pt")
    data = load_for_checkpoint(checkpoint)
    x, y, val_mask, test_mask = data.eval_set()
    batch = config["eval_batch_size"]
    original_logits = logits_in_batches(build_format_model(base_model, {}, "dense").to(device), x, batch, device)
    original_accuracy = accuracies(original_logits, y, val_mask, test_mask)

    specs = [spec for experiment in (args.experiments or config["experiments"])
             for spec in variant_plan(experiment, config, args.seed)]
    missing = [spec["tag"] for spec in specs if not (variants_dir / f"{spec['tag']}.pt").exists()]
    if missing:
        sys.exit(f"missing variants (run the variants stage first): {missing}")

    failed = []
    for i, spec in enumerate(specs, 1):
        output = output_dir / f"{spec['tag']}.json"
        if output.exists():
            print(f"[{i}/{len(specs)}] exists, skipping {output.name}")
            continue
        variant = load_variant_file(variants_dir / f"{spec['tag']}.pt")
        record = {"schema_version": SCHEMA_VERSION, "experiment": "mlp_followup_evaluate", "status": "ok",
                  "timestamp": datetime.now(timezone.utc).isoformat(), **device_info(device), **precision,
                  "dtype": config["dtype"], "eval_batch_size": batch,
                  "eval_set": {"val": int(val_mask.sum()), "test": int(test_mask.sum())},
                  "variant": variant_summary(variant), "original_accuracy": original_accuracy,
                  "original_sweep": original_sweep(variant, sweep_dir), "formats": {}}
        references = {}
        for storage_format in variant["deploy_formats"]:
            reference_format = REFERENCE_FORMAT[storage_format]
            if reference_format not in references:
                reference = build_variant(variant, reference_format)
                references[reference_format] = (logits_in_batches(reference.to(device), x, batch, device),
                                                reference.storage())
            reference_logits, reference_storage = references[reference_format]
            deployed = build_variant(variant, storage_format).to(device)
            if not check_resident(deployed):
                raise RuntimeError(f"{spec['tag']} {storage_format}: unexpected tensors on the device")
            logits = logits_in_batches(deployed, x, batch, device)
            check = format_comparison(reference_logits, logits, y, test_mask)
            check["reference"] = f"{reference_format} on {device}"
            record["formats"][storage_format] = {
                "accuracy": accuracies(logits, y, val_mask, test_mask),
                "reference_format": reference_format,
                "reference_accuracy": accuracies(reference_logits, y, val_mask, test_mask),
                "check": check, "storage": deployed.storage(), "reference_storage": reference_storage,
                "effect_vs_original": pruning_comparison(original_logits, logits, y, val_mask, test_mask),
                "int_mm_support": getattr(deployed, "support", None),
                "logits_sha256": tensor_sha256(logits),
            }
        record["all_formats_equivalent"] = all(f["check"]["equivalent"] for f in record["formats"].values())
        output.write_text(json.dumps(record, indent=2))

        line = ", ".join(
            f"{fmt} test acc {r['accuracy']['test']:.4f} (eps {r['check']['epsilon_logit']:.1e}, "
            f"equivalent {r['check']['equivalent']}), stored {r['storage']['stored_bytes']:,} B"
            for fmt, r in record["formats"].items())
        sweep = record["original_sweep"]
        if sweep and sweep.get("found"):
            line += f"; masks match original sweep: {sweep['masks_match']}"
        print(f"[{i}/{len(specs)}] {spec['tag']}: {line}", flush=True)
        if not record["all_formats_equivalent"] or (sweep and sweep.get("found") and not sweep["masks_match"]):
            failed.append(spec["tag"])

    print(f"\nDone: {len(failed)} variant(s) with a failed check.")
    for tag in failed:
        print(f"  CHECK FAILED: {tag}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

"""Make the model variants of one follow-up experiment for one model seed.

Prunes (and for finetune / protect_fc3 / lora, trains on --device) and saves
each variant to checkpoints/variants/<run>/<tag>.pt, with its metadata (no
tensors) in results/mlp_followup/<run>/variants/<tag>.json. Existing variants
are skipped, so a rerun resumes. Selection uses validation data only.

Example (lab GPU):
    python scripts/make_mlp_variants.py --experiment statistical --seed 42 --device cuda:1
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

from src.gcn.config import PROJECT_ROOT
from src.gcn.measurement import configure_precision, resolve_device
from src.mlp.config import FOLLOWUP_CONFIG_PATH, FOLLOWUP_EXPERIMENTS, load_followup_config
from src.mlp.data import load_for_checkpoint
from src.mlp.model import checkpoint_identity, load_mlp_checkpoint
from src.mlp.variants import make_variant, save_variant, variant_plan, variant_summary


def describe(variant):
    parts = []
    pruning = variant["pruning"]
    if pruning:
        layers = " / ".join(f"{100 * layer['actual_sparsity']:.1f}" for layer in pruning["layers"].values())
        parts.append(f"retained {pruning['retained_weights']:,} ({100 * pruning['actual_sparsity']:.2f}% sparse; "
                     f"per layer {layers}%)")
    if variant["matching"] and "mismatch_bytes" in variant["matching"]:
        parts.append(f"stored bytes {variant['matching']['achieved_weight_bytes']:,} vs target "
                     f"{variant['matching']['target_weight_bytes']:,} ({variant['matching']['mismatch_bytes']:+,} B)")
    if variant["recovery"]:
        selected = variant["recovery"]["selected"]
        parts.append(f"selected lr {selected['learning_rate']:g}, epoch {selected['epoch']}, "
                     f"val acc {selected['val_accuracy']:.4f}")
    return "; ".join(parts) or "unpruned"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, choices=FOLLOWUP_EXPERIMENTS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N (recovery training runs here)")
    parser.add_argument("--checkpoint-run", default=None,
                        help="which trained checkpoints to start from (default: the device tag, e.g. cuda1)")
    parser.add_argument("--config", default=str(FOLLOWUP_CONFIG_PATH))
    args = parser.parse_args()

    config = load_followup_config(args.config)
    device = resolve_device(args.device)
    configure_precision(device, config["tf32"])
    run = args.device.replace(":", "")
    checkpoint_path = PROJECT_ROOT / "checkpoints" / f"fashion_mlp_seed{args.seed}_{args.checkpoint_run or run}.pt"
    variants_dir = PROJECT_ROOT / "checkpoints" / "variants" / run
    meta_dir = PROJECT_ROOT / "results" / "mlp_followup" / run / "variants"
    meta_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_mlp_checkpoint(checkpoint_path)
    identity = checkpoint_identity(checkpoint_path, checkpoint)
    data = load_for_checkpoint(checkpoint)
    plan = variant_plan(args.experiment, config, args.seed)
    for i, spec in enumerate(plan, 1):
        path = variants_dir / f"{spec['tag']}.pt"
        if path.exists():
            print(f"[{i}/{len(plan)}] exists, skipping {path.name}")
            continue
        variant = make_variant(spec, model, checkpoint, identity, data, config, device)
        save_variant(variant, path)
        (meta_dir / f"{spec['tag']}.json").write_text(json.dumps(variant_summary(variant), indent=2))
        print(f"[{i}/{len(plan)}] {spec['tag']}: {describe(variant)}", flush=True)


if __name__ == "__main__":
    main()

"""Tables, CSVs and charts for the MLP weight-pruning experiment.

Reads results/mlp/<device>/{baseline,evaluate,benchmark}/*.json and writes
summary.md, one CSV per table and the charts.

Aggregation:
  unit       one (model seed, pruning seed) pair; pruning seed only for random
  accuracy   mean over units, whiskers = min-max over units
  latency    per unit: median of the per-repetition medians; then the mean
             over units, whiskers = min-max over units
  peak       per unit: max over repetitions; then mean, min-max over units
             (likewise resident-before-inference and peak - resident)
  storage    exact bytes (identical across seeds for a given method and level)
  epsilon    worst case over units (max epsilon, min agreement, ...)

Sparsity axis: positions are -log10(1 - sparsity), i.e. a log scale of the
retained fraction, so 90 / 95 / 99% are readable; ticks show the sparsity.
"""

import csv
import json
import math
import statistics
import textwrap
from collections import defaultdict
from pathlib import Path

from src.gcn.reporting import (BASELINE, GRID, INK, INK_SECONDARY, MUTED, SURFACE, _pyplot, human_bytes,
                               markdown_table, text_table)
from src.gcn.synthetic_reporting import _spread

from .formats import FORMAT_LABELS, FORMATS
from .weight_pruning import METHODS

# Categorical slots 1-3 (methods, validated all-pairs) and 1-4 (formats, validated
# adjacent). Aqua and yellow are under 3:1 contrast, so every chart also has a
# distinct marker per series, direct end labels, and a table with every value.
METHOD_STYLE = {
    "magnitude": {"color": "#2a78d6", "marker": "o", "label": "magnitude"},
    "topk_per_neuron": {"color": "#eb6834", "marker": "s", "label": "top-K per neuron"},
    "random": {"color": "#1baf7a", "marker": "^", "label": "random"},
}
FORMAT_STYLE = {
    "dense": {"color": "#2a78d6", "marker": "o", "label": FORMAT_LABELS["dense"]},
    "dense_mask": {"color": "#eb6834", "marker": "s", "label": FORMAT_LABELS["dense_mask"]},
    "coo": {"color": "#1baf7a", "marker": "^", "label": FORMAT_LABELS["coo"]},
    "csr": {"color": "#eda100", "marker": "D", "label": FORMAT_LABELS["csr"]},
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read(directory, experiment):
    for path in sorted(Path(directory).glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"skipping unreadable {path}")
            continue
        if data.get("experiment") == experiment and data.get("status", "ok") == "ok":
            yield data


def load_mlp_results(results_dir):
    """(baselines, evaluations, benchmarks); the latest file wins for a repeated run."""
    root = Path(results_dir)
    baseline_dir = root / "baseline"
    if not baseline_dir.exists() and root.name.startswith("pilot_"):  # the pilot shares the main run's checkpoints
        baseline_dir = root.parent / root.name[len("pilot_"):] / "baseline"
    baselines = {r["model_seed"]: r for r in _read(baseline_dir, "mlp_baseline")}
    evaluations = {(r["model_seed"], r["method"], r["requested_sparsity"], r["pruning_seed"]): r
                   for r in _read(root / "evaluate", "mlp_evaluate")}
    benchmarks = {}
    for r in _read(root / "benchmark", "mlp_benchmark"):
        key = (r["model_seed"], r["method"], r["requested_sparsity"], r["pruning_seed"], r["format"],
               r["batch_size"], r["repetition"])
        if key not in benchmarks or r["timestamp"] > benchmarks[key]["timestamp"]:
            benchmarks[key] = r
    return list(baselines.values()), list(evaluations.values()), list(benchmarks.values())


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": statistics.fmean(values), "min": min(values), "max": max(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0, "n": len(values)}


def accuracy_points(evaluations):
    """{(method, sparsity): stats of test accuracy, loss, disagreement, actual sparsity}; 'none' at 0."""
    groups = defaultdict(list)
    for r in evaluations:
        groups[(r["method"], r["requested_sparsity"])].append(r)
    out = {}
    for key, runs in groups.items():
        pruned = key[0] != "none"
        out[key] = {
            "test": _stats([r["accuracy"]["test"] for r in runs]),
            "val": _stats([r["accuracy"]["val"] for r in runs]),
            "loss_pp": _stats([r["pruning_effect"]["test_accuracy_loss_pp"] for r in runs]) if pruned else None,
            "disagreement": _stats([r["pruning_effect"]["prediction_disagreement_test"] for r in runs])
            if pruned else None,
            "actual_sparsity": _stats([r["actual_sparsity"] for r in runs]),
            "units": len(runs),
        }
    return out


def storage_points(evaluations):
    """{(method, sparsity, format): weight bytes}; the original model under ('none', 0, 'dense')."""
    groups = defaultdict(set)
    for r in evaluations:
        for storage_format, storage in r["storage"].items():
            groups[(r["method"], r["requested_sparsity"], storage_format)].add(storage["weight_bytes"])
    return {key: max(values) for key, values in groups.items()}  # identical across seeds


def epsilon_points(evaluations):
    """Worst case over units of each format check, {(method, sparsity, format): {...}}."""
    groups = defaultdict(list)
    for r in evaluations:
        for storage_format, check in (r.get("format_checks") or {}).items():
            groups[(r["method"], r["requested_sparsity"], storage_format)].append(check)
    return {key: {
        "units": len(checks),
        "max_epsilon": max(c["epsilon_logit"] for c in checks),
        "max_mean_abs": max(c["mean_abs_logit_diff"] for c in checks),
        "all_allclose": all(c["allclose"] for c in checks),
        "min_agreement": min(c["prediction_agreement_all"] for c in checks),
        "max_abs_accuracy_pp": max(abs(c["accuracy_difference_pp"]) for c in checks),
        "min_margin_condition": min(c["margin"]["test"]["margin_condition_fraction"] for c in checks),
        "violations": sum(c["margin"]["all"]["disagreements_satisfying_condition"] for c in checks),
        "all_equivalent": all(c["equivalent"] for c in checks),
        "all_bitwise": all(c.get("bitwise_equal", False) for c in checks),
    } for key, checks in groups.items()}


def benchmark_points(benchmarks):
    """{(method, sparsity, format, batch): {'latency': stats, 'peak': stats, ...}} over units."""
    units = defaultdict(list)
    for r in benchmarks:
        units[(r["method"], r["requested_sparsity"], r["format"], r["batch_size"],
               r["model_seed"], r["pruning_seed"])].append(r)
    groups = defaultdict(lambda: {"latency": [], "peak": [], "resident": [], "extra": [], "reps": [],
                                  "debug": False, "checks": []})
    for (method, sparsity, storage_format, batch, *_), reps in units.items():
        g = groups[(method, sparsity, storage_format, batch)]
        g["latency"].append(statistics.median(r["latency_ms"]["median"] for r in reps))
        for key, field in (("peak", "cuda_peak_allocated_bytes"),
                           ("resident", "cuda_allocated_before_inference_bytes")):
            values = [r[field] for r in reps if r.get(field) is not None]
            g[key].append(max(values) if values else None)
        extra = [r["cuda_peak_allocated_bytes"] - r["cuda_allocated_before_inference_bytes"] for r in reps
                 if r.get("cuda_peak_allocated_bytes") is not None
                 and r.get("cuda_allocated_before_inference_bytes") is not None]
        g["extra"].append(max(extra) if extra else None)
        g["reps"].append(len(reps))
        g["debug"] |= any(r["latency_is_debug_only"] for r in reps)
        g["checks"] += [r["format_check"] for r in reps if r.get("format_check")]
    return {key: {"latency": _stats(g["latency"]), "peak": _stats(g["peak"]), "resident": _stats(g["resident"]),
                  "peak_minus_resident": _stats(g["extra"]), "units": len(g["latency"]),
                  "min_reps": min(g["reps"]), "debug": g["debug"], "checks": g["checks"]}
            for key, g in groups.items()}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _pct(x, digits=2):
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def _range(stats, fmt):
    if stats is None:
        return "n/a"
    if stats["n"] == 1 or stats["min"] == stats["max"]:
        return fmt(stats["mean"])
    return f"{fmt(stats['mean'])} ({fmt(stats['min'])} - {fmt(stats['max'])})"


def _memory_row(method_label, sparsity_label, format_label, batch, point):
    """Resident, peak and their difference, each mean (min - max) over units."""
    resident, peak = point["resident"], point["peak"]
    if resident is None:
        return [method_label, sparsity_label, format_label, batch, "n/a", _range(peak, _compact_bytes), "n/a"]
    extra = point.get("peak_minus_resident")
    return [method_label, sparsity_label, format_label, batch, _range(resident, _compact_bytes),
            _range(peak, _compact_bytes), _range(extra, _compact_bytes) if extra else "n/a"]


def _sparsity_label(s):
    return f"{100 * s:g}%"


def _methods_present(points):
    return [m for m in METHODS if any(key[0] == m for key in points)]


def _sparsities(points, method=None):
    return sorted({key[1] for key in points if key[0] != "none" and (method is None or key[0] == method)})


RUN_COLUMNS = (
    "experiment", "dataset", "device", "device_name", "model_seed", "method", "pruning_seed", "requested_sparsity",
    "actual_sparsity", "format", "batch_size", "repetition", "weight_storage_bytes", "bias_bytes",
    "resident_tensor_bytes", "cuda_allocated_before_inference_bytes", "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes", "latency_median_ms", "latency_mean_ms", "latency_std_ms", "measured_runs",
    "energy_joules", "format_check_equivalent", "format_check_epsilon", "latency_is_debug_only", "checkpoint_sha256",
)


def run_rows(benchmarks):
    """One flat row per benchmark run (RUN_COLUMNS), for merging with the harness CSV."""
    rows = []
    for r in sorted(benchmarks, key=lambda r: (r["repetition"], r["model_seed"], r["method"], r["requested_sparsity"],
                                                 str(r["pruning_seed"]), r["format"], r["batch_size"])):
        check = r.get("format_check") or {}
        rows.append([
            r["experiment"], r.get("dataset", "fashion_mnist"), r["device"], r.get("device_name"), r["model_seed"],
            r["method"], r["pruning_seed"], r["requested_sparsity"], r.get("actual_sparsity"), r["format"],
            r["batch_size"], r["repetition"], r.get("weight_storage_bytes"), r.get("bias_bytes"),
            r.get("resident_tensor_bytes"), r.get("cuda_allocated_before_inference_bytes"),
            r.get("cuda_peak_allocated_bytes"), r.get("cuda_peak_reserved_bytes"), r["latency_ms"]["median"],
            r["latency_ms"].get("mean"), r["latency_ms"].get("std"), r.get("measured_runs"), r.get("energy_joules"),
            check.get("equivalent"), check.get("epsilon_logit"), r["latency_is_debug_only"],
            (r.get("checkpoint") or {}).get("sha256"),
        ])
    return rows


def _write_csv(path, headers, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _x(sparsity):
    return -math.log10(1 - sparsity)


def line_panels(plt, path, panels, styles, title, subtitle, y_fmt, log_y=False, reference_label="original dense"):
    """Small multiples sharing one legend. panels: [(panel title, {key: [(sparsity, mean, low, high)]}, ref)].

    `ref` is a neutral horizontal reference (the original dense model) or None.
    Line ends in the last panel are labeled directly.
    """
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator

    n = len(panels)
    panel_in = 5.2 if n == 1 else 3.0
    gap_in, left_in, right_in, bottom_in = 0.75, 0.9, 1.8, 0.65
    width_in = left_in + n * panel_in + (n - 1) * gap_in + right_in
    subtitle_lines = textwrap.wrap(subtitle, int((width_in - left_in - 0.3) * 14.5))
    top_in = 1.2 + 0.17 * len(subtitle_lines)
    height_in = top_in + 3.0 + bottom_in
    fig = plt.figure(figsize=(width_in, height_in), dpi=100)
    keys = []

    for k, (panel_title, series, ref) in enumerate(panels):
        left = left_in + k * (panel_in + gap_in)
        ax = fig.add_axes([left / width_in, bottom_in / height_in, panel_in / width_in,
                           1 - (top_in + bottom_in) / height_in])
        sparsities = sorted({p[0] for points in series.values() for p in points})
        for key, points in series.items():
            if key not in keys:
                keys.append(key)
            style = styles[key]
            xs, means = [_x(p[0]) for p in points], [p[1] for p in points]
            ax.plot(xs, means, color=style["color"], linewidth=1.5, solid_joinstyle="round",
                    solid_capstyle="round", zorder=3)
            ax.plot(xs, means, linestyle="none", marker=style["marker"], markersize=6.5,
                    markerfacecolor=style["color"], markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
            for x, (_, _, low, high) in zip(xs, points):
                if high > low:
                    ax.plot([x, x], [low, high], color=style["color"], linewidth=1, zorder=2)
        if ref is not None:
            ax.axhline(ref, color=MUTED, linewidth=1, zorder=1)
        ax.set_xticks([_x(s) for s in sparsities])
        ax.set_xticklabels([_sparsity_label(s) for s in sparsities])
        ax.set_xlim(_x(sparsities[0]) - 0.12, _x(sparsities[-1]) + 0.12)
        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: y_fmt(v)))
        ax.tick_params(which="both", length=0, labelsize=9, labelcolor=MUTED, pad=6)
        ax.yaxis.grid(True, which="major", color=GRID, linewidth=0.75)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
            ax.spines[side].set_linewidth(0.75)
        ax.set_xlabel("sparsity (log scale of retained fraction)", color=MUTED, fontsize=8.5, labelpad=6)
        if n > 1:
            ax.set_title(panel_title, loc="left", fontsize=10, color=INK, pad=8)

        if k == n - 1:
            # Direct labels in ink just right of the plot area (clear of the reference line), spread
            # apart and kept within the plot's height; the legend carries the colors.
            fig.canvas.draw()
            to_pixels, to_data = ax.transData.transform, ax.transData.inverted().transform
            ends = {styles[key]["label"]: points[-1][1] for key, points in series.items()}
            if ref is not None:
                ends[reference_label] = ref
            box = ax.get_window_extent()
            x_end = _x(sparsities[-1])
            ys = _spread([to_pixels((x_end, value))[1] for value in ends.values()], min_gap=13)
            shift = max(0.0, box.y0 + 6 - min(ys)) - max(0.0, max(ys) - (box.y1 - 6))
            gap_px = box.x1 - to_pixels((x_end, 0))[0] + 8
            for (label, value), y_px in zip(ends.items(), ys):
                y_data = to_data((to_pixels((x_end, 0))[0], y_px + shift))[1]
                ax.annotate(f"{label}  {y_fmt(value)}", xy=(x_end, y_data), xytext=(gap_px * 72 / fig.dpi, 0),
                            textcoords="offset points", va="center", ha="left", fontsize=8.5,
                            color=INK_SECONDARY, annotation_clip=False)

    left = left_in / width_in
    fig.text(left, 1 - 0.32 / height_in, title, fontsize=12.5, fontweight="bold", color=INK, va="baseline")
    for i, line in enumerate(subtitle_lines):
        fig.text(left, 1 - (0.55 + 0.17 * i) / height_in, line, fontsize=9, color=INK_SECONDARY, va="baseline")
    handles = [Line2D([], [], color=styles[key]["color"], linewidth=1.5, marker=styles[key]["marker"],
                      markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.5) for key in keys]
    labels = [styles[key]["label"] for key in keys]
    if any(ref is not None for _, _, ref in panels):
        handles.append(Line2D([], [], color=MUTED, linewidth=1))
        labels.append(reference_label)
    legend_top = 0.68 + 0.17 * (len(subtitle_lines) - 1)
    legend = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(left - 0.01, 1 - legend_top / height_in),
                        ncol=len(labels), frameon=False, fontsize=9.5, handlelength=2.2, columnspacing=1.6)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _compact_bytes(v):
    return human_bytes(v).replace(".0 ", " ")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def summarize_mlp(results_dir, out_dir):
    """Write summary.md, CSVs and charts. Returns (text report, written files)."""
    baselines, evaluations, benchmarks = load_mlp_results(results_dir)
    if not (baselines or evaluations or benchmarks):
        raise FileNotFoundError(f"no MLP baseline, evaluate or benchmark results under {results_dir}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    text, written = [], []
    debug = any(r["latency_is_debug_only"] for r in benchmarks)
    device = (benchmarks or evaluations or baselines)[0]["device"]
    seeds = sorted({r["model_seed"] for r in evaluations or baselines})
    md = ["# MLP weight pruning: Fashion-MNIST, dense vs sparse weight storage", "",
          f"Results from `{results_dir}` on `{device}`. Model seeds: {', '.join(map(str, seeds))}. "
          "Ranges in parentheses and chart whiskers are min - max over model seeds (and pruning seeds for "
          "random). No fine-tuning after pruning.", ""]
    if debug:
        md += ["**Latency and memory here are CPU debug numbers, not GPU results.**", ""]

    def section(title, headers, rows, note=None, image=None, csv_name=None):
        text.extend([title, text_table(headers, rows), ""])
        md.extend([f"## {title}", ""])
        if image:
            md.extend([f"![{title}]({image})", ""])
        md.extend([markdown_table(headers, rows), ""])
        if note:
            md.extend([note, ""])
            text.extend([note, ""])
        if csv_name:
            _write_csv(out_dir / csv_name, headers, rows)
            written.append(out_dir / csv_name)

    def chart(name, *args, **kwargs):
        if plt is None:
            return None
        line_panels(plt, out_dir / name, *args, **kwargs)
        written.append(out_dir / name)
        return name

    # 1. Dense baseline
    if baselines:
        rows = [[r["model_seed"], r["best_epoch"], _pct(r["validation_accuracy"]), _pct(r["test_accuracy"])]
                for r in sorted(baselines, key=lambda r: r["model_seed"])]
        test = _stats([r["test_accuracy"] for r in baselines])
        note = f"Test accuracy over seeds: mean {_pct(test['mean'])}, std {100 * test['std']:.2f} pp."
        section("Dense baseline", ["Model seed", "Best epoch", "Val acc", "Test acc"], rows, note,
                csv_name="baseline.csv")

    # 2. Accuracy vs sparsity
    acc = accuracy_points(evaluations)
    original = acc.get(("none", 0.0))
    if acc:
        methods = _methods_present(acc)
        rows = [["original", "0%", "0%", _range(original["test"], _pct), "-", "-", original["units"]]] if original else []
        for method in methods:
            for s in _sparsities(acc, method):
                p = acc[(method, s)]
                rows.append([METHOD_STYLE[method]["label"], _sparsity_label(s), _pct(p["actual_sparsity"]["mean"]),
                             _range(p["test"], _pct), _range(p["loss_pp"], lambda v: f"{v:+.2f}"),
                             _range(p["disagreement"], _pct), p["units"]])
        series = {m: [(s, 100 * acc[(m, s)]["test"]["mean"], 100 * acc[(m, s)]["test"]["min"],
                       100 * acc[(m, s)]["test"]["max"]) for s in _sparsities(acc, m)] for m in methods}
        image = chart("accuracy_vs_sparsity.png", [("", series, 100 * original["test"]["mean"] if original else None)],
                      METHOD_STYLE, "Test accuracy vs sparsity, by pruning method",
                      f"Fashion-MNIST, MLP 784-300-100-10, no fine-tuning. Mean over {len(seeds)} model "
                      f"seed{'s' if len(seeds) != 1 else ''}; "
                      "whiskers min - max.", lambda v: f"{v:.0f}%")
        section("Accuracy vs sparsity", ["Method", "Target", "Actual", "Test acc", "Loss (pp)", "Disagreement",
                                         "Units"], rows,
                "Loss = original - pruned test accuracy in percentage points (negative = pruning helped). "
                "Disagreement = share of test images whose predicted class changed.", image, "accuracy.csv")

    # 3. Weight storage
    storage = storage_points(evaluations)
    if storage:
        original_bytes = storage.get(("none", 0.0, "dense"))
        rows, series = [], defaultdict(list)
        for method in _methods_present(storage):
            for s in _sparsities(storage, method):
                cells = []
                for storage_format in FORMATS:
                    b = storage.get((method, s, storage_format))
                    cells.append("n/a" if b is None else
                                 f"{b:,} ({100 * b / original_bytes:.1f}%)" if original_bytes else f"{b:,}")
                    if method == "magnitude" and b is not None:
                        series[storage_format].append((s, b, b, b))
                rows.append([METHOD_STYLE[method]["label"], _sparsity_label(s), *cells])
        image = chart("weight_storage_vs_sparsity.png", [("", dict(series), original_bytes)], FORMAT_STYLE,
                      "Weight storage vs sparsity, by format",
                      "Exact bytes of the stored weight tensors (FP32 values, int64 indices, bool mask), magnitude "
                      "pruning; biases excluded.", _compact_bytes, log_y=True) if series else None
        section("Weight storage (bytes, % of original dense)",
                ["Method", "Sparsity", *[FORMAT_LABELS[f] for f in FORMATS]], rows,
                f"Original dense weights: {original_bytes:,} bytes. Biases "
                f"({evaluations[0]['storage']['dense']['bias_bytes']:,} bytes) are dense in every format."
                if original_bytes else None, image, "weight_storage.csv")

    # 4. Latency and peak memory
    bench = benchmark_points(benchmarks)
    if bench:
        batches = sorted({key[3] for key in bench})
        device_note = ("CPU debug timing, not a GPU result. " if debug else "") + (
            f"Per unit: median of per-repetition medians ({benchmarks[0]['measured_runs']} timed passes each); "
            "mean over units, whiskers min - max.")
        lat_rows, peak_rows = [], []
        for method in _methods_present(bench):
            panels, peak_panels = [], []
            for batch in batches:
                ref = bench.get(("none", 0.0, "dense", batch))
                series, peak_series = defaultdict(list), defaultdict(list)
                for s in _sparsities(bench, method):
                    for storage_format in FORMATS:
                        p = bench.get((method, s, storage_format, batch))
                        if p is None:
                            continue
                        lat = p["latency"]
                        series[storage_format].append((s, lat["mean"], lat["min"], lat["max"]))
                        lat_rows.append([METHOD_STYLE[method]["label"], _sparsity_label(s),
                                         FORMAT_LABELS[storage_format], batch,
                                         _range(lat, lambda v: f"{v:.4f}"), p["units"], p["min_reps"]])
                        if p["peak"]:
                            peak = p["peak"]
                            peak_series[storage_format].append((s, peak["mean"], peak["min"], peak["max"]))
                            peak_rows.append(_memory_row(METHOD_STYLE[method]["label"], _sparsity_label(s),
                                                         FORMAT_LABELS[storage_format], batch, p))
                panels.append((f"batch size {batch}", dict(series), ref["latency"]["mean"] if ref else None))
                peak_panels.append((f"batch size {batch}", dict(peak_series),
                                    ref["peak"]["mean"] if ref and ref["peak"] else None))
            label = METHOD_STYLE[method]["label"]
            chart(f"latency_vs_sparsity_{method}.png", panels, FORMAT_STYLE,
                  f"Inference latency vs sparsity and batch size ({label} pruning)", device_note,
                  lambda v: f"{v:.3g} ms")
            if any(series for _, series, _ in peak_panels):
                chart(f"peak_memory_vs_sparsity_{method}.png", peak_panels, FORMAT_STYLE,
                      f"Total peak GPU memory vs sparsity and batch size ({label} pruning)",
                      "torch.cuda.max_memory_allocated during the timed passes: weights, biases, input, "
                      "activations and library workspace (the dense formats include cuBLAS's fixed 8.1 MiB "
                      "workspace). Mean over units, whiskers min - max.", _compact_bytes)
        original_rows = [["original", "0%", "dense", batch, _range(bench[("none", 0.0, "dense", batch)]["latency"],
                                                                  lambda v: f"{v:.4f}"),
                          bench[("none", 0.0, "dense", batch)]["units"], bench[("none", 0.0, "dense", batch)]["min_reps"]]
                         for batch in batches if ("none", 0.0, "dense", batch) in bench]
        section("Latency (ms)", ["Method", "Sparsity", "Format", "Batch", "Latency (ms)", "Units", "Reps"],
                original_rows + lat_rows, device_note + " Charts: latency_vs_sparsity_<method>.png.",
                csv_name="latency.csv")
        if peak_rows:
            original_peak = [_memory_row("original", "0%", "dense", batch, bench[("none", 0.0, "dense", batch)])
                             for batch in batches
                             if ("none", 0.0, "dense", batch) in bench and bench[("none", 0.0, "dense", batch)]["peak"]]
            section("Total peak GPU memory",
                    ["Method", "Sparsity", "Format", "Batch", "Resident before inference", "Peak allocated",
                     "Peak - resident"], original_peak + peak_rows,
                    "Resident = torch.cuda.memory_allocated after loading, before the first pass: the stored "
                    "weights, biases and input batch. Peak = torch.cuda.max_memory_allocated over the timed "
                    "passes (reset after warm-up). Peak - resident = activations, temporaries and library "
                    "workspace. For pruned dense and original dense it includes cuBLAS's workspace, which "
                    "PyTorch allocates through its caching allocator on the first dense matmul and keeps "
                    "(8,519,680 B = 2 x 4 MiB + 8 x 16 KiB by default, the same in every run). Dense+mask "
                    "adds the W*mask temporary on top of that. The sparse formats run on cuSPARSE and do not "
                    "use that workspace, so compare formats on both columns; the weights alone are in the "
                    "weight storage table. Charts: peak_memory_vs_sparsity_<method>.png.",
                    csv_name="peak_memory.csv")
        else:
            md.extend(["## Total peak GPU memory", "", "Not available: these runs are not on CUDA.", ""])

    # 5. Storage equivalence and epsilon
    eps = epsilon_points(evaluations)
    if eps:
        rows = [[METHOD_STYLE[m]["label"], _sparsity_label(s), FORMAT_LABELS[f], f"{e['max_epsilon']:.1e}",
                 f"{e['max_mean_abs']:.1e}", "yes" if e["all_allclose"] else "NO", _pct(e["min_agreement"]),
                 f"{e['max_abs_accuracy_pp']:.2f}", _pct(e["min_margin_condition"]), e["violations"],
                 "yes" if e["all_equivalent"] else "NO", "yes" if e["all_bitwise"] else "no"]
                for (m, s, f), e in sorted(eps.items(), key=lambda kv: (METHODS.index(kv[0][0]), kv[0][1],
                                                                         FORMATS.index(kv[0][2])))]
        section("Storage-format equivalence and epsilon (vs pruned dense, same device)",
                ["Method", "Sparsity", "Format", "Max epsilon", "Max mean |dz|", "allclose", "Min agreement",
                 "Max |acc diff| pp", "Min margin cond. (test)", "Violations", "Equivalent", "Bitwise equal"], rows,
                "Epsilon = max |logit - pruned-dense logit| over validation + test images. Margin condition: "
                "margin > 2 x epsilon guarantees an unchanged prediction; violations (a changed prediction that "
                "met the condition) must be 0. Tolerance: allclose(atol=1e-5, rtol=1e-4).", csv_name="epsilon.csv")
        bench_checks = defaultdict(list)
        for (_, _, storage_format, batch), p in bench.items():
            bench_checks[(storage_format, batch)] += p["checks"]
        if bench_checks:
            rows = [[FORMAT_LABELS[f], b, len(c), "yes" if all(x["equivalent"] for x in c) else "NO",
                     f"{max(x['epsilon_logit'] for x in c):.1e}"]
                    for (f, b), c in sorted(bench_checks.items(), key=lambda kv: (FORMATS.index(kv[0][0]), kv[0][1]))
                    if c]
            section("Benchmark-batch format checks", ["Format", "Batch", "Runs", "All equivalent", "Max epsilon"],
                    rows, "Each benchmark run compares its timed batch against pruned dense on the same device.")

    # 6. Energy
    md.extend(["## Energy", "", "Not measured: no reliable energy method has been established for the harness yet.",
               ""])

    # 7. Every benchmark run, flat, for the merged results CSV
    if benchmarks:
        _write_csv(out_dir / "runs.csv", RUN_COLUMNS, run_rows(benchmarks))
        written.append(out_dir / "runs.csv")
        md.extend(["## Per-run data", "", f"`runs.csv`: one row per benchmark run ({len(benchmarks)} runs), "
                   "columns for merging with the timing/energy harness results.", ""])

    summary = out_dir / "summary.md"
    summary.write_text("\n".join(md))
    written.insert(0, summary)
    return "\n".join(text), written

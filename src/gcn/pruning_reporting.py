"""Tables and charts for the pruning experiments and the epsilon/margin analysis.

Reads, for each dataset and device:
  results/pruning/<dataset>_<device>/         scripts/benchmark_pruning.py runs (CSR) + logits/ + orderings/
  results/pruning/verify_<dataset>_<device>/  scripts/compare_gcn_formats.py on the selected pruned graphs
  <formats dir> (results/gpu)                 the unpruned dense/COO/CSR comparison and benchmarks

Aggregation:
  - accuracy, disagreement and logit changes come from one repetition
    (they are identical across repetitions; logits_sha256 checks it);
  - latency per condition: median of the repetition medians, range = min-max
    of the repetition medians. Random pruning pools its three pruning seeds;
  - random pruning across seeds: mean, with the min-max over the three seeds.
Ranges are observed spreads, not confidence intervals.

The margin tables are recomputed in float64 from the saved logits, so they
can be regenerated without the GPU.
"""

import json
import statistics
import textwrap
from collections import defaultdict
from pathlib import Path

import torch

from .config import PROJECT_ROOT
from .error_analysis import exact_accuracy, logit_change, margin_analysis
from .reporting import (BASELINE, GRID, INK, INK_SECONDARY, MUTED, SURFACE, _ordered_formats, _pyplot, _yes_no,
                        human_bytes, latency_stats, load_results, markdown_table, text_table)
from .synthetic_reporting import _spread

# Fixed per method (categorical slots 1-2, the validated dense/COO pair). The
# original graph is a neutral reference, not a third series.
METHOD_STYLE = {
    "random": {"color": "#2a78d6", "marker": "o", "label": "random"},
    "similarity": {"color": "#eb6834", "marker": "s", "label": "feature similarity"},
}
METHOD_ORDER = ("none", "random", "similarity")


# ---------------------------------------------------------------------------
# Loading and aggregation
# ---------------------------------------------------------------------------

def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError):
        print(f"skipping unreadable {path}")
        return None


def load_pruning_runs(folder):
    """Successful pruning runs, plus the files of failed ones."""
    runs, failed = [], []
    for path in sorted(Path(folder).glob("*.json")):
        data = _read_json(path)
        if not data or data.get("experiment") != "pruning":
            continue
        if data.get("status") == "ok":
            runs.append(data)
        else:
            failed.append(path.name)
    return runs, failed


def load_verifications(folder):
    """Format comparisons on (possibly pruned) graphs, keyed by (method, seed, requested fraction)."""
    out = {}
    for path in sorted(Path(folder).glob("*.json")):
        data = _read_json(path)
        if data and "formats" in data and "pruning" in data:
            p = data["pruning"]
            out[(p["method"], p["seed"], p["requested_removal_fraction"])] = data
    return dict(sorted(out.items(), key=lambda kv: _condition_sort_key(kv[0])))


def _condition_sort_key(key):
    method, seed, fraction = key
    return METHOD_ORDER.index(method), fraction, seed or 0


def condition_name(method, seed, fraction):
    if method == "none":
        return "original graph"
    name = METHOD_STYLE[method]["label"]
    return f"{name}{'' if seed is None else f' s{seed}'} {100 * fraction:g}%"


def aggregate_conditions(runs):
    """{(method, seed, requested fraction): one summary dict per condition}."""
    groups = defaultdict(list)
    for run in runs:
        p = run["pruning"]
        groups[(p["method"], p["seed"], p["requested_removal_fraction"])].append(run)

    conditions = {}
    for key, reps in sorted(groups.items(), key=lambda kv: _condition_sort_key(kv[0])):
        reps = sorted(reps, key=lambda r: r["repetition"])
        first, effect = reps[0], reps[0]["pruning_effect"]
        timing = lambda name: statistics.median(r["timing_ms"][name] for r in reps)  # noqa: E731
        conditions[key] = {
            "method": key[0], "seed": key[1], "fraction": key[2],
            "actual_fraction": first["pruning"]["actual_removal_fraction"],
            "original_edges": first["pruning"]["original_unique_edges"],
            "retained_edges": first["pruning"]["retained_unique_edges"],
            "removed_edges": first["pruning"]["removed_unique_edges"],
            "graph": first["graph"]["pruned"],
            "val_acc": first["accuracy"]["val"], "test_acc": first["accuracy"]["test"],
            "original_test_acc": effect["original_test_accuracy"],
            "val_loss_pp": effect["val_accuracy_loss_pp"], "test_loss_pp": effect["test_accuracy_loss_pp"],
            "disagree_all": effect["prediction_disagreement_all"],
            "disagree_test": effect["prediction_disagreement_test"],
            "adjacency_bytes": first["adjacency_bytes"],
            "latency": latency_stats([{**r, "prediction": r["format_check"]} for r in reps]),
            "rep_medians": [r["latency_ms"]["median"] for r in reps],
            "order_ms": timing("order_ms"),
            "prep_ms": timing("selection_ms") + timing("normalization_ms") + timing("conversion_ms"),
            "transfer_ms": timing("transfer_ms"),
            "logits_consistent": len({r["logits_sha256"] for r in reps}) == 1,
            "original_logits_shas": {r["original_logits_sha256"] for r in reps},
            "logits_sha": first["logits_sha256"],
            "format_equivalent": all(r["format_check"]["equivalent"] for r in reps),
            "logits_file": next((r["logits_file"] for r in reps if r.get("logits_file")), None),
            "device": first["device"], "device_name": first["device_name"],
            "debug_only": first["latency_is_debug_only"],
            "checkpoint": first["checkpoint"],
        }
    return conditions


def method_series(conditions, value, spread=None, center=statistics.fmean):
    """{method: [(x = % removed, y, low, high)]} from the original graph up to 75%.

    Random pruning: y = center (mean by default) over pruning seeds, low/high =
    min/max over seeds. `spread(condition) -> (low, high)` overrides the range
    for a single condition.
    """
    baseline = next(c for (m, _, _), c in conditions.items() if m == "none")
    series = {}
    for method in ("random", "similarity"):
        by_fraction = defaultdict(list)
        for (m, _, fraction), c in conditions.items():
            if m == method:
                by_fraction[fraction].append(c)
        if not by_fraction:
            continue
        points = [(0.0, *_point([baseline], value, spread, center))]
        for fraction in sorted(by_fraction):
            group = by_fraction[fraction]
            points.append((100 * group[0]["actual_fraction"], *_point(group, value, spread, center)))
        series[method] = points
    return series


def _point(group, value, spread, center):
    values = [value(c) for c in group]
    if spread is not None:
        ranges = [spread(c) for c in group]
        return center(values), min(r[0] for r in ranges), max(r[1] for r in ranges)
    return center(values), min(values), max(values)


# ---------------------------------------------------------------------------
# Offline float64 analysis from saved logits
# ---------------------------------------------------------------------------

def load_labels(folder, dataset):
    path = Path(folder) / "logits" / f"{dataset}_labels.pt"
    return torch.load(path, weights_only=True) if path.exists() else None


def pruning_margin_rows(folder, dataset, conditions):
    """Pruning effects recomputed from saved logits; reference = the original-graph run's logits."""
    labels = load_labels(folder, dataset)
    baseline = next(c for (m, _, _), c in conditions.items() if m == "none")
    if labels is None or not baseline["logits_file"]:
        return None
    reference = torch.load(Path(folder) / baseline["logits_file"], weights_only=True)["logits"]
    y, test_mask = labels["y"], labels["test_mask"]
    rows = []
    for key, c in conditions.items():
        if key[0] == "none" or not c["logits_file"]:
            continue
        logits = torch.load(Path(folder) / c["logits_file"], weights_only=True)["logits"]
        margin = margin_analysis(reference, logits, {"all": None, "test": test_mask})
        change = logit_change(reference, logits)
        loss_pp = 100 * (exact_accuracy(reference, y, test_mask) - exact_accuracy(logits, y, test_mask))
        rows.append({"name": condition_name(*key), **change, "loss_pp": loss_pp, "margin": margin,
                     "bound_holds": abs(loss_pp) <= 100 * margin["test"]["disagreement_rate"] + 1e-9})
    return rows


def format_margin_rows(folder, verifications):
    """Format errors recomputed from the saved logits of each verification (reference: COO, same device)."""
    rows = []
    for key, data in verifications.items():
        if not data.get("logits_file"):
            continue
        saved = torch.load(Path(folder) / data["logits_file"], weights_only=True)
        y, test_mask, logits = saved["y"], saved["test_mask"], saved["logits"]
        for storage_format in _ordered_formats(logits):
            if storage_format == "coo":
                continue
            margin = margin_analysis(logits["coo"], logits[storage_format], {"all": None, "test": test_mask})
            diff_pp = 100 * (exact_accuracy(logits[storage_format], y, test_mask) - exact_accuracy(logits["coo"], y, test_mask))
            rows.append({"graph": condition_name(*key), "format": storage_format,
                         **logit_change(logits["coo"], logits[storage_format]),
                         "allclose": data["formats"][storage_format]["allclose"],
                         "diff_pp": diff_pp, "margin": margin})
    return rows


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def _range_ms(values):
    return f"{statistics.median(values):.3f} [{min(values):.3f}–{max(values):.3f}]"


def unpruned_format_table(comparison, by_format):
    headers = ["Format", "Test acc", "Acc diff (pp)", "Agree (all)", "Agree (test)", "Mean |Δlogit|",
               "ε_logit (max)", "allclose", "Adjacency", "Peak alloc", "Peak reserved", "Latency ms [rep range]"]
    rows = []
    for storage_format in _ordered_formats(comparison["formats"]):
        m = comparison["formats"][storage_format]
        reps = by_format.get(storage_format)
        s = latency_stats(reps) if reps else None
        rows.append([storage_format, _pct(m["test_accuracy"]), f"{m['accuracy_difference_pp']:+.2f}",
                     _pct(m["prediction_agreement_all"], 2), _pct(m["prediction_agreement_test"], 2),
                     f"{m['mean_abs_logit_diff']:.1e}", f"{m['max_abs_logit_diff']:.1e}", _yes_no(m["allclose"]),
                     human_bytes(m["adjacency_bytes"]),
                     human_bytes(s["peak_allocated"]) if s else "n/a",
                     human_bytes(s["peak_reserved"]) if s else "n/a",
                     f"{s['median']:.3f} [{s['rep_median_min']:.3f}–{s['rep_median_max']:.3f}]" if s else "n/a"])
    return headers, rows


def pruning_table(conditions):
    headers = ["Method", "Seed", "Removed", "Test acc", "Loss test (pp)", "Loss val (pp)", "Disagree (all)",
               "Disagree (test)", "Adjacency", "Peak alloc", "Latency ms [rep range]"]
    rows = []
    for (method, seed, _), c in conditions.items():
        rows.append([METHOD_STYLE.get(method, {"label": "original graph"})["label"], seed or "-",
                     _pct(c["actual_fraction"]), _pct(c["test_acc"]), f"{c['test_loss_pp']:+.1f}",
                     f"{c['val_loss_pp']:+.1f}", _pct(c["disagree_all"]), _pct(c["disagree_test"]),
                     human_bytes(c["adjacency_bytes"]), human_bytes(c["latency"]["peak_allocated"]),
                     _range_ms(c["rep_medians"])])
    return headers, rows


def random_seed_table(conditions):
    headers = ["Removed", "Seeds", "Test acc mean [min–max]", "Loss test pp mean [min–max]",
               "Disagree test mean [min–max]"]
    by_fraction = defaultdict(list)
    for (method, _, fraction), c in conditions.items():
        if method == "random":
            by_fraction[fraction].append(c)

    def cell(values, fmt):
        return f"{fmt(statistics.fmean(values))} [{fmt(min(values))}–{fmt(max(values))}]"

    rows = []
    for fraction, group in sorted(by_fraction.items()):
        rows.append([_pct(group[0]["actual_fraction"]), len(group),
                     cell([c["test_acc"] for c in group], _pct),
                     cell([c["test_loss_pp"] for c in group], lambda v: f"{v:+.1f}"),
                     cell([c["disagree_test"] for c in group], _pct)])
    return headers, rows


def graph_table(conditions):
    headers = ["Condition", "Removed / original edges", "Retained", "nnz (normalized)", "Density",
               "Isolated before self-loops", "Order ms", "Select+normalize+convert ms", "Transfer ms",
               "Rep medians (ms)"]
    rows = []
    for key, c in conditions.items():
        g = c["graph"]
        rows.append([condition_name(*key), f"{c['removed_edges']:,} / {c['original_edges']:,}",
                     f"{c['retained_edges']:,}", f"{g['normalized_nnz']:,}",
                     f"{100 * g['density_excluding_self_loops']:.4f}%", f"{g['isolated_nodes_before_self_loops']:,}",
                     f"{c['order_ms']:.1f}", f"{c['prep_ms']:.1f}", f"{c['transfer_ms']:.1f}",
                     ", ".join(f"{v:.3f}" for v in c["rep_medians"])])
    return headers, rows


def _coverage(margin):
    return f"{_pct(margin['all']['margin_condition_fraction'])} / {_pct(margin['test']['margin_condition_fraction'])}"


def format_epsilon_table(rows):
    headers = ["Graph", "Format vs COO", "Mean |Δlogit|", "ε_logit", "allclose", "Acc diff test (pp)",
               "|Acc diff| (pp)", "Disagree all / test", "margin > 2ε all / test", "Disagreements with margin > 2ε"]
    return headers, [[r["graph"], r["format"], f"{r['mean_abs_logit_diff']:.1e}", f"{r['epsilon_logit']:.1e}",
                      _yes_no(r["allclose"]), f"{r['diff_pp']:+.2f}", f"{abs(r['diff_pp']):.2f}",
                      f"{_pct(r['margin']['all']['disagreement_rate'], 2)} / "
                      f"{_pct(r['margin']['test']['disagreement_rate'], 2)}",
                      _coverage(r["margin"]),
                      r["margin"]["all"]["disagreements_satisfying_condition"]] for r in rows]


def pruning_epsilon_table(rows):
    headers = ["Pruned graph vs original (CSR)", "Mean |Δlogit|", "ε_logit", "Loss test (pp)", "Disagree all / test",
               "margin > 2ε all / test", "Disagreements with margin > 2ε", "|loss| ≤ disagreement"]
    return headers, [[r["name"], f"{r['mean_abs_logit_diff']:.3f}", f"{r['epsilon_logit']:.3f}",
                      f"{r['loss_pp']:+.1f}",
                      f"{_pct(r['margin']['all']['disagreement_rate'])} / {_pct(r['margin']['test']['disagreement_rate'])}",
                      _coverage(r["margin"]), r["margin"]["all"]["disagreements_satisfying_condition"],
                      _yes_no(r["bound_holds"])] for r in rows]


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def method_chart(plt, path, series, title, subtitle, x_label, y_fmt, x_fmt, baseline=None, x_notes=None):
    """One line per pruning method with range whiskers and end labels.

    series: {method: [(x, y, low, high)]}. baseline: optional (label, y) drawn as
    a dashed reference line, or (label, x, y) drawn as a reference point.
    x_notes: optional [(x, text)] written along the bottom of the plot instead
    of end labels (for charts whose lines do not end at the right edge).
    """
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, NullLocator

    subtitle_lines = textwrap.wrap(subtitle, 100)
    width_in, height_in = 7.5, 4.8 + 0.17 * (len(subtitle_lines) - 1)
    left_in, right_in, top_in, bottom_in = 0.95, 1.55, 0.98 + 0.17 * len(subtitle_lines), 0.6
    fig = plt.figure(figsize=(width_in, height_in), dpi=100)
    ax = fig.add_axes([left_in / width_in, bottom_in / height_in,
                       1 - (left_in + right_in) / width_in, 1 - (top_in + bottom_in) / height_in])

    for method, points in series.items():
        style = METHOD_STYLE[method]
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        ax.plot(xs, ys, color=style["color"], linewidth=1.5, solid_joinstyle="round", solid_capstyle="round",
                zorder=3)
        ax.plot(xs, ys, linestyle="none", marker=style["marker"], markersize=6.5, markerfacecolor=style["color"],
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        for x, _, low, high in points:
            if high > low:
                ax.plot([x, x], [low, high], color=style["color"], linewidth=1, zorder=2)

    handles = [Line2D([], [], color=METHOD_STYLE[m]["color"], linewidth=1.5, marker=METHOD_STYLE[m]["marker"],
                      markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.5) for m in series]
    names = [METHOD_STYLE[m]["label"] for m in series]
    if baseline and len(baseline) == 2:
        ax.axhline(baseline[1], color=INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        handles.append(Line2D([], [], color=INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3))))
        names.append(baseline[0])
    elif baseline:
        ax.plot([baseline[1]], [baseline[2]], linestyle="none", marker="D", markersize=7, markerfacecolor=INK,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
        handles.append(Line2D([], [], linestyle="none", marker="D", markersize=7, markerfacecolor=INK,
                              markeredgecolor=SURFACE))
        names.append(baseline[0])

    ax.margins(x=0.04, y=0.12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: x_fmt(v)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: y_fmt(v)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(which="both", length=0, labelsize=9, labelcolor=MUTED, pad=6)
    ax.yaxis.grid(True, color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.75)
    ax.set_xlabel(x_label, color=MUTED, fontsize=9, labelpad=6)

    if x_notes:
        for x, note in x_notes:
            ax.annotate(note, xy=(x, 0), xycoords=("data", "axes fraction"), xytext=(0, 6),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=MUTED)
    else:
        # Direct labels at each line's last point, in ink; the marker beside each carries the color.
        fig.canvas.draw()
        to_pixels, to_data = ax.transData.transform, ax.transData.inverted().transform
        ends = [to_pixels(points[-1][:2]) for points in series.values()]
        labels_y = _spread([p[1] for p in ends], min_gap=13)
        for (method, points), end, y_px in zip(series.items(), ends, labels_y):
            ax.annotate(f"{METHOD_STYLE[method]['label']}  {y_fmt(points[-1][1])}",
                        xy=to_data((end[0], y_px)), xytext=(10, 0), textcoords="offset points", va="center",
                        ha="left", fontsize=9, color=INK_SECONDARY, annotation_clip=False)

    left = left_in / width_in
    fig.text(left, 1 - 0.32 / height_in, title, fontsize=12.5, fontweight="bold", color=INK, va="baseline")
    for k, line in enumerate(subtitle_lines):
        fig.text(left, 1 - (0.55 + 0.17 * k) / height_in, line, fontsize=9, color=INK_SECONDARY, va="baseline")
    legend_top = 0.68 + 0.17 * (len(subtitle_lines) - 1)
    legend = fig.legend(handles, names, loc="upper left", bbox_to_anchor=(left - 0.01, 1 - legend_top / height_in),
                        ncol=len(names), frameon=False, fontsize=9.5, handlelength=2.2, columnspacing=1.6)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _removed_fmt(v):
    return f"{v:g}%"


def _mb(v):
    # Pruning changes peak memory by small amounts, so keep two decimals.
    return f"{v / 1e6:.2f} MB"


def pruning_charts(plt, out_dir, dataset, conditions):
    baseline = next(c for (m, _, _), c in conditions.items() if m == "none")
    device = baseline["device"]
    seeds = len({k[1] for k in conditions if k[0] == "random"})
    removed = "Edges removed (% of unique undirected edges)"
    seed_note = f"random: mean of {seeds} pruning seeds, whisker = seed range"
    timing = "FP32, TF32 off" if not baseline["debug_only"] else f"DEBUG ONLY: {device} timing, not a GPU result"
    has_peak = baseline["latency"]["peak_allocated"] is not None
    written = {}

    def chart(name, *args, **kwargs):
        path = Path(out_dir) / f"{dataset}_{name}.png"
        method_chart(plt, path, *args, **kwargs)
        written[name] = path

    acc = lambda v: f"{v:.1f}%"  # noqa: E731
    chart("accuracy_vs_removed", method_series(conditions, lambda c: 100 * c["test_acc"]),
          f"Test accuracy vs edges removed · {dataset}",
          f"Frozen checkpoint, no retraining · CSR · {seed_note}", removed, acc, _removed_fmt,
          baseline=(f"original graph {100 * baseline['test_acc']:.1f}%", 100 * baseline["test_acc"]))
    if has_peak:
        chart("peak_memory_vs_removed",
              method_series(conditions, lambda c: c["latency"]["peak_allocated"], center=statistics.median),
              f"Peak GPU memory vs edges removed · {dataset}",
              "CSR · torch.cuda.max_memory_allocated during the measured passes (adjacency + features + "
              "weights + temporaries) · random: median of seeds, whisker = seed range · both methods keep the "
              "same number of edges at each rate, so their lines can overlap",
              removed, _mb, _removed_fmt)
    latency_range = lambda c: (min(c["rep_medians"]), max(c["rep_medians"]))  # noqa: E731
    chart("latency_vs_removed",
          method_series(conditions, lambda c: c["latency"]["median"], spread=latency_range,
                        center=statistics.median),
          f"Inference latency vs edges removed · {dataset} · {device}",
          f"CSR · median of the {len(baseline['rep_medians'])} repetition medians (random: median over seeds too) "
          f"· whisker = range of repetition medians · {timing}",
          removed, lambda v: f"{v:.3g} ms", _removed_fmt)
    if has_peak:
        series, notes = {}, {}
        for method, points in method_series(conditions, lambda c: 100 * c["test_acc"]).items():
            peaks = method_series(conditions, lambda c: c["latency"]["peak_allocated"], center=statistics.median)[method]
            series[method] = [(peak[1], p[1], p[2], p[3]) for p, peak in zip(points, peaks)]
            notes.update({f"{p[0]:.0f}%": peak[1] for p, peak in zip(points[1:], peaks[1:])})
        chart("accuracy_vs_peak_memory", series, f"Test accuracy vs peak GPU memory · {dataset}",
              f"Each point is one removal rate; the % removed is written along the bottom · {seed_note}",
              "Peak GPU memory allocated during inference (MB = 10^6 bytes)", acc, _mb,
              baseline=("original graph", baseline["latency"]["peak_allocated"], 100 * baseline["test_acc"]),
              x_notes=[(x, label) for label, x in notes.items()])
    return written


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def find_datasets(pruning_dir):
    """[(dataset, device tag)] for every full-stage folder (<dataset>_<device>)."""
    found = []
    for folder in sorted(Path(pruning_dir).iterdir()):
        if folder.is_dir() and not folder.name.startswith(("pilot_", "verify_", "summary")) and "_" in folder.name:
            dataset, device_tag = folder.name.rsplit("_", 1)
            found.append((dataset, device_tag))
    return found


def summarize_pruning(pruning_dir, formats_dir, out_dir):
    """Write summary.md and charts to out_dir. Returns (text report, list of written files)."""
    pruning_dir, out_dir = Path(pruning_dir), Path(out_dir)
    datasets = find_datasets(pruning_dir)
    if not datasets:
        raise FileNotFoundError(f"no <dataset>_<device> pruning folders in {pruning_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    comparisons, benchmarks = load_results(formats_dir) if Path(formats_dir).exists() else ({}, {})
    plt = _pyplot()
    written, text = [], []
    md = ["# Pruning and error-analysis results", "",
          "Generated by `scripts/summarize_pruning_results.py`. One frozen checkpoint per dataset, no retraining. "
          "Pruning runs use CSR. Format error compares storage formats on the same graph; pruning effects compare "
          "a pruned graph with the original graph. Accuracy differences are in percentage points (pp); logit "
          "differences are raw logit units. Ranges are observed min–max spreads, not confidence intervals. "
          "Method and settings: `docs/pruning_and_error_analysis.md`.", ""]
    if plt is None:
        md += ["_matplotlib is not installed, so no charts were drawn._", ""]

    def section(title, headers, rows, note=None, image=None):
        text.extend([title, text_table(headers, rows)])
        md.extend([f"### {title}", ""])
        if image:
            md.extend([f"![{title}]({image.name})", ""])
        md.extend([markdown_table(headers, rows), ""])
        if note:
            text.append(note)
            md.extend([note, ""])
        text.append("")

    for dataset, device_tag in datasets:
        runs, failed = load_pruning_runs(pruning_dir / f"{dataset}_{device_tag}")
        if not runs:
            continue
        conditions = aggregate_conditions(runs)
        if not any(m == "none" for m, _, _ in conditions):
            print(f"{dataset}_{device_tag}: no original-graph run yet, skipping")
            continue
        baseline = next(c for (m, _, _), c in conditions.items() if m == "none")
        device = baseline["device"]
        text += ["=" * 72, f"DATASET: {dataset} ({device})", "=" * 72, ""]
        md += [f"## {dataset}", ""]
        ck = baseline["checkpoint"]
        intro = (f"Device {device} ({baseline['device_name']}). Checkpoint `{ck['path']}` (sha256 {ck['sha256']}, "
                 f"model seed {ck['model_seed']}, best epoch {ck['best_epoch']}, val acc "
                 f"{100 * ck['best_validation_accuracy']:.1f}%). {len(runs)} pruning runs"
                 + (f", **{len(failed)} failed: {', '.join(failed)}**" if failed else ", none failed") + ".")
        if baseline["debug_only"]:
            intro += " **Latency is DEBUG ONLY (not a GPU result).**"
        text += [intro, ""]
        md += [intro, ""]

        comparison = comparisons.get((dataset, device))
        if comparison:
            section("1. Unpruned format comparison (dense / COO / CSR)",
                    *unpruned_format_table(comparison, benchmarks.get((dataset, device), {})),
                    note=f"Reference: COO on {device}. allclose(atol=1e-5, rtol=1e-4). Latency and peak memory from "
                         "scripts/benchmark_gcn.py (peak counters reset before warm-up, as in the Cora baseline).")
        else:
            note = f"_No unpruned format comparison for {dataset} on {device} in `{formats_dir}`._"
            text += [note, ""]
            md += [note, ""]

        charts = pruning_charts(plt, out_dir, dataset, conditions) if plt else {}
        written.extend(charts.values())
        section("2. Pruning results (CSR)", *pruning_table(conditions),
                note="Loss = original − pruned accuracy, in pp; negative means pruning improved accuracy. "
                     "Disagreement is against the original graph's predictions with the same checkpoint.")
        section("Random pruning across pruning seeds", *random_seed_table(conditions))
        for name, title in (("accuracy_vs_removed", "3. Accuracy vs edges removed"),
                            ("peak_memory_vs_removed", "4. Peak memory vs edges removed"),
                            ("latency_vs_removed", "5. Inference latency vs edges removed"),
                            ("accuracy_vs_peak_memory", "6. Accuracy vs peak memory")):
            if name in charts:
                md += [f"### {title}", "", f"![{title}]({charts[name].name})", ""]
        section("Pruned graphs, preparation time and latency repetitions", *graph_table(conditions),
                note="Order ms = scoring/ordering the edges (0 for the original graph). None of these are "
                     "included in inference latency.")

        verifications = load_verifications(pruning_dir / f"verify_{dataset}_{device_tag}")
        format_rows = format_margin_rows(pruning_dir / f"verify_{dataset}_{device_tag}", verifications)
        if format_rows:
            all_ok = all(v["all_equivalent"] for v in verifications.values())
            section("7a. Format error: same graph, different format (reference COO on the same device)",
                    *format_epsilon_table(format_rows),
                    note=f"All formats equivalent on every verified graph: {_yes_no(all_ok)}.")
        prune_rows = pruning_margin_rows(pruning_dir / f"{dataset}_{device_tag}", dataset, conditions)
        if prune_rows:
            section("7b. Pruning effects: pruned vs original graph (CSR, same checkpoint)",
                    *pruning_epsilon_table(prune_rows),
                    note="margin > 2ε is sufficient for an unchanged prediction, so the 'Disagreements with "
                         "margin > 2ε' column must be 0. Recomputed in float64 from the saved logits.")

        checks = [
            ("every pruning run matches COO on CPU for its own graph",
             all(c["format_equivalent"] for c in conditions.values())),
            ("logits identical across repetitions", all(c["logits_consistent"] for c in conditions.values())),
            ("original-graph logits identical in every run",
             len(set().union(*(c["original_logits_shas"] for c in conditions.values()))) == 1
             and baseline["logits_sha"] in baseline["original_logits_shas"]),
            ("same checkpoint in every run", len({c["checkpoint"]["sha256"] for c in conditions.values()}) == 1),
        ]
        lines = [f"- {name}: {_yes_no(ok)}" for name, ok in checks]
        text += ["Checks:", *lines, ""]
        md += ["### Checks", "", *lines, ""]

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(md))
    written.insert(0, summary_path)
    return "\n".join(text), written


def default_dirs():
    root = PROJECT_ROOT / "results"
    return root / "pruning", root / "gpu", root / "pruning" / "summary"

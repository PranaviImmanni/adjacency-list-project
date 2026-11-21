"""Tables, CSVs and charts for the follow-up MLP experiments.

Reads results/mlp_followup/<run>/{evaluate,benchmark} and, as baselines, the
original sweep's results/mlp/<baseline run>/{evaluate,benchmark}: the original
dense model and global-magnitude pruning without fine-tuning. Baseline timing
is used only when both runs are on the same device.

Writes summary.md, combined.csv (one row per model and format, every metric at
every batch size), one CSV per section, and:
  accuracy_vs_memory.png        test accuracy vs stored weight bytes (top) and peak GPU
                                memory at batch 256 (bottom), one panel per experiment
  latency_by_model.png          latency of every model at batch 1 / 32 / 256
  statistical_vs_magnitude.png  experiment 1 against the original magnitude sweep

Aggregation as in the original report: a unit is one model seed; values are
the mean over units, ranges min - max. Latency per unit is the median of the
per-repetition medians; memory per unit is the maximum over repetitions.
"""

import json
import statistics
import textwrap
from collections import defaultdict
from pathlib import Path

from src.gcn.reporting import BASELINE, GRID, INK, INK_SECONDARY, MUTED, SURFACE, _pyplot, markdown_table

from .reporting import _compact_bytes, _range, _stats, _write_csv, _x
from .variants import s_label

# Categorical slots 1-3 (validated all-pairs for scatter); gray is reserved for the baselines.
SLOTS = [{"color": "#2a78d6", "marker": "o"}, {"color": "#eb6834", "marker": "s"},
         {"color": "#1baf7a", "marker": "^"}]
BATCH_STYLE = {1: SLOTS[0], 32: SLOTS[1], 256: SLOTS[2]}
EXPERIMENT_TITLES = {
    "baseline": "Baselines (original sweep)", "statistical": "1. Statistical pruning",
    "finetune": "2. Pruning + fine-tuning", "protect_fc3": "3. Protect the output layer",
    "lora": "4. Pruning + LoRA", "int8": "5. INT8 without pruning",
}
FORMAT_TEXT = {"dense": "FP32 dense", "csr": "CSR", "int8_weight_only": "INT8 weight-only",
               "int8_w8a8": "INT8 W8A8"}


# ---------------------------------------------------------------------------
# Loading and aggregation
# ---------------------------------------------------------------------------

def _load(directory):
    out = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"skipping unreadable {path}")
            continue
        if record.get("status", "ok") == "ok":
            out.append(record)
    return out


def _row_id(tag):
    return tag.split("_", 1)[1]


def _k_text(k):
    return f"k = {k:g}"


def row_label(row_id, storage_format):
    parts = row_id.split("_")
    s = lambda p: f"{p[1:]}%"  # noqa: E731
    if row_id == "original":
        return {"dense": "Original, FP32 dense", "int8_weight_only": "Original, INT8 weight-only",
                "int8_w8a8": "Original, INT8 W8A8"}[storage_format]
    if parts[0] == "magnitude":
        return f"Magnitude {s(parts[1])}, pruning only"
    if parts[0] == "statistical":
        return f"Statistical, {_k_text(float(parts[1][1:].replace('p', '.')))}"
    if parts[0] == "magmatch":
        return f"Magnitude, count of {_k_text(float(parts[1][1:].replace('p', '.')))}"
    if parts[0] == "finetune":
        return f"Magnitude {s(parts[1])} + fine-tune"
    if parts[0] == "fc3dense":
        return f"fc3 dense, {s(parts[1])} bytes" + (" + fine-tune" if row_id.endswith("finetune") else ", pruning only")
    if parts[0] == "lora":
        return f"Magnitude {s(parts[1])} + LoRA"
    return row_id


def format_text(row_id, storage_format):
    if row_id.startswith("fc3dense"):
        return "CSR + dense fc3"
    if row_id.startswith("lora"):
        return "CSR + adapters"
    return FORMAT_TEXT[storage_format]


def collect(followup_dir, baseline_dir, include_baseline_timing):
    """{(row id, format): {'experiment', 'kind', 'params', 'units': {seed: {...}}}}."""
    rows = {}

    def unit(key, experiment, seed, kind=None, params=None):
        row = rows.setdefault(key, {"experiment": experiment, "kind": kind, "params": params or {}, "units": {}})
        if kind is not None:
            row["kind"], row["params"] = kind, params or {}
        return row["units"].setdefault(seed, {"bench": defaultdict(list)})

    for r in _load(Path(followup_dir) / "evaluate"):
        variant = r["variant"]
        for storage_format, f in r["formats"].items():
            u = unit((_row_id(variant["tag"]), storage_format), variant["experiment"], variant["model_seed"],
                     variant["kind"], variant["params"])
            check = f["check"]
            u.update(test=f["accuracy"]["test"], val=f["accuracy"]["val"], sparsity=variant["model_sparsity"],
                     stored=f["storage"]["stored_bytes"], weight=f["storage"]["weight_bytes"],
                     adapter=f["storage"]["adapter_bytes"], delta=-f["effect_vs_original"]["test_accuracy_loss_pp"],
                     epsilon=check["epsilon_logit"], equivalent=check["equivalent"],
                     bitwise=check["bitwise_equal"], agreement=check["prediction_agreement_all"],
                     variant=variant, int_mm=f.get("int_mm_support"))
    for r in _load(Path(followup_dir) / "benchmark"):
        unit((_row_id(r["tag"]), r["format"]), r["followup_experiment"], r["model_seed"])["bench"][
            r["batch_size"]].append(r)

    for r in _load(Path(baseline_dir) / "evaluate"):
        if r["method"] == "none":
            key, kind, storage = ("original", "dense"), "original", r["storage"]["dense"]
            check, delta = None, 0.0
        elif r["method"] == "magnitude":
            key, kind = (f"magnitude_{s_label(r['requested_sparsity'])}", "csr"), "magnitude"
            storage, check = r["storage"]["csr"], r["format_checks"]["csr"]
            delta = -r["pruning_effect"]["test_accuracy_loss_pp"]
        else:
            continue
        u = unit(key, "baseline", r["model_seed"], kind, {"sparsity": r["requested_sparsity"]})
        u.update(test=r["accuracy"]["test"], val=r["accuracy"]["val"], sparsity=r["actual_sparsity"],
                 stored=storage["weight_bytes"], weight=storage["weight_bytes"], adapter=0, delta=delta)
        if check:
            u.update(epsilon=check["epsilon_logit"], equivalent=check["equivalent"],
                     bitwise=check.get("bitwise_equal", False), agreement=check["prediction_agreement_all"])
    if include_baseline_timing:
        for r in _load(Path(baseline_dir) / "benchmark"):
            if r["method"] == "none" and r["format"] == "dense":
                key = ("original", "dense")
            elif r["method"] == "magnitude" and r["format"] == "csr":
                key = (f"magnitude_{s_label(r['requested_sparsity'])}", "csr")
            else:
                continue
            if key in rows:
                rows[key]["units"].setdefault(r["model_seed"], {"bench": defaultdict(list)})["bench"][
                    r["batch_size"]].append(r)
    return rows


def aggregate(row, batches):
    units = [u for u in row["units"].values()]
    agg = {key: _stats([u.get(key) for u in units]) for key in
           ("test", "val", "sparsity", "stored", "weight", "adapter", "delta")}
    checked = [u for u in units if "epsilon" in u]
    agg["epsilon"] = max((u["epsilon"] for u in checked), default=None)
    agg["equivalent"] = all(u["equivalent"] for u in checked) if checked else None
    agg["bitwise"] = all(u["bitwise"] for u in checked) if checked else None
    agg["agreement"] = min((u["agreement"] for u in checked), default=None)
    agg["seeds"] = sorted(row["units"])
    agg["batches"] = {}
    for batch in batches:
        reps_per_unit = [u["bench"].get(batch, []) for u in units if u["bench"].get(batch)]
        if not reps_per_unit:
            continue

        def per_unit(field, reduce):
            values = [reduce([r[field] for r in reps if r.get(field) is not None]) for reps in reps_per_unit
                      if any(r.get(field) is not None for r in reps)]
            return _stats(values)

        agg["batches"][batch] = {
            "latency": _stats([statistics.median(r["latency_ms"]["median"] for r in reps) for reps in reps_per_unit]),
            "resident": per_unit("cuda_allocated_before_inference_bytes", max),
            "peak": per_unit("cuda_peak_allocated_bytes", max),
            "reps": min(len(reps) for reps in reps_per_unit), "units": len(reps_per_unit),
            "debug": any(r["latency_is_debug_only"] for reps in reps_per_unit for r in reps),
            "checks_equivalent": all(r["format_check"]["equivalent"] for reps in reps_per_unit for r in reps
                                     if r.get("format_check")),
            "check_epsilon": max((r["format_check"]["epsilon_logit"] for reps in reps_per_unit for r in reps
                                  if r.get("format_check")), default=None),
        }
    return agg


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(stats, digits=2):
    return _range(stats, lambda v: f"{100 * v:.{digits}f}%")


def _pp(stats):
    return _range(stats, lambda v: f"{v:+.2f}")


def _ms(stats):
    return _range(stats, lambda v: f"{v:.4f}")


def _bytes(stats):
    return _range(stats, lambda v: f"{v:,.0f}")


def _mem(stats):
    return _range(stats, _compact_bytes)


def _batch(agg, batch, field, fmt):
    point = agg["batches"].get(batch)
    return fmt(point[field]) if point and point[field] else "n/a"


def _mean(stats):
    return None if stats is None else stats["mean"]


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _style(ax, log_x=False):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator

    ax.tick_params(which="both", length=0, labelsize=8.5, labelcolor=MUTED, pad=5)
    ax.yaxis.grid(True, which="major", color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.75)
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 3)))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _compact_bytes(v)))


def _header(fig, width_in, height_in, left_in, title, subtitle):
    lines = textwrap.wrap(subtitle, int((width_in - left_in - 0.3) * 14.5))
    left = left_in / width_in
    fig.text(left, 1 - 0.32 / height_in, title, fontsize=12.5, fontweight="bold", color=INK, va="baseline")
    for i, line in enumerate(lines):
        fig.text(left, 1 - (0.55 + 0.17 * i) / height_in, line, fontsize=9, color=INK_SECONDARY, va="baseline")
    return len(lines)


def _series_plot(ax, xs, ys, lows, highs, style, line=True):
    order = sorted(range(len(xs)), key=xs.__getitem__)
    xs, ys = [xs[i] for i in order], [ys[i] for i in order]
    lows, highs = [lows[i] for i in order], [highs[i] for i in order]
    if line and len(xs) > 1:
        ax.plot(xs, ys, color=style["color"], linewidth=1.5, solid_joinstyle="round", zorder=3)
    for x, low, high in zip(xs, lows, highs):
        if high > low:
            ax.plot([x, x], [low, high], color=style["color"], linewidth=1, zorder=2)
    ax.plot(xs, ys, linestyle="none", marker=style["marker"], markersize=6.5, markerfacecolor=style["color"],
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)


def accuracy_memory_figure(plt, path, panels, curve, original, has_peak, subtitle):
    """panels: [(title, [(label, style, [(stored, peak, mean, low, high, point label)])])]."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    n, n_rows = len(panels), 2 if has_peak else 1
    max_entries = max(len(series) for _, series in panels)
    panel_w, panel_h, gap_w, gap_h = 2.75, 2.55, 0.45, 0.85
    left_in, right_in, bottom_in = 0.85, 0.3, 0.6
    width_in = left_in + n * panel_w + (n - 1) * gap_w + right_in
    n_sub = len(textwrap.wrap(subtitle, int((width_in - left_in - 0.3) * 14.5)))
    title_pad = 10 + 12 * max_entries  # points: room for the panel legend between title and plot
    top_in = 1.0 + 0.17 * n_sub + (title_pad + 14) / 72
    height_in = top_in + n_rows * panel_h + (n_rows - 1) * gap_h + bottom_in + 0.55
    fig = plt.figure(figsize=(width_in, height_in), dpi=100)
    _header(fig, width_in, height_in, left_in, "Test accuracy vs memory, by experiment", subtitle)

    metrics = [("stored", "stored weight bytes (log)")] + ([("peak", "peak GPU memory, batch 256 (log)")]
                                                            if has_peak else [])
    for r, (metric, xlabel) in enumerate(metrics):
        all_x = [p[0 if metric == "stored" else 1] for _, series in panels for _, _, pts in series for p in pts]
        all_x += [x for x, _ in curve.get(metric, [])] + ([original[metric]] if original.get(metric) else [])
        all_x = [x for x in all_x if x]
        x_lo, x_hi = min(all_x) / 1.6, max(all_x) * 1.6
        for c, (title, series) in enumerate(panels):
            left = left_in + c * (panel_w + gap_w)
            bottom = bottom_in + 0.55 + (n_rows - 1 - r) * (panel_h + gap_h)
            ax = fig.add_axes([left / width_in, bottom / height_in, panel_w / width_in, panel_h / height_in])
            _style(ax, log_x=True)
            ax.set_xlim(x_lo, x_hi)
            if x_hi / x_lo > 10 ** 2.5:  # wide range: decade ticks only, so labels do not collide
                from matplotlib.ticker import LogLocator
                ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1,)))
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
            if c:
                ax.set_yticklabels([])
            points = curve.get(metric, [])
            if points:
                ax.plot([x for x, _ in points], [y for _, y in points], color=MUTED, linewidth=1, zorder=1)
                ax.plot([x for x, _ in points], [y for _, y in points], linestyle="none", marker="o",
                        markersize=3.5, color=MUTED, zorder=1)
            if original.get(metric):
                ax.plot([original[metric]], [original["acc"]], linestyle="none", marker="X", markersize=8,
                        markerfacecolor=MUTED, markeredgecolor=SURFACE, markeredgewidth=1, zorder=2)
            for label, style, pts in series:
                idx = 0 if metric == "stored" else 1
                pts = [p for p in pts if p[idx]]
                if pts:
                    _series_plot(ax, [p[idx] for p in pts], [p[2] for p in pts], [p[3] for p in pts],
                                 [p[4] for p in pts], style)
                    if style is SLOTS[0]:
                        for p in pts:
                            if p[5]:
                                ax.annotate(p[5], (p[idx], p[2]), xytext=(6, 5), textcoords="offset points",
                                            fontsize=7.5, color=INK_SECONDARY)
            ax.set_xlabel(xlabel, color=MUTED, fontsize=8, labelpad=5)
            if r == 0:
                ax.set_title(title, loc="left", fontsize=10, color=INK, pad=title_pad)
                handles = [Line2D([], [], color=s["color"], marker=s["marker"], linewidth=1.5, markersize=6,
                                  markeredgecolor=SURFACE) for _, s, _ in series]
                legend = ax.legend(handles, [label for label, _, _ in series], loc="lower left",
                                   bbox_to_anchor=(-0.02, 1.0), frameon=False, fontsize=7.5, handlelength=1.8,
                                   borderaxespad=0.2, labelspacing=0.25)
                for text in legend.get_texts():
                    text.set_color(INK)
    handles = [Line2D([], [], color=MUTED, linewidth=1, marker="o", markersize=3.5),
               Line2D([], [], color=MUTED, linestyle="none", marker="X", markersize=8, markeredgecolor=SURFACE)]
    legend = fig.legend(handles, ["global magnitude, pruning only (original sweep, 50-99%)",
                                  "original model, FP32 dense"], loc="lower left",
                        bbox_to_anchor=(left_in / width_in, 0.12 / height_in), ncol=2, frameon=False, fontsize=8.5)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def latency_figure(plt, path, groups, subtitle):
    """groups: [(group title, [(row label, {batch: latency stats})])]."""
    from matplotlib.lines import Line2D

    labels, positions, headers = [], [], []
    y = 0
    for title, items in groups:
        headers.append((y, title))
        y += 1
        for label, by_batch in items:
            labels.append((y, label, by_batch))
            y += 1
        y += 0.4
    row_in, left_in, right_in = 0.24, 2.6, 0.4
    width_in = 8.6
    n_sub = len(textwrap.wrap(subtitle, int((width_in - left_in - 0.3) * 14.5)))
    top_in, bottom_in = 1.25 + 0.17 * n_sub, 0.6
    height_in = top_in + y * row_in + bottom_in
    fig = plt.figure(figsize=(width_in, height_in), dpi=100)
    _header(fig, width_in, height_in, left_in, "Inference latency by model and batch size", subtitle)
    ax = fig.add_axes([left_in / width_in, bottom_in / height_in, (width_in - left_in - right_in) / width_in,
                       (y * row_in) / height_in])
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.75)
    x_max = 0
    for pos, label, by_batch in labels:
        for batch, stats in by_batch.items():
            if stats is None:
                continue
            style = BATCH_STYLE[batch]
            offset = {1: -0.22, 32: 0, 256: 0.22}[batch]
            if stats["max"] > stats["min"]:
                ax.plot([stats["min"], stats["max"]], [pos + offset] * 2, color=style["color"], linewidth=1, zorder=2)
            ax.plot([stats["mean"]], [pos + offset], linestyle="none", marker=style["marker"], markersize=5.5,
                    markerfacecolor=style["color"], markeredgecolor=SURFACE, markeredgewidth=1, zorder=3)
            x_max = max(x_max, stats["max"])
    ax.set_ylim(y - 0.4, -0.8)
    ax.set_xlim(0, x_max * 1.08 if x_max else 1)
    ax.set_yticks([p for p, _, _ in labels])
    ax.set_yticklabels([label for _, label, _ in labels], fontsize=8.5, color=INK)
    for pos, title in headers:
        ax.text(-0.01, pos, title, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.5,
                fontweight="bold", color=INK)
    ax.set_xlabel("latency (ms)", color=MUTED, fontsize=8.5)
    handles = [Line2D([], [], linestyle="none", color=BATCH_STYLE[b]["color"], marker=BATCH_STYLE[b]["marker"],
                      markersize=6, markeredgecolor=SURFACE) for b in (1, 32, 256)]
    legend = fig.legend(handles, ["batch 1", "batch 32", "batch 256"], loc="upper left", ncol=3, frameon=False,
                        fontsize=9, bbox_to_anchor=(left_in / width_in - 0.01, 1 - (0.7 + 0.17 * n_sub) / height_in))
    for text in legend.get_texts():
        text.set_color(INK)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def statistical_figure(plt, path, curve, statistical, matched, subtitle):
    """Accuracy vs sparsity: the original magnitude sweep (gray), statistical (blue), matched magnitude (orange)."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    width_in, left_in, panel_w, panel_h = 7.2, 0.85, 5.0, 3.2
    n_sub = len(textwrap.wrap(subtitle, int((width_in - left_in - 0.3) * 14.5)))
    top_in, bottom_in = 1.55 + 0.17 * n_sub, 0.65
    height_in = top_in + panel_h + bottom_in
    fig = plt.figure(figsize=(width_in, height_in), dpi=100)
    _header(fig, width_in, height_in, left_in, "Statistical rule vs global magnitude at the same retained count",
            subtitle)
    ax = fig.add_axes([left_in / width_in, bottom_in / height_in, panel_w / width_in, panel_h / height_in])
    _style(ax)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.plot([_x(s) for s, _ in curve], [a for _, a in curve], color=MUTED, linewidth=1, marker="o", markersize=3.5,
            zorder=1)
    for points, style in ((statistical, SLOTS[0]), (matched, SLOTS[1])):
        _series_plot(ax, [_x(p[0]) for p in points], [p[1] for p in points], [p[2] for p in points],
                     [p[3] for p in points], style)
    for s, acc, _, _, label in statistical:
        ax.annotate(label, (_x(s), acc), xytext=(0, -15), textcoords="offset points", fontsize=8,
                    color=INK_SECONDARY, ha="center")
    ticks = sorted({s for s, _ in curve})
    ax.set_xticks([_x(s) for s in ticks])
    ax.set_xticklabels([f"{100 * s:.0f}%" for s in ticks], fontsize=8)
    ax.set_xlim(min(_x(s) for s in ticks + [p[0] for p in statistical]) - 0.1, _x(ticks[-1]) + 0.1)
    ax.set_xlabel("sparsity of the whole model (log scale of retained fraction)", color=MUTED, fontsize=8.5)
    handles = [Line2D([], [], color=SLOTS[0]["color"], marker="o", markeredgecolor=SURFACE),
               Line2D([], [], color=SLOTS[1]["color"], marker="s", markeredgecolor=SURFACE),
               Line2D([], [], color=MUTED, marker="o", markersize=3.5, linewidth=1)]
    legend = fig.legend(handles, ["statistical rule (|w - mean| > k std, per matrix)",
                                  "global magnitude, same retained count", "global magnitude, original sweep"],
                        loc="upper left", frameon=False, fontsize=8.5,
                        bbox_to_anchor=(left_in / width_in - 0.01, 1 - (0.7 + 0.17 * n_sub) / height_in))
    for text in legend.get_texts():
        text.set_color(INK)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def summarize_followup(followup_dir, baseline_dir, out_dir, config, include_baseline_timing):
    followup_dir, out_dir = Path(followup_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = list(config["batch_sizes"])
    rows = collect(followup_dir, baseline_dir, include_baseline_timing)
    if not any(row["experiment"] != "baseline" for row in rows.values()):
        raise FileNotFoundError(f"no follow-up results in {followup_dir}")
    aggs = {key: aggregate(row, batches) for key, row in rows.items()}
    seeds = sorted({seed for key, row in rows.items() if row["experiment"] != "baseline" for seed in row["units"]})
    sweep_s = [s_label(s) for s in config["recovery_sparsities"]]
    debug = any(p["debug"] for a in aggs.values() for p in a["batches"].values())
    has_peak = any(p["peak"] for a in aggs.values() for p in a["batches"].values())
    md, written = [], []

    def table(title, headers, body, note=None, csv_name=None):
        md.extend([f"## {title}", "", markdown_table(headers, body), ""])
        if note:
            md.extend([note, ""])
        if csv_name:
            _write_csv(out_dir / csv_name, headers, body)
            written.append(out_dir / csv_name)

    def get(row_id, storage_format):
        return aggs.get((row_id, storage_format))

    # Row order for the combined table.
    order = [("original", "dense")] + [(f"magnitude_{s}", "csr") for s in sweep_s]
    experiments = [e for e in config["experiments"] if any(r["experiment"] == e for r in rows.values())]
    for experiment in experiments:
        keys = sorted((k for k, r in rows.items() if r["experiment"] == experiment),
                      key=lambda k: (rows[k]["kind"] != "statistical", json.dumps(rows[k]["params"], sort_keys=True),
                                     k[0], k[1]))
        if experiment == "statistical":
            keys = sorted(keys, key=lambda k: (rows[k]["params"].get("k", 0), rows[k]["kind"] != "statistical"))
        if experiment == "int8":
            keys = [k for k in (("original", "int8_weight_only"), ("original", "int8_w8a8")) if k in keys]
        order += keys
    order = [k for k in order if k in aggs]

    def shown(path):
        path = Path(path).resolve()
        try:
            return path.relative_to(Path(__file__).resolve().parents[2])
        except ValueError:
            return path.name

    md += ["# MLP follow-up experiments: statistical pruning, recovery, protected output layer, LoRA, INT8", "",
           f"Results from `{shown(followup_dir)}`; baselines (original dense model, magnitude pruning without "
           f"fine-tuning) from `{shown(baseline_dir)}`. Model seeds: {', '.join(map(str, seeds))}. Values are means over model seeds, "
           "ranges in parentheses are min - max. Test accuracy is reported for the configuration selected on "
           "validation data. Every deployed format is checked against a dense implementation of the same "
           "modified model.", ""]
    if debug:
        md += ["**CPU debug run: latency is for checking the pipeline only, not a GPU result. Peak GPU memory is "
               "not available.**", ""]
    if not include_baseline_timing:
        md += ["Baseline timing is omitted because the baseline run is on a different device.", ""]

    # 1. Combined table
    combined_headers = ["Experiment", "Model", "Format", "Sparsity", "Test acc", "vs original (pp)",
                        "Stored bytes", "Resident b256", "Peak b1", "Peak b256",
                        "Latency b1 (ms)", "b32", "b256", "Matches reference"]
    body, csv_rows = [], []
    for key in order:
        a, row = aggs[key], rows[key]
        matches = "-" if a["equivalent"] is None else ("yes" if a["equivalent"] else "NO")
        body.append([EXPERIMENT_TITLES[row["experiment"]].split(". ")[-1], row_label(*key), format_text(*key),
                     _pct(a["sparsity"], 1), _pct(a["test"]), _pp(a["delta"]), _bytes(a["stored"]),
                     _batch(a, 256, "resident", _mem), _batch(a, 1, "peak", _mem), _batch(a, 256, "peak", _mem),
                     _batch(a, 1, "latency", _ms), _batch(a, 32, "latency", _ms), _batch(a, 256, "latency", _ms),
                     matches])
        csv_row = [row["experiment"], key[0], key[1], row_label(*key), format_text(*key),
                   _mean(a["sparsity"]), _mean(a["val"]), _mean(a["test"]),
                   a["test"]["min"] if a["test"] else None, a["test"]["max"] if a["test"] else None,
                   _mean(a["delta"]), _mean(a["stored"]), _mean(a["weight"]), _mean(a["adapter"]),
                   a["epsilon"], a["equivalent"], len(a["seeds"])]
        for batch in batches:
            p = a["batches"].get(batch, {})
            csv_row += [_mean(p.get("latency")), _mean(p.get("resident")), _mean(p.get("peak"))]
        csv_rows.append(csv_row)
    table("Combined results", combined_headers, body,
          "Stored bytes: weight storage including indices, INT8 scales and LoRA adapters (biases, 1,640 B, "
          "excluded). Resident: torch.cuda.memory_allocated before the first pass (stored tensors, biases and the "
          "input batch). Peak: torch.cuda.max_memory_allocated over the timed passes; formats that run a dense "
          "matmul (FP32 dense, dense fc3, LoRA adapters, INT8) also hold the cuBLAS workspace. Latency: median "
          "of 50 synchronized passes after 20 warm-ups, median over repetitions, mean over seeds.")
    csv_headers = ["experiment", "row_id", "format", "model", "format_label", "sparsity", "val_accuracy",
                   "test_accuracy", "test_accuracy_min", "test_accuracy_max", "vs_original_pp", "stored_bytes",
                   "weight_bytes", "adapter_bytes", "max_epsilon", "equivalent", "seeds"]
    for batch in batches:
        csv_headers += [f"latency_ms_b{batch}", f"resident_bytes_b{batch}", f"peak_bytes_b{batch}"]
    _write_csv(out_dir / "combined.csv", csv_headers, csv_rows)
    written.append(out_dir / "combined.csv")

    plt = _pyplot()
    curve_rows = sorted(((rows[k]["params"]["sparsity"], aggs[k]) for k in aggs
                         if rows[k]["experiment"] == "baseline" and rows[k]["kind"] == "magnitude"),
                        key=lambda t: t[0])
    original = get("original", "dense")
    if plt is not None:
        def pts(keys, label_fn=lambda key: None):
            out = []
            for key in keys:
                a = aggs.get(key)
                if a is None or a["test"] is None:
                    continue
                peak = a["batches"].get(256, {}).get("peak")
                out.append((_mean(a["stored"]), _mean(peak), a["test"]["mean"], a["test"]["min"], a["test"]["max"],
                            label_fn(key)))
            return out

        k_label = lambda key: _k_text(rows[key]["params"]["k"])  # noqa: E731
        s_text = lambda key: key[0].split("_")[1][1:] + "%"  # noqa: E731
        magnitude = [(f"magnitude_{s}", "csr") for s in sweep_s]
        panels = []
        if "statistical" in experiments:
            panels.append((EXPERIMENT_TITLES["statistical"], [
                ("statistical rule", SLOTS[0], pts([k for k in order if k[0].startswith("statistical_")], k_label)),
                ("magnitude, same count", SLOTS[1], pts([k for k in order if k[0].startswith("magmatch_")]))]))
        if "finetune" in experiments:
            panels.append((EXPERIMENT_TITLES["finetune"], [
                ("+ fine-tune", SLOTS[0], pts([k for k in order if k[0].startswith("finetune_")], s_text)),
                ("pruning only", SLOTS[1], pts(magnitude))]))
        if "protect_fc3" in experiments:
            panels.append((EXPERIMENT_TITLES["protect_fc3"], [
                ("fc3 dense + fine-tune", SLOTS[0],
                 pts([k for k in order if k[0].startswith("fc3dense_") and k[0].endswith("finetune")], s_text)),
                ("fc3 dense, pruning only", SLOTS[1],
                 pts([k for k in order if k[0].startswith("fc3dense_") and not k[0].endswith("finetune")])),
                ("all pruned + fine-tune", SLOTS[2], pts([k for k in order if k[0].startswith("finetune_")]))]))
        if "lora" in experiments:
            panels.append((EXPERIMENT_TITLES["lora"], [
                ("+ LoRA (rank 4)", SLOTS[0], pts([k for k in order if k[0].startswith("lora_")], s_text)),
                ("pruning only", SLOTS[1], pts(magnitude)),
                ("+ fine-tune", SLOTS[2], pts([k for k in order if k[0].startswith("finetune_")]))]))
        if "int8" in experiments:
            panels.append((EXPERIMENT_TITLES["int8"], [
                ("INT8 weight-only", SLOTS[0], pts([("original", "int8_weight_only")])),
                ("INT8 W8A8", SLOTS[1], pts([("original", "int8_w8a8")]))]))
        curve = {"stored": [(_mean(a["stored"]), a["test"]["mean"]) for _, a in curve_rows],
                 "peak": [(_mean(a["batches"][256]["peak"]), a["test"]["mean"]) for _, a in curve_rows
                          if a["batches"].get(256, {}).get("peak")]}
        orig = {"acc": original["test"]["mean"], "stored": _mean(original["stored"]),
                "peak": _mean(original["batches"].get(256, {}).get("peak"))} if original else {}
        accuracy_memory_figure(plt, out_dir / "accuracy_vs_memory.png", panels, curve, orig, has_peak,
                               f"Fashion-MNIST, MLP 784-300-100-10. Mean over model seeds "
                               f"({', '.join(map(str, seeds))}); whiskers min - max. Gray: the original sweep's "
                               "global magnitude pruning without fine-tuning (CSR) and the original FP32 model.")
        written.append(out_dir / "accuracy_vs_memory.png")

        groups = []
        for experiment in ["baseline"] + experiments:
            items = [(row_label(*k), {b: aggs[k]["batches"].get(b, {}).get("latency") for b in batches})
                     for k in order if rows[k]["experiment"] == experiment]
            items = [(label, by_batch) for label, by_batch in items if any(by_batch.values())]
            if items:
                groups.append((EXPERIMENT_TITLES[experiment], items))
        if groups:
            latency_figure(plt, out_dir / "latency_by_model.png", groups,
                           ("CPU DEBUG RUN, not a GPU result. " if debug else "") +
                           "Per seed: median of per-repetition medians (50 timed passes each); marker = mean "
                           "over seeds, whiskers min - max. Deployed formats as in the combined table.")
            written.append(out_dir / "latency_by_model.png")

    # Experiment sections
    if "statistical" in experiments:
        stat_keys = [k for k in order if k[0].startswith("statistical_")]
        body, stat_pts, match_pts, wins = [], [], [], 0
        for key in stat_keys:
            match_key = ("magmatch_" + key[0].split("_", 1)[1], "csr")
            a, m = aggs[key], aggs.get(match_key)
            k = rows[key]["params"]["k"]

            def layers(row_key):
                per = defaultdict(list)
                for u in rows[row_key]["units"].values():
                    for name, layer in u["variant"]["pruning"]["layers"].items():
                        per[name].append(layer["actual_sparsity"])
                return " / ".join(f"{100 * statistics.fmean(v):.1f}" for v in per.values())

            diff = _stats([rows[key]["units"][s]["test"] - rows[match_key]["units"][s]["test"]
                           for s in rows[key]["units"] if m and s in rows[match_key]["units"]])
            wins += bool(diff and diff["mean"] > 0)
            body.append([_k_text(k), _pct(a["sparsity"], 2), layers(key), layers(match_key) if m else "n/a",
                         _pct(a["test"]), _pct(m["test"]) if m else "n/a",
                         _range(diff, lambda v: f"{100 * v:+.2f}") if diff else "n/a", _bytes(a["stored"])])
            stat_pts.append((a["sparsity"]["mean"], a["test"]["mean"], a["test"]["min"], a["test"]["max"], _k_text(k)))
            if m:
                match_pts.append((m["sparsity"]["mean"], m["test"]["mean"], m["test"]["min"], m["test"]["max"]))
        verdict = (f"The statistical rule scores higher than global magnitude with the same retained count at "
                   f"{wins} of {len(stat_keys)} thresholds (by the mean over seeds).")
        image = None
        if plt is not None and stat_pts:
            statistical_figure(plt, out_dir / "statistical_vs_magnitude.png",
                               [(s, a["test"]["mean"]) for s, a in curve_rows], stat_pts, match_pts,
                               "Test accuracy without fine-tuning; mean over model seeds, whiskers min - max. "
                               "Each statistical point is paired with a global-magnitude model that keeps exactly "
                               "the same number of weights.")
            written.append(out_dir / "statistical_vs_magnitude.png")
            image = "statistical_vs_magnitude.png"
        md += [f"## {EXPERIMENT_TITLES['statistical']} vs matched global magnitude", "", verdict, ""]
        if image:
            md += [f"![Statistical vs magnitude]({image})", ""]
        table_headers = ["k", "Sparsity", "Per-layer sparsity, statistical (fc1 / fc2 / fc3 %)",
                         "Per-layer, matched magnitude", "Test acc, statistical", "Test acc, matched magnitude",
                         "Difference (pp)", "Stored bytes (both, CSR)"]
        md += [markdown_table(table_headers, body), "",
               "Per matrix: keep w when |w - mean| > k * std (population std of that matrix's original weights, "
               "biases excluded). The matched baseline is global magnitude pruning that keeps exactly the same "
               "total number of weights, so both store the same CSR bytes. No fine-tuning.", ""]
        _write_csv(out_dir / "statistical.csv", table_headers, body)
        written.append(out_dir / "statistical.csv")

    def recovery_rows(prefix, compare):
        body = []
        for s in sweep_s:
            key = (f"{prefix}_{s}", "csr")
            if key not in aggs:
                continue
            a = aggs[key]
            selected = ", ".join(f"m{seed}: lr {u['variant']['recovery']['selected']['learning_rate']:g} "
                                 f"ep {u['variant']['recovery']['selected']['epoch']}"
                                 for seed, u in sorted(rows[key]["units"].items()) if u["variant"]["recovery"])
            base = get(*compare(s))
            body.append([f"{s[1:]}%", _pct(base["test"]) if base else "n/a", _pct(a["test"]),
                         _pct(a["val"]), _bytes(a["stored"]), selected])
        return body

    if "finetune" in experiments:
        body = recovery_rows("finetune", lambda s: (f"magnitude_{s}", "csr"))
        md += [f"## {EXPERIMENT_TITLES['finetune']}", ""]
        headers = ["Sparsity", "Pruning only (test)", "Fine-tuned (test)", "Fine-tuned (val)", "Stored bytes",
                   "Selected on validation"]
        md += [markdown_table(headers, body), "",
               f"Masks fixed during fine-tuning ({config['recovery_epochs']} epochs per learning rate in "
               f"{config['recovery_learning_rates']}; Adam, batch {config['recovery_batch_size']}); removed weights "
               "stay exactly zero, so the CSR bytes equal the pruning-only model's.", ""]
        _write_csv(out_dir / "finetune.csv", headers, body)
        written.append(out_dir / "finetune.csv")

    if "protect_fc3" in experiments:
        body = []
        for s in sweep_s:
            before, after = get(f"fc3dense_{s}", "csr"), get(f"fc3dense_{s}_finetune", "csr")
            if not before:
                continue
            matching = [u["variant"]["matching"] for u in rows[(f"fc3dense_{s}", "csr")]["units"].values()]
            target = matching[0]["target_weight_bytes"]
            mismatch = sorted({m["mismatch_bytes"] for m in matching})
            exp2 = get(f"finetune_{s}", "csr")
            body.append([f"{s[1:]}%", f"{target:,}", ", ".join(f"{m:+,}" for m in mismatch), _pct(before["sparsity"], 2),
                         _pct(before["test"]), _pct(after["test"]) if after else "n/a",
                         _pct(exp2["test"]) if exp2 else "n/a"])
        headers = ["Matched to", "Target bytes (exp. 2)", "Byte mismatch", "Model sparsity",
                   "fc3 dense, pruning only", "fc3 dense + fine-tune", "All pruned + fine-tune (exp. 2)"]
        md += [f"## {EXPERIMENT_TITLES['protect_fc3']}", "", markdown_table(headers, body), "",
               "fc3 stays dense and unpruned; fc1 and fc2 are pruned by global magnitude with the retained count "
               "chosen so the stored weight bytes (CSR fc1 + fc2, dense fc3) match experiment 2's all-CSR model. "
               "The same fine-tuning procedure as experiment 2.", ""]
        _write_csv(out_dir / "protect_fc3.csv", headers, body)
        written.append(out_dir / "protect_fc3.csv")

    if "lora" in experiments:
        body = []
        for s in sweep_s:
            lora, base, ft = get(f"lora_{s}", "csr"), get(f"magnitude_{s}", "csr"), get(f"finetune_{s}", "csr")
            if not lora:
                continue
            row = [f"{s[1:]}%", _pct(base["test"]) if base else "n/a", _pct(lora["test"]),
                   _pct(ft["test"]) if ft else "n/a", _bytes(lora["adapter"]), _bytes(lora["stored"])]
            for batch in (1, 256):
                lat_l = lora["batches"].get(batch, {}).get("latency")
                lat_b = base["batches"].get(batch, {}).get("latency") if base else None
                row.append(f"{_mean(lat_l):.4f} vs {_mean(lat_b):.4f}" if lat_l and lat_b else
                           (_ms(lat_l) if lat_l else "n/a"))
            body.append(row)
        headers = ["Sparsity", "Pruning only (test)", "LoRA (test)", "Fine-tune (test)", "Adapter bytes",
                   "Stored bytes (CSR + adapters)", "Latency b1: LoRA vs pruning only (ms)", "Latency b256"]
        md += [f"## {EXPERIMENT_TITLES['lora']}", "", markdown_table(headers, body), "",
               f"Rank-{config['lora_rank']} adapters (alpha {config['lora_alpha']}) on "
               f"{', '.join(config['lora_layers'])}; pruned weights and biases frozen; same budget as experiment 2. "
               "At inference the adapters run as a separate low-rank product next to the CSR weights.", ""]
        _write_csv(out_dir / "lora.csv", headers, body)
        written.append(out_dir / "lora.csv")

    if "int8" in experiments:
        body = []
        for key in [("original", "dense"), ("original", "int8_weight_only"), ("original", "int8_w8a8")]:
            a = aggs.get(key)
            if not a:
                continue
            body.append([format_text(*key), _pct(a["test"]), _pp(a["delta"]), _bytes(a["stored"]),
                         _batch(a, 1, "latency", _ms), _batch(a, 32, "latency", _ms), _batch(a, 256, "latency", _ms),
                         _batch(a, 1, "peak", _mem), _batch(a, 256, "peak", _mem)])
        w8a8_units = rows.get(("original", "int8_w8a8"), {}).get("units", {}).values()
        probes = [r["int_mm_support"] for u in w8a8_units for reps in u["bench"].values() for r in reps
                  if r.get("int_mm_support")] + [u["int_mm"] for u in w8a8_units if u.get("int_mm")]
        probe = probes[0] if probes else None
        headers = ["Format", "Test acc", "vs FP32 (pp)", "Stored bytes", "Latency b1 (ms)", "b32", "b256",
                   "Peak b1", "Peak b256"]
        md += [f"## {EXPERIMENT_TITLES['int8']}", "", markdown_table(headers, body), ""]
        if probe:
            md += [f"INT8 matmul support check on `{probe['device']}`: `torch._int_mm` "
                   f"{'ran and matched an exact int64 product' if probe.get('supported') else 'NOT supported'}"
                   f" (smallest batch accepted: {probe.get('min_rows', 'n/a')} rows; smaller batches are zero-padded).",
                   ""]
        md += ["Weight-only: int8 weights with one FP32 scale per output row, dequantized on the device each pass "
               "(FP32 compute). W8A8: activations quantized per sample at run time, int8 x int8 -> int32 matmul "
               "(torch._int_mm), weights zero-padded to multiples of 8. Stored bytes include the scales and padding.",
               ""]
        _write_csv(out_dir / "int8.csv", headers, body)
        written.append(out_dir / "int8.csv")

    images =[name for name in ("accuracy_vs_memory.png", "latency_by_model.png") if (out_dir / name).exists()]
    if images:
        insert_at = md.index("## Combined results")
        md[insert_at:insert_at] = [f"![{name}]({name})" for name in images] + [""]

    # Memory by batch
    if has_peak:
        body = []
        for key in order:
            a = aggs[key]
            if not a["batches"]:
                continue
            body.append([row_label(*key), format_text(*key)] +
                        [_batch(a, b, field, _mem) for b in batches for field in ("resident", "peak")])
        headers = ["Model", "Format"] + [f"{name} b{b}" for b in batches for name in ("Resident", "Peak")]
        table("GPU memory by batch size", headers, body, None, "gpu_memory.csv")

    # Equivalence
    body = []
    for key in order:
        a = aggs[key]
        if a["epsilon"] is None:
            continue
        bench_eps = [p["check_epsilon"] for p in a["batches"].values() if p.get("check_epsilon") is not None]
        body.append([row_label(*key), format_text(*key), f"{a['epsilon']:.1e}",
                     "yes" if a["equivalent"] else "NO", "yes" if a["bitwise"] else "no",
                     f"{100 * a['agreement']:.2f}%",
                     ("yes" if all(p["checks_equivalent"] for p in a["batches"].values()) else "NO")
                     if a["batches"] else "n/a",
                     f"{max(bench_eps):.1e}" if bench_eps else "n/a"])
    table("Equivalence with a dense implementation of the same modified model",
          ["Model", "Format", "Max epsilon (eval)", "Equivalent", "Bitwise", "Min agreement",
           "Benchmark batches equivalent", "Max epsilon (benchmark)"], body,
          "Reference: pruned dense with adapters kept separate (CSR), FP32 dense with the dequantized int8 weights "
          "(INT8 weight-only), the same activation quantization with an exact integer matmul (INT8 W8A8). "
          "Evaluation over 5,000 validation + 10,000 test images; tolerance allclose(atol=1e-5, rtol=1e-4).",
          "equivalence.csv")

    md += ["## Energy", "", "Not measured: no reliable energy method has been established for the harness yet.", ""]
    (out_dir / "summary.md").write_text("\n".join(md))
    written.insert(0, out_dir / "summary.md")
    return written

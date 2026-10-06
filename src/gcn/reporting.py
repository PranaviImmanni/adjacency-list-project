"""Turn the JSON results into tables and charts.

Reads the files written by scripts/compare_gcn_formats.py and
scripts/benchmark_gcn.py (results/raw/*.json) and produces:
  - plain-text tables (returned as a string, printed by the script),
  - summary.md: Markdown tables + embedded charts (open with a Markdown preview),
  - PNG charts: adjacency memory per dataset, latency per dataset and device.

Latency from any non-CUDA device is labeled debug-only everywhere it appears.
"""

import json
import statistics
import textwrap
from collections import defaultdict
from pathlib import Path

import torch

from .adjacency import STORAGE_FORMATS
from .config import PROJECT_ROOT

# Chart styling: one series, so one validated color for every bar; text in ink tones.
SERIES = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(results_dir):
    """Return (comparisons, benchmarks), keeping only the latest file per run.

    comparisons: {(dataset, device): comparison dict}
    benchmarks:  {(dataset, device): {format: [one dict per repetition]}}
    """
    comparisons, runs = {}, {}
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"skipping unreadable {path}")
            continue
        if "formats" in data and "all_equivalent" in data:
            data["_mtime"] = path.stat().st_mtime
            key = (data["dataset"], data["device"])
            if key not in comparisons or data["_mtime"] > comparisons[key]["_mtime"]:
                comparisons[key] = data
        elif "latency_ms" in data and "schema_version" in data:
            key = (data["dataset"], data["device"], data["format"], data["repetition"])
            if key not in runs or data["timestamp"] > runs[key]["timestamp"]:
                runs[key] = data

    benchmarks = defaultdict(lambda: defaultdict(list))
    for (dataset, device, storage_format, _), run in sorted(runs.items()):
        benchmarks[(dataset, device)][storage_format].append(run)
    return comparisons, benchmarks


def _load_checkpoint_info(checkpoint_path):
    path = Path(checkpoint_path)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def human_bytes(n):
    if n is None:
        return "n/a"
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} B"


def _pct(x):
    return f"{100 * x:.2f}%"


def _yes_no(flag):
    return "yes" if flag else "NO"


def _ordered_formats(formats):
    return [f for f in STORAGE_FORMATS if f in formats]


def text_table(headers, rows):
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]

    def line(cells):
        return "  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w)
                         for i, (c, w) in enumerate(zip(cells, widths)))

    return "\n".join([line(headers), "  ".join("-" * w for w in widths), *(line(r) for r in rows)])


def markdown_table(headers, rows):
    def cells(values):
        return "| " + " | ".join(str(v).replace("|", "\\|") for v in values) + " |"

    out = [cells(headers), "|" + "|".join([":---"] + ["---:"] * (len(headers) - 1)) + "|"]
    out += [cells(row) for row in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def training_table(checkpoint):
    headers = ["Best epoch", "Val acc", "Test acc", "Seed", "Hidden", "Dropout", "Trained with"]
    rows = [[checkpoint["best_epoch"], _pct(checkpoint["best_validation_accuracy"]),
             _pct(checkpoint["test_accuracy"]), checkpoint["seed"], checkpoint["hidden_dim"],
             checkpoint["dropout"], checkpoint.get("train_adjacency_format", "coo").upper()]]
    return headers, rows


def equivalence_table(comparison):
    headers = ["Format", "Test acc", "Acc diff (pp)", "Agree (all)", "Agree (test)",
               "Mean |Δlogit|", "Max |Δlogit|", "allclose", "Equivalent"]
    rows = []
    for storage_format in _ordered_formats(comparison["formats"]):
        m = comparison["formats"][storage_format]
        rows.append([storage_format, _pct(m["test_accuracy"]), f"{m['accuracy_difference_pp']:+.2f}",
                     _pct(m["prediction_agreement_all"]), _pct(m["prediction_agreement_test"]),
                     f"{m['mean_abs_logit_diff']:.1e}", f"{m['max_abs_logit_diff']:.1e}",
                     _yes_no(m["allclose"]), _yes_no(m["equivalent"])])
    return headers, rows


def memory_rows(dataset, comparisons, benchmarks):
    """{format: (bytes, breakdown or None)} plus the dense size, from any device's results.

    Adjacency bytes do not depend on the device, so any run will do.
    """
    sizes, dense_bytes = {}, None
    for (ds, _), by_format in benchmarks.items():
        if ds != dataset:
            continue
        for storage_format, reps in by_format.items():
            sizes[storage_format] = (reps[0]["adjacency_bytes"], reps[0]["adjacency_bytes_breakdown"])
            dense_bytes = reps[0]["dense_adjacency_bytes_estimate"]
    for (ds, _), comparison in comparisons.items():
        if ds != dataset:
            continue
        for storage_format, m in comparison["formats"].items():
            sizes.setdefault(storage_format, (m["adjacency_bytes"], None))
    if "dense" in sizes:
        dense_bytes = sizes["dense"][0]
    return {f: sizes[f] for f in _ordered_formats(sizes)}, dense_bytes


def memory_table(sizes, dense_bytes):
    headers = ["Format", "Bytes", "Size", "% of dense", "Components (bytes)"]
    rows = []
    for storage_format, (nbytes, breakdown) in sizes.items():
        components = " + ".join(f"{k} {v:,}" for k, v in breakdown.items()) if breakdown else "-"
        share = _pct(nbytes / dense_bytes) if dense_bytes else "n/a"
        rows.append([storage_format, f"{nbytes:,}", human_bytes(nbytes), share, components])
    return headers, rows


def latency_stats(reps):
    """Headline = median of per-repetition medians; other stats pool every timed pass."""
    samples = [t for r in reps for t in r["latency_samples_ms"]]
    rep_medians = [r["latency_ms"]["median"] for r in reps]
    peak_alloc = [r["cuda_peak_allocated_bytes"] for r in reps if r["cuda_peak_allocated_bytes"] is not None]
    peak_reserved = [r["cuda_peak_reserved_bytes"] for r in reps if r["cuda_peak_reserved_bytes"] is not None]
    return {
        "reps": len(reps),
        "runs_per_rep": reps[0]["measured_runs"],
        "median": statistics.median(rep_medians),
        "rep_median_min": min(rep_medians),
        "rep_median_max": max(rep_medians),
        "mean": statistics.fmean(samples),
        "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min": min(samples),
        "max": max(samples),
        "peak_allocated": max(peak_alloc) if peak_alloc else None,
        "peak_reserved": max(peak_reserved) if peak_reserved else None,
        "equivalent": all(r["prediction"]["equivalent"] for r in reps),
    }


def latency_table(by_format):
    headers = ["Format", "Reps", "Median (ms)", "Mean (ms)", "Std (ms)", "Min (ms)", "Max (ms)",
               "Peak CUDA alloc", "Peak CUDA reserved", "Matches COO"]
    rows = []
    for storage_format in _ordered_formats(by_format):
        s = latency_stats(by_format[storage_format])
        rows.append([storage_format, s["reps"], f"{s['median']:.3f}", f"{s['mean']:.3f}", f"{s['std']:.3f}",
                     f"{s['min']:.3f}", f"{s['max']:.3f}", human_bytes(s["peak_allocated"]),
                     human_bytes(s["peak_reserved"]), _yes_no(s["equivalent"])])
    return headers, rows


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })
    return plt


def _rounded_bar(ax, y, length, thickness, color, radius_px=4):
    """Horizontal bar, square at the x=0 baseline, 4px rounded at the data end."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    if length <= 0:
        return
    box = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    rx = min(radius_px * (x1 - x0) / box.width, length)
    ry = min(radius_px * abs(y1 - y0) / box.height, thickness / 2)
    top, bottom = y - thickness / 2, y + thickness / 2
    vertices = [(0, top), (length - rx, top), (length, top), (length, top + ry),
                (length, bottom - ry), (length, bottom), (length - rx, bottom), (0, bottom), (0, top)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
             MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=color, edgecolor="none", zorder=3))


def _bar_chart(plt, path, labels, values, tip_labels, title, subtitle, x_label, ranges=None):
    """One-series horizontal bar chart with value labels at the bar tips.

    ranges: optional [(low, high)] per bar, drawn as a thin whisker.
    """
    from matplotlib.ticker import MaxNLocator

    subtitle_lines = textwrap.wrap(subtitle, 100)
    n = len(labels)
    width_in, left_in, right_in = 7.5, 0.9, 1.9
    row_in, bar_in = 0.5, 0.22  # bar ~22px thick at 100 px/in, under the 24px cap
    top_in = 0.45 + 0.17 * len(subtitle_lines)
    bottom_in = 0.6
    height_in = top_in + n * row_in + bottom_in

    fig = plt.figure(figsize=(width_in, height_in), dpi=100)
    ax = fig.add_axes([left_in / width_in, bottom_in / height_in,
                       1 - (left_in + right_in) / width_in, n * row_in / height_in])

    highest = max([*values, *(hi for _, hi in (ranges or []))])
    ticks = MaxNLocator(nbins=5).tick_values(0, highest)
    ax.set_xlim(0, ticks[-1] if ticks[-1] >= highest else highest)
    ax.set_ylim(n - 0.5, -0.5)  # first label at the top
    thickness = bar_in / row_in

    for i, value in enumerate(values):
        _rounded_bar(ax, i, value, thickness, SERIES)
        tip = value
        if ranges and ranges[i][1] > ranges[i][0]:
            low, high = ranges[i]
            ax.plot([low, high], [i, i], color=INK_SECONDARY, linewidth=1, zorder=4, solid_capstyle="butt")
            ax.plot([low, low], [i - 0.12, i + 0.12], color=INK_SECONDARY, linewidth=1, zorder=4)
            ax.plot([high, high], [i - 0.12, i + 0.12], color=INK_SECONDARY, linewidth=1, zorder=4)
            tip = max(value, high)
        ax.annotate(tip_labels[i], xy=(tip, i), xytext=(6, 0), textcoords="offset points",
                    va="center", ha="left", color=INK, fontsize=9.5, annotation_clip=False)

    ax.set_yticks(range(n), labels)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="y", length=0, labelsize=10.5, labelcolor=INK, pad=8)
    ax.tick_params(axis="x", length=0, labelsize=9, labelcolor=MUTED, pad=6)
    ax.xaxis.grid(True, color=GRID, linewidth=0.75, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["left"].set_zorder(5)
    ax.set_xlabel(x_label, color=MUTED, fontsize=9, labelpad=6)

    fig.text(left_in / width_in, 1 - 0.3 / height_in, title, fontsize=12.5, fontweight="bold",
             color=INK, va="baseline")
    for k, line in enumerate(subtitle_lines):
        fig.text(left_in / width_in, 1 - (0.52 + 0.17 * k) / height_in, line, fontsize=9,
                 color=INK_SECONDARY, va="baseline")

    fig.savefig(path, dpi=200)
    plt.close(fig)


def memory_chart(plt, path, dataset, sizes, dense_bytes):
    labels = list(sizes)
    values = [sizes[f][0] / 1e6 for f in labels]
    tips = []
    for f in labels:
        nbytes = sizes[f][0]
        share = f" · {100 * nbytes / dense_bytes:.1f}% of dense" if dense_bytes and f != "dense" else ""
        tips.append(human_bytes(nbytes) + share)
    _bar_chart(plt, path, labels, values, tips,
               title=f"Adjacency memory · {dataset}",
               subtitle="Exact bytes of the stored tensors (FP32 values, int64 indices). Same on every device.",
               x_label="Adjacency size (MB, 1 MB = 1,000,000 bytes)")


def latency_chart(plt, path, dataset, device, by_format):
    labels = _ordered_formats(by_format)
    stats = [latency_stats(by_format[f]) for f in labels]
    reps = max(s["reps"] for s in stats)
    runs = stats[0]["runs_per_rep"]
    reps_text = f"{reps} repetition{'s' if reps != 1 else ''} × {runs} timed forward passes"
    if device.startswith("cuda"):
        subtitle = (f"Median of per-repetition medians ({reps_text}). "
                    "Whisker = range of repetition medians. FP32, TF32 off.")
    else:
        subtitle = (f"DEBUG ONLY: {device.upper()} timing on the development machine, not a GPU result. "
                    f"Median of per-repetition medians ({reps_text}).")
    _bar_chart(plt, path, labels, [s["median"] for s in stats], [f"{s['median']:.2f} ms" for s in stats],
               title=f"Inference latency · {dataset} · {device}",
               subtitle=subtitle, x_label="Forward-pass latency (ms)",
               ranges=[(s["rep_median_min"], s["rep_median_max"]) for s in stats])


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def summarize(results_dir, out_dir):
    """Write summary.md and charts to out_dir. Returns (text report, list of written files)."""
    comparisons, benchmarks = load_results(results_dir)
    if not comparisons and not benchmarks:
        raise FileNotFoundError(f"no comparison or benchmark JSON files in {results_dir}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    written = []
    text = []
    try:
        source = Path(results_dir).resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        source = results_dir
    md = ["# GCN results summary", "",
          f"Generated by `scripts/summarize_gcn_results.py` from `{source}`. "
          "Latency measured on CPU (or any non-CUDA device) is for debugging only and is not a GPU result.", ""]
    if plt is None:
        md += ["_matplotlib is not installed, so no charts were drawn._", ""]

    def section(title, headers, rows, level=3, note=None, image=None):
        text.extend([title, text_table(headers, rows)])
        md.extend([f"{'#' * level} {title}", ""])
        if image:
            md.extend([f"![{title}]({image})", ""])
        md.extend([markdown_table(headers, rows), ""])
        if note:
            text.append(note)
            md.extend([note, ""])
        text.append("")

    datasets = sorted({k[0] for k in comparisons} | {k[0] for k in benchmarks})
    for dataset in datasets:
        text += ["=" * 72, f"DATASET: {dataset}", "=" * 72, ""]
        md += [f"## {dataset}", ""]

        checkpoint = next((_load_checkpoint_info(c["checkpoint"]) for (ds, _), c in comparisons.items()
                           if ds == dataset), None)
        if checkpoint:
            section("Training (checkpoint)", *training_table(checkpoint))

        for (ds, device), comparison in sorted(comparisons.items()):
            if ds != dataset:
                continue
            verdict = (f"All formats equivalent: {_yes_no(comparison['all_equivalent'])} "
                       f"(reference: {comparison['reference_format'].upper()}; "
                       f"allclose atol=1e-5, rtol=1e-4)")
            section(f"Prediction equivalence · {device}", *equivalence_table(comparison), note=verdict)

        sizes, dense_bytes = memory_rows(dataset, comparisons, benchmarks)
        if sizes:
            image = None
            if plt:
                image = f"{dataset}_adjacency_memory.png"
                memory_chart(plt, out_dir / image, dataset, sizes, dense_bytes)
                written.append(out_dir / image)
            section("Adjacency memory", *memory_table(sizes, dense_bytes), image=image)

        for (ds, device), by_format in sorted(benchmarks.items()):
            if ds != dataset:
                continue
            debug = not device.startswith("cuda")
            image = None
            if plt:
                image = f"{dataset}_latency_{device.replace(':', '')}.png"
                latency_chart(plt, out_dir / image, dataset, device, by_format)
                written.append(out_dir / image)
            title = f"Inference latency · {device}" + (" (DEBUG ONLY, not a GPU result)" if debug else "")
            section(title, *latency_table(by_format), image=image)

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(md))
    written.insert(0, summary_path)
    return "\n".join(text), written

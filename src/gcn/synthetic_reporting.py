"""Tables and charts for the synthetic pilot, size and density sweeps.

Reads the per-run JSON files written by scripts/benchmark_synthetic.py, one
sub-folder per sweep (results/synthetic/<sweep>_<device>/), and produces for
each sweep:
  - tables of the graphs, latency, adjacency bytes, peak GPU memory and equivalence,
  - where COO and CSR cross dense on latency, adjacency bytes and peak memory,
  - line charts of each of those against the swept variable.

Aggregation: per graph seed, the median of the repetition medians; the
headline is the median over graph seeds, and the range shown is the spread
over graph seeds (how much the result depends on the random graph).
"""

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from .config import PROJECT_ROOT
from .reporting import (BASELINE, GRID, INK, INK_SECONDARY, MUTED, SURFACE, _ordered_formats, _pyplot,
                        _yes_no, human_bytes, markdown_table, text_table)

SWEEP_ORDER = ("pilot", "size", "density")

# Fixed per format (categorical slots 1-3, validated all-pairs on SURFACE).
# CSR's aqua is under 3:1 contrast, so every chart also has direct labels,
# a distinct marker per format, and the tables carry every value.
FORMAT_STYLE = {
    "dense": {"color": "#2a78d6", "marker": "o"},
    "coo": {"color": "#eb6834", "marker": "s"},
    "csr": {"color": "#1baf7a", "marker": "^"},
}

METRICS = (
    # key, table/crossover label, chart file suffix, chart title
    ("latency", "Latency", "latency", "Inference latency"),
    ("adjacency_bytes", "Adjacency bytes", "adjacency_memory", "Adjacency memory"),
    ("peak", "Peak GPU memory", "peak_memory", "Peak GPU memory during inference"),
)


# ---------------------------------------------------------------------------
# Loading and aggregation
# ---------------------------------------------------------------------------

def load_sweeps(results_dir):
    """{sweep folder name: [run dicts]}, for every sub-folder holding synthetic benchmark JSON."""
    sweeps = {}
    for folder in sorted(p for p in Path(results_dir).iterdir() if p.is_dir()):
        runs = []
        for path in sorted(folder.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                print(f"skipping unreadable {path}")
                continue
            if data.get("dataset") == "synthetic" and "latency_ms" in data:
                runs.append(data)
        if runs:
            sweeps[folder.name] = runs
    return dict(sorted(sweeps.items(), key=lambda kv: (_sweep_rank(kv[0]), kv[0])))


def _sweep_rank(name):
    return next((i for i, prefix in enumerate(SWEEP_ORDER) if name.startswith(prefix)), len(SWEEP_ORDER))


def sweep_variable(runs):
    """The graph parameter that varies: 'nodes', 'density' or 'avg_degree' (None for a single graph)."""
    if len({r["num_nodes"] for r in runs}) > 1:
        return "nodes"
    for variable in ("density", "avg_degree"):
        if len({r["graph"][f"requested_{variable}"] for r in runs}) > 1:
            return variable
    return None


def x_value(run, variable):
    if variable in ("density", "avg_degree"):
        return run["graph"][f"requested_{variable}"]
    return run["num_nodes"]


def format_x(x, variable):
    if variable == "density":
        return f"{100 * x:g}%"
    if variable == "avg_degree":
        return f"{x:g}"
    return f"{round(x):,}"


def _median_and_range(values):
    return statistics.median(values), min(values), max(values)


def aggregate(runs, variable):
    """Returns (graphs, points).

    graphs: {x: graph description}
    points: {x: {format: stats}}, stats holding (median, low, high) over
            graph seeds for latency and peak memory, plus bytes and equivalence.
    """
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # x -> format -> seed -> runs
    for run in runs:
        grouped[x_value(run, variable)][run["format"]][run["graph"]["graph_seed"]].append(run)

    graphs, points = {}, {}
    for x in sorted(grouped):
        by_format = grouped[x]
        everything = [r for seeds in by_format.values() for rs in seeds.values() for r in rs]
        fingerprints = defaultdict(set)
        for r in everything:
            fingerprints[r["graph"]["graph_seed"]].add(r["graph"]["graph_fingerprint"])
        g = everything[0]["graph"]
        graphs[x] = {
            "nodes": g["num_nodes"],
            "edges": g["num_edges"],
            "avg_degree": g["avg_degree"],
            "density": g["density"],
            "nnz": everything[0]["nnz"],
            "seeds": sorted(fingerprints),
            "same_graph": all(len(f) == 1 for f in fingerprints.values()),
            "device": everything[0]["device"],
            "debug_only": any(r["latency_is_debug_only"] for r in everything),
        }

        points[x] = {}
        for storage_format in _ordered_formats(by_format):
            seeds = by_format[storage_format]
            fmt_runs = [r for rs in seeds.values() for r in rs]
            latency = [statistics.median(r["latency_ms"]["median"] for r in rs) for rs in seeds.values()]
            peaks = [max(r["cuda_peak_allocated_bytes"] for r in rs) for rs in seeds.values()
                     if all(r["cuda_peak_allocated_bytes"] is not None for r in rs)]
            stats = {
                "latency": _median_and_range(latency),
                "adjacency_bytes": (statistics.median(r["adjacency_bytes"] for r in fmt_runs),) * 3,
                "peak": _median_and_range(peaks) if peaks else None,
                "max_abs_diff": max(r["prediction"]["max_abs_logit_diff"] for r in fmt_runs),
                "equivalent": all(r["prediction"]["equivalent"] for r in fmt_runs),
                "runs": len(fmt_runs),
            }
            points[x][storage_format] = stats
    return graphs, points


# ---------------------------------------------------------------------------
# Crossovers
# ---------------------------------------------------------------------------

# Differences smaller than this fraction of the dense value count as a tie. At
# the fixed-overhead floor (~0.45 ms) formats differ by well under 1%, which is
# within the run-to-run spread and must not be reported as a crossover.
TIE_TOLERANCE = 0.02


def compare_to_dense(sparse, dense, tolerance=TIE_TOLERANCE):
    """Per point: -1 where sparse is lower than dense, +1 where higher, 0 for a tie."""
    return [0 if abs(s - d) <= tolerance * d else (-1 if s < d else 1) for s, d in zip(sparse, dense)]


def find_crossovers(xs, sparse, dense, log_x=False, tolerance=TIE_TOLERANCE):
    """Where sparse switches between lower and higher than dense, ignoring ties.

    Returns [(x_low, x_high, estimate, ties, sparse_better_below)]: the last
    point on one side, the first on the other, and the tied points between
    them. With no ties in between, estimate interpolates linearly between the
    two points (in log x when log_x), so it is only as good as the sweep's
    spacing; with ties, estimate is None and the ties are the answer.
    """
    signs = compare_to_dense(sparse, dense, tolerance)
    decided = [i for i, s in enumerate(signs) if s != 0]
    found = []
    for a, b in zip(decided, decided[1:]):
        if signs[a] == signs[b]:
            continue
        ties = [xs[i] for i in range(a + 1, b)]
        estimate = None
        if not ties:
            d0, d1 = sparse[a] - dense[a], sparse[b] - dense[b]
            t = d0 / (d0 - d1)
            x0, x1 = xs[a], xs[b]
            estimate = (math.exp(math.log(x0) + t * (math.log(x1) - math.log(x0))) if log_x
                        else x0 + t * (x1 - x0))
        found.append((xs[a], xs[b], estimate, ties, signs[a] < 0))
    return found


def crossover_table(xs, points, variable, log_x):
    headers = ["Metric", "Format", "Crosses dense", "Estimate", "Better than dense"]
    rows = []

    def listed(values):
        return ", ".join(format_x(v, variable) for v in values)

    for key, label, _, _ in METRICS:
        for storage_format in ("coo", "csr"):
            if any(points[x].get(storage_format, {}).get(key) is None or points[x].get("dense", {}).get(key) is None
                   for x in xs):
                continue
            sparse = [points[x][storage_format][key][0] for x in xs]
            dense = [points[x]["dense"][key][0] for x in xs]
            found = find_crossovers(xs, sparse, dense, log_x)
            if not found:
                signs = compare_to_dense(sparse, dense)
                ties = [x for x, s in zip(xs, signs) if s == 0]
                verdict = ("at every point" if -1 in signs else "at no point") if len(ties) < len(xs) else "tied"
                if ties and len(ties) < len(xs):
                    verdict += f" (tied at {listed(ties)})"
                rows.append([label, storage_format, "not in this range", "-", verdict])
                continue
            for x0, x1, estimate, ties, better_below in found:
                rows.append([label, storage_format, f"between {format_x(x0, variable)} and {format_x(x1, variable)}",
                             f"≈ {format_x(_round_sig(estimate), variable)}" if estimate is not None
                             else f"tied at {listed(ties)}",
                             f"{'below' if better_below else 'above'} the crossover"])
    return headers, rows


def _round_sig(x, digits=2):
    return float(f"{x:.{digits}g}") if x else x


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def graph_table(graphs):
    """Actual graph sizes; the swept variable is one of these columns."""
    headers = ["Nodes", "Edges (m)", "Avg degree", "Density", "nnz(Â)", "Graph seeds", "Same graph in every format"]
    rows = [[f"{g['nodes']:,}", f"{g['edges']:,}", f"{g['avg_degree']:.2f}", f"{100 * g['density']:.3g}%",
             f"{g['nnz']:,}", ", ".join(map(str, g["seeds"])), _yes_no(g["same_graph"])]
            for g in graphs.values()]
    return headers, rows


def _x_header(variable):
    return {"nodes": "Nodes", "density": "Density", "avg_degree": "Avg degree"}.get(variable, "Graph")


def _cell(value, low, high, fmt, show_range):
    return f"{fmt(value)} ({fmt(low)}–{fmt(high)})" if show_range and high > low else fmt(value)


def metric_table(points, variable, key, fmt, best_label):
    """One row per x, one column per format, plus which format is lowest."""
    formats = _ordered_formats({f for p in points.values() for f in p})
    headers = [_x_header(variable), *formats, best_label]
    rows = []
    for x, by_format in points.items():
        row = [format_x(x, variable)]
        values = {}
        for f in formats:
            stats = by_format.get(f, {}).get(key)
            if stats is None:
                row.append("n/a")
                continue
            values[f] = stats[0]
            row.append(_cell(*stats, fmt, show_range=key != "adjacency_bytes"))
        row.append(min(values, key=values.get) if values else "-")
        rows.append(row)
    return headers, rows


def equivalence_table(points, variable):
    formats = _ordered_formats({f for p in points.values() for f in p})
    headers = [_x_header(variable), *[f"Max |Δlogit| {f}" for f in formats], "Runs", "All equivalent"]
    rows = []
    for x, by_format in points.items():
        rows.append([format_x(x, variable), *[f"{by_format[f]['max_abs_diff']:.1e}" for f in formats],
                     sum(by_format[f]["runs"] for f in formats),
                     _yes_no(all(by_format[f]["equivalent"] for f in formats))])
    return headers, rows


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _compact_bytes(n):
    return human_bytes(n).replace(".0 ", " ")


def _spread(positions, min_gap):
    """Push label positions (pixels, any order) apart by at least min_gap, keeping their order."""
    order = sorted(range(len(positions)), key=positions.__getitem__)
    placed = [positions[i] for i in order]
    for k in range(1, len(placed)):
        placed[k] = max(placed[k], placed[k - 1] + min_gap)
    shift = (sum(placed) - sum(positions[i] for i in order)) / len(placed)  # re-center the group
    out = [0.0] * len(positions)
    for k, i in enumerate(order):
        out[i] = placed[k] - shift
    return out


def line_chart(plt, path, xs, series, title, subtitle, x_label, variable, log_x, log_y, y_fmt):
    """One line per format against the swept variable, with seed-range whiskers and end labels.

    series: {format: [(median, low, high) per x]}.
    """
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator

    width_in, height_in = 7.5, 4.8
    left_in, right_in, top_in, bottom_in = 0.95, 1.0, 1.15, 0.6
    fig = plt.figure(figsize=(width_in, height_in), dpi=100)
    ax = fig.add_axes([left_in / width_in, bottom_in / height_in,
                       1 - (left_in + right_in) / width_in, 1 - (top_in + bottom_in) / height_in])

    for storage_format, values in series.items():
        style = FORMAT_STYLE[storage_format]
        medians = [v[0] for v in values]
        ax.plot(xs, medians, color=style["color"], linewidth=1.5, solid_joinstyle="round",
                solid_capstyle="round", zorder=3)
        ax.plot(xs, medians, linestyle="none", marker=style["marker"], markersize=6.5,
                markerfacecolor=style["color"], markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        for x, (_, low, high) in zip(xs, values):
            if high > low:
                ax.plot([x, x], [low, high], color=style["color"], linewidth=1, zorder=2)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
        low, high = ax.get_ylim()
        # 1-2-5 ticks over a few decades; powers of ten only when it spans more.
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5) if math.log10(high / low) <= 3 else (1,)))
        ax.yaxis.set_minor_locator(NullLocator())
    if variable == "nodes":
        ax.set_xticks(xs)  # few, log-spaced points: label each one
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    else:
        ax.set_xlim(left=0)  # low densities sit close together, so use round ticks instead
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: format_x(v, variable)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: y_fmt(v)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    if not log_y:
        ax.set_ylim(bottom=0)
    ax.tick_params(which="both", length=0, labelsize=9, labelcolor=MUTED, pad=6)
    ax.yaxis.grid(True, which="major", color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.75)
    ax.set_xlabel(x_label, color=MUTED, fontsize=9, labelpad=6)

    # Direct labels at the line ends, in ink (the marker beside each carries the color).
    fig.canvas.draw()
    to_pixels = ax.transData.transform
    ends = {f: to_pixels((xs[-1], values[-1][0]))[1] for f, values in series.items()}
    labels_y = _spread(list(ends.values()), min_gap=13)
    to_data = ax.transData.inverted().transform
    for (storage_format, values), y_px in zip(series.items(), labels_y):
        y_data = to_data((to_pixels((xs[-1], 0))[0], y_px))[1]
        ax.annotate(f"{storage_format}  {y_fmt(values[-1][0])}", xy=(xs[-1], y_data), xytext=(10, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=9, color=INK_SECONDARY,
                    annotation_clip=False)

    left = left_in / width_in
    fig.text(left, 1 - 0.32 / height_in, title, fontsize=12.5, fontweight="bold", color=INK, va="baseline")
    fig.text(left, 1 - 0.55 / height_in, subtitle, fontsize=9, color=INK_SECONDARY, va="baseline")
    handles = [Line2D([], [], color=FORMAT_STYLE[f]["color"], linewidth=1.5, marker=FORMAT_STYLE[f]["marker"],
                      markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.5) for f in series]
    legend = fig.legend(handles, list(series), loc="upper left", bbox_to_anchor=(left - 0.01, 1 - 0.68 / height_in),
                        ncol=len(series), frameon=False, fontsize=9.5, handlelength=2.2, columnspacing=1.6)
    for text in legend.get_texts():
        text.set_color(INK)

    fig.savefig(path, dpi=200)
    plt.close(fig)


def _latency_fmt(v):
    return f"{v:.3g} ms"


def sweep_charts(plt, out_dir, sweep, xs, points, graphs, variable, log_x):
    """Write one chart per metric. Returns the written paths keyed by metric."""
    first = graphs[xs[0]]
    device = first["device"]
    fixed = (f"{first['nodes']:,} nodes" if variable != "nodes"
             else f"average degree {first['avg_degree']:g}")
    seeds = len(first["seeds"])
    x_label = {"nodes": "Number of nodes (log scale)", "density": "Density (% of possible edges)",
               "avg_degree": "Average degree"}[variable]
    timing = ("FP32, TF32 off" if device.startswith("cuda")
              else f"DEBUG ONLY: {device} timing, not a GPU result")
    written = {}
    for key, _, suffix, title in METRICS:
        if any(points[x][f].get(key) is None for x in xs for f in points[x]):
            continue
        series = {f: [points[x][f][key] for x in xs] for f in _ordered_formats(points[xs[0]])}
        if key == "latency":
            subtitle = f"{fixed} · median over {seeds} graph seeds, whisker = seed range · {timing}"
            y_fmt, log_y = _latency_fmt, variable == "nodes"
        elif key == "adjacency_bytes":
            subtitle = f"{fixed} · exact bytes of the stored tensors (FP32 values, int64 indices)"
            y_fmt, log_y = _compact_bytes, variable == "nodes"
        else:
            subtitle = f"{fixed} · torch.cuda.max_memory_allocated, median over {seeds} graph seeds"
            y_fmt, log_y = _compact_bytes, variable == "nodes"
        path = Path(out_dir) / f"{sweep}_{suffix}.png"
        name = f"{sweep.split('_')[0]} sweep · {device}"
        line_chart(plt, path, xs, series, f"{title} · {name}", subtitle, x_label, variable, log_x, log_y, y_fmt)
        written[key] = path
    return written


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def summarize_synthetic(results_dir, out_dir):
    """Write summary.md and charts to out_dir. Returns (text report, list of written files)."""
    sweeps = load_sweeps(results_dir)
    if not sweeps:
        raise FileNotFoundError(f"no synthetic benchmark JSON files in sub-folders of {results_dir}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    written, text = [], []
    try:
        source = Path(results_dir).resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        source = results_dir
    md = ["# Synthetic sweep results", "",
          f"Generated by `scripts/summarize_synthetic_results.py` from `{source}`. Uniform random G(n, m) graphs, "
          "untrained two-layer GCN with fixed random weights, logits checked against COO on CPU. "
          "Latency and peak memory: median over graph seeds of the per-seed median, range = spread over seeds. "
          "Crossover estimates interpolate between neighboring sweep points.", ""]
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

    for sweep, runs in sweeps.items():
        variable = sweep_variable(runs)
        graphs, points = aggregate(runs, variable)
        xs = list(points)
        log_x = variable == "nodes"
        devices = {g["device"] for g in graphs.values()}

        text += ["=" * 72, f"SWEEP: {sweep}", "=" * 72, ""]
        md += [f"## {sweep}", ""]
        device_note = f"Device: {', '.join(sorted(devices))}. {len(runs)} runs."
        if any(g["debug_only"] for g in graphs.values()):
            device_note += " **Latency is DEBUG ONLY (not a GPU result).**"
        text += [device_note, ""]
        md += [device_note, ""]

        charts = sweep_charts(plt, out_dir, sweep, xs, points, graphs, variable, log_x) if plt and len(xs) > 1 else {}
        written.extend(charts.values())

        section("Graphs", *graph_table(graphs))
        section("Latency (ms)", *metric_table(points, variable, "latency", lambda v: f"{v:.3f}", "Fastest"),
                image=charts.get("latency"))
        section("Adjacency memory", *metric_table(points, variable, "adjacency_bytes", human_bytes, "Smallest"),
                image=charts.get("adjacency_bytes"))
        if all(points[x][f]["peak"] is not None for x in xs for f in points[x]):
            section("Peak GPU memory", *metric_table(points, variable, "peak", human_bytes, "Smallest"),
                    image=charts.get("peak"))
        verdict = (f"All runs equivalent to the CPU COO reference: "
                   f"{_yes_no(all(p[f]['equivalent'] for p in points.values() for f in p))} "
                   "(allclose atol=1e-5, rtol=1e-4, 100% argmax agreement)")
        section("Equivalence", *equivalence_table(points, variable), note=verdict)
        if len(xs) > 1 and "dense" in points[xs[0]]:
            section("Crossovers with dense", *crossover_table(xs, points, variable, log_x),
                    note="Estimates interpolate between the two neighboring sweep points"
                         + (" in log(nodes)" if log_x else "")
                         + f". Differences under {TIE_TOLERANCE:.0%} of dense count as ties.")

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(md))
    written.insert(0, summary_path)
    return "\n".join(text), written

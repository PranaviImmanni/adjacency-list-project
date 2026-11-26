"""Report for the GPU format comparison (results/mlp_gpu/<run>) with energy.

Per model, format and batch size: test accuracy, stored weight bytes, resident and peak CUDA
memory, latency, average power, energy per batch and per example (gross and net of idle).
Per model seed the repetitions are reduced by their median; values are then the mean over
seeds with the min - max range. Random pruning: accuracy over every pruning seed, timing over
the benchmarked one (as in the original sweep).

Also checks: equivalence of every format with its reference, agreement with Pranavi's CPU
results, the original sweep's accuracy and latency for the four standard formats.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

from src.gcn.config import PROJECT_ROOT
from src.gcn.reporting import GRID, INK, INK_SECONDARY, MUTED, SURFACE, BASELINE, _pyplot, markdown_table

from .formats import FORMAT_LABELS as PRUNED_FORMAT_LABELS
from .low_rank import FORMAT_LABELS as FACTOR_FORMAT_LABELS
from .low_rank import rule_label
from .reporting import FORMAT_STYLE, _compact_bytes, _range, _stats, _write_csv, line_panels

FORMAT_LABELS = {**PRUNED_FORMAT_LABELS, **FACTOR_FORMAT_LABELS}
PRUNED_ORDER = ("dense", "dense_mask", "coo", "csr", "adjacency_list")
FACTOR_ORDER = ("factor_adjacency", "factor_dense")
METHOD_LABELS = {"none": "Original", "magnitude": "Global magnitude", "topk_per_neuron": "Top-K per neuron",
                 "random": "Global random"}
# Fixed categorical order (validated): the four standard formats keep the original report's colors,
# the adjacency list takes slot 5.
STYLES = {**FORMAT_STYLE, "adjacency_list": {"color": "#e87ba4", "marker": "v", "label": "adjacency list"}}
FACTOR_STYLES = {"factor_adjacency": {"color": "#2a78d6", "marker": "o", "label": "factor adjacency list"},
                 "factor_dense": {"color": "#eb6834", "marker": "s", "label": "dense factors"}}
ENERGY_FIELDS = ("energy_per_example_j", "energy_per_batch_j", "average_power_w", "net_energy_per_example_j",
                 "net_energy_per_batch_j", "net_average_power_w", "idle_power_w")


def _load(directory):
    ok, failed = [], []
    for path in sorted(Path(directory).glob("*.json")):
        record = json.loads(path.read_text())
        (ok if record.get("status") == "ok" else failed).append(record)
    return ok, failed


def row_key(record):
    """('original'|'pruned', method, sparsity) or ('low_rank', rank, rule label)."""
    if record["kind"] == "low_rank":
        return ("low_rank", record["rank"], record["rule_label"])
    return (record["kind"], record["method"], record["requested_sparsity"])


def row_label(key):
    if key[0] == "low_rank":
        return f"rank {key[1]}, {key[2].replace('abs_std_k', 'abs-std k = ').replace('p', '.')}"
    if key[0] == "original":
        return "Original model"
    return f"{METHOD_LABELS[key[1]]} {100 * key[2]:g}%"


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def aggregate_benchmarks(benches):
    """{(row key, format, batch): {metric: stats over seeds}} with per-seed medians over repetitions."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in benches:
        groups[(row_key(r), r["format"], r["batch_size"])][r["model_seed"]].append(r)
    out = {}
    for key, by_seed in groups.items():
        per_seed = defaultdict(list)
        for runs in by_seed.values():
            per_seed["latency"].append(_median([r["latency_ms"]["median"] for r in runs]))
            per_seed["resident"].append(_median([r.get("cuda_allocated_before_inference_bytes") for r in runs]))
            per_seed["peak"].append(_median([r.get("cuda_peak_allocated_bytes") for r in runs]))
            for field in ENERGY_FIELDS:
                per_seed[field].append(_median([r["energy"].get(field) for r in runs if r["energy"].get("available")]))
        reps = [len(runs) for runs in by_seed.values()]
        checks = [r["format_check"] for runs in by_seed.values() for r in runs if r.get("format_check")]
        out[key] = {**{metric: _stats(values) for metric, values in per_seed.items()},
                    "seeds": sorted(by_seed), "runs": sum(reps),
                    "stored_bytes": max(r["weight_storage_bytes"] for runs in by_seed.values() for r in runs),
                    "checks": len(checks), "checks_within": sum(c["equivalent"] for c in checks),
                    "equivalent": all(c["equivalent"] for c in checks) if checks else None,
                    "same_predictions": all(c.get("same_predictions", True) for c in checks) if checks else None,
                    "max_epsilon": max((c["epsilon_logit"] for c in checks), default=None)}
    return out


def aggregate_evaluations(evals):
    """{(row key, format): {test accuracy stats, stored bytes, worst check, units}} plus per-row extras."""
    groups = defaultdict(list)
    for r in evals:
        groups[row_key(r)].append(r)
    out = {}
    for key, records in groups.items():
        formats = records[0]["formats"]
        for storage_format in formats:
            accs = [(r.get("format_accuracy") or {storage_format: r["accuracy"]}).get(storage_format, r["accuracy"])
                    for r in records]
            checks = [r["format_checks"][storage_format] for r in records if storage_format in r["format_checks"]]
            out[(key, storage_format)] = {
                "test": _stats([a["test"] for a in accs]), "val": _stats([a["val"] for a in accs]),
                "stored_bytes": max(r["storage"][storage_format]["weight_bytes"] for r in records),
                "units": len(records), "checks": len(checks),
                "equivalent": all(c["equivalent"] for c in checks) if checks else None,
                "strict": sum(c["equivalent"] for c in checks),
                "same_predictions": all(c.get("same_predictions", c["prediction_agreement_all"] == 1.0)
                                        for c in checks) if checks else None,
                "max_epsilon": max((c["epsilon_logit"] for c in checks), default=None),
                "float64_error": max((c.get("max_abs_error_vs_float64") or 0 for c in checks), default=None),
                "reference_float64_error": max((c.get("reference_max_abs_error_vs_float64") or 0 for c in checks),
                                               default=None),
                "max_logit": max((c.get("max_abs_exact_logit") or 0 for c in checks), default=None),
                "bitwise": all(c.get("bitwise_equal") for c in checks) if checks else None,
            }
        effects = [r["effect_vs_original"] for r in records if r.get("effect_vs_original")]
        out[(key, None)] = {
            "loss_pp": _stats([e["test_accuracy_loss_pp"] for e in effects]) if effects else None,
            "disagreement": _stats([e["prediction_disagreement_test"] for e in effects]) if effects else None,
            "edges": _stats([r.get("edges") for r in records]),
            "factor_sparsity": _stats([(r.get("low_rank") or {}).get("factor_sparsity") for r in records]),
            "reconstruction_mse": _stats([(r.get("low_rank") or {}).get("mean_weight_reconstruction_mse")
                                          for r in records]),
            "actual_sparsity": _stats([r.get("actual_sparsity") for r in records]),
        }
    return out


def baseline_latency(baseline_dir):
    """{(method, sparsity, pruning seed, format, batch, seed): median over repetitions} of the original sweep."""
    benches, _ = _load(Path(baseline_dir) / "benchmark") if Path(baseline_dir, "benchmark").exists() else ([], [])
    groups = defaultdict(list)
    for r in benches:
        groups[(r["method"], r["requested_sparsity"], r["pruning_seed"], r["format"], r["batch_size"],
                r["model_seed"])].append(r["latency_ms"]["median"])
    return {key: statistics.median(values) for key, values in groups.items()}, {r["device"] for r in benches}


# ---------------------------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------------------------

def _pct(stats):
    return _range(stats, lambda v: f"{100 * v:.2f}%")


def _ms(stats):
    return _range(stats, lambda v: f"{v:.4f}")


def _uj(stats):
    return _range(stats, lambda v: f"{1e6 * v:.2f}")


def _mj(stats):
    return _range(stats, lambda v: f"{1e3 * v:.3f}")


def _w(stats):
    return _range(stats, lambda v: f"{v:.1f}")


def _mem(stats):
    return _range(stats, _compact_bytes)


def _get(agg, key, metric):
    return (agg.get(key) or {}).get(metric)


# ---------------------------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------------------------

def _series(bench_agg, method, formats, batch, metric, scale=1.0):
    series = {}
    for storage_format in formats:
        points = []
        for (key, fmt, b), point in bench_agg.items():
            if key[0] == "pruned" and key[1] == method and fmt == storage_format and b == batch and point.get(metric):
                s = point[metric]
                points.append((key[2], scale * s["mean"], scale * s["min"], scale * s["max"]))
        points.sort()
        if points:
            series[storage_format] = points
    return series


def low_rank_figure(plt, path, eval_agg, curve, original, ranks, rules, subtitle):
    """Accuracy vs stored bytes, one panel per factor rule; rank labels on the factor-adjacency points."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator

    n = len(rules)
    panel_w, panel_h, gap, left_in, right_in, bottom_in = 3.1, 2.8, 0.45, 0.85, 0.3, 0.75
    width = left_in + n * panel_w + (n - 1) * gap + right_in
    top_in = 1.45
    height = top_in + panel_h + bottom_in + 0.45
    fig = plt.figure(figsize=(width, height), dpi=100)
    xs_all = [p[0] for p in curve] + [original[0]] + [
        v["stored_bytes"] for (key, fmt), v in eval_agg.items() if fmt and key[0] == "low_rank"]
    x_lo, x_hi = min(xs_all) / 1.6, max(xs_all) * 1.6
    for c, rule in enumerate(rules):
        ax = fig.add_axes([(left_in + c * (panel_w + gap)) / width, (bottom_in + 0.45) / height,
                           panel_w / width, panel_h / height])
        ax.set_xscale("log")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 3)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _compact_bytes(v)))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _, first=(c == 0): f"{100 * v:.0f}%" if first else ""))
        ax.tick_params(which="both", length=0, labelsize=8.5, labelcolor=MUTED, pad=5)
        ax.yaxis.grid(True, color=GRID, linewidth=0.75)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
            ax.spines[side].set_linewidth(0.75)
        ax.plot([p[0] for p in curve], [p[1] for p in curve], color=MUTED, linewidth=1, marker="o", markersize=3.5,
                zorder=1)
        ax.plot([original[0]], [original[1]], linestyle="none", marker="X", markersize=8, markerfacecolor=MUTED,
                markeredgecolor=SURFACE, zorder=2)
        for storage_format in FACTOR_ORDER:
            pts = [(eval_agg[(("low_rank", r, rule), storage_format)], r) for r in ranks
                   if (("low_rank", r, rule), storage_format) in eval_agg]
            if not pts:
                continue
            style = FACTOR_STYLES[storage_format]
            xs, ys = [p["stored_bytes"] for p, _ in pts], [p["test"]["mean"] for p, _ in pts]
            ax.plot(xs, ys, color=style["color"], linewidth=1.5, zorder=3)
            ax.plot(xs, ys, linestyle="none", marker=style["marker"], markersize=6, markerfacecolor=style["color"],
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
            for (p, _), x in zip(pts, xs):
                if p["test"]["max"] > p["test"]["min"]:
                    ax.plot([x, x], [p["test"]["min"], p["test"]["max"]], color=style["color"], linewidth=1, zorder=2)
            if storage_format == "factor_adjacency":
                for (p, r), x, y in zip(pts, xs, ys):
                    ax.annotate(f"r{r}", (x, y), xytext=(5, -11), textcoords="offset points", fontsize=7.5,
                                color=INK_SECONDARY)
        ax.set_title(rule.replace("abs_std_k", "abs-std, k = ").replace("p", ".").replace("all", "all entries"),
                     loc="left", fontsize=10, color=INK, pad=8)
        ax.set_xlabel("stored weight bytes (log)", color=MUTED, fontsize=8, labelpad=5)
    fig.text(left_in / width, 1 - 0.32 / height, "Low-rank factor networks: test accuracy vs storage", fontsize=12.5,
             fontweight="bold", color=INK, va="baseline")
    fig.text(left_in / width, 1 - 0.55 / height, subtitle, fontsize=9, color=INK_SECONDARY, va="baseline")
    handles = [Line2D([], [], color=s["color"], marker=s["marker"], linewidth=1.5, markersize=6,
                      markeredgecolor=SURFACE) for s in FACTOR_STYLES.values()]
    handles += [Line2D([], [], color=MUTED, linewidth=1, marker="o", markersize=3.5),
                Line2D([], [], color=MUTED, linestyle="none", marker="X", markersize=8, markeredgecolor=SURFACE)]
    labels = [s["label"] for s in FACTOR_STYLES.values()] + ["global magnitude pruning, CSR", "original model"]
    legend = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(left_in / width - 0.01, 1 - 0.75 / height),
                        ncol=4, frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------

def summarize_comparison(results_dir, baseline_dir, out_dir, config):
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evals, failed_evals = _load(results_dir / "evaluate")
    benches, failed_benches = _load(results_dir / "benchmark")
    if not evals and not benches:
        raise SystemExit(f"no results in {results_dir}")
    eval_agg, bench_agg = aggregate_evaluations(evals), aggregate_benchmarks(benches)
    batches = config["batch_sizes"]
    debug = any(r["latency_is_debug_only"] for r in benches)
    devices = sorted({f"{r['device']} ({r.get('device_name')})" for r in benches + evals})
    seeds = sorted({r["model_seed"] for r in evals + benches})
    written, lines = [], []

    def shown(path):
        try:
            return str(Path(path).resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    def table(title, headers, rows, note=None, csv_name=None, csv_rows=None):
        lines.extend([f"## {title}", "", markdown_table(headers, rows), ""])
        if note:
            lines.extend([note, ""])
        if csv_name:
            _write_csv(out_dir / csv_name, headers, csv_rows if csv_rows is not None else rows)
            written.append(out_dir / csv_name)

    lines += ["# MLP GPU format comparison: dense, dense+mask, COO, CSR, adjacency list, low-rank factors", "",
              f"Results from `{shown(results_dir)}`. Devices: {', '.join(devices)}. Model seeds: "
              f"{', '.join(map(str, seeds))}. {len(evals)} evaluations, {len(benches)} benchmark runs "
              f"({len(failed_evals) + len(failed_benches)} failed). Values are means over model seeds; ranges in "
              "parentheses are min - max. Per seed, repetitions are reduced by their median.", ""]
    if debug:
        lines += ["**CPU debug run: latency and memory are not GPU results and energy is not measured.**", ""]

    # -- energy method ---------------------------------------------------------------------------
    energies = [r["energy"] for r in benches]
    measured = [e for e in energies if e.get("available")]
    lines += ["## Energy measurement", ""]
    if measured:
        methods = sorted({e["method"] for e in measured})
        intervals = [e["idle"].get("counter_update_interval_ms") for e in measured
                     if e["idle"].get("counter_update_interval_ms")]
        idle = _stats([e["idle_power_w"] for e in measured])
        idle_reading = _stats([e.get("idle_power_reading_w") for e in measured])
        active_reading = _stats([e.get("active_power_reading_w") for e in measured])
        active = _stats([e["average_power_w"] for e in measured])
        pids = sorted({p for e in measured for p in e.get("other_processes", [])})
        busy = sum(e["other_processes_present"] and ((e.get("state_before") or {}).get("gpu_utilization_pct") or 0) > 0
                   for e in measured)
        window = statistics.median([e["duration_s"] for e in measured])
        others = sum(e["other_processes_present"] for e in measured)
        lines += [f"- Measured in {len(measured)} of {len(energies)} runs with NVML ({', '.join(methods)}), "
                  f"device matched by {', '.join(sorted({e['device']['matched_by'] for e in measured}))}: "
                  f"{', '.join(sorted({e['device']['name'] for e in measured}))}.",
                  f"- Active window: synchronized inference of the benchmark batch for at least "
                  f"{config['energy_seconds']:g} s (median {window:.2f} s); "
                  f"idle window {config['energy_idle_seconds']:g} s before it, with the model loaded.",
                  f"- Idle power from the counter: {_w(idle)} W; NVML's power reading at the end of the idle window: "
                  f"{_w(idle_reading)} W. Active: counter {_w(active)} W, power reading at the end of the window "
                  f"{_w(active_reading)} W."
                  + (f" Energy counter update interval: median {statistics.median(intervals):.1f} ms."
                     if intervals else ""),
                  f"- Runs with another process on the GPU: {others} (PIDs {', '.join(map(str, pids)) or '-'}). NVML "
                  "measures the whole card, so work by another process would raise the idle power measured in every "
                  f"run (range above). GPU utilization above 0 right before the measurement: {busy} runs (this "
                  "sample also covers our own latency timing, which ends just before).",
                  "- Gross = what the card drew during the window; net = gross minus idle power x window length.", ""]
    else:
        reasons = sorted({e.get("reason", "unknown") for e in energies})
        lines += [f"Not measured in this run: {'; '.join(reasons) or 'no benchmark runs'}.", ""]

    # -- pruned formats ------------------------------------------------------------------------------
    original_key = ("original", "none", 0.0)
    pruned_rows = sorted({key for key, fmt in eval_agg if fmt and key[0] == "pruned"},
                         key=lambda k: (list(METHOD_LABELS).index(k[1]), k[2]))
    methods = [m for m in METHOD_LABELS if m != "none" and any(k[1] == m for k in pruned_rows)]

    def format_row(key, storage_format):
        e = eval_agg.get((key, storage_format), {})
        label = "FP32 dense" if key == original_key else FORMAT_LABELS[storage_format]
        cells = [row_label(key), label, _pct(e.get("test")),
                 f"{e['stored_bytes']:,}" if e else "n/a"]
        for batch in batches:
            cells.append(_ms(_get(bench_agg, (key, storage_format, batch), "latency")))
        for batch in batches:
            cells.append(_uj(_get(bench_agg, (key, storage_format, batch), "energy_per_example_j")))
        cells += [_w(_get(bench_agg, (key, storage_format, batches[-1]), "average_power_w")),
                  _mem(_get(bench_agg, (key, storage_format, batches[0]), "resident")),
                  _mem(_get(bench_agg, (key, storage_format, batches[0]), "peak")),
                  _mem(_get(bench_agg, (key, storage_format, batches[-1]), "peak"))]
        return cells

    headers = (["Model", "Format", "Test acc", "Stored bytes"] + [f"Latency b{b} (ms)" for b in batches]
               + [f"Energy/example b{b} (uJ)" for b in batches]
               + [f"Avg power b{batches[-1]} (W)", f"Resident b{batches[0]}", f"Peak b{batches[0]}",
                  f"Peak b{batches[-1]}"])
    main_rows = [format_row(original_key, "dense")] if (original_key, "dense") in eval_agg else []
    main_rows += [format_row(key, storage_format) for key in pruned_rows if key[1] == "magnitude"
                  for storage_format in PRUNED_ORDER if (key, storage_format) in eval_agg]
    table("Pruned weights: all five formats (global magnitude)", headers, main_rows,
          "Energy per example is gross (whole card during synchronized inference) divided by the batch size; "
          "net values (minus idle power) and every method are in `pruned.csv`. Random pruning is timed with "
          f"pruning seed {', '.join(map(str, config['benchmark_random_seeds']))}.")
    csv_headers = ["row_kind", "method", "sparsity", "format", "test_accuracy", "test_accuracy_min",
                   "test_accuracy_max", "stored_bytes"] + [f"{m}_b{b}" for b in batches for m in (
                       "latency_ms", "energy_per_example_j", "energy_per_batch_j", "average_power_w",
                       "net_energy_per_example_j", "net_energy_per_batch_j", "net_average_power_w", "idle_power_w",
                       "resident_bytes", "peak_bytes")]
    csv_rows = []
    for key in ([original_key] if (original_key, "dense") in eval_agg else []) + pruned_rows:
        for storage_format in PRUNED_ORDER:
            e = eval_agg.get((key, storage_format))
            if not e:
                continue
            row = [key[0], key[1], key[2], storage_format, e["test"]["mean"], e["test"]["min"], e["test"]["max"],
                   e["stored_bytes"]]
            for batch in batches:
                b = bench_agg.get((key, storage_format, batch), {})
                for metric in ("latency", "energy_per_example_j", "energy_per_batch_j", "average_power_w",
                               "net_energy_per_example_j", "net_energy_per_batch_j", "net_average_power_w",
                               "idle_power_w", "resident", "peak"):
                    row.append((b.get(metric) or {}).get("mean"))
            csv_rows.append(row)
    _write_csv(out_dir / "pruned.csv", csv_headers, csv_rows)
    written.append(out_dir / "pruned.csv")

    acc_rows = []
    for method in methods:
        for key in (k for k in pruned_rows if k[1] == method):
            e = eval_agg[(key, "dense")]
            extra = eval_agg[(key, None)]
            acc_rows.append([METHOD_LABELS[method], f"{100 * key[2]:g}%", _pct(e["test"]),
                             _range(extra["loss_pp"], lambda v: f"{v:+.2f}"), e["units"],
                             _range(extra["edges"], lambda v: f"{v:,.0f}")]
                            + [f"{eval_agg[(key, f)]['stored_bytes']:,}" if (key, f) in eval_agg else "n/a"
                               for f in PRUNED_ORDER])
    table("Accuracy and storage, every pruning method",
          ["Method", "Sparsity", "Test acc", "Loss vs original (pp)", "Units", "Edges"]
          + [f"{FORMAT_LABELS[f]} bytes" for f in PRUNED_ORDER], acc_rows,
          "Every format holds the same retained weights, so they share one accuracy (checked below). Adjacency "
          "list bytes: 4 B value + 8 B neighbor index per edge + 8 B per offset, the same as CSR.")

    # -- low rank ------------------------------------------------------------------------------------
    rule_order = [rule_label(r["rule"], r.get("k")) for r in config["low_rank_rules"]]
    low_rows = sorted({key for key, fmt in eval_agg if fmt and key[0] == "low_rank"},
                      key=lambda k: (rule_order.index(k[2]) if k[2] in rule_order else len(rule_order), k[2], k[1]))
    dense_bytes = eval_agg.get((original_key, "dense"), {}).get("stored_bytes")
    if low_rows:
        rows, csv_rows = [], []
        for key in low_rows:
            lists, dense = eval_agg.get((key, "factor_adjacency")), eval_agg.get((key, "factor_dense"))
            extra = eval_agg[(key, None)]
            share = (lambda v: f" ({100 * v / dense_bytes:.1f}%)") if dense_bytes else (lambda v: "")
            rows.append([row_label(key), _pct(lists["test"] if lists else dense["test"]),
                         _range(extra["loss_pp"], lambda v: f"{-v:+.2f}"),
                         _range(extra["disagreement"], lambda v: f"{100 * (1 - v):.2f}%"),
                         _range(extra["edges"], lambda v: f"{v:,.0f}"),
                         _range(extra["factor_sparsity"], lambda v: f"{100 * v:.1f}%"),
                         (f"{lists['stored_bytes']:,}" + share(lists["stored_bytes"])) if lists else "n/a",
                         (f"{dense['stored_bytes']:,}" + share(dense["stored_bytes"])) if dense else "n/a"])
        table("Low-rank factor networks: accuracy and storage",
              ["Model", "Test acc", "Change vs original (pp)", "Agreement with original (test)", "Factor edges",
               "Factor sparsity", "Factor adjacency bytes", "Dense factor bytes"], rows,
              "W ~= B A from a truncated SVD of each trained weight matrix (rank capped at 10 for fc3), factor "
              "entries kept by the rule, no fine-tuning (Pranavi's definition). Percentages are of the original "
              "dense weight bytes. Dense factors store every factor entry (unselected ones as 0).")
        rows = []
        for key in low_rows:
            for storage_format in FACTOR_ORDER:
                if (key, storage_format) not in eval_agg:
                    continue
                rows.append([row_label(key), FACTOR_FORMAT_LABELS[storage_format]]
                            + [_ms(_get(bench_agg, (key, storage_format, b), "latency")) for b in batches]
                            + [_uj(_get(bench_agg, (key, storage_format, b), "energy_per_example_j")) for b in batches]
                            + [_mem(_get(bench_agg, (key, storage_format, batches[0]), "peak")),
                               _mem(_get(bench_agg, (key, storage_format, batches[-1]), "peak"))])
                e = eval_agg[(key, storage_format)]
                row = [key[1], key[2], storage_format, e["test"]["mean"], e["stored_bytes"],
                       (eval_agg[(key, None)]["edges"] or {}).get("mean")]
                for batch in batches:
                    b = bench_agg.get((key, storage_format, batch), {})
                    for metric in ("latency", "energy_per_example_j", "energy_per_batch_j", "average_power_w",
                                   "net_energy_per_example_j", "resident", "peak"):
                        row.append((b.get(metric) or {}).get("mean"))
                csv_rows.append(row)
        table("Low-rank factor networks: latency, energy, memory",
              ["Model", "Format"] + [f"Latency b{b} (ms)" for b in batches]
              + [f"Energy/example b{b} (uJ)" for b in batches] + [f"Peak b{batches[0]}", f"Peak b{batches[-1]}"], rows)
        _write_csv(out_dir / "low_rank.csv",
                   ["rank", "rule", "format", "test_accuracy", "stored_bytes", "factor_edges"]
                   + [f"{m}_b{b}" for b in batches for m in ("latency_ms", "energy_per_example_j",
                                                             "energy_per_batch_j", "average_power_w",
                                                             "net_energy_per_example_j", "resident_bytes",
                                                             "peak_bytes")], csv_rows)
        written.append(out_dir / "low_rank.csv")

    # -- checks ------------------------------------------------------------------------------------
    rows = []
    for storage_format in PRUNED_ORDER[1:] + FACTOR_ORDER[:1]:
        evaluated = [v for (key, fmt), v in eval_agg.items() if fmt == storage_format and v["checks"]]
        benched = [v for (key, fmt, b), v in bench_agg.items() if fmt == storage_format and v["checks"]]
        if not evaluated and not benched:
            continue
        n = sum(v["checks"] for v in evaluated)
        rows.append([FORMAT_LABELS[storage_format], "dense factors" if storage_format == "factor_adjacency"
                     else "pruned dense", n,
                     "yes" if all(v["same_predictions"] for v in evaluated) else "NO",
                     f"{sum(v['strict'] for v in evaluated)} / {n}",
                     f"{max((v['max_epsilon'] for v in evaluated), default=0):.1e}",
                     f"{max((v['float64_error'] or 0 for v in evaluated), default=0):.1e}",
                     f"{max((v['reference_float64_error'] or 0 for v in evaluated), default=0):.1e}",
                     f"{max((v['max_logit'] or 0 for v in evaluated), default=0):.0f}",
                     f"{sum(v['checks'] for v in benched)}",
                     "yes" if all(v["same_predictions"] for v in benched) else "NO",
                     f"{sum(v['checks_within'] for v in benched)} / {sum(v['checks'] for v in benched)}"])
    table("Equivalence with the reference implementation",
          ["Format", "Reference", "Evaluations", "Same predictions", "Within tolerance", "Max epsilon",
           "Max error vs float64", "Reference's error vs float64", "Max |logit|", "Benchmark batches",
           "Batches: same predictions", "Batches within tolerance"], rows,
          "Evaluations cover 5,000 validation + 10,000 test images per model. Same predictions: identical "
          "predicted class for every image and identical accuracy. Within tolerance: the project's criterion, "
          "allclose(atol=1e-5, rtol=1e-4) against the FP32 reference as well. Float64 columns: largest distance "
          "of the format and of the FP32 reference from an exact float64 computation of the same model, which "
          "separates rounding (different summation order, large logits) from errors.", "equivalence.csv")

    rows = []
    for r in sorted(evals, key=lambda r: r["tag"]):
        check = r.get("pranavi_cpu") or {}
        if not check.get("found"):
            continue
        rows.append([r["tag"], FORMAT_LABELS["factor_adjacency" if r["kind"] == "low_rank" else "adjacency_list"],
                     f"{100 * check['pranavi_test_accuracy']:.2f}%", f"{100 * check['test_accuracy']:.2f}%",
                     f"{check['difference_images']:+d}", "yes" if check["edges_equal"] else "NO",
                     {True: "yes", False: "NO"}.get(check.get("stored_bytes_equal"), "-"), check["source"]])
    if rows:
        same = sum(row[4] == "+0" for row in rows)
        table("Agreement with Pranavi's CPU results", ["Model", "Format", "Her CPU test acc", "GPU test acc",
                                                       "Difference (images)", "Same edges", "Same bytes", "Source"],
              rows, f"{same} of {len(rows)} models give exactly the same test accuracy. Her runs compute in float64 "
              "on the FP32 values in NumPy; these run in FP32 on the device, so a few borderline images can flip.",
              "pranavi_crosscheck.csv")

    sweep = [r["original_sweep"] for r in evals if r.get("original_sweep", {}).get("found")]
    stability_rows = []
    old, old_devices = baseline_latency(baseline_dir)
    same_device = {r["device"] for r in benches} <= old_devices if old else False
    if old and same_device:
        ratios = defaultdict(list)
        for r in benches:
            if r["kind"] == "low_rank" or r["format"] not in PRUNED_ORDER[:4]:
                continue
            key = (r["method"], r["requested_sparsity"], r["pruning_seed"], r["format"], r["batch_size"],
                   r["model_seed"])
            if key in old:
                ratios[(r["format"], r["batch_size"])].append((key, r["latency_ms"]["median"]))
        for (storage_format, batch), values in sorted(ratios.items(), key=lambda kv: (PRUNED_ORDER.index(kv[0][0]),
                                                                                      kv[0][1])):
            per_key = defaultdict(list)
            for key, value in values:
                per_key[key].append(value)
            r_values = [statistics.median(v) / old[k] for k, v in per_key.items()]
            stability_rows.append([FORMAT_LABELS[storage_format], batch, len(r_values),
                                   f"{statistics.median(r_values):.3f}", f"{min(r_values):.3f} - {max(r_values):.3f}",
                                   f"{100 * sum(abs(x - 1) <= 0.05 for x in r_values) / len(r_values):.0f}%"])
    lines += ["## Consistency with the original sweep", ""]
    if sweep:
        sources = sorted({str(Path(s["path"]).parent) for s in sweep})
        lines += [f"- Accuracy: {sum(s['test_accuracy_equal'] for s in sweep)} of {len(sweep)} models give the same "
                  f"test accuracy as the original sweep (`{'`, `'.join(sources)}`, same checkpoints); "
                  f"{sum(s['logits_sha256_equal'] for s in sweep)} give bit-identical logits.", ""]
    if stability_rows:
        lines.append(markdown_table(["Format", "Batch", "Models", "Latency now / before (median)", "Range",
                                     "Within 5%"], stability_rows))
        lines.append("")
        _write_csv(out_dir / "stability.csv", ["format", "batch", "models", "median_ratio", "range", "within_5pct"],
                   stability_rows)
        written.append(out_dir / "stability.csv")
    elif not sweep:
        lines += ["No original-sweep results on the same device to compare with.", ""]

    # -- runs.csv ------------------------------------------------------------------------------------
    run_headers = ["tag", "kind", "model_seed", "method", "pruning_seed", "requested_sparsity", "rank", "rule",
                   "format", "batch_size", "repetition", "device", "device_name", "weight_storage_bytes", "edges",
                   "latency_median_ms", "cuda_allocated_before_inference_bytes", "cuda_peak_allocated_bytes",
                   "energy_method", "energy_batches", "energy_duration_s", "energy_j", "average_power_w",
                   "energy_per_batch_j", "energy_per_example_j", "idle_power_w", "net_energy_per_batch_j",
                   "net_energy_per_example_j", "other_processes_present", "format_check_equivalent",
                   "latency_is_debug_only", "checkpoint_sha256"]
    run_rows = []
    for r in sorted(benches, key=lambda r: (r["repetition"], r["model_seed"], r["tag"], r["format"], r["batch_size"])):
        e = r["energy"] if r["energy"].get("available") else {}
        run_rows.append([r["tag"], r["kind"], r["model_seed"], r["method"], r["pruning_seed"], r["requested_sparsity"],
                         r["rank"], r["rule_label"], r["format"], r["batch_size"], r["repetition"], r["device"],
                         r.get("device_name"), r["weight_storage_bytes"], r.get("edges"), r["latency_ms"]["median"],
                         r.get("cuda_allocated_before_inference_bytes"), r.get("cuda_peak_allocated_bytes"),
                         e.get("method"), e.get("batches"), e.get("duration_s"), e.get("energy_j"),
                         e.get("average_power_w"), e.get("energy_per_batch_j"), e.get("energy_per_example_j"),
                         e.get("idle_power_w"), e.get("net_energy_per_batch_j"), e.get("net_energy_per_example_j"),
                         e.get("other_processes_present"), (r.get("format_check") or {}).get("equivalent"),
                         r["latency_is_debug_only"], (r.get("checkpoint") or {}).get("sha256")])
    _write_csv(out_dir / "runs.csv", run_headers, run_rows)
    written.append(out_dir / "runs.csv")
    lines += ["`runs.csv` has one row per benchmark run (latency, memory and every energy field).", ""]

    # -- charts --------------------------------------------------------------------------------------
    plt = _pyplot()
    if plt is not None and bench_agg:
        original_latency = {b: _get(bench_agg, (original_key, "dense", b), "latency") for b in batches}
        panels = [(f"batch {b}", _series(bench_agg, "magnitude", PRUNED_ORDER, b, "latency"),
                   original_latency[b]["mean"] if original_latency[b] else None) for b in batches]
        if any(series for _, series, _ in panels):
            path = out_dir / "latency_vs_sparsity.png"
            line_panels(plt, path, panels, STYLES, "Inference latency by format, global magnitude pruning",
                        f"Median of {config['measured_runs']} synchronized passes per repetition; mean over model "
                        "seeds, whiskers min - max." + (" CPU DEBUG RUN." if debug else ""),
                        lambda v: f"{v:.2g} ms", log_y=True)
            written.append(path)
        original_energy = {b: _get(bench_agg, (original_key, "dense", b), "energy_per_example_j") for b in batches}
        panels = [(f"batch {b}", _series(bench_agg, "magnitude", PRUNED_ORDER, b, "energy_per_example_j", 1e6),
                   1e6 * original_energy[b]["mean"] if original_energy[b] else None) for b in batches]
        if any(series for _, series, _ in panels):
            path = out_dir / "energy_vs_sparsity.png"
            line_panels(plt, path, panels, STYLES, "GPU energy per example by format, global magnitude pruning",
                        "NVML, whole card, gross; synchronized inference of the benchmark batch for at least "
                        f"{config['energy_seconds']:g} s, divided by the examples processed. Mean over model seeds.",
                        lambda v: f"{v:.3g} uJ", log_y=True)
            written.append(path)
    if plt is not None and low_rows:
        curve = sorted((eval_agg[(k, "csr")]["stored_bytes"], eval_agg[(k, "csr")]["test"]["mean"])
                       for k in pruned_rows if k[1] == "magnitude" and (k, "csr") in eval_agg)
        if (original_key, "dense") in eval_agg:
            reference = eval_agg[(original_key, "dense")]
            original = (reference["stored_bytes"], reference["test"]["mean"])
            path = out_dir / "low_rank_accuracy_vs_storage.png"
            rules = list(dict.fromkeys(k[2] for k in low_rows))
            low_rank_figure(plt, path, eval_agg, curve, original, config["low_rank_ranks"], rules,
                            "Mean over model seeds, whiskers min - max. Labels: rank. No fine-tuning.")
            written.append(path)

    figures = [p.name for p in written if p.suffix == ".png"]
    if figures:
        lines[2:2] = [" ".join(f"![{name}]({name})" for name in figures), ""]
    (out_dir / "summary.md").write_text("\n".join(lines))
    return [out_dir / "summary.md"] + written

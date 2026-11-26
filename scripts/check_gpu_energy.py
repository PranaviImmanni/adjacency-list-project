"""Check NVML's energy readings on one GPU: idle, under a short synthetic load, then idle again.

Every --interval seconds it prints the power from the energy counter over that interval,
NVML's power reading (averaged over about 1 s on Ampere), the driver's instantaneous power
field when it exists, the performance state, clocks, utilization and other processes.
At the end it compares, per phase, the counter's average power with the power readings.
Used to decide which source to trust when they disagree (for example at idle).

Nothing is written; the load is a few seconds of FP32 matrix multiplies on --device.

    python scripts/check_gpu_energy.py --device cuda:1
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gcn.measurement import resolve_device
from src.mlp.energy import close_meter, open_meter

POWER_INSTANT_FIELD = 186  # NVML_FI_DEV_POWER_INSTANT (newer drivers only)


def instant_power_w(nvml, handle):
    try:
        value = nvml.nvmlDeviceGetFieldValues(handle, [POWER_INSTANT_FIELD])[0]
        return None if value.nvmlReturn != 0 else value.value.uiVal / 1e3
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--idle", type=float, default=10.0, help="seconds idle before the load")
    parser.add_argument("--load", type=float, default=6.0, help="seconds of matrix multiplies")
    parser.add_argument("--after", type=float, default=8.0, help="seconds idle after the load")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    a = torch.randn(4096, 4096, device=device)  # creates the CUDA context, like a benchmark run
    torch.cuda.synchronize(device)
    meter, reason = open_meter(device)
    if meter is None:
        sys.exit(f"energy unavailable: {reason}")
    nvml, handle = meter.nvml, meter.handle
    print(f"{meter.device['name']} (NVML index {meter.device['nvml_index']}, matched by "
          f"{meter.device['matched_by']}), method: {meter.method}")
    if meter.method != "total_energy_counter":
        close_meter(meter)
        sys.exit("this GPU has no energy counter; nothing to compare")
    print(f"{'phase':>10} {'t (s)':>6} {'counter W':>9} {'reading W':>9} {'instant W':>9} {'P':>2} "
          f"{'SM MHz':>6} {'mem MHz':>7} {'util':>4}  other processes")

    rows = []
    t_start = time.perf_counter()
    e_prev, t_prev = meter._energy_mj(), t_start
    for phase, seconds in (("idle", args.idle), ("load", args.load), ("idle again", args.after)):
        phase_start = time.perf_counter()
        next_print = phase_start + args.interval
        while True:
            now = time.perf_counter()
            if phase == "load":
                a @ a
                torch.cuda.synchronize(device)
            else:
                time.sleep(0.01)
            if now >= next_print:
                e = meter._energy_mj()
                state = meter.state()
                row = {"phase": phase, "t": now - t_start, "counter_w": (e - e_prev) / 1e3 / (now - t_prev),
                       "reading_w": state["power_w"], "instant_w": instant_power_w(nvml, handle), **state}
                rows.append(row)
                e_prev, t_prev = e, now
                next_print += args.interval
                fmt = lambda v, spec: "n/a" if v is None else format(v, spec)  # noqa: E731
                print(f"{phase:>10} {row['t']:6.1f} {row['counter_w']:9.1f} {fmt(row['reading_w'], '9.1f')} "
                      f"{fmt(row['instant_w'], '9.1f')} {fmt(row['performance_state'], '2.0f')} "
                      f"{fmt(row['sm_clock_mhz'], '6.0f')} {fmt(row['memory_clock_mhz'], '7.0f')} "
                      f"{fmt(row['gpu_utilization_pct'], '4d')}  {row['other_processes']}", flush=True)
            if now - phase_start >= seconds:
                break
    close_meter(meter)

    print("\nSecond half of each phase (the 1 s averaged reading has settled):")
    for phase in ("idle", "load", "idle again"):
        part = [r for r in rows if r["phase"] == phase]
        part = part[len(part) // 2:]
        if not part:
            continue
        mean = lambda key: statistics.fmean(r[key] for r in part if r[key] is not None)  # noqa: E731
        instants = [r["instant_w"] for r in part if r["instant_w"] is not None]
        print(f"  {phase:>10}: counter {mean('counter_w'):6.1f} W, power reading {mean('reading_w'):6.1f} W"
              + (f", instantaneous {statistics.fmean(instants):6.1f} W" if instants else ""))


if __name__ == "__main__":
    main()

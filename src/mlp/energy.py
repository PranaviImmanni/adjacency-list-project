"""GPU energy during synchronized inference, measured with NVML (package nvidia-ml-py, module pynvml).

What is measured, for one model / format / batch on one CUDA device:
  1. idle: nothing runs for `idle_seconds` (the model is loaded); average idle power.
  2. active: the batch is run back to back, each pass followed by a device
     synchronize (the same synchronized inference as the latency timing),
     for at least `seconds`. Energy, time and the number of passes are recorded.
Reported: average power (W), energy per batch (J) and per example (J), each
gross (what the card drew) and net of idle power (gross - idle power x time).

Two NVML methods, chosen per run and recorded:
  total_energy_counter  nvmlDeviceGetTotalEnergyConsumption (mJ since driver
                        load). The counter only changes every few ms, so both
                        ends of each window are aligned to a counter change: the
                        window starts right after a change and the workload
                        runs until the counter changes again after `seconds`.
  power_sampling        fallback when the counter is not supported: a thread
                        samples nvmlDeviceGetPowerUsage every
                        `sample_interval_ms` and the power is integrated over
                        the window. NVML power readings can be averaged over up
                        to ~1 s, so the workload runs for 1 s before the window.

The NVML device is matched to the CUDA device by UUID (or PCI address), never
by index, so CUDA_VISIBLE_DEVICES or a different enumeration order cannot
make it read another GPU. NVML reports power for the whole card: other
processes on the same GPU are recorded and flagged. Clocks are not locked
(that needs root).

Off CUDA, or without NVML, every energy field is None with the reason.
"""

import statistics
import threading
import time


def unavailable(reason):
    return {"available": False, "reason": reason}


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _uuid_key(value):
    text = _text(value).strip().lower()
    for prefix in ("gpu-", "mig-"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text


def cuda_identity(device):
    """UUID and PCI address of a CUDA device, from PyTorch."""
    import torch
    props = torch.cuda.get_device_properties(device)
    uuid = getattr(props, "uuid", None)
    pci = None
    if all(hasattr(props, a) for a in ("pci_domain_id", "pci_bus_id", "pci_device_id")):
        pci = (int(props.pci_domain_id), int(props.pci_bus_id), int(props.pci_device_id))
    return {"uuid": _uuid_key(uuid) if uuid is not None else None, "pci": pci, "name": props.name}


def match_handle(nvml, identity):
    """(handle, how) of the NVML device with this UUID / PCI address; raises if there is none."""
    candidates = []
    for index in range(nvml.nvmlDeviceGetCount()):
        handle = nvml.nvmlDeviceGetHandleByIndex(index)
        candidates.append((index, handle))
        if identity.get("uuid") and _uuid_key(nvml.nvmlDeviceGetUUID(handle)) == identity["uuid"]:
            return index, handle, "uuid"
    if identity.get("pci"):
        for index, handle in candidates:
            info = nvml.nvmlDeviceGetPciInfo(handle)
            if (int(info.domain), int(info.bus), int(info.device)) == identity["pci"]:
                return index, handle, "pci"
    raise RuntimeError(f"no NVML device matches the CUDA device {identity}")


class EnergyMeter:
    def __init__(self, nvml, handle, nvml_index, matched_by, sample_interval_ms=10, clock=time.perf_counter):
        self.nvml, self.handle, self.clock = nvml, handle, clock
        self.sample_interval_s = sample_interval_ms / 1e3
        self.device = {"nvml_index": nvml_index, "matched_by": matched_by,
                       "name": _text(nvml.nvmlDeviceGetName(handle)), "uuid": _text(nvml.nvmlDeviceGetUUID(handle))}
        try:
            nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            self.method = "total_energy_counter"
        except Exception:  # NVMLError_NotSupported (or an older driver)
            self.method = "power_sampling"

    # -- device state -------------------------------------------------------------------------
    def _safe(self, fn, *args, scale=1.0):
        try:
            value = fn(self.handle, *args)
            return value * scale if isinstance(value, (int, float)) else value
        except Exception:
            return None

    def other_processes(self):
        import os
        pids = []
        for fn in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
            try:
                pids += [int(p.pid) for p in getattr(self.nvml, fn)(self.handle)]
            except Exception:
                continue
        return sorted({p for p in pids if p != os.getpid()})

    def state(self):
        nvml = self.nvml
        utilization = self._safe(nvml.nvmlDeviceGetUtilizationRates)
        return {
            "temperature_c": self._safe(nvml.nvmlDeviceGetTemperature, getattr(nvml, "NVML_TEMPERATURE_GPU", 0)),
            "sm_clock_mhz": self._safe(nvml.nvmlDeviceGetClockInfo, getattr(nvml, "NVML_CLOCK_SM", 1)),
            "memory_clock_mhz": self._safe(nvml.nvmlDeviceGetClockInfo, getattr(nvml, "NVML_CLOCK_MEM", 2)),
            "power_limit_w": self._safe(nvml.nvmlDeviceGetEnforcedPowerLimit, scale=1e-3),
            "power_w": self._safe(nvml.nvmlDeviceGetPowerUsage, scale=1e-3),
            "performance_state": self._safe(nvml.nvmlDeviceGetPerformanceState),
            "gpu_utilization_pct": None if utilization is None else getattr(utilization, "gpu", None),
            "other_processes": self.other_processes(),
        }

    # -- total energy counter ------------------------------------------------------------------
    def _energy_mj(self):
        return self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)

    def _next_change(self, step=None, timeout_s=2.0):
        """(energy mJ, time, intervals) at the next counter change; runs `step` meanwhile if given."""
        start_value, start = self._energy_mj(), self.clock()
        extra = 0
        while True:
            if step is not None:
                step()
                extra += 1
            else:
                time.sleep(0.0005)
            value, now = self._energy_mj(), self.clock()
            if value != start_value:
                return value, now, extra
            if now - start > timeout_s:
                raise RuntimeError("the NVML energy counter did not change within "
                                   f"{timeout_s:g} s; use power sampling")

    def _idle_counter(self, seconds):
        e0, t0, _ = self._next_change()
        changes, last = [t0], e0
        while self.clock() - t0 < seconds:
            time.sleep(0.001)
            value = self._energy_mj()
            if value != last:
                changes.append(self.clock())
                last = value
        e1, t1, _ = self._next_change()
        changes.append(t1)
        intervals = [b - a for a, b in zip(changes, changes[1:]) if b > a]
        return {"duration_s": t1 - t0, "energy_j": (e1 - e0) / 1e3,
                "counter_update_interval_ms": 1e3 * statistics.median(intervals) if intervals else None}

    def _active_counter(self, step, seconds):
        e0, t0, _ = self._next_change()
        passes = 0
        while self.clock() - t0 < seconds:
            step()
            passes += 1
        e1, t1, extra = self._next_change(step=step)
        return {"duration_s": t1 - t0, "energy_j": (e1 - e0) / 1e3, "batches": passes + extra}

    # -- power sampling ------------------------------------------------------------------------
    def _sampler(self):
        samples, stop = [], threading.Event()

        def run():
            while not stop.is_set():
                try:
                    samples.append((self.clock(), self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1e3))
                except Exception:
                    pass
                stop.wait(self.sample_interval_s)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return samples, stop, thread

    @staticmethod
    def _integrate(samples, t0, t1):
        window = [(t, p) for t, p in samples if t0 <= t <= t1]
        if len(window) < 2:
            raise RuntimeError("too few power samples in the window")
        energy = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(window, window[1:]))
        return energy, window[-1][0] - window[0][0], len(window)

    def _idle_sampling(self, seconds):
        samples, stop, thread = self._sampler()
        t0 = self.clock()
        time.sleep(seconds)
        t1 = self.clock()
        stop.set()
        thread.join()
        energy, duration, n = self._integrate(samples, t0, t1)
        return {"duration_s": duration, "energy_j": energy, "power_samples": n, "counter_update_interval_ms": None}

    def _active_sampling(self, step, seconds, preroll_s=1.0):
        samples, stop, thread = self._sampler()
        start = self.clock()
        while self.clock() - start < preroll_s:
            step()
        t0, passes = self.clock(), 0
        while self.clock() - t0 < seconds:
            step()
            passes += 1
        t1 = self.clock()
        stop.set()
        thread.join()
        energy, sampled_s, n = self._integrate(samples, t0, t1)
        # the samples span slightly less than the window: scale the integral to the whole window
        return {"duration_s": t1 - t0, "energy_j": energy * (t1 - t0) / sampled_s,
                "batches": passes, "power_samples": n, "preroll_s": preroll_s}

    # -- measurement ---------------------------------------------------------------------------
    def measure(self, step, batch_size, seconds, idle_seconds):
        """Energy of synchronized inference. `step` runs one batch and synchronizes the device."""
        before = self.state()
        if self.method == "total_energy_counter":
            idle = self._idle_counter(idle_seconds)
            active = self._active_counter(step, seconds)
        else:
            idle = self._idle_sampling(idle_seconds)
            active = self._active_sampling(step, seconds)
        after = self.state()
        idle_power = idle["energy_j"] / idle["duration_s"]
        energy, duration, batches = active["energy_j"], active["duration_s"], active["batches"]
        net_energy = energy - idle_power * duration
        others = sorted(set(before["other_processes"]) | set(after["other_processes"]))
        return {
            "available": True,
            "method": self.method,
            "device": self.device,
            "batch_size": batch_size,
            "batches": batches,
            "examples": batches * batch_size,
            "duration_s": duration,
            "energy_j": energy,
            "average_power_w": energy / duration,
            "energy_per_batch_j": energy / batches,
            "energy_per_example_j": energy / (batches * batch_size),
            "idle_power_w": idle_power,
            "net_average_power_w": net_energy / duration,
            "net_energy_per_batch_j": net_energy / batches,
            "net_energy_per_example_j": net_energy / (batches * batch_size),
            "seconds_per_batch_in_loop": duration / batches,
            "idle": idle,
            "active": active,
            "state_before": before,
            "state_after": after,
            "other_processes": others,
            "other_processes_present": bool(others),
        }


def open_meter(device, sample_interval_ms=10, nvml=None):
    """(EnergyMeter, None) for a CUDA device, or (None, reason) when energy cannot be measured."""
    if getattr(device, "type", str(device)) != "cuda":
        return None, "energy is measured on CUDA devices only"
    if nvml is None:
        try:
            import pynvml as nvml
        except ImportError:
            return None, "pynvml is not installed (pip install nvidia-ml-py)"
    try:
        nvml.nvmlInit()
    except Exception as error:
        return None, f"NVML could not be initialized: {error!r}"
    try:
        index, handle, how = match_handle(nvml, cuda_identity(device))
        return EnergyMeter(nvml, handle, index, how, sample_interval_ms), None
    except Exception as error:  # no matching NVML device: measure nothing rather than another GPU
        close_meter_nvml(nvml)
        return None, f"no NVML device could be matched to {device}: {error!r}"


def close_meter_nvml(nvml):
    try:
        nvml.nvmlShutdown()
    except Exception:
        pass


def close_meter(meter):
    if meter is not None:
        close_meter_nvml(meter.nvml)

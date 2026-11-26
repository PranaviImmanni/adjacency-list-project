"""GPU format comparison: adjacency list, low-rank factors (vs Pranavi's script), energy meter, plan."""

import importlib.util
import warnings

import numpy as np
import pytest
import torch

from src.gcn.config import PROJECT_ROOT
from src.mlp.config import load_comparison_config
from src.mlp.energy import EnergyMeter, match_handle, open_meter
from src.mlp.formats import AdjacencyList, build_format_model
from src.mlp.gpu_comparison import (build, cross_check, equivalence, exact_forward, pilot_plan, plan, prepare,
                                    pranavi_reference, resident_ok)
from src.mlp.low_rank import build_low_rank_model, low_rank_factors
from src.mlp.model import MLP
from src.mlp.weight_pruning import prune_layers

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")


def _model(seed=0):
    torch.manual_seed(seed)
    return MLP(20, [12, 8], 4)


def _pranavi(script):
    spec = importlib.util.spec_from_file_location(script, PROJECT_ROOT / "scripts" / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- adjacency list --------------------------------------------------------------------------

@pytest.mark.parametrize("sparsity", [0.5, 0.9, 0.99])
def test_adjacency_list_matches_pruned_dense_and_csr_bytes(sparsity):
    model = _model()
    weights = {n: getattr(model, n).weight.detach() for n in model.layer_names}
    masks = prune_layers(weights, "magnitude", sparsity)["masks"]
    x = torch.randn(7, 20)
    with torch.inference_mode():
        dense = build_format_model(model, masks, "dense")(x)
        lists = build_format_model(model, masks, "adjacency_list")
        assert torch.allclose(lists(x), dense, atol=1e-6, rtol=1e-5)
    assert lists.storage()["weight_bytes"] == build_format_model(model, masks, "csr").storage()["weight_bytes"]
    assert resident_ok(lists, "adjacency_list")


def test_adjacency_list_structure_and_empty_rows():
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    mask = torch.tensor([[True, False, True, False], [False] * 4, [False, False, False, True]])
    lists = AdjacencyList.from_mask(weight, mask)
    assert lists.offsets.tolist() == [0, 2, 2, 3]
    assert lists.neighbors.tolist() == [0, 2, 3] and lists.weights.tolist() == [0.0, 2.0, 11.0]
    assert torch.equal(lists.to_dense(), weight * mask)
    h = torch.randn(4, 5)
    out = lists.aggregate(h)
    assert torch.allclose(out, (weight * mask) @ h) and torch.equal(out[1], torch.zeros(5))
    empty = AdjacencyList.from_mask(weight, torch.zeros(3, 4, dtype=torch.bool))
    assert torch.equal(empty.aggregate(h), torch.zeros(3, 5))
    # 4 B value + 8 B neighbor per edge, 8 B per offset
    assert sum(lists.storage_parts().values()) == 3 * 12 + 4 * 8


# --- low-rank factors: the same numbers as Pranavi's script ----------------------------------

@pytest.mark.parametrize("rank,rule,k", [(2, "all", None), (3, "abs_std", 0.0), (8, "abs_std", 1.0), (50, "all", None)])
def test_low_rank_matches_pranavi_script(rank, rule, k):
    theirs = _pranavi("benchmark_low_rank_adjacency")
    model = _model(1)
    ours = low_rank_factors(model, rank, rule, k)
    x = torch.randn(9, 20)
    weights = [layer.weight.detach().double().numpy() for layer in model.layers()]
    biases = [layer.bias.detach().double().numpy() for layer in model.layers()]
    factor_layers = []
    for weight in weights:
        left, right = theirs.factorize(weight, rank)
        lm, _ = theirs.factor_mask(left, rule, 0.0 if k is None else k)
        rm, _ = theirs.factor_mask(right, rule, 0.0 if k is None else k)
        factor_layers.append((theirs.make_adjacency(left, lm), theirs.make_adjacency(right, rm)))
    their_logits = theirs.forward_factors(factor_layers, biases, x.double().numpy())
    their_bytes, their_edges = theirs.structural_factor_bytes(factor_layers)

    lists = build_low_rank_model(ours["layers"], "factor_adjacency")
    dense = build_low_rank_model(ours["layers"], "factor_dense")
    with torch.inference_mode():
        out_lists, out_dense = lists(x), dense(x)
    assert ours["summary"]["factor_edges"] == their_edges
    assert lists.storage()["weight_bytes"] == their_bytes
    assert np.allclose(out_lists.double().numpy(), their_logits, atol=1e-4, rtol=1e-4)
    assert torch.allclose(out_lists, out_dense, atol=1e-5, rtol=1e-4)
    assert resident_ok(lists, "factor_adjacency")
    if rank >= 12 and rule == "all":  # full rank: the product reproduces W
        assert ours["summary"]["mean_weight_reconstruction_mse"] < 1e-12


def test_low_rank_rank_is_capped_per_layer():
    summary = low_rank_factors(_model(), 64, "all")["summary"]
    assert [layer["rank"] for layer in summary["layers"].values()] == [12, 8, 4]


# --- plan and config -------------------------------------------------------------------------

def test_plan_counts_and_pilot():
    config = load_comparison_config()
    specs = plan(config, 42, "benchmark")
    runs = sum(len(s["formats"]) for s in specs) * len(config["batch_sizes"]) * config["repetitions"]
    assert runs == (1 + 15 * 5 + 15 * 2) * 3 * 3
    assert len(plan(config, 42, "evaluate", ["pruned"])) == 1 + 5 + 5 + 15  # random: 3 pruning seeds
    assert {s["tag"] for s in pilot_plan(config, 42)} == {"m42_none_s0", "m42_magnitude_s90",
                                                         "m42_lowrank_r32_all", "m42_lowrank_r32_abs_std_k0"}
    spec = next(s for s in specs if s["tag"] == "m42_lowrank_r8_abs_std_k1")
    assert spec["k"] == 1.0 and spec["reference_format"] == "factor_dense"


def test_prepare_and_build_every_format():
    config = load_comparison_config()
    model = _model()
    x = torch.randn(3, 20)
    for spec in (s for s in plan(config, 0, "benchmark") if s["tag"] in ("m0_magnitude_s90", "m0_lowrank_r4_all")):
        prepared = prepare(spec, model, {**config, "pruned_layers": model.layer_names})
        reference = build(spec, prepared, model, spec["reference_format"])
        with torch.inference_mode():
            for storage_format in spec["formats"]:
                out = build(spec, prepared, model, storage_format)(x)
                assert torch.allclose(out, reference(x), atol=1e-5, rtol=1e-4)


def test_exact_forward_matches_both_parts():
    config = {**load_comparison_config(), "pruned_layers": ["fc1", "fc2", "fc3"]}
    model = _model()
    x = torch.randn(5, 20)
    for spec in (s for s in plan(config, 0, "benchmark") if s["tag"] in ("m0_magnitude_s90", "m0_lowrank_r4_all")):
        prepared = prepare(spec, model, config)
        with torch.inference_mode():
            fp32 = build(spec, prepared, model, spec["reference_format"])(x)
        exact = exact_forward(spec, prepared, model)(x)
        assert exact.dtype == torch.float64 and torch.allclose(exact, fp32.double(), atol=1e-5)


def test_equivalence_strict_rounding_and_failure():
    torch.manual_seed(0)
    exact = torch.randn(200, 10, dtype=torch.float64) * 60  # large logits, as in the low-rank models
    y = exact.argmax(1)
    mask = torch.ones(200, dtype=torch.bool)
    reference = (exact + 1e-4 * torch.randn_like(exact)).float()
    close = (exact + 1e-4 * torch.randn_like(exact)).float()
    check = equivalence(reference, close, exact, y, mask)
    assert not check["strict_equivalent"] and check["within_fp32_rounding"] and check["equivalent"]
    assert check["equivalence_basis"] == "fp32_rounding"
    assert equivalence(reference, reference.clone(), exact, y, mask)["equivalence_basis"] == "tolerance"
    far = (exact + 1e-2 * torch.randn_like(exact)).float()  # 100x the reference's rounding error
    assert not equivalence(reference, far, exact, y, mask)["equivalent"]
    flipped = reference.clone()
    flipped[0] = -flipped[0]  # one changed prediction
    assert not equivalence(reference, flipped, exact, y, mask)["equivalent"]


def test_pranavi_reference_rows_and_cross_check():
    config = load_comparison_config()
    specs = {s["tag"]: s for s in plan(config, 42, "evaluate")}
    ref = pranavi_reference(specs["m42_lowrank_r32_abs_std_k0"])
    assert ref["edges"] == 19527 and ref["stored_bytes"] == 238244 and ref["test_accuracy"] == pytest.approx(0.6156)
    assert pranavi_reference(specs["m42_lowrank_r8_abs_std_k1"])["source"].endswith("_k1.csv")
    assert pranavi_reference(specs["m42_random_p101_s90"])["edges"] == 26620
    assert pranavi_reference(specs["m42_random_p102_s90"]) is None
    check = cross_check(ref, 0.6157, 19527, 10000, stored_bytes=238244)
    assert check["difference_images"] == 1 and check["edges_equal"] and check["stored_bytes_equal"]


# --- energy ----------------------------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeNVML:
    """Energy counter (mJ) that only changes every `interval` s; `idle_w` while idle, `active_w` while busy."""

    NVML_TEMPERATURE_GPU, NVML_CLOCK_SM, NVML_CLOCK_MEM = 0, 1, 2

    def __init__(self, clock, idle_w=30.0, active_w=110.0, interval=0.02, counter=True, uuid="GPU-ABC-123"):
        self.clock, self.idle_w, self.active_w, self.interval = clock, idle_w, active_w, interval
        self.counter, self.uuid = counter, uuid
        self.segments = [(0.0, 0.0, idle_w)]  # (start time, energy at start in mJ, power in W)

    def set_busy(self, busy):
        t0, e0, power = self.segments[-1]
        now = self.clock()
        self.segments.append((now, e0 + power * (now - t0) * 1e3, self.active_w if busy else self.idle_w))

    def energy_at(self, t):
        t0, e0, power = [s for s in self.segments if s[0] <= t][-1]
        return e0 + power * (t - t0) * 1e3

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetCount(self):
        return 2

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetUUID(self, handle):
        return self.uuid if handle == 1 else "GPU-other"

    def nvmlDeviceGetName(self, handle):
        return "Fake 3090"

    def nvmlDeviceGetTotalEnergyConsumption(self, handle):
        if not self.counter:
            raise RuntimeError("NotSupported")
        return int(self.energy_at(int(self.clock() / self.interval) * self.interval))

    def nvmlDeviceGetPowerUsage(self, handle):
        return int(self.segments[-1][2] * 1e3)

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        return []

    def nvmlDeviceGetGraphicsRunningProcesses(self, handle):
        return []

    def nvmlDeviceGetTemperature(self, handle, sensor):
        return 40

    def nvmlDeviceGetClockInfo(self, handle, clock):
        return 1695

    def nvmlDeviceGetEnforcedPowerLimit(self, handle):
        return 350000

    def nvmlDeviceGetPerformanceState(self, handle):
        return 2

    def nvmlDeviceGetUtilizationRates(self, handle):
        return type("U", (), {"gpu": 0})()


def test_energy_counter_measurement(monkeypatch):
    clock = FakeClock()
    nvml = FakeNVML(clock)
    monkeypatch.setattr("time.sleep", lambda s: setattr(clock, "t", clock.t + s))
    meter = EnergyMeter(nvml, 1, 1, "uuid", clock=clock)
    assert meter.method == "total_energy_counter"

    def step():  # one synchronized batch: the GPU is busy for 0.25 ms
        nvml.set_busy(True)
        clock.t += 0.00025
        nvml.set_busy(False)

    result = meter.measure(step, batch_size=32, seconds=0.5, idle_seconds=0.2)
    assert result["idle_power_w"] == pytest.approx(30.0, rel=0.02)
    assert result["average_power_w"] == pytest.approx(110.0, rel=0.03)
    assert result["energy_per_batch_j"] == pytest.approx(110.0 * 0.00025, rel=0.03)
    assert result["energy_per_example_j"] == pytest.approx(result["energy_per_batch_j"] / 32)
    assert result["net_energy_per_batch_j"] == pytest.approx(80.0 * 0.00025, rel=0.05)
    assert result["idle"]["counter_update_interval_ms"] == pytest.approx(20.0, rel=0.1)
    assert result["active"]["preroll_s"] == 0.5 and len(result["active"]["trace"]) > 10
    assert result["idle_power_reading_w"] == pytest.approx(30.0)
    assert not result["other_processes_present"]


def test_energy_falls_back_to_power_sampling():
    meter = EnergyMeter(FakeNVML(FakeClock(), counter=False), 1, 1, "uuid")
    assert meter.method == "power_sampling"


def test_device_matching_by_uuid_and_off_cuda():
    nvml = FakeNVML(FakeClock(), uuid="GPU-AbC-123")
    assert match_handle(nvml, {"uuid": "abc-123", "pci": None})[1:] == (1, "uuid")
    with pytest.raises(RuntimeError):
        match_handle(nvml, {"uuid": "nope", "pci": None})
    meter, reason = open_meter(torch.device("cpu"))
    assert meter is None and "CUDA" in reason


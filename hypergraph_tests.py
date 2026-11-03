"""Reproducible incidence propagation experiments. See README.md."""
import argparse
import csv
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import torch

METHODS = ['dense', 'coo', 'csr', 'membership', 'explicit_dense', 'explicit_sparse']
ATOL, RTOL = 1e-5, 1e-4


def graph(n, edges):
    """Deduplicate memberships; preserve separate columns for identical groups."""
    rows, cols = [], []
    for e, members in enumerate(edges):
        members = sorted(set(members))
        if not members:
            raise ValueError('Empty hyperedges are not allowed')
        if any(v < 0 or v >= n for v in members):
            raise ValueError('Node ID outside graph')
        rows.extend(members)
        cols.extend([e] * len(members))
    return torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)


def synthetic(n, m, k, f, seed):
    if not 1 <= k <= n or m < 1:
        raise ValueError('Require 1 <= k <= n and m >= 1')
    rng = torch.Generator().manual_seed(seed)
    # Draw memberships directly; no N x N or N x M allocation.
    groups = [torch.randperm(n, generator=rng)[:k].tolist() for _ in range(m)]
    r, c = graph(n, groups)
    x = torch.randn(n, f, generator=rng)
    w = 0.5 + torch.rand(m, generator=rng)
    return r, c, x, w


def degrees(n, m, r, c, w, normalized):
    if w.numel() != m or not bool(torch.isfinite(w).all()) or not bool((w > 0).all()):
        raise ValueError('Require one finite positive weight per hyperedge')
    de = torch.bincount(c, minlength=m).to(w.dtype)
    dv = torch.zeros(n, dtype=w.dtype, device=w.device)
    dv.index_add_(0, r, w[c])
    d = torch.zeros_like(dv)
    d[dv > 0] = dv[dv > 0].rsqrt()
    return (d, w / de) if normalized else (torch.ones_like(d), w)


def incidence(n, m, r, c, dtype):
    return torch.sparse_coo_tensor(torch.stack([r, c]),
                                   torch.ones(r.numel(), dtype=dtype, device=r.device),
                                   (n, m)).coalesce()


def reference(n, m, r, c, x, w, normalized):
    """Independent dense FP64 reference on small cases; sparse FP64 for scaling."""
    x, w = x.double(), w.double()
    d, a = degrees(n, m, r, c, w, normalized)
    h = incidence(n, m, r, c, x.dtype)
    if n * m <= 1_000_000:
        h = h.to_dense()
        return d[:, None] * (h @ (a[:, None] * (h.T @ (d[:, None] * x))))
    return d[:, None] * torch.sparse.mm(h, a[:, None] * torch.sparse.mm(h.T, d[:, None] * x))


def storage_bytes(tensors):
    """Actual distinct tensor buffers, including index buffers and transposes."""
    seen, total = set(), 0
    for t in tensors:
        if t.layout == torch.sparse_coo:
            buffers = [t.indices(), t.values()]
        elif t.layout == torch.sparse_csr:
            buffers = [t.crow_indices(), t.col_indices(), t.values()]
        else:
            buffers = [t]
        for b in buffers:
            s = b.untyped_storage()
            key = (str(b.device), s.data_ptr())
            if key not in seen:
                seen.add(key)
                total += s.nbytes()
    return total


def build(method, n, m, r_cpu, c_cpu, w_cpu, device, normalized=True):
    r, c, w = r_cpu.to(device), c_cpu.to(device), w_cpu.to(device)
    d, a = degrees(n, m, r, c, w, normalized)
    if method == 'membership':
        def operation(x):
            scaled = d[:, None] * x
            z = x.new_zeros(m, x.shape[1])
            z.index_add_(0, c, scaled[r])
            weighted = a[:, None] * z
            out = x.new_zeros(n, x.shape[1])
            out.index_add_(0, r, weighted[c])
            return d[:, None] * out
        return operation, [r, c, d, a], None
    h = incidence(n, m, r, c, w.dtype)
    if method.startswith('explicit'):
        if method == 'explicit_dense':
            hd = h.to_dense()
            p = (hd * a[None, :]) @ hd.T
            p *= d[:, None]
            p *= d[None, :]
            fn = lambda x: p @ x
            nnz = int(torch.count_nonzero(p))
        else:
            left = torch.sparse_coo_tensor(torch.stack([r, c]), d[r] * a[c], (n, m)).coalesce()
            right = torch.sparse_coo_tensor(torch.stack([c, r]), d[r], (m, n)).coalesce()
            p = torch.sparse.mm(left, right).coalesce()
            fn = lambda x: torch.sparse.mm(p, x)
            nnz = p._nnz()
        return fn, [p], nnz
    if method == 'dense':
        h = h.to_dense()
        ht = h.T  # View, shares storage.
        mm = lambda a, b: a @ b
    else:
        ht = h.T.coalesce()
        if method == 'csr':
            h, ht = h.to_sparse_csr(), ht.to_sparse_csr()
        mm = torch.sparse.mm
    def operation(x):
        z = mm(ht, d[:, None] * x)
        return d[:, None] * mm(h, a[:, None] * z)
    return operation, [h, ht, d, a], None


def check(actual, expected):
    a, b = actual.detach().cpu().double(), expected.cpu().double()
    error = (a - b).abs()
    return dict(max_abs_error=float(error.max()), mean_abs_error=float(error.mean()),
                passed=bool(torch.allclose(a, b, atol=ATOL, rtol=RTOL)))


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


class RSSSampler:
    """Observed process RSS, not exact per-operation native allocator peak."""
    def __init__(self):
        self.process = None if Path('/proc/self/status').exists() else psutil.Process()
        self.baseline = self.current_rss()
        self.peak = self.baseline
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
    def sample(self):
        self.peak = max(self.peak, self.current_rss())
    def current_rss(self):
        if self.process is not None:
            return self.process.memory_info().rss
        # /proc/self works even when the host procfs and PID namespace differ.
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
        raise RuntimeError('Process RSS unavailable')
    def run(self):
        while not self.stop.wait(0.001):
            self.sample()
    def __enter__(self):
        self.thread.start()
        return self
    def __exit__(self, *args):
        self.sample()
        self.stop.set()
        self.thread.join()


def worker(args):
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only CPU and CUDA are benchmarked; use CPU on Apple M1')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(args.threads)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    r, c, x_cpu, w = synthetic(args.n, args.m, args.k, args.f, args.seed)
    expected = reference(args.n, args.m, r, c, x_cpu, w, True)
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    with RSSSampler() as rss, torch.inference_mode():
        x = x_cpu.to(device)
        sync(device)
        transfer_start = time.perf_counter()
        r_device, c_device, w_device = r.to(device), c.to(device), w.to(device)
        sync(device)
        graph_transfer_ms = (time.perf_counter() - transfer_start) * 1000
        sync(device)
        start = time.perf_counter()
        fn, buffers, nnz = build(args.method, args.n, args.m, r_device, c_device, w_device, device)
        sync(device)
        build_ms = (time.perf_counter() - start) * 1000
        del r_device, c_device, w_device
        rss.sample()
        y = fn(x)
        sync(device)
        validation = check(y, expected)
        del y
        if not validation['passed']:
            raise AssertionError(f'Correctness failure: {validation}')
        for _ in range(args.warmup):
            y = fn(x)
            del y
        sync(device)
        times = []
        for _ in range(args.repeats):
            sync(device)
            start = time.perf_counter()
            y = fn(x)
            sync(device)
            times.append((time.perf_counter() - start) * 1000)
            del y  # One returned output per invocation, no old output overlap.
        rss.sample()
    q = statistics.quantiles(times, n=4)
    result = dict(status='ok', method=args.method, device=str(device), n=args.n, m=args.m,
                  k=args.k, f=args.f, seed=args.seed, memberships=r.numel(),
                  incidence_density=r.numel() / (args.n * args.m), operator_nnz=nnz,
                  representation_bytes=storage_bytes(buffers), build_ms=build_ms,
                  graph_transfer_ms=graph_transfer_ms,
                  median_ms=statistics.median(times), iqr_ms=q[2] - q[0],
                  cpu_baseline_rss_bytes=rss.baseline, cpu_sampled_peak_rss_bytes=rss.peak,
                  cpu_observed_rss_increase_bytes=rss.peak - rss.baseline,
                  cuda_peak_allocated_bytes=None, cuda_peak_reserved_bytes=None,
                  warmup=args.warmup, repeats=args.repeats, threads=args.threads,
                  **validation)
    if device.type == 'cuda':
        result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    result['total_ms_1_call'] = build_ms + result['median_ms']
    result['total_ms_10_calls_estimate'] = build_ms + 10 * result['median_ms']
    result['total_ms_100_calls_estimate'] = build_ms + 100 * result['median_ms']
    print(json.dumps(result))


def test1(args):
    device = torch.device(args.device)
    cases = [
        ('worked_example', 4, [[0, 1, 2], [1, 3]], [1., 1.]),
        ('isolated_node', 5, [[0, 1, 2], [1, 3]], [1., 1.]),
        ('singleton', 4, [[0], [1, 2], [2, 3]], [1., 1., 1.]),
        ('weighted_overlap', 4, [[0, 1, 2], [1, 3]], [0.5, 2.]),
        ('duplicate_membership', 4, [[0, 0, 1], [1, 2, 3]], [1., 1.]),
        ('identical_distinct_edges', 4, [[0, 1], [0, 1], [2, 3]], [1., 2., 1.]),
        ('all_nodes_isolated_except_one', 4, [[2]], [3.]),
    ]
    results = []
    with torch.inference_mode():
        for name, n, edges, weights in cases:
            r, c = graph(n, edges)
            w = torch.tensor(weights)
            for features in (1, 7):
                x = (torch.arange(1, n + 1, dtype=torch.float32)[:, None] if features == 1
                     else torch.randn(n, features, generator=torch.Generator().manual_seed(42)))
                for normalized in (False, True):
                    expected = reference(n, len(edges), r, c, x, w, normalized)
                    if name == 'worked_example' and features == 1 and not normalized:
                        torch.testing.assert_close(expected.flatten(), torch.tensor([6., 12., 6., 6.], dtype=torch.float64))
                    for method in METHODS:
                        try:
                            fn, _, _ = build(method, n, len(edges), r, c, w, device, normalized)
                            result = dict(case=name, features=features, normalized=normalized,
                                          method=method, status='ok', **check(fn(x.to(device)), expected))
                        except (NotImplementedError, RuntimeError) as e:
                            result = dict(case=name, features=features, normalized=normalized,
                                          method=method, status='error', passed=False, error=str(e))
                        results.append(result)
        for bad in ([[]], [[-1]], [[4]]):
            try:
                graph(4, bad)
            except ValueError:
                continue
            raise AssertionError('Invalid graph accepted')
        r, c = graph(2, [[0, 1]])
        for bad_weight in (0., -1., float('nan')):
            try:
                degrees(2, 1, r, c, torch.tensor([bad_weight]), True)
            except ValueError:
                continue
            raise AssertionError('Invalid weight accepted')
    write_results(args, 'test1', results)
    failures = [r for r in results if not r['passed']]
    print(f'Test 1: {len(results) - len(failures)}/{len(results)} comparisons passed; invalid-input checks passed.')
    if failures:
        raise SystemExit('See test1.json for errors; no silent fallbacks were used.')


def write_results(args, name, rows):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{name}.json').write_text(json.dumps(rows, indent=2))
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (out / f'{name}.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def benchmark(args, name, configs):
    rows = []
    for n, m, k, f in configs:
        for seed in args.seeds:
            for method in args.methods:
                # Conservative heuristic, not an allocator guarantee. Includes a
                # possible dense sparse-product intermediate and gather buffers.
                estimate = 4 * (6*(n+m)*f + 3*m*k*f) + 48*m*k
                if method == 'dense':
                    estimate += 12*n*m
                elif method == 'explicit_dense':
                    estimate += 12*n*m + 12*n*n
                elif method == 'explicit_sparse':
                    estimate += 72*min(n*n, m*k*k)
                info = dict(method=method, n=n, m=m, k=k, f=f, seed=seed)
                if estimate > args.budget_mb * 1024**2:
                    row = dict(**info, status='skipped_preflight', estimated_working_bytes=estimate)
                else:
                    cmd = [sys.executable, str(Path(__file__).resolve()), '_worker', '--device', args.device,
                           '--method', method, '--n', str(n), '--m', str(m), '--k', str(k), '--f', str(f),
                           '--seed', str(seed), '--threads', str(args.threads), '--warmup', str(args.warmup),
                           '--repeats', str(args.repeats)]
                    p = subprocess.run(cmd, capture_output=True, text=True)
                    if p.returncode:
                        row = dict(**info, status='error', error=p.stderr[-3000:])
                    else:
                        row = json.loads(p.stdout.strip().splitlines()[-1])
                rows.append(row)
                write_results(args, name, rows)  # Checkpoint each independent subprocess.
                latency = f"{row['median_ms']:.3f} ms" if row['status'] == 'ok' else row['status']
                print(f'{name} N={n} M={m} k={k} F={f} seed={seed}: {method}: {latency}', flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('test', choices=['test1', 'test2', 'test3', 'all', '_worker'])
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', default='results')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeats', type=int, default=50)
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    p.add_argument('--full', action='store_true', help='Larger sweeps; use --seeds 42 43 44 for three seeds')
    p.add_argument('--budget-mb', type=float, default=512, help='Conservative preflight working-memory ceiling')
    for key, default in [('n', 256), ('m', 256), ('k', 8), ('f', 64), ('seed', 42)]:
        p.add_argument('--' + key, type=int, default=default)
    p.add_argument('--method', choices=METHODS, default='coo')
    args = p.parse_args()
    if args.repeats < 4 or args.warmup < 0 or args.threads < 1 or args.budget_mb <= 0:
        p.error('Require repeats >= 4, warmup >= 0, threads >= 1 and a positive budget')
    torch.set_num_threads(args.threads)
    if args.test == '_worker':
        worker(args)
        return
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(python=sys.version, torch=torch.__version__, platform=platform.platform(),
                    cpu=platform.processor(), cuda_available=torch.cuda.is_available(),
                    cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    arguments=vars(args), atol=ATOL, rtol=RTOL)
    (out / 'environment.json').write_text(json.dumps(metadata, indent=2))
    if args.test in ('test1', 'all'):
        test1(args)
    if args.test in ('test2', 'all'):
        rows2 = benchmark(args, 'test2', [(args.n, args.m, args.k, args.f)])
    if args.test in ('test3', 'all'):
        if args.full:
            configs = [(n, n, 8, 64) for n in (100, 1000, 5000)]
            configs += [(1000, m, 8, 64) for m in (500, 1000, 2000)]
            configs += [(1000, 1000, k, 64) for k in (2, 4, 8, 16, 32)]
            configs += [(1000, 1000, 8, f) for f in (16, 64, 128)]
        else:
            configs = [(128, 128, 8, 16), (256, 256, 8, 16), (128, 64, 8, 16),
                       (128, 128, 2, 16), (128, 128, 16, 16), (128, 128, 8, 64)]
        rows3 = benchmark(args, 'test3', list(dict.fromkeys(configs)))
    print(f'Results saved in {out.resolve()}')
    if any(row['status'] == 'error' for rows in (locals().get('rows2', []), locals().get('rows3', [])) for row in rows):
        raise SystemExit('Some methods failed; inspect result JSON files. No fallback used.')


if __name__ == '__main__':
    main()

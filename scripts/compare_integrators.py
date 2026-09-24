#!/usr/bin/env python3
"""Paired Platen/EM comparison; no training or checkpoint writes.

Run from the repository root with uv run scripts/compare_integrators.py.
Defaults: smoke (B=16, T=3, Delta=1). --profile extended selects B=256,
T=20, Delta=0.01,0.1,1. Arguments use argparse, not key=value overrides.
CPU float64 noise is generated one controller interval at a time; every
coarser Wiener increment is a sum of the same dt=0.01/256 increments.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import torch

from learn_to_cool.gaussian_integrators import (
    METHODS, action_controls, advance_gaussian, covariance_health,
)

CONTROL_DT = 0.01
FINE_SUBSTEPS = 256
SCENARIOS = ("no_feedback", "modulation_minus", "modulation_plus", "seeded_policy")


def frequencies(delta):
    a, b = np.indices((5, 5))
    wz = 1 + ((a + b) % 5) * delta
    return np.stack((4.5 * wz, 4.1 * wz, wz), axis=-1)


def policy(seed, dtype, device):
    # Construct in float64 first: all dtypes/methods use identical weights.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(150, 32, dtype=torch.float64),
                                  torch.nn.Tanh(),
                                  torch.nn.Linear(32, 20, dtype=torch.float64))
    return net.to(device=device, dtype=dtype)


def model(delta, batch, dtype, device):
    omega = torch.as_tensor(frequencies(delta), dtype=dtype, device=device)
    zero = torch.zeros((batch, 5, 5, 3), dtype=dtype, device=device)
    state = (zero.clone(), zero.clone(), zero + 11, zero + 11, zero.clone())
    bx = omega[..., 0] * (omega[..., 0] / omega[0, 0, 0]).sqrt()
    by = omega[..., 1] * (omega[..., 1] / omega[0, 0, 1]).sqrt()
    return state, omega, bx, by


def controls(state, scenario, net, bx, by):
    if scenario == "seeded_policy":
        raw = net(torch.cat((state[0].flatten(1), state[1].flatten(1)), dim=1))
        # Fixed, untrained, modest cold forces. Parametric channels sum to +/-0.1.
        raw = torch.cat((0.05 * torch.tanh(raw[:, :10]), raw[:, 10:]), dim=1)
        force, modulation = action_controls(raw, bx, by, 0.05)
    else:
        force = torch.zeros_like(state[0])
        eps = {"no_feedback": 0., "modulation_minus": -0.1,
               "modulation_plus": 0.1}[scenario]
        modulation = state[0].new_tensor(eps)
    return force, modulation, 0.05 * (1 + modulation)


def noise_blocks(batch, intervals, substeps, seed, dtype, device):
    if FINE_SUBSTEPS % substeps:
        raise ValueError("Substeps must divide 256")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(intervals):
        fine = torch.randn((FINE_SUBSTEPS, batch, 5, 5, 3), generator=generator,
                           dtype=torch.float64) * math.sqrt(CONTROL_DT / FINE_SUBSTEPS)
        coarse = fine.reshape(substeps, FINE_SUBSTEPS // substeps, batch, 5, 5, 3).sum(1)
        yield coarse.to(device=device, dtype=dtype)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def interval_count(duration):
    count = round(duration / CONTROL_DT)
    if count < 1 or not math.isclose(count * CONTROL_DT, duration, abs_tol=1e-12):
        raise ValueError("Duration must be a positive multiple of 0.01")
    return count


def occupation(history):
    # history: time, component, batch, row, col, direction.
    return (history[:, 0]**2 + history[:, 1]**2 + history[:, 2] + history[:, 3]) / 4 - .5


def per_direction(values):
    return values.mean(axis=(-3, -2))


def mean_sem(values):
    return values.mean(0), values.std(0, ddof=1) / math.sqrt(values.shape[0])


def run_accuracy(method, substeps, scenario, delta, args):
    dtype, device = getattr(torch, args.dtype), args.device
    state, omega, bx, by = model(delta, args.batch, dtype, device)
    net = policy(args.seed + 1, dtype, device)
    samples = [torch.stack(state).double().cpu().numpy()]
    failures = dict(nonfinite_modes=0, nonpositive_diagonals=0, uncertainty_violations=0)
    minimum_det, minimum_var, first_failure = math.inf, math.inf, None
    solver_time = 0.
    completed_steps = 0
    started = time.perf_counter()
    with torch.no_grad():
        for interval, increments in enumerate(noise_blocks(
                args.batch, interval_count(args.duration), substeps, args.seed, dtype, device)):
            # Only solver/controller work is timed; noise, diagnostics and transfers are excluded.
            synchronize(device)
            tick = time.perf_counter()
            force, eps, gamma = controls(state, scenario, net, bx, by)
            synchronize(device)
            solver_time += time.perf_counter() - tick
            failed = False
            for j in range(substeps):
                synchronize(device)
                tick = time.perf_counter()
                state = advance_gaussian(state, omega, force, eps, gamma, .5,
                                         CONTROL_DT / substeps, increments[j], method)
                synchronize(device)
                solver_time += time.perf_counter() - tick
                completed_steps += 1
                health = {k: v.item() for k, v in covariance_health(
                    state, 1e-5 if dtype == torch.float32 else 1e-10).items()}
                for key in failures:
                    failures[key] += health[key]
                minimum_det = min(minimum_det, health['min_determinant'])
                minimum_var = min(minimum_var, health['min_variance'])
                if any(health[key] for key in failures) and first_failure is None:
                    first_failure = interval * CONTROL_DT + (j + 1) * CONTROL_DT / substeps
                if health['nonfinite_modes']:
                    failed = True
                    break
            if failed:
                break
            samples.append(torch.stack(state).double().cpu().numpy())
    history = np.stack(samples)
    status = 'nonfinite' if failures['nonfinite_modes'] else (
        'covariance_failure' if first_failure is not None else 'ok')
    row = dict(method=method, substeps=substeps, dt=CONTROL_DT/substeps,
               scenario=scenario, delta=delta, dtype=args.dtype, status=status,
               first_failure_time=first_failure, completed_steps=completed_steps,
               completed_time=(len(history)-1)*CONTROL_DT,
               forward_seconds=solver_time, diagnostic_wall_seconds=time.perf_counter()-started,
               min_determinant=minimum_det, min_variance=minimum_var, **failures)
    n = per_direction(occupation(history))
    avg, sem = mean_sem(n[-1])
    for m, label in enumerate('xyz'):
        row[f'n_{label}'], row[f'n_{label}_sem'] = avg[m], sem[m]
        if scenario == 'no_feedback':
            row[f'heating_residual_{label}'] = avg[m] - (5 + .05 * row['completed_time'])
    return row, history


def compare(row, history, reference):
    # Only compare common, fully completed controller intervals after failure.
    n = min(len(history), len(reference))
    difference = history[:n] - reference[:n]
    row['comparison_time'] = (n-1) * CONTROL_DT
    row['trajectory_mean_rms'] = float(np.sqrt(np.mean(difference[:, :2]**2)))
    row['terminal_mean_rms'] = float(np.sqrt(np.mean(difference[-1, :2]**2)))
    row['covariance_rms'] = float(np.sqrt(np.mean(difference[:, 2:]**2)))
    row['covariance_max_abs'] = float(np.max(np.abs(difference[:, 2:])))
    for component, label in enumerate(('Vxx', 'Vpp', 'Cxp'), 2):
        row[f'{label}_max_abs'] = float(np.max(np.abs(difference[:, component])))
    paired = per_direction(occupation(history[:n])[-1] - occupation(reference[:n])[-1])
    mean, sem = mean_sem(paired)
    for m, label in enumerate('xyz'):
        row[f'paired_n_{label}'], row[f'paired_n_{label}_sem'] = mean[m], sem[m]
    return row


def performance_worker(config):
    """Fresh process: full untruncated BPTT over an explicitly reported window."""
    torch.set_num_threads(config['threads'])
    dtype, device = getattr(torch, config['dtype']), config['device']
    state, omega, bx, by = model(config['delta'], config['batch'], dtype, device)
    net = policy(config['seed'] + 1, dtype, device)
    blocks = list(noise_blocks(config['batch'], interval_count(config['benchmark_duration']),
                              config['substeps'], config['seed'], dtype, device))
    # Warm up operations without retaining a graph.
    with torch.no_grad():
        force, eps, gamma = controls(state, 'seeded_policy', net, bx, by)
        advance_gaussian(state, omega, force, eps, gamma, .5,
                         CONTROL_DT/config['substeps'], blocks[0][0], config['method'])
    synchronize(device)
    if torch.device(device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    begin = time.perf_counter()
    for increments in blocks:
        force, eps, gamma = controls(state, 'seeded_policy', net, bx, by)
        for dW in increments:
            state = advance_gaussian(state, omega, force, eps, gamma, .5,
                                     CONTROL_DT/config['substeps'], dW, config['method'])
    # Terminal frequency-weighted mean energy; includes all five state gradients.
    loss = (omega * (state[0]**2 + state[1]**2 + state[2] + state[3]) / 4).mean()
    synchronize(device)
    forward = time.perf_counter() - begin
    begin = time.perf_counter()
    loss.backward()
    synchronize(device)
    backward = time.perf_counter() - begin
    finite = bool(torch.isfinite(loss).item() and all(
        p.grad is not None and torch.isfinite(p.grad).all().item() for p in net.parameters()))
    # ru_maxrss includes Python/PyTorch, noise, native tensor allocations and autograd.
    # Each worker is isolated, so no earlier comparison contaminates this peak.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == 'darwin' else rss * 1024
    result = dict(bptt_duration=config['benchmark_duration'],
                  bptt_forward_seconds=forward, backward_seconds=backward,
                  bptt_total_seconds=forward+backward, bptt_finite=finite,
                  peak_process_rss_bytes=rss_bytes,
                  peak_cuda_allocated_bytes=(torch.cuda.max_memory_allocated(device)
                      if torch.device(device).type == 'cuda' else None))
    print(json.dumps(result))


def benchmark(method, substeps, delta, args):
    config = vars(args) | dict(method=method, substeps=substeps, delta=delta)
    completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                '--worker', json.dumps(config)],
                               capture_output=True, text=True, check=False)
    if completed.returncode:
        return {'benchmark_status': 'failed', 'benchmark_error': completed.stderr[-2000:]}
    return {'benchmark_status': 'ok', **json.loads(completed.stdout)}


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def save_results(output, metadata, rows):
    output.mkdir(parents=True, exist_ok=True)
    (output/'results.json').write_text(json.dumps(clean_json(
        {'metadata': metadata, 'results': rows}), indent=2, allow_nan=False)+'\n')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (output/'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def write_report(output, metadata, rows):
    """Compare speed only among complete, valid runs meeting the same errors."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lines = ['# Platen versus Euler–Maruyama', '',
             f"Profile: {metadata['profile']}; batch {metadata['batch']}; "
             f"T={metadata['duration']}; {metadata['device']}; {metadata['dtype']}.", '',
             'Fixed seeded policy is untrained. Times are single eager trials. '
             'BPTT times and peak memory cover the reported benchmark window, not full T.', '',
             'Errors use paired controller-time samples, including the terminal state. '
             'RMS means error and maximum absolute covariance error are tested together. '
             'Both have to fall below each listed tolerance; all requested scenarios '
             'must complete without covariance failures. Reference-halving errors '
             'must be below one fifth of the tolerance.', '',
             '| Delta | tolerance | method | dt | worst means RMS | worst covariance error | BPTT forward+backward (s) |',
             '|---:|---:|---|---:|---:|---:|---:|']
    candidates = [r for r in rows if r.get('role') == 'candidate']
    for delta in metadata['splittings']:
        for tolerance in (.1, .05, .01):
            for method in METHODS:
                options = []
                for level in metadata['levels']:
                    group = [r for r in candidates if r['delta'] == delta
                             and r['method'] == method and r['substeps'] == 2**level]
                    if len(group) != len(metadata['scenarios']):
                        continue
                    if not all(r['status'] == 'ok' and r['reference_valid']
                               and r['trajectory_mean_rms'] <= tolerance
                               and r['covariance_max_abs'] <= tolerance
                               and r['reference_trajectory_mean_rms'] <= tolerance/5
                               and r['reference_covariance_max_abs'] <= tolerance/5 for r in group):
                        continue
                    bench = next((r for r in group if r['scenario'] == 'seeded_policy'), {})
                    if bench.get('bptt_finite') is False or bench.get('benchmark_status') == 'failed':
                        continue
                    score = bench.get('bptt_total_seconds', sum(r['forward_seconds'] for r in group))
                    options.append((score, group, bench))
                if options:
                    _, group, bench = min(options, key=lambda x: x[0])
                    timing = bench.get('bptt_total_seconds')
                    lines.append(f"| {delta:g} | {tolerance:g} | {method} | {group[0]['dt']:.8g} | "
                                 f"{max(r['trajectory_mean_rms'] for r in group):.3g} | "
                                 f"{max(r['covariance_max_abs'] for r in group):.3g} | "
                                 f"{timing if timing is not None else 'not measured'} |")
                else:
                    lines.append(f'| {delta:g} | {tolerance:g} | {method} | no qualifying run | — | — | — |')
    lines += ['', '## Reference halving', '',
              '| Delta | scenario | means RMS | covariance max | status |',
              '|---:|---|---:|---:|---|']
    for r in rows:
        if r.get('role') == 'reference_halving_check':
            lines.append(f"| {r['delta']:g} | {r['scenario']} | {r['trajectory_mean_rms']:.3g} | "
                         f"{r['covariance_max_abs']:.3g} | {r['status']} |")
    lines += ['', 'Full direction occupations and standard errors, paired occupation differences, '
              'covariance components, failure counts/times, timing and memory are in results.csv/json.', '',
              'Counts are mode-step observations, not unique failed trajectories. CPU peak RSS '
              'includes the interpreter, libraries, input noise and autograd. CUDA peak allocated '
              'memory is reported separately. Partial failed trajectories never qualify for timing comparisons.', '']
    (output/'report.md').write_text('\n'.join(lines))
    for delta in metadata['splittings']:
        fig, axes = plt.subplots(2, len(metadata['scenarios']), figsize=(13, 6), squeeze=False)
        for j, scenario in enumerate(metadata['scenarios']):
            for method in METHODS:
                group = sorted((r for r in candidates if r['delta'] == delta and
                                r['scenario'] == scenario and r['method'] == method and
                                r['status'] == 'ok'), key=lambda r: r['dt'])
                for i, metric in enumerate(('trajectory_mean_rms', 'covariance_max_abs')):
                    axes[i, j].loglog([r['dt'] for r in group], [r[metric] for r in group],
                                      'o-', label=method.replace('_', ' '))
                    axes[i, j].grid(alpha=.2)
                    axes[i, j].set_xlabel('dt')
            axes[0, j].set_title(scenario.replace('_', ' '))
        axes[0, 0].set_ylabel('Paired means RMS error')
        axes[1, 0].set_ylabel('Max covariance error')
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"Delta={delta:g}; B={metadata['batch']}; T={metadata['duration']} — valid runs only")
        fig.tight_layout()
        fig.savefig(output/f'convergence_delta{delta:g}.png', dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('smoke', 'extended'), default='smoke')
    parser.add_argument('--batch', type=int)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--splittings', type=float, nargs='+')
    parser.add_argument('--levels', type=int, nargs='+', default=list(range(7)))
    parser.add_argument('--scenarios', choices=SCENARIOS, nargs='+', default=list(SCENARIOS))
    parser.add_argument('--dtype', choices=('float32', 'float64'), default='float64')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=20260924)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--benchmark-duration', type=float, default=.1)
    parser.add_argument('--skip-benchmark', action='store_true')
    parser.add_argument('--output', default=None)
    parser.add_argument('--report-only', type=Path, help='Regenerate report/plots from an existing results directory')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.report_only:
        saved = json.loads((args.report_only/'results.json').read_text())
        write_report(args.report_only, saved['metadata'], saved['results'])
        return
    if args.worker:
        performance_worker(json.loads(args.worker))
        return
    smoke = args.profile == 'smoke'
    args.batch = args.batch if args.batch is not None else (16 if smoke else 256)
    args.duration = args.duration if args.duration is not None else (3. if smoke else 20.)
    args.splittings = args.splittings or ([1.] if smoke else [.01, .1, 1.])
    if args.batch < 2 or args.threads < 1 or any(k not in range(7) for k in args.levels):
        parser.error('Use batch >= 2, threads >= 1 and levels 0..6')
    if any(d <= 0 for d in args.splittings):
        parser.error('Splittings must be positive')
    interval_count(args.duration)
    interval_count(args.benchmark_duration)
    torch.set_num_threads(args.threads)
    output = Path(args.output or f'notes/integrator_comparison/{args.profile}_{time.strftime("%Y%m%d_%H%M%S")}')
    if (output/'results.json').exists():
        parser.error('Output already contains results; choose a new directory')
    metadata = vars(args).copy() | dict(torch_version=torch.__version__,
        platform=platform.platform(), control_dt=CONTROL_DT, gamma0=.05, eta=.5,
        initial_occupation=5., frequency_formula='wz=1+((a+b)%5)*delta; wx=4.5*wz; wy=4.1*wz',
        intensity_recoil='gamma=0.05*(1+row_modulation+column_modulation)',
        reference_dt=CONTROL_DT/128, check_dt=CONTROL_DT/256,
        uncertainty_tolerance=1e-5 if args.dtype == 'float32' else 1e-10,
        memory_scope='Isolated .1-time-unit (or --benchmark-duration) seeded-policy BPTT worker; CPU RSS includes runtime and noise.',
        timing_scope='Eager, no compilation; accuracy forward excludes noise/health/transfers, synchronizes each CUDA step. BPTT synchronizes window boundaries.',
        policy='Fixed seeded 150-32-20 tanh MLP; no training; force outputs 0.05*tanh(raw).')
    rows = []
    for delta in args.splittings:
        for scenario in args.scenarios:
            print(f'Delta={delta:g} {scenario}: float64 reference and halving', flush=True)
            requested_dtype = args.dtype
            args.dtype = 'float64'
            ref_row, reference = run_accuracy('platen', 128, scenario, delta, args)
            check_row, check = run_accuracy('platen', 256, scenario, delta, args)
            args.dtype = requested_dtype
            compare(ref_row, reference, check)
            ref_row['role'], check_row['role'] = 'reference_halving_check', 'finer_reference'
            reference_ok = (ref_row['status'] == check_row['status'] == 'ok'
                            and len(reference) == len(check) == interval_count(args.duration)+1)
            rows.extend((ref_row, check_row))
            save_results(output, metadata, rows)
            # Keep the reference trajectory for audit/reanalysis without storing fine-step paths.
            np.savez_compressed(output/f'reference_delta{delta:g}_{scenario}.npz',
                                reference=reference, finer=check,
                                times=np.arange(len(reference))*CONTROL_DT)
            for method in METHODS:
                for level in args.levels:
                    row, history = run_accuracy(method, 2**level, scenario, delta, args)
                    row['role'] = 'candidate'
                    compare(row, history, reference)
                    row['reference_valid'] = reference_ok
                    row['reference_trajectory_mean_rms'] = ref_row['trajectory_mean_rms']
                    row['reference_covariance_max_abs'] = ref_row['covariance_max_abs']
                    row['trajectory_error_above_reference_floor'] = bool(
                        row['trajectory_mean_rms'] > 5 * ref_row['trajectory_mean_rms'])
                    if scenario == 'seeded_policy' and not args.skip_benchmark:
                        row.update(benchmark(method, 2**level, delta, args))
                    rows.append(row)
                    save_results(output, metadata, rows)
                    print(f"  {method:15s} dt={row['dt']:.8g} {row['status']:18s} "
                          f"path={row['trajectory_mean_rms']:.3g} cov={row['covariance_max_abs']:.3g} "
                          f"forward={row['forward_seconds']:.2f}s", flush=True)
    write_report(output, metadata, rows)
    print(f'Results: {output}', flush=True)


if __name__ == '__main__':
    main()

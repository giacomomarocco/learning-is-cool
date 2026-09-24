#!/usr/bin/env python3
"""Train residual LQR + parametric array feedback on a CPU or CUDA server.

Only --smoke is intended for local execution. See docs/array_training.md.
Normal training requires an explicit batch, iteration budget and fresh output.
Importing this module does not launch any work.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
import tempfile
import time

import torch

from learn_to_cool.array_training import (
    ControllerInterval, TrainingConfig, TrainingSession, integer_steps,
    read_checkpoint, rollout,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true', help='CPU eager only: batch 2, T=0.02, two updates')
    mode.add_argument('--dry-run', action='store_true', help='Print configuration; no model, output files or rollout')
    mode.add_argument('--benchmark', action='store_true', help='REMOTE: compare eager and compiled forward/backward')
    mode.add_argument('--evaluate', type=Path, metavar='CHECKPOINT', help='REMOTE: paired evaluation versus LQR')
    p.add_argument('--resume', type=Path, help='Resume checkpoint.pt in its existing run directory')
    p.add_argument('--output', type=Path, help='Fresh run directory; existing directory only with --resume')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--base-iterations', type=int, help='K gives 20K, 5K, K updates at T=5,20,100')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--compile', action='store_true', help='Compile controller intervals (training or evaluation)')
    p.add_argument('--seed', type=int)
    p.add_argument('--g-fb', type=float)
    p.add_argument('--dt', type=float)
    p.add_argument('--dtype', choices=('float32', 'float64'))
    p.add_argument('--threads', type=int, default=1, help='PyTorch intra-op CPU threads')
    p.add_argument('--duration', type=float, help='Benchmark/evaluation duration only; defaults 1/100 respectively')
    p.add_argument('--repeats', type=int, default=3, help='Warmed benchmark trials per execution mode')
    return p


def make_config(args):
    if args.smoke:
        return TrainingConfig(batch_size=2, base_iterations=1, smoke=True)
    values = {name: getattr(args, name) for name in
              ('seed', 'g_fb', 'dt', 'dtype') if getattr(args, name) is not None}
    return TrainingConfig(batch_size=args.batch_size,
                          base_iterations=args.base_iterations if args.base_iterations is not None else 1,
                          **values)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def describe(config):
    return {
        'config': asdict(config),
        'network': [15 * config.grid_size**2, config.hidden_width,
                    config.hidden_width, 4 * config.grid_size],
        'schedule': [{'duration': duration, 'iterations': count,
                      'controller_intervals': integer_steps(duration, config.control_dt),
                      'integration_steps': integer_steps(duration, config.dt),
                      'simulated_time_per_trajectory_in_stage': duration * count}
                     for duration, count in config.schedule],
        'state_order': ['xc', 'pc', 'Vxx', 'Vpp', 'Cxp'],
        'normalization': {'mean_scale': (2 * config.initial_occupation + 1)**0.5,
                          'covariance_scale': 2 * config.initial_occupation + 1},
        'costs': 'time-averaged centroid energy and physical squared cold forces, per particle',
        'microbatching': False, 'time_checkpointing': False,
    }


def progress_callback(label):
    last = time.perf_counter()

    def report(done, total):
        nonlocal last
        now = time.perf_counter()
        if now - last >= 30:
            print(f'{label}: {done}/{total} controller intervals ({100*done/total:.1f}%)', flush=True)
            last = now
    return report


def train(args, config, output):
    if args.resume:
        if args.resume.resolve() != (output / 'checkpoint.pt').resolve():
            raise ValueError('--resume must point to OUTPUT/checkpoint.pt')
        session = TrainingSession.resume(args.resume, args.device, args.compile)
    else:
        output.mkdir(parents=True, exist_ok=False)
        session = TrainingSession(config, args.device, args.compile)
        write_json(output / 'config.json', describe(config))
        session.save(output / 'checkpoint.pt')
    metadata = {'event': 'resume' if args.resume else 'start',
                'global_step': session.global_step, 'device': str(session.device),
                'compiled': args.compile, 'torch_version': str(torch.__version__),
                'threads': torch.get_num_threads()}
    print(json.dumps(metadata), flush=True)
    with (output / 'metrics.jsonl').open('a') as log:
        log.write(json.dumps(metadata) + '\n')
        log.flush()
        try:
            while not session.complete:
                started = time.perf_counter()
                record = session.step(progress_callback(f'update {session.global_step + 1}'))
                record['update_seconds'] = time.perf_counter() - started
                session.save(output / 'checkpoint.pt')
                record['event'] = 'update'
                line = json.dumps(record, allow_nan=False)
                log.write(line + '\n')
                log.flush()
                print(line, flush=True)
        except (Exception, KeyboardInterrupt) as error:
            log.write(json.dumps({'event': 'interrupted', 'reason': str(error),
                                  'last_saved_checkpoint': str(output / 'checkpoint.pt')}) + '\n')
            log.flush()
            raise
        log.write(json.dumps({'event': 'complete', 'global_step': session.global_step}) + '\n')
    print(f'Completed curriculum; checkpoint: {output / "checkpoint.pt"}', flush=True)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def benchmark_trial(session, duration, seed):
    session.optimizer.zero_grad(set_to_none=True)
    generator = torch.Generator(device=session.device).manual_seed(seed)
    synchronize(session.device)
    if session.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(session.device)
    start = time.perf_counter()
    result = rollout(session.model, session.interval, duration, session.config.batch_size,
                     generator=generator)
    loss = result.loss
    synchronize(session.device)
    forward_end = time.perf_counter()
    loss.backward()
    synchronize(session.device)
    end = time.perf_counter()
    gradients = torch.cat([p.grad.detach().flatten().cpu() for p in session.policy.parameters()])
    if not bool(torch.isfinite(gradients).all()):
        raise FloatingPointError('Nonfinite benchmark gradients')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = {'forward_seconds': forward_end - start, 'backward_seconds': end - forward_end,
              'total_seconds': end - start, 'objective': loss.item(),
              'process_lifetime_peak_rss_bytes': rss * (1 if sys.platform == 'darwin' else 1024),
              'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(session.device)
              if session.device.type == 'cuda' else None}
    return record, result.costs.detach().cpu(), gradients


def benchmark(args, config):
    args.output.mkdir(parents=True, exist_ok=False)
    duration = args.duration if args.duration is not None else 1.0
    report = {'configuration': describe(config), 'duration': duration, 'device': args.device,
              'torch_version': str(torch.__version__), 'threads': torch.get_num_threads(),
              'notes': ['No optimizer updates; identical seeded policy and Wiener paths.',
                        'First call includes cold execution/compilation; warmed trials are separate.',
                        'CPU RSS is a cumulative process high-water mark, not isolated per mode.',
                        'Times include noise generation and covariance checks; exclude gradient copies.'],
              'modes': {}}
    reference_costs = reference_gradients = None
    for compiled in (False, True):
        label = 'compiled' if compiled else 'eager'
        print(f'Benchmarking {label}; first forward/backward may compile.', flush=True)
        session = TrainingSession(config, args.device, compiled)
        first, costs, gradients = benchmark_trial(session, duration, config.seed + 1000)
        if not compiled:
            reference_costs, reference_gradients = costs, gradients
        else:
            torch.testing.assert_close(costs, reference_costs, rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(gradients, reference_gradients, rtol=1e-4, atol=1e-6)
        # An extra warmup also exposes specialization changes before timed trials.
        benchmark_trial(session, duration, config.seed + 1000)
        trials = [benchmark_trial(session, duration, config.seed + 1000)[0]
                  for _ in range(args.repeats)]
        report['modes'][label] = {'first_call': first, 'warmed_trials': trials}
        write_json(args.output / 'benchmark.json', report)
        del session
    report['matched_loss_and_gradient_check'] = 'passed'
    write_json(args.output / 'benchmark.json', report)
    print(json.dumps(report, indent=2), flush=True)


def mean_sem(samples):
    samples = samples.detach().cpu()
    return {'mean': samples.mean(0).tolist(),
            'sem': (samples.std(0, unbiased=True) / samples.shape[0]**0.5).tolist()
            if samples.shape[0] > 1 else None}


def evaluate(args):
    payload = read_checkpoint(args.evaluate)
    config = TrainingConfig(**payload['config'])
    duration = args.duration if args.duration is not None else 100.0
    batch = args.batch_size if args.batch_size is not None else config.batch_size
    if batch < 1:
        raise ValueError('batch-size must be positive')
    integer_steps(duration, config.control_dt)
    args.output.mkdir(parents=True, exist_ok=False)
    session = TrainingSession(config, args.device)
    session.model.load_state_dict(payload['model'])
    session.policy.load_state_dict(payload['policy'])
    session.policy.eval()
    seed = args.seed if args.seed is not None else config.seed + 100000
    report = {'checkpoint': str(args.evaluate), 'checkpoint_global_step': payload['global_step'],
              'configuration': describe(config), 'duration': duration, 'batch_size': batch,
              'evaluation_seed': seed, 'device': args.device, 'compiled': args.compile,
              'torch_version': str(torch.__version__), 'controllers': {},
              'sem_convention': 'Independent trajectories; average sites within each trajectory.'}
    values = {}
    with torch.no_grad():
        for label, policy in (('learned', session.policy), ('lqr', None)):
            interval = ControllerInterval(session.model, policy)
            if args.compile:
                interval = torch.compile(interval, fullgraph=True)
            generator = torch.Generator(device=session.device).manual_seed(seed)
            result = rollout(session.model, interval, duration, batch, generator=generator,
                             progress=progress_callback(label))
            n_xyz = session.model.occupations(result.state).mean((1, 2))
            values[label] = (result.costs, n_xyz)
            report['controllers'][label] = {
                'centroid_cost': mean_sem(result.costs[:, 0]),
                'force_cost': mean_sem(result.costs[:, 1]),
                'objective': mean_sem(result.costs.sum(1)),
                'terminal_n_xyz': mean_sem(n_xyz),
                'health': {key: value.item() for key, value in result.health.items()},
            }
        difference = values['learned'][0] - values['lqr'][0]
        report['paired_learned_minus_lqr'] = {
            'centroid_cost': mean_sem(difference[:, 0]),
            'force_cost': mean_sem(difference[:, 1]),
            'objective': mean_sem(difference.sum(1)),
            'terminal_n_xyz': mean_sem(values['learned'][1] - values['lqr'][1]),
        }
    write_json(args.output / 'evaluation.json', report)
    print(json.dumps(report, indent=2), flush=True)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.threads < 1 or args.repeats < 1:
        p.error('threads and repeats must be positive')
    if args.duration is not None and not (args.benchmark or args.evaluate):
        p.error('--duration applies only to benchmark/evaluation; training uses the fixed curriculum')
    if args.resume:
        if args.smoke or args.benchmark or args.evaluate or args.dry_run:
            p.error('--resume is only for normal training')
        if any(getattr(args, key) is not None for key in
               ('batch_size', 'base_iterations', 'seed', 'g_fb', 'dt', 'dtype')):
            p.error('Resume reads training configuration from the checkpoint; do not override it')
    if args.smoke:
        if args.compile or args.device != 'cpu' or any(getattr(args, key) is not None for key in
                ('batch_size', 'base_iterations', 'seed', 'g_fb', 'dt', 'dtype')):
            p.error('--smoke uses fixed tiny CPU/eager settings; only output and threads may be set')
    elif args.output is None and not args.dry_run:
        p.error('--output is required')
    if args.evaluate and any(getattr(args, key) is not None for key in
                             ('base_iterations', 'g_fb', 'dt', 'dtype')):
        p.error('Evaluation reads physical parameters and dtype from its checkpoint')
    if not (args.smoke or args.resume or args.evaluate):
        if args.batch_size is None:
            p.error('--batch-size is required')
        if not args.benchmark and args.base_iterations is None:
            p.error('--base-iterations is required')
    try:
        config = None if args.resume or args.evaluate else make_config(args)
        if args.dry_run:
            print(json.dumps(describe(config), indent=2))
            return
        if args.device == 'cuda' and not torch.cuda.is_available():
            p.error('CUDA was requested but is unavailable; choose a CUDA server or --device cpu')
        torch.set_num_threads(args.threads)
        if args.benchmark:
            integer_steps(args.duration if args.duration is not None else 1.0, config.control_dt)
            benchmark(args, config)
        elif args.evaluate:
            evaluate(args)
        elif args.smoke and args.output is None:
            with tempfile.TemporaryDirectory(prefix='array-feedback-smoke-') as temporary:
                train(args, config, Path(temporary) / 'run')
            print('Temporary smoke outputs removed; use --output to retain them.', flush=True)
        else:
            train(args, config, args.output)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        p.error(str(error))


if __name__ == '__main__':
    main()

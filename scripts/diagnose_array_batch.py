#!/usr/bin/env python3
"""REMOTE fixed-policy gradient noise, throughput, and memory diagnostics.

No optimizer steps, clipping, or physics changes. Use --smoke locally only.
"""

import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
import time

import torch

from learn_to_cool.array_training import TrainingConfig, TrainingSession, integer_steps, read_checkpoint, rollout
from train_array_feedback import progress_callback, synchronize, write_json


def summarize_gradients(gradients, batch_size):
    """Independent batch-mean gradients; unbiased noise and squared-mean estimates.

    S estimates tr Cov(g_B). ||mean(g_B)||² - S/R estimates ||E g_B||².
    B*S / signal estimates the simple gradient noise scale (not an Adam optimum).
    Nonpositive signal estimates are preserved and their ratios left undefined.
    """
    g = gradients.detach().cpu().double()
    if g.ndim != 2 or len(g) < 2 or batch_size < 1 or not bool(torch.isfinite(g).all()):
        raise ValueError('Need at least two finite gradient vectors and a positive batch size')
    r = len(g)
    mean = g.mean(0)
    noise = ((g - mean).square().sum() / (r - 1)).item()
    mean_squared = mean.square().sum().item()
    signal = mean_squared - noise / r
    norms = g.norm(dim=1)
    valid = norms > 0
    cosine = None
    if valid.sum() >= 2:
        unit = g[valid] / norms[valid, None]
        i, j = torch.triu_indices(len(unit), len(unit), offset=1)
        pairwise = (unit @ unit.T)[i, j].clamp(-1, 1)
        cosine = {'mean': pairwise.mean().item(), 'median': pairwise.median().item(),
                  'negative_fraction': (pairwise < 0).double().mean().item()}
    # Delete-one jackknife uncertainty for the unbiased squared signal.
    # Off-diagonal dot-product mean is algebraically the same estimator.
    signal_se = None
    if r > 2:
        total = g.sum(0)
        sum_squares = g.square().sum()
        deleted = ((total - g).square().sum(1) - (sum_squares - g.square().sum(1))) / ((r - 1) * (r - 2))
        signal_se = (((r - 1) / r * (deleted - deleted.mean()).square().sum()).sqrt().item())
    return {'replicates': r, 'batch_size': batch_size,
            'batch_gradient_noise_trace': noise, 'mean_gradient_norm': math.sqrt(mean_squared),
            'signal_squared_unbiased': signal, 'signal_squared_jackknife_se': signal_se,
            'relative_rms_noise': math.sqrt(noise / signal) if signal > 0 else None,
            'simple_noise_scale': batch_size * noise / signal if signal > 0 else None,
            'gradient_norm_mean': norms.mean().item(), 'gradient_norm_min': norms.min().item(),
            'gradient_norm_max': norms.max().item(), 'pairwise_cosine': cosine}


def parameter_groups(policy):
    masks = {'all': [], 'hidden': [], 'cold_output': [], 'parametric_output': []}
    offset = 0
    output_names = {f'net.{len(policy.net)-1}.weight', f'net.{len(policy.net)-1}.bias'}
    for name, value in policy.named_parameters():
        indices = torch.arange(offset, offset + value.numel())
        masks['all'].append(indices)
        if name in output_names:
            indices = indices.reshape(value.shape)
            half = value.shape[0] // 2
            masks['cold_output'].append(indices[:half].flatten())
            masks['parametric_output'].append(indices[half:].flatten())
        else:
            masks['hidden'].append(indices)
        offset += value.numel()
    return {key: torch.cat(value) for key, value in masks.items()}


def policy_digest(policy):
    digest = hashlib.sha256()
    for key, value in policy.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gradient_trial(session, duration, seed):
    session.optimizer.zero_grad(set_to_none=True)
    generator = torch.Generator(device=session.device).manual_seed(seed)
    synchronize(session.device)
    if session.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(session.device)
    start = time.perf_counter()
    result = rollout(session.model, session.interval, duration, session.config.batch_size,
                     generator=generator, progress=progress_callback(f'T={duration} seed={seed}'))
    loss = result.loss
    synchronize(session.device)
    forward_end = time.perf_counter()
    loss.backward()
    synchronize(session.device)
    end = time.perf_counter()
    gradients = torch.cat([p.grad.detach().flatten().cpu() for p in session.policy.parameters()])
    if not bool(torch.isfinite(gradients).all()) or not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite loss or gradients')
    record = {'seed': seed, 'objective': loss.item(),
              'forward_seconds': forward_end-start, 'backward_seconds': end-forward_end,
              'total_seconds': end-start,
              'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(session.device)
              if session.device.type == 'cuda' else None,
              'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(session.device)
              if session.device.type == 'cuda' else None,
              'health': {key: value.item() for key, value in result.health.items()}}
    return record, gradients


def diagnose(args, output):
    payload = read_checkpoint(args.checkpoint) if args.checkpoint else None
    config = TrainingConfig(**payload['config']) if payload else TrainingConfig(args.batch_size, 1)
    config = replace(config, batch_size=args.batch_size)
    for duration in args.durations:
        integer_steps(duration, config.control_dt)
    output.mkdir(parents=True, exist_ok=False)
    session = TrainingSession(config, args.device, args.compile)
    if payload:
        session.model.load_state_dict(payload['model'])
        session.policy.load_state_dict(payload['policy'])
    digest = policy_digest(session.policy)
    masks = parameter_groups(session.policy)
    report = {'configuration': asdict(config), 'checkpoint': str(args.checkpoint) if payload else None,
              'checkpoint_global_step': payload['global_step'] if payload else 0,
              'policy_sha256': digest, 'seed_base': args.seed, 'device': args.device,
              'compiled': args.compile, 'torch_version': str(torch.__version__),
              'cuda_version': torch.version.cuda, 'threads': torch.get_num_threads(),
              'gpu_name': torch.cuda.get_device_name(session.device) if args.device == 'cuda' else None,
              'gpu_total_bytes': torch.cuda.get_device_properties(session.device).total_memory
              if args.device == 'cuda' else None,
              'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
              'notes': ['Independent noise seeds per replicate and batch size; weights fixed.',
                        'Same seeds at initialization/checkpoint allow paired policy comparisons.',
                        'Raw total-objective gradients before clipping; no optimizer updates.',
                        'Noise scale is diagnostic, not an optimal batch size for Adam.',
                        'Output groups select head parameters; hidden parameters serve both controls.',
                        'Short compilation warmup excluded; full-duration trials include RNG and health checks.',
                        'GPU peaks exclude Adam moment allocation; reserved memory is cumulative allocator state.'],
              'cases': []}
    write_json(output/'diagnostic.json', report)
    print(f'Warming B={args.batch_size}, fixed policy {digest[:12]}', flush=True)
    report['warmup'], _ = gradient_trial(session, 2*config.control_dt, args.seed-1)
    for index, (duration, repeats) in enumerate(zip(args.durations, args.repeats)):
        case = {'duration': duration, 'requested_replicates': repeats, 'status': 'running', 'trials': []}
        report['cases'].append(case)
        gradients = []
        try:
            for replicate in range(repeats):
                seed = args.seed + args.batch_size*1000000 + index*10000 + replicate
                trial, grad = gradient_trial(session, duration, seed)
                case['trials'].append(trial)
                gradients.append(grad)
                print(json.dumps({'duration': duration, 'replicate': replicate+1, **trial}), flush=True)
                write_json(output/'diagnostic.json', report)
        except torch.cuda.OutOfMemoryError as error:
            case.update(status='out_of_memory', error=str(error))
            session.optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        if gradients:
            samples = torch.stack(gradients)
            torch.save({'gradients': samples, 'parameter_names': [n for n, _ in session.policy.named_parameters()],
                        'groups': masks, 'seeds': [t['seed'] for t in case['trials']]},
                       output/f'gradients_t{duration:g}.pt')
            if len(samples) >= 2:
                case['groups'] = {name: summarize_gradients(samples[:, mask], args.batch_size)
                                  for name, mask in masks.items()}
                case['fraction_would_clip_at_one'] = (samples.double().norm(dim=1) > 1).double().mean().item()
            seconds = [t['total_seconds'] for t in case['trials']]
            case['median_seconds'] = statistics.median(seconds)
            case['trajectories_per_second'] = args.batch_size/case['median_seconds']
            case['max_cuda_peak_allocated_bytes'] = max(t['cuda_peak_allocated_bytes'] or 0 for t in case['trials'])
        if case['status'] == 'running':
            case['status'] = 'complete'
        assert policy_digest(session.policy) == digest, 'Diagnostic changed policy weights'
        write_json(output/'diagnostic.json', report)
        print(json.dumps({'duration': duration, 'status': case['status'], 'groups': case.get('groups')}), flush=True)
    report['weights_unchanged'] = True
    write_json(output/'diagnostic.json', report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='Tiny CPU/eager diagnostic, temporary output by default')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--durations', type=float, nargs='+')
    parser.add_argument('--repeats', type=int, nargs='+', help='One count for all durations, or one per duration')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--seed', type=int, default=30260924)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.smoke:
        if args.checkpoint or args.compile or args.device != 'cpu' or any(
                value is not None for value in (args.batch_size, args.durations, args.repeats)):
            parser.error('--smoke requires fixed CPU/eager settings')
        args.batch_size, args.durations, args.repeats = 2, [.02], [3]
    elif any(value is None for value in (args.output, args.batch_size, args.durations, args.repeats)):
        parser.error('Remote diagnostics require --output, --batch-size, --durations, --repeats')
    if args.batch_size < 1 or args.threads < 1 or any(r < 2 for r in args.repeats):
        parser.error('Positive batch/threads and at least two independent repeats required')
    if len(args.repeats) == 1:
        args.repeats *= len(args.durations)
    if len(args.repeats) != len(args.durations):
        parser.error('Provide one repeat count or one per duration')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    torch.set_num_threads(args.threads)
    try:
        if args.output is None:
            with tempfile.TemporaryDirectory(prefix='array-batch-smoke-') as temporary:
                diagnose(args, Path(temporary)/'run')
        else:
            diagnose(args, args.output)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()

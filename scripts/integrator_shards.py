#!/usr/bin/env python3
"""Run independent extended-profile cases inside an interactive Slurm allocation.

Launch with srun, one rank per CPU case or GPU. Each rank processes its assigned
cases sequentially. After srun completes, invoke --merge to validate coverage
and generate the combined report. Never submit work from this script itself.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from compare_integrators import METHODS, SCENARIOS, save_results, write_report

SPLITTINGS = (.01, .1, 1.)
CASES = tuple((delta, scenario) for delta in SPLITTINGS for scenario in SCENARIOS)


def case_directory(root, delta, scenario):
    return root / f'delta{delta:g}_{scenario}'


def merge(root):
    rows, metadata = [], None
    invariant = ('batch', 'duration', 'dtype', 'device', 'seed', 'levels', 'threads',
                 'benchmark_duration', 'skip_benchmark', 'torch_version', 'control_dt',
                 'frequency_formula', 'gamma0', 'eta', 'initial_occupation')
    for delta, scenario in CASES:
        path = case_directory(root, delta, scenario) / 'results.json'
        saved = json.loads(path.read_text())
        current = saved['metadata']
        if metadata is None:
            metadata = current.copy()
        if any(current[key] != metadata[key] for key in invariant):
            raise ValueError(f'Inconsistent configuration: {path}')
        expected = {('candidate', method, 2**level) for method in METHODS
                    for level in current['levels']}
        expected |= {('reference_halving_check', 'platen', 128), ('finer_reference', 'platen', 256)}
        actual = {(r['role'], r['method'], r['substeps']) for r in saved['results']}
        if actual != expected or len(saved['results']) != len(expected):
            raise ValueError(f'Incomplete or duplicate results: {path}')
        if any(r['delta'] != delta or r['scenario'] != scenario for r in saved['results']):
            raise ValueError(f'Incorrect case labels: {path}')
        rows.extend(saved['results'])
    metadata.update(splittings=list(SPLITTINGS), scenarios=list(SCENARIOS),
                    output=str(root/'summary'), execution='Independent Slurm ranks in an interactive allocation')
    destination = root/'summary'
    if (destination/'results.json').exists():
        raise FileExistsError(f'Summary already exists: {destination}')
    save_results(destination, metadata, rows)
    write_report(destination, metadata, rows)
    print(f'Merged {len(rows)} verified result records: {destination}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--dtype', choices=('float32', 'float64'), default='float64')
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='Append logs and resume each case using its saved configuration')
    args = parser.parse_args()
    if args.merge:
        merge(args.output)
        return
    if 'SLURM_JOB_ID' not in os.environ:
        parser.error('Run inside an interactive Slurm allocation')
    rank, workers = int(os.environ.get('SLURM_PROCID', 0)), int(os.environ.get('SLURM_NTASKS', 1))
    if not 0 <= rank < workers <= len(CASES):
        parser.error('Use 1..12 Slurm tasks with valid rank IDs')
    args.output.mkdir(parents=True, exist_ok=True)
    for index in range(rank, len(CASES), workers):
        delta, scenario = CASES[index]
        destination = case_directory(args.output, delta, scenario)
        with (args.output/f'{destination.name}.log').open('a' if args.resume else 'x') as log:
            command = [sys.executable, str(Path(__file__).with_name('compare_integrators.py')),
                       '--profile', 'extended', '--device', args.device, '--dtype', args.dtype,
                       '--splittings', str(delta), '--scenarios', scenario,
                       '--output', str(destination)]
            if args.resume:
                command = [sys.executable, str(Path(__file__).with_name('compare_integrators.py')),
                           '--resume-from', str(destination)]
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(f'[rank {rank} {destination.name}] {line}', end='', flush=True)
                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, command)


if __name__ == '__main__':
    main()

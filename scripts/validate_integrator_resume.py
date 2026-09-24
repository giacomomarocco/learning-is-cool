#!/usr/bin/env python3
"""Regression checks for interrupted sweeps and TeX-independent reports."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
from matplotlib.texmanager import TexManager

import compare_integrators as diagnostic


def main():
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)/'case'
        command = ['compare_integrators.py', '--batch', '2', '--duration', '.02',
                   '--levels', '0', '1', '--scenarios', 'seeded_policy',
                   '--device', 'cpu', '--skip-benchmark', '--disk-history',
                   '--output', str(output)]
        # Reproduce a site configuration enabling TeX; any external TeX call
        # would fail even if the developer happens to have TeX installed.
        with matplotlib.rc_context({'text.usetex': True}), \
                patch.object(TexManager, 'make_dvi', side_effect=AssertionError('External TeX called')):
            with patch.object(sys, 'argv', command):
                diagnostic.main()
            assert matplotlib.rcParams['text.usetex'] is True
            original = json.loads((output/'results.json').read_text())
            reference = output/'reference_delta1_seeded_policy.npz'
            reference_hash = hashlib.sha256(reference.read_bytes()).digest()
            partial = original | {'results': original['results'][:-2]}
            (output/'results.json').write_text(json.dumps(partial))
            actual_runs = []
            run_accuracy = diagnostic.run_accuracy
            def counted(method, substeps, scenario, delta, args, *rest, **kwargs):
                if args.duration == .02:
                    actual_runs.append((method, substeps))
                return run_accuracy(method, substeps, scenario, delta, args, *rest, **kwargs)
            with patch.object(diagnostic, 'run_accuracy', side_effect=counted):
                diagnostic.resume_from(output)
            assert actual_runs == [('platen', 1), ('platen', 2)]
            resumed = json.loads((output/'results.json').read_text())
            assert resumed['results'][:4] == original['results'][:4]
            timing_fields = {'forward_seconds', 'diagnostic_wall_seconds'}
            for got, expected in zip(resumed['results'], original['results']):
                assert {k:v for k,v in got.items() if k not in timing_fields} == \
                       {k:v for k,v in expected.items() if k not in timing_fields}
            assert hashlib.sha256(reference.read_bytes()).digest() == reference_hash
            assert not list(output.glob('*.tmp'))
            assert (output/'convergence_delta1.png').stat().st_size > 0
            with patch.object(diagnostic, 'run_accuracy', side_effect=AssertionError('Unexpected rerun')):
                diagnostic.resume_from(output)
            invalid = resumed | {'results': resumed['results'] + resumed['results'][:1]}
            (output/'results.json').write_text(json.dumps(invalid))
            try:
                diagnostic.resume_from(output)
            except ValueError as error:
                assert 'Duplicate' in str(error)
            else:
                raise AssertionError('Duplicate records were accepted')
    print('Resume preserves saved records/references and exactly reproduces missing numerical results.')
    print('Completed cases require no rerun; duplicate records are rejected; reports never call TeX.')


if __name__ == '__main__':
    main()

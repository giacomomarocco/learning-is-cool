#!/usr/bin/env python3
"""Tiny CPU/eager checks only. No compilation, benchmarks, or long trajectories.

uv run python -B scripts/validate_array_training.py
"""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from scipy.linalg import solve_continuous_lyapunov
import torch

from learn_to_cool.array_training import (
    ArrayModel, ControllerInterval, ResidualPolicy, TrainingConfig,
    TrainingSession, integer_steps, lqr_matrices, read_checkpoint, rollout,
)
from learn_to_cool.gaussian_integrators import action_controls
import train_array_feedback as cli


def physics_and_lqr():
    cfg = TrainingConfig(2, 1, dtype='float64', smoke=True)
    model = ArrayModel(cfg)
    omega = model.omega.numpy()
    expected = np.arange(5) * .1 + 1
    for row in range(5):
        np.testing.assert_allclose(np.sort(omega[row, :, 2]), expected)
        np.testing.assert_allclose(np.sort(omega[:, row, 2]), expected)
    np.testing.assert_allclose(omega[..., 0], 4.5 * omega[..., 2])
    np.testing.assert_allclose(omega[..., 1], 4.1 * omega[..., 2])
    initial = model.initial_state(2)
    torch.testing.assert_close(model.occupations(initial), torch.full_like(initial[0], 5))
    torch.testing.assert_close(model.centroid_cost(initial), torch.zeros(2, dtype=torch.float64))

    # A single displacement per row/column: independent physical force formula.
    displacement = torch.tensor([.03, -.02, .01, .04, -.01], dtype=torch.float64)
    raw = torch.zeros((2, 20), dtype=torch.float64)
    raw[:, :5] = math.sqrt(2 * omega[0, 0, 0]) * displacement
    raw[:, 5:10] = math.sqrt(2 * omega[0, 0, 1]) * displacement
    force, modulation = action_controls(raw, model.bx, model.by, .05)
    physical_x = model.omega[..., 0].square() * displacement[:, None]
    physical_y = model.omega[..., 1].square() * displacement[None, :]
    expected_cost = (physical_x.square().sum() + physical_y.square().sum()) / 25
    torch.testing.assert_close(model.force_cost(force), expected_cost.expand(2))
    torch.testing.assert_close(force[..., 2], torch.zeros_like(force[..., 2]))
    torch.testing.assert_close(modulation, torch.zeros_like(modulation))
    # Equal-frequency and single-particle physical-unit reductions.
    for n in (1, 5):
        w, d = 3.0, .2
        _, _, _, r = lqr_matrices(np.full(n, w), np.full(n, w), n*n, 1.)
        assert math.isclose(r.item() * (math.sqrt(2*w)*d)**2, w**4 * d**2 / n)
    # All saturated channels give the combined bound, with no z cold force.
    raw[:, 10:] = 100
    _, modulation = action_controls(raw, model.bx, model.by, .05)
    torch.testing.assert_close(modulation, torch.full_like(modulation, .1))

    # Independent optimality check: closed-loop Lyapunov value must recover K.
    for direction, gains, coupling in ((0, model.kx, model.bx), (1, model.ky, model.by)):
        for group in range(5):
            w = omega[group, :, 0] if direction == 0 else omega[:, group, 1]
            b_values = coupling[group].numpy() if direction == 0 else coupling[:, group].numpy()
            a, b, q, r = lqr_matrices(w, b_values, 25, cfg.g_fb)
            k = gains[group].numpy().reshape(1, 10)
            closed = a - b @ k
            assert np.linalg.eigvals(closed).real.max() < 0
            p = solve_continuous_lyapunov(closed.T, -(q + k.T @ r @ k))
            np.testing.assert_allclose(k, np.linalg.solve(r, b.T @ p), rtol=1e-9, atol=1e-10)

    # Nonzero, asymmetric state checks row/column gathering and action signs.
    generator = torch.Generator().manual_seed(32)
    x = torch.randn(initial[0].shape, generator=generator, dtype=torch.float64)
    p = torch.randn(x.shape, generator=generator, dtype=torch.float64)
    state = (x, p, *initial[2:])
    commands = model.lqr_commands(state)
    for group in range(5):
        for batch in range(2):
            row_state = torch.stack((x[batch, group, :, 0], p[batch, group, :, 0]), dim=-1)
            col_state = torch.stack((x[batch, :, group, 1], p[batch, :, group, 1]), dim=-1)
            torch.testing.assert_close(commands[batch, group], -(row_state * model.kx[group]).sum())
            torch.testing.assert_close(commands[batch, 5+group], -(col_state * model.ky[group]).sum())
    policy = ResidualPolicy(cfg).double()
    torch.testing.assert_close(policy(state), torch.zeros_like(commands))
    noise = torch.randn((16, 2, 5, 5, 3), generator=generator, dtype=torch.float64) * math.sqrt(cfg.dt)
    base = ControllerInterval(model, None)(state, noise)
    learned = ControllerInterval(model, policy)(state, noise)
    for lhs, rhs in zip(base[0], learned[0]):
        torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
    torch.testing.assert_close(base[1], learned[1], rtol=0, atol=0)
    print('Physics, force weights, LQR optimality and zero-residual baseline: passed')


def feedback_gradient():
    cfg = TrainingConfig(2, 1, dtype='float64', smoke=True)
    session = TrainingSession(cfg)
    bias = session.policy.net[-1].bias
    with torch.no_grad():
        bias.copy_(torch.linspace(-.01, .01, 20, dtype=torch.float64))
    noise = torch.randn((2, 16, 2, 5, 5, 3), dtype=torch.float64,
                        generator=torch.Generator().manual_seed(33)) * math.sqrt(cfg.dt)

    def loss():
        return rollout(session.model, session.interval, .02, 2, noise=noise).loss

    derivative = torch.autograd.grad(loss(), bias)[0][10].item()
    original, h = bias[10].item(), 1e-6
    with torch.no_grad():
        bias[10] = original + h
        plus = loss().item()
        bias[10] = original - h
        minus = loss().item()
        bias[10] = original
    finite_difference = (plus-minus)/(2*h)
    assert abs(derivative) > 1e-8
    assert math.isclose(derivative, finite_difference, rel_tol=1e-5, abs_tol=1e-8)
    print(f'Fixed-noise parametric gradient: autograd={derivative:.8g}, finite difference={finite_difference:.8g}')


def resume_and_training(directory):
    cfg = TrainingConfig(2, 1, smoke=True)
    session = TrainingSession(cfg)
    before = {key: value.clone() for key, value in session.policy.state_dict().items()}
    first = session.step()  # First of only TWO optimizer updates in this script.
    assert first['global_step'] == 1 and first['health']['nonfinite_modes'] == 0
    assert first['gradient_norm_before_clipping'] > 0
    assert any(not torch.equal(before[key], value) for key, value in session.policy.state_dict().items())
    checkpoint = directory / 'checkpoint.pt'
    session.save(checkpoint)
    restored = TrainingSession.resume(checkpoint)
    assert restored.global_step == 1 and restored.stage_iteration == 1
    assert torch.equal(restored.generator.get_state(), session.generator.get_state())
    for key, value in session.policy.state_dict().items():
        torch.testing.assert_close(restored.policy.state_dict()[key], value, rtol=0, atol=0)
    for lhs, rhs in zip(session.optimizer.state.values(), restored.optimizer.state.values()):
        for key in lhs:
            torch.testing.assert_close(lhs[key], rhs[key], rtol=0, atol=0)
    # Predict the next cost/gradient without another optimizer update.
    session.optimizer.zero_grad(set_to_none=True)
    expected = rollout(session.model, session.interval, .02, 2, generator=session.generator)
    expected.loss.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(session.policy.parameters(), 1.0).item()
    second = restored.step()  # Second optimizer update; same next path after reload.
    expected_costs = expected.costs.detach().mean(0).tolist()
    assert second['centroid_cost'] == expected_costs[0]
    assert second['force_cost'] == expected_costs[1]
    assert second['gradient_norm_before_clipping'] == expected_norm
    assert restored.complete and restored.global_step == 2
    assert all(state['step'].item() == 2 for state in restored.optimizer.state.values())
    restored.save(checkpoint)
    completed = TrainingSession.resume(checkpoint)
    assert completed.complete
    assert read_checkpoint(checkpoint)['global_step'] == 2
    # No legacy checkpoint format or path is involved.
    assert sorted(p.name for p in directory.iterdir()) == ['checkpoint.pt']
    # Exercise the CLI resume path on an already completed run: zero updates.
    with redirect_stdout(io.StringIO()):
        cli.main(['--resume', str(checkpoint), '--output', str(directory)])
        cli.main(['--evaluate', str(checkpoint), '--output', str(directory/'evaluation'),
                  '--batch-size', '2', '--duration', '.02'])
    evaluation = json.loads((directory/'evaluation'/'evaluation.json').read_text())
    assert evaluation['batch_size'] == 2 and evaluation['duration'] == .02
    assert set(evaluation['controllers']) == {'learned', 'lqr'}
    assert math.isfinite(evaluation['paired_learned_minus_lqr']['objective']['mean'])
    print('Two tiny optimizer updates, checkpoint round-trip and next-path replay: passed')


def cli_and_curriculum(directory):
    script = Path(cli.__file__).resolve()
    imported = subprocess.run(
        [sys.executable, '-B', '-c',
         'import runpy, sys; runpy.run_path(sys.argv[1], run_name="import_check")', str(script)],
        cwd=directory, capture_output=True, text=True, check=True)
    assert imported.stdout == '' and list(directory.iterdir()) == []
    stream = io.StringIO()
    with redirect_stdout(stream):
        cli.main(['--dry-run', '--batch-size', '2', '--base-iterations', '3'])
    description = json.loads(stream.getvalue())
    assert [s['iterations'] for s in description['schedule']] == [60, 15, 3]
    assert [s['integration_steps'] for s in description['schedule']] == [8000, 32000, 160000]
    assert [s['simulated_time_per_trajectory_in_stage'] for s in description['schedule']] == [300, 300, 300]
    assert list(directory.iterdir()) == []
    for arguments in ([], ['--dry-run', '--batch-size', '2'],
                      ['--dry-run', '--batch-size', '2', '--base-iterations', '0'],
                      ['--smoke', '--compile'], ['--smoke', '--duration', '5'],
                      ['--smoke', '--output', str(directory)]):
        with redirect_stderr(io.StringIO()):
            try:
                cli.main(arguments)
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError(f'Should reject {arguments}')
    cfg = TrainingConfig(2, 1)
    for changes in ({'g_fb': 0}, {'dt': .003}, {'batch_size': 0}, {'eta': 2}):
        try:
            replace(cfg, **changes)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Should reject {changes}')
    try:
        integer_steps(.015, .01)
    except ValueError:
        pass
    else:
        raise AssertionError('Nonintegral controller horizon was accepted')
    print('Import/CLI safety, overwrite refusal and fixed-step curriculum counts: passed')


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='array-training-validation-') as name:
        root = Path(name)
        empty, checkpoints = root/'empty', root/'checkpoints'
        empty.mkdir()
        checkpoints.mkdir()
        cli_and_curriculum(empty)
        physics_and_lqr()
        feedback_gradient()
        resume_and_training(checkpoints)
    print('All CPU/eager smoke checks passed. Compilation and remote-scale behavior remain untested.')


if __name__ == '__main__':
    main()

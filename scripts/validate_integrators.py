#!/usr/bin/env python3
"""Small assertion-based integrator checks; no training/checkpoint side effects.

uv run scripts/validate_integrators.py
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp
import torch

from learn_to_cool.gaussian_integrators import (
    METHODS, action_controls, advance_gaussian, covariance_health, drift,
)
from learn_to_cool.torch_oscillator_env import TorchOscillatorEnv
from compare_integrators import frequencies, noise_blocks, policy, controls, model


def assert_close(a, b, **kwargs):
    torch.testing.assert_close(a, b, **kwargs)


def deterministic_check():
    # Gamma=0: exact harmonic rotation and stationary isotropic covariance.
    errors = {}
    for method in METHODS:
        errors[method] = []
        for steps in (20, 40, 80):
            x, p = torch.tensor([1.], dtype=torch.float64), torch.tensor([.3], dtype=torch.float64)
            state = (x, p, x*0+2, x*0+2, x*0)
            for _ in range(steps):
                state = advance_gaussian(state, 2., 0., 0., 0., .5, .4/steps, x*0, method)
            exact = torch.tensor([np.cos(.8)+.3*np.sin(.8), .3*np.cos(.8)-np.sin(.8)], dtype=torch.float64)
            errors[method].append(torch.linalg.vector_norm(torch.cat(state[:2])-exact).item())
            assert_close(state[2], x*0+2, rtol=0, atol=1e-14)
            assert_close(state[3], x*0+2, rtol=0, atol=1e-14)
        ratio = errors[method][-2] / errors[method][-1]
        assert (1.9 < ratio < 2.1) if method == 'euler_maruyama' else (3.8 < ratio < 4.2)
    print('Deterministic convergence:', errors)


def heating_and_numpy_check():
    omega = torch.tensor([4.5, 4.1, 1.], dtype=torch.float64)
    state = tuple(torch.tensor(v, dtype=torch.float64).expand(3).clone()
                  for v in (0., 0., 11., 11., 0.))
    # Expected one-step energy drift from covariance + Ito quadratic variation.
    a = drift(state, omega, 0., 0., .05, .5)
    g2 = 4 * .5 * .05 * (state[2]**2 + state[4]**2)
    assert_close((a[2]+a[3]+g2)/4, omega*0+.05, atol=1e-14, rtol=0)
    xi = torch.sqrt(1 + 16*.5*.05**2/omega**2)
    vx = 2/(np.sqrt(2*.5)*torch.sqrt(xi+1))
    vp = xi*vx
    c = torch.sqrt(xi-1)/(np.sqrt(.5)*torch.sqrt(xi+1))
    stationary = drift((omega*0, omega*0, vx, vp, c), omega, 0., 0., .05, .5)
    for derivative in stationary[2:]:
        assert_close(derivative, omega*0, atol=2e-13, rtol=0)
    # Independent NumPy Riccati solver, same normalization and gamma(epsilon).
    for eps in (-.1, 0., .1):
        gamma = .05 * (1+eps)
        def rhs(t, flat):
            vx, vp, c = flat.reshape(3, 3)
            w = omega.numpy()
            q = 4*.5*gamma
            return np.stack((2*w*c-q*vx**2, -2*w*(1+eps)*c+4*gamma-q*c**2,
                             w*vp-w*(1+eps)*vx-q*vx*c)).ravel()
        ref = solve_ivp(rhs, (0, .1), np.stack([s.numpy() for s in state[2:]]).ravel(),
                        rtol=1e-12, atol=1e-13).y[:, -1].reshape(3, 3)
        errors = []
        for steps in (50, 100):
            y = state
            for _ in range(steps):
                y = advance_gaussian(y, omega, 0., eps, gamma, .5, .1/steps, omega*0, 'platen')
            errors.append(np.max(np.abs(np.stack([s.numpy() for s in y[2:]])-ref)))
        assert 3.8 < errors[0]/errors[1] < 4.2, errors
        assert errors[-1] < 2e-5, errors
    print('Ito heating identity and independent NumPy covariance convergence: passed')


def weak_heating_check():
    # Propagate the exact expectation of each DISCRETE scheme, without sampling
    # noise: linear mean map M and noise vector q imply Snew=M*S*M.T+q*q.T.
    # This is a validation calculation, not an additional simulation integrator.
    results = {}
    for method in METHODS:
        errors = []
        for steps in (20, 40, 80):
            y = tuple(torch.tensor([v], dtype=torch.float64) for v in (0, 0, 11, 11, 0))
            second = torch.zeros((2, 2), dtype=torch.float64)
            h = .2/steps
            for _ in range(steps):
                zero = advance_gaussian(y, 4.5, 0., 0., .05, .5, h, y[0]*0, method)
                columns = []
                for i in range(2):
                    basis = list(y)
                    basis[i] = basis[i]+1
                    shifted = advance_gaussian(tuple(basis), 4.5, 0., 0., .05, .5, h, y[0]*0, method)
                    columns.append(torch.cat([shifted[j]-zero[j] for j in range(2)]))
                matrix = torch.stack(columns, dim=1)
                noisy = advance_gaussian(y, 4.5, 0., 0., .05, .5, h, y[0]*0+np.sqrt(h), method)
                q = torch.cat([noisy[j]-zero[j] for j in range(2)])
                second = matrix @ second @ matrix.T + torch.outer(q, q)
                y = zero
            expected_n = (second.trace()+y[2]+y[3])/4-.5
            errors.append(abs(expected_n.item()-(5+.05*.2)))
        ratio = errors[-2]/errors[-1]
        assert (1.8 < ratio < 2.2) if method == 'euler_maruyama' else (3.5 < ratio < 4.5), errors
        results[method] = errors
    print('No-feedback expected occupation convergence (no Monte Carlo error):', results)


def noise_and_gradient_check():
    fine = list(noise_blocks(2, 3, 256, 31, torch.float64, 'cpu'))
    coarse = list(noise_blocks(2, 3, 4, 31, torch.float64, 'cpu'))
    again = list(noise_blocks(2, 3, 4, 31, torch.float64, 'cpu'))
    for f, c, repeat in zip(fine, coarse, again):
        assert_close(f.reshape(4, 64, 2, 5, 5, 3).sum(1), c, rtol=0, atol=0)
        assert_close(c, repeat, rtol=0, atol=0)
    gradient_errors = {}
    for method in METHODS:
        # Nonzero means/covariances and fixed dW expose drift AND diffusion derivatives.
        q = torch.tensor([.2, -.3, 2., 3., .1, .02], dtype=torch.float64, requires_grad=True)
        def step(q):
            eps = q[5].tanh()*.1
            return torch.stack(advance_gaussian(tuple(q[:5]), 3., q[5], eps,
                                                 .05*(1+eps), .5, .007,
                                                 q.new_tensor(.12), method))
        assert torch.autograd.gradcheck(step, (q,), eps=1e-6, atol=1e-7, rtol=1e-5)
        # Paired full policy BPTT versus a central finite difference, then dt convergence.
        def loss(scale, substeps):
            state, w, bx, by = model(1., 2, torch.float64, 'cpu')
            net = policy(17, torch.float64, 'cpu')
            for dws in noise_blocks(2, 3, substeps, 33, torch.float64, 'cpu'):
                force, eps, _ = controls(state, 'seeded_policy', net, bx, by)
                force, eps = force*scale, eps*scale
                for dw in dws:
                    state = advance_gaussian(state, w, force, eps, .05*(1+eps), .5,
                                             .01/substeps, dw, method)
            return (w*(state[0]**2+state[1]**2+state[2]+state[3])/4).mean()
        grads = []
        for substeps in (1, 4, 16, 128):
            scale = torch.tensor(1., dtype=torch.float64, requires_grad=True)
            grad, = torch.autograd.grad(loss(scale, substeps), scale)
            grads.append(grad.item())
        h = 1e-4
        finite_difference = ((loss(torch.tensor(1.+h, dtype=torch.float64), 4)
                              - loss(torch.tensor(1.-h, dtype=torch.float64), 4))/(2*h)).item()
        assert abs(finite_difference-grads[1]) < 2e-7
        assert abs(grads[2]-grads[3]) < abs(grads[0]-grads[3])
        gradient_errors[method] = dict(gradients=grads, finite_difference=finite_difference)
    print('Paired noise, repeatability, autograd/finite differences, gradient convergence:', gradient_errors)


def environment_check():
    osc = SimpleNamespace(omegas=frequencies(.1), eta=.5, gamma_meas=.05)
    for method in METHODS:
        env = TorchOscillatorEnv(osc, 2, horizon=2, dt=.01, dtype=torch.float64,
                                integration_method=method, intensity_dependent_recoil=True,
                                modulation_depth=.05, pregenerate_noise=False)
        for derivative in drift(env.gaussian_state(), env.omegas, 0., 0., .05, .5)[2:]:
            assert_close(derivative, torch.zeros_like(derivative), atol=2e-12, rtol=0)
        initial = torch.zeros((2, 5, 5, 3, 2), dtype=torch.float64)
        vx = torch.full((2, 5, 5, 3), 11., dtype=torch.float64)
        covariances = (vx, vx, torch.zeros_like(vx))
        dw = torch.randn(vx.shape, generator=torch.Generator().manual_seed(55), dtype=torch.float64)*.1
        action = torch.full((2, 20), .2, dtype=torch.float64, requires_grad=True)
        env.reset(initial, covariances=covariances)
        force, eps = action_controls(action, env.B_x, env.B_y, .05)
        expected = advance_gaussian(env.gaussian_state(), env.omegas, force, eps,
                                    .05*(1+eps), .5, .01, dw, method)
        if method == 'euler_maruyama':
            # In particular dx must use OLD p, with no momentum-first h**2 term.
            old_x, old_p, old_vx, _, _ = env.gaussian_state()
            assert_close(expected[0], old_x + .01*env.omegas*old_p
                         + 2*torch.sqrt(.5*.05*(1+eps))*old_vx*dw, rtol=0, atol=0)
        env.step(action, dw)
        for got, want in zip(env.gaussian_state(), expected):
            assert_close(got, want, rtol=0, atol=0)
        env.mean_energy().sum().backward()
        assert torch.isfinite(action.grad).all() and action.grad[:, 10:].abs().sum() > 0
        first = tuple(s.clone() for s in env.gaussian_state())
        env.reset(initial, covariances=covariances, noise=torch.stack((dw, dw)))
        env.step(action)
        for got, want in zip(env.gaussian_state(), first):
            assert_close(got, want, rtol=0, atol=0)
    # A bad covariance is reported, never repaired by the new steppers.
    state = tuple(torch.tensor([v], dtype=torch.float64) for v in (0, 0, -.1, 1, 0))
    bad = advance_gaussian(state, 1., 0., 0., .05, .5, .001, state[0], 'platen')
    health = covariance_health(bad)
    assert health['nonpositive_diagonals'] == 1 and health['uncertainty_violations'] == 1
    assert bad[2].item() < 0
    # Default legacy rule uses the updated momentum in dx, unlike true EM.
    env = TorchOscillatorEnv(osc, 2, horizon=2, dtype=torch.float64)
    assert env.integration_method == 'legacy'
    print('Environment dispatch, external noise, gradients and unmasked covariance failure: passed')


def legacy_regression(path):
    """Optional compare against a saved pre-change module, with identical RNG."""
    spec = importlib.util.spec_from_file_location('old_env', path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    osc = SimpleNamespace(omegas=frequencies(.1), eta=.5, gamma_meas=.05)
    torch.manual_seed(132)
    before = old.TorchOscillatorEnv(osc, 2, horizon=3)
    torch.manual_seed(132)
    after = TorchOscillatorEnv(osc, 2, horizon=3)
    assert_close(before._noise, after._noise, rtol=0, atol=0)
    for _ in range(3):
        action = torch.full((2, 20), .2)
        assert_close(before.step(action), after.step(action), rtol=0, atol=0)
        for name in ('xc', 'pc', 'Vxx', 'Vpp', 'Cxp'):
            assert_close(getattr(before, name), getattr(after, name), rtol=0, atol=0)
    print('Legacy default is bitwise unchanged against saved baseline')


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-baseline', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    deterministic_check()
    heating_and_numpy_check()
    weak_heating_check()
    noise_and_gradient_check()
    environment_check()
    if args.legacy_baseline:
        legacy_regression(args.legacy_baseline)
    print('All integrator checks passed.')


if __name__ == '__main__':
    main()

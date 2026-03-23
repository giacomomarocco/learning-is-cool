#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:43:22 2026

@author: giacomomarocco
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'learn_to_cool'))

import torch
import torch.nn as nn
import numpy as np
from gaussian_oscillator_array import GaussianOscillatorArray
from torch_oscillator_env import TorchOscillatorEnv
from optimal_feedback_n_oscillators import OptimalFeedbackNOscillators
import matplotlib.pyplot as plt
import time

dirname = os.path.dirname(os.path.abspath(__file__))

# Hyperparameters
omegas = [[1.0, 1.01], [1.05, 1.06]]
initial_temperature = 10
gamma_BA = 18.8/104
g_fb = 0.2
eta = 0.2
dt = 0.05
n_th = 10
horizon = 300
batch_size = 32768
learning_rate = 5e-5
n_iterations = 500
eval_rate = 50

# CLI overrides
from cli_utils import parse_overrides
overrides = parse_overrides()
n_th = overrides.get('n_th', n_th)
initial_temperature = overrides.get('initial_temperature', initial_temperature)
gamma_BA = overrides.get('gamma_BA', gamma_BA)
eta = overrides.get('eta', eta)
dt = overrides.get('dt', dt)
horizon = overrides.get('horizon', horizon)
batch_size = overrides.get('batch_size', batch_size)
g_fb = overrides.get('g_fb', g_fb)
learning_rate = overrides.get('learning_rate', learning_rate)
n_iterations = overrides.get('n_iterations', n_iterations)
eval_rate = overrides.get('eval_rate', eval_rate)
omegas = overrides.get('omegas', omegas)
if overrides:
    print(f"CLI overrides: {overrides}")

omegas = np.asarray(omegas)
if omegas.ndim == 1:
    omegas = np.column_stack([omegas, omegas * 1.01])
N = len(omegas)

# Device detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Environment and Cost
osc_array = GaussianOscillatorArray(omegas=omegas, n_thermal=n_th, gamma_meas=gamma_BA, eta=eta)
q_cost = (g_fb)**(-2)

def torch_rollout_env(policy, env, horizon):
    env.reset()
    total_cost = 0.0
    for _ in range(horizon):
        state = env.state()
        u = policy(state)  # (batch_size, 2)
        cost = env.step(u)
        total_cost += cost
    return total_cost.mean()

def train_policy(policy, env, n_iterations=n_iterations, lr=learning_rate, eval_every=eval_rate):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []

    time_forward = 0.0
    time_backward = 0.0
    time_eval = 0.0
    t_start = time.perf_counter()

    for iteration in range(n_iterations):
        # Forward pass
        t0 = time.perf_counter()
        cost = torch_rollout_env(policy, env, horizon=horizon)
        t1 = time.perf_counter()
        time_forward += t1 - t0

        # Backward pass
        optimizer.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        t2 = time.perf_counter()
        time_backward += t2 - t1

        if iteration % eval_every == 0:
            t3 = time.perf_counter()
            with torch.no_grad():
                eval_cost = torch_rollout_env(policy, env, horizon=200)
            t4 = time.perf_counter()
            time_eval += t4 - t3
            history.append((iteration, eval_cost.item()))
            print(f"  Iter {iteration:4d}, avg cost: {eval_cost.item():.2f}")

    t_total = time.perf_counter() - t_start

    print(f"\n  Timing breakdown:")
    print(f"    Total:    {t_total:.1f}s")
    print(f"    Forward:  {time_forward:.1f}s ({100*time_forward/max(t_total, 1e-6):.0f}%)")
    print(f"    Backward: {time_backward:.1f}s ({100*time_backward/max(t_total, 1e-6):.0f}%)")
    print(f"    Eval:     {time_eval:.1f}s ({100*time_eval/max(t_total, 1e-6):.0f}%)")
    print(f"    Per iteration: {1000*t_total/n_iterations:.1f}ms")

    return history

# Neural network policy (input size 4*N, output 2: u_cd, u_param)
nn_policy = nn.Sequential(
    nn.Linear(4 * N, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 2)
).to(device)

# Training environment
train_env = TorchOscillatorEnv(osc_array, batch_size=batch_size, dt=dt,
                               horizon=horizon, cost_u_weight=q_cost,
                               phase_space_range=5, device=device)

print(f"Training RL policy for {N} oscillators...")
train_policy(nn_policy, train_env, n_iterations=n_iterations, lr=learning_rate, eval_every=eval_rate)

# Save the final weights
weights_dir = os.path.join(os.path.dirname(dirname), 'weights')
os.makedirs(weights_dir, exist_ok=True)
torch.save(nn_policy.state_dict(), os.path.join(weights_dir, 'two_particles_rl_final.pth'))
print(f"Saved final policy weights to {os.path.join(weights_dir, 'two_particles_rl_final.pth')}")

# Evaluation and Comparison
n_traj = 100
# Initial conditions: (n_traj, N, 2, 2) — [oscillator, mode, x/p]
initial_conditions = np.random.randn(n_traj, N, 2, 2)
for a in range(N):
    initial_conditions[:, a, :, 0] *= np.sqrt(initial_temperature + 0.5)
    initial_conditions[:, a, :, 1] *= np.sqrt(initial_temperature + 0.5)

# Optimal baseline (x-mode only)
opt_ctrl = OptimalFeedbackNOscillators(omegas=omegas, q=q_cost)
n_steady_opt = opt_ctrl.mean_nbar(eta=eta, Gamma_BA=gamma_BA)

# RL performance
eval_env = TorchOscillatorEnv(osc_array, batch_size=n_traj, dt=dt,
                              horizon=horizon, phase_space_range=5, device=device)

# Initial conditions tensor: (n_traj, N, 2, 2)
init_tensor = torch.tensor(initial_conditions, dtype=torch.float32, device=device)

eval_env.reset(initial_conditions=init_tensor)

n_history = []
with torch.no_grad():
    for t in range(horizon):
        state = eval_env.state()
        u = nn_policy(state)  # (batch, 2)
        eval_env.step(u)
        # eval_env.n_bar() returns (batch, N) — per-oscillator sum over modes
        n_history.append(eval_env.n_bar().mean(dim=1).cpu().numpy())

n_history = np.array(n_history).T # (n_traj, horizon)
n_rl_traj = []
for i in range(n_traj):
    n_rl_traj.append(np.mean(n_history[i, -horizon // 4:]))

print(f"\ng_fb = {g_fb}")
print(f"Optimal x-mode (steady-state): n_bar = {n_steady_opt:.3f}")
print(f"RL (simulated):                n_bar = {np.mean(n_rl_traj):.3f} +/- {np.std(n_rl_traj)/np.sqrt(n_traj):.3f}")

# Trajectory simulation for plotting
def simulate_trajectory(policy, osc_array, horizon=500, dt=0.05):
    sim_env = TorchOscillatorEnv(osc_array, batch_size=1, dt=dt,
                                 horizon=horizon, phase_space_range=5, device=device)

    # Custom initial conditions: (1, N, 2, 2)
    init_cond = torch.zeros(1, N, 2, 2, dtype=torch.float32, device=device)
    init_cond[0, 0, 0, 0] = 10.0  # osc 0, x-mode, position
    init_cond[0, 1, 0, 1] = 10.0  # osc 1, x-mode, momentum
    sim_env.reset(initial_conditions=init_cond)

    x_history = [[] for _ in range(N)]
    u_cd_history = []
    u_param_history = []
    times = []

    with torch.no_grad():
        for t in range(horizon):
            for a in range(N):
                x_history[a].append(sim_env.xc[0, a, 0].item())  # x-mode
            times.append(t * dt)

            state = sim_env.state()
            u_raw = policy(state)  # (1, 2)
            u_cd_history.append(u_raw[0, 0].item())
            u_param_history.append(sim_env.modulation_depth * torch.tanh(u_raw[0, 1]).item())
            sim_env.step(u_raw)

    return np.array(times), np.array(x_history), np.array(u_cd_history), np.array(u_param_history)

times, xs, u_cd, u_param = simulate_trajectory(nn_policy, osc_array, horizon=1000, dt=dt)

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for a in range(N):
    axes[0].plot(times, xs[a], label=rf'Oscillator {a+1} ($\omega_x={omegas[a,0]:.2f}$)')
axes[0].set_ylabel('Position (x-mode)')
axes[0].legend()
axes[0].set_title(f'RL Cooling of {N} Oscillators (2D modes)')

axes[1].plot(times, u_cd, color='C3', label='Cold damping')
axes[1].set_ylabel(r'$u_{cd}$')
axes[1].legend()

axes[2].plot(times, u_param, color='C4', label='Parametric mod.')
axes[2].set_ylabel(r'$u_{param}$')
axes[2].set_xlabel('Time')
axes[2].legend()

plt.tight_layout()
os.makedirs(os.path.join(dirname, 'figures'), exist_ok=True)
plt.savefig(os.path.join(dirname, 'figures/two_particles_rl_trajectory.png'), dpi=150)
print(f"Saved trajectory plot to {os.path.join(dirname, 'figures/two_particles_rl_trajectory.png')}")
plt.close()

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
from gaussian_oscillator import GaussianOscillator
from n_oscillators import GaussianOscillatorArray
from torch_oscillator_env import TorchOscillatorEnv
import copy
import matplotlib.pyplot as plt
import time

dirname = os.path.dirname(os.path.abspath(__file__))

horizon = 300
dt = 0.1
phase_space_max = 5.
batch_size = 128
lr = 1e-4

osc_array = GaussianOscillatorArray(omegas=[1.0, 1.05])

def torch_rollout_env(policy, env, horizon):
    env.reset()
    total_cost = 0.0
    for _ in range(horizon):
        state = env.state()
        u = policy(state).squeeze(-1)
        cost = env.step(u)
        total_cost += cost
    return total_cost.mean()

def train_policy(policy, env, n_iterations=10, lr=0.0001, eval_every=2, horizon=horizon):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []

    time_forward = 0.0
    time_backward = 0.0
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
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
        t2 = time.perf_counter()
        time_backward += t2 - t1

        if iteration % eval_every == 0:
            print(f"  iter {iteration:4d}, cost={cost.item():.2f}")
            history.append((iteration, cost.item()))

    t_total = time.perf_counter() - t_start
    print(f"\n  Timing breakdown (Training):")
    print(f"    Total:    {t_total:.1f}s")
    print(f"    Forward:  {time_forward:.1f}s ({100*time_forward/max(t_total, 1e-6):.0f}%)")
    print(f"    Backward: {time_backward:.1f}s ({100*time_backward/max(t_total, 1e-6):.0f}%)")
    print(f"    Per iteration: {1000*t_total/n_iterations:.1f}ms")

    return history

feedback_gains_rl = np.linspace(0.5, 5, 5)
n_steady_nn = []
trained_policies = {}

for g_fb in feedback_gains_rl:
    q_cost = osc_array.omegas[0] / g_fb**2

    policy_sweep = nn.Sequential(
        nn.Linear(4, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1)
    )

    train_env = TorchOscillatorEnv(osc_array, batch_size=batch_size, dt=dt,
                                   mode='cold_damping', horizon=horizon,
                                   cost_u_weight=q_cost,
                                   phase_space_range=phase_space_max)

    print(f"Training for g_fb={g_fb:.2f}...")
    train_policy(policy_sweep, train_env, n_iterations=10, lr=lr, eval_every=2, horizon=horizon)

    trained_policies[g_fb] = copy.deepcopy(policy_sweep)

    # Save the weights for the first gain
    if g_fb == feedback_gains_rl[0]:
        weights_dir = os.path.join(os.path.dirname(dirname), 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        torch.save(policy_sweep.state_dict(), os.path.join(weights_dir, 'two_oscillators_rl_final.pth'))
        print(f"Saved weights for g_fb={g_fb:.2f} to {os.path.join(weights_dir, 'two_oscillators_rl_final.pth')}")

    # Evaluate steady-state phonon number
    n_eval_traj = 50
    eval_horizon = 2000
    eval_env = TorchOscillatorEnv(osc_array, batch_size=n_eval_traj, dt=dt,
                                  mode='cold_damping', horizon=eval_horizon,
                                  phase_space_range=phase_space_max)
    eval_env.reset()

    n_bars_history = []
    with torch.no_grad():
        for t in range(eval_horizon):
            state = eval_env.state()
            u = policy_sweep(state).squeeze(-1)
            eval_env.step(u)
            if t >= 3 * eval_horizon // 4:
                # Average n_bar across oscillators for each batch element
                n_bars_history.append(eval_env.n_bar().mean(dim=1).numpy()) # (n_eval_traj,)

    n_steady_nn.append(np.mean(n_bars_history))
    print(f"g_fb={g_fb:.2f}, q={q_cost:.4f}, n_bar={n_steady_nn[-1]:.3f}\n")

# Trajectory simulation
def simulate_trajectory(policy, osc_array, n_periods=50, dt=0.01):
    sim_horizon = int(2 * np.pi * n_periods / dt)
    sim_env = TorchOscillatorEnv(osc_array, batch_size=1, dt=dt,
                                 mode='cold_damping', horizon=sim_horizon,
                                 phase_space_range=phase_space_max)

    # Custom initial conditions
    init_cond = torch.tensor([[[15.0, 0.0], [-1.0, 13.0]]], dtype=torch.float32) # (1, 2, 2)
    sim_env.reset(initial_conditions=init_cond)

    x1_history = []
    x2_history = []
    u_history = []
    times = []

    with torch.no_grad():
        for t in range(sim_horizon):
            x1_history.append(sim_env.xc[0, 0].item())
            x2_history.append(sim_env.xc[0, 1].item())
            times.append(t * dt)

            state = sim_env.state()
            u_raw = policy(state).squeeze(-1)
            u_history.append(u_raw.item())
            sim_env.step(u_raw)

    return np.array(times), np.array(x1_history), np.array(x2_history), np.array(u_history)

times, x1, x2, u = simulate_trajectory(trained_policies[feedback_gains_rl[0]], osc_array, n_periods=50, dt=0.1)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axes[0].plot(times, x1, label=rf'Oscillator 1 ($\omega={osc_array.omegas[0]}$)')
axes[0].plot(times, x2, label=rf'Oscillator 2 ($\omega={osc_array.omegas[1]}$)')
axes[0].set_ylabel('Oscillator position')
axes[0].legend()
axes[1].plot(times, u, label='Applied force')
axes[1].set_ylabel('u')
axes[1].set_xlabel('Time')
axes[1].legend()
plt.tight_layout()
os.makedirs(os.path.join(dirname, 'figures'), exist_ok=True)
plt.savefig(os.path.join(dirname, 'figures/two_oscillators_trajectory.png'), dpi=150)
plt.close()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined Parametric and Cold Damping RL Cooling.
Learns an optimal policy for both direct force and trap frequency modulation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'learn_to_cool'))

from gaussian_oscillator import GaussianOscillator
from torch_oscillator_env import TorchOscillatorEnv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time

dirname = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Parameters
# ============================================================
n_th = 10
initial_temperature = 10
gamma_BA = 18.8 / 104
eta = 0.2
dt = 0.05
horizon = 300
batch_size = 512
modulation_depth = 0.5  # max fractional change in omega^2
g_fb = 0.2
q_cost = (g_fb)**(-2)

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
modulation_depth = overrides.get('modulation_depth', modulation_depth)
g_fb = overrides.get('g_fb', g_fb)
q_cost = (g_fb)**(-2)
if overrides:
    print(f"CLI overrides: {overrides}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

osc = GaussianOscillator(n_thermal=n_th, gamma_meas=gamma_BA, eta=eta)

def torch_rollout_env(policy, env, horizon):
    env.reset()
    total_cost = 0.0
    for _ in range(horizon):
        state = env.state()
        u = policy(state) # (batch_size, 2)
        cost = env.step(u)
        total_cost += cost
    return total_cost.mean()

# ============================================================
# Training
# ============================================================
def train_policy(policy, env, n_iterations=1000, lr=0.0005, eval_every=100):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []

    t_start = time.perf_counter()

    for iteration in range(n_iterations):
        # Forward pass
        cost = torch_rollout_env(policy, env, horizon=horizon)

        # Backward pass
        optimizer.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        if iteration % eval_every == 0:
            with torch.no_grad():
                avg = sum(
                    torch_rollout_env(policy, env, horizon=horizon).item()
                    for _ in range(5)
                ) / 5
            history.append((iteration, avg))
            print(f"  Iter {iteration:4d}, avg cost: {avg:.2f}")

    t_total = time.perf_counter() - t_start
    print(f"\nTraining completed in {t_total:.1f}s")
    return history

# --- Combined policy (2 inputs, 2 outputs) ---
print("=" * 60)
print("Training combined feedback policy (xc, pc -> u_cold, u_param)")
print("=" * 60)

combined_policy = nn.Sequential(
    nn.Linear(2, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 2)
).to(device)

train_env = TorchOscillatorEnv(osc, batch_size=batch_size, dt=dt,
                               mode='combined', horizon=horizon,
                               modulation_depth=modulation_depth,
                               cost_u_weight=q_cost,
                               phase_space_range=np.sqrt(initial_temperature),
                               device=device)

history_combined = train_policy(
    combined_policy,
    train_env,
    n_iterations=1000,
    lr=0.0005
)

# ============================================================
# Save weights
# ============================================================
weights_dir = os.path.join(os.path.dirname(dirname), 'weights')
os.makedirs(weights_dir, exist_ok=True)
torch.save(combined_policy.state_dict(),
           os.path.join(weights_dir, 'combined_rl.pth'))
print(f"\nSaved weights to {os.path.join(weights_dir, 'combined_rl.pth')}")

# ============================================================
# Learning curves
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
iters, costs = zip(*history_combined)
ax.plot(iters, costs, label="Combined RL")
ax.set_xlabel("Iteration")
ax.set_ylabel("Average cost")
ax.set_title("Learning curve")
ax.legend()
ax.set_yscale('log')
plt.tight_layout()
os.makedirs(os.path.join(dirname, 'figures'), exist_ok=True)
plt.savefig(os.path.join(dirname, 'figures/combined_learning_curves.png'), dpi=150)
plt.close()

# ============================================================
# Simulate and compare trajectories
# ============================================================
def simulate_combined_trajectory(policy, env, horizon_sim):
    state = env.reset()

    x_history = []
    p_history = []
    u_cold_history = []
    u_param_history = []
    n_history = []
    times = []

    with torch.no_grad():
        for t in range(horizon_sim):
            x_history.append(env.xc.item())
            p_history.append(env.pc.item())
            times.append(t * env.dt)

            n_bar = env.n_bar().item()
            n_history.append(n_bar)

            u_raw = policy(state)
            u_cold = u_raw[0, 0]
            u_param = env.modulation_depth * torch.tanh(u_raw[0, 1])
            u_cold_history.append(u_cold.item())
            u_param_history.append(u_param.item())

            env.step(u_raw)
            state = env.state()

    return (np.array(times), np.array(x_history), np.array(p_history),
            np.array(u_cold_history), np.array(u_param_history), np.array(n_history))

sim_horizon = 2000
eval_env = TorchOscillatorEnv(osc, batch_size=1, dt=dt,
                               mode='combined', horizon=sim_horizon,
                               modulation_depth=modulation_depth,
                               phase_space_range=np.sqrt(initial_temperature),
                               device=device)

times, x_sim, p_sim, u_c_sim, u_p_sim, n_sim = simulate_combined_trajectory(
    combined_policy, eval_env, sim_horizon
)

fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

axes[0].plot(times, x_sim, lw=0.5)
axes[0].set_ylabel(r'$\langle x \rangle_c$')
axes[0].set_title('Combined feedback trajectory')

axes[1].plot(times, u_c_sim, lw=0.5, color='C1')
axes[1].set_ylabel(r'$u_{cold}$')
axes[1].axhline(0, color='k', lw=0.5)

axes[2].plot(times, u_p_sim, lw=0.5, color='C2')
axes[2].set_ylabel(r'$\delta\omega^2/\omega_0^2$')
axes[2].axhline(0, color='k', lw=0.5)

axes[3].plot(times, n_sim, lw=0.5, color='C3')
axes[3].set_ylabel(r'$\bar{n}$')
axes[3].set_xlabel(r'$t$')
axes[3].set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(dirname, 'figures/combined_trajectory.png'), dpi=150)
plt.close()

# ============================================================
# Final Comparison
# ============================================================
print("\n" + "=" * 60)
print("Final performance comparison")
print("=" * 60)

n_traj = 100
n_compare_horizon = 2000

compare_env = TorchOscillatorEnv(osc, batch_size=n_traj, dt=dt,
                                 mode='combined', horizon=n_compare_horizon,
                                 modulation_depth=modulation_depth,
                                 phase_space_range=np.sqrt(initial_temperature),
                                 device=device)
compare_env.reset()

n_bar_combined = np.zeros((n_traj, n_compare_horizon))

with torch.no_grad():
    for t in range(n_compare_horizon):
        n_bar_combined[:, t] = compare_env.n_bar().squeeze(-1).cpu().numpy()
        state = compare_env.state()
        u_raw = combined_policy(state)
        compare_env.step(u_raw)

n_final_combined = np.mean(n_bar_combined[:, -n_compare_horizon // 4:], axis=1)

n_min_theory_cd = (eta ** (-0.5) - 1) / 2
print(f"Combined RL:  n_bar = {np.mean(n_final_combined):.4f} ± {np.std(n_final_combined)/np.sqrt(n_traj):.4f}")
print(f"Theoretical min (CD only): n_min = {n_min_theory_cd:.4f}")

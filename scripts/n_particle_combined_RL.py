#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scale Combined RL Cooling to N Particles (Shared Control).
Trains an RL agent to cool an array of N particles using a shared pair
of control signals (u_cold, u_param).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'learn_to_cool'))

from n_oscillators import GaussianOscillatorArray
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

# Array of N particles with different frequencies
omegas = [1.0, 1.05, 1.1]
N = len(omegas)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Cooling {N} oscillators with frequencies: {omegas}")

osc_array = GaussianOscillatorArray(omegas=omegas, n_thermal=n_th, gamma_meas=gamma_BA, eta=eta)

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
def train_policy(policy, env, n_iterations=1500, lr=0.0005, eval_every=100):
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

# --- N-particle policy (2N inputs, 2 outputs) ---
print("=" * 60)
print(f"Training shared combined feedback policy (2*{N} inputs -> 2 outputs)")
print("=" * 60)

# Input dimension is 2*N (all xc and all pc)
combined_policy = nn.Sequential(
    nn.Linear(2 * N, 128),
    nn.Tanh(),
    nn.Linear(128, 128),
    nn.Tanh(),
    nn.Linear(128, 2)
).to(device)

train_env = TorchOscillatorEnv(osc_array, batch_size=batch_size, dt=dt,
                               mode='combined', horizon=horizon,
                               modulation_depth=modulation_depth,
                               cost_u_weight=q_cost,
                               phase_space_range=np.sqrt(initial_temperature),
                               device=device)

history_combined = train_policy(
    combined_policy,
    train_env,
    n_iterations=1500,
    lr=0.0005
)

# ============================================================
# Save weights
# ============================================================
weights_dir = os.path.join(os.path.dirname(dirname), 'weights')
os.makedirs(weights_dir, exist_ok=True)
torch.save(combined_policy.state_dict(),
           os.path.join(weights_dir, 'n_particle_combined_rl.pth'))
print(f"\nSaved weights to {os.path.join(weights_dir, 'n_particle_combined_rl.pth')}")

# ============================================================
# Learning curves
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
iters, costs = zip(*history_combined)
ax.plot(iters, costs, label=f"N={N} Combined RL")
ax.set_xlabel("Iteration")
ax.set_ylabel("Average cost")
ax.set_title(f"Learning curve ({N} particles)")
ax.legend()
ax.set_yscale('log')
plt.tight_layout()
os.makedirs(os.path.join(dirname, 'figures'), exist_ok=True)
plt.savefig(os.path.join(dirname, 'figures/n_particle_combined_learning_curves.png'), dpi=150)
plt.close()

# ============================================================
# Simulate and compare trajectories
# ============================================================
def simulate_n_particle_trajectory(policy, env, horizon_sim):
    state = env.reset()

    # Store histories for each oscillator: list of length N
    x_history = [[] for _ in range(N)]
    p_history = [[] for _ in range(N)]
    n_history = [[] for _ in range(N)]

    u_cold_history = []
    u_param_history = []
    times = []

    with torch.no_grad():
        for t in range(horizon_sim):
            # env.xc is (1, N)
            for i in range(N):
                x_history[i].append(env.xc[0, i].item())
                p_history[i].append(env.pc[0, i].item())

            times.append(t * env.dt)

            n_bars = env.n_bar() # (1, N)
            for i in range(N):
                n_history[i].append(n_bars[0, i].item())

            u_raw = policy(state) # (1, 2)
            u_cold = u_raw[0, 0]
            u_param = env.modulation_depth * torch.tanh(u_raw[0, 1])
            u_cold_history.append(u_cold.item())
            u_param_history.append(u_param.item())

            env.step(u_raw)
            state = env.state()

    return (np.array(times), np.array(x_history), np.array(p_history),
            np.array(u_cold_history), np.array(u_param_history), np.array(n_history))

sim_horizon = 2000
eval_env = TorchOscillatorEnv(osc_array, batch_size=1, dt=dt,
                               mode='combined', horizon=sim_horizon,
                               modulation_depth=modulation_depth,
                               phase_space_range=np.sqrt(initial_temperature),
                               device=device)

times, x_sim, p_sim, u_c_sim, u_p_sim, n_sim = simulate_n_particle_trajectory(
    combined_policy, eval_env, sim_horizon
)

fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

for i in range(N):
    axes[0].plot(times, x_sim[i], lw=0.5, label=f"$\omega_{i}={omegas[i]:.2f}$")
axes[0].set_ylabel(r'$\langle x \rangle_c$')
axes[0].set_title(f'N={N} Combined feedback trajectory')
axes[0].legend(loc='upper right', ncol=N, fontsize='small')

axes[1].plot(times, u_c_sim, lw=0.5, color='C3')
axes[1].set_ylabel(r'$u_{cold}$')
axes[1].axhline(0, color='k', lw=0.5)

axes[2].plot(times, u_p_sim, lw=0.5, color='C4')
axes[2].set_ylabel(r'$\delta\omega^2/\omega_0^2$')
axes[2].axhline(0, color='k', lw=0.5)

for i in range(N):
    axes[3].plot(times, n_sim[i], lw=0.5)
axes[3].set_ylabel(r'$\bar{n}$')
axes[3].set_xlabel(r'$t$')
axes[3].set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(dirname, 'figures/n_particle_combined_trajectory.png'), dpi=150)
plt.close()

# ============================================================
# Final Comparison
# ============================================================
print("\n" + "=" * 60)
print(f"Final performance comparison (N={N})")
print("=" * 60)

n_traj = 128
n_compare_horizon = 2000

compare_env = TorchOscillatorEnv(osc_array, batch_size=n_traj, dt=dt,
                                 mode='combined', horizon=n_compare_horizon,
                                 modulation_depth=modulation_depth,
                                 phase_space_range=np.sqrt(initial_temperature),
                                 device=device)
compare_env.reset()

n_bars_history = np.zeros((N, n_traj, n_compare_horizon))

with torch.no_grad():
    for t in range(n_compare_horizon):
        # env.n_bar() returns (n_traj, N)
        n_bars_history[:, :, t] = compare_env.n_bar().cpu().numpy().T
        state = compare_env.state()
        u_raw = combined_policy(state)
        compare_env.step(u_raw)

n_final_avg = np.mean(n_bars_history[:, :, -n_compare_horizon // 4:], axis=(1, 2))

n_min_theory_cd = (eta ** (-0.5) - 1) / 2
print(f"Theoretical min (CD only, single particle): n_min = {n_min_theory_cd:.4f}")
print("-" * 60)
for i in range(N):
    print(f"Oscillator {i} (w={omegas[i]:.2f}): n_bar = {n_final_avg[i]:.4f}")
print("-" * 60)
print(f"Average n_bar across all oscillators: {np.mean(n_final_avg):.4f}")

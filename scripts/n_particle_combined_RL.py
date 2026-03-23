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

from gaussian_oscillator_array import GaussianOscillatorArray
from optimal_feedback_n_oscillators import OptimalFeedbackNOscillators
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
horizon = 400
batch_size = 8192
modulation_depth = 0.5  # max fractional change in omega^2
g_fb = 1.0
q_cost = (g_fb)**(-2)
n_iterations = 500

# Array of N particles with different frequencies — [omega_x, omega_y] per oscillator
omegas = [[1.0, 1.01], [1.05, 1.06], [1.1, 1.11]]

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
n_iterations = overrides.get('n_iterations', n_iterations)
omegas = overrides.get('omegas', omegas)
omegas = np.asarray(omegas)
if omegas.ndim == 1:
    omegas = np.column_stack([omegas, omegas * 1.01])
q_cost = (g_fb)**(-2)
N = len(omegas)
if overrides:
    print(f"CLI overrides: {overrides}")

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
def train_policy(policy, env, n_iterations=1500, lr=0.0005, eval_every=50):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50)
    current_lr = lr

    t_start = time.perf_counter()

    for iteration in range(n_iterations):
        # Forward pass
        cost = torch_rollout_env(policy, env, horizon=horizon)

        # Backward pass
        optimizer.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(cost.item())  # then adjust the learning rate
        
        if (lr_ := optimizer.param_groups[0]['lr']) != current_lr:
            print(f"  ** LR: {current_lr:.6f} -> {lr_:.6f}")
            current_lr = lr_

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

# --- N-particle policy (4N inputs, 2 outputs) ---
print("=" * 60)
print(f"Training shared combined feedback policy (4*{N} inputs -> 2 outputs)")
print("=" * 60)

# Input dimension is 4*N (xc and pc for both modes of each oscillator)
combined_policy = nn.Sequential(
    nn.Linear(4 * N, 128),
    nn.Tanh(),
    nn.Linear(128, 128),
    nn.Tanh(),
    nn.Linear(128, 2)
).to(device)

train_env = TorchOscillatorEnv(osc_array, batch_size=batch_size, dt=dt,
                               horizon=horizon,
                               modulation_depth=modulation_depth,
                               cost_u_weight=q_cost,
                               phase_space_range=np.sqrt(initial_temperature),
                               device=device)

history_combined = train_policy(
    combined_policy,
    train_env,
    n_iterations=n_iterations,
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
    x_history = []
    p_history = []
    n_history = []
    u_cold_history = []
    u_param_history = []
    times = []

    with torch.no_grad():
        for t in range(horizon_sim):
            # env.xc is (1, N, 2) — record x-mode
            # for i in range(N):
            #     x_history[i].append(env.xc[0, i, 0].item())
            #     p_history[i].append(env.pc[0, i, 0].item())

            # times.append(t * env.dt)

            # n_bars = env.n_bar() # (1, N)
            # for i in range(N):
            #     n_history[i].append(n_bars[0, i].item())

            # u_raw = policy(state) # (1, 2)
            # u_cold = u_raw[0, 0]
            # u_param = env.modulation_depth * torch.tanh(u_raw[0, 1])
            # u_cold_history.append(u_cold.item())
            # u_param_history.append(u_param.item())

            # env.step(u_raw)
            # state = env.state()
            x_history.append(env.xc[0].cpu().numpy().copy())   # (N, 2)
            p_history.append(env.pc[0].cpu().numpy().copy())   # (N, 2)
            times.append(t * env.dt)
            
            # Per-mode n_bar: (1, N, 2)
            n_per_mode = (env.xc**2 + env.Vxx + env.pc**2 + env.Vpp) / 4.0 - 0.5
            n_history.append(n_per_mode[0].cpu().numpy().copy())  # (N, 2)
            
            u_raw = policy(state)
            u_cold = u_raw[0, 0]
            u_param = env.modulation_depth * torch.tanh(u_raw[0, 1])
            u_cold_history.append(u_cold.item())
            u_param_history.append(u_param.item())
            
            env.step(u_raw)
            state = env.state()

        # Stack: (horizon, N, 2) -> transpose to (N, 2, horizon)
        x_hist = np.array(x_history).transpose(1, 2, 0)
        p_hist = np.array(p_history).transpose(1, 2, 0)
        n_hist = np.array(n_history).transpose(1, 2, 0)
    return (np.array(times), x_hist, p_hist,
            np.array(u_cold_history), np.array(u_param_history), n_hist)


sim_horizon = 2000
eval_env = TorchOscillatorEnv(osc_array, batch_size=1, dt=dt,
                               horizon=sim_horizon,
                               modulation_depth=modulation_depth,
                               phase_space_range=np.sqrt(initial_temperature),
                               device=device)

times, x_sim, p_sim, u_c_sim, u_p_sim, n_sim = simulate_n_particle_trajectory(
    combined_policy, eval_env, sim_horizon
)

fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

for i in range(N):
    axes[0].plot(times, x_sim[i, 0], lw=0.5, label=fr"$\omega_{{x,{i}}}={omegas[i,0]:.2f}$")
    axes[0].plot(times, x_sim[i, 1], lw=0.5, ls='--', label=fr"$\omega_{{y,{i}}}={omegas[i,1]:.2f}$")
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
    axes[3].plot(times, n_sim[i, 0], lw=0.5, label=f'osc {i} x')
    axes[3].plot(times, n_sim[i, 1], lw=0.5, ls='--', label=f'osc {i} y')
axes[3].legend(fontsize='small')
axes[3].set_ylabel(r'$\bar{n}$')
axes[3].set_xlabel(r'$t$')
axes[3].set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(dirname, 'figures/n_particle_combined_trajectory.png'), dpi=150)
plt.close()


n_traj = 128
n_compare_horizon = 2000

compare_env = TorchOscillatorEnv(osc_array, batch_size=n_traj, dt=dt,
                                 horizon=n_compare_horizon,
                                 modulation_depth=modulation_depth,
                                 phase_space_range=np.sqrt(initial_temperature),
                                 device=device)
compare_env.reset()

n_bars_history = np.zeros((N, 2, n_traj, n_compare_horizon))

with torch.no_grad():
    for t in range(n_compare_horizon):
        n_per_mode = (compare_env.xc**2 + compare_env.Vxx
                      + compare_env.pc**2 + compare_env.Vpp) / 4.0 - 0.5  # (n_traj, N, 2)
        n_bars_history[:, :, :, t] = n_per_mode.cpu().numpy().transpose(1, 2, 0)
        state = compare_env.state()
        u_raw = combined_policy(state)
        compare_env.step(u_raw)

n_final_avg = np.mean(n_bars_history[:, :, :, -n_compare_horizon // 4:], axis=(2, 3))  # (N, 2)


lqr = OptimalFeedbackNOscillators(omegas, q=q_cost)
n_min_theory_cd = lqr.steady_state_nbar(eta, gamma_BA)

# ============================================================
# Zero-policy baseline
# ============================================================
print("\n" + "=" * 60)
print("Zero-policy baseline (no feedback)")
print("=" * 60)

zero_env = TorchOscillatorEnv(osc_array, batch_size=n_traj, dt=dt,
                              horizon=n_compare_horizon,
                              modulation_depth=modulation_depth,
                              phase_space_range=np.sqrt(initial_temperature),
                              device=device)
zero_env.reset()

n_bars_zero = np.zeros((N, 2, n_traj, n_compare_horizon))

with torch.no_grad():
    for t in range(n_compare_horizon):
        n_per_mode = (zero_env.xc**2 + zero_env.Vxx
                      + zero_env.pc**2 + zero_env.Vpp) / 4.0 - 0.5
        n_bars_zero[:, :, :, t] = n_per_mode.cpu().numpy().transpose(1, 2, 0)
        state = zero_env.state()
        u_zero = torch.zeros(n_traj, 2, device=device)
        zero_env.step(u_zero)

n_final_zero = np.mean(n_bars_zero[:, :, :, -n_compare_horizon // 4:], axis=(2, 3))  # (N, 2)

for i in range(N):
    print(f"Oscillator {i} (w={omegas[i,0]:.2f},{omegas[i,1]:.2f}):")
    print(f"  x-mode: n_bar (no feedback) = {n_final_zero[i,0]:.4f}")
    print(f"  y-mode: n_bar (no feedback) = {n_final_zero[i,1]:.4f}")
# ============================================================
# Final Comparison
# ============================================================
print("\n" + "=" * 60)
print(f"Final performance comparison (N={N})")
print("=" * 60)

for i in range(N):
    print(f"Oscillator {i} (w={omegas[i,0]:.2f},{omegas[i,1]:.2f}):")
    print(f"  x-mode: n_bar = {n_final_avg[i,0]:.4f}, n_min (LQR) = {n_min_theory_cd[i]:.4f}")
    print(f"  y-mode: n_bar = {n_final_avg[i,1]:.4f}  (no direct feedback)")
    
print("-" * 60)
print(f"Average n_bar (x-mode): {np.mean(n_final_avg[:, 0]):.4f}")
print(f"Average n_bar (y-mode): {np.mean(n_final_avg[:, 1]):.4f}")
print(f"Average n_bar (overall): {np.mean(n_final_avg):.4f}")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scale Combined RL Cooling to NxN Grid of 3D Oscillators.
Trains an RL agent to cool a 2D array of particles using
row/column cold damping and parametric modulation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'learn_to_cool'))

from gaussian_oscillator_array import GaussianOscillatorArray
from optimal_feedback_n_oscillators import OptimalFeedbackNOscillators, GridFeedbackLQR
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

# 2D NxN grid — [omega_x, omega_y, omega_z] per site
# Example: 2x2 grid
N_grid = 2
base_x = np.linspace(1.0, 1.1, N_grid)
base_y = np.linspace(1.01, 1.11, N_grid)
base_z = np.linspace(1.02, 1.12, N_grid)
omegas = np.zeros((N_grid, N_grid, 3))
for a in range(N_grid):
    for b in range(N_grid):
        omegas[a, b, 0] = base_x[a] + 0.01 * b  # omega_x
        omegas[a, b, 1] = base_y[a] + 0.01 * b  # omega_y
        omegas[a, b, 2] = base_z[a] + 0.01 * b  # omega_z

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
    # Single list of frequencies — build NxN grid with slight offsets
    N_grid = int(np.sqrt(len(omegas)))
    omegas_3d = np.zeros((N_grid, N_grid, 3))
    for a in range(N_grid):
        for b in range(N_grid):
            w = omegas[a * N_grid + b]
            omegas_3d[a, b] = [w, w * 1.01, w * 1.02]
    omegas = omegas_3d
elif omegas.ndim == 2:
    # (N, 3) — reshape to (sqrt(N), sqrt(N), 3) or (N, 1, 3)
    N_lin = omegas.shape[0]
    N_grid = int(np.sqrt(N_lin))
    if N_grid * N_grid == N_lin and omegas.shape[1] == 3:
        omegas = omegas.reshape(N_grid, N_grid, 3)
    else:
        # (N, 2) old format — expand to (N, 1, 3) with z = mean(x,y)
        omegas_3d = np.zeros((N_lin, 1, 3))
        omegas_3d[:, 0, 0] = omegas[:, 0]
        omegas_3d[:, 0, 1] = omegas[:, 1] if omegas.shape[1] > 1 else omegas[:, 0] * 1.01
        omegas_3d[:, 0, 2] = np.mean(omegas, axis=1) * 1.02
        omegas = omegas_3d
q_cost = (g_fb)**(-2)
N_grid = omegas.shape[0]
N = N_grid  # grid size
if overrides:
    print(f"CLI overrides: {overrides}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Cooling {N}x{N} grid ({N*N} oscillators, 3 modes each)")
print(f"Frequencies shape: {omegas.shape}")

osc_array = GaussianOscillatorArray(omegas=omegas, n_thermal=n_th, gamma_meas=gamma_BA, eta=eta)

def torch_rollout_env(policy, env, horizon):
    env.reset()
    total_cost = 0.0
    for _ in range(horizon):
        state = env.state()
        u = policy(state)  # (batch_size, 4N)
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
    t_forward_total = 0.0
    t_backward_total = 0.0

    for iteration in range(n_iterations):
        # Forward pass
        t_fwd0 = time.perf_counter()
        cost = torch_rollout_env(policy, env, horizon=horizon)
        t_fwd1 = time.perf_counter()

        # Backward pass
        optimizer.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        t_bwd1 = time.perf_counter()

        t_forward_total += t_fwd1 - t_fwd0
        t_backward_total += t_bwd1 - t_fwd1

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
            elapsed = time.perf_counter() - t_start
            per_iter = elapsed / (iteration + 1)
            print(f"  Iter {iteration:4d}, avg cost: {avg:.2f}  "
                  f"[{elapsed:.0f}s elapsed, {per_iter:.2f}s/iter]")

    t_total = time.perf_counter() - t_start
    avg_fwd = t_forward_total / n_iterations
    avg_bwd = t_backward_total / n_iterations
    print(f"\nTraining completed in {t_total:.1f}s ({t_total/n_iterations:.2f}s/iter)")
    print(f"  Forward rollout: {avg_fwd:.2f}s/iter ({100*t_forward_total/t_total:.0f}%)")
    print(f"  Backward + optim: {avg_bwd:.2f}s/iter ({100*t_backward_total/t_total:.0f}%)")
    return history

# --- Grid policy: 6N^2 inputs, 4N outputs ---
input_dim = 6 * N * N
output_dim = 4 * N
print("=" * 60)
print(f"Training grid feedback policy ({input_dim} inputs -> {output_dim} outputs)")
print("=" * 60)

combined_policy = nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.Tanh(),
    nn.Linear(128, 128),
    nn.Tanh(),
    nn.Linear(128, output_dim)
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
ax.plot(iters, costs, label=f"{N}x{N} Grid Combined RL")
ax.set_xlabel("Iteration")
ax.set_ylabel("Average cost")
ax.set_title(f"Learning curve ({N}x{N} grid, {N*N} oscillators)")
ax.legend()
ax.set_yscale('log')
plt.tight_layout()
os.makedirs(os.path.join(dirname, 'figures'), exist_ok=True)
plt.savefig(os.path.join(dirname, 'figures/n_particle_combined_learning_curves.png'), dpi=150)
plt.close()

# ============================================================
# Simulate and compare trajectories
# ============================================================
def simulate_grid_trajectory(policy, env, horizon_sim):
    state = env.reset()

    x_history = []
    p_history = []
    n_history = []
    u_cd_x_history = []
    u_cd_y_history = []
    u_param_x_history = []
    u_param_y_history = []
    times = []

    with torch.no_grad():
        for t_step in range(horizon_sim):
            x_history.append(env.xc[0].cpu().numpy().copy())   # (N, N, 3)
            p_history.append(env.pc[0].cpu().numpy().copy())
            times.append(t_step * env.dt)

            # Per-mode n_bar: (1, N, N, 3)
            n_per_mode = (env.xc**2 + env.Vxx + env.pc**2 + env.Vpp) / 4.0 - 0.5
            n_history.append(n_per_mode[0].cpu().numpy().copy())  # (N, N, 3)

            u_raw = policy(state)  # (1, 4N)
            u_cd_x = u_raw[0, :N]
            u_cd_y = u_raw[0, N:2*N]
            u_param_x = env.modulation_depth * torch.tanh(u_raw[0, 2*N:3*N])
            u_param_y = env.modulation_depth * torch.tanh(u_raw[0, 3*N:])
            u_cd_x_history.append(u_cd_x.cpu().numpy().copy())
            u_cd_y_history.append(u_cd_y.cpu().numpy().copy())
            u_param_x_history.append(u_param_x.cpu().numpy().copy())
            u_param_y_history.append(u_param_y.cpu().numpy().copy())

            env.step(u_raw)
            state = env.state()

    # x_history: list of (N, N, 3) -> stack to (horizon, N, N, 3)
    x_hist = np.array(x_history)
    p_hist = np.array(p_history)
    n_hist = np.array(n_history)
    return (np.array(times), x_hist, p_hist,
            np.array(u_cd_x_history), np.array(u_cd_y_history),
            np.array(u_param_x_history), np.array(u_param_y_history),
            n_hist)


sim_horizon = 2000
eval_env = TorchOscillatorEnv(osc_array, batch_size=1, dt=dt,
                               horizon=sim_horizon,
                               modulation_depth=modulation_depth,
                               phase_space_range=np.sqrt(initial_temperature),
                               device=device)

times, x_sim, p_sim, u_cx, u_cy, u_px, u_py, n_sim = simulate_grid_trajectory(
    combined_policy, eval_env, sim_horizon
)

# Plot: one row per grid site, showing all 3 modes
n_sites = N * N
fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

# Panel 0: conditional means (x-mode of each site)
for a in range(N):
    for b in range(N):
        for m, mode_name in enumerate(['x', 'y', 'z']):
            ls = ['-', '--', ':'][m]
            axes[0].plot(times, x_sim[:, a, b, m], lw=0.5, ls=ls,
                        label=f'({a},{b}) {mode_name}' if m == 0 else None)
axes[0].set_ylabel(r'$\langle x \rangle_c$')
axes[0].set_title(f'{N}x{N} Grid combined feedback trajectory')
axes[0].legend(loc='upper right', ncol=N*N, fontsize='x-small')

# Panel 1: cold damping forces
for a in range(N):
    axes[1].plot(times, u_cx[:, a], lw=0.5, label=f'u_cd_x row {a}')
    axes[1].plot(times, u_cy[:, a], lw=0.5, ls='--', label=f'u_cd_y col {a}')
axes[1].set_ylabel(r'$u_{cd}$')
axes[1].axhline(0, color='k', lw=0.5)
axes[1].legend(fontsize='x-small')

# Panel 2: parametric modulation
for a in range(N):
    axes[2].plot(times, u_px[:, a], lw=0.5, label=f'u_p_x row {a}')
    axes[2].plot(times, u_py[:, a], lw=0.5, ls='--', label=f'u_p_y col {a}')
axes[2].set_ylabel(r'$\delta\omega^2/\omega_0^2$')
axes[2].axhline(0, color='k', lw=0.5)
axes[2].legend(fontsize='x-small')

# Panel 3: phonon numbers per site per mode
for a in range(N):
    for b in range(N):
        for m, mode_name in enumerate(['x', 'y', 'z']):
            ls = ['-', '--', ':'][m]
            axes[3].plot(times, n_sim[:, a, b, m], lw=0.5, ls=ls,
                        label=f'({a},{b}) {mode_name}')
axes[3].legend(fontsize='x-small', ncol=3)
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

# n_bars: (N, N, 3, n_traj, n_compare_horizon)
n_bars_history = np.zeros((N, N, 3, n_traj, n_compare_horizon))

with torch.no_grad():
    for t in range(n_compare_horizon):
        n_per_mode = (compare_env.xc**2 + compare_env.Vxx
                      + compare_env.pc**2 + compare_env.Vpp) / 4.0 - 0.5  # (n_traj, N, N, 3)
        n_bars_history[:, :, :, :, t] = n_per_mode.cpu().numpy().transpose(1, 2, 3, 0)
        state = compare_env.state()
        u_raw = combined_policy(state)
        compare_env.step(u_raw)

# Average over trajectories and last quarter: (N, N, 3)
n_final_avg = np.mean(n_bars_history[:, :, :, :, -n_compare_horizon // 4:], axis=(3, 4))

# LQR comparison for x-mode rows
lqr_grid = GridFeedbackLQR(omegas, q=q_cost)

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

n_bars_zero = np.zeros((N, N, 3, n_traj, n_compare_horizon))

with torch.no_grad():
    for t in range(n_compare_horizon):
        n_per_mode = (zero_env.xc**2 + zero_env.Vxx
                      + zero_env.pc**2 + zero_env.Vpp) / 4.0 - 0.5
        n_bars_zero[:, :, :, :, t] = n_per_mode.cpu().numpy().transpose(1, 2, 3, 0)
        state = zero_env.state()
        u_zero = torch.zeros(n_traj, 4 * N, device=device)
        zero_env.step(u_zero)

n_final_zero = np.mean(n_bars_zero[:, :, :, :, -n_compare_horizon // 4:], axis=(3, 4))  # (N, N, 3)

for a in range(N):
    for b in range(N):
        print(f"Site ({a},{b}) (wx={omegas[a,b,0]:.3f}, wy={omegas[a,b,1]:.3f}, wz={omegas[a,b,2]:.3f}):")
        for m, mode_name in enumerate(['x', 'y', 'z']):
            print(f"  {mode_name}-mode: n_bar (no feedback) = {n_final_zero[a,b,m]:.4f}")

# ============================================================
# Final Comparison
# ============================================================
print("\n" + "=" * 60)
print(f"Final performance comparison ({N}x{N} grid)")
print("=" * 60)

for a in range(N):
    for b in range(N):
        print(f"Site ({a},{b}) (wx={omegas[a,b,0]:.3f}, wy={omegas[a,b,1]:.3f}, wz={omegas[a,b,2]:.3f}):")
        for m, mode_name in enumerate(['x', 'y', 'z']):
            print(f"  {mode_name}-mode: n_bar = {n_final_avg[a,b,m]:.4f}")

print("-" * 60)
print(f"Average n_bar (x-mode): {np.mean(n_final_avg[:, :, 0]):.4f}")
print(f"Average n_bar (y-mode): {np.mean(n_final_avg[:, :, 1]):.4f}")
print(f"Average n_bar (z-mode): {np.mean(n_final_avg[:, :, 2]):.4f}")
print(f"Average n_bar (overall): {np.mean(n_final_avg):.4f}")

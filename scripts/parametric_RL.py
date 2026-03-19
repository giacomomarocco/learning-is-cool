#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parametric feedback cooling via RL.
Learns to modulate trap frequency omega^2 based on conditional means (xc, pc).
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
horizon = 200
batch_size = 512
modulation_depth = 0.5  # max fractional change in omega^2

osc = GaussianOscillator(n_thermal=n_th, gamma_meas=gamma_BA, eta=eta)

def torch_rollout_env(policy, env, horizon):
    env.reset()
    total_cost = 0.0
    for _ in range(horizon):
        state = env.state()
        u = policy(state).squeeze(-1)
        cost = env.step(u)
        total_cost += cost
    return total_cost.mean()
# ============================================================
# Training
# ============================================================
def train_policy(policy, env, n_iterations=500, lr=0.0005, eval_every=50):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []


    time_forward = 0.0
    time_backward = 0.0
    time_eval = 0.0
    t_start = time.perf_counter()
    
    for iteration in range(n_iterations):
        # Forward pass (simulation + policy)
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
                avg = sum(
                    torch_rollout_env(policy, env, horizon=horizon).item()
                    for _ in range(5)
                ) / 5
            t4 = time.perf_counter()
            time_eval += t4 - t3
            history.append((iteration, avg))
            print(f"  Iter {iteration:4d}, avg cost: {avg:.2f}")

    t_total = time.perf_counter() - t_start
    
    print(f"\n  Timing breakdown:")
    print(f"    Total:    {t_total:.1f}s")
    print(f"    Forward:  {time_forward:.1f}s ({100*time_forward/t_total:.0f}%)")
    print(f"    Backward: {time_backward:.1f}s ({100*time_backward/t_total:.0f}%)")
    print(f"    Eval:     {time_eval:.1f}s ({100*time_eval/t_total:.0f}%)")
    print(f"    Per iteration: {1000*t_total/n_iterations:.1f}ms")

    return history


# --- Parametric policy (2 inputs) ---
print("=" * 60)
print("Training parametric feedback policy (2 inputs: xc, pc)")
print("=" * 60)

parametric_policy_2d = nn.Sequential(
    nn.Linear(2, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
)

train_env = TorchOscillatorEnv(osc, batch_size=batch_size, dt=dt,
                               mode='parametric', horizon=horizon,
                               modulation_depth=modulation_depth,
                               phase_space_range=n_th)

history_param_2d = train_policy(
    parametric_policy_2d,
    train_env,
    n_iterations=500,
    lr=0.0005
)


# ============================================================
# Save weights
# ============================================================
weights_dir = os.path.join(os.path.dirname(dirname), 'weights')
os.makedirs(weights_dir, exist_ok=True)
torch.save(parametric_policy_2d.state_dict(),
           os.path.join(weights_dir, 'parametric_2d.pth'))
print(f"\nSaved all weights to {weights_dir}")

# ============================================================
# Learning curves
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
for label, hist in [("Parametric 2D", history_param_2d)]:
    iters, costs = zip(*hist)
    ax.plot(iters, costs, label=label)
ax.set_xlabel("Iteration")
ax.set_ylabel("Average cost")
ax.set_title("Learning curves")
ax.legend()
ax.set_yscale('log')
plt.tight_layout()
os.makedirs(os.path.join(dirname, 'figures'), exist_ok=True)
plt.savefig(os.path.join(dirname, 'figures/learning_curves.png'), dpi=150)
plt.close()

# ============================================================
# Simulate and compare trajectories
# ============================================================
def simulate_parametric_trajectory(policy, env, horizon_sim):
    state = env.reset()

    x_history = []
    p_history = []
    u_history = []
    n_history = []
    times = []

    with torch.no_grad():
        for t in range(horizon_sim):
            x_history.append(env.xc.item())
            p_history.append(env.pc.item())
            times.append(t * env.dt)

            n_bar = env.n_bar().item()
            n_history.append(n_bar)

            u_raw = policy(state).squeeze()
            u = env.modulation_depth * torch.tanh(u_raw)
            u_history.append(u.item())

            env.step(u_raw)
            state = env.state()

    return (np.array(times), np.array(x_history), np.array(p_history),
            np.array(u_history), np.array(n_history))

sim_horizon = 5000
eval_env = TorchOscillatorEnv(osc, batch_size=1, dt=dt,
                               mode='parametric', horizon=sim_horizon,
                               modulation_depth=modulation_depth,
                               phase_space_range=n_th)

times, x_sim, p_sim, u_sim, n_sim = simulate_parametric_trajectory(
    parametric_policy_2d, eval_env, sim_horizon
)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(times, x_sim, lw=0.5)
axes[0].set_ylabel(r'$\langle x \rangle_c$')
axes[0].set_title('Parametric feedback trajectory')

axes[1].plot(times, u_sim, lw=0.5, color='C1')
axes[1].set_ylabel(r'$\delta\omega^2/\omega_0^2$')
axes[1].axhline(0, color='k', lw=0.5)

axes[2].plot(times, n_sim, lw=0.5, color='C2')
axes[2].set_ylabel(r'$\bar{n}$')
axes[2].set_xlabel(r'$t$')
axes[2].set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(dirname, 'figures/parametric_trajectory.png'), dpi=150)
plt.close()

# ============================================================
# Compare final n_bar: parametric vs cold damping
# ============================================================
print("\n" + "=" * 60)
print("Final performance comparison")
print("=" * 60)

n_traj = 100
n_compare_horizon = 2000

compare_env = TorchOscillatorEnv(osc, batch_size=n_traj, dt=dt,
                                 mode='parametric', horizon=n_compare_horizon,
                                 modulation_depth=modulation_depth,
                                 phase_space_range=np.sqrt(initial_temperature))
compare_env.reset()

n_bar_parametric = np.zeros((n_traj, n_compare_horizon))

with torch.no_grad():
    for t in range(n_compare_horizon):
        n_bar_parametric[:, t] = compare_env.n_bar().squeeze(-1).numpy()
        state = compare_env.state()
        u_raw = parametric_policy_2d(state).squeeze(-1)
        compare_env.step(u_raw)

n_final_parametric = np.mean(n_bar_parametric[:, -n_compare_horizon // 4:], axis=1)

n_min_theory = (eta ** (-0.5) - 1) / 2
print(f"Parametric (2D):  n_bar = {np.mean(n_final_parametric):.3f} ± {np.std(n_final_parametric)/np.sqrt(n_traj):.3f}")
print(f"Theoretical min (CD):    n_min = {n_min_theory:.3f}")

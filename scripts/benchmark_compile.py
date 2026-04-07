#!/usr/bin/env python3
"""
Benchmark torch.compile speedup for the oscillator RL training loop.
Compares wall time per iteration with and without compilation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'learn_to_cool'))

from gaussian_oscillator_array import GaussianOscillatorArray
from torch_oscillator_env import TorchOscillatorEnv
import numpy as np
import torch
import torch.nn as nn
import time


# --- Parameters (small enough for a quick test) ---
N_grid = 3
base_x = np.linspace(1.0, 1.1, N_grid)
base_y = np.linspace(1.63, 1.79, N_grid)
base_z = np.linspace(2.46, 2.64, N_grid)
omegas = np.zeros((N_grid, N_grid, 3))
for a in range(N_grid):
    for b in range(N_grid):
        omegas[a, b, 0] = base_x[a] + 0.01 * b
        omegas[a, b, 1] = base_y[a] + 0.01 * b
        omegas[a, b, 2] = base_z[a] + 0.01 * b

batch_size = 8192
horizon = 50
dt = 0.01
modulation_depth = 0.25
g_fb = 1.0
q_cost = g_fb**(-2)

N = N_grid
input_dim = 6 * N * N
output_dim = 4 * N
n_warmup = 5
n_bench = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Grid: {N}x{N}, batch_size={batch_size}, horizon={horizon}")
print(f"Warmup: {n_warmup} iters, Benchmark: {n_bench} iters\n")

osc_array = GaussianOscillatorArray(omegas=omegas, gamma_meas=18.8/104, eta=0.2)


def rollout(policy, env, horizon):
    env.reset()
    total_cost = 0.0
    for _ in range(horizon):
        state = env.state()
        u = policy(state)
        cost = env.step(u)
        total_cost += cost
    return total_cost.mean()


def bench_iterations(policy, env, n_warmup, n_bench, label):
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0005)

    # Warmup (includes compile overhead if compiled)
    for _ in range(n_warmup):
        cost = rollout(policy, env, horizon)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    # Timed iterations
    t0 = time.perf_counter()
    for _ in range(n_bench):
        cost = rollout(policy, env, horizon)
        optimizer.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
    elapsed = time.perf_counter() - t0

    ms_per_iter = 1000 * elapsed / n_bench
    print(f"  {label}: {ms_per_iter:.1f} ms/iter  ({elapsed:.2f}s total for {n_bench} iters)")
    return ms_per_iter


# --- Eager (no compile) ---
print("=" * 50)
print("Eager mode (no torch.compile)")
print("=" * 50)

policy_eager = nn.Sequential(
    nn.Linear(input_dim, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, output_dim),
).to(device)

env_eager = TorchOscillatorEnv(osc_array, batch_size=batch_size, dt=dt,
                                horizon=horizon, modulation_depth=modulation_depth,
                                cost_u_weight=q_cost, device=device)

eager_ms = bench_iterations(policy_eager, env_eager, n_warmup, n_bench, "Eager")

# --- Compiled ---
print()
print("=" * 50)
print("Compiled mode (torch.compile)")
print("=" * 50)

policy_compiled_base = nn.Sequential(
    nn.Linear(input_dim, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, output_dim),
).to(device)

env_compiled = TorchOscillatorEnv(osc_array, batch_size=batch_size, dt=dt,
                                   horizon=horizon, modulation_depth=modulation_depth,
                                   cost_u_weight=q_cost, device=device)

policy_compiled = torch.compile(policy_compiled_base)
env_compiled.step = torch.compile(env_compiled.step)

compiled_ms = bench_iterations(policy_compiled, env_compiled, n_warmup, n_bench, "Compiled")

# --- Summary ---
print()
print("=" * 50)
speedup = eager_ms / compiled_ms
print(f"Speedup: {speedup:.2f}x  ({eager_ms:.1f} ms -> {compiled_ms:.1f} ms per iter)")
if speedup > 1:
    print(f"torch.compile is {speedup:.2f}x faster")
else:
    print(f"torch.compile is {1/speedup:.2f}x slower (compile overhead may dominate at this scale)")
print("=" * 50)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:24:18 2026

@author: giacomomarocco
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'learn_to_cool'))

from gaussian_oscillator import GaussianOscillator
from torch_oscillator_env import TorchOscillatorEnv
import numpy as np
import matplotlib.pyplot as plt
import os
from optimal_control import FeedbackForces
import torch
import torch.nn as nn
import time

dirname = os.path.dirname(os.path.abspath(__file__))

omega = 1.0
initial_temperature = 10
gamma_BA = 18.8/104 # Value used in Magrini et al.
g_fb = 0.2
eta = 0.2
dt = 0.05
n_th = 10
horizon = 300
batch_size = int(4096*2)
learning_rate = 1e-3
n_iterations = 1000
eval_rate = 100

osc = GaussianOscillator(omega = omega,n_thermal=n_th, gamma_meas=gamma_BA, eta=eta)
q_cost = (g_fb)**(-2)

def torch_rollout_env(policy, env, horizon):
    env.reset()
    total_cost = 0.0
    for _ in range(horizon):
        state = env.state()
        u = policy(state).squeeze(-1)
        cost = env.step(u)
        total_cost += cost
    return total_cost.mean()

def train_policy(policy, env, n_iterations=n_iterations, lr=0.0005, eval_every=eval_rate):
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
                # Use a smaller horizon for evaluation
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

# Neural network policy
nn_policy = nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    nn.Linear(16, 16),
    nn.Tanh(),
    nn.Linear(16, 1)
)
    
# nn_policy = nn.Linear(2, 1, bias=False)

# Training environment
train_env = TorchOscillatorEnv(osc, batch_size=batch_size, dt=dt,
                               horizon=horizon, cost_u_weight=q_cost,
                               phase_space_range=5)

print("Training RL policy...")
train_policy(nn_policy, train_env, n_iterations=n_iterations, lr=learning_rate, eval_every=eval_rate)

# Save the final weights
weights_dir = os.path.join(os.path.dirname(dirname), 'weights')
os.makedirs(weights_dir, exist_ok=True)
torch.save(nn_policy.state_dict(), os.path.join(weights_dir, 'single_particle_rl_final.pth'))
print(f"Saved final policy weights to {os.path.join(weights_dir, 'single_particle_rl_final.pth')}")

# # Check gradient at learned weights
# learned = nn_policy.weight.data.clone()
# grad_acc = torch.zeros_like(nn_policy.weight)
# n_samples = 100
# for _ in range(n_samples):
#     nn_policy.zero_grad()
#     cost = torch_rollout_env(nn_policy, train_env, horizon=horizon)
#     cost.backward()
#     grad_acc += nn_policy.weight.grad
# grad_acc /= n_samples
# print(f"Mean gradient at learned: {grad_acc}")
# print(f"Learned weights: {learned}")

# # Check gradient at optimal weights
fb = FeedbackForces(g_fb=g_fb, omega=osc.omega)
opt = fb.optimal_feedback()
# nn_policy.weight.data = torch.tensor([[opt(1, 0), opt(0, 1)]], dtype=torch.float32)
# grad_acc = torch.zeros_like(nn_policy.weight)
# for _ in range(n_samples):
#     nn_policy.zero_grad()
#     cost = torch_rollout_env(nn_policy, train_env, horizon=horizon)
#     cost.backward()
#     grad_acc += nn_policy.weight.grad
# grad_acc /= n_samples
# print(f"Mean gradient at optimal: {grad_acc}")
# # # Check gradients manually
# # nn_policy.weight.data = torch.tensor([[opt(1, 0), opt(0, 1)]], dtype=torch.float32)

# optimizer = torch.optim.Adam(nn_policy.parameters(), lr=1e-4)
# optimizer.zero_grad()
# cost = torch_rollout_env(nn_policy, train_env, horizon=horizon)
# cost.backward()
# print(f"Gradient at optimal weights: {nn_policy.weight.grad}")

# nn_policy.weight.data = learned

# Evaluation and Comparison
n_traj = 100
initial_conditions = np.random.randn(n_traj, 2)
initial_conditions[:, 0] *= np.sqrt(initial_temperature + 0.5)
initial_conditions[:, 1] *= np.sqrt(initial_temperature + 0.5)

# Optimal baseline
fb = FeedbackForces(g_fb=g_fb, omega=osc.omega)
n_opt_traj = []
horizon_periods = horizon * dt / (2 * np.pi)
res_var = osc.variance_solver(n_periods=horizon_periods, dt=dt)
covs_baseline_x = res_var['Vxx']
covs_baseline_p = res_var['Vpp']

for i in range(n_traj):
    init = np.array([initial_conditions[i, 0], initial_conditions[i, 1], 0.0])
    res_exp = osc.expectation_solver(feedback_fn=fb.optimal_feedback(),
                                       initial_conditions=init,
                                       gamma_fb=g_fb, n_periods=horizon_periods, dt=dt)
    n_bar = osc.find_n_bar(res_exp['xc'], res_exp['pc'], covs_baseline_x, covs_baseline_p)
    n_opt_traj.append(np.mean(n_bar[-len(n_bar) // 4:]))

# RL performance
eval_env = TorchOscillatorEnv(osc, batch_size=n_traj, dt=dt,
                              horizon=horizon, phase_space_range=5)
init_tensor = torch.tensor(initial_conditions, dtype=torch.float32).unsqueeze(1) # (n_traj, 1, 2)
eval_env.reset(initial_conditions=init_tensor)

n_rl_traj = []
n_history = []
with torch.no_grad():
    for t in range(horizon):
        state = eval_env.state()
        u = nn_policy(state).squeeze(-1)
        eval_env.step(u)
        n_history.append(eval_env.n_bar().squeeze(-1).numpy()) # (n_traj,)

n_history = np.array(n_history).T # (n_traj, horizon)
for i in range(n_traj):
    n_rl_traj.append(np.mean(n_history[i, -horizon // 4:]))

print(f"\ng_fb = {g_fb}")
print(f"Optimal: n_bar = {np.mean(n_opt_traj):.3f} ± {np.std(n_opt_traj)/np.sqrt(n_traj):.3f}")
print(f"RL:      n_bar = {np.mean(n_rl_traj):.3f} ± {np.std(n_rl_traj)/np.sqrt(n_traj):.3f}")

# print(f"Optimal force at (1,0): {opt(1, 0):.4f}")
# print(f"Optimal force at (0,1): {opt(0, 1):.4f}")
# print(f"Learned weights: {nn_policy.weight.data}")
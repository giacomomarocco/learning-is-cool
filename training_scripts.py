#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 20:56:42 2026

@author: giacomomarocco
"""

import torch
import torch.nn as nn
import numpy as np
from GaussianOscillator import GaussianOscillator
import copy
import matplotlib.pyplot as plt

horizon = 200

#%%
def raw_rollout(policy, oscillator, horizon, dt, phase_space_range=5.0, window=10):
    """
    Policy sees only (y, ydot) from raw measurement record.
    No Kalman filter. Cost is still on (xc, pc).
    """
    covs, _ = oscillator.variance_solver(n_periods=horizon*dt/(2*np.pi), dt=dt)
    sqrt_term = np.sqrt(8 * oscillator.eta * oscillator.k_jacobs)

    xc = torch.FloatTensor(1).uniform_(-phase_space_range, phase_space_range).squeeze()
    pc = torch.FloatTensor(1).uniform_(-phase_space_range, phase_space_range).squeeze()

    cost = 0.0
    y_prev = torch.tensor(0.0)
    
    for t in range(horizon):
        dW = torch.randn(1).squeeze() * np.sqrt(dt)
        
        # Measurement record (what a detector actually gives you)
        y_current = xc * dt + dW / sqrt_term
        photocurrent = y_current / dt
        ydot = (photocurrent - y_prev) / dt
        y_prev = photocurrent

        # Policy sees only raw measurement signal and its derivative
        obs = torch.stack([photocurrent, ydot])
        u = policy(obs).squeeze()
        
        # Limit the maximum possible force. This will probably come from wanting the trap to remain linear
        # umax = 5.0
        # u = u_max * torch.tanh(policy(state).squeeze())

        # True dynamics (policy doesn't see these directly)
        dpc = (-oscillator.omega**2 * xc * dt
               + sqrt_term * covs[2][t] * dW
               - oscillator.gamma * pc * dt
               + u * dt)
        pc = pc + dpc
        xc = xc + pc * dt + sqrt_term * covs[0][t] * dW
        cost += xc**2 + pc**2

    return cost

raw_policy = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)
raw_optimizer = torch.optim.Adam(raw_policy.parameters(), lr=1e-3)

oscillator = GaussianOscillator()
saved_nn_policies = []


for iteration in range(2500):
    if iteration % 100 == 0:
        saved_nn_policies.append(copy.deepcopy(raw_policy))
        avg = sum(raw_rollout(raw_policy, oscillator, 200, 0.05).item() 
                  for _ in range(50)) / 50
        print(f"Iter {iteration}, avg cost: {avg:.2f}")
    cost = raw_rollout(raw_policy, oscillator, horizon=200, dt=0.05)
    raw_optimizer.zero_grad()
    cost.backward()
    raw_optimizer.step()

        
#%%
with torch.no_grad():
    ps = torch.linspace(-5, 5, 200)
    states = torch.stack([torch.zeros(200), ps], dim=1)

    plt.figure()
    # Plot NN snapshots
    for i, saved in enumerate(saved_nn_policies[-5:]):
        us = saved(states).squeeze().numpy()
        plt.plot(ps.numpy(), us, alpha=0.5, label=f'NN iter {i*500}')

    # Plot final linear policy
    # us_lin = linear_policy(states).squeeze().numpy()
    # plt.plot(ps.numpy(), us_lin, 'k--', linewidth=2, label='Linear')
    plt.xlabel('p')
    plt.ylabel('u')
    plt.legend()
    plt.show()


#%%

def torch_rollout(policy, oscillator, horizon, dt, phase_space_range=5.0):
    covs, _ = oscillator.variance_solver(n_periods=horizon*dt/(2*np.pi), dt=dt)
    xc = torch.FloatTensor(1).uniform_(-phase_space_range, phase_space_range).squeeze()
    pc = torch.FloatTensor(1).uniform_(-phase_space_range, phase_space_range).squeeze()
    cost = 0.0
    sqrt_term = np.sqrt(8 * oscillator.eta * oscillator.k_jacobs)
    for t in range(horizon):
        state = torch.stack([xc, pc])
        u = policy(state).squeeze()
        dW = torch.randn(1).squeeze() * np.sqrt(dt)
        dpc = (-oscillator.omega**2 * xc * dt
               + sqrt_term * covs[2][t] * dW
               - oscillator.gamma * pc * dt
               + u * dt)
        pc = pc + dpc
        xc = xc + pc * dt + sqrt_term * covs[0][t] * dW
        cost += xc**2 + pc**2
    return cost

def evaluate_average_performance(policy, oscillator, n_episodes=100, horizon=200, dt=0.05):
    total_cost = 0.0
    with torch.no_grad():
        for _ in range(n_episodes):
            cost = torch_rollout(policy, oscillator, horizon, dt)
            total_cost += cost.item()
    return total_cost / n_episodes

def get_local_gain(policy, delta=0.1):
    with torch.no_grad():
        u_plus = policy(torch.tensor([0.0, delta])).item()
        u_minus = policy(torch.tensor([0.0, -delta])).item()
        return (u_plus - u_minus) / (2 * delta)

# Neural network policy
nn_policy = nn.Sequential(
    nn.Linear(2, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
)

# Linear policy
linear_policy = nn.Linear(2, 1, bias=False)

nn_optimizer = torch.optim.Adam(nn_policy.parameters(), lr=0.001)
linear_optimizer = torch.optim.Adam(linear_policy.parameters(), lr=0.001)

oscillator = GaussianOscillator()
saved_nn_policies = []

for iteration in range(5000):
    # Train neural network
    cost_nn = torch_rollout(nn_policy, oscillator, horizon=200, dt=0.05)
    nn_optimizer.zero_grad()
    cost_nn.backward()
    nn_optimizer.step()

    # Train linear policy
    cost_lin = torch_rollout(linear_policy, oscillator, horizon=200, dt=0.05)
    linear_optimizer.zero_grad()
    cost_lin.backward()
    linear_optimizer.step()

    if iteration % 500 == 0:
        saved_nn_policies.append(copy.deepcopy(nn_policy))
        with torch.no_grad():
            lin_gain_x = linear_policy.weight[0, 0].item()
            lin_gain_p = linear_policy.weight[0, 1].item()
            nn_gain = get_local_gain(nn_policy)
        avg_cost_nn = evaluate_average_performance(nn_policy, oscillator)
        avg_cost_lin = evaluate_average_performance(linear_policy, oscillator)
        print(f"Iteration {iteration}")
        print(f"  Linear gains: x={lin_gain_x:.4f}, p={lin_gain_p:.4f}, cost={avg_cost_lin:.4f}")
        print(f"  NN local gain on p: {nn_gain:.4f}, cost={avg_cost_nn:.4f}")

#%% Compare policies

with torch.no_grad():
    ps = torch.linspace(-5, 5, 200)
    states = torch.stack([torch.zeros(200), ps], dim=1)

    plt.figure()
    # Plot NN snapshots
    for i, saved in enumerate(saved_nn_policies):
        us = saved(states).squeeze().numpy()
        plt.plot(ps.numpy(), us, alpha=0.5, label=f'NN iter {i*500}')

    # Plot final linear policy
    us_lin = linear_policy(states).squeeze().numpy()
    plt.plot(ps.numpy(), us_lin, 'k--', linewidth=2, label='Linear')
    plt.xlabel('p')
    plt.ylabel('u')
    plt.legend()
    plt.show()

#%% Compare trajectories

def make_feedback(policy):
    def feedback(xc, pc):
        with torch.no_grad():
            state = torch.tensor([xc, pc], dtype=torch.float32)
            return policy(state).squeeze().item()
    return feedback

y_classical, times = oscillator.expectation_solver(
    feedback_fn=lambda xc, pc: -0.1 * pc,
    n_periods=50, dt=0.01
)
y_nn, times = oscillator.expectation_solver(
    feedback_fn=make_feedback(nn_policy),
    n_periods=50, dt=0.01
)
y_lin, times = oscillator.expectation_solver(
    feedback_fn=make_feedback(linear_policy),
    n_periods=50, dt=0.01
)

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
axes[0].plot(times, y_classical[0], label='Classical (g=0.1)')
axes[0].plot(times, y_nn[0], label='NN')
axes[0].plot(times, y_lin[0], label='Learned linear')
axes[0].set_ylabel('x')
axes[0].legend()

axes[1].plot(times, y_classical[1], label='Classical (g=0.1)')
axes[1].plot(times, y_nn[1], label='NN')
axes[1].plot(times, y_lin[1], label='Learned linear')
axes[1].set_ylabel('p')
axes[1].legend()
axes[1].set_xlabel('Time')
plt.show()

#%%
covs, _ = oscillator.variance_solver(n_periods=horizon*0.05/(2*np.pi), dt=0.05)
print(f"var_x range: {covs[0].min():.6f} to {covs[0].max():.6f}")
print(f"cov_xp range: {covs[2].min():.6f} to {covs[2].max():.6f}")
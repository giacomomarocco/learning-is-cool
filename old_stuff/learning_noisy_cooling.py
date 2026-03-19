import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 19:25:32 2026

@author: giacomomarocco
"""

import torch
import torch.nn as nn
import numpy as np
from gaussian_oscillator import GaussianOscillator
import copy

horizon = 200


def torch_rollout(policy, oscillator, horizon, dt, phase_space_range = 5.0):
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
optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
oscillator = GaussianOscillator()

def evaluate_average_performance(policy, oscillator, n_episodes=100, horizon=200, dt=0.05):
    total_cost = 0.0
    with torch.no_grad():
        for _ in range(n_episodes):
            cost = torch_rollout(policy, oscillator, horizon, dt)
            total_cost += cost.item()
    return total_cost / n_episodes


saved_policies = []

for iteration in range(1000):
    cost = torch_rollout(policy, oscillator, horizon=200, dt=0.05)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if iteration % 100 == 0:
        saved_policies.append(copy.deepcopy(policy))
        avg_cost = evaluate_average_performance(policy, oscillator)
        print(f"Iteration {iteration}, avg cost: {avg_cost:.4f}")        
        
        
        
#%%


def learned_feedback(xc, pc):
    with torch.no_grad():
        state = torch.tensor([xc, pc], dtype=torch.float32)
        return policy(state).squeeze().item()

# Compare against classical linear feedback
y_classical, times = oscillator.expectation_solver(
    feedback_fn=lambda xc, pc: -0.1 * pc,
    n_periods=50, dt=0.01
)

y_learned, times = oscillator.expectation_solver(
    feedback_fn=learned_feedback,
    n_periods=50, dt=0.01
)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 6))

axes[0].plot(times, y_classical[0], label='Classical')
axes[0].plot(times, y_learned[0], label='Learned')
axes[0].set_ylabel('x')
axes[0].legend()

axes[1].plot(times, y_classical[1], label='Classical')
axes[1].plot(times, y_learned[1], label='Learned')
axes[1].set_ylabel('p')
axes[1].legend()
axes[1].set_xlabel('Time')

plt.show()

with torch.no_grad():
    ps = torch.linspace(-5, 5, 100)
    us = [policy(torch.tensor([0.0, p])).item() for p in ps]
    plt.plot(ps.numpy(), us)
    plt.xlabel('p')
    plt.ylabel('u')
    plt.title('Policy at x=0')
    plt.show()
    
    #%%
    
with torch.no_grad():
    ps = torch.linspace(-5, 5, 200)
    states = torch.stack([torch.zeros(200), ps], dim=1)

    plt.figure()
    for i, saved in enumerate(saved_policies):
        us = saved(states).squeeze().numpy()
        plt.plot(ps.numpy(), us, label=f'iter {i*100}')
    plt.xlabel('p')
    plt.ylabel('u')
    plt.legend()
    plt.show()
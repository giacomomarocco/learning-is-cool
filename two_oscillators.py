#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 11:14:54 2026

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
def torch_rollout_two(policy, osc1, osc2, horizon, dt, phase_space_max = 20.0, u_max=5.0):
    covs1, _ = osc1.variance_solver(n_periods=horizon*dt/(2*np.pi), dt=dt)
    covs2, _ = osc2.variance_solver(n_periods=horizon*dt/(2*np.pi), dt=dt)

    xc1 = torch.FloatTensor(1).uniform_(-phase_space_max, phase_space_max).squeeze()
    pc1 = torch.FloatTensor(1).uniform_(-phase_space_max, phase_space_max).squeeze()
    xc2 = torch.FloatTensor(1).uniform_(-phase_space_max, phase_space_max).squeeze()
    pc2 = torch.FloatTensor(1).uniform_(-phase_space_max, phase_space_max).squeeze()
    cost = 0.0

    sqrt_term1 = np.sqrt(8 * osc1.eta * osc1.k_jacobs)
    sqrt_term2 = np.sqrt(8 * osc2.eta * osc2.k_jacobs)

    for t in range(horizon):
        state = torch.stack([xc1, pc1, xc2, pc2])
        raw_u = policy(state).squeeze()
        u = u_max * torch.tanh(raw_u)

        dW1 = torch.randn(1).squeeze() * np.sqrt(dt)
        dW2 = torch.randn(1).squeeze() * np.sqrt(dt)

        # Oscillator 1
        dpc1 = (-osc1.omega**2 * xc1 * dt
                + sqrt_term1 * covs1[2][t] * dW1
                - osc1.gamma * pc1 * dt
                + u * dt)
        pc1 = pc1 + dpc1
        xc1 = xc1 + pc1 * dt + sqrt_term1 * covs1[0][t] * dW1

        # Oscillator 2
        dpc2 = (-osc2.omega**2 * xc2 * dt
                + sqrt_term2 * covs2[2][t] * dW2
                - osc2.gamma * pc2 * dt
                + u * dt)
        pc2 = pc2 + dpc2
        xc2 = xc2 + pc2 * dt + sqrt_term2 * covs2[0][t] * dW2

        cost += xc1**2 + pc1**2 + xc2**2 + pc2**2

    return cost

osc1 = GaussianOscillator(omega=1.0)
osc2 = GaussianOscillator(omega=1.1)

policy = nn.Sequential(
    nn.Linear(4, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
)

optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

#%%
saved_nn_policies = []


for iteration in range(2400):
    cost = torch_rollout_two(policy, osc1, osc2, horizon=200, dt=0.05)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if iteration % 200 == 0:
        saved_nn_policies.append(copy.deepcopy(policy))
        avg = sum(torch_rollout_two(policy, osc1, osc2, 200, 0.05).item() 
                  for _ in range(50)) / 50
        print(f"Iter {iteration}, avg cost: {avg:.2f}")
        
#%%
def torch_rollout_batched(policy, osc1, osc2, horizon, dt, u_max=5.0,  phase_space_max = 20.0, batch_size=32):
    covs1, _ = osc1.variance_solver(n_periods=horizon*dt/(2*np.pi), dt=dt)
    covs2, _ = osc2.variance_solver(n_periods=horizon*dt/(2*np.pi), dt=dt)

    xc1 = torch.FloatTensor(batch_size).uniform_(-phase_space_max, phase_space_max)
    pc1 = torch.FloatTensor(batch_size).uniform_(-phase_space_max, phase_space_max)
    xc2 = torch.FloatTensor(batch_size).uniform_(-phase_space_max, phase_space_max)
    pc2 = torch.FloatTensor(batch_size).uniform_(-phase_space_max, phase_space_max)
    cost = torch.zeros(batch_size)

    sqrt_term1 = np.sqrt(8 * osc1.eta * osc1.k_jacobs)
    sqrt_term2 = np.sqrt(8 * osc2.eta * osc2.k_jacobs)

    for t in range(horizon):
        state = torch.stack([xc1, pc1, xc2, pc2], dim=1)
        raw_u = policy(state).squeeze(-1)
        u = u_max * torch.tanh(raw_u)

        dW1 = torch.randn(batch_size) * np.sqrt(dt)
        dW2 = torch.randn(batch_size) * np.sqrt(dt)

        dpc1 = (-osc1.omega**2 * xc1 * dt
                + sqrt_term1 * covs1[2][t] * dW1
                - osc1.gamma * pc1 * dt
                + u * dt)
        pc1 = pc1 + dpc1
        xc1 = xc1 + pc1 * dt + sqrt_term1 * covs1[0][t] * dW1

        dpc2 = (-osc2.omega**2 * xc2 * dt
                + sqrt_term2 * covs2[2][t] * dW2
                - osc2.gamma * pc2 * dt
                + u * dt)
        pc2 = pc2 + dpc2
        xc2 = xc2 + pc2 * dt + sqrt_term2 * covs2[0][t] * dW2

        cost += xc1**2 + pc1**2 + xc2**2 + pc2**2

    return cost.mean()

saved_nn_policies = []


for iteration in range(1000):
    cost = torch_rollout_batched(policy, osc1, osc2, horizon=200, dt=0.05, batch_size=16)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if iteration % 200 == 0:
        saved_nn_policies.append(copy.deepcopy(policy))
        avg = sum(torch_rollout_two(policy, osc1, osc2, 200, 0.05).item() 
                  for _ in range(10)) / 10
        print(f"Iter {iteration}, avg cost: {avg:.2f}")

#%%


def greedy_rollout_batched(g, osc1, osc2, horizon, dt,  phase_space_max = 20.0, u_max=5.0, batch_size=32):
    covs1, _ = osc1.variance_solver(n_periods=horizon*dt/(2*np.pi), dt=dt)
    covs2, _ = osc2.variance_solver(n_periods=horizon*dt/(2*np.pi), dt=dt)


    xc1 = torch.FloatTensor(batch_size).uniform_(-phase_space_max, phase_space_max)
    pc1 = torch.FloatTensor(batch_size).uniform_(-phase_space_max, phase_space_max)
    xc2 = torch.FloatTensor(batch_size).uniform_(-phase_space_max, phase_space_max)
    pc2 = torch.FloatTensor(batch_size).uniform_(-phase_space_max, phase_space_max)
    cost = torch.zeros(batch_size)

    sqrt_term1 = np.sqrt(8 * osc1.eta * osc1.k_jacobs)
    sqrt_term2 = np.sqrt(8 * osc2.eta * osc2.k_jacobs)

    for t in range(horizon):
        u = u_max * torch.tanh(-g * (pc1 + pc2))

        dW1 = torch.randn(batch_size) * np.sqrt(dt)
        dW2 = torch.randn(batch_size) * np.sqrt(dt)

        dpc1 = (-osc1.omega**2 * xc1 * dt
                + sqrt_term1 * covs1[2][t] * dW1
                - osc1.gamma * pc1 * dt
                + u * dt)
        pc1 = pc1 + dpc1
        xc1 = xc1 + pc1 * dt + sqrt_term1 * covs1[0][t] * dW1

        dpc2 = (-osc2.omega**2 * xc2 * dt
                + sqrt_term2 * covs2[2][t] * dW2
                - osc2.gamma * pc2 * dt
                + u * dt)
        pc2 = pc2 + dpc2
        xc2 = xc2 + pc2 * dt + sqrt_term2 * covs2[0][t] * dW2

        cost += xc1**2 + pc1**2 + xc2**2 + pc2**2

    return cost.mean()


g = torch.tensor(1.0, requires_grad=True)
g_optimizer = torch.optim.Adam([g], lr=0.01)

for iteration in range(2000):
    cost = greedy_rollout_batched(g, osc1, osc2, horizon=200, dt=0.05, batch_size=64)
    g_optimizer.zero_grad()
    cost.backward()
    g_optimizer.step()

    if iteration % 200 == 0:
        avg = sum(greedy_rollout_batched(g, osc1, osc2, 200, 0.05, batch_size=1).item()
                  for _ in range(50)) / 50
        print(f"Iter {iteration}, g: {g.item():.4f}, avg cost: {avg:.2f}")


#%%
# Final comparison
avg_nn = sum(torch_rollout_batched(policy, osc1, osc2, 200, 0.05, batch_size=1).item()
             for _ in range(100)) / 100
avg_greedy = sum(greedy_rollout_batched(g, osc1, osc2, 200, 0.05, batch_size=1).item()
                 for _ in range(100)) / 100
print(f"NN cost:     {avg_nn:.2f}")
print(f"Greedy cost: {avg_greedy:.2f}")
print(f"Optimal g:   {g.item():.4f}")
        
#%%
def simulate_trajectory(policy, osc1, osc2, n_periods=50, dt=0.01, u_max=5.0):
    covs1, _ = osc1.variance_solver(n_periods=n_periods, dt=dt)
    covs2, _ = osc2.variance_solver(n_periods=n_periods, dt=dt)

    n_times = int(2 * np.pi * n_periods / dt)

    xc1 = torch.tensor(15.0)
    pc1 = torch.tensor(0.0)
    xc2 = torch.tensor(-1.0)
    pc2 = torch.tensor(13.0)

    sqrt_term1 = np.sqrt(8 * osc1.eta * osc1.k_jacobs)
    sqrt_term2 = np.sqrt(8 * osc2.eta * osc2.k_jacobs)

    x1_history = []
    x2_history = []
    u_history = []
    times = []

    with torch.no_grad():
        for t in range(n_times):
            x1_history.append(xc1.item())
            x2_history.append(xc2.item())
            times.append(t * dt)

            state = torch.stack([xc1, pc1, xc2, pc2])
            raw_u = policy(state).squeeze()
            u = u_max * torch.tanh(raw_u)
            u_history.append(u.item())

            dW1 = torch.randn(1).squeeze() * np.sqrt(dt)
            dW2 = torch.randn(1).squeeze() * np.sqrt(dt)

            dpc1 = (-osc1.omega**2 * xc1 * dt
                    + sqrt_term1 * covs1[2][t] * dW1
                    - osc1.gamma * pc1 * dt
                    + u * dt)
            pc1 = pc1 + dpc1
            xc1 = xc1 + pc1 * dt + sqrt_term1 * covs1[0][t] * dW1

            dpc2 = (-osc2.omega**2 * xc2 * dt
                    + sqrt_term2 * covs2[2][t] * dW2
                    - osc2.gamma * pc2 * dt
                    + u * dt)
            pc2 = pc2 + dpc2
            xc2 = xc2 + pc2 * dt + sqrt_term2 * covs2[0][t] * dW2

    return np.array(times), np.array(x1_history), np.array(x2_history), np.array(u_history)


times, x1, x2, u = simulate_trajectory(policy, osc1, osc2)
times, x1G, x2G, uG = simulate_trajectory(policy, osc1, osc2)


fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(times, x1, label=f'Oscillator 1 (ω={osc1.omega})')
axes[0].set_ylabel('Oscillator position')

axes[0].plot(times, x2, label=f'Oscillator 2 (ω={osc2.omega})')
axes[0].legend()

axes[1].plot(times, u, label='Applied force')
axes[1].set_ylabel('u')
axes[1].set_xlabel('Time')
axes[1].legend()

plt.tight_layout()
plt.show()

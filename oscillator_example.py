#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 15:46:04 2026

@author: giacomomarocco
"""
import torch
import torch.nn as nn
import numpy as np

# Parameters
omega_n = 1.0
dt = 0.05
horizon = 200

def step(state, u):
    x = state[0]
    x_dot = state[1]
    x_ddot = -omega_n**2 * x + u
    new_x = x + x_dot * dt
    new_x_dot = x_dot + x_ddot * dt
    return torch.stack([new_x, new_x_dot])

# Input is 2D vector of x, v
#Output is 1D vector of energy
policy = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

for iteration in range(500):
    x0 = torch.FloatTensor(1).uniform_(-2.0, 2.0)
    xdot0 = torch.FloatTensor(1).uniform_(-2.0, 2.0)
    state = torch.cat([x0, xdot0])
    cost = 0.

    for t in range(horizon):
        u = policy(state).squeeze()
        state = step(state, u)
        cost += state[0]**2 + state[1]**2 + 0.01 * u**2

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if iteration % 50 == 0:
        print(f"Iteration {iteration}, cost: {cost.item():.4f}")
        
#%%
with torch.no_grad():
    state = torch.tensor([1.0, 0.0])
    states = [state]
    actions = []

    for t in range(horizon):
        u = policy(state).squeeze()
        state = step(state, u)
        states.append(state)
        actions.append(u)

    states = torch.stack(states)
    actions = torch.stack(actions)

import matplotlib.pyplot as plt

time = [i * dt for i in range(horizon + 1)]

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, states[:, 0].numpy(), label='x')
plt.plot(time, states[:, 1].numpy(), label='x_dot')
plt.legend()
plt.ylabel('State')

plt.subplot(2, 1, 2)
plt.plot(time[:-1], np.log(np.abs(actions.numpy()/states[:-1, 1].numpy())), label='u/v')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Control force')

plt.show()
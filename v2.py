#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:03:43 2026

@author: giacomomarocco
"""

import torch
import torch.nn as nn
import numpy as np
from GaussianOscillator import GaussianOscillator
import copy
import matplotlib.pyplot as plt


class CoupledOscillatorSim:
    def __init__(self, oscillators, dt=0.05, u_max=5.0):
        self.oscillators = oscillators
        self.n_osc = len(oscillators)
        self.dt = dt
        self.u_max = u_max
        
        self.sqrt_terms = [np.sqrt(8 * osc.eta * osc.k_jacobs) for osc in oscillators]
    
    def precompute_covs(self, horizon):
        self.covs = []
        for osc in self.oscillators:
            covs, _ = osc.variance_solver(n_periods=horizon*self.dt/(2*np.pi), dt=self.dt)
            self.covs.append(covs)
    
    def step(self, state, u_raw, t):
        u = self.u_max * torch.tanh(u_raw)
        new_state = torch.zeros_like(state)
        
        for i in range(self.n_osc):
            x = state[..., 2*i]
            p = state[..., 2*i+1]
            osc = self.oscillators[i]
            
            dW = torch.randn_like(x) * np.sqrt(self.dt)
            
            dp = (-osc.omega**2 * x * self.dt
                  + self.sqrt_terms[i] * self.covs[i][2][t] * dW
                  - osc.gamma * p * self.dt
                  + u * self.dt)
            new_p = p + dp
            new_x = x + new_p * self.dt + self.sqrt_terms[i] * self.covs[i][0][t] * dW
            
            new_state[..., 2*i] = new_x
            new_state[..., 2*i+1] = new_p
        
        return new_state
    
    def rollout(self, policy, horizon, batch_size=1, phase_space_max=20.0):
        self.precompute_covs(horizon)
        state = torch.FloatTensor(batch_size, 2*self.n_osc).uniform_(-phase_space_max, phase_space_max)
        cost = torch.zeros(batch_size)
        
        for t in range(horizon):
            u_raw = policy(state).squeeze(-1)
            state = self.step(state, u_raw, t)
            cost += (state**2).sum(dim=-1)
        
        return cost.mean()
    
    def simulate(self, policy, horizon, initial_state=None):
        self.precompute_covs(horizon)
        if initial_state is None:
            initial_state = torch.zeros(2*self.n_osc)
        state = initial_state.unsqueeze(0)
        
        states = []
        forces = []
        
        with torch.no_grad():
            for t in range(horizon):
                states.append(state.squeeze(0).numpy().copy())
                u_raw = policy(state).squeeze(-1)
                forces.append((self.u_max * torch.tanh(u_raw)).item())
                state = self.step(state, u_raw, t)
        
        return np.array(states), np.array(forces)
    
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


sim = CoupledOscillatorSim([osc1, osc2], dt=0.05, u_max=5.0)

# Train NN
for iteration in range(1000):
    cost = sim.rollout(policy, horizon=200, batch_size=64)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if iteration % 200 == 0:
        # saved_nn_policies.append(copy.deepcopy(policy))
        avg = sum(sim.rollout(policy, horizon=200, batch_size=1).item()
                  for _ in range(50)) / 50
        print(f"NN Iter {iteration}, avg cost: {avg:.2f}")
        


g = torch.tensor(1.0, requires_grad=True)
g_optimizer = torch.optim.Adam([g], lr=0.01)

greedy = lambda state: -g * (state[..., 1] + state[..., 3]).unsqueeze(-1)
for iteration in range(1000):
    cost = sim.rollout(greedy, horizon=200, batch_size=64)
    g_optimizer.zero_grad()
    cost.backward()
    g_optimizer.step()
    
    

    if iteration % 200 == 0:
        avg = sum(sim.rollout(greedy, horizon=200, batch_size=1).item()
                  for _ in range(50)) / 50
        print(f"Greedy Iter {iteration}, g: {g.item():.4f}, avg cost: {avg:.2f}")

# Compare

avg_nn = sum(sim.rollout(policy, horizon=200, batch_size=1).item()
             for _ in range(100)) / 100
avg_greedy = sum(sim.rollout(greedy, horizon=200, batch_size=1).item()
                 for _ in range(100)) / 100
print(f"NN cost:     {avg_nn:.2f}")
print(f"Greedy cost: {avg_greedy:.2f}")
print(f"Optimal g:   {g.item():.4f}")

#%%

initial = torch.tensor([15.0, 0.0, -1.0, 13.0])

torch.manual_seed(42)
states_nn, forces_nn = sim.simulate(policy, horizon=10000, initial_state=initial)
torch.manual_seed(42)
states_gr, forces_gr = sim.simulate(greedy, horizon=10000, initial_state=initial)

times = np.arange(len(forces_nn)) * sim.dt

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axes[0].plot(times, states_nn[:, 0], label='Osc 1 (NN)')
axes[0].plot(times, states_gr[:, 0], '--', label='Osc 1 (Greedy)')
axes[0].set_ylabel('x₁')
axes[0].legend()

axes[1].plot(times, states_nn[:, 2], label='Osc 2 (NN)')
axes[1].plot(times, states_gr[:, 2], '--', label='Osc 2 (Greedy)')
axes[1].set_ylabel('x₂')
axes[1].legend()

axes[2].plot(times, forces_nn, label='Force (NN)')
axes[2].plot(times, forces_gr, '--', label='Force (Greedy)')
axes[2].set_ylabel('u')
axes[2].set_xlabel('Time')
axes[2].legend()

plt.tight_layout()
plt.show()
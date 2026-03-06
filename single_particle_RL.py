#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:24:18 2026

@author: giacomomarocco
"""

from gaussian_oscillator import GaussianOscillator
import numpy as np
import matplotlib.pyplot as plt
import os
from optimal_control import FeedbackForces
import torch
import torch.nn as nn
import copy

dirname = os.path.dirname(os.path.abspath(__file__))

initial_temperature = 10
gamma_BA = 18.8/104 # Value used in Magrini et al.
g_fb = 0.2
# kappa = 5e-2
g_fb_str = f"{g_fb}".replace(".", "")
eta = 0.2
n_min = (eta**(-1/2) - 1)/2
dt = 0.05
n_th = 10
horizon= 600

osc = GaussianOscillator(n_thermal = n_th, gamma_meas= gamma_BA, eta = eta, n_periods=horizon*dt/(2*np.pi))

fb = FeedbackForces(g_fb=g_fb, omega=osc.omega)

#%%
covs = osc.variance_solver()

def torch_rollout(policy, oscillator, horizon, batch_size = 256, phase_space_range = n_th):
    xc = torch.FloatTensor(batch_size).uniform_(-phase_space_range, phase_space_range).squeeze()
    pc = torch.FloatTensor(batch_size).uniform_(-phase_space_range, phase_space_range).squeeze()
    cost = 0.0
    sqrt_term = np.sqrt(2 * oscillator.eta * oscillator.gamma_meas)
    for t in range(horizon):
        state = torch.stack([xc, pc], dim = 1) # Create stack of (xc, pc) pairs
        u = policy(state).squeeze()
        dW = torch.randn(batch_size).squeeze() * np.sqrt(dt)
        dpc = (-oscillator.omega**2 * xc * dt
               + sqrt_term * covs['Cxp'][t] * dW
               # - oscillator.gamma * pc * dt
               + u * dt)
        pc = pc + dpc
        xc = xc + pc * dt + sqrt_term * covs['Vxx'][t] * dW
        cost += xc**2 + pc**2 + (g_fb)**(-2) * u**2
    return cost.mean()

def evaluate_average_performance(policy, oscillator, n_episodes=100, horizon=200, dt=0.05):
    total_cost = 0.0
    with torch.no_grad():
        for _ in range(n_episodes):
            cost = torch_rollout(policy, oscillator, horizon)
            total_cost += cost.item()
    return total_cost / n_episodes

# Neural network policy
nn_policy = nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    nn.Linear(16, 16),
    nn.Tanh(),
    nn.Linear(16, 1)
)

# nn_policy = nn.Linear(2, 1, bias=False)



nn_optimizer = torch.optim.Adam(nn_policy.parameters(), lr=0.0005)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(nn_optimizer, patience=50, factor=0.5)
#%%
saved_nn_policies = []

for iteration in range(2500):
    cost = torch_rollout(nn_policy, osc, horizon=horizon)
    nn_optimizer.zero_grad()
    cost.backward()
    nn_optimizer.step()
    
    # old_lr = nn_optimizer.param_groups[0]['lr']
    # scheduler.step(cost.item())
    # new_lr = nn_optimizer.param_groups[0]['lr']
    # if new_lr < old_lr:
    #     print(f"Iter {iteration}: lr reduced from {old_lr:.6f} to {new_lr:.6f}")    
    if iteration % 200 == 0:
        saved_nn_policies.append(copy.deepcopy(nn_policy))
        with torch.no_grad():
            avg = sum(torch_rollout(nn_policy, osc, 200).item() 
                      for _ in range(10)) / 10
        print(f"Iter {iteration}, avg cost: {avg:.2f}")

#%%
def simulate_trajectory(policy, oscillator, covs, horizon, dt=0.05, phase_space_range=n_th):
    xc = torch.FloatTensor(1).uniform_(-phase_space_range, phase_space_range).squeeze()
    pc = torch.FloatTensor(1).uniform_(-phase_space_range, phase_space_range).squeeze()
    x0 = xc.item()
    p0 = pc.item()
    sqrt_term = np.sqrt(2 * oscillator.eta * oscillator.gamma_meas)

    x_history = []
    p_history = []
    u_history = []
    times = []
    osc = GaussianOscillator(n_thermal = n_th, gamma_meas= gamma_BA, eta = eta, dt = dt, n_periods=horizon*dt/(2*np.pi))

    covs = osc.variance_solver()
    with torch.no_grad():
        for t in range(horizon):
            x_history.append(xc.item())
            p_history.append(pc.item())
            times.append(t * dt)

            state = torch.stack([xc, pc])
            u = policy(state).squeeze()
            u_history.append(u.item())

            dW = torch.randn(1).squeeze() * np.sqrt(dt)

            dpc = (-oscillator.omega**2 * xc * dt
                   + sqrt_term * covs['Cxp'][t] * dW
                   + u * dt)
            pc = pc + dpc
            xc = xc + pc * dt + sqrt_term * covs['Vxx'][t] * dW
    
    x_opt = osc.expectation_solver(initial_conditions = [x0,p0, 0])['xc']

    return np.array(times), np.array(x_history), np.array(p_history), x_opt 


times, x, p, x_opt = simulate_trajectory(nn_policy, osc, covs, 10000)


plt.plot(times, x, label='RL')
plt.plot(times, x_opt, label='optimal')
plt.legend()
plt.tight_layout()
plt.show()

#%%

covs = osc.variance_solver()
n_times = len(covs['Vxx'])
sqrt_term = np.sqrt(2 * osc.eta * osc.gamma_meas)

n_traj = 100
initial_conditions = np.random.randn(n_traj, 3)
initial_conditions[:, 0] *= np.sqrt(initial_temperature + 0.5)
initial_conditions[:, 1] *= np.sqrt(initial_temperature + 0.5)
initial_conditions[:, 2] = 0

fb = FeedbackForces(g_fb=g_fb, omega=osc.omega)

n_opt_traj = []
for i in range(n_traj):
    result = osc.expectation_solver(feedback_fn=fb.optimal_feedback(),
                                    initial_conditions=initial_conditions[i],
                                    gamma_fb=g_fb, covariances=covs)
    n_bar = osc.find_n_bar(result['xc'], result['pc'], covs['Vxx'], covs['Vpp'])
    n_opt_traj.append(np.mean(n_bar[-len(n_bar) // 4:]))

# RL — all trajectories at once
xc = torch.tensor(initial_conditions[:, 0], dtype=torch.float32)  # (n_traj,)
pc = torch.tensor(initial_conditions[:, 1], dtype=torch.float32)  # (n_traj,)
sqrt_term = np.sqrt(2 * osc.eta * osc.gamma_meas)

x_rl = torch.zeros(n_traj, n_times)
p_rl = torch.zeros(n_traj, n_times)

with torch.no_grad():
    for t in range(n_times):
        x_rl[:, t] = xc
        p_rl[:, t] = pc

        state = torch.stack([xc, pc], dim=-1)  # (n_traj, 2)
        u = nn_policy(state).squeeze(-1)        # (n_traj,)

        dW = torch.randn(n_traj) * np.sqrt(osc.dt)

        dpc = (-osc.omega ** 2 * xc * osc.dt
               + sqrt_term * covs['Cxp'][t] * dW
               + u * osc.dt)
        pc = pc + dpc
        xc = xc + pc * osc.dt + sqrt_term * covs['Vxx'][t] * dW

x_rl = x_rl.numpy()
p_rl = p_rl.numpy()

n_rl_traj = []
for i in range(n_traj):
    n_bar = osc.find_n_bar(x_rl[i], p_rl[i], covs['Vxx'], covs['Vpp'])
    n_rl_traj.append(np.mean(n_bar[-len(n_bar) // 4:]))

print(f"g_fb = {g_fb}")
print(f"Optimal: n_bar = {np.mean(n_opt_traj):.3f} ± {np.std(n_opt_traj)/np.sqrt(n_traj):.3f}")
print(f"RL:      n_bar = {np.mean(n_rl_traj):.3f} ± {np.std(n_rl_traj)/np.sqrt(n_traj):.3f}")
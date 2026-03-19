#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 19:03:41 2026

@author: giacomomarocco
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/giacomomarocco/Code/2026/learn_to_cool/src/learn_to_cool')
# Parameters
omega = 1.0
eta = 0.2
gamma_BA = 18.8 / 104
g_fb = 0.2
n_th = 10
dt = 0.05
q_cost = g_fb**(-2)

sqrt_2eta_gm = np.sqrt(2 * eta * gamma_BA)

# Steady-state covariances (you may need to adjust these to match your simulation)
# For now, compute them from your GaussianOscillator
# Placeholder: use the no-feedback steady state
# You should replace these with the actual values from osc
from gaussian_oscillator import GaussianOscillator
osc = GaussianOscillator(omega=omega, n_thermal=n_th, gamma_meas=gamma_BA, eta=eta)
res = osc.variance_solver(n_periods=50, dt=dt)
Vxx = res['Vxx'][-1]
Cxp = res['Cxp'][-1]

# Analytical optimum
from optimal_control import FeedbackForces
fb = FeedbackForces(g_fb=g_fb, omega=osc.omega)
opt = fb.optimal_feedback()
w1_opt = opt(1, 0)
w2_opt = opt(0, 1)
print(f"Analytical optimal: w1 = {w1_opt:.4f}, w2 = {w2_opt:.4f}")

# Simulation
def evaluate_cost(w1, w2, n_traj=5000, horizon=300):
    xc = np.random.randn(n_traj) * np.sqrt(n_th + 0.5)
    pc = np.random.randn(n_traj) * np.sqrt(n_th + 0.5)
    
    total_cost = np.zeros(n_traj)
    for t in range(horizon):
        u = w1 * xc + w2 * pc
        state_cost = omega * (xc**2 + pc**2)
        total_cost += state_cost + q_cost * u**2
        
        dW = np.random.randn(n_traj) * np.sqrt(dt)
        noise_x = sqrt_2eta_gm * Vxx * dW
        noise_p = sqrt_2eta_gm * Cxp * dW
        
        new_xc = xc + omega * pc * dt + noise_x
        new_pc = pc + (-omega * xc + g_fb * u) * dt + noise_p
        xc = new_xc
        pc = new_pc
    
    return np.mean(total_cost)

# Sweep
n_grid = 31
w1_range = np.linspace(w1_opt - 1.0, w1_opt + 1.0, n_grid)
w2_range = np.linspace(w2_opt - 1.0, w2_opt + 1.0, n_grid)
cost_grid = np.zeros((n_grid, n_grid))

for i, w1 in enumerate(w1_range):
    for j, w2 in enumerate(w2_range):
        cost_grid[i, j] = evaluate_cost(w1, w2)
    print(f"Row {i+1}/{n_grid} done")

# Find numerical minimum
i_min, j_min = np.unravel_index(np.argmin(cost_grid), cost_grid.shape)
w1_num_opt = w1_range[i_min]
w2_num_opt = w2_range[j_min]
print(f"Numerical optimum:  w1 = {w1_num_opt:.4f}, w2 = {w2_num_opt:.4f}")
print(f"Analytical optimum: w1 = {w1_opt:.4f}, w2 = {w2_opt:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
cs = ax.contourf(w2_range, w1_range, cost_grid, levels=50, cmap='viridis')
plt.colorbar(cs, ax=ax, label='Cost')
ax.plot(w2_opt, w1_opt, 'r*', markersize=15, label='Analytical')
ax.plot(w2_num_opt, w1_num_opt, 'wx', markersize=15, label='Numerical')
ax.set_xlabel('$w_2$ (momentum)')
ax.set_ylabel('$w_1$ (position)')
ax.set_title('Cost landscape')
ax.legend()
plt.tight_layout()
plt.savefig('cost_landscape.png', dpi=150)
plt.show()
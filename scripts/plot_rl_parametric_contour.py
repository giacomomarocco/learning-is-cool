#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizes the RL-learned parametric feedback policy as a contour plot in phase space.
Loads weights from weights/parametric_2d.pth.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Add src to path to import learn_to_cool
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src'))
from learn_to_cool.gaussian_oscillator import GaussianOscillator
from learn_to_cool.torch_oscillator_env import TorchOscillatorEnv

def plot_rl_contour():
    dirname = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(os.path.dirname(dirname), 'weights', 'parametric_2d.pth')

    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        print("Please run scripts/parametric_RL.py first to train the policy.")
        return

    # 1. Define Policy Architecture (matching parametric_RL.py)
    policy = nn.Sequential(
        nn.Linear(2, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1)
    )

    # 2. Load Weights
    policy.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    policy.eval()
    print(f"Loaded weights from {weights_path}")

    # 3. Parameters (matching parametric_RL.py defaults)
    n_th = 10
    modulation_depth = 0.5
    dt = 0.05
    sim_horizon = 2000

    osc = GaussianOscillator(n_thermal=n_th, gamma_meas=18.8/104, eta=0.2)
    env = TorchOscillatorEnv(osc, batch_size=1, dt=dt, mode='parametric',
                           modulation_depth=modulation_depth)

    # 4. Generate Grid for Contour
    limit = 6.0 # Range for x and p
    n_grid = 100
    x_grid = np.linspace(-limit, limit, n_grid)
    p_grid = np.linspace(-limit, limit, n_grid)
    X, P = np.meshgrid(x_grid, p_grid)

    # Prepare input tensor: shape (n_grid*n_grid, 2)
    grid_points = np.stack([X.ravel(), P.ravel()], axis=1)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

    # 5. Compute Policy Output
    with torch.no_grad():
        u_raw = policy(grid_tensor).squeeze(-1)
        # Apply the same scaling as in the environment
        u_mod = modulation_depth * torch.tanh(u_raw)
        U = u_mod.numpy().reshape(X.shape)

    # 6. Simulate a short trajectory to overlay
    state = env.reset()
    # Start from a specific point for a nice spiral
    env.xc = torch.tensor([[4.0]])
    env.pc = torch.tensor([[0.0]])

    traj_x = []
    traj_p = []
    with torch.no_grad():
        for _ in range(sim_horizon):
            traj_x.append(env.xc.item())
            traj_p.append(env.pc.item())
            u_step = policy(env.state())
            env.step(u_step)

    # 7. Plotting
    plt.figure(figsize=(9, 7))
    contour = plt.contourf(X, P, U, cmap='RdBu_r', levels=100)
    cbar = plt.colorbar(contour)
    cbar.set_label(r'$\delta\omega^2/\omega_0^2$')

    plt.plot(traj_x, traj_p, color='black', alpha=0.6, lw=1.0, label='RL Trajectory')

    plt.xlabel(r'$\langle x \rangle_c$')
    plt.ylabel(r'$\langle p \rangle_c$')
    plt.title('RL-based Parametric Feedback Policy')
    plt.grid(True, alpha=0.2)
    plt.legend(loc='upper right')

    # Save the figure
    fig_dir = os.path.join(os.path.dirname(dirname), 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    save_path = os.path.join(fig_dir, 'rl_parametric_contour.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Contour plot saved to {save_path}")

if __name__ == "__main__":
    plot_rl_contour()

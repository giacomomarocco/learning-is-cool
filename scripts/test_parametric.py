#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for parametric feedback in GaussianOscillator.
Outputs x(t) and a contour plot of delta_omega^2/omega^2.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path to import learn_to_cool
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from learn_to_cool.gaussian_oscillator import GaussianOscillator

def run_test():
    # Parameters
    omega = 1.0
    epsilon = 0.5
    dt = 0.01
    n_periods = 20
    initial_temp = 100
    phi_0 = np.pi/2

    osc = GaussianOscillator(omega=omega, n_thermal=initial_temp, gamma_meas=0.05, eta=1.0)

    # Run simulation
    # expectation_solver returns dict
    init_cond = np.array([np.sqrt(initial_temp), 0, 0])
    res = osc.expectation_solver(
        parametric=True,
        epsilon=epsilon,
        dt=dt,
        n_periods=n_periods,
        initial_conditions=init_cond
    )

    x_traj = res['xc']
    p_traj = res['pc']
    times = res['times']
    var_x = res['Vxx']
    var_p = res['Vpp']
    cov_xp = res['Cxp']

    # Calculate n_bar
    n_bar = (x_traj**2 + var_x + p_traj**2 + var_p) / 4 - 0.5

    # Create figures directory if it doesn't exist
    fig_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Plot 1: x(t) Trajectory
    plt.figure(figsize=(10, 4))
    plt.plot(times, x_traj, label='x(t)')
    plt.xlabel('Time ($\omega t$)')
    plt.ylabel('Position x')
    plt.title(f'Parametric Feedback Trajectory ($\epsilon={epsilon}$)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, 'parametric_xt.png'))
    plt.close()

    # Plot 2: n_bar(t)
    plt.figure(figsize=(10, 4))
    plt.plot(times, n_bar, label='$\\\\bar{n}(t)$')
    plt.yscale('log')
    plt.xlabel('Time ($\omega t$)')
    plt.ylabel('Phonon number $\\\\bar{n}$')
    plt.title('Cooling performance (n_bar)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, 'parametric_nbar.png'))
    plt.close()

    # Plot 3: Contour plot of delta_omega^2 / omega^2 vs (x, p)
    # At t=0 for simplicity, to see the spatial dependence
    x_grid = np.linspace(-max(abs(x_traj)), max(abs(x_traj)), 100)
    p_grid = np.linspace(-max(abs(p_traj)), max(abs(p_traj)), 100)
    X, P = np.meshgrid(x_grid, p_grid)

    t_fixed = 0.0
    # Replicate the logic from expectation_solver
    X_slow = X * np.cos(omega * t_fixed) - P * np.sin(omega * t_fixed)
    Y_slow = X * np.sin(omega * t_fixed) + P * np.cos(omega * t_fixed)
    phi_pll = np.arctan2(-Y_slow, X_slow)
    u_param = epsilon * np.cos(2 * omega * t_fixed + 2 * phi_pll + phi_0)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, P, u_param, cmap='RdBu_r', levels=100)
    plt.colorbar(contour, label='$\delta \omega^2 / \omega^2$')
    plt.xlabel('x')
    plt.ylabel('p')
    plt.title(f'Parametric Modulation')

    # Overlay the trajectory
    plt.plot(x_traj[100:5000], p_traj[100:5000], color='black', alpha=0.5, lw=1, label='Trajectory')
    # plt.legend()

    plt.savefig(os.path.join(fig_dir, 'parametric_contour.png'))
    plt.close()

    print(f"Test completed. Figures saved to {fig_dir}")

if __name__ == "__main__":
    run_test()

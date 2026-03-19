#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:47:50 2026

@author: giacomomarocco
"""

#!/usr/bin/env python3
"""
Simulate N coupled oscillators with common optimal LQR feedback
and plot all conditional mean positions on one figure.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'learn_to_cool'))

import numpy as np
import matplotlib.pyplot as plt
from n_oscillators import GaussianOscillatorArray
from optimal_feedback_n_oscillators import OptimalFeedbackNOscillators

import os

dirname = os.path.dirname(os.path.abspath(__file__))
#%%
# --- Parameters ---
omegas = np.array([1.0, 1.05, 1.1])
N = len(omegas)
q = 1.0
gamma_meas = 5e-2
n_periods = 300
dt = 0.01
n_thermal = 100

# --- Build the coupled optimal feedback ---
lqr = OptimalFeedbackNOscillators(omegas, q=q)
common_force = lqr.common_force_fn()

# --- Simulate the array ---
array = GaussianOscillatorArray(
    omegas,
    gamma_meas=gamma_meas,
    n_periods=n_periods,
    dt=dt,
    n_thermal=n_thermal,
)

covs = array.variance_solver()
result = array.expectation_solver(common_feedback_fn=common_force, covariances=covs)

times = result['times']
xc = result['xc']

# --- Plot all positions on one figure ---
plt.figure(figsize=(10, 5))
for j in range(N):
    plt.plot(times, xc[j], label=rf'$\omega_{j+1} = {omegas[j]:.2f}$', alpha=0.8)
plt.xlabel(r'Time [$1/\omega$]')
plt.ylabel(r'$\langle x \rangle_c$')
plt.title('Conditional mean positions with common LQR feedback')
plt.legend()
plt.tight_layout()
plt.show()

#%%
N = 2
omegas = np.array([1.0, 1.05])
eta = 0.2
gamma_BA = 0.18
n_thermal = 10
initial_temperature = n_thermal

osc_array = GaussianOscillatorArray(omegas=omegas, n_thermal=n_thermal, gamma_meas=gamma_BA, eta=eta, n_periods=20)

n_traj = 10
initial_conditions = np.random.randn(n_traj, N, 3)
initial_conditions[:, :, 0] *= np.sqrt(initial_temperature + 0.5)
initial_conditions[:, :, 1] *= np.sqrt(initial_temperature + 0.5)
initial_conditions[:, :, 2] = 0

feedback_gains = np.linspace(0.5, 5, 5)

covs = osc_array.variance_solver()
n_steady_opt = []
n_steady_err = []

for g_fb in feedback_gains:
    n_opt_traj = []
    
    q = omegas[0] / g_fb**2
    lqr = OptimalFeedbackNOscillators(omegas, q=q)
    common_force_fn = lqr.common_force_fn()
    
    for i in range(n_traj):
        result = osc_array.expectation_solver(
            common_feedback_fn=common_force_fn,
            initial_conditions=initial_conditions[i],
            gamma_fb=g_fb,
            covariances=covs,
        )
        n_bar = osc_array.find_n_bar(result['xc'], result['pc'], covs['Vxx'], covs['Vpp'])
        n_opt_traj.append(np.mean(n_bar[:, -n_bar.shape[1]//4:]))
    
    n_steady_opt.append(np.mean(n_opt_traj))
    n_steady_err.append(np.std(n_opt_traj) / np.sqrt(n_traj))

#%%
feedback_gains_fine = np.linspace(0.2, 5.1, 50)

n_analytic_1 = []
n_analytic_2 = []

for g_fb in feedback_gains_fine:
    q = omegas[0] / g_fb**2
    
    lqr1 = OptimalFeedbackNOscillators(np.array([omegas[0]]), q=q)
    n_analytic_1.append(lqr1.mean_nbar(eta, gamma_BA))
    
    lqr2 = OptimalFeedbackNOscillators(omegas, q=q)
    n_analytic_2.append(lqr2.mean_nbar(eta, gamma_BA))
#%%
plt.figure()
plt.plot(feedback_gains_fine, 2*np.array(n_analytic_1), '--', label='Optimal w/ independent control', color = 'black')
plt.plot(feedback_gains_fine, n_analytic_2, '-', label='Optimal w/ common control', color = 'black')
plt.errorbar(feedback_gains, n_steady_opt, yerr=n_steady_err, fmt='o', label='Simulation w/ common control', color='black', capsize=3)
plt.xlabel(r'$g_\mathrm{fb}/\omega_1$')
plt.ylabel(r'Total phonon number')
plt.xlim([0.2, 5.1])
plt.ylim([0,4])
plt.legend(loc = 'lower center')
plt.tight_layout()
plt.title(rf'Two oscillators with common feedback -- $\omega_1 = {omegas[0]}$, $\omega_2 = {omegas[1]}$')
os.makedirs(os.path.join(dirname, 'figures'), exist_ok=True)
plt.savefig(os.path.join(dirname, "figures/two_oscillator_cooling.pdf"), bbox_inches = 'tight')
plt.show()

#%%
import torch
import torch.nn as nn

n_steady_nn_eval = []
n_steady_nn_err = []

# Load trained policy for RL evaluation
weights_path = os.path.join(os.path.dirname(dirname), 'weights', 'two_oscillators_rl_final.pth')
if os.path.exists(weights_path):
    # This assumes the same architecture as in two_oscillators_RL.py
    policy_trained = nn.Sequential(
        nn.Linear(4, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1)
    )
    try:
        policy_trained.load_state_dict(torch.load(weights_path))
        policy_trained.eval()
        print(f"Loaded weights from {weights_path}")
    except Exception as e:
        print(f"Could not load weights: {e}. Using random policy.")
else:
    print(f"No weights found at {weights_path}. Using random policy.")

n_traj_nn = 10
feedback_gains_rl = [0.5, 1.0, 2.0, 5.0] # Define missing variable
for g_fb in feedback_gains_rl:
    q = omegas[0] / g_fb**2
    lqr = OptimalFeedbackNOscillators(omegas, q=q)
    # Use the loaded policy
    common_force_fn_nn = lqr.optimal_feedback_nn(policy_trained, u_max=25.0)

    n_nn_traj = []
    for i in range(n_traj_nn):
        result = osc_array.expectation_solver(
            common_feedback_fn=common_force_fn_nn,
            initial_conditions=initial_conditions[i],
            gamma_fb=g_fb,
            covariances=covs,
        )
        n_bar = osc_array.find_n_bar(result['xc'], result['pc'], covs['Vxx'], covs['Vpp'])
        n_nn_traj.append(np.mean(n_bar[:, -n_bar.shape[1]//4:]))

    n_steady_nn_eval.append(np.mean(n_nn_traj))
    n_steady_nn_err.append(np.std(n_nn_traj) / np.sqrt(n_traj))
    
#%%
plt.figure()
plt.plot(feedback_gains_fine, 2*np.array(n_analytic_1), '--', label='Independent control', color = 'black')
plt.plot(feedback_gains_fine, n_analytic_2, '-', label='Common control', color = 'black')
# plt.errorbar(feedback_gains, n_steady_opt, yerr=n_steady_err, fmt='o', label='Simulation', color='black', capsize=3)
plt.errorbar(feedback_gains_rl, n_steady_nn_eval, yerr=n_steady_nn_err, 
             fmt='o', label='RL policy', color='black', capsize=3)
plt.xlabel(r'$g_\mathrm{fb}/\omega_1$')
plt.ylabel(r'Total phonon number')
plt.xlim([0.2, 5.1])
plt.ylim([0,15])
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.title(rf'Two oscillators with common feedback -- $\omega_1 = {omegas[0]}$, $\omega_2 = {omegas[1]}$')
os.makedirs(os.path.join(dirname, 'figures'), exist_ok=True)
plt.savefig(os.path.join(dirname, "figures/two_oscillator_cooling.pdf"), bbox_inches = 'tight')
plt.show()

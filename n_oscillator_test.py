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
common_force = lqr.feedback_for_array()

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

n_traj = 100
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
    common_force_fn = lqr.feedback_for_array()
    
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
plt.savefig(os.path.join(dirname, "figures/two_oscillator_cooling.pdf"), bbox_inches = 'tight')
plt.show()
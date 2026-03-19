#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:34:21 2026

@author: giacomomarocco
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'learn_to_cool'))

from gaussian_oscillator import GaussianOscillator
import numpy as np
import matplotlib.pyplot as plt
import os
from optimal_control import FeedbackForces


dirname = os.path.dirname(os.path.abspath(__file__))

initial_temperature = 10
gamma_BA = 18.8/104 # Value used in Magrini et al.
g_fb = 0.1
# kappa = 5e-2
g_fb_str = f"{g_fb}".replace(".", "")
eta = 0.2
n_min = (eta**(-1/2) - 1)/2
dt = 0.05

osc = GaussianOscillator(n_thermal = 10, gamma_meas= gamma_BA, eta = eta)

fb = FeedbackForces(g_fb=g_fb, omega=osc.omega)


# simulate 100 trajectories
n_traj = 100

# a set of n_traj random initial conditions
initial_conditions = np.random.randn(n_traj, 3)
initial_conditions[:, 0] *= np.sqrt(initial_temperature + 0.5)  # x
initial_conditions[:, 1] *= np.sqrt(initial_temperature + 0.5)  # p
initial_conditions[:, 2] = 0  # photocurrent starts at 0

#%%
n_bars = []
horizon_periods = 250
covs = osc.variance_solver(n_periods=horizon_periods, dt=dt)
for i in range(n_traj):
    result = osc.expectation_solver(feedback_fn = fb.momentum_feedback(), initial_conditions=initial_conditions[i], gamma_fb = g_fb, n_periods=horizon_periods, dt=dt)
    n_bar = osc.find_n_bar(result['xc'], result['pc'],  covs['Vxx'], covs['Vpp'])
    n_bars.append(n_bar)

# Average over trajectories
n_bar_avg = np.mean(n_bars, axis=0)

times = result['times']
#%%
for n_bar in n_bars[:10]:
    plt.plot(times, n_bar, alpha=0.15, color='blue')
plt.plot(times, n_bar_avg, color='black', linewidth=2, label='Average')
plt.xlabel(r'$t \, \omega$')
plt.ylabel(r'$\bar{n}$')
plt.yscale('log')
plt.hlines(n_min, times[0], times[-1], ls = 'dashed', color = 'black', label = r'$\bar{n}_\mathrm{min}(\eta)$')
plt.xlim([times[0], times[-1]])
plt.ylim([0.1,1e3])
plt.title(rf'Cooling via velocity damping -- $g_\mathrm{{fb}} = {g_fb}$, $\eta = {eta}$')
plt.legend()
plt.savefig(os.path.join(dirname, f"figures/occupation_number_with_velocity_damping_g{g_fb_str}.pdf"), bbox_inches = 'tight')
plt.show()

#%%


n_bars_optimal = []
for i in range(n_traj):
    result = osc.expectation_solver(feedback_fn=fb.optimal_feedback(), initial_conditions=initial_conditions[i], n_periods=horizon_periods, dt=dt)
    n_bar = osc.find_n_bar(result['xc'], result['pc'], covs['Vxx'], covs['Vpp'])
    n_bars_optimal.append(n_bar)

n_bar_avg_optimal = np.mean(n_bars_optimal, axis=0)

#%%
for n_bar in n_bars_optimal[:10]:
    plt.plot(times, n_bar, alpha=0.15, color='blue')
plt.plot(times, n_bar_avg_optimal, color='black', linewidth=2, label='Average')
plt.xlabel(r'$t \, \omega$')
plt.ylabel(r'$\bar{n}$')
plt.yscale('log')
plt.hlines(n_min, times[0], times[-1], ls = 'dashed', color = 'black', label = r'$\bar{n}_\mathrm{min}(\eta)$')
plt.xlim([times[0], times[-1]])
plt.ylim([0.1,1e3])
plt.title(rf'Cooling via optimal damping -- $g_\mathrm{{fb}} = {g_fb}$, $\eta = {eta}$')
plt.legend()
plt.savefig(os.path.join(dirname, f"figures/occupation_number_with_optimal_damping_g{g_fb_str}.pdf"), bbox_inches = 'tight')
plt.show()

#%%
plt.plot(times, n_bar_avg, color = 'gray', label = 'Velocity damping')
plt.plot(times, n_bar_avg_optimal, color = 'black', label = 'Optimal damping')
plt.xlabel(r'$t \, \omega$')
plt.ylabel(r'$\bar{n}$')
plt.yscale('log')
plt.hlines(n_min, times[0], times[-1], ls = 'dashed', color = 'black', label = r'$\bar{n}_\mathrm{min}(\eta)$')
plt.xlim([times[0], times[-1]])
plt.ylim([0.1,1e3])
plt.title(rf'Cooling comparison -- $g_\mathrm{{fb}} = {g_fb}$, $\eta = {eta}$')
plt.legend()
plt.savefig(os.path.join(dirname, f"figures/occupation_number_comparison_g{g_fb_str}.pdf"), bbox_inches = 'tight')
plt.show()

#%%
osc = GaussianOscillator(n_thermal = 10, gamma_meas= gamma_BA, eta = eta)
fb = FeedbackForces(g_fb=g_fb, omega=osc.omega)


# simulate 100 trajectories
n_traj = 100
# a set of n_traj random initial conditions
initial_conditions = np.random.randn(n_traj, 3)
initial_conditions[:, 0] *= np.sqrt(initial_temperature + 0.5)  # x
initial_conditions[:, 1] *= np.sqrt(initial_temperature + 0.5)  # p
initial_conditions[:, 2] = 0  # photocurrent starts at 0


feedback_gains = np.linspace(0.2, 5, 15)

horizon_periods_short = 50
dt_eval = 0.01
covs = osc.variance_solver(n_periods=horizon_periods_short, dt=dt_eval)
n_steady_opt = []
n_steady_v = []

for g_fb in feedback_gains:
    n_opt_traj = []
    n_v_traj = []

    fb = FeedbackForces(g_fb=g_fb, omega=osc.omega)
    for i in range(n_traj):
        result = osc.expectation_solver(feedback_fn=fb.optimal_feedback(), initial_conditions=initial_conditions[i], gamma_fb=g_fb, n_periods=horizon_periods_short, dt=dt_eval)
        n_bar = osc.find_n_bar(result['xc'], result['pc'], covs['Vxx'], covs['Vpp'])
        n_opt_traj.append(np.mean(n_bar[-len(n_bar)//4:]))

        result = osc.expectation_solver(feedback_fn=fb.momentum_feedback(), initial_conditions=initial_conditions[i], gamma_fb=g_fb, n_periods=horizon_periods_short, dt=dt_eval)
        n_bar = osc.find_n_bar(result['xc'], result['pc'], covs['Vxx'], covs['Vpp'])
        n_v_traj.append(np.mean(n_bar[-len(n_bar)//4:]))
        
    n_steady_opt.append(np.mean(n_opt_traj))
    n_steady_v.append(np.mean(n_v_traj))
#%%
def nbar_analytic(Gamma, eta, gfb, omega):
    xi = np.sqrt(1 + 4 * eta * Gamma**2 / omega**2)
    s = np.sqrt(1 + gfb**2 / omega**2)
    zeta = np.sqrt(gfb**2 / omega**2 - 2 + 2 * s)
    
    # Conditional variances
    Vxx = np.sqrt(2) / (np.sqrt(eta) * np.sqrt(1 + xi))
    Vpp = np.sqrt(2 + 8 * eta * Gamma**2 / omega**2) / (np.sqrt(eta) * np.sqrt(1 + xi))
    
    # Sigma components
    sigma_xx = (2 * gfb**2 * Gamma + Gamma * (-5 + 6 * s + 2 * np.sqrt(2) * np.sqrt(xi - 1) * zeta + xi) * omega**2) / ((1 + xi) * s * zeta * omega**3)
    sigma_pp = Gamma * (-1 + 2 * s + xi) / ((1 + xi) * zeta * omega)
    
    return -0.5 + 0.25 * (Vxx + Vpp + sigma_xx + sigma_pp)

g_fb_fine = np.linspace(0.2, 5, 100)
n_analytic = [nbar_analytic(gamma_BA, eta, g, osc.omega) for g in g_fb_fine]

plt.plot(feedback_gains, n_steady_opt, 'o', label='Optimal feedback', color = 'black')
plt.plot(feedback_gains, n_steady_v, 's', label='Velocity feedback')
plt.plot(g_fb_fine, n_analytic, '-', label='LQG theory', color = 'black')
plt.hlines(n_min, 0, feedback_gains[-1], ls = 'dashed', color = 'black', label = r'$\bar{n}_\mathrm{min}(\eta)$')
plt.xlim([0,5])
# plt.ylim([0.2,1])
plt.xlabel(r'$g_{\mathrm{fb}}/\omega$')
plt.ylabel(r'$\bar{n}$')
plt.title(f'Cooling performance for $\eta = {eta}$ and $\Gamma_\mathrm{{BA}}/\omega = {gamma_BA:.2f}$')
plt.legend(frameon = False)
plt.savefig(os.path.join(dirname, "figures/cooling_comparison.pdf"), bbox_inches = 'tight')
plt.show()
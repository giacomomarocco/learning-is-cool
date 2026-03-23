import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:20:42 2026

@author: giacomomarocco
"""

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
from gaussian_oscillator_array import GaussianOscillatorArray
from optimal_feedback_n_oscillators import OptimalFeedbackNOscillators
import seaborn as sns

import os

dirname = os.path.dirname(os.path.abspath(__file__))

#%%
N = 2
omega1 = 1.0
splitting = np.logspace(-3, 0, num = 20)
omegas_list = [[omega1, omega1 + split] for split in splitting]
eta = 0.2
gamma_BA = 0.2

#%%
feedback_gains_fine = [0.01, 0.1, 1., 10]

n_analytic_1 = []
n_analytic_2 = []


# Store results as 2D arrays: (n_gains, n_splittings)
n_analytic_1 = np.zeros((len(feedback_gains_fine), len(splitting)))
n_analytic_2 = np.zeros((len(feedback_gains_fine), len(splitting)))

for i, g_fb in enumerate(feedback_gains_fine):
    for j, omegas in enumerate(omegas_list):
        q = omegas[0] / g_fb**2

        lqrInd1 = OptimalFeedbackNOscillators(np.array([omegas[0]]), q=q)
        lqrInd2 = OptimalFeedbackNOscillators(np.array([omegas[1]]), q=q)

        n_analytic_1[i, j] = lqrInd1.mean_nbar(eta, gamma_BA) + lqrInd2.mean_nbar(eta, gamma_BA) 
        n_analytic_1[i, j] *= 1/2

        lqr2 = OptimalFeedbackNOscillators(omegas, q=q)
        n_analytic_2[i, j] = lqr2.mean_nbar(eta, gamma_BA)
        
        
#%%
sns.set_palette('colorblind')
plt.figure()
# plt.plot(splitting, np.array(n_analytic_1[0,:]), '--', label='Optimal w/ independent control', color = 'black')
plt.plot(splitting, n_analytic_2[0,:], '-', label=rf'$g_\mathrm{{fb}}/\omega_1$  = {feedback_gains_fine[0]} ')
# plt.plot(splitting, np.array(n_analytic_1[10,:]), '--', label='Optimal w/ independent control', color = 'black')
plt.plot(splitting, n_analytic_2[1,:], '-', label=rf'$g_\mathrm{{fb}}/\omega_1$ = {feedback_gains_fine[1]}')
# plt.plot(splitting, np.array(n_analytic_1[20,:]), '--', label='Optimal w/ independent control', color = 'black')
plt.plot(splitting, n_analytic_2[2,:], '-', label=rf'$g_\mathrm{{fb}}/\omega_1$ = {feedback_gains_fine[2]}')
# plt.plot(splitting, np.array(n_analytic_1[30,:]), '--', label='Optimal w/ independent control', color = 'black')
# plt.plot(splitting, n_analytic_2[3,:], '-', label=rf'$g_\mathrm{{fb}}$ = {feedback_gains_fine[3]}')

# plt.plot(splitting, n_analytic_2[40,:], '-', label='Optimal w/ common control', color = 'black')
# plt.errorbar(feedback_gains, n_steady_opt, yerr=n_steady_err, fmt='o', label='Simulation w/ common control', color='black', capsize=3)
plt.xlabel(r'$|\omega_2 - \omega_1|/\omega_1$')
plt.ylabel(r'Mean phonon number')
plt.xlim([0.001, 1])
plt.ylim([0.1,500])
plt.xscale('log')
plt.yscale('log')
plt.legend(loc = 'lower left')
plt.tight_layout()
plt.title(rf'Two oscillator cooling -- $\eta = {eta}$, $\Gamma_\mathrm{{BA}} = {gamma_BA} \,\omega_1$')
plt.savefig(os.path.join(dirname, "figures/cooling_vs_splitting.pdf"), bbox_inches = 'tight')

#%%
N_values = range(1, 20)
deltas = [ 0.01, 0.1, 1]
g_fb = 1.0
q = omega1 / g_fb**2

plt.figure()
for delta in deltas:
    n_common = []
    n_ind = []
    for N in N_values:
        omegas = np.array([omega1 + i * delta * omega1 for i in range(N)])
        lqr = OptimalFeedbackNOscillators(omegas, q=q)
        n_common.append(lqr.mean_nbar(eta, gamma_BA))
        n_ind.append(np.mean([OptimalFeedbackNOscillators(np.array([om]), q=q).mean_nbar(eta, gamma_BA)
                              for om in omegas]))

    plt.plot(list(N_values), n_common, 'o-', label=rf'$\Delta\omega = {delta}\,\omega_1$')

# Independent control (doesn't depend on delta for equal masses)
# plt.hlines(n_ind, 0, 20, ls='dashed', color='black')

plt.xlabel(r'$N_\mathrm{osc}$')
plt.ylabel(r'Mean phonon number')
plt.legend(loc = 'lower center')
plt.title(rf'$\eta = {eta}$, $\Gamma_\mathrm{{BA}} = {gamma_BA} \,\omega_1$, $g_{{\mathrm{{fb}}}}/\omega_1 = {g_fb}$')
plt.ylim([0.1, 100])
plt.xlim([0, 20])
plt.yscale('log')
plt.savefig(os.path.join(dirname, "figures/cooling_vs_N_oscillator.pdf"), bbox_inches = 'tight')
plt.show()


#%%
N_values = range(1, 200, 10)
deltas = [0.01, 0.1, 1]
g_fb = 1.0
q = omega1 / g_fb**2

plt.figure()
for delta in deltas:
    n_common = []
    n_ind = []
    for N in N_values:
        omegas = np.array([omega1 + i * delta * omega1 for i in range(N)])
        lqr = OptimalFeedbackNOscillators(omegas, q=q)
        n_common.append(lqr.mean_nbar(eta, gamma_BA))
        n_ind.append(np.mean([OptimalFeedbackNOscillators(np.array([om]), q=q).mean_nbar(eta, gamma_BA)
                              for om in omegas]))

    plt.plot(list(N_values), n_common, 'o-', label=rf'$\Delta\omega = {delta}\,\omega_1$')

# Independent control (doesn't depend on delta for equal masses)
# plt.hlines(n_ind, 0, 20, ls='dashed', color='black')

plt.xlabel('N')
plt.ylabel(r'$\bar{n}$ (mean per oscillator)')
plt.legend(loc = 'lower center')
plt.title(rf'$g_{{\mathrm{{fb}}}}/\omega_1 = {g_fb}$')
plt.ylim([0.1, 30])
plt.xlim([0, 200])
plt.yscale('log')
plt.show()


#%%
N_values = range(1, 20)
deltas = [0.01, 0.1, 1]
g_fb = 0.1
q = omega1 / g_fb**2
n_ind = []
# for N in N_values:
#     omegas = np.array([omega1 + i * delta * omega1 for i in range(N)])
#     n_ind.append(np.mean([OptimalFeedbackNOscillators(np.array([om]), q=q).mean_nbar(eta, gamma_BA)
#                           for om in omegas]))

plt.figure()
for delta in deltas:
    n_common = []
    for N in N_values:
        omegas = np.array([omega1 + i * delta * omega1 for i in range(N)])
        lqr = OptimalFeedbackNOscillators(omegas, q=q)
        n_common.append(lqr.mean_nbar(eta, gamma_BA))
    plt.plot(list(N_values), n_common, 'o-', label=rf'$\Delta\omega = {delta}\,\omega_1$')

# Independent control (doesn't depend on delta for equal masses)

# plt.hlines(n_ind, 0, 20, ls='dashed', color='black')

plt.xlabel('N')
plt.ylabel(r'$\bar{n}$ (mean per oscillator)')
plt.legend(loc = 'lower center')
plt.title(rf'$g_{{\mathrm{{fb}}}}/\omega_1 = {g_fb}$')
plt.ylim([0.1, 30])
plt.xlim([0, 20])
plt.yscale('log')
plt.show()
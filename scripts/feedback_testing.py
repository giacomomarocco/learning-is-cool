#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:55:18 2026

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
dt = 0.005
initial_temperature = 25
gamma_BA = 0.18 # Value used in Magrini et al.
g_fb = 0.1
# kappa = 5e-2
g_fb_str = f"{g_fb}".replace(".", "")
eta = 0.2
n_min = (eta**(-1/2) - 1)/2
initial_x_conditions = np.sqrt(initial_temperature)*np.random.randn(2)
init = np.append(initial_x_conditions,0)

osc = GaussianOscillator(n_thermal = initial_temperature, gamma_meas= gamma_BA, eta = eta)
#
fb = FeedbackForces(g_fb=g_fb, omega=osc.omega)


# a set of n_traj random initial conditions
# initial_conditions = np.random.randn(n_traj, 3)
# initial_conditions[:, 0] *= np.sqrt(initial_temperature + 0.5)  # x
# initial_conditions[:, 1] *= np.sqrt(initial_temperature + 0.5)  # p
# initial_conditions[:, 2] = 0  # photocurrent starts at 0
covs = osc.variance_solver(dt=dt, n_periods=50)
expectations = osc.expectation_solver(parametric = True, epsilon = 0.1, gamma_fb = 0.001, initial_conditions = init, dt=dt, n_periods=50 )

# plt.plot(expectations['times'], covs['Vxx']/4, label = 'Vxx')
plt.plot(expectations['times'], osc.find_n_bar(expectations['xc'], expectations['pc'], covs['Vxx'], covs['Vpp']), label = 'nbar')
plt.yscale('log')
plt.xlabel(r'$ t \omega$')
plt.ylabel(r'$\bar{n}$')
plt.title('Parametric feedback example')
plt.savefig(os.path.join(dirname, f"figures/feedback_cooling_example.pdf"), bbox_inches = 'tight')
# plt.legend()
plt.show()
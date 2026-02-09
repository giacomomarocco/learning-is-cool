#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 09:59:00 2026

@author: giacomomarocco
"""
import numpy as np
import matplotlib.pyplot as plt
from GaussianOscillator import GaussianOscillator
import scipy


#%%
osc = GaussianOscillator(gamma_meas=5e-2, quality_factor=1e4)
# plt.plot(osc.expectation_solver(n_periods = 100, dt = 0.008, gamma_fb = 0.02, steady_state_covs = False)[2][::50])
(x, p, r), t = osc.expectation_solver(n_periods = 1000, dt = 0.1, gamma_fb = 0.01)
#%%
plt.plot(t[-500:], x[-500:])
plt.plot(t[-500:], r[-500:])

#%%
f, Sxx = osc.position_PSD(t[2000:], x[2000:], nperseg = int(5e4))
f, Srr = osc.position_PSD(t[2000:], r[2000:], nperseg = int(5e4))

plt.loglog(2*np.pi*f, Sxx)    
plt.loglog(2*np.pi*f, Srr)    

plt.xlim([0.1,100])    
temp = scipy.integrate.simpson(Sxx, f)
print(temp)
#%%
osc = GaussianOscillator(gamma_meas=1e-2)
# plt.plot(osc.expectation_solver(n_periods = 100, dt = 0.008, gamma_fb = 0.02, steady_state_covs = False)[2][::50])
x, p, _ = osc.expectation_solver(n_periods = 200, dt = 0.01, gamma_fb = 0.1)
plt.plot(x[::50])
plt.plot(p[::50])
print(np.mean(x[200:]), np.mean(p[200:]))

#%%
def T_theory(gamma_fb, k):
    Snn = 1/(8*k)
    Sff = 1/(4* Snn)
    return ( Sff / gamma_fb + gamma_fb * Snn ) / 2


#%%
damping_rates = np.logspace(-2., 1., 10)
temperatures = np.zeros_like(damping_rates)
damping_rates_long = np.logspace(-2., 1., 20)
theory_temps = np.zeros_like(damping_rates_long)
for i, damping_rate in enumerate(damping_rates):
    temperatures[i] = osc.find_temperature(gamma_fb = damping_rate, n_periods=1000, dt=0.005, steady_state_covs=False)

#%%
for i, damping_rate in enumerate(damping_rates_long):
    theory_temps[i] = T_theory(damping_rate, 0.02)

plt.loglog(damping_rates, temperatures)
plt.loglog(damping_rates_long, theory_temps)
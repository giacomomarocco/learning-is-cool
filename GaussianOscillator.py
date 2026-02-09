#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 10:30:41 2026

@author: giacomomarocco
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

class GaussianOscillator:
    def __init__(self, omega=1.0, quality_factor = 1e4, gamma_meas = 5e-2, eta = 1.0, n_thermal = 100):
        self.omega = omega
        self.gamma = self.omega/quality_factor
        self.gamma_meas = gamma_meas
        self.eta = eta
        self.n_thermal = 100
        # The variable k in quant-ph/9812004 that matches our definition in variance eom
        self.k_jacobs = self.gamma_meas * self.omega/2
        
    def variance_solver(self, n_periods = 10, dt = 0.05):
        '''Returns an array of the variances as a function of time.
        Can specify the number of periods for which to simulate n_periods, and the time step dt.'''
        init_cond = np.array([(1 + self.n_thermal)/2, 
                             (1 + self.n_thermal)/2, 
                             0])
        n_times = int(2*np.pi*n_periods/dt)
        y = np.zeros((3, n_times))
        y[:, 0] = init_cond
        times = [0]
        
        def variance_DE(t, y_vec):
            Vx, Vp, Cxp = y_vec
            
            x0Squared = 1/(self.omega)

            dVx = 2 * Cxp - 4 * self.eta * self.gamma_meas * Vx**2 / x0Squared
            dVp = -2 * self.omega**2 * Cxp +  self.gamma_meas / x0Squared - 4 * self.eta * self.gamma_meas * Cxp**2 / x0Squared
            dCxp = Vp - self.omega**2 * Vx - 4 * self.eta * self.gamma_meas * Vx * Cxp / x0Squared
                
            return np.array([dVx, dVp, dCxp])
        
        for i in range(1, n_times):
            t = (i-1) * dt
            current_y = y[:, i-1]
            
            # RK4 method (note: your code says RK2 but implements RK4)
            k1 = variance_DE(t, current_y)
            k2 = variance_DE(t + dt/2, current_y + dt/2 * k1)
            k3 = variance_DE(t + dt/2, current_y + dt/2 * k2)
            k4 = variance_DE(t + dt, current_y + dt * k3)
            times.append(t)
            y[:, i] = current_y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        #Work in dimensionless variables, with unit variance
        
        y[0,:] *= self.omega*2
        y[1,:] *= 2/self.omega
        
        return y, np.array(times)
    
    def steady_state_variances(self):
        ''' Steady state variances as given in eq. (51) of quant-ph/quant-ph/9812004'''

        r = self.omega**2 / (2 * self.eta * self.k_jacobs)
        xi = np.sqrt(1 + 4 / (self.eta * r**2))
        Vx = 1 / (np.sqrt(2 * self.eta) * self.omega) / np.sqrt(self.eta + 1)
        Vp = self.omega / np.sqrt(2 * self.eta) * xi / np.sqrt(self.eta + 1)
        Cxp = 1 / (2 * np.sqrt(self.eta)) * np.sqrt(self.eta - 1)/np.sqrt(self.eta + 1)
        return [Vx, Vp, Cxp]
    
    def expectation_solver(self, gamma_fb = 0.1, n_periods = 50, dt = 0.01, initial_conditions =  np.array([20, 
                         20,0]), steady_state_covs = False):
        
        n_times = int(2*np.pi*n_periods / dt)

        if steady_state_covs == True:
            # Assume the covariances have reached their steady state values
            ss = self.steady_state_variances()
            var_x = np.full(n_times, ss[0])
            cov_xp = np.full(n_times, ss[2])
            
        if steady_state_covs == False:
            covs, t = self.variance_solver(n_periods, dt)
            var_x = covs[0]
            cov_xp = covs[2]
            
        
        dW = np.sqrt(dt) * np.random.randn(n_times)
        
        # Initial conditions 
        init_cond = initial_conditions
        y = np.zeros((3, n_times))
        y[:, 0] = init_cond
        times = [0]

        # Semi-implicit Euler method for deterministic pieces to avoid late-time blowups
        # Explicit Euler for stochastic
        for i in range(1, n_times):
            x, p, record = y[:, i-1]
            dW_i = dW[i-1]
            
            dp = - self.omega**2 * x * dt  + np.sqrt(8*self.eta*self.k_jacobs) * cov_xp[i-1] * dW_i - (gamma_fb + self.gamma) * p * dt
            p_new = p + dp
            
            dx = p_new * dt + np.sqrt(8*self.eta*self.k_jacobs) * var_x[i-1] * dW_i
            x_new = x + dx
            
            drecord = x * dt + dW_i/(np.sqrt(8*self.eta*self.k_jacobs))
            photocurrent = drecord/dt 
            
            y[0, i] = x_new
            y[1, i] = p_new
            y[2, i] = photocurrent
            times.append(i*dt)
            
        return y, np.array(times)

    def find_temperature(self, **kwargs):
        y = self.expectation_solver(**kwargs)
        x_meas = y[0, :]
        
        # Discard initial transient
        x_meas_steady = x_meas[len(x_meas)//2:]
        
        # Time-averaged variance of conditioned mean
        V_x_e = np.mean(x_meas_steady**2)
        
        # Intrinsic variance from measurement
        covs, t = self.variance_solver(n_periods = kwargs['n_periods'], dt = kwargs['dt'])
        V_x = np.mean(covs[0][len(x_meas)//2:])/2
        V_x = self.steady_state_variances()[0]
        V_x = 0
        
        # Total
        return V_x + V_x_e

    def current_energy(self, xc, pc):
        '''Returns an estimate of the energy given estimates of the position and momentum'''
        return (xc**2 + pc**2)/2
    
    def position_PSD(self, times, positions, nperseg = int(2**16)):
        dt = times[1] - times[0]
        fs = 1.0/dt                   # sampling frequency

        f, Sxx = scipy.signal.welch(positions, fs=fs,
                       nperseg=nperseg, return_onesided=True)

        nu  =  2* np.pi*f
        # gamma = omega/Q_factor
        # chi = 1.0 / (m*(omega**2 - nu**2 + 1j*gamma*nu))
        # Sff = Sxx / (np.abs(chi)**2)     
        
        return f, Sxx

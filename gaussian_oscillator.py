#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 10:30:41 2026

@author: giacomomarocco
"""

import numpy as np
import scipy
from optimal_control import FeedbackForces

class GaussianOscillator:
    def __init__(self, omega=1.0, quality_factor = 1e4, gamma_meas = 5e-2, eta = 1.0, n_thermal = 100, dt = 0.01, n_periods = 100):
        self.omega = omega
        self.gamma = self.omega/quality_factor
        self.gamma_meas = gamma_meas
        self.eta = eta
        self.n_thermal = 100
        self.dt = dt
        self.n_periods = n_periods
        # The variable k in quant-ph/9812004 that matches our definition in variance eom
        self.k_jacobs = self.gamma_meas * self.omega/2
        
    def dvariance(self, time, covariances):
        # Returns the difference equation obeyed by the covariances
        Vx, Vp, Cxp = covariances
        
        x0Squared = 1/(self.omega)

        dVx = 2 * Cxp - 4 * self.eta * self.gamma_meas * Vx**2 / x0Squared
        dVp = -2 * self.omega**2 * Cxp +  self.gamma_meas / x0Squared - 4 * self.eta * self.gamma_meas * Cxp**2 / x0Squared
        dCxp = Vp - self.omega**2 * Vx - 4 * self.eta * self.gamma_meas * Vx * Cxp / x0Squared
        return np.array([dVx, dVp, dCxp])

        
        
    def variance_solver(self):
        '''Returns an array of the variances as a function of time.
        Can specify the number of periods for which to simulate n_periods, and the time step dt.'''
        init_cond = np.array([(1 + self.n_thermal)/2, 
                             (1 + self.n_thermal)/2, 
                             0])
        dt = self.dt
        n_times = int(2*np.pi*self.n_periods/dt)
        y = np.zeros((3, n_times))
        y[:, 0] = init_cond
        times = [0]
        
        # def variance_DE(t, y_vec):
        #     #This depends on time because there may be something (gas collision) that increases the variance at random times
        #     Vx, Vp, Cxp = y_vec
            
        #     x0Squared = 1/(self.omega)

        #     dVx = 2 * Cxp - 4 * self.eta * self.gamma_meas * Vx**2 / x0Squared
        #     dVp = -2 * self.omega**2 * Cxp +  self.gamma_meas / x0Squared - 4 * self.eta * self.gamma_meas * Cxp**2 / x0Squared
        #     dCxp = Vp - self.omega**2 * Vx - 4 * self.eta * self.gamma_meas * Vx * Cxp / x0Squared
                
        #     return np.array([dVx, dVp, dCxp])
        
        for i in range(1, n_times):
            t = (i-1) * dt
            current_y = y[:, i-1]
            
            # RK4 method (note: your code says RK2 but implements RK4)
            k1 = self.dvariance(t, current_y)
            k2 = self.dvariance(t + dt/2, current_y + dt/2 * k1)
            k3 = self.dvariance(t + dt/2, current_y + dt/2 * k2)
            k4 = self.dvariance(t + dt, current_y + dt * k3)
            times.append(t)
            y[:, i] = current_y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        #Work in dimensionless variables, with unit variance
        
        y[0,:] *= self.omega*2
        y[1,:] *= 2/self.omega
        
        return {'Vxx': y[0], 'Vpp': y[1], 'Cxp': y[2], 'times': np.array(times)}
    
    def steady_state_variances(self):
        ''' Steady state variances as given in eq. (51) of quant-ph/quant-ph/9812004'''

        r = self.omega**2 / (2 * self.eta * self.k_jacobs)
        xi = np.sqrt(1 + 4 / (self.eta * r**2))
        Vx = 1 / (np.sqrt(2 * self.eta) * self.omega) / np.sqrt(xi + 1)
        Vp = self.omega / np.sqrt(2 * self.eta) * xi / np.sqrt(xi + 1)
        Cxp = 1 / (2 * np.sqrt(self.eta)) * np.sqrt(xi - 1)/np.sqrt(xi + 1)
        return [Vx, Vp, Cxp]
    
    def expectation_solver(self, feedback_fn=None, gamma_fb=0.1, initial_conditions=np.array([20, 20, 0]),
                           covariances=None):       
        
        dt = self.dt
        # Default to optimal feedback if no function provided
        if feedback_fn is None:
            fb = FeedbackForces(g_fb = gamma_fb, omega = self.omega)
            feedback_fn = fb.optimal_feedback()
        n_times = int(2*np.pi*self.n_periods / self.dt)

        # if steady_state_covs == True:
        #     # Assume the covariances have reached their steady state values
        #     ss = self.steady_state_variances()
        #     var_x = np.full(n_times, ss[0])
        #     cov_xp = np.full(n_times, ss[2])
            
        # if steady_state_covs == False:
        #     covs = self.variance_solver()
        #     var_x = covs['Vxx']
        #     cov_xp = covs['Cxp']            
        
        if covariances is None:
            covariances = self.variance_solver()
        
        var_x = covariances['Vxx']
        cov_xp = covariances['Cxp']
        
        dW = np.sqrt(self.dt) * np.random.randn(n_times)
        
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
            u = feedback_fn(x, p)
            

            dp = (-self.omega**2 * x * dt
                  + np.sqrt(2*self.eta*self.gamma_meas) * cov_xp[i-1] * dW_i
                  - (self.gamma) * p * dt
                  + u * dt)
            p_new = p + dp
            
            dx = p_new * dt + np.sqrt(2*self.eta*self.gamma_meas) * var_x[i-1] * dW_i
            x_new = x + dx
            
            drecord = x * dt + dW_i/(np.sqrt(2*self.eta*self.gamma_meas))
            photocurrent = drecord/dt 
            
            y[0, i] = x_new
            y[1, i] = p_new
            y[2, i] = photocurrent
            times.append(i*dt)
            
        return {'xc': y[0], 'pc': y[1], 'photocurrent': y[2], 'times': np.array(times)}

    def find_n_bar(self, xc, pc, Vx, Vp):
        '''Calculates <x^2 + p^2>/2 - 1/2 from Kalman filter and variances'''
        x2 = xc ** 2 + Vx
        p2 = pc ** 2 + Vp
        return (x2 + p2)/4 - 1/2

    def find_temperature_ergodic(self, **kwargs):
        ''' Calculates the temperature by time-integrating, assuming ergodicity/steady-state reached '''
        y = self.expectation_solver(**kwargs)
        x_meas = y['xc']        
        
        # Discard initial transient
        x_meas_steady = x_meas[len(x_meas)//2:]
        
        # Time-averaged variance of conditioned mean
        V_x_e = np.mean(x_meas_steady**2)
        
        # Intrinsic variance from measurement
        # covs = self.variance_solver()
        # V_x = np.mean(covs['Vxx'][len(x_meas)//2:])/2        # V_x = np.mean(covs[0][len(x_meas)//2:])/2
        V_x = self.steady_state_variances()[0]
        # V_x = 0
        
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
    

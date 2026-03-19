#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 10:30:41 2026

@author: giacomomarocco
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
# import seaborn as sns

class GaussianOscillator:
    def __init__(self, omega=1.0, quality_factor = 1e4, gamma_meas = 5e-2, eta = 1.0, n_thermal = 100):
        self.omega = omega
        self.gamma = self.omega/quality_factor
        self.gamma_meas = gamma_meas
        self.eta = eta
        self.n_thermal = n_thermal
        # The variable k in quant-ph/9812004 that matches our definition in variance eom
        self.k_jacobs = self.gamma_meas * self.omega/2
        
    def dvariance(self, time, covariances):
        # Returns the difference equation obeyed by the covariances
        Vx, Vp, Cxp = covariances

        # dVx = 2*omega*Cxp - 4*eta*Gamma_BA*Vx^2
        dVx = 2 * self.omega * Cxp - 4 * self.eta * self.gamma_meas * Vx**2
        # dVp = -2*omega*Cxp + 4*Gamma_BA - 4*eta*Gamma_BA*Cxp^2
        dVp = -2 * self.omega * Cxp + 4 * self.gamma_meas - 4 * self.eta * self.gamma_meas * Cxp**2
        # dCxp = omega(Vp - Vx) - 4*eta*Gamma_BA*Vx*Cxp
        dCxp = self.omega * (Vp - Vx) - 4 * self.eta * self.gamma_meas * Vx * Cxp
        return np.array([dVx, dVp, dCxp])

        
        
    def variance_solver(self, n_periods = 10, dt = 0.05, n_thermal = None):
        '''Returns an array of the variances as a function of time.
        Can specify the number of periods for which to simulate n_periods, and the time step dt.'''
        if n_thermal == None:
            n_thermal = self.n_thermal
        init_cond = np.array([(1 + n_thermal), 
                             (1 + n_thermal), 
                             0])
        n_times = int(2*np.pi*n_periods/dt)
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

        return {'Vxx': y[0, :], 'Vpp': y[1, :], 'Cxp': y[2, :], 'times': np.array(times)}

    def steady_state_variances(self):
        ''' Steady state variances as given in eq. (51) of quant-ph/9812004'''

        xi = np.sqrt(1 + 4 * self.eta * self.gamma_meas**2 / self.omega**2)
        Vx = 2 / (np.sqrt(2 * self.eta) * np.sqrt(xi + 1))
        Vp = 2 * xi / (np.sqrt(2 * self.eta) * np.sqrt(xi + 1))
        Cxp = np.sqrt(xi - 1) / (np.sqrt(self.eta) * np.sqrt(xi + 1))
        return [Vx, Vp, Cxp]
    
    # def expectation_solver(self, feedback_fn=None, gamma_fb=0.1, n_periods=50, dt=0.01,
    #                        initial_conditions=np.array([20, 20, 0]),
    #                        steady_state_covs=False, parametric=False):       
        
    #     # Default to linear feedback if no function provided
    #     if feedback_fn is None:
    #         feedback_fn = lambda xc, pc: -gamma_fb * pc
    #     n_times = int(2*np.pi*n_periods / dt)

    #     if steady_state_covs == True:
    #         # Assume the covariances have reached their steady state values
    #         ss = self.steady_state_variances()
    #         var_x = np.full(n_times, ss[0])
    #         cov_xp = np.full(n_times, ss[2])
            
    #     if steady_state_covs == False:
    #         covs, t = self.variance_solver(n_periods, dt)
    #         var_x = covs[0]
    #         cov_xp = covs[2]
            
        
    #     dW = np.sqrt(dt) * np.random.randn(n_times)
        
    #     # Initial conditions 
    #     init_cond = initial_conditions
    #     y = np.zeros((3, n_times))
    #     y[:, 0] = init_cond
    #     times = [0]

    #     # Semi-implicit Euler method for deterministic pieces to avoid late-time blowups
    #     # Explicit Euler for stochastic
    #     sqrt_2eta_gm = 2 * np.sqrt(self.eta * self.gamma_meas)
    #     for i in range(1, n_times):
    #         x, p, record = y[:, i-1]
    #         dW_i = dW[i-1]
    #         u = feedback_fn(x, p)

    #         dp = (-self.omega * x * dt
    #               + sqrt_2eta_gm * cov_xp[i-1] * dW_i
    #               + u * dt)
    #         p_new = p + dp

    #         dx = self.omega * p_new * dt + sqrt_2eta_gm * var_x[i-1] * dW_i
    #         x_new = x + dx

    #         drecord = x * dt + dW_i/sqrt_2eta_gm
    #         photocurrent = drecord/dt
            
    #         y[0, i] = x_new
    #         y[1, i] = p_new
    #         y[2, i] = photocurrent
    #         times.append(i*dt)
            
    #     return y, np.array(times)
    
    def expectation_solver(self, feedback_fn=None, gamma_fb=0.1, n_periods=50, dt=0.01,
                       initial_conditions=np.array([20, 20, 0]),
                       steady_state_covs=False, parametric=False,
                       epsilon=0.0, parametric_fn=None):

        # Default to nothing feedback if no function provided
        if feedback_fn is None:
            feedback_fn = lambda xc, pc: 0
        n_times = int(2*np.pi*n_periods / dt)

        if not parametric and steady_state_covs:
            ss = self.steady_state_variances()
            var_x = np.full(n_times, ss[0])
            var_p = np.full(n_times, ss[1])
            cov_xp = np.full(n_times, ss[2])
        elif not parametric and not steady_state_covs:
            res_var = self.variance_solver(n_periods, dt)
            var_x = res_var['Vxx']
            var_p = res_var['Vpp']
            cov_xp = res_var['Cxp']
        else:
            # Parametric case: variances will be evolved step by step
            # Initialize from thermal state
            var_x = np.zeros(n_times)
            var_p = np.zeros(n_times)
            cov_xp = np.zeros(n_times)
            var_x[0] = (1 + self.n_thermal) / 2
            var_p[0] = (1 + self.n_thermal) / 2
            cov_xp[0] = 0.0

        dW = np.sqrt(dt) * np.random.randn(n_times)

        y = np.zeros((3, n_times))
        y[:, 0] = initial_conditions
        times = np.zeros(n_times)

        sqrt_2eta_gm = 2 * np.sqrt(self.eta * self.gamma_meas)

        for i in range(1, n_times):
            x, p, record = y[:, i-1]
            t = times[i-1]
            dW_i = dW[i-1]

            Vx = var_x[i-1]
            Vp = var_p[i-1]
            Cxp = cov_xp[i-1]

            # Compute effective spring constant
            if parametric:
                if parametric_fn is not None:
                    u_param = parametric_fn(x, p, t)
                    # We assume the parametric_fn returns the factor multiplying omega^2
                    # or it can return the delta if we want to be consistent with epsilon logic.
                    # Let's say if it's the RL policy, it returns (1+u)**2
                    omega2_eff = self.omega**2 * u_param
                else:
                    X_slow = x * np.cos(self.omega * t) - p * np.sin(self.omega * t)
                    Y_slow = x * np.sin(self.omega * t) + p * np.cos(self.omega * t)
                    phi_pll = np.arctan2(-Y_slow, X_slow)
                    phi_0 = np.pi/2
                    u_param = epsilon * np.cos(2 * self.omega * t + 2 * phi_pll + phi_0)
                    omega2_eff = self.omega**2 * (1 + u_param)
            else:
                omega2_eff = self.omega**2
    
            # --- Conditional means ---
            u = feedback_fn(x, p)
    
            dp = (-(omega2_eff / self.omega) * x * dt
                  + sqrt_2eta_gm * Cxp * dW_i
                  + u * dt)
            p_new = p + dp
    
            # x equation is unchanged (kinematic, not spring)
            dx = self.omega * p_new * dt + sqrt_2eta_gm * Vx * dW_i
            x_new = x + dx
    
            drecord = x * dt + dW_i / sqrt_2eta_gm
            photocurrent = drecord / dt
            
            y[0, i] = x_new
            y[1, i] = p_new
            y[2, i] = photocurrent
            times[i] = t + dt
    
            # --- Co-evolve variances if parametric ---
            if parametric:
                # Vxx equation: unchanged (comes from x eom)
                dVx = (2 * self.omega * Cxp 
                       - 4 * self.eta * self.gamma_meas * Vx**2) * dt
                
                # Vpp equation: omega -> omega2_eff/omega
                dVp = (-2 * (omega2_eff / self.omega) * Cxp 
                       + 4 * self.gamma_meas 
                       - 4 * self.eta * self.gamma_meas * Cxp**2) * dt
                
                # Cxp equation: mixed
                dCxp = (self.omega * Vp - (omega2_eff / self.omega) * Vx
                        - 4 * self.eta * self.gamma_meas * Vx * Cxp) * dt

                Vx = np.maximum(Vx + dVx, 1e-4)
                Vp = np.maximum(Vp + dVp, 1e-4)
                Cxp = Cxp + dCxp

                var_x[i] = Vx
                var_p[i] = Vp
                cov_xp[i] = Cxp

        return {
            'xc': y[0, :],
            'pc': y[1, :],
            'photocurrent': y[2, :],
            'times': times,
            'Vxx': var_x,
            'Vpp': var_p,
            'Cxp': cov_xp
        }

    def find_temperature(self, **kwargs):
        res = self.expectation_solver(**kwargs)
        x_means = res['xc']
        p_means = res['pc']
        var_x = res['Vxx']
        var_p = res['Vpp']

        # Discard initial transient (last half)
        n = len(x_means)
        x_steady = x_means[n//2:]
        p_steady = p_means[n//2:]
        vx_steady = var_x[n//2:]
        vp_steady = var_p[n//2:]

        # Calculate mean phonon number in steady state
        n_bars = (x_steady**2 + vx_steady + p_steady**2 + vp_steady) / 4 - 0.5
        return np.mean(n_bars)

    def find_n_bar(self, xc, pc, Vx, Vp):
        """
        Returns the mean phonon number given conditional means and variances.
        """
        x2 = xc**2 + Vx
        p2 = pc**2 + Vp
        return (x2 + p2) / 4 - 0.5

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
    

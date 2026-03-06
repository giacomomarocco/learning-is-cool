#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:42:58 2026

@author: giacomomarocco
"""

import numpy as np
from optimal_control import FeedbackForces


class GaussianOscillatorArray:
    def __init__(self, omegas, quality_factor=1e4, gamma_meas=5e-2, eta=1.0,
                 n_thermal=100, dt=0.01, n_periods=100):
        """
        Parameters
        ----------
        omegas : array-like
            Array of angular frequencies for each oscillator.
        quality_factor, gamma_meas, eta, n_thermal, dt, n_periods :
            Shared parameters (same meaning as in GaussianOscillator).
        """
        self.omegas = np.asarray(omegas, dtype=float)
        self.N = len(self.omegas)
        self.quality_factor = quality_factor
        self.gammas = self.omegas / quality_factor
        self.gamma_meas = gamma_meas
        self.eta = eta
        self.n_thermal = n_thermal
        self.dt = dt
        self.n_periods = n_periods
        self.k_jacobs = self.gamma_meas * self.omegas / 2  # per-oscillator

    # ----- variance equations (vectorised over oscillators) -----

    def dvariance(self, time, covariances):
        """
        covariances : ndarray of shape (N, 3)
            Each row is [Vx, Vp, Cxp] for one oscillator.
        Returns ndarray of the same shape.
        """
        Vx = covariances[:, 0]
        Vp = covariances[:, 1]
        Cxp = covariances[:, 2]

        x0Sq = 1.0 / self.omegas  # per-oscillator

        dVx = (2 * Cxp
               - 4 * self.eta * self.gamma_meas * Vx**2 / x0Sq)
        dVp = (-2 * self.omegas**2 * Cxp
               + self.gamma_meas / x0Sq
               - 4 * self.eta * self.gamma_meas * Cxp**2 / x0Sq)
        dCxp = (Vp
                - self.omegas**2 * Vx
                - 4 * self.eta * self.gamma_meas * Vx * Cxp / x0Sq)

        return np.column_stack([dVx, dVp, dCxp])

    def variance_solver(self):
        """
        Returns
        -------
        dict with keys 'Vxx', 'Vpp', 'Cxp' (each shape (N, n_times))
        and 'times' (shape (n_times,)).
        """
        dt = self.dt
        n_times = int(2 * np.pi * self.n_periods / dt)

        # initial conditions: shape (N, 3)
        init_val = (1 + self.n_thermal) / 2
        y = np.zeros((n_times, self.N, 3))
        y[0, :, 0] = init_val
        y[0, :, 1] = init_val
        y[0, :, 2] = 0.0
        times = np.arange(n_times) * dt

        for i in range(1, n_times):
            t = (i - 1) * dt
            cur = y[i - 1]

            k1 = self.dvariance(t, cur)
            k2 = self.dvariance(t + dt / 2, cur + dt / 2 * k1)
            k3 = self.dvariance(t + dt / 2, cur + dt / 2 * k2)
            k4 = self.dvariance(t + dt, cur + dt * k3)
            y[i] = cur + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Rescale to dimensionless units (per-oscillator omega)
        Vxx = y[:, :, 0].T * self.omegas[:, None] * 2        # shape (N, n_times)
        Vpp = y[:, :, 1].T * 2 / self.omegas[:, None]
        Cxp = y[:, :, 2].T

        return {'Vxx': Vxx, 'Vpp': Vpp, 'Cxp': Cxp, 'times': times}

    def steady_state_variances(self):
        """
        Returns arrays of shape (N,) for Vx, Vp, Cxp
        (eq. 51 of quant-ph/9812004, evaluated per oscillator).
        """
        r = self.omegas**2 / (2 * self.eta * self.k_jacobs)
        xi = np.sqrt(1 + 4 / (self.eta * r**2))
        Vx = 1 / (np.sqrt(2 * self.eta) * self.omegas) / np.sqrt(xi + 1)
        Vp = self.omegas / np.sqrt(2 * self.eta) * xi / np.sqrt(xi + 1)
        Cxp = 1 / (2 * np.sqrt(self.eta)) * np.sqrt(xi - 1) / np.sqrt(xi + 1)
        return Vx, Vp, Cxp

    def expectation_solver(self, feedback_fns=None, common_feedback_fn=None,
                       gamma_fb=0.1, initial_conditions=None,
                       covariances=None):
        """
    Parameters
    ----------
    feedback_fns : list of callables or None
        Independent per-oscillator feedback: fn(x_a, p_a) -> force.
        Ignored if common_feedback_fn is provided.
    common_feedback_fn : callable or None
        A coupled feedback function with signature fn(xs, ps) -> scalar,
        where xs and ps are length-N arrays.  The same scalar force is
        applied to every oscillator.  Takes precedence over feedback_fns.
    gamma_fb : float
        Used only to build default independent feedback when both
        feedback_fns and common_feedback_fn are None.
    initial_conditions : ndarray of shape (N, 3) or None
    covariances : dict or None

    Returns
    -------
    dict with 'xc', 'pc', 'photocurrent' (each (N, n_times)) and 'times'.
    """
        dt = self.dt
        n_times = int(2 * np.pi * self.n_periods / dt)
    
        use_common = common_feedback_fn is not None
    
        if not use_common:
            if feedback_fns is None:
                feedback_fns = []
                for omega in self.omegas:
                    fb = FeedbackForces(g_fb=gamma_fb, omega=omega)
                    feedback_fns.append(fb.optimal_feedback())
    
        if covariances is None:
            covariances = self.variance_solver()
    
        var_x = covariances['Vxx']      # (N, n_times)
        cov_xp = covariances['Cxp']     # (N, n_times)
    
        if initial_conditions is None:
            initial_conditions = np.tile([20.0, 20.0, 0.0], (self.N, 1))
        initial_conditions = np.asarray(initial_conditions, dtype=float)
    
        y = np.zeros((self.N, 3, n_times))
        y[:, :, 0] = initial_conditions
        times = np.arange(n_times) * dt
    
        dW = np.sqrt(dt) * np.random.randn(self.N, n_times)
        sqrt_2eta_gm = np.sqrt(2 * self.eta * self.gamma_meas)
    
        for i in range(1, n_times):
            x = y[:, 0, i - 1]
            p = y[:, 1, i - 1]
            dW_i = dW[:, i - 1]
    
            if use_common:
                force = common_feedback_fn(x, p)       # scalar
                u = np.full(self.N, force)              # same force on all
            else:
                u = np.array([fn(x[j], p[j])
                              for j, fn in enumerate(feedback_fns)])
    
            dp = (-self.omegas**2 * x * dt
                  + sqrt_2eta_gm * cov_xp[:, i - 1] * dW_i
                  - self.gammas * p * dt
                  + u * dt)
            p_new = p + dp
    
            dx = (p_new * dt
                  + sqrt_2eta_gm * var_x[:, i - 1] * dW_i)
            x_new = x + dx
    
            drecord = x * dt + dW_i / sqrt_2eta_gm
            photocurrent = drecord / dt
    
            y[:, 0, i] = x_new
            y[:, 1, i] = p_new
            y[:, 2, i] = photocurrent
            times[i] = i * dt
    
        return {'xc': y[:, 0, :],
            'pc': y[:, 1, :],
            'photocurrent': y[:, 2, :],
            'times': times}

    def find_n_bar(self, xc, pc, Vx, Vp):
        """
        Parameters can be arrays of shape (N, n_times) or (N,).
        Returns the same shape.
        """
        x2 = xc**2 + Vx
        p2 = pc**2 + Vp
        return (x2 + p2) / 4 - 0.5
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:42:58 2026

@author: giacomomarocco

2D NxN oscillator array: each site (a,b) has three uncoupled spatial modes
(omega_x, omega_y, omega_z). Row-wise x cold damping, column-wise y cold
damping, row-wise and column-wise parametric modulation. z-modes are
measured but only parametrically cooled (no cold damping).
"""

import numpy as np
from optimal_control import FeedbackForces


class GaussianOscillatorArray:
    def __init__(self, omegas, quality_factor=1e4, gamma_meas=5e-2, eta=1.0,
                 n_thermal=100, dt=0.01, n_periods=100):
        """
        Parameters
        ----------
        omegas : array-like, shape (N, N, 3)
            Angular frequencies [omega_x, omega_y, omega_z] per site (a,b).
        quality_factor, gamma_meas, eta, n_thermal, dt, n_periods :
            Shared parameters (same meaning as in GaussianOscillator).
        """
        self.omegas = np.asarray(omegas, dtype=float)  # (N, N, 3)
        self.N = self.omegas.shape[0]
        self.omegas_flat = self.omegas.reshape(-1)  # (3N^2,)
        self.quality_factor = quality_factor
        self.gammas = self.omegas_flat / quality_factor
        self.gamma_meas = gamma_meas
        self.eta = eta
        self.n_thermal = n_thermal
        self.dt = dt
        self.n_periods = n_periods
        self.k_jacobs = self.gamma_meas * self.omegas_flat / 2  # per mode

        # Coupling vectors for cold damping
        omega_bar_x = self.omegas[0, 0, 0]
        omega_bar_y = self.omegas[0, 0, 1]
        omegas_x = self.omegas[:, :, 0]  # (N, N)
        omegas_y = self.omegas[:, :, 1]  # (N, N)
        self.B_x = omegas_x * np.sqrt(omegas_x / omega_bar_x)  # (N, N)
        self.B_y = omegas_y * np.sqrt(omegas_y / omega_bar_y)  # (N, N)

    # ----- variance equations (vectorised over 3N^2 modes) -----

    def dvariance(self, time, covariances):
        """
        covariances : ndarray of shape (3N^2, 3)
            Each row is [Vx, Vp, Cxp] for one mode.
        Returns ndarray of the same shape.
        """
        Vx = covariances[:, 0]
        Vp = covariances[:, 1]
        Cxp = covariances[:, 2]

        dVx = (2 * self.omegas_flat * Cxp
               - 4 * self.eta * self.gamma_meas * Vx**2)
        dVp = (-2 * self.omegas_flat * Cxp
               + 4 * self.gamma_meas
               - 4 * self.eta * self.gamma_meas * Cxp**2)
        dCxp = (self.omegas_flat * (Vp - Vx)
                - 4 * self.eta * self.gamma_meas * Vx * Cxp)

        return np.column_stack([dVx, dVp, dCxp])

    def variance_solver(self, n_periods=None, dt=None):
        """
        Returns
        -------
        dict with keys 'Vxx', 'Vpp', 'Cxp' (each shape (N, N, 3, n_times))
        and 'times' (shape (n_times,)).
        """
        if n_periods is None:
            n_periods = self.n_periods
        if dt is None:
            dt = self.dt
        n_modes = 3 * self.N**2
        n_times = int(2 * np.pi * n_periods / dt)

        # initial conditions: shape (3N^2, 3)
        init_val = (1 + self.n_thermal) / 2
        y = np.zeros((n_times, n_modes, 3))
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

        # y[:, :, k].T -> (3N^2, n_times), reshape to (N, N, 3, n_times)
        Vxx = y[:, :, 0].T.reshape(self.N, self.N, 3, n_times)
        Vpp = y[:, :, 1].T.reshape(self.N, self.N, 3, n_times)
        Cxp = y[:, :, 2].T.reshape(self.N, self.N, 3, n_times)

        return {'Vxx': Vxx, 'Vpp': Vpp, 'Cxp': Cxp, 'times': times}

    def steady_state_variances(self):
        """
        Returns arrays of shape (N, N, 3) for Vx, Vp, Cxp per mode.
        """
        xi = np.sqrt(1 + 4 * self.eta * self.gamma_meas**2 / self.omegas_flat**2)
        Vx = 2 / (np.sqrt(2 * self.eta) * np.sqrt(xi + 1))
        Vp = 2 * xi / (np.sqrt(2 * self.eta) * np.sqrt(xi + 1))
        Cxp = np.sqrt(xi - 1) / (np.sqrt(self.eta) * np.sqrt(xi + 1))
        return (Vx.reshape(self.N, self.N, 3),
                Vp.reshape(self.N, self.N, 3),
                Cxp.reshape(self.N, self.N, 3))

    def expectation_solver(self, cold_damping_fn=None, parametric_fn=None,
                           gamma_fb=0.1, initial_conditions=None):
        """
        Parameters
        ----------
        cold_damping_fn : callable or None
            fn(xc, pc) -> (u_cd_x, u_cd_y), where xc, pc are (N, N, 3).
            u_cd_x: (N,) — one force per row.
            u_cd_y: (N,) — one force per column.
            Default: GridFeedbackLQR.
        parametric_fn : callable or None
            fn(xc, pc, t) -> (u_param_x, u_param_y).
            u_param_x: (N,) — per-row modulation.
            u_param_y: (N,) — per-column modulation.
            Default: no modulation.
        gamma_fb : float
            Used to build default cold damping if cold_damping_fn is None.
        initial_conditions : ndarray of shape (N, N, 3, 3) or None
            Per mode: [xc, pc, photocurrent]. Default: [20, 20, 0].

        Returns
        -------
        dict with 'xc', 'pc', 'photocurrent' each (N, N, 3, n_times),
        'Vxx', 'Vpp', 'Cxp' each (N, N, 3, n_times), and 'times'.
        """
        N = self.N
        dt = self.dt
        n_times = int(2 * np.pi * self.n_periods / dt)

        # Default cold damping: use GridFeedbackLQR
        if cold_damping_fn is None:
            from optimal_feedback_n_oscillators import GridFeedbackLQR
            lqr = GridFeedbackLQR(self.omegas,
                                  q=self.omegas[0, 0, 0] / gamma_fb**2)
            cold_damping_fn = lqr.cold_damping_fn()

        # Default parametric: no modulation
        if parametric_fn is None:
            def parametric_fn(xc, pc, t):
                return np.zeros(N), np.zeros(N)

        # Initial conditions: (N, N, 3, 3)
        if initial_conditions is None:
            initial_conditions = np.tile([20.0, 20.0, 0.0], (N, N, 3, 1))
        initial_conditions = np.asarray(initial_conditions, dtype=float)

        # State array: (N, N, 3, 3, n_times) — [row, col, mode, xc/pc/photo, time]
        y = np.zeros((N, N, 3, 3, n_times))
        y[:, :, :, :, 0] = initial_conditions
        times = np.arange(n_times) * dt

        # Covariance arrays: (N, N, 3, n_times)
        init_val = (1 + self.n_thermal) / 2
        Vxx = np.zeros((N, N, 3, n_times))
        Vpp = np.zeros((N, N, 3, n_times))
        Cxp_arr = np.zeros((N, N, 3, n_times))
        Vxx[:, :, :, 0] = init_val
        Vpp[:, :, :, 0] = init_val

        # Independent Wiener processes per site per mode
        dW = np.sqrt(dt) * np.random.randn(N, N, 3, n_times)
        sqrt_2eta_gm = 2 * np.sqrt(self.eta * self.gamma_meas)

        for i in range(1, n_times):
            x = y[:, :, :, 0, i - 1]     # (N, N, 3)
            p = y[:, :, :, 1, i - 1]     # (N, N, 3)
            dW_i = dW[:, :, :, i - 1]    # (N, N, 3)
            vxx = Vxx[:, :, :, i - 1]    # (N, N, 3)
            vpp = Vpp[:, :, :, i - 1]    # (N, N, 3)
            cxp = Cxp_arr[:, :, :, i - 1]  # (N, N, 3)

            t = (i - 1) * dt

            # Get control signals
            u_cd_x, u_cd_y = cold_damping_fn(x, p)      # (N,), (N,)
            u_param_x, u_param_y = parametric_fn(x, p, t)  # (N,), (N,)

            # Modified frequencies: omega_mod = omega * (1 + u_param_x + u_param_y)
            omega_mod = self.omegas * (1 + u_param_x[:, None, None]
                                       + u_param_y[None, :, None])  # (N, N, 3)

            # Noise terms
            noise_p = sqrt_2eta_gm * cxp * dW_i   # (N, N, 3)
            noise_x = sqrt_2eta_gm * vxx * dW_i   # (N, N, 3)

            # Momentum update (all modes)
            dp = -omega_mod * x * dt + noise_p
            # x-mode: u_cd_x[A] acts on all (A,b) via B_x
            dp[:, :, 0] += self.B_x * u_cd_x[:, None] * dt
            # y-mode: u_cd_y[B] acts on all (a,B) via B_y
            dp[:, :, 1] += self.B_y * u_cd_y[None, :] * dt
            p_new = p + dp

            # Position update (all modes)
            dx = self.omegas * p_new * dt + noise_x
            x_new = x + dx

            # Photocurrent (per mode, independent measurements)
            drecord = x * dt + dW_i / sqrt_2eta_gm
            photocurrent = drecord / dt

            y[:, :, :, 0, i] = x_new
            y[:, :, :, 1, i] = p_new
            y[:, :, :, 2, i] = photocurrent

            # Covariance co-evolution (all modes, with omega_mod)
            dVxx = (2 * self.omegas * cxp
                    - 4 * self.eta * self.gamma_meas * vxx**2) * dt
            dVpp = (-2 * omega_mod * cxp
                    + 4 * self.gamma_meas
                    - 4 * self.eta * self.gamma_meas * cxp**2) * dt
            dCxp = (self.omegas * vpp - omega_mod * vxx
                    - 4 * self.eta * self.gamma_meas * vxx * cxp) * dt

            Vxx[:, :, :, i] = vxx + dVxx
            Vpp[:, :, :, i] = vpp + dVpp
            Cxp_arr[:, :, :, i] = cxp + dCxp

        return {
            'xc': y[:, :, :, 0, :],
            'pc': y[:, :, :, 1, :],
            'photocurrent': y[:, :, :, 2, :],
            'times': times,
            'Vxx': Vxx,
            'Vpp': Vpp,
            'Cxp': Cxp_arr,
        }

    def find_n_bar(self, xc, pc, Vx, Vp):
        """
        Per-mode phonon number.

        Parameters can be arrays of shape (N, N, 3, n_times) or (N, N, 3).
        Returns per-mode n_bar (same shape), and per-site n_bar
        is obtained by summing over axis 2 (mode axis).
        """
        x2 = xc**2 + Vx
        p2 = pc**2 + Vp
        return (x2 + p2) / 4 - 0.5

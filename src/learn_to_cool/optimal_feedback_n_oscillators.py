#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:45:28 2026

@author: giacomomarocco
"""

import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.linalg import solve_continuous_lyapunov



class OptimalFeedbackNOscillators:
    """
    Compute optimal LQR feedback for N non-degenerate oscillators
    subject to a common force.

    State vector: [xc1, pc1, xc2, pc2, ..., xcN, pcN]
    """

    def __init__(self, omegas, q=1.0):
        """
        Parameters
        ----------
        omegas : array-like, shape (N, 2)
            Frequencies [omega_x, omega_y] per oscillator.
            LQR uses x-mode frequencies (column 0) only.
        q : float
            Control cost parameter (higher = more expensive control).
        """
        omegas = np.asarray(omegas, dtype=float)
        if omegas.ndim == 1:
            omegas = omegas[:, np.newaxis] * np.ones(2)  # broadcast to (N, 2)
        self.omegas_full = omegas          # (N, 2)
        self.omegas = omegas[:, 0]         # x-mode freqs for LQR
        self.N = omegas.shape[0]
        self.q = q

        self.A_full = self._build_A()
        self.B_full = self._build_B()
        self.H_full = self._build_H()
        self.Q_full = self._build_Q()

        self.P = self._solve_care()
        self.K = self._compute_gain()

    def _build_A(self):
        """Block-diagonal drift matrix: diag(A1, A2, ..., AN)."""
        dim = 2 * self.N
        A = np.zeros((dim, dim))
        for a, omega in enumerate(self.omegas):
            i = 2 * a
            A[i, i + 1] = omega
            A[i + 1, i] = -omega
        return A

    def _build_B(self, omega_bar=None):
        """
        Control vector B (2N x 1) from notes/multiparticle_feedback.md.
        B_a = omega_a * sqrt(omega_a / omega_bar) * [0, 1]^T
        """
        if omega_bar is None:
            omega_bar = self.omegas[0]
        B = np.zeros((2 * self.N, 1))
        for a, omega in enumerate(self.omegas):
            B[2 * a + 1, 0] = omega * np.sqrt(omega / omega_bar)
        return B

    def _build_H(self):
        """
        State cost matrix H from notes/multiparticle_feedback.md.
        H = (1/4N) * diag(omega_1 I_2, ..., omega_N I_2)
        """
        dim = 2 * self.N
        H = np.zeros((dim, dim))
        for a, omega in enumerate(self.omegas):
            i = 2 * a
            H[i:i + 2, i:i + 2] = omega * np.eye(2)
        return H / (4.0 * self.N)


    # def _build_Q(self):
    #     """Control cost: q * I_{2N}."""
    #     return self.q * np.eye(2 * self.N)
    def _build_Q(self):
        """Control cost scalar q"""
        return np.array([[self.q/4]])

    def _solve_care(self):
        """
        Solve the CARE:
          A^T P + P A - P B  B^T P / q + H = 0
        """
        return solve_continuous_are(self.A_full, self.B_full, self.H_full, self.Q_full)

    # def _compute_gain(self):
    #     """K = Q^{-1} B^T U."""
    #     Q_inv = np.linalg.inv(self.Q_full)
    #     return Q_inv @ self.B_full.T @ self.U

    def _compute_gain(self):
        """K = (1/q) * B^T @ U, a 1 x 2N row vector."""
        return (1.0 / self.q) * self.B_full.T @ self.P

    # def optimal_feedback_vec(self):
    #     """
    #     Return the effective gain vector g of length 2N such that
    #     force = -g @ state.

    #     The common scalar force on every oscillator is
    #       force = (1/N) * sum_a u_{p_a}
    #     where u = -K @ state. So g = (1/N) * sum_a K[2a+1, :].
    #     """
    #     p_rows = self.K[1::2, :]  # rows 1, 3, 5, ... (momentum components)
    #     g = p_rows.mean(axis=0)   # (1/N) * sum
    #     return g

    def optimal_feedback_vec(self):
        """Return the gain vector g (length 2N) such that u = -g @ state."""
        return self.K.flatten()

    # def optimal_feedback(self):
    #     """
    #     Return a function f(state) -> scalar force, where
    #     state is a 1-D array [xc1, pc1, xc2, pc2, ..., xcN, pcN].
    #     """
    #     g = self.optimal_feedback_vec()

    #     def feedback_fn(state):
    #         return -g @ state

    #     return feedback_fn


    def optimal_feedback(self):
        """Return a function f(state) -> scalar force."""
        g = self.optimal_feedback_vec()
        def feedback_fn(state):
            return -g @ state
        return feedback_fn
    
    
    # def feedback_for_array(self):
    #     """
    #     Return a list of N feedback functions compatible with
    #     GaussianOscillatorArray.expectation_solver, which expects
    #     one function per oscillator with signature fn(x_a, p_a).

    #     Each returned function closes over the *shared* state and
    #     must be called together inside a wrapper that first assembles
    #     the full state, computes the common force once, and distributes
    #     it.  This method therefore returns a single callable

    #         common_force_fn(xs, ps) -> scalar

    #     where xs and ps are length-N arrays of the current conditional
    #     means, plus a helper that wraps it into per-oscillator callables.
    #     """
    #     g = self.optimal_feedback_vec()

    #     def common_force(xs, ps):
    #         state = np.empty(2 * self.N)
    #         state[0::2] = xs
    #         state[1::2] = ps
    #         return -g @ state

    #     return common_force
    
    
    def common_force_fn(self):
        """Return common_force(xs, ps) -> scalar, compatible with array solver."""
        g = self.optimal_feedback_vec()
        def common_force(xs, ps):
            state = np.empty(2 * self.N)
            state[0::2] = xs
            state[1::2] = ps
            return -g @ state
        return common_force
    
    def optimal_feedback_nn(self, policy, u_max=25.0):
        """
        Return a feedback function that uses a trained neural network policy.
        
        Parameters
        ----------
        policy : torch.nn.Module
            Trained policy network that takes state [xc1, pc1, xc2, pc2, ...]
            and outputs a raw control signal.
        u_max : float
            Maximum control force (policy output is passed through u_max * tanh).
        
        Returns
        -------
        common_force_fn : callable
            Function with signature common_force(xs, ps) -> scalar,
            compatible with feedback_for_array().
        """
        import torch
    
        def common_force(xs, ps):
            state = np.empty(2 * self.N)
            state[0::2] = xs
            state[1::2] = ps
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():
                raw_u = policy(state_tensor).squeeze()
                u = u_max * torch.tanh(raw_u)
            return u.item()
    
        return common_force
    
# Steady-state conditional covariance (Kalman filter)
    # ----------------------------------------------------------------

    @staticmethod
    def _conditional_covariance_single(omega, eta, Gamma_BA):
        """
        Steady-state conditional covariance matrix for a single oscillator
        measured with efficiency eta and back-action rate Gamma_BA.

        Returns the 2x2 matrix V = [[Vxx, Cxp], [Cxp, Vpp]].
        """
        xi = np.sqrt(1.0 + 4.0 * eta * Gamma_BA**2 / omega**2)

        Vxx = 2.0 / (np.sqrt(2.0 * eta) * np.sqrt(xi + 1.0))
        Vpp = 2.0 * xi / (np.sqrt(2.0 * eta) * np.sqrt(xi + 1.0))
        Cxp = np.sqrt(xi - 1.0) / (np.sqrt(eta) * np.sqrt(xi + 1.0))

        return np.array([[Vxx, Cxp],
                         [Cxp, Vpp]])

    # ----------------------------------------------------------------
    # Noise matrix for the conditioned-mean SDE
    # ----------------------------------------------------------------

    def _build_noise_covariance(self, eta, Gamma_BA):
        """
        Build the diffusion matrix GG^T for the full 2N-dimensional
        conditioned-mean SDE.

        Each oscillator contributes independently (block-diagonal),
        with the a-th 2x2 block being

            2 * eta * Gamma_BA * [[Vxx^2,      Vxx*Cxp],
                                  [Vxx*Cxp,    Cxp^2  ]]

        (rank-1: G_a G_a^T where G_a = sqrt(2*eta*Gamma_BA) * [Vxx, Cxp]^T).
        """
        dim = 2 * self.N
        D = np.zeros((dim, dim))
        for a, omega in enumerate(self.omegas):
            V = self._conditional_covariance_single(omega, eta, Gamma_BA)
            Vxx = V[0, 0]
            Cxp = V[0, 1]
            g_vec = np.array([Vxx, Cxp]) * np.sqrt(2.0 * eta * Gamma_BA)
            i = 2 * a
            D[i:i + 2, i:i + 2] = np.outer(g_vec, g_vec)
        return D

    # ----------------------------------------------------------------
    # Steady-state total covariance and phonon number
    # ----------------------------------------------------------------

    def steady_state_covariance(self, eta, Gamma_BA):
        """
        Compute the total steady-state covariance matrix (2N x 2N).

        This is Sigma_c + V, where:
          - Sigma_c solves the Lyapunov equation for the conditioned-mean
            fluctuations under the closed-loop dynamics,
          - V is the block-diagonal conditional covariance from the
            Kalman filter.

        Parameters
        ----------
        eta : float
            Measurement efficiency (0 < eta <= 1).
        Gamma_BA : float
            Measurement back-action rate.

        Returns
        -------
        Sigma_total : ndarray, shape (2N, 2N)
        """
        A_cl = self.A_full - self.B_full @ self.K  # closed-loop A
        D = self._build_noise_covariance(eta, Gamma_BA)

        # Lyapunov equation: A_cl @ Sigma_c + Sigma_c @ A_cl^T + D = 0
        # scipy convention: A X + X A^H + Q = 0
        Sigma_c = solve_continuous_lyapunov(A_cl, -D)

        # Add the conditional covariance (block-diagonal)
        dim = 2 * self.N
        V_full = np.zeros((dim, dim))
        for a, omega in enumerate(self.omegas):
            i = 2 * a
            V_full[i:i + 2, i:i + 2] = self._conditional_covariance_single(
                omega, eta, Gamma_BA
            )
        return Sigma_c + V_full

    def steady_state_nbar(self, eta, Gamma_BA):
        """
        Compute the steady-state mean phonon number for each oscillator.

        nbar_a = (1/2)(Sigma_{x_a x_a} + Sigma_{p_a p_a}) - 1/2

        Parameters
        ----------
        eta : float
            Measurement efficiency.
        Gamma_BA : float
            Measurement back-action rate.

        Returns
        -------
        nbars : ndarray, shape (N,)
            Mean phonon number for each oscillator.
        """
        Sigma = self.steady_state_covariance(eta, Gamma_BA)
        nbars = np.empty(self.N)
        for a in range(self.N):
            i = 2 * a
            nbars[a] = 0.25 * (Sigma[i, i] + Sigma[i + 1, i + 1]) - 0.5
        return nbars

    def total_nbar(self, eta, Gamma_BA):
        """
        Total mean phonon number summed over all oscillators.
        """
        return np.sum(self.steady_state_nbar(eta, Gamma_BA))

    def mean_nbar(self, eta, Gamma_BA):
        """
        Mean phonon number averaged over all oscillators.
        """
        return np.mean(self.steady_state_nbar(eta, Gamma_BA))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:18:51 2026

@author: giacomomarocco
"""

import numpy as np
import torch


class TorchOscillatorEnv:
    """
    Batched Gaussian oscillator array environment in PyTorch.

    Supports:
      - Multiple oscillators (N)
      - Cold damping (action = force on momentum)
      - Parametric feedback (action = fractional modulation of omega^2)
      - Precomputed or co-evolved covariances
    """

    def __init__(self, oscillator, batch_size, dt=0.05,
                 phase_space_range=10.0, n_thermal=10,
                 mode='cold_damping', horizon=600,
                 modulation_depth=0.5, cost_u_weight=0.0,
                 omega_bar=None):
        """
        Parameters
        ----------
        oscillator : GaussianOscillator or GaussianOscillatorArray
            Source of physical parameters.
        batch_size : int
        dt : float
        phase_space_range : float
            Initial conditions drawn from [-r, r].
        n_thermal : float
            Thermal occupation for initial covariances.
        mode : str
            'cold_damping' or 'parametric'.
        horizon : int
            Number of timesteps (needed to precompute covariances in cold_damping mode).
        modulation_depth : float
            Max fractional change in omega^2 (parametric mode only).
        cost_u_weight : float
            Weight on u^2 in the cost (q).
        omega_bar : float, optional
            Reference frequency for the shared force B_a = omega_a * sqrt(omega_a/omega_bar).
            If None, uses the first oscillator's frequency.
        """
        if hasattr(oscillator, 'omegas'):
            # It's an array
            self.omegas = torch.tensor(oscillator.omegas, dtype=torch.float32)
            self.N = len(self.omegas)
        else:
            # Single oscillator
            self.omegas = torch.tensor([oscillator.omega], dtype=torch.float32)
            self.N = 1

        self.eta = oscillator.eta
        self.gamma_meas = oscillator.gamma_meas
        self.dt = dt
        self.batch_size = batch_size
        self.phase_space_range = phase_space_range
        self.n_thermal = n_thermal
        self.mode = mode
        self.horizon = horizon
        self.modulation_depth = modulation_depth
        self.cost_u_weight = cost_u_weight
        self.sqrt_2eta_gm = 2.0 * np.sqrt(self.eta * self.gamma_meas)

        # B vector for shared force: B_a = omega_a * sqrt(omega_a / omega_bar)
        if omega_bar is None:
            omega_bar = self.omegas[0].item()
        self.B = self.omegas * torch.sqrt(self.omegas / omega_bar)

        if self.mode == 'cold_damping':
            horizon_periods = horizon * dt / (2 * np.pi)
            if hasattr(oscillator, 'variance_solver'):
                # Handle both GaussianOscillator and GaussianOscillatorArray
                res = oscillator.variance_solver(n_periods=horizon_periods, dt=dt)

                # Both return dictionaries now.
                # GaussianOscillatorArray returns dict with (N, n_times)
                # GaussianOscillator returns dict with (n_times,)

                vxx, vpp, cxp = res['Vxx'], res['Vpp'], res['Cxp']

                if vxx.ndim == 1:
                    # Single oscillator (GaussianOscillator)
                    # Reshape to (n_times, N, 3) where N=1
                    covs = np.stack([vxx, vpp, cxp], axis=-1)[:, np.newaxis, :]
                else:
                    # Array of oscillators (GaussianOscillatorArray)
                    # Transpose (N, n_times, 3) -> (n_times, N, 3)
                    covs = np.stack([vxx, vpp, cxp], axis=-1).transpose(1, 0, 2)

                # We need (horizon, N) for each covariance component
                self._Vxx_precomputed = torch.tensor(covs[:, :, 0], dtype=torch.float32)
                self._Vpp_precomputed = torch.tensor(covs[:, :, 1], dtype=torch.float32)
                self._Cxp_precomputed = torch.tensor(covs[:, :, 2], dtype=torch.float32)

        self.reset()

    def reset(self, initial_conditions=None):
        """Reset state. Optionally pass (batch_size, N, 2) tensor of [x, p]."""
        if initial_conditions is not None:
            self.xc = initial_conditions[:, :, 0].clone()
            self.pc = initial_conditions[:, :, 1].clone()
        else:
            r = self.phase_space_range
            self.xc = torch.zeros((self.batch_size, self.N)).uniform_(-r, r)
            self.pc = torch.zeros((self.batch_size, self.N)).uniform_(-r, r)
    
        
        self.x_true = self.xc.clone().detach()
        self.p_true = self.pc.clone().detach()

        if self.mode == 'parametric':
            # # Initialize covariances to thermal state
            # self.Vxx = torch.full((self.batch_size, self.N), (self.n_thermal + 0.5) / 2.0)
            # self.Vpp = torch.full((self.batch_size, self.N), (self.n_thermal + 0.5) / 2.0)
            # self.Cxp = torch.zeros((self.batch_size, self.N))
            # Initialize covariances to coherent state
            self.Vxx = torch.full((self.batch_size, self.N), 1.0)
            self.Vpp = torch.full((self.batch_size, self.N), 1.0)
            self.Cxp = torch.zeros((self.batch_size, self.N))


        self.t = 0
        return self.state()

    def state(self):
        """Return (batch_size, 2*N) state vector for the policy."""
        return torch.cat([self.xc, self.pc], dim=1)

    def _get_covariances(self):
        """Return (Vxx, Vpp, Cxp) for the current timestep."""
        if self.mode == 'cold_damping':
            t_idx = min(self.t, len(self._Vxx_precomputed) - 1)
            # Precomputed covariances are (horizon, N)
            # We want (batch_size, N) to match xc/pc
            return (self._Vxx_precomputed[t_idx].expand(self.batch_size, -1),
                    self._Vpp_precomputed[t_idx].expand(self.batch_size, -1),
                    self._Cxp_precomputed[t_idx].expand(self.batch_size, -1))
        else:
            return self.Vxx, self.Vpp, self.Cxp

    def n_bar(self):
        """Return (batch_size, N) phonon numbers."""
        Vxx, Vpp, _ = self._get_covariances()
        return (self.xc**2 + Vxx + self.pc**2 + Vpp) / 4.0 - 0.5

    def step(self, raw_action):
        """
        Advance one timestep.

        Parameters
        ----------
        raw_action : (batch_size,) or scalar tensor
            In cold_damping mode: the common force u applied (used directly).
            In parametric mode: the common fractional modulation u passed through modulation_depth * tanh.

        Returns
        -------
        cost : (batch_size,) tensor
            The average energy cost across N oscillators + control cost.
        """
        Vxx, Vpp, Cxp = self._get_covariances()

        # Ensure raw_action is (batch_size,)
        raw_action = raw_action.view(self.batch_size)

        # State cost: Average energy (sum_a omega_a * E_a) / (4N)
        # Note: (xc^2 + Vxx + pc^2 + Vpp) is 4 * (n_a + 0.5)
        # Average energy = (1/4N) * sum_a omega_a * (xc_a^2 + Vxx_a + pc_a^2 + Vpp_a)
        # We can calculate it as sum_a omega_a * (n_a + 0.5) / N
        n_a = self.n_bar()
        state_cost = torch.mean(self.omegas * (self.xc**2 + self.pc**2), dim=1)

        dW = torch.randn((self.batch_size, self.N)) * np.sqrt(self.dt)

        # if self.mode == 'cold_damping':
            # u = raw_action
            # cost = state_cost + self.cost_u_weight * u**2

            # # Shared force: B_a * u
            # # pc is (batch_size, N), xc is (batch_size, N)
            # # omegas is (N,), B is (N,)
            # # u is (batch_size,) -> expand to (batch_size, N)
            # u_expanded = u.unsqueeze(1).expand(-1, self.N)

            # dpc = (-self.omegas * self.xc * self.dt
            #        + self.sqrt_2eta_gm * Cxp * dW
            #        + self.B * u_expanded * self.dt)
            # self.pc = self.pc + dpc
            # self.xc = self.xc + (self.omegas * self.pc * self.dt
            #                       + self.sqrt_2eta_gm * Vxx * dW)
            
        if self.mode == 'cold_damping':
            u = raw_action
            cost = state_cost + self.cost_u_weight * u**2
        
            u_expanded = u.unsqueeze(1).expand(-1, self.N)
        
        #     dW = torch.randn((self.batch_size, self.N)) * np.sqrt(self.dt)
        #     noise_x = (self.sqrt_2eta_gm * Vxx * dW).detach()
        #     noise_p = (self.sqrt_2eta_gm * Cxp * dW).detach()
        
        #     # new_pc = self.pc + (-self.omegas * self.xc * self.dt
        #     #                      + noise_p
        #     #                      + self.B * u_expanded * self.dt)
        #     # new_xc = self.xc + (self.omegas * new_pc * self.dt + noise_x)
        
        
        #     # Construct measurement outcome dY from current state + noise, then detach
        #     dW_raw = torch.randn((self.batch_size, self.N)) * np.sqrt(self.dt)
        #     dY = (self.sqrt_2eta_gm * self.xc * self.dt + dW_raw).detach()
        
        #     # Innovation — gradients flow through xc
        #     innovation = dY - self.sqrt_2eta_gm * self.xc * self.dt
        
        #     # Conditional mean update
        #     new_xc = self.xc + (self.omegas * self.pc * self.dt
        #                          + self.sqrt_2eta_gm * Vxx * innovation / self.dt)
        #     new_pc = self.pc + (-self.omegas * self.xc * self.dt
        #                          + self.sqrt_2eta_gm * Cxp * innovation / self.dt
        #                          + self.B * u_expanded * self.dt)
        
            dW = torch.randn((self.batch_size, self.N)) * np.sqrt(self.dt)
            noise_x = (self.sqrt_2eta_gm * Vxx * dW).detach()
            noise_p = (self.sqrt_2eta_gm * Cxp * dW).detach()
            
            new_xc = self.xc + (self.omegas * self.pc * self.dt + noise_x)
            new_pc = self.pc + (-self.omegas * self.xc * self.dt
                                 + noise_p
                                 + self.B * u_expanded * self.dt)
            
            self.xc = new_xc
            self.pc = new_pc
        elif self.mode == 'parametric':
            # In parametric mode, modulation is usually fractional modulation of omega^2
            # u is (batch_size,)
            u = self.modulation_depth * torch.tanh(raw_action)
            cost = state_cost + self.cost_u_weight * u**2

            # Effective omega^2 factor: (1+u)
            u_expanded = u.unsqueeze(1).expand(-1, self.N)
            omega_mod = self.omegas * (1.0 + u_expanded)

            # dpc = (-omega_mod * self.xc * self.dt
            #        + self.sqrt_2eta_gm * Cxp * dW)
            # self.pc = self.pc + dpc
            # self.xc = self.xc + (self.omegas * self.pc * self.dt
            #                       + self.sqrt_2eta_gm * Vxx * dW)
            
            new_pc = self.pc + (-self.omegas * self.xc * self.dt
                     + noise_p
                     + self.B * u_expanded * self.dt)
            new_xc = self.xc + (self.omegas * self.pc * self.dt + noise_x)  # use old pc
            
            self.pc = new_pc
            self.xc = new_xc

            # Co-evolve covariances for each oscillator
            # Vxx is (batch_size, N)
            dVxx = (2.0 * self.omegas * Cxp
                    - 4.0 * self.eta * self.gamma_meas * Vxx**2) * self.dt
            dVpp = (-2.0 * omega_mod * Cxp
                    + 4.0 * self.gamma_meas
                    - 4.0 * self.eta * self.gamma_meas * Cxp**2) * self.dt
            dCxp = (self.omegas * Vpp - omega_mod * Vxx
                    - 4.0 * self.eta * self.gamma_meas * Vxx * Cxp) * self.dt

            self.Vxx = torch.clamp(Vxx + dVxx, min=1e-4)
            self.Vpp = torch.clamp(Vpp + dVpp, min=1e-4)
            self.Cxp = Cxp + dCxp

        self.t += 1
        return cost

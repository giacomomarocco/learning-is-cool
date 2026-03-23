import numpy as np
import torch


class TorchOscillatorEnv:
    """
    Batched 2D Gaussian oscillator array environment in PyTorch.

    Each oscillator has two uncoupled spatial modes (x, y).
    Both modes feel scalar parametric modulation u_param;
    only x-mode additionally feels cold damping force u_cd.

    States: (batch, N, 2) where [:,:,0] = x-mode, [:,:,1] = y-mode.
    Action: (batch, 2) — [u_cd, u_param_raw].
    """

    def __init__(self, oscillator, batch_size, dt=0.05,
                 phase_space_range=10.0, n_thermal=10,
                 horizon=600, modulation_depth=0.5, cost_u_weight=0.0,
                 omega_bar=None, device=None):
        # --- Device setup ---
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # omegas: (N, 2)
        self.omegas = torch.tensor(oscillator.omegas, dtype=torch.float32, device=self.device)  # (N, 2)
        self.N = self.omegas.shape[0]

        self.eta = oscillator.eta
        self.gamma_meas = oscillator.gamma_meas
        self.dt = dt
        self.batch_size = batch_size
        self.phase_space_range = phase_space_range
        self.n_thermal = n_thermal
        self.horizon = horizon
        self.modulation_depth = modulation_depth
        self.cost_u_weight = cost_u_weight
        self.sqrt_2eta_gm = 2.0 * np.sqrt(self.eta * self.gamma_meas)

        # B coefficients for cold damping (x-mode only)
        if omega_bar is None:
            omega_bar = self.omegas[0, 0].item()
        omegas_x = self.omegas[:, 0]  # (N,)
        self.B = omegas_x * torch.sqrt(omegas_x / omega_bar)  # (N,)

        self.reset()

    def reset(self, initial_conditions=None):
        """
        initial_conditions : tensor of shape (batch, N, 2, 2) — [osc, mode, x/p]
            or None for random initialization.
        """
        if initial_conditions is not None:
            self.xc = initial_conditions[:, :, :, 0].clone().to(self.device)  # (batch, N, 2)
            self.pc = initial_conditions[:, :, :, 1].clone().to(self.device)  # (batch, N, 2)
        else:
            r = self.phase_space_range
            self.xc = (2 * r) * torch.rand((self.batch_size, self.N, 2), device=self.device) - r
            self.pc = (2 * r) * torch.rand((self.batch_size, self.N, 2), device=self.device) - r

        # Co-evolve covariances, initialized from thermal state
        self.Vxx = torch.full((self.batch_size, self.N, 2), (1.0 + self.n_thermal), device=self.device)
        self.Vpp = torch.full((self.batch_size, self.N, 2), (1.0 + self.n_thermal), device=self.device)
        self.Cxp = torch.zeros((self.batch_size, self.N, 2), device=self.device)

        self.t = 0
        return self.state()

    def state(self):
        """Flat state: (batch, 4N) — [xc_x1, xc_y1, xc_x2, xc_y2, ..., pc_x1, pc_y1, ...]"""
        return torch.cat([self.xc.flatten(1), self.pc.flatten(1)], dim=1)

    def n_bar(self):
        """
        Per-mode phonon number: (batch, N, 2).
        Returns per-oscillator sum: (batch, N).
        """
        n_per_mode = (self.xc**2 + self.Vxx + self.pc**2 + self.Vpp) / 4.0 - 0.5
        return n_per_mode.sum(dim=2)  # sum over modes

    def mean_energy(self):
        """Mean energy: average of omega * (n_bar_mode + 1/2) over all modes and oscillators."""
        n_per_mode = (self.xc**2 + self.Vxx + self.pc**2 + self.Vpp) / 4.0 - 0.5
        energy_per_mode = self.omegas * (n_per_mode + 0.5)  # (batch, N, 2)
        return torch.mean(energy_per_mode, dim=(1, 2))  # average over all modes

    def step(self, raw_action):
        """
        raw_action : (batch, 2) — [u_cd, u_param_raw]
        """
        Vxx, Vpp, Cxp = self.Vxx, self.Vpp, self.Cxp

        u_cold = raw_action[:, 0]      # (batch,)
        u_param = self.modulation_depth * torch.tanh(raw_action[:, 1])  # (batch,)

        state_cost = self.mean_energy()
        cost = state_cost + self.cost_u_weight * u_cold**2 / 4

        # Modified frequencies: (batch, N, 2)
        omega_mod = self.omegas * (1.0 + u_param[:, None, None])

        # Independent noise per oscillator per mode: (batch, N, 2)
        dW = torch.randn((self.batch_size, self.N, 2), device=self.device) * np.sqrt(self.dt)

        noise_x = (self.sqrt_2eta_gm * Vxx * dW).detach()
        noise_p = (self.sqrt_2eta_gm * Cxp * dW).detach()

        # Momentum update (both modes)
        new_pc = self.pc + (-omega_mod * self.xc * self.dt + noise_p)
        # x-mode only: add cold damping force
        new_pc[:, :, 0] = new_pc[:, :, 0] + self.B * u_cold[:, None] * self.dt

        # Position update (both modes)
        new_xc = self.xc + (self.omegas * new_pc * self.dt + noise_x)

        self.pc = new_pc
        self.xc = new_xc

        # Covariance co-evolution (both modes, with omega_mod)
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

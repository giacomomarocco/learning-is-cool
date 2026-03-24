import numpy as np
import torch


class TorchOscillatorEnv:
    """
    Batched 2D NxN Gaussian oscillator array environment in PyTorch.

    Each site (a,b) has three uncoupled spatial modes (x, y, z).
    Row-wise x cold damping, column-wise y cold damping,
    row-wise and column-wise parametric modulation.
    z-modes are measured but only parametrically cooled.

    States: (batch, N, N, 3).
    Action: (batch, 4N) — [u_cd_x(N), u_cd_y(N), u_param_x_raw(N), u_param_y_raw(N)].
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

        # omegas: (N, N, 3)
        self.omegas = torch.tensor(oscillator.omegas, dtype=torch.float32, device=self.device)
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

        # B coefficients for cold damping
        if omega_bar is None:
            omega_bar_x = self.omegas[0, 0, 0].item()
            omega_bar_y = self.omegas[0, 0, 1].item()
        else:
            omega_bar_x = omega_bar
            omega_bar_y = omega_bar
        omegas_x = self.omegas[:, :, 0]  # (N, N)
        omegas_y = self.omegas[:, :, 1]  # (N, N)
        self.B_x = omegas_x * torch.sqrt(omegas_x / omega_bar_x)  # (N, N)
        self.B_y = omegas_y * torch.sqrt(omegas_y / omega_bar_y)  # (N, N)

        self.reset()

    def reset(self, initial_conditions=None):
        """
        initial_conditions : tensor of shape (batch, N, N, 3, 2) — [..., x/p]
            or None for random initialization.
        """
        N = self.N
        if initial_conditions is not None:
            self.xc = initial_conditions[..., 0].clone().to(self.device)  # (batch, N, N, 3)
            self.pc = initial_conditions[..., 1].clone().to(self.device)
        else:
            r = self.phase_space_range
            self.xc = (2 * r) * torch.rand((self.batch_size, N, N, 3), device=self.device) - r
            self.pc = (2 * r) * torch.rand((self.batch_size, N, N, 3), device=self.device) - r

        # Co-evolve covariances, initialized from thermal state
        self.Vxx = torch.full((self.batch_size, N, N, 3), (1.0 + self.n_thermal), device=self.device)
        self.Vpp = torch.full((self.batch_size, N, N, 3), (1.0 + self.n_thermal), device=self.device)
        self.Cxp = torch.zeros((self.batch_size, N, N, 3), device=self.device)

        self.t = 0
        return self.state()

    def state(self):
        """Flat state: (batch, 6N^2) — [xc.flatten, pc.flatten]"""
        return torch.cat([self.xc.flatten(1), self.pc.flatten(1)], dim=1)

    def n_bar(self):
        """
        Per-mode phonon number: (batch, N, N, 3).
        Returns per-site sum: (batch, N, N).
        """
        n_per_mode = (self.xc**2 + self.Vxx + self.pc**2 + self.Vpp) / 4.0 - 0.5
        return n_per_mode.sum(dim=3)  # sum over modes

    def mean_energy(self):
        """Mean energy: average of omega * (n_bar_mode + 1/2) over all sites and modes."""
        n_per_mode = (self.xc**2 + self.Vxx + self.pc**2 + self.Vpp) / 4.0 - 0.5
        energy_per_mode = self.omegas * (n_per_mode + 0.5)  # (batch, N, N, 3)
        return torch.mean(energy_per_mode, dim=(1, 2, 3))  # average over all sites and modes

    def step(self, raw_action):
        """
        raw_action : (batch, 4N) — [u_cd_x(N), u_cd_y(N), u_param_x_raw(N), u_param_y_raw(N)]
        """
        N = self.N
        Vxx, Vpp, Cxp = self.Vxx, self.Vpp, self.Cxp

        u_cd_x = raw_action[:, :N]                                           # (batch, N)
        u_cd_y = raw_action[:, N:2*N]                                        # (batch, N)
        u_param_x = self.modulation_depth * torch.tanh(raw_action[:, 2*N:3*N])  # (batch, N)
        u_param_y = self.modulation_depth * torch.tanh(raw_action[:, 3*N:])     # (batch, N)

        state_cost = self.mean_energy()
        control_cost = ((u_cd_x**2).sum(1) + (u_cd_y**2).sum(1)) / (4 * N)
        cost = state_cost + self.cost_u_weight * control_cost

        # Modified frequencies: (batch, N, N, 3)
        omega_mod = self.omegas * (1.0 + u_param_x[:, :, None, None]
                                       + u_param_y[:, None, :, None])

        # Independent noise per site per mode: (batch, N, N, 3)
        dW = torch.randn((self.batch_size, N, N, 3), device=self.device) * np.sqrt(self.dt)

        noise_x = (self.sqrt_2eta_gm * Vxx * dW).detach()
        noise_p = (self.sqrt_2eta_gm * Cxp * dW).detach()

        # Momentum update (all modes)
        new_pc = self.pc + (-omega_mod * self.xc * self.dt + noise_p)
        # x-mode: u_cd_x[A] acts on all (A,b) via B_x — row force
        new_pc[:, :, :, 0] = new_pc[:, :, :, 0] + self.B_x * u_cd_x[:, :, None] * self.dt
        # y-mode: u_cd_y[B] acts on all (a,B) via B_y — col force
        new_pc[:, :, :, 1] = new_pc[:, :, :, 1] + self.B_y * u_cd_y[:, None, :] * self.dt

        # Position update (all modes)
        new_xc = self.xc + (self.omegas * new_pc * self.dt + noise_x)

        self.pc = new_pc
        self.xc = new_xc

        # Covariance co-evolution (all modes, with omega_mod)
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

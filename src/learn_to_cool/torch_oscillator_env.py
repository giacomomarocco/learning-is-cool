import numpy as np
import torch


class TorchOscillatorEnv:
    """
    Batched Gaussian oscillator array environment in PyTorch.
    GPU-compatible version.
    """

    def __init__(self, oscillator, batch_size, dt=0.05,
                 phase_space_range=10.0, n_thermal=10,
                 mode='cold_damping', horizon=600,
                 modulation_depth=0.5, cost_u_weight=0.0,
                 omega_bar=None, device=None):
        """
        Parameters
        ----------
        device : torch.device or str, optional
            Device to run on. If None, auto-detects GPU.
        (all other parameters unchanged)
        """
        # --- Device setup ---
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        if hasattr(oscillator, 'omegas'):
            self.omegas = torch.tensor(oscillator.omegas, dtype=torch.float32, device=self.device)
            self.N = len(self.omegas)
        else:
            self.omegas = torch.tensor([oscillator.omega], dtype=torch.float32, device=self.device)
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

        if omega_bar is None:
            omega_bar = self.omegas[0].item()
        self.B = self.omegas * torch.sqrt(self.omegas / omega_bar)

        if self.mode == 'cold_damping':
            horizon_periods = horizon * dt / (2 * np.pi)
            if hasattr(oscillator, 'variance_solver'):
                res = oscillator.variance_solver(n_periods=horizon_periods, dt=dt)
                vxx, vpp, cxp = res['Vxx'], res['Vpp'], res['Cxp']

                if vxx.ndim == 1:
                    covs = np.stack([vxx, vpp, cxp], axis=-1)[:, np.newaxis, :]
                else:
                    covs = np.stack([vxx, vpp, cxp], axis=-1).transpose(1, 0, 2)

                # Move precomputed covariances to GPU
                self._Vxx_precomputed = torch.tensor(covs[:, :, 0], dtype=torch.float32, device=self.device)
                self._Vpp_precomputed = torch.tensor(covs[:, :, 1], dtype=torch.float32, device=self.device)
                self._Cxp_precomputed = torch.tensor(covs[:, :, 2], dtype=torch.float32, device=self.device)

        self.reset()

    def reset(self, initial_conditions=None):
        if initial_conditions is not None:
            self.xc = initial_conditions[:, :, 0].clone().to(self.device)
            self.pc = initial_conditions[:, :, 1].clone().to(self.device)
        else:
            r = self.phase_space_range
            # Use torch.rand on-device instead of .uniform_()
            self.xc = (2 * r) * torch.rand((self.batch_size, self.N), device=self.device) - r
            self.pc = (2 * r) * torch.rand((self.batch_size, self.N), device=self.device) - r

        self.x_true = self.xc.clone().detach()
        self.p_true = self.pc.clone().detach()

        if self.mode == 'parametric':
            self.Vxx = torch.full((self.batch_size, self.N), 1.0, device=self.device)
            self.Vpp = torch.full((self.batch_size, self.N), 1.0, device=self.device)
            self.Cxp = torch.zeros((self.batch_size, self.N), device=self.device)

        self.t = 0
        return self.state()

    def state(self):
        return torch.cat([self.xc, self.pc], dim=1)

    def _get_covariances(self):
        if self.mode == 'cold_damping':
            t_idx = min(self.t, len(self._Vxx_precomputed) - 1)
            return (self._Vxx_precomputed[t_idx].expand(self.batch_size, -1),
                    self._Vpp_precomputed[t_idx].expand(self.batch_size, -1),
                    self._Cxp_precomputed[t_idx].expand(self.batch_size, -1))
        else:
            return self.Vxx, self.Vpp, self.Cxp

    def n_bar(self):
        Vxx, Vpp, _ = self._get_covariances()
        return (self.xc**2 + Vxx + self.pc**2 + Vpp) / 4.0 - 0.5

    def step(self, raw_action):
        Vxx, Vpp, Cxp = self._get_covariances()

        raw_action = raw_action.view(self.batch_size)

        n_a = self.n_bar()
        state_cost = torch.mean(self.omegas * (self.xc**2 + self.pc**2), dim=1)

        # Generate noise on the correct device
        dW = torch.randn((self.batch_size, self.N), device=self.device) * np.sqrt(self.dt)

        if self.mode == 'cold_damping':
            u = raw_action
            cost = state_cost + self.cost_u_weight * u**2

            u_expanded = u.unsqueeze(1).expand(-1, self.N)

            dW = torch.randn((self.batch_size, self.N), device=self.device) * np.sqrt(self.dt)
            noise_x = (self.sqrt_2eta_gm * Vxx * dW).detach()
            noise_p = (self.sqrt_2eta_gm * Cxp * dW).detach()

            new_xc = self.xc + (self.omegas * self.pc * self.dt + noise_x)
            new_pc = self.pc + (-self.omegas * self.xc * self.dt
                                 + noise_p
                                 + self.B * u_expanded * self.dt)

            self.xc = new_xc
            self.pc = new_pc

        elif self.mode == 'parametric':
            u = self.modulation_depth * torch.tanh(raw_action)
            cost = state_cost + self.cost_u_weight * u**2

            u_expanded = u.unsqueeze(1).expand(-1, self.N)
            omega_mod = self.omegas * (1.0 + u_expanded)

            noise_x = (self.sqrt_2eta_gm * Vxx * dW).detach()
            noise_p = (self.sqrt_2eta_gm * Cxp * dW).detach()

            new_pc = self.pc + (-self.omegas * self.xc * self.dt
                     + noise_p
                     + self.B * u_expanded * self.dt)
            new_xc = self.xc + (self.omegas * self.pc * self.dt + noise_x)

            self.pc = new_pc
            self.xc = new_xc

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
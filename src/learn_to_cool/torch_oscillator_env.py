import numpy as np
import torch

try:
    from .gaussian_integrators import action_controls, advance_gaussian, covariance_health
except ImportError:  # Existing scripts also import this module directly.
    from gaussian_integrators import action_controls, advance_gaussian, covariance_health


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
                 omega_bar=None, device=None, integration_method="legacy",
                 intensity_dependent_recoil=False, dtype=torch.float32,
                 pregenerate_noise=True):
        if integration_method not in ("legacy", "euler_maruyama", "platen"):
            raise ValueError("integration_method must be legacy, euler_maruyama, or platen")
        if integration_method == "legacy" and intensity_dependent_recoil:
            raise ValueError("Intensity-dependent recoil requires a new integrator")
        if intensity_dependent_recoil and not 0 <= modulation_depth < 0.5:
            raise ValueError("Each modulation channel must be in [0, 0.5)")
        self.integration_method = integration_method
        self.intensity_dependent_recoil = intensity_dependent_recoil
        self.dtype = dtype
        self.pregenerate_noise = pregenerate_noise
        # --- Device setup ---
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # omegas: (N, N, 3)
        self.omegas = torch.tensor(oscillator.omegas, dtype=self.dtype, device=self.device)
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

        # Steady-state covariances (no feedback), shape (N, N, 3)
        omegas_np = oscillator.omegas.ravel()  # (3N^2,)
        xi = np.sqrt(1 + 4 * self.eta * self.gamma_meas**2 / omegas_np**2)
        if self.integration_method != "legacy":
            # Match the Riccati equations used by both new integration methods.
            xi = np.sqrt(1 + 16 * self.eta * self.gamma_meas**2 / omegas_np**2)
        Vx_ss = 2 / (np.sqrt(2 * self.eta) * np.sqrt(xi + 1))
        Vp_ss = 2 * xi / (np.sqrt(2 * self.eta) * np.sqrt(xi + 1))
        Cxp_ss = np.sqrt(xi - 1) / (np.sqrt(self.eta) * np.sqrt(xi + 1))
        N = self.N
        self.Vxx_ss = torch.tensor(Vx_ss.reshape(N, N, 3), dtype=self.dtype, device=self.device)
        self.Vpp_ss = torch.tensor(Vp_ss.reshape(N, N, 3), dtype=self.dtype, device=self.device)
        self.Cxp_ss = torch.tensor(Cxp_ss.reshape(N, N, 3), dtype=self.dtype, device=self.device)

        self.reset()

    def reset(self, initial_conditions=None, *, covariances=None, noise=None,
              generator=None):
        """
        initial_conditions : tensor of shape (batch, N, N, 3, 2) — [..., x/p]
            or None for random initialization.
        covariances : optional (Vxx, Vpp, Cxp), each (batch, N, N, 3).
        noise : optional Wiener increments (horizon, batch, N, N, 3), variance dt.
            Set pregenerate_noise=False and pass dW to step() to stream noise.
        generator : optional torch.Generator for initialization and internal noise.
        """
        N = self.N
        if initial_conditions is not None:
            self.xc = initial_conditions[..., 0].clone().to(device=self.device, dtype=self.dtype)  # (batch, N, N, 3)
            self.pc = initial_conditions[..., 1].clone().to(device=self.device, dtype=self.dtype)
        else:
            r = self.phase_space_range
            self.xc = (2 * r) * torch.rand((self.batch_size, N, N, 3), device=self.device,
                                       dtype=self.dtype, generator=generator) - r
            self.pc = (2 * r) * torch.rand((self.batch_size, N, N, 3), device=self.device,
                                       dtype=self.dtype, generator=generator) - r

        # Co-evolve covariances, initialized from measurement steady state
        self.Vxx = self.Vxx_ss.unsqueeze(0).expand(self.batch_size, -1, -1, -1).clone()
        self.Vpp = self.Vpp_ss.unsqueeze(0).expand(self.batch_size, -1, -1, -1).clone()
        self.Cxp = self.Cxp_ss.unsqueeze(0).expand(self.batch_size, -1, -1, -1).clone()

        if covariances is not None:
            shape = (self.batch_size, N, N, 3)
            if len(covariances) != 3 or any(v.shape != shape for v in covariances):
                raise ValueError(f"Covariances must contain three tensors of shape {shape}")
            self.Vxx, self.Vpp, self.Cxp = (
                v.clone().to(device=self.device, dtype=self.dtype) for v in covariances
            )

        shape = (self.horizon, self.batch_size, N, N, 3)
        if noise is not None:
            if noise.shape != shape:
                raise ValueError(f"Noise must have shape {shape}")
            self._noise = noise.to(device=self.device, dtype=self.dtype)
        elif self.pregenerate_noise:
            self._noise = torch.randn(shape, device=self.device, dtype=self.dtype,
                                      generator=generator) * getattr(self, "sqrt_dt", np.sqrt(self.dt))
        else:
            self._noise = None

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

    def gaussian_state(self):
        """Joint state in integrator order; tensors retain their autograd graphs."""
        return self.xc, self.pc, self.Vxx, self.Vpp, self.Cxp

    def covariance_health(self, tolerance=1e-10):
        return covariance_health(self.gaussian_state(), tolerance)

    def _integrated_step(self, raw_action, dW):
        """New methods only. Compile this or step(), not the legacy core."""
        N = self.N
        ux, uy = raw_action[:, :N], raw_action[:, N:2*N]
        force, modulation = action_controls(
            raw_action, self.B_x, self.B_y, self.modulation_depth)
        gamma = self.gamma_meas * (1 + modulation) if self.intensity_dependent_recoil else self.gamma_meas
        cost = self.mean_energy() + self.cost_u_weight * (
            ux.square().sum(1) + uy.square().sum(1)) / (4 * N)
        self.xc, self.pc, self.Vxx, self.Vpp, self.Cxp = advance_gaussian(
            self.gaussian_state(), self.omegas, force, modulation, gamma,
            self.eta, self.dt, dW, self.integration_method)
        return cost

    def step(self, raw_action, dW=None):
        """Advance with an optional Wiener increment of variance dt.

        The default legacy method retains momentum-first stepping, detached
        diffusion amplitudes and covariance clipping. The new methods use the
        old joint state in the predictor and never clip covariances.
        """
        if self.integration_method != "legacy":
            if self.t >= self.horizon:
                raise RuntimeError("step() called past horizon; call reset()")
            if dW is None:
                if self._noise is None:
                    raise ValueError("Supply dW when pregenerate_noise=False")
                dW = self._noise[self.t]
            if dW.shape != self.xc.shape:
                raise ValueError(f"dW must have shape {self.xc.shape}")
            cost = self._integrated_step(raw_action, dW)
            self.t += 1
            return cost
        if self._noise is None and dW is None:
            raise ValueError("Supply dW when pregenerate_noise=False")
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

        if dW is None:
            # Use pre-generated noise if available, else generate on the fly
            if self.t < self._noise.shape[0]:
                dW = self._noise[self.t]
            else:
                dW = torch.randn((self.batch_size, N, N, 3), device=self.device) * np.sqrt(self.dt)

        noise_x = (self.sqrt_2eta_gm * Vxx * dW).detach()
        noise_p = (self.sqrt_2eta_gm * Cxp * dW).detach()

        # Cold damping force as full (batch, N, N, 3) tensor — no in-place indexing
        cd_x = self.B_x * u_cd_x[:, :, None]       # (batch, N, N)
        cd_y = self.B_y * u_cd_y[:, None, :]        # (batch, N, N)
        cd_force = torch.stack([cd_x, cd_y, torch.zeros_like(cd_x)], dim=3)  # (batch, N, N, 3)

        # Momentum update (all modes in one shot)
        new_pc = self.pc + (-omega_mod * self.xc * self.dt + noise_p + cd_force * self.dt)

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

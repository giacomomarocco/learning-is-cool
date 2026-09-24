"""Differentiable array-feedback training, independent of the legacy trainer.

Quadratures have vacuum variance one. State costs sum all three directions
per particle; force costs sum physical squared cold forces per particle.
Parametric modulation changes recoil and covariances, including their gradients.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.linalg import solve_continuous_are
import torch
from torch import nn

from .gaussian_integrators import action_controls, advance_gaussian, covariance_health


State = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    base_iterations: int
    seed: int = 20260924
    g_fb: float = 1.0
    dt: float = 0.000625
    control_dt: float = 0.01
    dtype: str = "float32"
    learning_rate: float = 0.0005
    smoke: bool = False
    # Persist the physical and architecture conventions with every checkpoint.
    grid_size: int = 5
    delta: float = 0.1
    radial_ratios: Tuple[float, float] = (4.5, 4.1)
    gamma0: float = 0.05
    eta: float = 0.5
    initial_occupation: float = 5.0
    modulation_depth: float = 0.05
    hidden_width: int = 128

    def __post_init__(self):
        for name in ("batch_size", "base_iterations", "grid_size", "hidden_width"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("g_fb", "dt", "control_dt", "learning_rate", "delta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if not 0 < self.eta <= 1 or not math.isfinite(self.gamma0) or self.gamma0 < 0:
            raise ValueError("Require 0 < eta <= 1 and finite gamma0 >= 0")
        if not math.isfinite(self.initial_occupation) or self.initial_occupation < 0:
            raise ValueError("initial_occupation must be finite and nonnegative")
        if not 0 <= self.modulation_depth < 0.5:
            raise ValueError("modulation_depth must be in [0, 0.5)")
        if len(self.radial_ratios) != 2 or any(
                not math.isfinite(r) or r <= 0 for r in self.radial_ratios):
            raise ValueError("Require two finite positive radial frequency ratios")
        integer_steps(self.control_dt, self.dt)
        for duration, _ in self.schedule:
            integer_steps(duration, self.control_dt)

    @property
    def substeps(self):
        return integer_steps(self.control_dt, self.dt)

    @property
    def schedule(self):
        if self.smoke:
            return ((2 * self.control_dt, 2),)
        k = self.base_iterations
        return ((5.0, 20 * k), (20.0, 5 * k), (100.0, k))

    @property
    def torch_dtype(self):
        return getattr(torch, self.dtype)


def integer_steps(duration, step):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    count = round(duration / step)
    if count < 1 or not math.isclose(count * step, duration, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"duration {duration} must be a positive integer multiple of {step}")
    return count


def lqr_matrices(omega, coupling, particle_count, g_fb):
    """CARE matrices for one shared cold actuator, using global cost units.

    Interleaved group state: x1,p1,x2,p2,... . In hbar=m=1 units,
    F_j = sqrt(omega_j/2) * coupling_j * u. Thus R includes every
    particle's squared physical force, not merely the shared command squared.
    """
    omega, coupling = np.asarray(omega), np.asarray(coupling)
    n = len(omega)
    a = np.zeros((2 * n, 2 * n))
    i = 2 * np.arange(n)
    a[i, i + 1], a[i + 1, i] = omega, -omega
    b = np.zeros((2 * n, 1))
    b[1::2, 0] = coupling
    q = np.diag(np.repeat(omega / (4 * particle_count), 2))
    r = np.array([[np.sum(omega * coupling**2) / (2 * particle_count * g_fb**2)]])
    return a, b, q, r


class ArrayModel(nn.Module):
    """Fixed physics and LQR buffers; no trainable parameters."""

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.n = n = config.grid_size
        self.particles = n * n
        a, b = np.indices((n, n))
        wz = 1 + ((a + b) % n) * config.delta
        omega = np.stack((config.radial_ratios[0] * wz,
                          config.radial_ratios[1] * wz, wz), axis=-1)
        bx = omega[..., 0] * np.sqrt(omega[..., 0] / omega[0, 0, 0])
        by = omega[..., 1] * np.sqrt(omega[..., 1] / omega[0, 0, 1])

        def gain(w, coupling):
            a, b, q, r = lqr_matrices(w, coupling, self.particles, config.g_fb)
            p = solve_continuous_are(a, b, q, r)
            return np.linalg.solve(r, b.T @ p).reshape(n, 2)

        kx = np.stack([gain(omega[row, :, 0], bx[row]) for row in range(n)])
        ky = np.stack([gain(omega[:, col, 1], by[:, col]) for col in range(n)])
        for name, value in (("omega", omega), ("bx", bx), ("by", by),
                            ("kx", kx), ("ky", ky)):
            self.register_buffer(name, torch.tensor(value, dtype=config.torch_dtype))

    def initial_state(self, batch_size):
        zero = self.omega.new_zeros((batch_size, self.n, self.n, 3))
        variance = 2 * self.config.initial_occupation + 1
        return zero, zero.clone(), zero + variance, zero + variance, zero.clone()

    def lqr_commands(self, state: State):
        x, p = state[:2]
        ux = -(x[..., 0] * self.kx[..., 0] + p[..., 0] * self.kx[..., 1]).sum(-1)
        # ky is stored (column, row, quadrature).
        uy = -(x[..., 1].transpose(1, 2) * self.ky[..., 0]
               + p[..., 1].transpose(1, 2) * self.ky[..., 1]).sum(-1)
        return torch.cat((ux, uy, torch.zeros_like(ux), torch.zeros_like(uy)), dim=1)

    def centroid_cost(self, state: State):
        return (self.omega * (state[0].square() + state[1].square())).sum((1, 2, 3)) / (4 * self.particles)

    def force_cost(self, force):
        return (self.omega * force.square()).sum((1, 2, 3)) / (2 * self.particles * self.config.g_fb**2)

    def occupations(self, state: State):
        x, p, vx, vp, _ = state
        return (x.square() + p.square() + vx + vp) / 4 - 0.5


class ResidualPolicy(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.variance_scale = 2 * config.initial_occupation + 1
        self.mean_scale = math.sqrt(self.variance_scale)
        self.net = nn.Sequential(
            nn.Linear(15 * config.grid_size**2, config.hidden_width), nn.Tanh(),
            nn.Linear(config.hidden_width, config.hidden_width), nn.Tanh(),
            nn.Linear(config.hidden_width, 4 * config.grid_size),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: State):
        x, p, vx, vp, c = state
        observation = torch.cat((x.flatten(1) / self.mean_scale,
                                 p.flatten(1) / self.mean_scale,
                                 vx.flatten(1) / self.variance_scale,
                                 vp.flatten(1) / self.variance_scale,
                                 c.flatten(1) / self.variance_scale), dim=1)
        return self.net(observation)


def merge_health(previous, current):
    if previous is None:
        return current
    return {key: (torch.minimum(previous[key], value) if key.startswith("min_")
                  else previous[key] + value) for key, value in current.items()}


def require_healthy(health):
    counts = torch.stack([health[key] for key in
                          ("nonfinite_modes", "nonpositive_diagonals", "uncertainty_violations")])
    if bool((counts != 0).any()):
        details = {key: value.item() for key, value in health.items()}
        raise FloatingPointError(f"Invalid Gaussian evolution (no clipping): {details}")


class ControllerInterval(nn.Module):
    """Compilation boundary: explicit tensor state/noise, no RNG or counters.

    Health summaries inspect EVERY integration step and are detached solely
    for diagnostics. The five evolving state tensors retain full gradients.
    """

    def __init__(self, model: ArrayModel, policy: Optional[ResidualPolicy]):
        super().__init__()
        self.model, self.policy = model, policy
        self.substeps = model.config.substeps

    def forward(self, state: State, noise):
        model, cfg = self.model, self.model.config
        raw = model.lqr_commands(state)
        if self.policy is not None:
            raw = raw + self.policy(state)
        force, modulation = action_controls(raw, model.bx, model.by, cfg.modulation_depth)
        gamma = cfg.gamma0 * (1 + modulation)
        control = model.force_cost(force) * cfg.control_dt
        energy = model.centroid_cost(state)
        integral = torch.zeros_like(energy)
        health = None
        for index in range(self.substeps):
            state = advance_gaussian(state, model.omega, force, modulation, gamma,
                                     cfg.eta, cfg.dt, noise[index], "platen")
            next_energy = model.centroid_cost(state)
            integral = integral + 0.5 * cfg.dt * (energy + next_energy)
            energy = next_energy
            health = merge_health(health, covariance_health(
                tuple(v.detach() for v in state),
                tolerance=1e-5 if state[0].dtype == torch.float32 else 1e-10))
        return state, torch.stack((integral, control), dim=1), health


@dataclass
class RolloutResult:
    costs: torch.Tensor  # (batch, 2): time-averaged centroid and force costs
    state: State
    health: dict

    @property
    def loss(self):
        return self.costs.sum(1).mean()


def rollout(model, interval, duration, batch_size, *, generator=None, noise=None,
            progress: Optional[Callable] = None):
    """Full BPTT; independent innovations per mode, shared by its x/p equations.

    Noise is drawn one controller interval at a time outside compilation.
    Autograd may retain it until backward; this is not constant-memory BPTT.
    Explicit noise is supported for small matched-path validation runs.
    """
    cfg = model.config
    intervals = integer_steps(duration, cfg.control_dt)
    shape = (cfg.substeps, batch_size, model.n, model.n, 3)
    if noise is not None and noise.shape != (intervals, *shape):
        raise ValueError(f"noise must have shape {(intervals, *shape)}")
    state = model.initial_state(batch_size)
    costs = model.omega.new_zeros((batch_size, 2))
    health = None
    for index in range(intervals):
        dw = noise[index] if noise is not None else torch.randn(
            shape, device=model.omega.device, dtype=model.omega.dtype,
            generator=generator) * math.sqrt(cfg.dt)
        state, contribution, current_health = interval(state, dw)
        costs = costs + contribution
        health = merge_health(health, current_health)
        if (index + 1) % 100 == 0 or index + 1 == intervals:
            require_healthy(health)
            if progress is not None:
                progress(index + 1, intervals)
    return RolloutResult(costs / duration, state, health)


class TrainingSession:
    """Single-process eager/compiled trainer. Microbatching is intentionally deferred."""

    def __init__(self, config: TrainingConfig, device="cpu", compiled=False):
        self.config, self.device = config, torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Supported training devices are cpu and cuda")
        self.model = ArrayModel(config).to(self.device)
        # Network initialization is isolated from caller RNG state and identical
        # for eager/compiled execution, regardless of target device.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.policy = ResidualPolicy(config).to(dtype=config.torch_dtype)
        self.policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed + 1)
        self.interval = ControllerInterval(self.model, self.policy)
        if compiled:
            self.interval = torch.compile(self.interval, fullgraph=True)
        self.stage, self.stage_iteration, self.global_step = 0, 0, 0

    @property
    def complete(self):
        return self.stage == len(self.config.schedule)

    def step(self, progress=None):
        if self.complete:
            raise RuntimeError("The curriculum is complete")
        duration, count = self.config.schedule[self.stage]
        self.optimizer.zero_grad(set_to_none=True)
        result = rollout(self.model, self.interval, duration, self.config.batch_size,
                         generator=self.generator, progress=progress)
        loss = result.loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        occupations = self.model.occupations(tuple(v.detach() for v in result.state))
        costs = result.costs.detach().mean(0).cpu().tolist()
        self.global_step += 1
        self.stage_iteration += 1
        record = {
            "global_step": self.global_step, "stage": self.stage,
            "stage_iteration": self.stage_iteration, "stage_iterations": count,
            "duration": duration, "centroid_cost": costs[0], "force_cost": costs[1],
            "objective": sum(costs), "gradient_norm_before_clipping": norm.item(),
            "terminal_n_xyz": occupations.mean((0, 1, 2)).cpu().tolist(),
            "health": {key: value.item() for key, value in result.health.items()},
        }
        if self.stage_iteration == count:
            self.stage += 1
            self.stage_iteration = 0
        return record

    def save(self, path):
        """Atomically replace this run's checkpoint; never write legacy weights."""
        path = Path(path)
        payload = {
            "version": CHECKPOINT_VERSION, "config": asdict(self.config),
            "torch_version": str(torch.__version__), "device_type": self.device.type,
            "model": self.model.state_dict(), "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(), "stage": self.stage,
            "stage_iteration": self.stage_iteration, "global_step": self.global_step,
            "noise_rng": self.generator.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
        }
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def resume(cls, path, device="cpu", compiled=False):
        payload = read_checkpoint(path)
        if torch.device(device).type != payload["device_type"]:
            raise ValueError("Training resume requires the same device type for noise replay; evaluation may change it")
        session = cls(TrainingConfig(**payload["config"]), device, compiled)
        session.model.load_state_dict(payload["model"])
        session.policy.load_state_dict(payload["policy"])
        session.optimizer.load_state_dict(payload["optimizer"])
        session.stage = payload["stage"]
        session.stage_iteration = payload["stage_iteration"]
        session.global_step = payload["global_step"]
        counts = [count for _, count in session.config.schedule]
        if (not 0 <= session.stage <= len(counts)
                or session.stage_iteration < 0
                or (session.stage == len(counts) and session.stage_iteration != 0)
                or (session.stage < len(counts) and session.stage_iteration >= counts[session.stage])
                or session.global_step != sum(counts[:session.stage]) + session.stage_iteration):
            raise ValueError("Invalid checkpoint curriculum position")
        session.generator.set_state(payload["noise_rng"])
        torch.set_rng_state(payload["torch_rng"])
        if session.device.type == "cuda":
            torch.cuda.set_rng_state(payload["cuda_rng"], session.device)
        return session


def read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported training checkpoint version")
    return payload

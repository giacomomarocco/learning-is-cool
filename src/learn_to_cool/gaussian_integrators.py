"""Differentiable Ito steps for conditional Gaussian oscillator states.

State order is (xc, pc, Vxx, Vpp, Cxp); each tensor is (batch, N, N, 3).
Controls and measurement strength must be held fixed for the entire step.
The diffusion acts only on means and depends only on covariances, which have
finite variation. Consequently all diffusion-direction derivatives vanish.
For this system the simplified weak-order-two Platen scheme reduces to an
Euler predictor and trapezoidal drift/diffusion corrector using the SAME dW.
This is not a general multiplicative-noise stochastic Heun implementation.
No covariance clipping or diffusion detaching is performed.
"""

import torch


METHODS = ("euler_maruyama", "platen")


def action_controls(raw_action, B_x, B_y, modulation_depth):
    """Map row/column actions to force and combined trap modulation.

    Each channel is bounded by modulation_depth; the combined bound is twice
    that value. Set modulation_depth=0.05 for combined modulation +/-0.1.
    """
    n = B_x.shape[0]
    ux, uy = raw_action[:, :n], raw_action[:, n:2*n]
    row = modulation_depth * torch.tanh(raw_action[:, 2*n:3*n])
    col = modulation_depth * torch.tanh(raw_action[:, 3*n:4*n])
    modulation = row[:, :, None, None] + col[:, None, :, None]
    fx = B_x * ux[:, :, None]
    fy = B_y * uy[:, None, :]
    return torch.stack((fx, fy, torch.zeros_like(fx)), dim=-1), modulation


def drift(state, omegas, force, modulation, gamma, eta):
    """Joint mean/Riccati drift, with gamma the recoil heating rate."""
    x, p, vx, vp, c = state
    omega_mod = omegas * (1 + modulation)
    measurement = 4 * eta * gamma
    return (
        omegas * p,
        -omega_mod * x + force,
        2 * omegas * c - measurement * vx.square(),
        -2 * omega_mod * c + 4 * gamma - measurement * c.square(),
        omegas * vp - omega_mod * vx - measurement * vx * c,
    )


def diffusion(state, gamma, eta):
    """Amplitudes for one independent Wiener process per oscillator mode."""
    amplitude = 2 * torch.sqrt(torch.as_tensor(gamma, dtype=state[0].dtype,
                                             device=state[0].device) * eta)
    return amplitude * state[2], amplitude * state[4]


def advance_gaussian(state, omegas, force, modulation, gamma, eta, dt, dW,
                     method="platen"):
    """Advance all five components together using supplied Wiener increments.

    dW has variance dt (not unit variance). ``gamma`` already includes any
    intensity dependence. Its autograd graph, like those of force/modulation,
    is retained, but its value is not recomputed at the predictor state.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown integration method {method!r}; choose {METHODS}")
    a = drift(state, omegas, force, modulation, gamma, eta)
    b = diffusion(state, gamma, eta)
    predicted = tuple(y + dt * ai + (b[i] * dW if i < 2 else 0)
                      for i, (y, ai) in enumerate(zip(state, a)))
    if method == "euler_maruyama":
        return predicted
    a_predicted = drift(predicted, omegas, force, modulation, gamma, eta)
    b_predicted = diffusion(predicted, gamma, eta)
    return tuple(y + 0.5 * dt * (ai + ap)
                 + (0.5 * (b[i] + b_predicted[i]) * dW if i < 2 else 0)
                 for i, (y, ai, ap) in enumerate(zip(state, a, a_predicted)))


def covariance_health(state, tolerance=1e-10):
    """Unmasked diagnostics: positive diagonals and Vxx*Vpp-Cxp**2 >= 1.

    Counts are per mode. Nonfinite states are reported separately; callers
    should retain both counts rather than interpreting NaN comparisons as OK.
    """
    _, _, vx, vp, c = state
    determinant = vx * vp - c.square()
    finite = torch.stack([torch.isfinite(y) for y in state]).all(0)
    return {
        "nonfinite_modes": (~finite).sum(),
        "nonpositive_diagonals": ((vx <= 0) | (vp <= 0)).sum(),
        "uncertainty_violations": (determinant < 1 - tolerance).sum(),
        "min_determinant": determinant.amin(),
        "min_variance": torch.minimum(vx, vp).amin(),
    }

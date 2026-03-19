import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from learn_to_cool.gaussian_oscillator import GaussianOscillator
from learn_to_cool.utils.plotting import plot_nbar_trajectories, plot_cooling_comparison

def run_test():
    print("Comparing Standard Parametric (PLL) and RL Parametric Policy...")

    # Parameters from user
    modulation_depth = 0.5
    eta = 0.2
    gamma_BA = 18.8 / 104
    n_th = 10

    # Simulation settings
    n_periods = 100
    dt = 0.05
    n_trajectories = 50 # Number of trajectories to average

    # Initialize oscillator
    osc = GaussianOscillator(omega=1.0, quality_factor=1e6, gamma_meas=gamma_BA, eta=eta, n_thermal=n_th)

    # 1. Load RL Policy
    policy = nn.Sequential(
        nn.Linear(2, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1)
    )
    weights_path = "weights/parametric_2d.pth"
    if os.path.exists(weights_path):
        policy.load_state_dict(torch.load(weights_path))
        policy.eval()
        print(f"Loaded RL policy from {weights_path}")
    else:
        print(f"Warning: Weights not found at {weights_path}. Using untrained policy.")

    def rl_parametric_fn(x, p, t):
        state = torch.FloatTensor([[x, p]])
        with torch.no_grad():
            raw_u = policy(state).item()
            u = modulation_depth * np.tanh(raw_u)
        # RL policy was trained with omega^2(t) = omega0^2 * (1+u)
        return (1.0 + u)

    # 2. Standard Parametric (PLL) as baseline
    # No extra function needed, just use built-in PLL in expectation_solver

    print(f"Simulating {n_trajectories} trajectories for each method...")

    all_n_bars_pll = []
    all_n_bars_rl = []

    # Initial conditions: thermal state
    initial_conds = np.random.normal(0, np.sqrt(n_th/2), (n_trajectories, 2))

    for i in range(n_trajectories):
        if i % 10 == 0:
            print(f"  Trajectory {i}/{n_trajectories}")

        # Standard Parametric (PLL) simulation
        res_pll = osc.expectation_solver(
            epsilon=modulation_depth,
            n_periods=n_periods,
            dt=dt,
            initial_conditions=np.append(initial_conds[i], 0),
            parametric=True,
            parametric_fn=None
        )
        n_bar_pll = (res_pll['xc']**2 + res_pll['Vxx'] + res_pll['pc']**2 + res_pll['Vpp']) / 4 - 0.5
        all_n_bars_pll.append(n_bar_pll)

        # RL Parametric simulation
        res_rl = osc.expectation_solver(
            parametric_fn=rl_parametric_fn,
            n_periods=n_periods,
            dt=dt,
            initial_conditions=np.append(initial_conds[i], 0),
            parametric=True
        )
        n_bar_rl = (res_rl['xc']**2 + res_rl['Vxx'] + res_rl['pc']**2 + res_rl['Vpp']) / 4 - 0.5
        all_n_bars_rl.append(n_bar_rl)

    all_n_bars_pll = np.array(all_n_bars_pll)
    all_n_bars_rl = np.array(all_n_bars_rl)

    n_bar_avg_pll = np.mean(all_n_bars_pll, axis=0)
    n_bar_avg_rl = np.mean(all_n_bars_rl, axis=0)

    # Theoretical minimum for LQG cold damping: n_min = (1/sqrt(eta) - 1)/2
    n_min = (1.0/np.sqrt(eta) - 1.0) / 2.0 + gamma_BA/2

    dirname = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(dirname, 'figures'), exist_ok=True)

    # Plot results
    print("Generating plots...")

    plot_nbar_trajectories(
        times, all_n_bars_rl, n_bar_avg_rl, n_min,
        f"RL Parametric Cooling (mod_depth={modulation_depth})",
        os.path.join(dirname, "figures/rl_parametric_trajectories.png")
    )

    plot_cooling_comparison(
        times, [n_bar_avg_pll, n_bar_avg_rl],
        ["Standard Parametric (PLL)", "RL Parametric Feedback"],
        n_min, "Cooling Method Comparison",
        os.path.join(dirname, "figures/cooling_method_comparison.png")
    )

    print(f"Final n_bar (PLL): {n_bar_avg_pll[-1]:.4f}")
    print(f"Final n_bar (RL):  {n_bar_avg_rl[-1]:.4f}")
    print(f"Theoretical n_min: {n_min:.4f}")
    print("Results saved in figures/ directory.")

if __name__ == "__main__":
    run_test()

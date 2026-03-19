import numpy as np
import matplotlib.pyplot as plt
import os

def plot_nbar_trajectories(times, n_bars, n_bar_avg, n_min, title, save_path=None):
    """
    Plot individual n_bar trajectories with an average overlay.
    """
    plt.figure(figsize=(10, 6))
    for i in range(min(n_bars.shape[0], 50)):  # Plot up to 50 individual trajectories
        plt.plot(times, n_bars[i, :], alpha=0.1, color='blue')

    plt.plot(times, n_bar_avg, label='Average', color='red', linewidth=2)
    plt.axhline(y=n_min, color='black', linestyle='--', label=rf'LQG Minimum ($n_{{min}}$={n_min:.2f})')

    plt.xlabel('Time (s)')
    plt.ylabel(r'Phonon Number ($n_{bar}$)')
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def plot_cooling_comparison(times, n_bar_avgs, labels, n_min, title, save_path=None):
    """
    Compare multiple n_bar_avg trajectories.
    """
    plt.figure(figsize=(10, 6))
    for i, n_bar_avg in enumerate(n_bar_avgs):
        plt.plot(times, n_bar_avg, label=labels[i])

    plt.axhline(y=n_min, color='black', linestyle='--', label=rf'LQG Minimum ($n_{{min}}$={n_min:.2f})')

    plt.xlabel('Time (s)')
    plt.ylabel(r'Average Phonon Number ($\bar{n}$)')
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def plot_gain_performance(gains, n_steady_results, analytic_fn, n_min, title, save_path=None):
    """
    Plot steady-state n_bar vs feedback gain, including analytic theory.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(gains, n_steady_results, label='Simulation', color='blue')

    g_fine = np.linspace(min(gains), max(gains), 100)
    plt.plot(g_fine, [analytic_fn(g) for g in g_fine], label='Analytic Theory', color='red')

    plt.axhline(y=n_min, color='black', linestyle='--', label=f'Quantum Limit ($n_{{min}}$={n_min:.2f})')

    plt.xlabel('Feedback Gain ($g_{fb}$)')
    plt.ylabel('Steady-state Phonon Number ($n_{bar}$)')
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

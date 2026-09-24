# Repository guide

## Purpose

This is a scientific Python project for precision simulations of quantum
harmonic oscillators and feedback cooling of neutral nanoparticle arrays under
continuous measurement. It compares analytic LQR/LQG feedback with learned
policies, including cold damping and parametric trap modulation.

Keep units, quadrature normalization, feedback signs, coordinate ordering, and
variable names consistent across the NumPy simulations and PyTorch training
environment. Read the relevant equations and implementation before changing
the dynamics.

## Where things live

| Path | Purpose |
| --- | --- |
| `src/learn_to_cool/` | Reusable oscillator models, feedback controllers, training environment, and utilities. |
| `scripts/` | Executable training, comparison, plotting, validation, and benchmarking experiments. Many run substantial work at module scope. |
| `weights/` | Saved PyTorch policies and checkpoints. Match the model architecture, grid size, and physical parameters when loading them. |
| `data/` | Experiment and benchmark outputs, including CSV tables and NumPy archives. |
| `figures/`, `scripts/figures/` | Generated plots; the destination varies by script. |
| `notes/` | Physics derivations, design documentation, and implementation/results notes. This directory is gitignored and may be absent in a fresh checkout. |
| `mma_scripts/` | Mathematica notebooks for symbolic work, including Lyapunov calculations. |
| `scaling.nb` | Root-level Mathematica notebook for scaling studies. |
| `old_stuff/` | Historical simulations and training experiments; useful context, but not the current API reference. |
| `numerical_approximation_to_RL.py` | Standalone exploration of a feedback cost landscape; contains a machine-specific import path. |
| `tex_to_md.py` | Converts a supplied TeX file to Markdown in `notes/`. |
| `pyproject.toml`, `uv.lock`, `.python-version` | Package/dependency configuration, locked environment, and Python selection. |
| `CLAUDE.md` | Companion agent instructions with project conventions and `uv` usage. |

Some research scripts and outputs may exist only in the local working tree.
Check their presence and Git status before relying on them. `README.md` is
currently empty; this guide and, when available, `notes/DESIGN.md` provide the
project overview.

### Core modules

All paths below are relative to `src/learn_to_cool/`.

| Module | Responsibility |
| --- | --- |
| `gaussian_oscillator.py` | `GaussianOscillator`: single-mode covariance evolution, stochastic conditional means, feedback simulations, occupation, and power spectra. |
| `gaussian_oscillator_array.py` | `GaussianOscillatorArray`: NumPy simulation of an N-by-N array with three modes per site. Includes covariance integration, stochastic trajectories, and `make_parametric_z_fn` for shared per-row z-mode feedback. |
| `optimal_control.py` | `FeedbackForces`: single-oscillator momentum damping, optimal feedback, and no-feedback laws. |
| `optimal_feedback_n_oscillators.py` | `OptimalFeedbackNOscillators`: common-control LQR gains and analytic steady-state covariances/occupations. `GridFeedbackLQR` combines row and column controllers. |
| `torch_oscillator_env.py` | `TorchOscillatorEnv`: batched differentiable array simulation for policy training, evolving means and covariances with pre-generated noise. |
| `cli_utils.py` | `parse_overrides`: parses `key=value` arguments into integers, floats, strings, comma-separated numeric lists, or JSON lists. Each script chooses which keys to use. |
| `array_training.py` | Functional Platen rollouts, physical-force-cost LQR, covariance-aware residual policy, and resumable array training. See `docs/array_training.md`. |
| `utils/plotting.py` | Shared occupation-trajectory, cooling-comparison, and gain-performance plots. |

The extensionless `src/learn_to_cool/parametric_feedback` is a separate
experimental Python source file, not a standard importable module.

### Script entry points

| Script(s) in `scripts/` | Purpose |
| --- | --- |
| `n_particle_combined_RL.py` | Main N-by-N array trainer combining cold damping and parametric feedback. Saves `weights/n_particle_combined_rl.pth`. |
| `train_array_feedback.py` | New 5x5 residual-LQR trainer with centroid/physical-force costs, a fixed-step horizon curriculum, explicit remote benchmark/evaluation modes, and tiny CPU smoke mode. Uses argparse. |
| `validate_array_training.py` | Tiny CPU/eager assertions for physical costs, LQR, feedback gradients, curriculum, CLI safety, and checkpoint replay. No compilation or long trajectories. |
| `diagnose_array_batch.py` | Remote fixed-policy gradient variance, direction agreement, timing and GPU memory by batch/horizon; no weight updates. `--smoke` is tiny CPU/eager only. |
| `combined_cooling_RL.py`, `parametric_RL.py` | Earlier combined-feedback and parametric-only training experiments. |
| `single_particle_RL.py`, `two_particle_RL.py` | Earlier small-system policy-learning experiments. |
| `compare_lqr_rl_cooling.py` | Compares analytic grid LQR feedback against a saved RL policy. |
| `plot_cooling_trajectories.py`, `plot_rl_parametric_contour.py` | Plots cooling trajectories and learned phase-space feedback. |
| `n_oscillator_test.py`, `feedback_testing.py` | Exploratory analytic-feedback simulations. |
| `test_steady_state_covariances.py` | Compares Torch covariance stationarity and thermal relaxation against analytic values and the NumPy solver; prints errors and writes a figure. |
| `test_parametric.py`, `test_parametric_utils.py` | Parametric-feedback diagnostics and PLL/RL comparisons. |
| `benchmark_compile.py`, `benchmark_forward_scaling.py` | Compilation and forward-simulation timing. |
| `benchmark_training.py`, `benchmark_training_cpu.py`, `benchmark_gpu.py` | Training scalability experiments with different execution strategies. |
| `benchmark_parametric_z.py` | Sweeps parametric gain and records axial/radial occupations, data, and plots. |
| `compare_integrators.py` | Paired-noise Platen/Euler–Maruyama comparison on a cyclic 75-mode model, with reference halving, covariance-health checks, and timing/memory reports. Uses argparse; smoke is the default, extended is opt-in. |
| `validate_integrators.py` | Focused assertions for deterministic/covariance convergence, fixed-noise gradients, reproducibility, and environment dispatch. |
| `integrator_shards.py` | Distributes extended-profile cases across ranks inside an interactive Slurm allocation, streams progress, and validates/merges completed results. |
| `run_integrator_interactive.sh` | Launches the extended CPU sweep from a persistent login-node tmux session, obtains an interactive allocation, merges results on the compute node, and releases the allocation. |

Older scripts may target earlier environment interfaces. Inspect their current
imports, constructor calls, parameter blocks, and checkpoint requirements before
running them; a filename or docstring alone does not establish compatibility.

## Physics and array conventions

- Dimensionless quadratures use coherent-state variance one. Per-mode phonon
  occupation is `(xc**2 + Vxx + pc**2 + Vpp) / 4 - 0.5`.
- Array frequencies have shape `(N, N, 3)` with mode order `[x, y, z]`.
  Torch means and covariances have shape `(batch, N, N, 3)`.
- The policy state is `(batch, 6*N*N)`, ordered as all flattened positions
  followed by all flattened momenta. Actions have shape `(batch, 4*N)`:
  row x cold damping, column y cold damping, row parametric signals, then
  column parametric signals.
- z modes have no direct cold damping. Parametric signals affect all three
  modes of the corresponding row/column.
- `TorchOscillatorEnv.n_bar()` sums the three modes per site;
  `mean_energy()` averages frequency-weighted energy across sites and modes.
  Preserve this distinction when defining losses and comparing plots.
- The new `array_training` policy observes all five Gaussian state components
  (375 values for 5x5). Its cost sums centroid energies over directions per
  particle, excludes covariance energy, and penalizes physical cold forces as
  `sum(omega*f_p**2)/(2*N*N*g_fb**2)`. It does not use the legacy energy/command
  cost. Its LQR gains use exactly the same actuator coefficients and cost.
- Frequency ordering varies by experiment: some combined-feedback studies use
  stiff z modes; axial parametric-cooling studies use soft z modes. Read the
  script's frequency construction rather than imposing one ordering globally.
- For the array drift `dp = -omega*(1 + epsilon)*x*dt`, the implemented
  z-feedback law uses `epsilon = clip(+g*sum(x_z*p_z), ...)`. Some notes and
  script docstrings retain a conflicting minus sign; check the energy drift
  before copying a formula.
- Time is dimensionless unless explicitly converted for a plot. State any
  reference frequency used for SI units.
- `TorchOscillatorEnv` defaults to the historical `legacy` momentum-first
  update. Opt-in `euler_maruyama` and `platen` methods share the equations in
  `gaussian_integrators.py`, retain diffusion gradients, and never clip
  covariance failures. Use `modulation_depth=0.05` for combined row/column
  modulation bounded by 0.1, and explicitly enable intensity-dependent recoil
  when required by the experiment.

## Python tooling and execution

Use `uv` exclusively for Python environments, dependencies, scripts, and tools;
do not use pip, Poetry, or Conda. `.python-version` selects Python 3.12, while
`pyproject.toml` currently declares Python >=3.9.

```sh
uv sync
uv run scripts/test_parametric.py
uv run scripts/n_particle_combined_RL.py
uv run scripts/n_particle_combined_RL.py batch_size=64 n_iterations=2 horizon=20
uv add <package>
uv remove <package>
```

Run commands from the repository root. The reduced training example still
compiles, saves weights, and runs subsequent analysis; it is not a side-effect-free
test. Inspect output paths before launching it to preserve existing policies.
Defaults can involve long trajectories, large batches, or compilation overhead.
The main environment selects CUDA when available and otherwise CPU.

CLI overrides use `key=value`, not conventional `--key value` flags, in scripts
that call `parse_overrides`. Only explicitly consumed keys take effect; for
example, the main trainer derives grid size from `omegas` rather than consuming
an `N_grid` override. Other scripts may have their own argument parser.

For PEP 723 scripts, use `uv add --script <script>.py <package>` or
`uv remove --script <script>.py <package>`, or edit the inline dependency block.
Run installed tools as `uv run <tool>`.

Imports currently mix package imports with direct module imports enabled by
adding `src/learn_to_cool` to `sys.path`. Follow the relevant entry point's
existing convention and verify imports when moving code.

## Validation and documentation

- There is no configured pytest suite or lint command in `pyproject.toml`.
  The `test_*.py` files are simulation/plotting diagnostics, not a uniform
  assertion-based test suite. Do not assume pytest collection is harmless.
- Choose validation for the changed behavior. For dynamics changes, compare
  analytic steady states, no-feedback evolution, and NumPy/Torch results with
  matching parameters; inspect numerical errors rather than only exit status.
- Use small runs first, set NumPy/Torch seeds for reproducibility when relevant,
  and use `MPLBACKEND=Agg uv run scripts/<name>.py` for headless plots.
- For the new array trainer, run only `validate_array_training.py` and
  `train_array_feedback.py --smoke` and `diagnose_array_batch.py --smoke` locally unless instructed otherwise.
  Training, compilation benchmarks, and scaling runs belong on remote compute
  servers; see `docs/array_training.md`. Normal runs require explicit batch,
  iteration budget, and output; never substitute old trainers as smoke tests.
- Check checkpoint availability: some diagnostics fall back to an untrained
  policy, which is not evidence of trained-policy performance.
- Keep research notes in `notes/`; major tasks get their own Markdown file
  describing implementation and results. Maintain `notes/DESIGN.md` when
  changing layout or module/script responsibilities, and keep this guide current.
- Useful local references include `notes/stochastic_evolution.md`,
  `notes/multiparticle_feedback.md`, `notes/parametric_z_cooling.md`,
  `notes/rl_cooling_trajectories.md`, and `notes/cli_overrides.md`. Reconcile
  stale notes against the actual equations and code.

## Git workflow

- Inspect `git status` before editing and preserve unrelated local changes.
- After each coherent, verified unit of work, create a small focused commit.
  Stage only related files; use an imperative subject and a body explaining
  what changed and why. Do not commit broken or incomplete work for cadence.
- Push completed commits to the configured upstream unless the user asks not
  to or the branch is not intended for direct pushes.
- Do not incidentally stage generated plots, checkpoints, benchmark data,
  bytecode, `.DS_Store`, or editor state. Some such files are already tracked;
  the ignore rules do not protect them from modification.
- `notes/`, `figures/`, PNGs, and PDFs are ignored. Do not force-add research
  notes or generated artifacts unless they are explicitly part of the task.

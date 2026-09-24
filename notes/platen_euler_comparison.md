# Platen versus Euler–Maruyama

This comparison adds opt-in methods to `TorchOscillatorEnv`; existing scripts
continue to use `integration_method="legacy"`. No policy checkpoint is loaded,
trained or overwritten. The scripts use argparse rather than the repository's
`key=value` helper.

## Model and conventions

There are 25 sites and three modes per site. With zero-based indices,

\[
\omega_{z,ab}=1+((a+b)\bmod5)\Delta,\qquad
\omega_{x,ab}=4.5\omega_{z,ab},\qquad
\omega_{y,ab}=4.1\omega_{z,ab}.
\]

Every row and column contains the same five distinct frequencies. The supplied
plan did not retain the original three splitting values; the configurable
extended defaults are 0.01, 0.1 and 1. The user confirmed the row/column
nondegeneracy intent; the cyclic modulo assignment was made explicit before
running. At Delta=1, frequencies range from 1 to 22.5. Time is dimensionless,
with the lowest axial angular frequency equal to one; no SI conversion is used.

Each trajectory starts with xc=pc=Cxp=0 and Vxx=Vpp=11, hence each mode has
occupation exactly 5. Efficiency is 0.5. There is no gas damping or thermal bath.
The recoil heating rate is gamma=0.05*(1+epsilon), with epsilon the combined
row/column trap modulation. Both the Riccati drift and mean diffusion use this
same gamma. Each parametric channel has depth 0.05, so the combined modulation
is bounded by +/-0.1. Cold-force geometry follows the existing Bx/By convention;
z has no direct cold force.

With Y=(xc,pc,Vxx,Vpp,Cxp), the joint drift is

\[
a(Y)=\begin{pmatrix}
\omega p_c\\
-\omega(1+\epsilon)x_c+F\\
2\omega C_{xp}-4\eta\gamma V_{xx}^2\\
-2\omega(1+\epsilon)C_{xp}+4\gamma-4\eta\gamma C_{xp}^2\\
\omega V_{pp}-\omega(1+\epsilon)V_{xx}-4\eta\gamma V_{xx}C_{xp}
\end{pmatrix},\quad
b(Y)=2\sqrt{\eta\gamma}\begin{pmatrix}V_{xx}\\C_{xp}\\0\\0\\0\end{pmatrix}.
\]

There is one independent Wiener process per mode; its same increment enters
both mean quadratures. Force, epsilon and gamma stay fixed throughout each
step, and throughout each 0.01 controller interval in this diagnostic.

## Integrators

True Euler–Maruyama is Ynew=Y+a(Y)h+b(Y)dW. All terms use the old joint state.
The legacy momentum-first update is a different method and retains its existing
noise detachment and diagonal covariance clipping for compatibility.

The simplified Platen method uses

\[
\widetilde Y=Y+a(Y)h+b(Y)\Delta W,\qquad
Y_{new}=Y+\tfrac12[a(Y)+a(\widetilde Y)]h
+\tfrac12[b(Y)+b(\widetilde Y)]\Delta W.
\]

The reduction is specific to this SDE: noise moves only means, while diffusion
depends only on finite-variation covariances. Diffusion-direction derivatives
therefore vanish. The two diffusion probe states of the general Platen scheme
have identical covariances, equal to the predictor covariances. Holding controls
fixed is essential: recomputing the neural policy at a predictor state would
invalidate this reduction. Autograd still differentiates through the held
controls and both diffusion amplitudes. There is no stop-gradient or clipping
in either new method.

For the general weak second-order Platen formula, see Särkkä and Solin,
[Applied Stochastic Differential Equations, section 8.6](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf).
The reduction above is derived from this model's diffusion structure. Weak
order two is not a claim of second-order strong path convergence: using only
the shared Wiener increment leaves stochastic-integral error in individual
paths. The deterministic limit reduces to explicit trapezoidal/Heun stepping.

## Running and interpreting the diagnostic

```sh
uv run scripts/validate_integrators.py
uv run scripts/compare_integrators.py --profile smoke
uv run scripts/compare_integrators.py --profile extended
uv run scripts/compare_integrators.py --profile smoke --dtype float32
uv run scripts/compare_integrators.py --report-only notes/integrator_comparison/smoke
```

Smoke means batch 16, T=3, Delta=1. Extended means batch 256, T=20 and all three
splittings. Both sweep dt=0.01/2**k, k=0..6. All scenarios use identical initial
states and paired Wiener paths: a CPU float64 generator streams 256 fine
increments per controller interval, and coarse increments are their sums.
The same seed recreates the identical fine path in each run without retaining
the full path in memory. Candidates of either dtype are compared with float64
Platen at 0.01/128 and that reference is checked against 0.01/256.

The scenarios are no feedback, prescribed epsilon=-0.1, prescribed epsilon=+0.1,
and a fixed seeded 150-32-20 tanh neural network. Its cold-force outputs are
0.05*tanh(raw), while parametric raw outputs use the usual channel mapping.
The network has not been trained. This tests numerical behavior only.

`results.json` and `results.csv` record directional occupations and standard
errors, paired terminal occupation differences and standard errors, controller-
time mean RMS errors, covariance RMS/max and component errors, validity counts,
first failure time, timing and memory. Reference trajectories are saved in NPZ
files at controller times. Occupation is per mode (not the three-mode site sum).
Standard errors treat each whole independent trajectory as one sample and
average 25 sites within each direction. No-feedback heating is compared against
5+0.05*t; finite Monte Carlo residuals must be judged against their SEM.

Validity is checked after every integration step, without clipping: positive
variance diagonals, determinant Vxx*Vpp-Cxp**2 >= 1 (tolerance 1e-10 for float64,
1e-5 for float32), and finite states. Counts are mode-step observations, not
unique failed paths. Nonfinite trajectories stop with explicit failure status;
errors for such runs cover only fully completed controller intervals. They
never qualify for matched-accuracy speed comparisons. JSON uses null for
nonfinite diagnostic scalars.

Accuracy-run forward timing excludes noise generation, transfers and health
checks and synchronizes CUDA at each step. Policy runs additionally launch a
fresh process per timestep for full, untruncated forward/backward autograd over
a 0.1 time-unit window (configurable with --benchmark-duration). This limits
measurement memory without conflating it with full-T training memory. These
window timings synchronize CUDA at the boundaries, warm up one step and exclude
input-noise generation. CPU peak RSS includes Python, libraries, noise and
native tensors; CUDA peak allocated memory is separate. All timings are single
trials of eager code, with one CPU thread by default; no runtime optimizations
or compilation are included.

`report.md` compares fastest valid runs meeting common joint mean/covariance
error thresholds across every requested scenario. Reference-halving errors
must be below one fifth of each threshold. The convergence plots omit invalid
runs; consult the CSV for their failures. No method is chosen on per-step speed.

## Environment API

```python
env = TorchOscillatorEnv(
    oscillator, batch_size=16, dt=0.00125, horizon=2400,
    integration_method="platen",  # or "euler_maruyama"
    intensity_dependent_recoil=True,
    modulation_depth=0.05, dtype=torch.float64,
    pregenerate_noise=False,
)
env.reset(initial_conditions, covariances=(Vxx, Vpp, Cxp))
env.step(raw_action, dW)  # dW variance is dt, not one
health = env.covariance_health()
```

Supply controller outputs only every 0.01 and reuse them for all intervening
steps. For precomputed noise, pass `noise=` to reset; for seeded internal draws,
pass `generator=`. Compile `step` or `_integrated_step` when later studying
compilation; the locally existing `_step_impl` is the legacy core. This task
makes no change to training defaults or checkpoint formats.

## Results (2026-09-24)

Completed on the local CPU, PyTorch 2.10.0, one thread, seed 20260924:

- Full smoke sweep: float64, batch 16, T=3, Delta=1, all four scenarios, seven
  timesteps, both reference levels, and isolated seeded-policy BPTT workers.
- Float32 seeded-policy sweep at the same batch, duration and splitting, with
  float64 references; the seven levels track the float64 errors closely.
- Focused validation checks passed for deterministic order (one for EM, two
  for Platen), analytic stationary covariance and Ito heating identity,
  independent NumPy covariance convergence, expected occupation convergence
  without Monte Carlo noise, paired-noise reproducibility, finite-difference
  gradient agreement and timestep refinement of full policy gradients.
- Legacy default outputs were bitwise identical to the saved pre-change local
  environment, including internally generated noise. The separately staged
  version is checked against its own committed baseline before commit.

The full extended profile (T=20, batch 256, three splittings) has **not** been
run locally. CUDA is unavailable. Consequently these results support a next
experiment, not a universal training-integrator choice or GPU speed claim.

### Stability and accuracy

Platen had no covariance or uncertainty failures at any smoke timestep.
Euler became nonfinite at dt=0.01 and 0.005 in every scenario. At dt=0.0025,
it had a covariance failure without overflow in the no-feedback case and
became nonfinite in the other scenarios. Its four finer timesteps stayed valid.

The reference-halving differences are 0.00046–0.00052 in controller-time means
RMS. Maximum covariance differences range from 4.27e-6 (no feedback) to 3.38e-4
(seeded policy). These floors are small enough for the common 0.01 tolerance,
but the finest candidate path errors should not be treated as exact.

Using the **same joint tolerance** for means RMS and covariance maximum error,
and requiring every scenario to pass:

| Method | dt | Worst means RMS | Worst covariance max | 0.1-window forward | Backward | Peak process RSS |
|---|---:|---:|---:|---:|---:|---:|
| Euler–Maruyama | 0.00015625 | 0.0576 | 0.0766 | 0.0673 s | 0.1301 s | 315.4 MiB |
| Platen | 0.0025 | 0.0247 | 0.0369 | 0.0102 s | 0.0235 s | 233.6 MiB |
| Platen | 0.000625 | 0.00488 | 0.00373 | 0.0358 s | 0.0703 s | 268.0 MiB |

At tolerance 0.1, Platen at 0.0025 is about **5.85 times faster** in this single
forward-plus-backward measurement, while producing smaller errors. At 0.05,
only Platen qualifies. At 0.01, Platen needs dt=0.000625; Euler has no qualifying
run in the requested sweep. These are matched acceptance thresholds rather
than an assertion that the two errors are numerically identical.

For the seeded policy, paired terminal directional occupation errors at
Platen dt=0.000625 are (+0.0000934, -0.000113, -0.000124). Euler at its finest
requested timestep still has errors (+0.2668, +0.2151, +0.01265). Directional
SEM and every other requested metric are preserved in the result tables.

### Heating check

At T=3 the analytic no-feedback expectation is 5.15 in every direction.
The float64 Platen reference yields (mean +/- one SEM):

| Direction | Occupation |
|---|---:|
| x | 5.1024 +/- 0.1993 |
| y | 5.1260 +/- 0.1327 |
| z | 5.2537 +/- 0.2550 |

All are consistent with the analytic value. To distinguish integrator bias
from Monte Carlo uncertainty, the validation also propagates the exact second
moments of each discrete linear mean update. Halving h reduces EM's expected
occupation error by about two and Platen's by about four, confirming first-
and second-order weak heating convergence without sampling noise.

### Next training experiment

Use a small **Platen float32 pilot at dt=0.000625**, with the controller updated
every 0.01 (16 integration steps per action). Keep combined modulation bounded
by +/-0.1 and intensity-dependent recoil enabled; log unmasked covariance
health and save to a new checkpoint path. Repeat a short matched-noise
validation at dt=0.0003125 as the policy learns, because the fixed untrained
network does not establish accuracy under a trained policy. The looser,
faster dt=0.0025 is useful for exploratory runs if its documented errors are
acceptable. Keep the repository's training default unchanged for now.

Before committing to a production timestep or long training horizon, run the
extended profile on the intended training hardware, including the two reference
levels. The current 0.1-window memory measurements do not predict full-T BPTT
memory, and eager CPU timings do not predict compiled CUDA throughput.

### Artifacts

Compact result tables are intentionally versioned as part of this comparison:

- `data/integrator_comparison/smoke_results.json` and `.csv`
- `data/integrator_comparison/smoke_float32_results.json` and `.csv`

The JSON includes parameters, machine/software metadata, the exact policy
construction, timing scope, and all results. Large reference trajectories,
plots and generated reports remain local under `notes/integrator_comparison/`.
The human-readable smoke report is `notes/integrator_comparison/smoke/report.md`
and its figure is `convergence_delta1.png` in that directory. Both reports and
plots can be regenerated from a directory containing `results.json` with
`--report-only`.

## Extended execution and running estimates

The diagnostic now calibrates expected runtime with short full-batch probes,
then writes `progress.log` with UTC timestamps and estimates every 30 seconds.
Use `--progress-seconds` to change this interval or `--estimate-only` to measure
runtime without starting the sweep. The estimate excludes later result I/O and
short BPTT timing workers; stopped numerical failures can shorten the sweep.
Extended runs automatically use disk-backed controller-time histories and
chunked error reductions. These changes preserve bitwise identical trajectories
and error metrics within floating-point reduction roundoff.

`scripts/integrator_shards.py` runs the twelve independent splitting/scenario
combinations on Slurm ranks in an **interactive allocation**, with unchanged
batch size, duration, timestep grid and noise pairing. It does not submit a job.
For example, from the repository root inside a CPU allocation with 12 tasks:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --unbuffered -n 12 -c 2 --cpu-bind=cores uv run --no-sync scripts/integrator_shards.py --device cpu --output /path/to/run
uv run --no-sync scripts/integrator_shards.py --output /path/to/run --merge
```

The merge validates configuration agreement and complete, nonduplicate coverage
before writing `summary/results.json`, CSV, report and plots. Each rank also
streams timestamped progress to the interactive session and its case log. CPU
and CUDA environment assertions are both exercised when CUDA is available.
Current execution details and estimate revisions are kept in
`notes/integrator_comparison/extended_run_log.md`.

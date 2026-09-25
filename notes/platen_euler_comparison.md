# Platen versus Euler–Maruyama

This comparison adds opt-in methods to `TorchOscillatorEnv`; existing scripts
continue to use `integration_method="legacy"`. No policy checkpoint is loaded,
trained or overwritten. The scripts use argparse rather than the repository's
`key=value` helper.

**Extended comparison completed:** all 192 records were verified on Perlmutter
on 2026-09-24. Platen at dt=0.0003125 passes the joint 0.01 error criterion
across all three splittings and all four scenarios. Euler–Maruyama has no
qualifying timestep at that tolerance. See the final results below; the generic
environment's legacy default was not changed by this comparison.

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

## Local smoke results (2026-09-24)

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

These smoke runs used the local CPU, where CUDA is unavailable. The extended
Perlmutter results below supersede the smoke-only timestep recommendation for
T=20. Neither CPU study establishes compiled CUDA training throughput.

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

### Smoke-only training recommendation (superseded for T=20)

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

### Surviving laptop disconnects

Use `scripts/run_integrator_interactive.sh` inside **tmux on a Perlmutter login
node**. The tmux server owns the interactive `salloc` process, so closing the
local SSH connection does not end the allocation. The four-hour allocation
limit and remote node availability still apply. No batch job is submitted.

The launcher expects an isolated run root containing `code/` (the repository
and its synced `.venv`) and `bin/uv`. It uses the run root for the uv cache,
outputs and logs. For example, after connecting to Perlmutter:

```sh
ssh -t login01
tmux new-session -s integrators-20260924
bash "$SCRATCH/integrator_comparison_20260924/code/scripts/run_integrator_interactive.sh" \
  "$SCRATCH/integrator_comparison_20260924" extended_cpu_persistent m5258
```

Detach with **Ctrl-b, then d**. To reattach, connect to Perlmutter, then to the
same login node (`ssh -t login01` in this example), and run
`tmux attach -t integrators-20260924`. Perlmutter's public SSH endpoint balances
connections across login nodes, so the node name matters. See the
[NERSC connection documentation](https://docs.nersc.gov/connect/).

The launcher refuses existing output names. It records `<output>.login_host`,
`<output>.job_id`, `<output>.launch.log` and, after exit, `<output>.status`.
Successful completion automatically validates and merges all twelve cases into
`<output>/summary/`, then releases the allocation. The merge also runs on the
compute node. A nonzero exit status records execution failure; numerical
failures within completed diagnostic cases remain scientific result records.
Progress and ETA logs continue to update remotely without a connected laptop.

**Initial persistent run, 2026-09-24 20:45 UTC:** job 58836116 on CPU node `nid004203`,
account `m5258`; detached tmux session `integrators-20260924` on `login01`.
Output root is `$SCRATCH/integrator_comparison_20260924/extended_cpu_persistent`.
All twelve cases progressed after the launching SSH connection exited,
verified from a fresh connection. Estimated completion is 22:10–22:30 UTC
(15:10–15:30 Pacific), subject to later I/O and timing overhead. This replaces
cancelled job 58835882, whose early partial output is preserved in
`extended_cpu/`. The launcher merges and releases the allocation automatically.
This attempt subsequently failed as described below; final results are now
available in the last section.

### Recovering the interrupted extended run

Job 58836116 failed at **2026-09-24 21:48:25 UTC**, after 1h04m48s. The first
completed case could not render its plot because inherited Matplotlib settings
enabled external TeX without the `cmr10.tfm` font. Slurm's failure propagation
then terminated the other ranks. This was a reporting/execution failure, not a
diagnostic numerical failure. The detached session survived; SSH disconnection
did not cause this stop. There were **156 of 192 saved records**, including all
24 reference runs. All twelve reference archives were intact.

Reports now explicitly use Matplotlib's built-in font rendering in a scoped
configuration, overriding external TeX settings. Result JSON/CSV files are
replaced atomically. `--resume-from` loads a case's saved configuration and
references, validates record identity, skips completed runs, and recomputes only
missing whole trajectories from the same seed; no physical parameters or
controller timing change. For example:

```sh
uv run scripts/compare_integrators.py --resume-from /path/to/case
srun -n 12 -c 2 uv run scripts/integrator_shards.py --output /path/to/run --resume
```

For the persistent interactive launcher, append `resume` to the previous
three-argument command. It appends per-case logs while creating new timestamped
allocation logs/status files (`<output>.resume_<UTC>.*`), preserving the failed
attempt's files. A short plotting/resume preflight runs on the compute node
before the sweep; `--kill-on-bad-exit=0` lets independent ranks finish if another
rank fails, while retaining the nonzero overall execution status.

`uv run scripts/validate_integrator_resume.py` passed locally: missing candidate
rows match uninterrupted numerical results exactly, saved rows/reference
archives remain unchanged, completed cases do no integration, duplicates are
rejected, and reports render even when site settings request TeX and all TeX
calls are forced to fail. The existing full integrator assertions also passed.

**Recovery launch, 2026-09-24 22:46 UTC:** interactive CPU job 58839589 on
`nid004142`, account `m5258`, in the same detached tmux session on `login01`.
The compute-node preflight passed and all unfinished cases are progressing.
Only 36 missing runs are being recomputed. The updated completion estimate is
23:15–23:30 UTC (16:15–16:30 Pacific), allowing for timing/merge overhead.

## Final extended results (completed 2026-09-24)

The resumed allocation **58839589 completed successfully at 23:13:18 UTC
(16:13:18 Pacific)**, after 28m48s, and was released automatically. The merge
contains all **192 records**: 168 candidates and 24 reference runs, covering
batch 256, T=20, three splittings, four scenarios, seven timesteps per method,
and both float64 reference resolutions. Candidate precision is also float64.
All 42 isolated short-window forward/backward measurements completed with
finite losses and gradients. These were eager CPU trials on Perlmutter with
PyTorch 2.10.0+cu128 and one Torch thread per case.

Downloaded JSON/CSV SHA-256 hashes match the remote files. Every merged record
matches its per-case file, and all 156 records saved before recovery remain
exactly unchanged, including their original timing measurements. The existing
integrator assertions, resume equivalence check, and compute-node plotting
preflight passed. Convergence figures were generated successfully; the Delta=1
figure was visually checked.

### Accuracy and covariance health

The strict criterion requires **both** controller-time means RMS and maximum
absolute covariance error to be at most 0.01 in **every scenario**. Reference
halving errors must be below 0.002. The fastest qualifying Platen choices are:

| Splitting Delta | Platen dt | Substeps per controller update | Worst means RMS | Worst covariance error |
|---:|---:|---:|---:|---:|
| 0.01 | 0.0025 | 4 | 0.008897 | 0.002490 |
| 0.1 | 0.00125 | 8 | 0.005149 | 0.001732 |
| 1 | 0.0003125 | 32 | 0.004042 | 0.003435 |

Euler–Maruyama has no qualifying run at tolerance 0.01 for any splitting in the
requested sweep. At Delta=1, even its finest dt=0.00015625 has worst means RMS
0.7345 and covariance error 0.7116. Platen at dt=0.000625 has errors 0.01187 and
0.01218 there: the T=3 smoke recommendation narrowly misses the T=20 criterion.

All 24 references completed without covariance/nonfinite failures. The largest
reference-halving differences are 0.000756 (means RMS) and 0.000547 (covariance
maximum), sufficiently below the strict threshold; finest-grid errors are
still relative to a finite-resolution reference.

Of 84 candidates per method, Platen has **80 valid runs**, two finite runs with
covariance failures, and two nonfinite runs. All four failures occur at
Delta=1, dt=0.01. Every finer Platen timestep is covariance-valid in all cases.
Euler has **49 valid runs**, four finite covariance failures, and 31 nonfinite
runs. Its largest timestep valid in every scenario is 0.0025 for Delta=0.01
and 0.1, and 0.00015625 for Delta=1. Numerical validity alone does not imply
acceptable accuracy. Every failed candidate recorded uncertainty violations
and nonpositive covariance diagonals without clipping; per-step counts and
first-failure times are retained in the tables.

For the fixed untrained policy at Delta=1, Platen dt=0.0003125 has paired
terminal occupation differences (x,y,z) of
(0.0000877, 0.0000983, 0.0000102), with paired SEM
(0.0002084, 0.0001756, 0.0000328). The corresponding reference occupations are
(7.3362, 6.8803, 6.0206). This is a numerical comparison, not evidence of learned
cooling performance.

### Heating check

The analytic no-feedback expectation at T=20 is 6 per direction. Float64
Platen reference results (mean +/- one SEM) are:

| Delta | x occupation | y occupation | z occupation |
|---:|---:|---:|---:|
| 0.01 | 6.0441 +/- 0.0728 | 6.0144 +/- 0.0683 | 6.0427 +/- 0.0726 |
| 0.1 | 5.9853 +/- 0.0701 | 6.0173 +/- 0.0831 | 6.0527 +/- 0.0745 |
| 1 | 5.9702 +/- 0.0737 | 5.9532 +/- 0.0756 | 5.9560 +/- 0.0761 |

All nine values are within 0.71 SEM of the analytic expectation.

### Speed at matched acceptance thresholds

These comparisons require all four scenarios to meet the same joint error
threshold and covariance checks. Times cover forward plus backward over the
0.1-unit seeded-policy window, with batch 256:

| Delta | Tolerance | Euler dt / seconds | Platen dt / seconds | Measured speed ratio |
|---:|---:|---:|---:|---:|
| 0.01 | 0.1 | 0.0003125 / 1.221 | 0.01 / 0.0852 | 14.34x |
| 0.01 | 0.05 | 0.00015625 / 2.469 | 0.005 / 0.1945 | 12.70x |
| 0.1 | 0.1 | 0.00015625 / 2.397 | 0.01 / 0.0820 | 29.24x |

There is no matched-threshold speed ratio at Delta=1 because Euler does not
qualify even at tolerance 0.1. Platen dt=0.00125 passes 0.1 and 0.05 there;
dt=0.0003125 passes 0.01. These are single eager measurements, including runs
on the original and resumed CPU allocations with different concurrent case
counts; they do not establish a GPU speed ratio or timing uncertainty.

At the strict Delta=1 Platen choice, forward/backward take **1.078/1.347 s**
over the 0.1-unit window. Reported peak process RSS is **3522.9 MiB**. RSS is a
whole-process high-water measure, including runtime/noise/launch overhead, not
the memory of live autograd tensors; repeated RSS plateaus make it unsuitable
for claiming a method-specific memory advantage here. Full-T training memory
and CUDA peak allocated memory were not measured by this CPU comparison.

### Recommendation for the next training experiment

Choose **Platen** for the next training experiment. For one timestep spanning
all three splittings, start at **dt=0.0003125**, with controller updates every
0.01 (32 held-action substeps). Retain the agreed recoil model, modulation
bounds, initial state and unmasked covariance-health logging. Use a new output
checkpoint path. Verify the selected float32 rollout against the float64 path
and repeat a short paired test at dt=0.00015625 as the policy changes.

The completed extended sweep establishes float64 accuracy for the prescribed
signals and fixed untrained network. The earlier float32 smoke and separate
GPU training pilot do not replace a float32/trained-policy timestep check at
the new horizon. The existing residual-array trainer documented in
`docs/array_training.md` already uses Platen at Delta=0.1, dt=0.000625; that is
finer than the 0.00125 threshold found here, so this comparison does not call
for changing its current timestep. Its trained policy still needs refinement
checks. This task leaves the generic environment's legacy default unchanged.

### Delivered artifacts

- Versioned full tables: `data/integrator_comparison/extended_results.json`
  and `extended_results.csv`. They include all requested occupations,
  covariance errors, paired path differences, failures, timing and memory.
- Local report and three convergence figures:
  `notes/integrator_comparison/extended_perlmutter/`.
- Local compact audit archive:
  `notes/integrator_comparison/completed_audit_20260924.tar.gz`.
- Remote case files and large reference trajectories:
  `$SCRATCH/integrator_comparison_20260924/extended_cpu_persistent/`.
- Running log, including the reporting failure and recovery:
  `notes/integrator_comparison/extended_run_log.md`, also copied to `RUN_LOG.md`
  at the remote run root.

The downloaded JSON SHA-256 is
`2145940bcf1ef92da2a8946bcdc5b7438d206d1582a60bb069d132c5387a3ab0`;
the CSV SHA-256 is
`b39e7b5d2fa4b87245fd4d2d026103d5d7559bbd4d3ee91902298ea91661b7dc`.

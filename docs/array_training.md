# Residual array-feedback training

This trainer is intended for remote CPU or CUDA servers. Only the small eager
smoke tests below have been run locally. Compilation, production horizons,
memory scaling, timestep refinement of trained policies, and cooling performance
must be evaluated remotely. No remote jobs are submitted by these scripts.

## Model and cost

There are 25 particles, three modes per particle, and 20 shared commands:
five row-x cold commands, five column-y cold commands, five row modulations,
and five column modulations. With zero-based indices,

```text
omega_z[a,b] = 1 + ((a+b) % 5) * 0.1
omega_x = 4.5 * omega_z
omega_y = 4.1 * omega_z
```

Each row and column contains the same five frequencies. Efficiency is 0.5;
recoil is `0.05*(1+epsilon)`, including its influence on measurement diffusion.
Each modulation channel is `0.05*tanh(raw)`, giving a combined bound of +/-0.1.
Time is dimensionless in inverse lowest axial angular frequency. There is no
delay, thermal bath, or interparticle interaction.

Every trajectory starts with zero means, `Vxx=Vpp=11`, and `Cxp=0`, giving
occupation five per mode. One independent Wiener innovation per mode enters
both of that mode's conditional-mean equations.

The per-trajectory objective is the time average of these two costs:

```text
centroid_cost = sum_over_particles_and_xyz(omega*(xc**2 + pc**2)/4) / 25
force_cost   = sum_over_particles_and_xy(omega*f_p**2/2) / (25*g_fb**2)
```

`f_p` is the additive term in the dimensionless momentum drift. Since
`F=sqrt(m*hbar*omega/2)*f_p`, the second expression is the requested physical
force penalty in `m=hbar=1` units. It includes both LQR and residual cold forces.
Covariance energy is excluded from the objective but included in occupation
diagnostics. Parametric modulation has its amplitude bound and recoil effect,
without a separate force penalty. The cold coupling uses the benchmark's
unmodulated `B=omega*sqrt(omega/omega_ref)` and an independent additive force;
no extra `(1+epsilon)` factor is applied to the cold force.

LQR solves the CARE separately for each shared cold actuator, with interleaved
group coordinates `[x1,p1,...,x5,p5]`, `Q=diag(repeat(omega,2))/100`, and
`R=sum(omega*B**2)/(50*g_fb**2)`. The gain is `solve(R, B.T@P)` and commands
are `-K@state`. All rows use the same x reference frequency; all columns use
the same y reference frequency. The legacy LQR implementation is not changed.

## Policy and rollout

The policy is a 375-128-128-20 MLP with tanh hidden layers. Inputs are flattened
`xc, pc, Vxx, Vpp, Cxp`, in that order. Means are divided by `sqrt(11)` and
covariances by 11. The last layer starts at zero: the initial policy is exactly
LQR cold damping, with no modulation. All 20 residual outputs are trained.

Platen integration uses `dt=0.000625`; the policy updates every `0.01` and holds
commands fixed for 16 substeps. State energy uses trapezoidal quadrature; the
held cold-force cost is integrated exactly over each interval. Both are divided
by trajectory duration. Full gradients pass through means, covariances, and
diffusion. There is no covariance clipping or detached diffusion.

The baseline processes the entire configured batch together. Noise is drawn
one controller interval at a time; autograd can retain it and intermediate
states until backward. Memory still grows with batch size and horizon.
Microbatching and time checkpointing are deferred until remote measurements
show that they are needed. Lowering batch size changes the effective batch;
the current implementation does not silently accumulate gradients.

`--compile` compiles a functional controller interval, including the policy,
integration, costs, and detached health diagnostics. RNG, counters, progress,
and optimization stay outside compilation. Full-graph capture is required.
The full trajectory is not unrolled into one compiled graph. No local compiled
execution has been validated yet.

Every integration step contributes to health counts and minima. The host checks
the accumulated results every 100 controller intervals and at rollout end,
and stops on nonfinite states, nonpositive diagonals, or uncertainty violations.
The determinant tolerance is `1e-5` in float32 and `1e-10` in float64. Counts
are mode-step observations, not numbers of independent failed trajectories.

## Local smoke checks

From the repository root, using the existing uv environment:

```sh
PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python -B scripts/validate_array_training.py
PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python -B scripts/train_array_feedback.py --smoke
```

Validation uses batch two and at most two controller intervals per rollout.
It performs two optimizer updates in total, checks physical force costs, LQR
optimality, a fixed-noise parametric finite difference, checkpoint replay, a
tiny paired evaluation, import/CLI safety, and curriculum arithmetic. No
compilation or timing benchmark is invoked. The CLI smoke command separately
performs two tiny updates, writing temporary outputs that are removed afterward.
Use `--smoke --output FRESH_DIRECTORY` to retain those smoke artifacts.

## Remote execution

Install the locked environment on the server using `uv sync --locked`.
Run from the repository root, within an appropriate compute allocation if
required by the server. CUDA must be available to that environment; requesting
an unavailable CUDA device fails explicitly. Start with small batches and
measure memory before running the long curriculum.

Configuration inspection creates no files or simulation tensors:

```sh
uv run scripts/train_array_feedback.py --dry-run --batch-size 16 --base-iterations 10
```

The following are example remote commands, not validated hardware choices:

```sh
uv run scripts/train_array_feedback.py --benchmark --device cuda --batch-size 16 --duration 0.1 --repeats 3 --output data/array_training/benchmark-001
uv run scripts/train_array_feedback.py --device cuda --batch-size 16 --base-iterations 10 --output data/array_training/train-001
```

The benchmark always compares eager and compiled modes with the same seeded,
untrained policy and noise, without optimizer updates. It checks loss and
gradient agreement, records first-call overhead separately, and reports warmed
forward/backward timings. CUDA timings synchronize at boundaries. CPU peak RSS
is a process-lifetime high-water mark, not an isolated comparison between modes.
Compilation and its backward pass can take substantial time.

Add `--compile` to a training command only after checking that comparison.
For CPU execution use `--device cpu --threads NUMBER`. No speedup is assumed.

For base count `K`, the training stages are:

| Duration | Optimizer updates | Total simulated time per trajectory |
|---:|---:|---:|
| 5 | 20K | 100K |
| 20 | 5K | 100K |
| 100 | K | 100K |

Timestep, control interval, and batch size remain fixed. The allocation is equal
simulated time, not guaranteed equal wall time. Adam uses learning rate 0.0005;
gradient norm is clipped to one before every update. Optimizer state carries
across stages. `--g-fb` defaults to 1. `--dt` and `--dtype` support explicit
remote numerical validation; changing timestep does not change controller rate.

Each fresh run refuses an existing output directory and writes `config.json`,
`metrics.jsonl`, and `checkpoint.pt`. Checkpoints are replaced atomically after
each completed update, and include physics, model buffers, architecture,
normalization, optimizer, curriculum position, and RNG state. A failed update
leaves the preceding saved checkpoint available; records include interruptions.
An interruption between checkpoint and log writes can leave a missing metric
record; the checkpoint is the source of truth for training progress.

Resume explicitly, with no overrides to checkpointed training settings:

```sh
uv run scripts/train_array_feedback.py --resume data/array_training/train-001/checkpoint.pt --output data/array_training/train-001 --device cuda
```

Pass `--compile` again if desired; execution mode is chosen at launch. Exact
replay is tested only for CPU eager execution in the same environment. Training
resume requires the same device type for RNG compatibility. Evaluation can use
a different device; cross-hardware or eager/compiled bitwise equivalence is not
promised. Use only one writer per run directory.

Independent evaluation replays identical fresh Wiener paths for learned and
LQR controllers and reports paired differences with trajectory-based SEM:

```sh
uv run scripts/train_array_feedback.py --evaluate data/array_training/train-001/checkpoint.pt --device cuda --batch-size 64 --duration 100 --output data/array_training/eval-001
```

It reports time-averaged centroid cost, force cost, their sum, terminal
directional occupations, and covariance health. It does not claim a steady
state from a finite duration. Use independent evaluation and timestep refinement
before interpreting trained-policy improvements as physical cooling results.

## Batch-size diagnostics

Run fixed-weight gradient diagnostics remotely, separately for each batch size
and policy snapshot. Omit `--checkpoint` for the original seeded initialization.
No optimizer updates are performed and no checkpoint is modified.

```sh
uv run scripts/diagnose_array_batch.py --device cuda --compile --batch-size 16 --durations 5 20 100 --repeats 16 16 8 --checkpoint data/array_training/train-001/checkpoint.pt --output data/array_training/batch-diagnostic-001
```

Repeat for batches 32 and 64. Independent noise batches estimate raw gradient
variance before clipping, with reports for all parameters, hidden layers,
cold-output parameters and parametric-output parameters. Hidden parameters serve
both control types; these groups are not gradients of separate cost components.
At zero-output initialization the hidden-layer gradients are exactly zero.

For R independent batch-mean gradients, S is the sample covariance trace.
The squared mean signal is estimated by `||mean_gradient||² - S/R`; the simple
gradient noise scale is `batch_size*S/signal_squared`. Ratios are left undefined
when the corrected signal is nonpositive. A delete-one jackknife standard error
helps identify weak signal estimates. Pairwise cosine similarities measure
direction agreement without fitting a reference direction to the same batch.
These statistics guide experiments, not an optimal batch-size claim for Adam.
See [McCandlish et al.](https://arxiv.org/abs/1812.06162) for the noise-scale idea.

The output includes raw gradient vectors, seeds, per-trial costs and covariance
health, full-horizon forward/backward timings, and CUDA peak allocated memory.
A two-interval compilation warmup is excluded from timings. Peaks exclude Adam
moment buffers; CUDA reserved memory reflects the allocator's history. An OOM
is recorded explicitly, never treated as a successful measurement. Evaluate
learning progress separately before choosing a production batch size.

Only `--smoke` is intended locally; it uses three batch-2, T=0.02 CPU/eager
trials and temporary outputs. Estimator and matched-path batch-averaging checks
are included in `validate_array_training.py`.

## First Perlmutter GPU pilot (2026-09-24)

Source `02c9336` completed an interactive Perlmutter run using one A100-SXM4-40GB
per compute step, PyTorch 2.10.0+cu128, and float32 with TF32 disabled. The
allocation was released after completion; no remote jobs are left running by
this pilot. No implementation changes were required.

For batch 16 and T=0.1, compiled and eager losses/gradients agreed within the
benchmark tolerances. Warmed forward/backward time averaged **0.0240 s compiled
versus 0.4953 s eager** (20.6x); the first compiled call took **170.35 s**.
These timings describe this short window and hardware, not a universal speedup.

Compiled batch-16 training completed all **20/5/1 updates at T=5/20/100** with
the fixed timestep and controller interval. Summed update time was **93.72 s**;
the complete training job step took **1m44s** including startup. There were no
covariance-health failures, and the T=100 update took 31.35 s. Microbatching and
time checkpointing were not needed at this batch size.

Independent paired evaluation (64 trajectories, T=100) gave total objectives
**15.8561 learned versus 15.8950 LQR**, a paired difference of **-0.0389 ± 0.0777**
(one SEM). This does not establish an overall improvement. Terminal x/y/z
occupations were approximately **0.475/0.494/9.923**; axial cooling remains
unresolved. Longer training, training-seed checks, and learned-policy timestep
refinement remain outstanding.

Local artifacts are in the ignored directory
`data/array_training/perlmutter_20260924T214052Z/`; the research run log is
`notes/array_training/perlmutter_pilot_20260924.md`. These generated artifacts
are not included in a fresh Git checkout.

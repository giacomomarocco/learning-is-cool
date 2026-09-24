#!/usr/bin/env bash
# Invoke from tmux on a Perlmutter login node. All computation runs under srun.
# Usage: bash scripts/run_integrator_interactive.sh RUN_ROOT OUTPUT_NAME ACCOUNT [resume]
# RUN_ROOT contains code/ (this repository), bin/uv, and the synced code/.venv.
set -euo pipefail

integrator_allocated=false
if [[ ${1:-} == --allocated ]]; then
    integrator_allocated=true
    shift
fi
if [[ $# -lt 3 || $# -gt 4 || ! $2 =~ ^[a-zA-Z0-9_-]+$ || ! ${4:-new} =~ ^(new|resume)$ ]]; then
    echo "Usage: $0 RUN_ROOT OUTPUT_NAME ACCOUNT [resume]" >&2
    exit 2
fi
integrator_root=$(cd -- "$1" && pwd)
integrator_name=$2
integrator_account=$3
integrator_mode=${4:-new}
integrator_output="$integrator_root/$integrator_name"
integrator_script="$integrator_root/code/scripts/run_integrator_interactive.sh"
export PATH="$integrator_root/bin:$PATH"
export UV_CACHE_DIR="$integrator_root/cache"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
cd -- "$integrator_root/code"

if [[ $integrator_allocated == false ]]; then
    if [[ -z ${TMUX:-} || -n ${SLURM_JOB_ID:-} ]]; then
        echo 'Start this launcher inside tmux on a login node, outside an allocation.' >&2
        exit 2
    fi
    export INTEGRATOR_LOG_BASE="$integrator_output"
    if [[ $integrator_mode == resume ]]; then
        test -d "$integrator_output"
        export INTEGRATOR_LOG_BASE="$integrator_output.resume_$(date -u +%Y%m%dT%H%M%SZ)"
    elif [[ -e $integrator_output ]]; then
        echo "Refusing to overwrite $integrator_output; use resume explicitly." >&2
        exit 2
    fi
    for integrator_artifact in "$INTEGRATOR_LOG_BASE.launch.log" "$INTEGRATOR_LOG_BASE.status" "$INTEGRATOR_LOG_BASE.job_id" "$INTEGRATOR_LOG_BASE.login_host"; do
        if [[ -e $integrator_artifact ]]; then
            echo "Refusing to overwrite $integrator_artifact; choose a fresh OUTPUT_NAME." >&2
            exit 2
        fi
    done
    exec > >(tee "$INTEGRATOR_LOG_BASE.launch.log") 2>&1
    trap 'integrator_rc=$?; printf "exit_code=%s\nfinished_utc=%s\n" "$integrator_rc" "$(date -u +%FT%TZ)" > "$INTEGRATOR_LOG_BASE.status"; exit "$integrator_rc"' EXIT
    hostname > "$INTEGRATOR_LOG_BASE.login_host"
    printf 'started_utc=%s\n' "$(date -u +%FT%TZ)"
    salloc -A "$integrator_account" -C cpu -q interactive -N 1 \
        --ntasks=12 --cpus-per-task=2 --time=04:00:00 \
        bash "$integrator_script" --allocated "$integrator_root" "$integrator_name" "$integrator_account" "$integrator_mode"
else
    : "${SLURM_JOB_ID:?The worker requires an interactive Slurm allocation}"
    : "${INTEGRATOR_LOG_BASE:?Missing launcher log prefix}"
    printf '%s\n' "$SLURM_JOB_ID" > "$INTEGRATOR_LOG_BASE.job_id"
    printf 'job_id=%s allocated_utc=%s\n' "$SLURM_JOB_ID" "$(date -u +%FT%TZ)"
    mkdir -p "$integrator_root/tmp"
    TMPDIR="$integrator_root/tmp" srun --unbuffered --ntasks=1 --cpus-per-task=2 --exact --cpu-bind=cores \
        uv run --no-sync scripts/validate_integrator_resume.py
    integrator_shard_args=()
    if [[ $integrator_mode == resume ]]; then
        integrator_shard_args+=(--resume)
    fi
    srun --unbuffered --kill-on-bad-exit=0 --ntasks=12 --cpus-per-task=2 --cpu-bind=cores \
        uv run --no-sync scripts/integrator_shards.py --device cpu --output "$integrator_output" "${integrator_shard_args[@]}"
    srun --unbuffered --ntasks=1 --cpus-per-task=2 --exact --cpu-bind=cores \
        uv run --no-sync scripts/integrator_shards.py --output "$integrator_output" --merge
fi

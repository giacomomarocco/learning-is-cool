#!/usr/bin/env bash
# Invoke from tmux on a Perlmutter login node. All computation runs under srun.
# Usage: bash scripts/run_integrator_interactive.sh RUN_ROOT OUTPUT_NAME ACCOUNT
# RUN_ROOT contains code/ (this repository), bin/uv, and the synced code/.venv.
set -euo pipefail

integrator_allocated=false
if [[ ${1:-} == --allocated ]]; then
    integrator_allocated=true
    shift
fi
if [[ $# != 3 || ! $2 =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Usage: $0 RUN_ROOT OUTPUT_NAME ACCOUNT" >&2
    exit 2
fi
integrator_root=$(cd -- "$1" && pwd)
integrator_name=$2
integrator_account=$3
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
    for integrator_artifact in "$integrator_output" "$integrator_output.launch.log" "$integrator_output.status" "$integrator_output.job_id" "$integrator_output.login_host"; do
        if [[ -e $integrator_artifact ]]; then
            echo "Refusing to overwrite $integrator_artifact; choose a fresh OUTPUT_NAME." >&2
            exit 2
        fi
    done
    exec > >(tee "$integrator_output.launch.log") 2>&1
    trap 'integrator_rc=$?; printf "exit_code=%s\nfinished_utc=%s\n" "$integrator_rc" "$(date -u +%FT%TZ)" > "$integrator_output.status"; exit "$integrator_rc"' EXIT
    hostname > "$integrator_output.login_host"
    printf 'started_utc=%s\n' "$(date -u +%FT%TZ)"
    salloc -A "$integrator_account" -C cpu -q interactive -N 1 \
        --ntasks=12 --cpus-per-task=2 --time=04:00:00 \
        bash "$integrator_script" --allocated "$integrator_root" "$integrator_name" "$integrator_account"
else
    : "${SLURM_JOB_ID:?The worker requires an interactive Slurm allocation}"
    printf '%s\n' "$SLURM_JOB_ID" > "$integrator_output.job_id"
    printf 'job_id=%s allocated_utc=%s\n' "$SLURM_JOB_ID" "$(date -u +%FT%TZ)"
    srun --unbuffered --ntasks=12 --cpus-per-task=2 --cpu-bind=cores \
        uv run --no-sync scripts/integrator_shards.py --device cpu --output "$integrator_output"
    srun --unbuffered --ntasks=1 --cpus-per-task=2 --exact --cpu-bind=cores \
        uv run --no-sync scripts/integrator_shards.py --output "$integrator_output" --merge
fi

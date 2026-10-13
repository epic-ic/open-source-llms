#!/usr/bin/env bash
# Resource requests come from uv run llm-submit; the job uses its recorded interpreter.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || $# -ne 3 ]]; then
    printf 'Use: uv run llm-submit PROFILE\n' >&2
    exit 1
fi
project_dir="$1"
job_config="$2"
python_bin="$3"
cd "$project_dir"
exec "$python_bin" -m slurm_llm.commands.serve --root "$project_dir" --job-config "$job_config"

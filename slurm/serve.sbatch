#!/usr/bin/env bash
# All resource requests come from `llm submit`, not model-specific #SBATCH blocks.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || $# -ne 3 ]]; then
    printf 'Use: python3 main.py submit PROFILE\n' >&2
    exit 1
fi
project_dir="$1"
job_config="$2"
python_bin="$3"
cd "$project_dir"
exec "$python_bin" "$project_dir/main.py" --root "$project_dir" serve --job-config "$job_config"

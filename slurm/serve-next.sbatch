#!/usr/bin/env bash
#SBATCH --job-name=qwen-next-api
#SBATCH --partition=main
#SBATCH --nodelist=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_pro_5000_blackwell:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/qwen-next-%j.log
#SBATCH --error=logs/qwen-next-%j.log

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_SUBMIT_DIR:-}" ]]; then
    printf 'Submit with sbatch slurm/serve-next.sbatch from the project directory.\n' >&2
    exit 1
fi
cd "$SLURM_SUBMIT_DIR"

if [[ -z "${LLM_MODEL_FILE:-}" ]]; then
    if [[ ! -r .tools/qwen-next/model.path ]]; then
        printf 'Download the model first: bash scripts/prepare-next.sh\n' >&2
        exit 1
    fi
    export LLM_MODEL_FILE="$(< .tools/qwen-next/model.path)"
fi

# Catch incomplete split-model downloads before starting the GPU runtime.
if [[ "$LLM_MODEL_FILE" =~ ^(.*)-00001-of-([0-9]{5})\.gguf$ ]]; then
    shard_prefix="${BASH_REMATCH[1]}"
    shard_count=$((10#${BASH_REMATCH[2]}))
    for ((index=1; index<=shard_count; index++)); do
        printf -v shard '%s-%05d-of-%05d.gguf' "$shard_prefix" "$index" "$shard_count"
        if [[ ! -s "$shard" ]]; then
            printf 'Missing or empty GGUF shard: %s\n' "$shard" >&2
            exit 1
        fi
    done
fi

export LLM_MODEL_ALIAS="${LLM_MODEL_ALIAS:-qwen3.8-flash-next}"
export LLM_CTX_SIZE="${LLM_CTX_SIZE:-32768}"
export LLM_PARALLEL="${LLM_PARALLEL:-1}"
export LLM_SPLIT_MODE="${LLM_SPLIT_MODE:-layer}"
export LLM_TENSOR_SPLIT="${LLM_TENSOR_SPLIT:-1,1,1,1}"
export LLM_LAZY_MODE="${LLM_LAZY_MODE:-auto}"

# Bash ignores the inner file's #SBATCH directives; this job's allocation applies.
exec bash slurm/serve.sbatch

#!/usr/bin/env bash
#SBATCH --job-name=glm-flash-api
#SBATCH --partition=main
#SBATCH --nodelist=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_pro_5000_blackwell:5
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=logs/glm-flash-%j.log
#SBATCH --error=logs/glm-flash-%j.log

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_SUBMIT_DIR:-}" ]]; then
    printf 'Submit with sbatch slurm/serve-glm.sbatch from the project directory.\n' >&2
    exit 1
fi
cd "$SLURM_SUBMIT_DIR"

if [[ -z "${LLM_MODEL_FILE:-}" ]]; then
    if [[ ! -r .tools/glm-flash/model.path ]]; then
        printf 'Download the model first: bash scripts/prepare-glm.sh\n' >&2
        exit 1
    fi
    export LLM_MODEL_FILE="$(< .tools/glm-flash/model.path)"
fi
if [[ ! -s "$LLM_MODEL_FILE" ]]; then
    printf 'Missing or empty model: %s\n' "$LLM_MODEL_FILE" >&2
    exit 1
fi
if [[ "$LLM_MODEL_FILE" =~ ^(.*)-00001-of-([0-9]{5})\.gguf$ ]]; then
    shard_prefix="${BASH_REMATCH[1]}"
    shard_count=$((10#${BASH_REMATCH[2]}))
    if ((shard_count < 1)); then
        printf 'Invalid GGUF shard count.\n' >&2
        exit 1
    fi
    for ((index=1; index<=shard_count; index++)); do
        printf -v shard '%s-%05d-of-%05d.gguf' "$shard_prefix" "$index" "$shard_count"
        if [[ ! -s "$shard" ]]; then
            printf 'Missing or empty GGUF shard: %s\n' "$shard" >&2
            exit 1
        fi
    done
fi

# Use the actual allocation when sbatch --gres overrides the five-GPU default.
if [[ -z "${LLM_TENSOR_SPLIT:-}" ]]; then
    gpu_count="${SLURM_GPUS_ON_NODE:-5}"
    if [[ ! "$gpu_count" =~ ^[1-9][0-9]*$ ]]; then
        printf 'Cannot determine GPU count; set LLM_TENSOR_SPLIT explicitly.\n' >&2
        exit 1
    fi
    split=1
    for ((index=1; index<gpu_count; index++)); do split+=,1; done
    export LLM_TENSOR_SPLIT="$split"
fi

export LLM_MODEL_ALIAS="${LLM_MODEL_ALIAS:-glm-5.3-flash}"
export LLM_CTX_SIZE="${LLM_CTX_SIZE:-131072}"
export LLM_PARALLEL="${LLM_PARALLEL:-1}"
export LLM_SPLIT_MODE="${LLM_SPLIT_MODE:-layer}"
export LLM_REASONING="${LLM_REASONING:-true}"
export LLM_REASONING_EFFORT="${LLM_REASONING_EFFORT:-high}"
case "$LLM_REASONING_EFFORT" in
    low|high|max) ;;
    *) printf 'GLM reasoning effort must be low, high, or max.\n' >&2; exit 1 ;;
esac
export LLM_TEMPERATURE="${LLM_TEMPERATURE:-1.0}"
export LLM_TOP_P="${LLM_TOP_P:-0.95}"
# Apply the documented temperature/top-p without Qwen's top-k restriction.
export LLM_TOP_K="${LLM_TOP_K:-0}"
export LLM_MIN_P="${LLM_MIN_P:-0}"
export LLM_PRESENCE_PENALTY="${LLM_PRESENCE_PENALTY:-0.0}"

# Only this file's #SBATCH directives apply to the shared serving command.
exec bash slurm/serve.sbatch

#!/usr/bin/env bash
#SBATCH --job-name=qwen-api
#SBATCH --partition=main
#SBATCH --nodelist=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_pro_5000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/qwen-%j.log
#SBATCH --error=logs/qwen-%j.log

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_SUBMIT_DIR:-}" ]]; then
    printf 'Submit with sbatch slurm/serve.sbatch from the project directory.\n' >&2
    exit 1
fi
cd "$SLURM_SUBMIT_DIR"
project_dir="$PWD"

if [[ -z "${LLAMA_SERVER:-}" ]]; then
    if [[ ! -r .tools/llama-cpp/server.path ]]; then
        printf 'Run bash scripts/prepare.sh first, or set LLAMA_SERVER.\n' >&2
        exit 1
    fi
    LLAMA_SERVER="$(< .tools/llama-cpp/server.path)"
fi
if [[ -r .tools/llama-cpp/library-path.txt ]]; then
    library_path="$(< .tools/llama-cpp/library-path.txt)"
    export LD_LIBRARY_PATH="$library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

model_file="${LLM_MODEL_FILE:-$project_dir/models/Qwen3.8-27B-UD-Q4_K_XL.gguf}"
key_file="${LLM_API_KEY_FILE:-$project_dir/.secrets/llm-api-key}"
port="${LLM_PORT:-8000}"
listen_host="${LLM_HOST:-0.0.0.0}"
context_size="${LLM_CTX_SIZE:-32768}"
parallel_slots="${LLM_PARALLEL:-1}"
model_alias="${LLM_MODEL_ALIAS:-qwen3.8-27b}"
reasoning="${LLM_REASONING:-false}"

if [[ ! -x "$LLAMA_SERVER" || ! -r "$model_file" ]]; then
    printf 'Cannot execute llama-server (%s) or read the model (%s).\n' "$LLAMA_SERVER" "$model_file" >&2
    printf 'Run the preparation script for the selected model.\n' >&2
    exit 1
fi
if [[ ! -s "$key_file" ]] || ! awk 'NF && $0 !~ /^#/ { found=1 } END { exit !found }' "$key_file"; then
    printf 'A nonempty API key file is required: %s\n' "$key_file" >&2
    exit 1
fi
case "$reasoning" in
    false) reasoning_mode=off; temperature=0.7; top_p=0.8; presence_penalty=1.5 ;;
    true) reasoning_mode=on; temperature=1.0; top_p=0.95; presence_penalty=0.0 ;;
    *) printf 'LLM_REASONING must be true or false.\n' >&2; exit 1 ;;
esac
temperature="${LLM_TEMPERATURE:-$temperature}"
top_p="${LLM_TOP_P:-$top_p}"
presence_penalty="${LLM_PRESENCE_PENALTY:-$presence_penalty}"
reasoning_effort="${LLM_REASONING_EFFORT:-medium}"
printf "Reasoning is %s; and thus temperature=%s, top_p=%s, presence_penalty=%s\n" "$reasoning" "$temperature" "$top_p" "$presence_penalty"

extra_args=()
if [[ -n "${LLM_SPLIT_MODE:-}" ]]; then
    extra_args+=(--split-mode "$LLM_SPLIT_MODE")
fi
if [[ -n "${LLM_TENSOR_SPLIT:-}" ]]; then
    extra_args+=(--tensor-split "$LLM_TENSOR_SPLIT")
fi
if [[ -n "${LLM_LAZY_MODE:-}" ]]; then
    extra_args+=(--lazy-mode "$LLM_LAZY_MODE")
fi

# Compute-node SSH names differ from Slurm node names on this cluster.
compute_host="$(hostname -f)"
printf 'Job %s; host %s; model %s; context %s; slots %s\n' \
    "$SLURM_JOB_ID" "$compute_host" "$model_alias" "$context_size" "$parallel_slots"
if [[ "$listen_host" == 0.0.0.0 ]]; then
    printf 'On your laptop (replace HEADNODE with your SSH login host):\n'
    printf 'ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:%s:%s:%s %s@HEADNODE\n' \
        "$port" "$compute_host" "$port" "$USER"
else
    printf 'Loopback mode: connect to this compute host with SSH ProxyJump; see README.md.\n'
fi
printf 'The API is ready when /health returns HTTP 200.\n'
srun nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

exec srun "$LLAMA_SERVER" \
    --model "$model_file" \
    --alias "$model_alias" \
    --host "$listen_host" --port "$port" \
    --api-key-file "$key_file" \
    --n-gpu-layers "${LLM_GPU_LAYERS:-99}" \
    --threads "$SLURM_CPUS_PER_TASK" \
    --ctx-size "$context_size" --parallel "$parallel_slots" \
    --cache-type-k bf16 --cache-type-v bf16 \
    --flash-attn on --jinja \
    --reasoning "$reasoning_mode" \
    --reasoning-effort "$reasoning_effort" \
    --temp "$temperature" --top-p "$top_p" --top-k "${LLM_TOP_K:-20}" --min-p "${LLM_MIN_P:-0}" \
    --presence-penalty "$presence_penalty" --repeat-penalty 1.0 \
    "${extra_args[@]}"

#!/usr/bin/env bash
# Run from a script-backend profile. Install vLLM in a shared environment first.
set -euo pipefail
: "${LLM_MODEL_REPO:?Set model.repo in the profile}"
: "${LLM_API_KEY_FILE:?Missing API key file}"
export VLLM_API_KEY="$(< "$LLM_API_KEY_FILE")"
# Add model-specific options (such as a tool parser) in server.extra_args.
exec "${VLLM_BIN:-vllm}" serve "$LLM_MODEL_REPO" \
    --served-model-name "$LLM_MODEL_ALIAS" \
    --host "$LLM_HOST" --port "$LLM_PORT" \
    --max-model-len "$LLM_CONTEXT_PER_REQUEST" \
    --max-num-seqs "$LLM_PARALLEL" \
    --tensor-parallel-size "$LLM_GPUS" \
    "$@"

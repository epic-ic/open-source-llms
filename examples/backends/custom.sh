#!/usr/bin/env bash
# Template for a script-backend profile; adapt the exec line to your serving engine.
set -euo pipefail
: "${SERVER_BIN:?Set SERVER_BIN to the serving engine executable}"
: "${LLM_API_KEY_FILE:?Missing API key file}"
# Hand the key over in the environment or a file, never on the command line.
export SERVER_API_KEY="$(< "$LLM_API_KEY_FILE")"
exec "$SERVER_BIN" \
    --model "${LLM_MODEL_FILE:-$LLM_MODEL_REPO}" \
    --served-model-name "$LLM_MODEL_ALIAS" \
    --host "$LLM_HOST" --port "$LLM_PORT" \
    --max-context "$LLM_CONTEXT_PER_REQUEST" \
    --max-concurrency "$LLM_PARALLEL" \
    --gpus "$LLM_GPUS" \
    "$@"

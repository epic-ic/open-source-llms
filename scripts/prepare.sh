#!/usr/bin/env bash
# Optional convenience wrapper; model selection is required.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${LLM_PYTHON:-python3}" "$project_dir/main.py" --root "$project_dir" prepare "$@"

#!/usr/bin/env bash
# Optional convenience wrapper; model selection is required.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
exec uv run llm-prepare --root "$project_dir" "$@"

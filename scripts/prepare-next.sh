#!/usr/bin/env bash
# Download only the chosen Flash-Next quant. Defaults to about 111.3 GB.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
quant="${1:-UD-Q4_K_XL}"
if [[ ! "$quant" =~ ^[A-Za-z0-9_-]+$ ]]; then
    printf 'Quant must be a repository directory name, for example UD-Q4_K_XL.\n' >&2
    exit 1
fi
if [[ ! -r .tools/llama-cpp/server.path ]]; then
    printf 'Prepare the existing CUDA runtime with bash scripts/prepare.sh first.\n' >&2
    exit 1
fi
model_dir="$project_dir/models/Qwen3.8-Flash-Next-GGUF"
mkdir -p "$model_dir" .tools/qwen-next logs
export HF_HOME="${HF_HOME:-$project_dir/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$project_dir/.cache/uv}"

uvx --from huggingface_hub hf download unsloth/Qwen3.8-Flash-Next-GGUF \
    --local-dir "$model_dir" --include "$quant/*.gguf"

# A split GGUF must have every shard. llama-server opens it through shard 00001.
python3 - "$model_dir/$quant" "$project_dir/.tools/qwen-next/model.path" <<'PY'
from pathlib import Path
import re
import sys

directory, metadata = map(Path, sys.argv[1:])
first_shards = sorted(directory.glob("*-00001-of-*.gguf"))
if len(first_shards) != 1:
    raise SystemExit(f"Expected exactly one first GGUF shard in {directory}")
first = first_shards[0]
match = re.fullmatch(r"(.+)-00001-of-(\d{5})\.gguf", first.name)
if not match:
    raise SystemExit(f"Unexpected shard filename: {first.name}")
prefix, count_text = match.groups()
count = int(count_text)
if count < 1:
    raise SystemExit("Invalid shard count")
for index in range(1, count + 1):
    shard = directory / f"{prefix}-{index:05d}-of-{count:05d}.gguf"
    if not shard.is_file() or shard.stat().st_size == 0:
        raise SystemExit(f"Missing or empty GGUF shard: {shard}")
metadata.write_text(str(first) + "\n")
print(f"Prepared {count} shards; model entry point: {first}")
PY
printf 'Submit from this directory: sbatch slurm/serve-next.sbatch\n'

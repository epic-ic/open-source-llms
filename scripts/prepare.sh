#!/usr/bin/env bash
# Downloads about 18.4 GB; requires about 25-30 GB of free shared storage.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
mkdir -p .tools/downloads .tools/llama-cpp models logs .secrets
chmod 700 .secrets

# Pin the binary and CUDA runtime together. Neither root nor nvcc is required.
release=b11472
release_url="https://github.com/ggml-org/llama.cpp/releases/download/$release"
install_dir="$project_dir/.tools/llama-cpp/$release-cuda12.8"
mkdir -p "$install_dir"

install_archive() {
    local name="$1" checksum="$2" archive="$project_dir/.tools/downloads/$1"
    if ! printf '%s  %s\n' "$checksum" "$archive" | sha256sum --check --status 2>/dev/null; then
        curl --fail --location --retry 3 --continue-at - \
            "$release_url/$name" --output "$archive"
        printf '%s  %s\n' "$checksum" "$archive" | sha256sum --check
    fi
    python3 - "$archive" "$install_dir" <<'PY'
import sys
import tarfile

with tarfile.open(sys.argv[1]) as archive:
    archive.extractall(sys.argv[2], filter="data")
PY
}

install_archive "llama-$release-bin-ubuntu-cuda-12.8-x64.tar.gz" \
    8a310ed6b6d50d84947477026819da69ec07d13e6607bf0dc26523ba55817aba
install_archive "cudart-llama-$release-bin-ubuntu-cuda-12.8-x64.tar.gz" \
    7713c7c7a74cf63e8e55e180aeb04b186d105f94b60ac3261cde0105cf1c7e2c

# Archive layouts can vary. Record the executable and all shared library paths.
python3 - "$install_dir" "$project_dir" <<'PY'
import os
from pathlib import Path
import secrets
import sys

install_dir, project_dir = map(Path, sys.argv[1:])
servers = [path for path in install_dir.rglob("llama-server") if path.is_file()]
if len(servers) != 1 or not os.access(servers[0], os.X_OK):
    raise SystemExit("Expected one executable llama-server in the release archive")
metadata_dir = project_dir / ".tools" / "llama-cpp"
(metadata_dir / "server.path").write_text(str(servers[0]) + "\n")
libraries = sorted({str(path.parent) for path in install_dir.rglob("*.so*") if path.is_file()})
(metadata_dir / "library-path.txt").write_text(os.pathsep.join(libraries) + "\n")
key_file = project_dir / ".secrets" / "llm-api-key"
try:
    descriptor = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
except FileExistsError:
    pass
else:
    with os.fdopen(descriptor, "w") as output:
        output.write(secrets.token_hex(32) + "\n")
key_file.chmod(0o600)
PY

model_name=Qwen3.8-27B-UD-Q4_K_XL.gguf
if [[ ! -s "models/$model_name" ]]; then
    curl --fail --location --retry 3 --continue-at - \
        "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/$model_name" \
        --output "models/$model_name.part"
    mv "models/$model_name.part" "models/$model_name"
fi
printf 'Prepared %s\nSubmit from this directory: sbatch slurm/serve.sbatch\n' "$model_name"

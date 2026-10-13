"""Install or reuse the pinned llama.cpp runtime independently of model weights."""

from __future__ import annotations

import hashlib
import os
import platform
import tarfile
from pathlib import Path

from .config import ConfigError, read_toml
from .process import atomic_write, resolve_path, run


def runtime_command(root: Path, override: str | None = None) -> tuple[Path, dict[str, str]]:
    env = dict(os.environ)
    metadata = root / ".tools" / "llama-cpp"
    executable = override or os.getenv("LLAMA_SERVER")
    if not executable:
        path = metadata / "server.path"
        if not path.is_file():
            raise ConfigError("Prepare the runtime with 'uv run llm-prepare PROFILE' or set LLAMA_SERVER")
        executable = path.read_text().strip()
    server = resolve_path(root, executable)
    if not server.is_file() or not os.access(server, os.X_OK):
        raise ConfigError(f"Cannot execute llama-server: {server}")
    recorded = metadata / "server.path"
    libraries = metadata / "library-path.txt"
    if recorded.is_file() and libraries.is_file() and resolve_path(root, recorded.read_text().strip()) == server:
        prefix = libraries.read_text().strip()
        env["LD_LIBRARY_PATH"] = prefix + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return server, env


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def prepare_runtime(root: Path) -> None:
    if os.getenv("LLAMA_SERVER"):
        runtime_command(root)
        print("Using LLAMA_SERVER")
        return
    settings = read_toml(root / "config" / "runtime.toml")
    release = settings["release"]
    install = root / ".tools" / "llama-cpp" / settings["install_directory"]
    # Existing installs, including those prepared by the previous scripts, remain usable.
    try:
        server, _ = runtime_command(root)
    except ConfigError:
        pass
    else:
        if server.is_relative_to(install):
            print(f"Runtime already prepared: {server}")
            return
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise ConfigError("Pinned binaries require Linux x86_64; provide your own LLAMA_SERVER")
    archives = settings.get("archives", [])
    if not archives:
        raise ConfigError("runtime.toml requires at least one archive")
    for archive in archives:
        if not isinstance(archive.get("name"), str) or Path(archive["name"]).name != archive["name"]:
            raise ConfigError("Runtime archive names must be filenames")
        digest = archive.get("sha256", "")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ConfigError("Runtime archives require a SHA-256 digest")
        destination = root / ".tools" / "downloads" / archive["name"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file() or sha256(destination) != digest:
            run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--retry",
                    "3",
                    "--continue-at",
                    "-",
                    f"https://github.com/ggml-org/llama.cpp/releases/download/{release}/{archive['name']}",
                    "--output",
                    str(destination),
                ]
            )
        if sha256(destination) != digest:
            raise ConfigError(f"SHA-256 mismatch: {destination}; remove the archive and retry")
        install.mkdir(parents=True, exist_ok=True)
        with tarfile.open(destination) as source:
            source.extractall(install, filter="data")
    servers = [path for path in install.rglob("llama-server") if path.is_file()]
    if len(servers) != 1 or not os.access(servers[0], os.X_OK):
        raise ConfigError("Expected exactly one executable llama-server in the runtime archives")
    metadata = root / ".tools" / "llama-cpp"
    libraries = sorted({str(path.parent) for path in install.rglob("*.so*") if path.is_file()})
    atomic_write(metadata / "server.path", str(servers[0]) + "\n")
    atomic_write(metadata / "library-path.txt", os.pathsep.join(libraries) + "\n")
    print(f"Prepared runtime: {servers[0]}")

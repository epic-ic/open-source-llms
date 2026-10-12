"""Download selected Hugging Face files and validate GGUF entry points."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from .config import ConfigError, Profile
from .process import atomic_write, resolve_path, run

SHARD = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf")


def validate_gguf(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise ConfigError(f"Missing or empty GGUF: {path}")
    match = SHARD.fullmatch(path.name)
    if match:
        prefix, first, count = match.groups()
        if int(first) != 1 or int(count) < 1:
            raise ConfigError("A split GGUF entry point must be shard 00001 with a positive count")
        for index in range(1, int(count) + 1):
            shard = path.parent / f"{prefix}-{index:05d}-of-{count}.gguf"
            if not shard.is_file() or shard.stat().st_size == 0:
                raise ConfigError(f"Missing or empty GGUF shard: {shard}")
    return path


def state_path(root: Path, profile: Profile) -> Path:
    return root / ".tools" / "profiles" / profile.id / "model.path"


def download_entry(root: Path, profile: Profile, quant: str | None = None) -> Path:
    directory = resolve_path(root, profile.format_download(profile.download["directory"], quant))
    pattern = profile.format_download(profile.download["entrypoint"], quant)
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ConfigError(f"Expected exactly one model entry point matching {directory / pattern}")
    return validate_gguf(matches[0])


def model_path(root: Path, profile: Profile, override: str | None = None) -> Path:
    if override or profile.model.get("file"):
        return validate_gguf(resolve_path(root, override or profile.model["file"]))
    state = state_path(root, profile)
    if state.is_file():
        return validate_gguf(resolve_path(root, state.read_text().strip()))
    # Reuse existing downloads even if they were made with the old scripts.
    return download_entry(root, profile)


def hf_command() -> list[str]:
    if shutil.which("hf"):
        return ["hf"]
    if shutil.which("uvx"):
        return ["uvx", "--from", "huggingface_hub", "hf"]
    raise ConfigError("Install uv to run the Hugging Face CLI through uvx")


def prepare_weights(root: Path, profile: Profile, quant: str | None = None) -> Path:
    if profile.model.get("file"):
        if quant:
            raise ConfigError("--quant requires a download profile without model.file")
        entry = model_path(root, profile)
    else:
        download = profile.download
        if not download:
            raise ConfigError("This profile has no download section")
        # Validate all substitutions before invoking the downloader.
        directory = resolve_path(root, profile.format_download(download["directory"], quant))
        patterns = [profile.format_download(pattern, quant) for pattern in download["include"]]
        profile.format_download(download["entrypoint"], quant)
        directory.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.setdefault("HF_HOME", str(root / ".cache" / "huggingface"))
        env.setdefault("UV_CACHE_DIR", str(root / ".cache" / "uv"))
        run(
            hf_command()
            + [
                "download",
                download["repo"],
                "--revision",
                download.get("revision", "main"),
                "--local-dir",
                str(directory),
                "--include",
                *patterns,
            ],
            env=env,
            cwd=root,
        )
        entry = download_entry(root, profile, quant)
    atomic_write(state_path(root, profile), str(entry) + "\n")
    print(f"Prepared model: {entry}")
    return entry

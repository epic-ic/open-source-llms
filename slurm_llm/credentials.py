"""Share one authentication convention across servers, clients, and harnesses."""

import os
import secrets
from pathlib import Path

from .config import ConfigError
from .process import resolve_path


def key_path(root: Path, value: str | None = None) -> Path:
    return resolve_path(root, value or os.getenv("LLM_API_KEY_FILE", ".secrets/llm-api-key"))


def read_key(path: Path) -> str:
    try:
        key = path.read_text().strip()
    except OSError as exc:
        raise ConfigError(f"Cannot read API key file {path}: {exc}") from exc
    return validate_key(key)


def validate_key(key: str) -> str:
    if not key or any(character.isspace() for character in key):
        raise ConfigError("API key must contain one nonempty token")
    return key


def ensure_key(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.name == ".secrets":
        path.parent.chmod(0o700)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        read_key(path)
    else:
        with os.fdopen(descriptor, "w") as output:
            output.write(secrets.token_hex(32) + "\n")
    path.chmod(0o600)


def client_key(root: Path, value: str | None = None) -> str:
    if value:
        return read_key(key_path(root, value))
    if os.getenv("LLM_API_KEY"):
        return validate_key(os.environ["LLM_API_KEY"].strip())
    return read_key(key_path(root))

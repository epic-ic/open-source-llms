"""Generate harness configuration and SSH commands from the selected profile."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from .client import validate_base_url
from .config import ConfigError, Profile


def provider_id(profile: Profile) -> str:
    return f"slurm-{profile.id}"


def codex_profile(profile: Profile) -> str:
    return f"slurm_{profile.id.replace('-', '_')}"


def opencode(profile: Profile, base_url: str) -> str:
    """Generate an OpenCode v2 configuration."""
    provider = provider_id(profile)
    alias = profile.model["alias"]
    model = {
        "name": profile.model["name"],
        "limit": {
            "context": profile.context_per_request,
            "output": profile.server["max_output"],
        },
        "capabilities": {
            "tools": profile.capabilities["tools"],
            "input": ["text"],
            "output": ["text"],
        },
    }
    if "reasoning_efforts" in profile.capabilities:
        model["variants"] = [
            {"id": effort, "settings": {"reasoningEffort": effort}}
            for effort in profile.capabilities["reasoning_efforts"]
        ]
    config: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "model": f"{provider}/{alias}",
        "providers": {
            provider: {
                "name": profile.model["name"] + " on Slurm",
                "env": ["LLM_API_KEY"],
                "package": "@opencode/ai/providers/openai-compatible",
                "settings": {"baseURL": validate_base_url(base_url)},
                "models": {alias: model},
            },
        },
    }
    return json.dumps(config, indent=2) + "\n"


def codex(profile: Profile, base_url: str) -> str:
    if not profile.capabilities["responses"] or not profile.capabilities["tools"]:
        raise ConfigError("Codex needs tools and Responses support; verify both before enabling capabilities")
    provider = codex_profile(profile)
    quote = json.dumps
    context = profile.context_per_request
    compact = min(int(context * 0.75), context - profile.server["max_output"])
    return f'''# Step 1: save the keys below as ~/.codex/{provider}.config.toml. Codex 0.134
# and later read --profile settings from that file, not from a [profiles] table
# in config.toml.
model_provider = "{provider}"
model = {quote(profile.model["alias"])}
model_context_window = {context}
model_auto_compact_token_limit = {compact}

# Step 2: merge the table below into ~/.codex/config.toml.
# Then start with: codex --profile {provider}
[model_providers.{provider}]
name = {quote(profile.model["name"] + " on Slurm")}
base_url = {quote(validate_base_url(base_url))}
env_key = "LLM_API_KEY"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
stream_idle_timeout_ms = 300000
'''


def shell_environment(profile: Profile, base_url: str, key_file: Path) -> str:
    # Emit commands, never credentials. This can be sourced after review.
    return (
        f"export LLM_BASE_URL={shlex.quote(validate_base_url(base_url))}\n"
        f"export LLM_MODEL={shlex.quote(profile.model['alias'])}\n"
        f"export LLM_API_KEY_FILE={shlex.quote(str(key_file))}\n"
        'export LLM_API_KEY="$(cat -- "$LLM_API_KEY_FILE")"\n'
    )


def tunnel_command(
    profile: Profile,
    *,
    login: str,
    compute_host: str,
    local_port: int | None = None,
    jump: bool = False,
) -> list[str]:
    if not login or not compute_host:
        raise ConfigError("Set SSH login and compute host in the cluster config or pass flags")
    for label, value in (("login", login), ("compute host", compute_host)):
        if value.startswith("-") or any(char.isspace() for char in value):
            raise ConfigError(f"Invalid SSH {label}")
    port = local_port or profile.server["port"]
    if type(port) is not int or not 1 <= port <= 65535:
        raise ConfigError("Local port must be between 1 and 65535")
    command = [
        "ssh",
        "-N",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
    ]
    if jump or profile.server["host"] in {"127.0.0.1", "localhost", "::1"}:
        user = login.rsplit("@", 1)[0] if "@" in login else ""
        target = compute_host if "@" in compute_host or not user else f"{user}@{compute_host}"
        command += [
            "-J",
            login,
            "-L",
            f"127.0.0.1:{port}:127.0.0.1:{profile.server['port']}",
            target,
        ]
    else:
        destination = compute_host.rsplit("@", 1)[-1]
        command += ["-L", f"127.0.0.1:{port}:{destination}:{profile.server['port']}", login]
    return command

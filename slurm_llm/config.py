"""Load and validate model profiles and cluster settings."""

from __future__ import annotations

import copy
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Configuration that cannot be used consistently."""


SECTIONS = {
    "model": {"name", "backend", "alias", "file", "repo", "quant"},
    "download": {"repo", "revision", "directory", "include", "entrypoint"},
    "server": {
        "host",
        "port",
        "context_size",
        "parallel",
        "max_output",
        "gpu_layers",
        "reasoning",
        "reasoning_effort",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "repeat_penalty",
        "split_mode",
        "tensor_split",
        "lazy_mode",
        "cache_type_k",
        "cache_type_v",
        "flash_attn",
        "extra_args",
    },
    "slurm": {
        "gpus",
        "cpus_per_task",
        "mem",
        "time",
        "partition",
        "account",
        "nodelist",
        "constraint",
        "gpu_type",
        "qos",
    },
    "capabilities": {"tools", "responses", "reasoning_efforts"},
    "scripts": {"prepare", "serve"},
}
SERVER_DEFAULTS = {
    "host": "0.0.0.0",
    "port": 8000,
    "context_size": 8192,
    "parallel": 1,
    "max_output": 2048,
    "gpu_layers": 99,
    "extra_args": [],
}
SLURM_DEFAULTS = {"gpus": 1, "cpus_per_task": 8, "mem": "64G", "time": "08:00:00"}
SERVER_ENV = {
    "LLM_HOST": ("host", str),
    "LLM_PORT": ("port", int),
    "LLM_CTX_SIZE": ("context_size", int),
    "LLM_PARALLEL": ("parallel", int),
    "LLM_MAX_OUTPUT": ("max_output", int),
    "LLM_GPU_LAYERS": ("gpu_layers", int),
    "LLM_REASONING": ("reasoning", str),
    "LLM_REASONING_EFFORT": ("reasoning_effort", str),
    "LLM_TEMPERATURE": ("temperature", float),
    "LLM_TOP_P": ("top_p", float),
    "LLM_TOP_K": ("top_k", int),
    "LLM_MIN_P": ("min_p", float),
    "LLM_PRESENCE_PENALTY": ("presence_penalty", float),
    "LLM_SPLIT_MODE": ("split_mode", str),
    "LLM_TENSOR_SPLIT": ("tensor_split", str),
    "LLM_LAZY_MODE": ("lazy_mode", str),
    "LLM_MODEL_ALIAS": ("alias", str),
}


def project_root(value: str | Path | None = None) -> Path:
    if value or os.environ.get("LLM_ROOT"):
        return Path(value or os.environ["LLM_ROOT"]).expanduser().resolve()
    for directory in [Path.cwd(), *Path.cwd().parents]:
        if (directory / "config" / "models").is_dir():
            return directory
    checkout = Path(__file__).resolve().parent.parent
    return checkout if (checkout / "config" / "models").is_dir() else Path.cwd()


def read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as source:
            return tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc


def valid_id(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value):
        raise ConfigError("Profile ID must use lowercase letters, digits, hyphens, or underscores")
    return value


def positive(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ConfigError(f"{label} must be a positive integer")
    return value


def strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ConfigError(f"{label} must be an array of strings")
    return value


def relative_path(value: str, label: str) -> str:
    if not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ConfigError(f"{label} must be a nonempty path relative to the repository")
    return value


@dataclass
class Profile:
    id: str
    data: dict[str, Any]

    def __post_init__(self) -> None:
        valid_id(self.id)
        if not isinstance(self.data, dict):
            raise ConfigError("A model profile must contain configuration tables")
        self.data = copy.deepcopy(self.data)
        for section, values in self.data.items():
            if section not in SECTIONS or not isinstance(values, dict):
                raise ConfigError(f"Unknown or invalid section: {section}")
            unknown = values.keys() - SECTIONS[section]
            if unknown:
                raise ConfigError(f"Unknown {section} setting(s): {', '.join(sorted(unknown))}")
        model = self.data.setdefault("model", {})
        model.setdefault("name", self.id)
        model.setdefault("alias", self.id)
        model.setdefault("backend", "llama.cpp")
        if model["backend"] not in ("llama.cpp", "script"):
            raise ConfigError("model.backend must be 'llama.cpp' or 'script'")
        for key, value in model.items():
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"model.{key} must be a nonempty string")
        self.data["server"] = SERVER_DEFAULTS | self.data.get("server", {})
        self.data["slurm"] = SLURM_DEFAULTS | self.data.get("slurm", {})
        self.data["capabilities"] = {"tools": False, "responses": False} | self.data.get(
            "capabilities", {}
        )
        self.data.setdefault("scripts", {})
        server = self.server
        for key in ("port", "context_size", "parallel", "max_output"):
            positive(server[key], f"server.{key}")
        if server["port"] > 65535:
            raise ConfigError("server.port must be at most 65535")
        if server["context_size"] // server["parallel"] <= server["max_output"]:
            raise ConfigError("max_output must be less than context_size / parallel")
        strings(server["extra_args"], "server.extra_args")
        if "reasoning" in server and server["reasoning"] not in ("on", "off", "auto"):
            raise ConfigError("server.reasoning must be 'on', 'off', or 'auto'")
        for key in (
            "host",
            "reasoning_effort",
            "split_mode",
            "tensor_split",
            "lazy_mode",
            "cache_type_k",
            "cache_type_v",
            "flash_attn",
        ):
            if key in server and (not isinstance(server[key], str) or not server[key].strip()):
                raise ConfigError(f"server.{key} must be a nonempty string")
        for key in ("gpu_layers", "top_k"):
            if key in server and (type(server[key]) is not int or server[key] < 0):
                raise ConfigError(f"server.{key} must be a nonnegative integer")
        for key in ("temperature", "top_p", "min_p", "presence_penalty", "repeat_penalty"):
            if key in server and type(server[key]) not in {int, float}:
                raise ConfigError(f"server.{key} must be a number")
        for key in ("gpus", "cpus_per_task"):
            positive(self.slurm[key], f"slurm.{key}")
        for key, value in self.slurm.items():
            if key not in {"gpus", "cpus_per_task"} and (
                not isinstance(value, str) or not value.strip() or "\n" in value
            ):
                raise ConfigError(f"slurm.{key} must be a nonempty string without newlines")
        for key in ("tools", "responses"):
            if type(self.capabilities[key]) is not bool:
                raise ConfigError(f"capabilities.{key} must be true or false")
        if "reasoning_efforts" in self.capabilities:
            efforts = strings(
                self.capabilities["reasoning_efforts"], "capabilities.reasoning_efforts"
            )
            if any(not effort.strip() for effort in efforts) or len(set(efforts)) != len(efforts):
                raise ConfigError("capabilities.reasoning_efforts must be unique nonempty strings")
        for key, value in self.scripts.items():
            if not isinstance(value, str):
                raise ConfigError(f"scripts.{key} must be a path string")
            relative_path(value, f"scripts.{key}")
        if self.backend == "script" and "serve" not in self.scripts:
            raise ConfigError("A script backend requires scripts.serve")
        if self.backend == "llama.cpp" and "file" not in model and not self.download:
            raise ConfigError("A llama.cpp profile requires model.file or a download section")
        if self.download:
            for key in ("repo", "directory", "entrypoint"):
                if not isinstance(self.download.get(key), str) or not self.download[key]:
                    raise ConfigError(f"download.{key} is required")
            for key in ("directory", "entrypoint"):
                relative_path(self.download[key], f"download.{key}")
            if not self.download.get("include"):
                raise ConfigError("download.include must select at least one file")
            strings(self.download["include"], "download.include")
            for pattern in self.download["include"]:
                relative_path(pattern, "download.include")
            if "revision" in self.download and not isinstance(self.download["revision"], str):
                raise ConfigError("download.revision must be a string")

    @property
    def model(self) -> dict[str, Any]:
        return self.data["model"]

    @property
    def backend(self) -> str:
        return self.model["backend"]

    @property
    def server(self) -> dict[str, Any]:
        return self.data["server"]

    @property
    def slurm(self) -> dict[str, Any]:
        return self.data["slurm"]

    @property
    def capabilities(self) -> dict[str, Any]:
        return self.data["capabilities"]

    @property
    def scripts(self) -> dict[str, str]:
        return self.data["scripts"]

    @property
    def download(self) -> dict[str, Any]:
        return self.data.get("download", {})

    @property
    def context_per_request(self) -> int:
        return self.server["context_size"] // self.server["parallel"]

    def format_download(self, value: str, quant: str | None = None) -> str:
        quant = quant or self.model.get("quant", "")
        if quant and not re.fullmatch(r"[A-Za-z0-9_-]+", quant):
            raise ConfigError("Quant must be a single repository directory or quant name")
        if "{quant}" in value and not quant:
            raise ConfigError("Set model.quant or pass --quant")
        return value.replace("{quant}", quant)


def load_profile(root: Path, name: str) -> Profile:
    path = Path(name).expanduser()
    if path.suffix == ".toml":
        path = path if path.is_absolute() else root / path
        ident = path.stem
    else:
        ident = valid_id(name)
        path = root / "config" / "models" / f"{ident}.toml"
    return Profile(ident, read_toml(path))


def load_cluster(root: Path, path: str | None = None) -> dict[str, Any]:
    cluster_path = Path(path).expanduser() if path else root / "config" / "cluster.local.toml"
    if not cluster_path.is_absolute():
        cluster_path = root / cluster_path
    data = read_toml(cluster_path) if path or cluster_path.exists() else {}
    if data.keys() - {"slurm", "ssh"}:
        raise ConfigError("Cluster configuration only supports [slurm] and [ssh]")
    for section, allowed in (("slurm", SECTIONS["slurm"]), ("ssh", {"login", "compute_host"})):
        values = data.get(section, {})
        if not isinstance(values, dict) or values.keys() - allowed:
            raise ConfigError(f"Invalid cluster [{section}] settings")
        if section == "ssh" and any(not isinstance(value, str) for value in values.values()):
            raise ConfigError("Cluster SSH settings must be strings")
    return data


def effective_profile(
    profile: Profile,
    *,
    server: dict[str, Any] | None = None,
    slurm: dict[str, Any] | None = None,
    cluster: dict[str, Any] | None = None,
    use_env: bool = True,
) -> Profile:
    data = copy.deepcopy(profile.data)
    if use_env:
        for variable, (key, convert) in SERVER_ENV.items():
            if variable not in os.environ:
                continue
            try:
                value = convert(os.environ[variable])
            except ValueError as exc:
                raise ConfigError(f"Invalid {variable}: {exc}") from exc
            if key == "alias":
                data["model"][key] = value
            else:
                if key == "reasoning":
                    value = {"true": "on", "false": "off"}.get(value, value)
                data["server"][key] = value
        if os.environ.get("LLM_MODEL_FILE"):
            data["model"]["file"] = os.environ["LLM_MODEL_FILE"]
    # Cluster values are local policy; flags take precedence over all configuration.
    data["slurm"].update((cluster or {}).get("slurm", {}))
    data["slurm"].update(slurm or {})
    data["server"].update(server or {})
    return Profile(profile.id, data)

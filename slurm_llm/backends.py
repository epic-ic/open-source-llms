"""Backend adapters: GGUF with llama.cpp, Hugging Face checkpoints with vLLM, or a script."""

from __future__ import annotations

import os
from pathlib import Path

from . import vllm
from .config import ConfigError, Profile
from .credentials import ensure_key, read_key
from .process import run
from .runtime import prepare_runtime, runtime_command
from .weights import model_path, prepare_weights

LLAMA_FLAGS = {
    "gpu_layers": "--n-gpu-layers",
    "reasoning": "--reasoning",
    "reasoning_effort": "--reasoning-effort",
    "temperature": "--temp",
    "top_p": "--top-p",
    "top_k": "--top-k",
    "min_p": "--min-p",
    "presence_penalty": "--presence-penalty",
    "repeat_penalty": "--repeat-penalty",
    "split_mode": "--split-mode",
    "tensor_split": "--tensor-split",
    "lazy_mode": "--lazy-mode",
    "cache_type_k": "--cache-type-k",
    "cache_type_v": "--cache-type-v",
    "flash_attn": "--flash-attn",
}


def script_environment(root: Path, profile: Profile, key_file: Path) -> dict[str, str]:
    env = dict(os.environ)
    values = {
        "LLM_ROOT": root,
        "LLM_PROFILE": profile.id,
        "LLM_MODEL_ALIAS": profile.model["alias"],
        "LLM_MODEL_REPO": profile.model.get("repo", ""),
        "LLM_MODEL_FILE": profile.model.get("file", ""),
        "LLM_API_KEY_FILE": key_file,
        "LLM_HOST": profile.server["host"],
        "LLM_PORT": profile.server["port"],
        "LLM_CTX_SIZE": profile.server["context_size"],
        "LLM_PARALLEL": profile.server["parallel"],
        "LLM_MAX_OUTPUT": profile.server["max_output"],
        "LLM_GPUS": profile.slurm["gpus"],
        "LLM_CONTEXT_PER_REQUEST": profile.context_per_request,
        "LLM_CPUS": profile.slurm["cpus_per_task"],
        "HF_HOME": env.get("HF_HOME", root / ".cache" / "huggingface"),
    }
    env.update({key: str(value) for key, value in values.items()})
    return env


def script_path(root: Path, profile: Profile, action: str) -> Path:
    path = root / profile.scripts[action]
    if not path.is_file():
        raise ConfigError(f"Missing backend script: {path}")
    return path


def prepare(
    root: Path,
    profile: Profile,
    key_file: Path,
    *,
    quant: str | None = None,
    runtime_only: bool = False,
    model_only: bool = False,
) -> None:
    if profile.backend in ("llama.cpp", "vllm"):
        if not model_only:
            if profile.backend == "vllm":
                vllm.prepare_runtime(root)
            else:
                prepare_runtime(root)
        hub_only = profile.backend == "vllm" and not profile.download and not profile.model.get("file")
        if hub_only and not runtime_only:
            if quant:
                raise ConfigError("--quant requires a download section")
            print(f"vLLM downloads {profile.model['repo']} into HF_HOME when the job starts")
        elif not runtime_only:
            prepare_weights(root, profile, quant)
    else:
        if quant or runtime_only or model_only:
            raise ConfigError("Script profiles use their prepare script; quant/runtime flags do not apply")
        if "prepare" in profile.scripts:
            run(
                ["bash", str(script_path(root, profile, "prepare"))],
                cwd=root,
                env=script_environment(root, profile, key_file),
            )
        else:
            print("No prepare script; use an existing serving environment")
    ensure_key(key_file)
    print(f"API key file: {key_file} (contents are never logged)")


def allocation_gpus(profile: Profile) -> int:
    value = os.getenv("SLURM_GPUS_ON_NODE")
    if value is None:
        return profile.slurm["gpus"]
    if value.isdigit() and int(value) > 0:
        return int(value)
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if visible and visible not in {"-1", "NoDevFiles"}:
        return len(visible.split(","))
    raise ConfigError("Cannot determine GPU count from SLURM_GPUS_ON_NODE or CUDA_VISIBLE_DEVICES")


def server_command(
    root: Path, profile: Profile, key_file: Path, executable: str | None = None
) -> tuple[list[str], dict[str, str]]:
    if profile.backend == "vllm":
        return vllm.server_command(root, profile, key_file, allocation_gpus(profile), executable)
    read_key(key_file)
    if profile.backend == "script":
        return (
            ["bash", str(script_path(root, profile, "serve")), *profile.server["extra_args"]],
            script_environment(root, profile, key_file),
        )
    server, env = runtime_command(root, executable)
    model = model_path(root, profile)
    settings = dict(profile.server)
    if "tensor_split" not in settings:
        count = allocation_gpus(profile)
        if count > 1:
            settings["tensor_split"] = ",".join(["1"] * count)
            settings.setdefault("split_mode", "layer")
    command = [
        str(server),
        "--model",
        str(model),
        "--alias",
        profile.model["alias"],
        "--host",
        settings["host"],
        "--port",
        str(settings["port"]),
        "--api-key-file",
        str(key_file),
        "--threads",
        str(profile.slurm["cpus_per_task"]),
        "--ctx-size",
        str(settings["context_size"]),
        "--parallel",
        str(settings["parallel"]),
        "--n-predict",
        str(settings["max_output"]),
        "--jinja",
    ]
    for key, flag in LLAMA_FLAGS.items():
        if key in settings:
            command.extend([flag, str(settings[key])])
    command.extend(settings["extra_args"])
    return command, env

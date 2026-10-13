"""Install, resolve, and launch vLLM from validated profile settings."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from .config import ConfigError, Profile, read_toml
from .credentials import read_key
from .process import atomic_write, resolve_path, run
from .weights import vllm_model

VALUE_FLAGS = {
    "pipeline_parallel_size": "--pipeline-parallel-size",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "max_num_batched_tokens": "--max-num-batched-tokens",
    "dtype": "--dtype",
    "quantization": "--quantization",
    "kv_cache_dtype": "--kv-cache-dtype",
    "reasoning_parser": "--reasoning-parser",
    "generation_config": "--generation-config",
}
SWITCHES = {
    "enable_expert_parallel": "--enable-expert-parallel",
    "enforce_eager": "--enforce-eager",
    "trust_remote_code": "--trust-remote-code",
    "language_model_only": "--language-model-only",
}
JSON_FLAGS = {
    "chat_template_kwargs": "--default-chat-template-kwargs",
    "speculative_config": "--speculative-config",
    "hf_overrides": "--hf-overrides",
}
# Profile sampling keys that vLLM accepts as server defaults (--override-generation-config).
SAMPLING = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "repeat_penalty": "repetition_penalty",
}
# Directory holding only the stdlib auth middleware, put on vLLM's PYTHONPATH.
MIDDLEWARE_PATH = Path(__file__).resolve().parent / "vllm_middleware"
MIDDLEWARE = "slurm_llm_auth.BearerAuth"


def metadata_path(root: Path) -> Path:
    return root / ".tools" / "vllm" / "vllm.path"


def runtime_command(root: Path, override: str | None = None) -> tuple[Path, dict[str, str]]:
    executable = override or os.getenv("VLLM_BIN")
    if not executable:
        metadata = metadata_path(root)
        if not metadata.is_file():
            raise ConfigError("Prepare vLLM with 'uv run llm-prepare PROFILE' or set VLLM_BIN")
        executable = metadata.read_text().strip()
    if os.sep not in executable and shutil.which(executable):
        executable = shutil.which(executable) or executable
    server = resolve_path(root, executable)
    if not server.is_file() or not os.access(server, os.X_OK):
        raise ConfigError(f"Cannot execute vllm: {server}")
    env = dict(os.environ)
    # The job inherits the slurm-llm environment from submission; run vLLM in its own.
    env.pop("VIRTUAL_ENV", None)
    env.pop("VLLM_BIN", None)  # vLLM warns about VLLM_* variables it does not know.
    venv = server.parent.parent
    if (venv / "pyvenv.cfg").is_file():
        env["VIRTUAL_ENV"] = str(venv)
        paths = [str(server.parent)]
        # vLLM disables FlashInfer when nvcc is missing, since it JIT-compiles kernels
        # (e.g. sparse MLA on sm_120). The nvidia-cuda-nvcc wheel ships one in the venv.
        nvcc = next(iter(sorted(venv.glob("lib/python*/site-packages/nvidia/cu*/bin/nvcc"))), None)
        if nvcc:
            env.setdefault("CUDA_HOME", str(nvcc.parent.parent))
            paths.append(str(nvcc.parent))
        env["PATH"] = os.pathsep.join([*paths, env.get("PATH", "")])
    return server, env


def prepare_runtime(root: Path) -> None:
    if os.getenv("VLLM_BIN"):
        server, _ = runtime_command(root)
        print(f"Using VLLM_BIN: {server}")
        return
    settings = read_toml(root / "config" / "runtime.toml").get("vllm")
    if not isinstance(settings, dict) or not isinstance(settings.get("version"), str):
        raise ConfigError("config/runtime.toml requires a [vllm] table with a version")
    version = settings["version"]
    if not re.fullmatch(r"[0-9][0-9A-Za-z.+-]*", version):
        raise ConfigError("vllm.version must be a release version such as 0.30.0")
    python = settings.get("python", "3.12")
    packages = settings.get("packages", [])
    backend = settings.get("torch_backend")
    if not isinstance(python, str) or not isinstance(packages, list):
        raise ConfigError("vllm.python must be a string and vllm.packages an array")
    if any(not isinstance(item, str) or item.startswith("-") for item in packages):
        raise ConfigError("vllm.packages must be package requirements")
    if backend is not None and not isinstance(backend, str):
        raise ConfigError("vllm.torch_backend must be a string")
    install = root / ".tools" / "vllm" / version
    server = install / "bin" / "vllm"
    metadata = metadata_path(root)
    if server.is_file() and metadata.is_file() and resolve_path(root, metadata.read_text().strip()) == server.resolve():
        print(f"vLLM already prepared: {server}")
        return
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    # Keep the interpreter, venv, and wheel cache on storage that compute nodes share.
    env.setdefault("UV_CACHE_DIR", str(root / ".cache" / "uv"))
    env.setdefault("UV_PYTHON_INSTALL_DIR", str(root / ".tools" / "python"))
    if not (install / "bin" / "python").exists():
        run(["uv", "venv", "--managed-python", "--python", python, str(install)], env=env)
    command = [
        "uv",
        "pip",
        "install",
        "--python",
        str(install / "bin" / "python"),
        f"vllm=={version}",
        *packages,
    ]
    if backend:
        command.extend(["--torch-backend", backend])
    run(command, env=env, cwd=root)
    if not server.is_file() or not os.access(server, os.X_OK):
        raise ConfigError(f"vLLM installation did not provide {server}")
    atomic_write(metadata, str(server) + "\n")
    print(f"Prepared vLLM {version}: {server}")


def parallel_sizes(profile: Profile, gpus: int) -> tuple[int, int]:
    pipeline = profile.vllm.get("pipeline_parallel_size", 1)
    tensor = profile.vllm.get("tensor_parallel_size", max(gpus // pipeline, 1))
    if tensor * pipeline > gpus:
        raise ConfigError(f"vLLM needs tensor x pipeline = {tensor * pipeline} GPUs, but {gpus} are allocated")
    return tensor, pipeline


def compact(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def server_command(
    root: Path, profile: Profile, key_file: Path, gpus: int, executable: str | None = None
) -> tuple[list[str], dict[str, str]]:
    key = read_key(key_file)
    server, env = runtime_command(root, executable)
    settings, options = profile.vllm, profile.server
    tensor, _ = parallel_sizes(profile, gpus)
    command = [
        str(server),
        "serve",
        vllm_model(root, profile),
        "--served-model-name",
        profile.model["alias"],
        "--host",
        options["host"],
        "--port",
        str(options["port"]),
        "--max-model-len",
        str(profile.context_per_request),
        "--max-num-seqs",
        str(options["parallel"]),
        "--tensor-parallel-size",
        str(tensor),
    ]
    for key_name, flag in VALUE_FLAGS.items():
        if key_name in settings:
            command.extend([flag, str(settings[key_name])])
    for key_name, flag in SWITCHES.items():
        if settings.get(key_name):
            command.append(flag)
    if "enable_prefix_caching" in settings:
        enabled = settings["enable_prefix_caching"]
        command.append("--enable-prefix-caching" if enabled else "--no-enable-prefix-caching")
    if "tool_call_parser" in settings:
        command.extend(["--enable-auto-tool-choice", "--tool-call-parser", settings["tool_call_parser"]])
    if "chat_template" in settings:
        template = resolve_path(root, settings["chat_template"])
        if not template.is_file():
            raise ConfigError(f"Missing chat template: {template}")
        command.extend(["--chat-template", str(template)])
    for key_name, flag in JSON_FLAGS.items():
        if key_name in settings:
            command.extend([flag, compact(settings[key_name])])
    # Server-side defaults for requests that omit them; max_new_tokens is max_tokens.
    generation = {name: options[source] for source, name in SAMPLING.items() if source in options}
    generation["max_new_tokens"] = options["max_output"]
    command.extend(["--override-generation-config", compact(generation)])
    command.extend(["--middleware", MIDDLEWARE, *options["extra_args"]])

    env.update(settings.get("env", {}))
    # The key travels in the environment, never argv, so it stays out of ps and logs.
    env["VLLM_API_KEY"] = key
    env["PYTHONPATH"] = os.pathsep.join(path for path in (str(MIDDLEWARE_PATH), env.get("PYTHONPATH")) if path)
    env.setdefault("HF_HOME", str(root / ".cache" / "huggingface"))
    env.setdefault("VLLM_NO_USAGE_STATS", "1")
    if Path(command[2]).is_absolute():
        # Prepared weights are local; compute nodes need not reach the Hub.
        env.setdefault("HF_HUB_OFFLINE", "1")
    return command, env

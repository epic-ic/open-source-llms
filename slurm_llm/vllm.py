"""Install, resolve, and launch vLLM from validated profile settings."""

from __future__ import annotations

import hashlib
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
SAMPLING = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "presence_penalty": "presence_penalty",
    "repeat_penalty": "repetition_penalty",
}
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
    if os.sep not in executable:
        executable = shutil.which(executable) or executable
    server = resolve_path(root, executable)
    if not server.is_file() or not os.access(server, os.X_OK):
        raise ConfigError(f"Cannot execute vllm: {server}")
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env.pop("VLLM_BIN", None)
    venv = server.parent.parent
    if (venv / "pyvenv.cfg").is_file():
        env["VIRTUAL_ENV"] = str(venv)
        paths = [str(server.parent)]
        nvcc = next(iter(sorted(venv.glob("lib/python*/site-packages/nvidia/cu*/bin/nvcc"))), None)
        if nvcc:
            env.setdefault("CUDA_HOME", str(nvcc.parent.parent))
            paths.append(str(nvcc.parent))
        env["PATH"] = os.pathsep.join([*paths, env.get("PATH", "")])
    return server, env


def link_cuda_wheels(install: Path) -> None:
    """Expose wheel libraries under the paths and names CUDA JIT linkers expect."""
    for toolkit in install.glob("lib/python*/site-packages/nvidia/cu*"):
        lib = toolkit / "lib"
        if not lib.is_dir():
            continue
        if not (toolkit / "lib64").exists():
            (toolkit / "lib64").symlink_to("lib")
        for versioned in sorted(lib.glob("lib*.so.*")):
            unversioned = lib / (versioned.name.split(".so.", 1)[0] + ".so")
            if not unversioned.exists() and not unversioned.is_symlink():
                unversioned.symlink_to(versioned.name)


def runtime_settings(root: Path) -> dict:
    settings = read_toml(root / "config" / "runtime.toml").get("vllm")
    if not isinstance(settings, dict) or not isinstance(settings.get("version"), str):
        raise ConfigError("config/runtime.toml requires a [vllm] table with a version")
    unknown = settings.keys() - {"version", "python", "packages", "torch_backend"}
    if unknown:
        raise ConfigError(f"Unknown vllm runtime setting(s): {', '.join(sorted(unknown))}")
    version = settings["version"]
    if not re.fullmatch(r"[0-9][0-9A-Za-z.+-]*", version):
        raise ConfigError("vllm.version must be a release version")
    python = settings.setdefault("python", "3.12")
    packages = settings.setdefault("packages", [])
    backend = settings.get("torch_backend")
    if not isinstance(python, str) or not python.strip():
        raise ConfigError("vllm.python must be a nonempty string")
    if not isinstance(packages, list) or any(
        not isinstance(item, str) or not item.strip() or item.startswith("-") for item in packages
    ):
        raise ConfigError("vllm.packages must be package requirements")
    if backend is not None and (not isinstance(backend, str) or not backend.strip()):
        raise ConfigError("vllm.torch_backend must be a nonempty string")
    return settings


def prepare_runtime(root: Path) -> None:
    if os.getenv("VLLM_BIN"):
        server, _ = runtime_command(root)
        print(f"Using VLLM_BIN: {server}")
        return
    settings = runtime_settings(root)
    version = settings["version"]
    encoded = compact(settings) + "\n"
    fingerprint = hashlib.sha256(encoded.encode()).hexdigest()[:12]
    install = root / ".tools" / "vllm" / f"{version}-{fingerprint}"
    server = install / "bin" / "vllm"
    metadata = metadata_path(root)
    marker = install / "settings.json"
    if server.is_file() and os.access(server, os.X_OK) and marker.is_file() and marker.read_text() == encoded:
        link_cuda_wheels(install)
        atomic_write(metadata, str(server) + "\n")
        print(f"vLLM already prepared: {server}")
        return
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env.setdefault("UV_CACHE_DIR", str(root / ".cache" / "uv"))
    env.setdefault("UV_PYTHON_INSTALL_DIR", str(root / ".tools" / "python"))
    if not (install / "bin" / "python").exists():
        run(["uv", "venv", "--managed-python", "--python", settings["python"], str(install)], env=env, cwd=root)
    command = [
        "uv",
        "pip",
        "install",
        "--python",
        str(install / "bin" / "python"),
        f"vllm=={version}",
        *settings["packages"],
    ]
    if settings.get("torch_backend"):
        command.extend(["--torch-backend", settings["torch_backend"]])
    run(command, env=env, cwd=root)
    if not server.is_file() or not os.access(server, os.X_OK):
        raise ConfigError(f"vLLM installation did not provide {server}")
    link_cuda_wheels(install)
    atomic_write(marker, encoded)
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
        command.extend(["--tool-call-parser", settings["tool_call_parser"]])
    if "tool_call_parser" in settings or any(
        arg.split("=", 1)[0] == "--tool-call-parser" for arg in options["extra_args"]
    ):
        command.append("--enable-auto-tool-choice")
    if "chat_template" in settings:
        template = resolve_path(root, settings["chat_template"])
        if not template.is_file():
            raise ConfigError(f"Missing chat template: {template}")
        command.extend(["--chat-template", str(template)])
    for key_name, flag in JSON_FLAGS.items():
        if key_name in settings:
            command.extend([flag, compact(settings[key_name])])
    generation = {name: options[source] for source, name in SAMPLING.items() if source in options}
    generation["max_new_tokens"] = options["max_output"]
    command.extend(["--override-generation-config", compact(generation)])
    command.extend(["--middleware", MIDDLEWARE, *options["extra_args"]])

    env.update(settings.get("env", {}))
    env["VLLM_API_KEY"] = key
    env["PYTHONPATH"] = os.pathsep.join(path for path in (str(MIDDLEWARE_PATH), env.get("PYTHONPATH")) if path)
    env.setdefault("HF_HOME", str(root / ".cache" / "huggingface"))
    env.setdefault("VLLM_NO_USAGE_STATS", "1")
    if Path(command[2]).is_absolute():
        env.setdefault("HF_HUB_OFFLINE", "1")
    return command, env

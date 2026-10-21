"""Shared parsing, profile resolution, and error handling for CLI entry points."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from . import slurm
from .client import APIError, Client, validate_base_url
from .config import (
    SECTIONS,
    ConfigError,
    Profile,
    effective_profile,
    load_profile,
    project_root,
)
from .credentials import client_key
from .process import resolve_path

SERVER_OPTIONS = (
    "port",
    "host",
    "context_size",
    "parallel",
    "max_output",
    "reasoning",
    "reasoning_effort",
)


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def server_options(parser: argparse.ArgumentParser, *, key_file: bool = True) -> None:
    parser.add_argument("--port", type=positive_int)
    parser.add_argument("--host")
    parser.add_argument("--context-size", type=positive_int)
    parser.add_argument("--parallel", type=positive_int)
    parser.add_argument("--max-output", type=positive_int)
    parser.add_argument("--reasoning", choices=["on", "off", "auto"])
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--model-file", help="use existing weights: a GGUF file, or a vLLM model directory")
    if key_file:
        parser.add_argument("--api-key-file", help="path relative to --root, or absolute")


def command_parser(name: str, description: str | None) -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(prog=f"llm-{name}", description=description)
    cli.add_argument("--root", help="repository root (or set LLM_ROOT)")
    return cli


def client_options(cli: argparse.ArgumentParser) -> None:
    cli.add_argument("--profile", help="select alias and endpoint port from a profile")
    cli.add_argument("--job-config", type=Path)
    cli.add_argument("--base-url")
    cli.add_argument("--api-key-file")
    cli.add_argument("--timeout", type=positive_int, default=300)


def read_job(args: argparse.Namespace) -> tuple[Path, Profile, Path, str | None] | None:
    path = getattr(args, "job_config", None)
    return slurm.read_snapshot(path.expanduser()) if path else None


def get_profile(
    args: argparse.Namespace,
    root: Path,
    *,
    cluster: dict | None = None,
    job: tuple[Path, Profile, Path, str | None] | None = None,
    frozen: bool = False,
) -> Profile:
    name = getattr(args, "profile", None)
    if job:
        if name:
            raise ConfigError("Use a profile or --job-config, not both")
        _, profile, _, _ = job
        if frozen:
            supplied = [
                key
                for key in (*SERVER_OPTIONS, "model_file", "llama_server", "vllm", "api_key_file")
                if getattr(args, key, None) is not None
            ]
            if supplied:
                raise ConfigError("Job snapshots cannot be overridden by serve flags")
            return profile
        use_env = False
    else:
        if not name:
            raise ConfigError("Provide a profile or --job-config")
        profile = load_profile(root, name)
        use_env = True
    server = {key: getattr(args, key) for key in SERVER_OPTIONS if getattr(args, key, None) is not None}
    resources = {key: getattr(args, key) for key in SECTIONS["slurm"] if getattr(args, key, None) is not None}
    profile = effective_profile(profile, server=server, slurm=resources, cluster=cluster, use_env=use_env)
    if getattr(args, "model_file", None):
        profile.model["file"] = str(resolve_path(root, args.model_file))
    return profile


def executable(args: argparse.Namespace, profile: Profile) -> str | None:
    """Return the runtime override that matches the profile's backend."""
    values = {"llama.cpp": args.llama_server, "vllm": args.vllm}
    for backend, value in values.items():
        if value is not None and backend != profile.backend:
            flag = "--llama-server" if backend == "llama.cpp" else "--vllm"
            raise ConfigError(f"{flag} does not apply to a {profile.backend} profile")
    return values.get(profile.backend)


def base_url(args: argparse.Namespace, profile: Profile | None) -> str:
    value = (
        args.base_url
        or os.getenv("LLM_BASE_URL")
        or (f"http://127.0.0.1:{profile.server['port'] if profile else 8000}/v1")
    )
    return validate_base_url(value)


def write_output(content: str, path: Path | None) -> None:
    if path:
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Never silently overwrite a user's existing harness configuration.
        with path.open("x") as output:
            output.write(content)
        print(f"Created {path}", file=sys.stderr)
    else:
        print(content, end="")


def api_context(args: argparse.Namespace, *, authenticated: bool = True) -> tuple[Client, Profile | None]:
    root = project_root(args.root)
    job = read_job(args)
    profile = get_profile(args, root, job=job) if args.profile or job else None
    key_file = args.api_key_file or (str(job[2]) if job else None)
    key = client_key(root, key_file) if authenticated else ""
    return Client(base_url(args, profile), key, args.timeout), profile


def run_cli(
    cli: argparse.ArgumentParser,
    execute: Callable[[argparse.Namespace], None],
    argv: list[str] | None = None,
) -> int:
    args = cli.parse_args(argv)
    try:
        execute(args)
    except (ConfigError, APIError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"Command failed (exit {exc.returncode}): {shlex.join(exc.cmd)}", file=sys.stderr)
        if exc.stderr:
            print(exc.stderr.strip(), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0

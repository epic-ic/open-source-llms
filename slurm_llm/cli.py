"""A single entry point for preparing, serving, and using Slurm models."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from . import backends, integrations, slurm
from .client import APIError, Client, chat_body, response_text, validate_base_url
from .config import (
    SECTIONS,
    SERVER_DEFAULTS,
    SLURM_DEFAULTS,
    ConfigError,
    Profile,
    effective_profile,
    load_cluster,
    load_profile,
    project_root,
    valid_id,
)
from .credentials import client_key, key_path
from .process import resolve_path, run
from .weights import SHARD

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
    parser.add_argument("--model-file", help="use existing GGUF weights")
    if key_file:
        parser.add_argument("--api-key-file", help="path relative to --root, or absolute")


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", help="repository root (or set LLM_ROOT)")
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list model profiles")
    init = commands.add_parser("init", help="create a new model profile")
    init.add_argument("name")
    init.add_argument("--repo", help="Hugging Face repository")
    init.add_argument("--file", help="GGUF path inside the HF repository")
    init.add_argument("--backend", choices=["llama.cpp", "script"], default="llama.cpp")
    init.add_argument("--serve-script", help="Bash script path relative to repository root")
    init.add_argument("--prepare-script", help="optional Bash preparation script")
    init.add_argument("--tools", action="store_true", help="declare verified tool-call support")
    init.add_argument(
        "--responses", action="store_true", help="declare verified Responses API support"
    )
    server_options(init, key_file=False)

    prepare = commands.add_parser("prepare", help="prepare runtime, weights, and authentication")
    prepare.add_argument("profile")
    prepare.add_argument("--quant", help="replace {quant} in download patterns")
    prepare.add_argument("--api-key-file")
    only = prepare.add_mutually_exclusive_group()
    only.add_argument("--runtime-only", action="store_true")
    only.add_argument("--model-only", action="store_true")

    submit = commands.add_parser("submit", help="submit one model with shared Slurm infrastructure")
    submit.add_argument("profile")
    submit.add_argument("--cluster", help="alternative cluster TOML file")
    for key in ("gpus", "cpus-per-task"):
        submit.add_argument(f"--{key}", type=positive_int)
    for key in ("partition", "account", "nodelist", "constraint", "gpu-type", "qos", "mem", "time"):
        submit.add_argument(f"--{key}")
    check = submit.add_mutually_exclusive_group()
    check.add_argument(
        "--dry-run", action="store_true", help="preview without writing or submitting"
    )
    check.add_argument(
        "--test-only", action="store_true", help="ask Slurm to validate without submitting"
    )
    submit.add_argument("--llama-server", help="override the runtime executable")
    server_options(submit)

    serve = commands.add_parser("serve", help="serve within an allocation (normally used by Slurm)")
    serve.add_argument("profile", nargs="?")
    serve.add_argument("--job-config", type=Path, help="frozen settings produced by submit")
    serve.add_argument("--local", action="store_true", help="run directly without srun")
    serve.add_argument("--dry-run", action="store_true")
    serve.add_argument("--llama-server")
    server_options(serve)

    config = commands.add_parser("config", help="generate harness config or environment commands")
    config.add_argument("format", choices=["opencode", "codex", "env"])
    config.add_argument("profile", nargs="?")
    config.add_argument("--job-config", type=Path, help="use the exact submitted settings")
    config.add_argument("--base-url")
    config.add_argument("--opencode-version", type=int, choices=[1, 2], default=2)
    config.add_argument("--output", type=Path, help="create a new file; stdout by default")
    server_options(config)

    tunnel = commands.add_parser("tunnel", help="print or start an SSH tunnel")
    tunnel.add_argument("profile", nargs="?")
    tunnel.add_argument("--job-config", type=Path)
    tunnel.add_argument("--cluster")
    tunnel.add_argument("--login")
    tunnel.add_argument("--compute-host")
    tunnel.add_argument("--local-port", type=positive_int)
    tunnel.add_argument("--jump", action="store_true", help="SSH all the way to the compute node")
    tunnel.add_argument(
        "--execute", action="store_true", help="keep the tunnel running in foreground"
    )
    server_options(tunnel)

    for name in ("health", "models", "chat"):
        client = commands.add_parser(
            name,
            help={
                "health": "check endpoint readiness",
                "models": "list models served by the endpoint",
                "chat": "send a prompt",
            }[name],
        )
        client.add_argument("--profile", help="select alias and endpoint port from a profile")
        client.add_argument("--job-config", type=Path)
        client.add_argument("--base-url")
        client.add_argument("--api-key-file")
        client.add_argument("--timeout", type=positive_int, default=300)
        if name == "health":
            client.add_argument("--wait", type=positive_int, default=0, help="wait up to N seconds")
        if name == "chat":
            client.add_argument("prompt", help="prompt text, or '-' to read stdin")
            client.add_argument("--model", help="API model ID; auto-discovered when omitted")
            client.add_argument("--responses", action="store_true")
            client.add_argument("--stream", action="store_true")
            client.add_argument("--reasoning-effort", help="request a model-supported effort level")
            client.add_argument("--max-tokens", type=positive_int)
    return cli


def get_profile(
    args: argparse.Namespace,
    root: Path,
    *,
    cluster: dict | None = None,
    job: tuple[Path, Profile, Path, str | None] | None = None,
) -> Profile:
    name = getattr(args, "profile", None)
    if job:
        if name:
            raise ConfigError("Use a profile or --job-config, not both")
        _, profile, _, _ = job
        if args.command == "serve":
            supplied = [
                key
                for key in (*SERVER_OPTIONS, "model_file", "llama_server", "api_key_file")
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
    server = {
        key: getattr(args, key) for key in SERVER_OPTIONS if getattr(args, key, None) is not None
    }
    resources = {
        key: getattr(args, key) for key in SECTIONS["slurm"] if getattr(args, key, None) is not None
    }
    profile = effective_profile(
        profile, server=server, slurm=resources, cluster=cluster, use_env=use_env
    )
    if getattr(args, "model_file", None):
        profile.model["file"] = str(resolve_path(root, args.model_file))
    return profile


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


def init_profile(args: argparse.Namespace, root: Path) -> None:
    ident = valid_id(args.name)
    if args.backend == "llama.cpp":
        if args.model_file:
            if args.repo or args.file:
                raise ConfigError("Use --model-file or --repo with --file")
        elif not args.repo or not args.file:
            raise ConfigError("Provide --repo with --file, or --model-file")
        if args.serve_script or args.prepare_script:
            raise ConfigError("Script flags require --backend script")
    elif not args.serve_script or args.file or args.model_file:
        raise ConfigError("A script profile needs --serve-script; GGUF file flags do not apply")
    model = {"name": ident, "alias": ident, "backend": args.backend}
    if args.model_file:
        model["file"] = args.model_file
    if args.backend == "script" and args.repo:
        model["repo"] = args.repo
    data: dict = {
        "model": model,
        "server": {
            "context_size": args.context_size or SERVER_DEFAULTS["context_size"],
            "max_output": args.max_output or SERVER_DEFAULTS["max_output"],
        },
        "slurm": dict(SLURM_DEFAULTS),
        "capabilities": {"tools": args.tools, "responses": args.responses},
    }
    for key in ("port", "host", "parallel", "reasoning", "reasoning_effort"):
        if getattr(args, key) is not None:
            data["server"][key] = getattr(args, key)
    if args.backend == "llama.cpp" and not args.model_file:
        pattern = args.file
        match = SHARD.fullmatch(Path(args.file).name)
        if match:
            prefix, first, count = match.groups()
            if first != "00001" or int(count) < 1:
                raise ConfigError("Use the first GGUF shard as --file")
            pattern = str(Path(args.file).parent / f"{prefix}-*-of-{count}.gguf")
        data["download"] = {
            "repo": args.repo,
            "revision": "main",
            "directory": f"models/{ident}",
            "include": [pattern],
            "entrypoint": args.file,
        }
    if args.backend == "script":
        data["scripts"] = {"serve": args.serve_script}
        if args.prepare_script:
            data["scripts"]["prepare"] = args.prepare_script
    Profile(ident, data)  # Fail before writing if the profile is inconsistent.
    lines = ["# Edit resource, context, and capability settings for your model.\n"]
    for section, values in data.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {json.dumps(value)}")
        lines.append("")
    write_output("\n".join(lines), root / "config" / "models" / f"{ident}.toml")


def execute(args: argparse.Namespace) -> None:
    root = project_root(args.root)
    if args.command == "list":
        profiles = sorted((root / "config" / "models").glob("*.toml"))
        if not profiles:
            raise ConfigError(f"No profiles in {root / 'config' / 'models'}; use 'llm init'")
        for path in profiles:
            profile = load_profile(root, str(path))
            print(
                f"{profile.id:20} {profile.backend:10} {profile.model['alias']:24} "
                f"{profile.slurm['gpus']} GPU(s)  {profile.model['name']}"
            )
        return
    if args.command == "init":
        init_profile(args, root)
        return
    job_path = getattr(args, "job_config", None)
    job = slurm.read_snapshot(job_path.expanduser()) if job_path else None
    if args.command in {"health", "models", "chat"}:
        profile = get_profile(args, root, job=job) if args.profile or job else None
        if job and not args.api_key_file:
            args.api_key_file = str(job[2])
        key = "" if args.command == "health" else client_key(root, args.api_key_file)
        client = Client(base_url(args, profile), key, args.timeout)
        if args.command == "health":
            client.health(args.wait)
            print("Ready")
        elif args.command == "models":
            print(json.dumps(client.json("/models"), indent=2))
        else:
            if args.responses and profile and not profile.capabilities["responses"]:
                raise ConfigError("This profile does not declare Responses API support")
            prompt = sys.stdin.read() if args.prompt == "-" else args.prompt
            if not prompt.strip():
                raise ConfigError("Prompt cannot be empty")
            model = client.model_id(
                args.model
                or os.getenv("LLM_MODEL")
                or (profile.model["alias"] if profile else None)
            )
            maximum = args.max_tokens or (profile.server["max_output"] if profile else 1024)
            endpoint, body = chat_body(
                model,
                prompt,
                maximum,
                responses=args.responses,
                reasoning_effort=args.reasoning_effort,
            )
            if args.stream:
                for chunk in client.stream(endpoint, body):
                    print(chunk, end="", flush=True)
                print()
            else:
                print(response_text(client.json(endpoint, body), args.responses))
        return
    cluster = (
        load_cluster(root, getattr(args, "cluster", None))
        if args.command in {"submit", "tunnel"}
        else {}
    )
    profile = get_profile(args, root, cluster=cluster, job=job)
    key_file = key_path(root, getattr(args, "api_key_file", None))
    executable = getattr(args, "llama_server", None)
    if job:
        snapshot_root, _, snapshot_key, snapshot_executable = job
        if args.command == "serve":
            root, key_file, executable = snapshot_root, snapshot_key, snapshot_executable
        elif not args.api_key_file:
            key_file = snapshot_key
    if args.command == "prepare":
        backends.prepare(
            root,
            profile,
            key_file,
            quant=args.quant,
            runtime_only=args.runtime_only,
            model_only=args.model_only,
        )
    elif args.command == "submit":
        slurm.submit(
            root,
            profile,
            key_file,
            executable=executable,
            dry_run=args.dry_run,
            test_only=args.test_only,
        )
    elif args.command == "serve":
        slurm.serve(
            root, profile, key_file, executable=executable, dry_run=args.dry_run, local=args.local
        )
    elif args.command == "config":
        url = base_url(args, profile)
        if args.format == "opencode":
            content = integrations.opencode(profile, url, version=args.opencode_version)
        elif args.format == "codex":
            content = integrations.codex(profile, url)
        else:
            content = integrations.shell_environment(profile, url, key_file)
        write_output(content, args.output)
    elif args.command == "tunnel":
        ssh = cluster.get("ssh", {})
        command = integrations.tunnel_command(
            profile,
            login=args.login or ssh.get("login", ""),
            compute_host=args.compute_host or ssh.get("compute_host", ""),
            local_port=args.local_port,
            jump=args.jump,
        )
        print(shlex.join(command), flush=True)
        if args.execute:
            run(command)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
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

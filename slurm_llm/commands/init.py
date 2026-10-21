"""Create a new GGUF, vLLM, or script model profile."""

import argparse
import json
from pathlib import Path

from ..cli import command_parser, run_cli, server_options, write_output
from ..config import (
    SERVER_DEFAULTS,
    SLURM_DEFAULTS,
    VLLM_INCLUDE,
    ConfigError,
    Profile,
    project_root,
    valid_id,
)
from ..weights import SHARD


def parser() -> argparse.ArgumentParser:
    cli = command_parser("init", __doc__)
    cli.add_argument("name")
    cli.add_argument("--repo", help="Hugging Face repository")
    cli.add_argument("--file", help="GGUF path inside the HF repository")
    cli.add_argument("--backend", choices=["llama.cpp", "vllm", "script"], default="llama.cpp")
    cli.add_argument("--tool-call-parser", help="vLLM tool parser, e.g. hermes or qwen3_coder")
    cli.add_argument("--reasoning-parser", help="vLLM reasoning parser, e.g. qwen3")
    cli.add_argument("--serve-script", help="Bash script path relative to repository root")
    cli.add_argument("--prepare-script", help="optional Bash preparation script")
    cli.add_argument("--tools", action="store_true", help="declare verified tool-call support")
    cli.add_argument("--responses", action="store_true", help="declare verified Responses API support")
    server_options(cli, key_file=False)
    return cli


def execute(args: argparse.Namespace) -> None:
    root = project_root(args.root)
    ident = valid_id(args.name)
    if args.backend == "llama.cpp":
        if args.model_file:
            if args.repo or args.file:
                raise ConfigError("Use --model-file or --repo with --file")
        elif not args.repo or not args.file:
            raise ConfigError("Provide --repo with --file, or --model-file")
    elif args.backend == "vllm":
        if bool(args.repo) == bool(args.model_file) or args.file:
            raise ConfigError("Provide --repo (a safetensors repository) or --model-file DIRECTORY")
    elif not args.serve_script or args.file or args.model_file:
        raise ConfigError("A script profile needs --serve-script; GGUF file flags do not apply")
    if args.backend != "script" and (args.serve_script or args.prepare_script):
        raise ConfigError("Script flags require --backend script")
    if args.backend != "vllm" and (args.tool_call_parser or args.reasoning_parser):
        raise ConfigError("Parser flags require --backend vllm")
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
    if args.backend == "vllm":
        if args.repo:
            data["download"] = {
                "repo": args.repo,
                "revision": "main",
                "directory": f"models/{ident}",
                "include": list(VLLM_INCLUDE),
            }
        data["vllm"] = {"gpu_memory_utilization": 0.9, "enable_prefix_caching": True}
        for key in ("tool_call_parser", "reasoning_parser"):
            if getattr(args, key):
                data["vllm"][key] = getattr(args, key)
    if args.backend == "script":
        data["scripts"] = {"serve": args.serve_script}
        if args.prepare_script:
            data["scripts"]["prepare"] = args.prepare_script
    Profile(ident, data)
    lines = []
    for section, values in data.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {json.dumps(value)}")
        lines.append("")
    write_output("\n".join(lines), root / "config" / "models" / f"{ident}.toml")


def main(argv: list[str] | None = None) -> int:
    return run_cli(parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())

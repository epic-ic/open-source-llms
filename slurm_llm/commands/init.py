"""Create a new GGUF or script model profile."""

import argparse
import json
from pathlib import Path

from ..cli import command_parser, run_cli, server_options, write_output
from ..config import SERVER_DEFAULTS, SLURM_DEFAULTS, ConfigError, Profile, project_root, valid_id
from ..weights import SHARD


def parser() -> argparse.ArgumentParser:
    cli = command_parser("init", __doc__)
    cli.add_argument("name")
    cli.add_argument("--repo", help="Hugging Face repository")
    cli.add_argument("--file", help="GGUF path inside the HF repository")
    cli.add_argument("--backend", choices=["llama.cpp", "script"], default="llama.cpp")
    cli.add_argument("--serve-script", help="Bash script path relative to repository root")
    cli.add_argument("--prepare-script", help="optional Bash preparation script")
    cli.add_argument("--tools", action="store_true", help="declare verified tool-call support")
    cli.add_argument(
        "--responses", action="store_true", help="declare verified Responses API support"
    )
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


def main(argv: list[str] | None = None) -> int:
    return run_cli(parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())

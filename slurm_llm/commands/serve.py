"""Serve within a Slurm allocation or directly on a compute node."""

import argparse
from pathlib import Path

from .. import slurm
from ..cli import command_parser, get_profile, read_job, run_cli, server_options
from ..config import project_root
from ..credentials import key_path


def parser() -> argparse.ArgumentParser:
    cli = command_parser("serve", __doc__)
    cli.add_argument("profile", nargs="?")
    cli.add_argument("--job-config", type=Path, help="frozen settings produced by llm-submit")
    cli.add_argument("--local", action="store_true", help="run directly without srun")
    cli.add_argument("--dry-run", action="store_true")
    cli.add_argument("--llama-server")
    server_options(cli)
    return cli


def execute(args: argparse.Namespace) -> None:
    root = project_root(args.root)
    job = read_job(args)
    profile = get_profile(args, root, job=job, frozen=True)
    if job:
        root, _, key_file, executable = job
    else:
        key_file = key_path(root, args.api_key_file)
        executable = args.llama_server
    slurm.serve(
        root, profile, key_file, executable=executable, dry_run=args.dry_run, local=args.local
    )


def main(argv: list[str] | None = None) -> int:
    return run_cli(parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())

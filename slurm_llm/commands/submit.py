"""Submit a model server to Slurm using frozen job settings."""

import argparse

from .. import slurm
from ..cli import command_parser, get_profile, positive_int, run_cli, server_options
from ..config import load_cluster, project_root
from ..credentials import key_path


def parser() -> argparse.ArgumentParser:
    cli = command_parser("submit", __doc__)
    cli.add_argument("profile")
    cli.add_argument("--cluster", help="alternative cluster TOML file")
    for key in ("gpus", "cpus-per-task"):
        cli.add_argument(f"--{key}", type=positive_int)
    for key in ("partition", "account", "nodelist", "constraint", "gpu-type", "qos", "mem", "time"):
        cli.add_argument(f"--{key}")
    check = cli.add_mutually_exclusive_group()
    check.add_argument(
        "--dry-run", action="store_true", help="preview without writing or submitting"
    )
    check.add_argument(
        "--test-only", action="store_true", help="ask Slurm to validate without submitting"
    )
    cli.add_argument("--llama-server", help="override the runtime executable")
    server_options(cli)
    return cli


def execute(args: argparse.Namespace) -> None:
    root = project_root(args.root)
    profile = get_profile(args, root, cluster=load_cluster(root, args.cluster))
    slurm.submit(
        root,
        profile,
        key_path(root, args.api_key_file),
        executable=args.llama_server,
        dry_run=args.dry_run,
        test_only=args.test_only,
    )


def main(argv: list[str] | None = None) -> int:
    return run_cli(parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())

"""Prepare runtime, model weights, and authentication."""

import argparse

from .. import backends
from ..cli import command_parser, get_profile, run_cli
from ..config import project_root
from ..credentials import key_path


def parser() -> argparse.ArgumentParser:
    cli = command_parser("prepare", __doc__)
    cli.add_argument("profile")
    cli.add_argument("--quant", help="replace {quant} in download patterns")
    cli.add_argument("--api-key-file")
    only = cli.add_mutually_exclusive_group()
    only.add_argument("--runtime-only", action="store_true")
    only.add_argument("--model-only", action="store_true")
    return cli


def execute(args: argparse.Namespace) -> None:
    root = project_root(args.root)
    profile = get_profile(args, root)
    backends.prepare(
        root,
        profile,
        key_path(root, args.api_key_file),
        quant=args.quant,
        runtime_only=args.runtime_only,
        model_only=args.model_only,
    )


def main(argv: list[str] | None = None) -> int:
    return run_cli(parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())

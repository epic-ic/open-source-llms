"""Check endpoint readiness without requiring authentication."""

import argparse

from ..cli import api_context, client_options, command_parser, positive_int, run_cli


def parser() -> argparse.ArgumentParser:
    cli = command_parser("health", __doc__)
    client_options(cli)
    cli.add_argument("--wait", type=positive_int, default=0, help="wait up to N seconds")
    return cli


def execute(args: argparse.Namespace) -> None:
    client, _ = api_context(args, authenticated=False)
    client.health(args.wait)
    print("Ready")


def main(argv: list[str] | None = None) -> int:
    return run_cli(parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())

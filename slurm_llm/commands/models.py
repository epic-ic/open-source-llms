"""List the models served by an API endpoint."""

import argparse
import json

from ..cli import api_context, client_options, command_parser, run_cli


def parser() -> argparse.ArgumentParser:
    cli = command_parser("models", __doc__)
    client_options(cli)
    return cli


def execute(args: argparse.Namespace) -> None:
    client, _ = api_context(args)
    print(json.dumps(client.json("/models"), indent=2))


def main(argv: list[str] | None = None) -> int:
    return run_cli(parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())

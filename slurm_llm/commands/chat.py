"""Send a prompt using Chat Completions or the Responses API."""

import argparse
import os
import sys

from ..cli import api_context, client_options, command_parser, positive_int, run_cli
from ..client import chat_body, response_text
from ..config import ConfigError


def parser() -> argparse.ArgumentParser:
    cli = command_parser("chat", __doc__)
    client_options(cli)
    cli.add_argument("prompt", help="prompt text, or '-' to read stdin")
    cli.add_argument("--model", help="API model ID; auto-discovered when omitted")
    cli.add_argument("--responses", action="store_true")
    cli.add_argument("--stream", action="store_true")
    cli.add_argument(
        "--reasoning-effort",
        dest="request_reasoning_effort",
        help="request a model-supported effort level",
    )
    cli.add_argument("--max-tokens", type=positive_int)
    return cli


def execute(args: argparse.Namespace) -> None:
    client, profile = api_context(args)
    if args.responses and profile and not profile.capabilities["responses"]:
        raise ConfigError("This profile does not declare Responses API support")
    prompt = sys.stdin.read() if args.prompt == "-" else args.prompt
    if not prompt.strip():
        raise ConfigError("Prompt cannot be empty")
    model = client.model_id(args.model or os.getenv("LLM_MODEL") or (profile.model["alias"] if profile else None))
    maximum = args.max_tokens or (profile.server["max_output"] if profile else 1024)
    endpoint, body = chat_body(
        model,
        prompt,
        maximum,
        responses=args.responses,
        reasoning_effort=args.request_reasoning_effort,
    )
    if args.stream:
        for chunk in client.stream(endpoint, body):
            print(chunk, end="", flush=True)
        print()
    else:
        print(response_text(client.json(endpoint, body), args.responses))


def main(argv: list[str] | None = None) -> int:
    return run_cli(parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())

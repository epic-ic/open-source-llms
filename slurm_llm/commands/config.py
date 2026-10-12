"""Generate harness configuration or shell environment commands."""

import argparse
from pathlib import Path

from .. import integrations
from ..cli import (
    base_url,
    command_parser,
    get_profile,
    read_job,
    run_cli,
    server_options,
    write_output,
)
from ..config import project_root
from ..credentials import key_path


def parser() -> argparse.ArgumentParser:
    cli = command_parser("config", __doc__)
    cli.add_argument("format", choices=["opencode", "codex", "env"])
    cli.add_argument("profile", nargs="?")
    cli.add_argument("--job-config", type=Path, help="use the exact submitted settings")
    cli.add_argument("--base-url")
    cli.add_argument("--output", type=Path, help="create a new file; stdout by default")
    server_options(cli)
    return cli


def execute(args: argparse.Namespace) -> None:
    root = project_root(args.root)
    job = read_job(args)
    profile = get_profile(args, root, job=job)
    key_file = job[2] if job and not args.api_key_file else key_path(root, args.api_key_file)
    url = base_url(args, profile)
    if args.format == "opencode":
        content = integrations.opencode(profile, url)
    elif args.format == "codex":
        content = integrations.codex(profile, url)
    else:
        content = integrations.shell_environment(profile, url, key_file)
    write_output(content, args.output)


def main(argv: list[str] | None = None) -> int:
    return run_cli(parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())

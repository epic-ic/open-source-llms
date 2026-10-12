"""List available model profiles."""

import argparse

from ..cli import command_parser, run_cli
from ..config import ConfigError, load_profile, project_root


def parser() -> argparse.ArgumentParser:
    return command_parser("list", __doc__)


def execute(args: argparse.Namespace) -> None:
    root = project_root(args.root)
    profiles = sorted((root / "config" / "models").glob("*.toml"))
    if not profiles:
        raise ConfigError(f"No profiles in {root / 'config' / 'models'}; use 'uv run llm-init'")
    for path in profiles:
        profile = load_profile(root, str(path))
        print(
            f"{profile.id:20} {profile.backend:10} {profile.model['alias']:24} "
            f"{profile.slurm['gpus']} GPU(s)  {profile.model['name']}"
        )


def main(argv: list[str] | None = None) -> int:
    return run_cli(parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())

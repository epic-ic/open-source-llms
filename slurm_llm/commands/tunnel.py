"""Print or start an SSH tunnel to a model server."""

import argparse
import shlex
from pathlib import Path

from .. import integrations
from ..cli import command_parser, get_profile, positive_int, read_job, run_cli, server_options
from ..config import load_cluster, project_root
from ..process import run


def parser() -> argparse.ArgumentParser:
    cli = command_parser("tunnel", __doc__)
    cli.add_argument("profile", nargs="?")
    cli.add_argument("--job-config", type=Path)
    cli.add_argument("--cluster")
    cli.add_argument("--login")
    cli.add_argument("--compute-host")
    cli.add_argument("--local-port", type=positive_int)
    cli.add_argument("--jump", action="store_true", help="SSH all the way to the compute node")
    cli.add_argument("--execute", action="store_true", help="keep the tunnel running in foreground")
    server_options(cli)
    return cli


def execute(args: argparse.Namespace) -> None:
    root = project_root(args.root)
    cluster = load_cluster(root, args.cluster)
    profile = get_profile(args, root, cluster=cluster, job=read_job(args))
    ssh = cluster.get("ssh", {})
    command = integrations.tunnel_command(
        profile,
        login=args.login or ssh.get("login", ""),
        compute_host=args.compute_host or ssh.get("compute_host", ""),
        local_port=args.local_port,
        jump=args.jump,
    )
    print(shlex.join(command), flush=True)
    if args.execute:
        run(command)


def main(argv: list[str] | None = None) -> int:
    return run_cli(parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())

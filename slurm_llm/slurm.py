"""Slurm submission with one shared batch entry point and per-job snapshots."""

from __future__ import annotations

import json
import os
import shlex
import socket
import sys
import uuid
from pathlib import Path
from typing import Any

from .backends import script_path, server_command
from .config import ConfigError, Profile
from .credentials import read_key
from .process import atomic_write, run
from .runtime import runtime_command
from .weights import model_path


def snapshot(
    root: Path,
    profile: Profile,
    key_file: Path,
    executable: str | None = None,
    *,
    validate: bool = True,
) -> dict[str, Any]:
    profile = Profile(profile.id, profile.data)
    if validate:
        read_key(key_file)
        if profile.backend == "llama.cpp":
            profile.model["file"] = str(model_path(root, profile))
            executable = str(runtime_command(root, executable)[0])
        else:
            script_path(root, profile, "serve")
    return {
        "version": 1,
        "root": str(root),
        "id": profile.id,
        "profile": profile.data,
        "key_file": str(key_file),
        "executable": executable,
    }


def read_snapshot(path: Path) -> tuple[Path, Profile, Path, str | None]:
    try:
        data = json.loads(path.read_text())
        if data["version"] != 1:
            raise ConfigError("Unsupported job snapshot version")
        return (
            Path(data["root"]),
            Profile(data["id"], data["profile"]),
            Path(data["key_file"]),
            data.get("executable"),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ConfigError(f"Cannot read job snapshot {path}: {exc}") from exc


def submit_command(
    root: Path, profile: Profile, snapshot_path: Path, *, test_only: bool = False
) -> list[str]:
    settings = profile.slurm
    gpu = (
        f"gpu:{settings['gpu_type']}:{settings['gpus']}"
        if settings.get("gpu_type")
        else (f"gpu:{settings['gpus']}")
    )
    log = root / "logs" / f"{profile.id}-%j.log"
    command = [
        "sbatch",
        "--parsable",
        f"--job-name=llm-{profile.id}",
        f"--chdir={root}",
        "--nodes=1",
        "--ntasks=1",
        f"--gres={gpu}",
        f"--cpus-per-task={settings['cpus_per_task']}",
        f"--mem={settings['mem']}",
        f"--time={settings['time']}",
        f"--output={log}",
        f"--error={log}",
        "--export=ALL",
    ]
    for key in ("partition", "account", "nodelist", "constraint", "qos"):
        if settings.get(key):
            command.append(f"--{key}={settings[key]}")
    if test_only:
        command.append("--test-only")
    command.extend(
        [str(root / "slurm" / "serve.sbatch"), str(root), str(snapshot_path), sys.executable]
    )
    return command


def submit(
    root: Path,
    profile: Profile,
    key_file: Path,
    *,
    executable: str | None = None,
    dry_run: bool = False,
    test_only: bool = False,
) -> str | None:
    data = snapshot(root, profile, key_file, executable, validate=not dry_run)
    path = root / ".tools" / "jobs" / f"{profile.id}-{uuid.uuid4().hex}.json"
    command = submit_command(root, profile, path, test_only=test_only)
    if dry_run:
        # Preview has no writes, downloads, key reads, or scheduler side effects.
        print(shlex.join(command))
        print(json.dumps(data, indent=2))
        return None
    if not (root / "slurm" / "serve.sbatch").is_file():
        raise ConfigError(f"Missing shared Slurm entry point: {root / 'slurm' / 'serve.sbatch'}")
    (root / "logs").mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(data, indent=2) + "\n")
    try:
        result = run(command, cwd=root, capture=True)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    if test_only:
        path.unlink(missing_ok=True)
        print(result.stdout or result.stderr or "Slurm submission check passed", end="\n")
        return None
    job = result.stdout.strip().split(";")[0]
    if not job.isdigit():
        raise ConfigError(f"Unexpected sbatch output: {result.stdout.strip()}")
    # Keep stdout usable in job_id=$(llm submit PROFILE).
    print(job)
    print(f"Log: {root / 'logs' / f'{profile.id}-{job}.log'}", file=sys.stderr)
    print(f"Snapshot: {path}", file=sys.stderr)
    return job


def serve(
    root: Path,
    profile: Profile,
    key_file: Path,
    *,
    executable: str | None = None,
    dry_run: bool = False,
    local: bool = False,
) -> None:
    command, env = server_command(root, profile, key_file, executable)
    if dry_run:
        print(shlex.join(command if local else ["srun", *command]))
        return
    if not local and not os.getenv("SLURM_JOB_ID"):
        raise ConfigError("Use 'llm submit PROFILE', or pass --local inside an existing allocation")
    host = socket.getfqdn()
    print(
        f"Job {os.getenv('SLURM_JOB_ID', 'local')}; host {host}; model {profile.model['alias']}; "
        f"port {profile.server['port']}; context/slot {profile.context_per_request}",
        flush=True,
    )
    print(
        f"Connect: llm tunnel {profile.id} --compute-host {host} --login USER@HEADNODE", flush=True
    )
    print("The API is ready when /health returns HTTP 200.", flush=True)
    os.chdir(root)
    # exec preserves Slurm signals and exit status; no background server process.
    os.execvpe(command[0] if local else "srun", command if local else ["srun", *command], env)

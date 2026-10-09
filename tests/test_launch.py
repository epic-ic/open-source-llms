import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from slurm_llm.backends import server_command
from slurm_llm.config import ConfigError, Profile
from slurm_llm.credentials import ensure_key, read_key
from slurm_llm.runtime import runtime_command
from slurm_llm.slurm import read_snapshot, snapshot, submit, submit_command


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="llm tests ")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.key = self.root / ".secrets" / "llm-api-key"
        ensure_key(self.key)
        self.binary = self.root / "runtime" / "llama-server"
        self.binary.parent.mkdir(parents=True)
        self.binary.write_text("#!/bin/sh\nexit 0\n")
        self.binary.chmod(0o755)
        self.model = self.root / "models" / "model.gguf"
        self.model.parent.mkdir()
        self.model.write_bytes(b"GGUF")
        self.profile = Profile("custom", {"model": {"file": str(self.model)}, "slurm": {"gpus": 3}})
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_key_is_private_and_retained(self):
        original = read_key(self.key)
        ensure_key(self.key)
        self.assertEqual(read_key(self.key), original)
        self.assertEqual(self.key.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.key.parent.stat().st_mode & 0o777, 0o700)

    def test_invalid_existing_key_is_not_silently_replaced(self):
        self.key.write_text("two keys\n")
        with self.assertRaises(ConfigError):
            ensure_key(self.key)
        self.assertEqual(self.key.read_text(), "two keys\n")

    def test_gpu_split_uses_actual_allocation(self):
        with patch.dict(os.environ, {"SLURM_GPUS_ON_NODE": "2"}):
            command, _ = server_command(self.root, self.profile, self.key, str(self.binary))
        self.assertEqual(command[command.index("--tensor-split") + 1], "1,1")
        self.assertNotIn(read_key(self.key), command)
        self.assertEqual(command[command.index("--model") + 1], str(self.model))

    def test_explicit_gpu_split_is_retained(self):
        self.profile.server["tensor_split"] = "2,1"
        with patch.dict(os.environ, {"SLURM_GPUS_ON_NODE": "2"}):
            command, _ = server_command(self.root, self.profile, self.key, str(self.binary))
        self.assertEqual(command[command.index("--tensor-split") + 1], "2,1")

    def test_runtime_libraries_survive_snapshot_executable_override(self):
        metadata = self.root / ".tools" / "llama-cpp"
        metadata.mkdir(parents=True)
        (metadata / "server.path").write_text(str(self.binary))
        (metadata / "library-path.txt").write_text("/test/cuda:/test/llama")
        _, env = runtime_command(self.root, str(self.binary))
        self.assertEqual(env["LD_LIBRARY_PATH"], "/test/cuda:/test/llama")

    def test_script_backend_receives_common_settings_and_arguments(self):
        script = self.root / "scripts" / "serve.sh"
        script.parent.mkdir()
        script.write_text("#!/bin/bash\n")
        selected = Profile(
            "other",
            {
                "model": {"backend": "script", "repo": "org/model"},
                "scripts": {"serve": "scripts/serve.sh"},
                "server": {"extra_args": ["--custom", "value with spaces"]},
            },
        )
        command, env = server_command(self.root, selected, self.key)
        self.assertEqual(command, ["bash", str(script), "--custom", "value with spaces"])
        self.assertEqual(env["LLM_MODEL_REPO"], "org/model")
        self.assertEqual(env["LLM_CONTEXT_PER_REQUEST"], "8192")
        self.assertEqual(env["LLM_API_KEY_FILE"], str(self.key))

    def test_snapshot_freezes_weights_and_profile(self):
        data = snapshot(self.root, self.profile, self.key, str(self.binary))
        self.profile.server["port"] = 9000
        self.profile.model["file"] = "another.gguf"
        self.assertEqual(data["profile"]["server"]["port"], 8000)
        self.assertEqual(data["profile"]["model"]["file"], str(self.model))
        self.assertNotIn(read_key(self.key), json.dumps(data))
        path = self.root / "job.json"
        path.write_text(json.dumps(data))
        loaded_root, loaded_profile, loaded_key, executable = read_snapshot(path)
        self.assertEqual(loaded_root, self.root)
        self.assertEqual(loaded_profile.server["port"], 8000)
        self.assertEqual(loaded_key, self.key)
        self.assertEqual(executable, str(self.binary))

    def test_submission_has_no_fixed_node_or_partition(self):
        command = submit_command(self.root, self.profile, self.root / "job.json")
        self.assertIn("--gres=gpu:3", command)
        self.assertFalse(
            any(arg.startswith("--nodelist") or arg.startswith("--partition") for arg in command)
        )
        self.assertIn(f"--chdir={self.root}", command)
        self.assertEqual(command[-3], str(self.root))

    def test_dry_run_has_no_filesystem_or_scheduler_side_effects(self):
        with patch("slurm_llm.slurm.run") as run, contextlib.redirect_stdout(io.StringIO()):
            submit(self.root, self.profile, self.root / "missing-key", dry_run=True)
        run.assert_not_called()
        self.assertFalse((self.root / ".tools").exists())
        self.assertFalse((self.root / "logs").exists())

    def test_submission_records_settings_and_returns_only_job_id_on_stdout(self):
        batch = self.root / "slurm" / "serve.sbatch"
        batch.parent.mkdir()
        batch.write_text("#!/bin/bash\n")
        output = io.StringIO()
        with (
            patch(
                "slurm_llm.slurm.run",
                return_value=subprocess.CompletedProcess([], 0, "12345;cluster\n", ""),
            ),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            submit(self.root, self.profile, self.key, executable=str(self.binary))
        self.assertEqual(output.getvalue(), "12345\n")
        self.assertEqual(len(list((self.root / ".tools" / "jobs").glob("*.json"))), 1)

    def test_failed_submission_removes_orphan_snapshot(self):
        batch = self.root / "slurm" / "serve.sbatch"
        batch.parent.mkdir()
        batch.write_text("#!/bin/bash\n")
        with (
            patch("slurm_llm.slurm.run", side_effect=subprocess.CalledProcessError(1, ["sbatch"])),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            submit(self.root, self.profile, self.key, executable=str(self.binary))
        self.assertEqual(list((self.root / ".tools" / "jobs").glob("*.json")), [])

import contextlib
import io
import json
import os
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

from slurm_llm.config import Profile, load_profile
from slurm_llm.credentials import ensure_key
from slurm_llm.slurm import read_snapshot, snapshot

ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def invoke(self, command, *args, root=None):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            entry_point = import_module(f"slurm_llm.commands.{command}").main
            status = entry_point(["--root", str(root or self.root), *args])
        return status, output.getvalue(), error.getvalue()

    def test_init_new_model_and_generate_harness_without_weights(self):
        status, _, error = self.invoke(
            "init",
            "new-model",
            "--repo",
            "org/repo",
            "--file",
            "weights/model.gguf",
            "--tools",
            "--responses",
        )
        self.assertEqual(status, 0, error)
        profile = load_profile(self.root, "new-model")
        self.assertEqual(profile.download["include"], ["weights/model.gguf"])
        status, output, error = self.invoke("config", "opencode", "new-model", "--port", "9000")
        self.assertEqual(status, 0, error)
        provider = json.loads(output)["providers"]["slurm-new-model"]
        self.assertEqual(provider["settings"]["baseURL"], "http://127.0.0.1:9000/v1")

    def test_init_split_model_selects_every_shard(self):
        status, _, error = self.invoke(
            "init", "split-model", "--repo", "org/repo", "--file", "Q4/model-00001-of-00004.gguf"
        )
        self.assertEqual(status, 0, error)
        profile = load_profile(self.root, "split-model")
        self.assertEqual(profile.download["include"], ["Q4/model-*-of-00004.gguf"])

    def test_config_does_not_overwrite_an_existing_file(self):
        output = self.root / "opencode.json"
        output.write_text("existing-config")
        status, _, _ = self.invoke(
            "config", "opencode", "glm-flash", "--output", str(output), root=ROOT
        )
        self.assertEqual(status, 1)
        self.assertEqual(output.read_text(), "existing-config")

    def test_missing_profile_has_a_readable_error(self):
        status, _, error = self.invoke("config", "opencode", "missing")
        self.assertEqual(status, 1)
        self.assertIn("Cannot read", error)
        self.assertNotIn("Traceback", error)

    def test_snapshot_config_ignores_current_server_environment(self):
        profile = load_profile(ROOT, "glm-flash")
        job = self.root / "job.json"
        job.write_text(
            json.dumps(
                {
                    "version": 1,
                    "root": str(ROOT),
                    "id": profile.id,
                    "profile": profile.data,
                    "key_file": "/local/key",
                }
            )
        )
        with patch.dict(os.environ, {"LLM_PORT": "9000", "LLM_CTX_SIZE": "8192"}):
            status, output, error = self.invoke("config", "codex", "--job-config", str(job))
        self.assertEqual(status, 0, error)
        self.assertIn("http://127.0.0.1:8001/v1", output)
        self.assertIn("model_context_window = 131072", output)
        self.assertEqual(read_snapshot(job)[1].server["port"], 8001)

    def test_explicit_model_file_takes_precedence_over_environment(self):
        self.invoke("init", "local-model", "--model-file", "old.gguf")
        with patch.dict(os.environ, {"LLM_MODEL_FILE": "environment.gguf"}):
            status, output, error = self.invoke(
                "submit", "local-model", "--model-file", "flag.gguf", "--dry-run"
            )
        self.assertEqual(status, 0, error)
        self.assertIn(str(self.root / "flag.gguf"), output)
        self.assertNotIn("environment.gguf", output)

    def test_init_script_model_is_independent_of_llama_cpp(self):
        status, _, error = self.invoke(
            "init",
            "other",
            "--backend",
            "script",
            "--repo",
            "org/model",
            "--serve-script",
            "scripts/other.sh",
        )
        self.assertEqual(status, 0, error)
        selected = load_profile(self.root, "other")
        self.assertEqual(selected.backend, "script")
        self.assertFalse(selected.download)

    def test_snapshot_serving_ignores_environment_and_rejects_overrides(self):
        key = self.root / ".secrets" / "llm-api-key"
        ensure_key(key)
        model = self.root / "model.gguf"
        model.write_bytes(b"GGUF")
        binary = self.root / "llama-server"
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
        selected = Profile("frozen", {"model": {"file": str(model)}, "server": {"port": 8001}})
        job = self.root / "job.json"
        job.write_text(json.dumps(snapshot(self.root, selected, key, str(binary))))
        with patch.dict(os.environ, {"LLM_PORT": "9999", "LLM_MODEL_FILE": "missing.gguf"}):
            status, output, error = self.invoke("serve", "--job-config", str(job), "--dry-run")
        self.assertEqual(status, 0, error)
        self.assertIn("--port 8001", output)
        self.assertNotIn("9999", output)
        status, _, error = self.invoke(
            "serve", "--job-config", str(job), "--port", "9999", "--dry-run"
        )
        self.assertEqual(status, 1)
        self.assertIn("cannot be overridden", error)

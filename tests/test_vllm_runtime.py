import contextlib
import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from slurm_llm import vllm
from slurm_llm.backends import prepare
from slurm_llm.config import ConfigError, Profile
from slurm_llm.credentials import ensure_key


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="vllm tests ")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.key = self.root / ".secrets" / "llm-api-key"
        ensure_key(self.key)
        self.binary = self.root / "venv" / "bin" / "vllm"
        self.binary.parent.mkdir(parents=True)
        self.binary.write_text("#!/bin/sh\n")
        self.binary.chmod(0o755)
        (self.root / "venv" / "pyvenv.cfg").touch()
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.calls = []

    def fake_install(self, command, **kwargs):
        self.calls.append(command)
        if command[:2] == ["uv", "venv"]:
            binary = Path(command[-1]) / "bin" / "python"
            binary.parent.mkdir(parents=True)
            binary.touch()
        else:
            binary = Path(command[4]).with_name("vllm")
            binary.write_text("#!/bin/sh\n")
            binary.chmod(0o755)

    def test_runtime_resolution_order(self):
        with self.assertRaisesRegex(ConfigError, "Prepare vLLM"):
            vllm.runtime_command(self.root)
        metadata = vllm.metadata_path(self.root)
        metadata.parent.mkdir(parents=True)
        metadata.write_text(str(self.binary))
        self.assertEqual(vllm.runtime_command(self.root)[0], self.binary)
        with patch.dict(os.environ, {"VLLM_BIN": str(self.root / "missing")}):
            with self.assertRaisesRegex(ConfigError, "Cannot execute"):
                vllm.runtime_command(self.root)

    def test_runtime_exposes_venv_cuda_toolkit(self):
        venv = self.root / "venv"
        server = venv / "bin" / "vllm"
        nvcc = venv / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13" / "bin" / "nvcc"
        for path in (server, nvcc):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("#!/bin/sh\n")
            path.chmod(0o755)
        (venv / "pyvenv.cfg").write_text("")
        with patch.dict(os.environ, {"PATH": "/usr/bin"}):
            os.environ.pop("CUDA_HOME", None)
            _, env = vllm.runtime_command(self.root, str(server))
        self.assertEqual(env["CUDA_HOME"], str(nvcc.parent.parent))
        self.assertEqual(env["PATH"].split(os.pathsep), [str(server.parent), str(nvcc.parent), "/usr/bin"])

    def test_prepare_installs_pinned_vllm_once(self):
        (self.root / "config").mkdir()
        (self.root / "config" / "runtime.toml").write_text('[vllm]\nversion = "0.30.0"\npackages = ["x>=1"]\n')
        output = io.StringIO()
        profile = Profile("served", {"model": {"backend": "vllm", "repo": "org/model"}})
        with patch("slurm_llm.vllm.run", side_effect=self.fake_install), contextlib.redirect_stdout(output):
            prepare(self.root, profile, self.key, runtime_only=True)
            prepare(self.root, profile, self.key, runtime_only=True)
        self.assertEqual(len(self.calls), 2)
        self.assertIn("vllm==0.30.0", self.calls[1])
        self.assertIn("x>=1", self.calls[1])
        install = Path(self.calls[0][-1])
        self.assertEqual(vllm.metadata_path(self.root).read_text().strip(), str(install / "bin" / "vllm"))
        self.assertIn("already prepared", output.getvalue())

    def test_changed_runtime_settings_preserve_previous_installations(self):
        (self.root / "config").mkdir()
        config = self.root / "config" / "runtime.toml"
        settings = (
            'version = "0.31.0"',
            'version = "0.31.0"\npackages = ["extra==1"]',
            'version = "0.31.0"\npython = "3.13"',
            'version = "0.31.0"\ntorch_backend = "cu130"',
        )
        with patch("slurm_llm.vllm.run", side_effect=self.fake_install), contextlib.redirect_stdout(io.StringIO()):
            for values in settings:
                config.write_text("[vllm]\n" + values + "\n")
                vllm.prepare_runtime(self.root)
            config.write_text("[vllm]\n" + settings[0] + "\n")
            vllm.prepare_runtime(self.root)
        installs = [Path(command[-1]) for command in self.calls if command[:2] == ["uv", "venv"]]
        self.assertEqual(len(installs), len(settings))
        self.assertEqual(len(set(installs)), len(settings))
        self.assertTrue(all((install / "bin" / "vllm").is_file() for install in installs))
        self.assertEqual(vllm.runtime_command(self.root)[0], installs[0] / "bin" / "vllm")

    def test_failed_install_is_retried_before_publishing_the_runtime(self):
        (self.root / "config").mkdir()
        (self.root / "config" / "runtime.toml").write_text('[vllm]\nversion = "0.31.0"\n')
        attempts = []

        def fake_run(command, **kwargs):
            self.fake_install(command, **kwargs)
            if command[:3] == ["uv", "pip", "install"]:
                attempts.append(command)
                if len(attempts) == 1:
                    raise subprocess.CalledProcessError(1, command)

        with patch("slurm_llm.vllm.run", side_effect=fake_run), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                vllm.prepare_runtime(self.root)
            self.assertFalse(vllm.metadata_path(self.root).exists())
            vllm.prepare_runtime(self.root)
        self.assertEqual(len(attempts), 2)
        self.assertTrue(vllm.metadata_path(self.root).exists())

    def test_invalid_runtime_settings_fail_before_installation(self):
        (self.root / "config").mkdir()
        config = self.root / "config" / "runtime.toml"
        for setting in ('python = ""', 'packages = ["--upgrade"]', 'torch_backend = ""', 'overrides = ["x==1"]'):
            config.write_text('[vllm]\nversion = "0.31.0"\n' + setting + "\n")
            with self.subTest(setting=setting), patch("slurm_llm.vllm.run") as run, self.assertRaises(ConfigError):
                vllm.prepare_runtime(self.root)
            run.assert_not_called()

    def test_cuda_wheels_get_a_linkable_toolkit_layout(self):
        toolkit = self.root / "venv" / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13"
        (toolkit / "lib").mkdir(parents=True)
        (toolkit / "lib" / "libcudart.so.13").write_text("")
        (toolkit / "lib" / "libnvrtc.so.13").write_text("")
        vllm.link_cuda_wheels(self.root / "venv")
        vllm.link_cuda_wheels(self.root / "venv")
        self.assertEqual(
            (toolkit / "lib64" / "libcudart.so").resolve(), (toolkit / "lib" / "libcudart.so.13").resolve()
        )
        self.assertEqual((toolkit / "lib64" / "libnvrtc.so").resolve(), (toolkit / "lib" / "libnvrtc.so.13").resolve())

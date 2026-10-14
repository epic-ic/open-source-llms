import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from slurm_llm.config import ConfigError, Profile, effective_profile, load_cluster, load_profile

ROOT = Path(__file__).resolve().parents[1]


def profile(**sections):
    return Profile("example", {"model": {"file": "models/test.gguf"}} | sections)


class ConfigTests(unittest.TestCase):
    def test_all_shipped_profiles_are_valid(self):
        for path in (ROOT / "config" / "models").glob("*.toml"):
            with self.subTest(path=path):
                loaded = load_profile(ROOT, str(path))
                self.assertGreater(loaded.context_per_request, loaded.server["max_output"])

    def test_script_and_gguf_templates_are_valid(self):
        for name in ("script-model", "gguf-model"):
            self.assertIsInstance(load_profile(ROOT, f"examples/{name}.toml"), Profile)

    def test_unknown_settings_fail(self):
        for data in ({"serve": {}}, {"server": {"context": 4096}}, {"model": {"mystery": "x"}}):
            with self.subTest(data=data), self.assertRaises(ConfigError):
                Profile("bad", data)

    def test_invalid_counts_and_context_budget_fail(self):
        for settings in (
            {"port": 70000},
            {"parallel": 0},
            {"context_size": True},
            {"context_size": 4096, "parallel": 2, "max_output": 2048},
            {"extra_args": "--foo"},
            {"reasoning": "false"},
        ):
            with self.subTest(settings=settings), self.assertRaises(ConfigError):
                profile(server=settings)

    def test_script_requires_a_serve_script(self):
        with self.assertRaises(ConfigError):
            Profile("custom", {"model": {"backend": "script"}})

    def test_capabilities_are_opt_in(self):
        self.assertEqual(profile().capabilities, {"tools": False, "responses": False})

    def test_invalid_reasoning_efforts_fail(self):
        for efforts in ("high", ["low", "low"], [""], [1]):
            with self.subTest(efforts=efforts), self.assertRaises(ConfigError):
                profile(capabilities={"reasoning_efforts": efforts})

    def test_flags_override_environment_and_cluster_policy(self):
        original = profile(server={"port": 8001}, slurm={"gpus": 2})
        with patch.dict(os.environ, {"LLM_PORT": "9000", "LLM_REASONING": "true"}, clear=True):
            result = effective_profile(
                original,
                server={"port": 9001},
                slurm={"gpus": 4},
                cluster={"slurm": {"gpus": 3, "partition": "gpu"}},
            )
        self.assertEqual(result.server["port"], 9001)
        self.assertEqual(result.server["reasoning"], "on")
        self.assertEqual(result.slurm["gpus"], 4)
        self.assertEqual(result.slurm["partition"], "gpu")
        self.assertEqual(original.server["port"], 8001)
        self.assertEqual(original.slurm["gpus"], 2)

    def test_quant_cannot_escape_download_directory(self):
        for quant in ("../bad", "/tmp/bad", "bad;command"):
            with self.subTest(quant=quant), self.assertRaises(ConfigError):
                profile().format_download("{quant}/*.gguf", quant)

    def test_missing_cluster_uses_scheduler_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_cluster(Path(directory)), {})

    def test_cluster_rejects_misspelled_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cluster.toml").write_text('[ssh]\nloginn = "user@host"\n')
            with self.assertRaises(ConfigError):
                load_cluster(root, "cluster.toml")

    def test_profile_name_cannot_escape_directory(self):
        with self.assertRaises(ConfigError):
            load_profile(ROOT, "../bad")

    def test_relative_paths_cannot_escape_through_home_expansion(self):
        for value in ("~/weights", "~user/weights"):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                Profile("custom", {"model": {"backend": "script"}, "scripts": {"serve": value}})

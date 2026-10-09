import json
import tempfile
import tomllib
import unittest
from pathlib import Path

from slurm_llm.config import ConfigError, Profile
from slurm_llm.integrations import codex, opencode, shell_environment, tunnel_command


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.profile = Profile(
            "my-model",
            {
                "model": {"file": "model.gguf", "alias": "org/model"},
                "server": {"context_size": 32768, "parallel": 2, "max_output": 4096, "port": 8001},
                "capabilities": {"tools": True, "responses": True},
            },
        )

    def test_opencode_versions_use_alias_and_context_per_slot(self):
        for version, key in ((1, "provider"), (2, "providers")):
            with self.subTest(version=version):
                data = json.loads(
                    opencode(self.profile, "http://localhost:9999/v1", version=version)
                )
                self.assertEqual(data["model"], "slurm-my-model/org/model")
                provider = data[key]["slurm-my-model"]
                self.assertEqual(
                    provider["models"]["org/model"]["limit"], {"context": 16384, "output": 4096}
                )

    def test_codex_config_is_valid_toml_and_uses_responses(self):
        data = tomllib.loads(codex(self.profile, "http://localhost:9999/v1"))
        provider = data["model_providers"]["slurm_my_model"]
        self.assertEqual(provider["env_key"], "LLM_API_KEY")
        self.assertEqual(provider["wire_api"], "responses")
        self.assertFalse(provider["supports_websockets"])
        profile = data["profiles"]["slurm_my_model"]
        self.assertEqual(profile["model_context_window"], 16384)
        self.assertLessEqual(profile["model_auto_compact_token_limit"], 16384 - 4096)

    def test_codex_requires_declared_capabilities(self):
        self.profile.capabilities["responses"] = False
        with self.assertRaisesRegex(ConfigError, "Responses"):
            codex(self.profile, "http://localhost:8000/v1")

    def test_env_output_contains_no_key_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / "key with spaces"
            key.write_text("secret-value")
            output = shell_environment(self.profile, "http://localhost:8001/v1", key)
        self.assertNotIn("secret-value", output)
        self.assertIn('export LLM_API_KEY="$(cat -- "$LLM_API_KEY_FILE")"', output)
        self.assertIn("export LLM_MODEL=org/model", output)

    def test_tunnel_uses_local_override_and_actual_server_port(self):
        command = tunnel_command(
            self.profile, login="alice@headnode", compute_host="compute", local_port=9000
        )
        self.assertIn("127.0.0.1:9000:compute:8001", command)
        self.assertEqual(command[-1], "alice@headnode")

    def test_loopback_server_uses_proxyjump_to_compute_host(self):
        self.profile.server["host"] = "127.0.0.1"
        command = tunnel_command(self.profile, login="alice@headnode", compute_host="compute")
        self.assertIn("-J", command)
        self.assertIn("127.0.0.1:8001:127.0.0.1:8001", command)
        self.assertEqual(command[-1], "alice@compute")

    def test_ssh_option_injection_is_rejected(self):
        with self.assertRaises(ConfigError):
            tunnel_command(self.profile, login="-bad", compute_host="compute")

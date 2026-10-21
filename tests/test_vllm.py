import asyncio
import contextlib
import io
import json
import os
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

from slurm_llm import vllm
from slurm_llm.backends import server_command
from slurm_llm.config import ConfigError, Profile, effective_profile, load_profile
from slurm_llm.credentials import ensure_key, read_key
from slurm_llm.slurm import snapshot
from slurm_llm.vllm_middleware.slurm_llm_auth import BearerAuth
from slurm_llm.weights import validate_checkpoint


def vllm_profile(**sections):
    data = {"model": {"backend": "vllm", "repo": "org/model", "alias": "my-model"}}
    for section, values in sections.items():
        data[section] = data.get(section, {}) | values
    return Profile("served", data)


class ProfileTests(unittest.TestCase):
    def test_context_is_per_request_and_not_divided_by_parallel(self):
        selected = vllm_profile(server={"context_size": 32768, "parallel": 8, "max_output": 8192})
        self.assertEqual(selected.context_per_request, 32768)
        self.assertNotIn("gpu_layers", selected.server)

    def test_llama_only_settings_fail_with_a_vllm_hint(self):
        for key, value, hint in (
            ("reasoning", "off", "chat_template_kwargs"),
            ("cache_type_k", "q8_0", "kv_cache_dtype"),
            ("tensor_split", "1,1", "tensor_parallel_size"),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ConfigError, hint):
                vllm_profile(server={key: value})

    def test_vllm_settings_are_validated(self):
        for settings in (
            {"gpu_memory_utilization": 1.5},
            {"tensor_parallel_size": 0},
            {"enforce_eager": "yes"},
            {"kv_cache_dtype": ""},
            {"chat_template_kwargs": "enable_thinking=false"},
            {"env": {"VLLM_API_KEY": "leak"}},
            {"env": {"BAD NAME": "x"}},
            {"max_model_len": 4096},
            {"hf_overrides": {"value": float("nan")}},
            {"speculative_config": {"value": object()}},
        ):
            with self.subTest(settings=settings), self.assertRaises(ConfigError):
                vllm_profile(vllm=settings)

    def test_vllm_section_requires_vllm_backend(self):
        for settings in ({}, {"enforce_eager": True}):
            with self.subTest(settings=settings), self.assertRaisesRegex(ConfigError, "only applies"):
                Profile("gguf", {"model": {"file": "model.gguf"}, "vllm": settings})

    def test_tools_require_a_parser(self):
        with self.assertRaisesRegex(ConfigError, "tool_call_parser"):
            vllm_profile(capabilities={"tools": True})
        vllm_profile(capabilities={"tools": True}, vllm={"tool_call_parser": "hermes"})
        vllm_profile(capabilities={"tools": True}, server={"extra_args": ["--tool-call-parser", "x"]})
        vllm_profile(capabilities={"tools": True}, server={"extra_args": ["--tool-call-parser=hermes"]})

    def test_reasoning_efforts_must_be_accepted_by_vllm(self):
        with self.assertRaisesRegex(ConfigError, "ultra"):
            vllm_profile(capabilities={"reasoning_efforts": ["low", "ultra"]})

    def test_model_source_is_required(self):
        with self.assertRaisesRegex(ConfigError, "download section"):
            Profile("served", {"model": {"backend": "vllm"}})

    def test_download_needs_no_entrypoint_and_defaults_include(self):
        selected = Profile(
            "served",
            {"model": {"backend": "vllm"}, "download": {"repo": "org/model", "directory": "models/x"}},
        )
        self.assertIn("*.safetensors", selected.download["include"])

    def test_shell_llama_tuning_does_not_break_vllm_profiles(self):
        with patch.dict(os.environ, {"LLM_REASONING": "off", "LLM_GPU_LAYERS": "10", "LLM_PORT": "9001"}, clear=True):
            result = effective_profile(vllm_profile())
        self.assertNotIn("reasoning", result.server)
        self.assertEqual(result.server["port"], 9001)

    def test_shipped_vllm_profile_and_template_are_valid(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("qwen-27b-vllm", "glm-flash-vllm"):
            selected = load_profile(root, name)
            self.assertEqual(selected.backend, "vllm")
            self.assertEqual(selected.context_per_request, selected.server["context_size"])
        self.assertEqual(load_profile(root, "examples/vllm-model.toml").backend, "vllm")


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = Path(self.directory.name) / "model"
        self.model.mkdir()

    def test_index_shards_are_required(self):
        (self.model / "config.json").write_text("{}")
        index = {"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}
        (self.model / "model.safetensors.index.json").write_text(json.dumps(index))
        (self.model / "model-1.safetensors").write_bytes(b"x")
        with self.assertRaisesRegex(ConfigError, "model-2.safetensors"):
            validate_checkpoint(self.model)
        (self.model / "model-2.safetensors").write_bytes(b"x")
        self.assertEqual(validate_checkpoint(self.model), self.model)

    def test_config_and_weights_are_required(self):
        with self.assertRaisesRegex(ConfigError, "config.json"):
            validate_checkpoint(self.model)
        (self.model / "params.json").write_text("{}")
        with self.assertRaisesRegex(ConfigError, "No model weights"):
            validate_checkpoint(self.model)
        (self.model / "consolidated.safetensors").write_bytes(b"x")
        validate_checkpoint(self.model)

    def test_malformed_indexes_report_configuration_errors(self):
        (self.model / "config.json").write_text("{}")
        index = self.model / "model.safetensors.index.json"
        for value in ([], {}, {"weight_map": {}}, {"weight_map": []}, {"weight_map": {"a": ["file"]}}):
            with self.subTest(value=value), self.assertRaisesRegex(ConfigError, "weight_map"):
                index.write_text(json.dumps(value))
                validate_checkpoint(self.model)

    def test_index_cannot_reference_weights_outside_the_checkpoint(self):
        (self.model / "config.json").write_text("{}")
        other = self.model.parent / "other.safetensors"
        other.write_bytes(b"x")
        for shard in (str(other), "../other.safetensors", ""):
            with self.subTest(shard=shard), self.assertRaisesRegex(ConfigError, "relative shard"):
                (self.model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": shard}}))
                validate_checkpoint(self.model)

    def test_a_directory_with_a_weight_suffix_is_not_a_checkpoint(self):
        (self.model / "config.json").write_text("{}")
        (self.model / "model.safetensors").mkdir()
        with self.assertRaisesRegex(ConfigError, "No model weights"):
            validate_checkpoint(self.model)


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="vllm tests ")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.key = self.root / ".secrets" / "llm-api-key"
        ensure_key(self.key)
        self.binary = self.root / "venv" / "bin" / "vllm"
        self.binary.parent.mkdir(parents=True)
        self.binary.write_text("#!/bin/sh\nexit 0\n")
        self.binary.chmod(0o755)
        (self.root / "venv" / "pyvenv.cfg").write_text("")
        self.model = self.root / "models" / "checkpoint"
        self.model.mkdir(parents=True)
        (self.model / "config.json").write_text("{}")
        (self.model / "model.safetensors").write_bytes(b"x")
        self.environment = patch.dict(os.environ, {"VIRTUAL_ENV": "/project/.venv"}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def command(self, selected, gpus="1"):
        with patch.dict(os.environ, {"SLURM_GPUS_ON_NODE": gpus}):
            return server_command(self.root, selected, self.key, str(self.binary))

    def option(self, command, flag):
        return command[command.index(flag) + 1]

    def test_command_maps_profile_settings(self):
        selected = vllm_profile(
            model={"file": str(self.model)},
            server={
                "context_size": 65536,
                "parallel": 4,
                "max_output": 8192,
                "temperature": 0.6,
                "presence_penalty": 0.5,
            },
            vllm={
                "kv_cache_dtype": "fp8",
                "enable_prefix_caching": False,
                "language_model_only": True,
                "enable_expert_parallel": True,
                "tool_call_parser": "qwen3_coder",
                "reasoning_parser": "qwen3",
                "chat_template_kwargs": {"enable_thinking": False},
                "env": {"VLLM_ATTENTION_BACKEND": "FLASHINFER"},
            },
        )
        selected.server["repeat_penalty"] = 1.05
        command, env = self.command(selected, gpus="2")
        self.assertEqual(command[:3], [str(self.binary), "serve", str(self.model)])
        self.assertEqual(self.option(command, "--served-model-name"), "my-model")
        self.assertEqual(self.option(command, "--max-model-len"), "65536")
        self.assertEqual(self.option(command, "--max-num-seqs"), "4")
        self.assertEqual(self.option(command, "--tensor-parallel-size"), "2")
        self.assertEqual(self.option(command, "--kv-cache-dtype"), "fp8")
        self.assertEqual(self.option(command, "--tool-call-parser"), "qwen3_coder")
        self.assertIn("--enable-auto-tool-choice", command)
        self.assertIn("--no-enable-prefix-caching", command)
        self.assertIn("--language-model-only", command)
        self.assertIn("--enable-expert-parallel", command)
        self.assertEqual(self.option(command, "--default-chat-template-kwargs"), '{"enable_thinking":false}')
        self.assertEqual(
            json.loads(self.option(command, "--override-generation-config")),
            {"temperature": 0.6, "presence_penalty": 0.5, "repetition_penalty": 1.05, "max_new_tokens": 8192},
        )
        self.assertEqual(self.option(command, "--middleware"), vllm.MIDDLEWARE)
        key = read_key(self.key)
        self.assertNotIn(key, " ".join(command))
        self.assertEqual(env["VLLM_API_KEY"], key)
        self.assertEqual(env["PYTHONPATH"], str(vllm.MIDDLEWARE_PATH))
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["VIRTUAL_ENV"], str(self.root / "venv"))
        self.assertEqual(env["VLLM_ATTENTION_BACKEND"], "FLASHINFER")

    def test_hub_model_is_downloaded_by_vllm(self):
        command, env = self.command(vllm_profile(server={"extra_args": ["--seed", "1"]}))
        self.assertEqual(command[2], "org/model")
        self.assertNotIn("HF_HUB_OFFLINE", env)
        self.assertEqual(command[-2:], ["--seed", "1"])

    def test_extra_args_tool_parser_enables_automatic_tool_choice(self):
        for args in (["--tool-call-parser", "hermes"], ["--tool-call-parser=hermes"]):
            with self.subTest(args=args):
                command, _ = self.command(vllm_profile(server={"extra_args": args}, capabilities={"tools": True}))
                self.assertIn("--enable-auto-tool-choice", command)
                self.assertEqual(command[-len(args) :], args)

    def test_parallelism_must_fit_the_allocation(self):
        selected = vllm_profile(vllm={"pipeline_parallel_size": 2})
        command, _ = self.command(selected, gpus="4")
        self.assertEqual(self.option(command, "--tensor-parallel-size"), "2")
        self.assertEqual(self.option(command, "--pipeline-parallel-size"), "2")
        with self.assertRaisesRegex(ConfigError, "4 GPUs"):
            self.command(vllm_profile(vllm={"tensor_parallel_size": 4}), gpus="2")

    def test_missing_chat_template_fails_before_launch(self):
        with self.assertRaisesRegex(ConfigError, "chat template"):
            self.command(vllm_profile(vllm={"chat_template": "templates/missing.jinja"}))

    def test_snapshot_freezes_checkpoint_and_executable(self):
        selected = vllm_profile(download={"repo": "org/model", "directory": "models/checkpoint"})
        data = snapshot(self.root, selected, self.key, str(self.binary))
        self.assertEqual(data["profile"]["model"]["file"], str(self.model))
        self.assertEqual(data["executable"], str(self.binary))
        with self.assertRaisesRegex(ConfigError, "2 GPUs"):
            snapshot(self.root, vllm_profile(vllm={"tensor_parallel_size": 2}), self.key, str(self.binary))


class MiddlewareTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {"VLLM_API_KEY": "secret"}):
            self.middleware = BearerAuth(self.app)
        self.reached = []

    async def app(self, scope, receive, send):
        self.reached.append(scope["path"] if "path" in scope else scope["type"])

    def call(self, scope):
        sent = []

        async def send(message):
            sent.append(message)

        asyncio.run(self.middleware(scope, None, send))
        return sent

    def http(self, path, token=None, method="POST"):
        headers = [(b"authorization", f"Bearer {token}".encode())] if token else []
        return {"type": "http", "path": path, "method": method, "headers": headers}

    def test_every_route_but_health_requires_the_key(self):
        sent = self.call(self.http("/invocations"))
        self.assertEqual(sent[0]["status"], 401)
        self.assertEqual(self.call(self.http("/invocations", "wrong"))[0]["status"], 401)
        self.assertEqual(self.reached, [])
        self.call(self.http("/invocations", "secret"))
        self.call(self.http("/health", method="GET"))
        self.call({"type": "lifespan"})
        self.assertEqual(self.reached, ["/invocations", "/health", "lifespan"])

    def test_only_cors_preflights_bypass_options_authentication(self):
        scope = self.http("/v1/models", method="OPTIONS")
        self.assertEqual(self.call(scope)[0]["status"], 401)
        scope["headers"] = [(b"origin", b"https://example.com"), (b"access-control-request-method", b"POST")]
        self.call(scope)
        self.assertEqual(self.reached, ["/v1/models"])

    def test_unauthenticated_websocket_is_closed(self):
        sent = self.call({"type": "websocket", "path": "/v1/realtime", "headers": []})
        self.assertEqual(sent, [{"type": "websocket.close", "code": 1008}])

    def test_missing_key_refuses_to_start(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            BearerAuth(self.app)


class CLITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def invoke(self, command, *args):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            status = import_module(f"slurm_llm.commands.{command}").main(["--root", str(self.root), *args])
        return status, output.getvalue(), error.getvalue()

    def test_init_creates_a_vllm_profile(self):
        status, _, error = self.invoke(
            "init", "served", "--backend", "vllm", "--repo", "org/model", "--tool-call-parser", "hermes", "--tools"
        )
        self.assertEqual(status, 0, error)
        selected = load_profile(self.root, "served")
        self.assertEqual(selected.backend, "vllm")
        self.assertEqual(selected.download["directory"], "models/served")
        self.assertEqual(selected.vllm["tool_call_parser"], "hermes")
        status, _, error = self.invoke("init", "broken", "--backend", "vllm", "--repo", "org/model", "--tools")
        self.assertEqual(status, 1)
        self.assertIn("tool_call_parser", error)

    def test_serve_dry_run_accepts_only_the_matching_runtime_flag(self):
        self.invoke("init", "served", "--backend", "vllm", "--repo", "org/model")
        ensure_key(self.root / ".secrets" / "llm-api-key")
        model = self.root / "models" / "served"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        (model / "model.safetensors").write_bytes(b"x")
        binary = self.root / "vllm"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        status, output, error = self.invoke("serve", "served", "--vllm", str(binary), "--local", "--dry-run")
        self.assertEqual(status, 0, error)
        self.assertTrue(output.startswith(f"{binary} serve {model}"))
        status, _, error = self.invoke("serve", "served", "--llama-server", str(binary), "--dry-run")
        self.assertEqual(status, 1)
        self.assertIn("does not apply", error)

import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from slurm_llm.client import APIError, Client, chat_body, response_text, validate_base_url
from slurm_llm.commands.chat import main
from slurm_llm.commands.health import main as health_main
from slurm_llm.commands.models import main as models_main
from slurm_llm.config import ConfigError

ROOT = Path(__file__).resolve().parents[1]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def authenticated(self):
        if self.headers.get("Authorization") != "Bearer test-key":
            self.send_json(401, {"error": "unauthorized"})
            return False
        return True

    def do_GET(self):
        if self.path == "/health":
            status = self.server.health_status
            self.send_json(status, {"status": "ok" if status == 200 else "loading"})
        elif self.authenticated():
            self.send_json(200, {"data": [{"id": "test-model"}]})

    def do_POST(self):
        if not self.authenticated():
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.received.append((self.path, body))
        responses = self.path == "/v1/responses"
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            prompt = body.get("input") or body.get("messages", [{}])[0].get("content")
            if prompt == "failed-stream":
                self.wfile.write(b'data: {"type":"error","error":{"message":"failed"}}\n\n')
                return
            if prompt == "truncated-stream":
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')
                return
            if prompt == "malformed-stream":
                self.wfile.write(b"data: []\n\n")
                return
            events = (
                [
                    {"type": "response.output_text.delta", "delta": "hello"},
                    {"type": "response.completed"},
                ]
                if responses
                else [
                    {"choices": [{"delta": {"content": "hello"}}]},
                    {"choices": [], "usage": {"completion_tokens": 1}},
                ]
            )
            for event in events:
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        elif responses:
            self.send_json(
                200,
                {
                    "output": [
                        {"type": "reasoning", "summary": []},
                        {"type": "message", "content": [{"type": "output_text", "text": "hello"}]},
                    ]
                },
            )
        else:
            self.send_json(200, {"choices": [{"message": {"content": "hello"}}]})


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.health_status = 200
        cls.server.received = []
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_models_authentication_and_discovery(self):
        self.assertEqual(Client(self.url, "test-key").model_id(), "test-model")
        with self.assertRaisesRegex(APIError, "401"):
            Client(self.url, "wrong-key").json("/models")

    def test_chat_and_responses_payloads_and_text(self):
        for responses in (False, True):
            with self.subTest(responses=responses):
                endpoint, body = chat_body(
                    "test-model", "test prompt", 123, responses=responses, reasoning_effort="high"
                )
                result = Client(self.url, "test-key").json(endpoint, body)
                self.assertEqual(response_text(result, responses), "hello")
                path, received = self.server.received[-1]
                self.assertEqual(path, "/v1" + endpoint)
                self.assertEqual(received["model"], "test-model")
                if responses:
                    self.assertFalse(received["store"])
                    self.assertEqual(received["reasoning"], {"effort": "high"})
                    self.assertEqual(received["max_output_tokens"], 123)
                else:
                    self.assertEqual(received["reasoning_effort"], "high")
                    self.assertEqual(received["max_tokens"], 123)

    def test_streaming_for_both_api_formats(self):
        for responses in (False, True):
            endpoint, body = chat_body("test-model", "test prompt", 123, responses=responses)
            self.assertEqual("".join(Client(self.url, "test-key").stream(endpoint, body)), "hello")

    def test_failed_truncated_and_malformed_streams_are_errors(self):
        for prompt in ("failed-stream", "truncated-stream", "malformed-stream"):
            with self.subTest(prompt=prompt), self.assertRaises(APIError):
                endpoint, body = chat_body("test-model", prompt, 123)
                list(Client(self.url, "test-key").stream(endpoint, body))

    def test_cli_can_discover_and_chat_without_any_model_specific_configuration(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=True),
        ):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main(
                    [
                        "--root",
                        directory,
                        "hello",
                        "--base-url",
                        self.url,
                        "--stream",
                        "--responses",
                    ]
                )
        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue(), "hello\n")

    def test_reasoning_effort_stays_a_request_option_under_a_profile(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=True):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main(
                    [
                        "--root",
                        str(ROOT),
                        "hello",
                        "--profile",
                        "glm-flash-vllm",
                        "--base-url",
                        self.url,
                        "--reasoning-effort",
                        "low",
                    ]
                )
        self.assertEqual(status, 0)
        path, body = self.server.received[-1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(body["reasoning_effort"], "low")

    def test_health_does_not_require_authentication(self):
        Client(self.url).health()

    def test_health_entry_point_does_not_need_a_key_file(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {}, clear=True),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            status = health_main(["--root", directory, "--base-url", self.url])
        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue(), "Ready\n")

    def test_models_entry_point_reads_the_project_secret_by_default(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {}, clear=True),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            key_file = Path(directory) / ".secrets" / "llm-api-key"
            key_file.parent.mkdir()
            key_file.write_text("test-key\n")
            status = models_main(["--root", directory, "--base-url", self.url])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["data"][0]["id"], "test-model")

    def test_loading_status_does_not_report_ready(self):
        self.server.health_status = 503
        try:
            with self.assertRaisesRegex(APIError, "loading"):
                Client(self.url).health()
        finally:
            self.server.health_status = 200

    def test_invalid_urls_are_rejected(self):
        for value in (
            "file:///tmp/v1",
            "http://localhost",
            "http://user:pass@host/v1",
            "http://localhost/v1?secret=true",
            "http://localhost:bad/v1",
        ):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                validate_base_url(value)

    def test_malformed_completion_is_a_useful_error(self):
        with self.assertRaisesRegex(APIError, "Unexpected API response"):
            response_text({"choices": []})

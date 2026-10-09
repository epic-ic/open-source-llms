"""Dependency-free client for OpenAI-compatible endpoints."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .config import ConfigError


class APIError(RuntimeError):
    """An HTTP, transport, or response-format failure."""


def validate_base_url(value: str) -> str:
    value = value.rstrip("/")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith("/v1")
    ):
        raise ConfigError("Base URL must be an HTTP(S) API URL ending in /v1")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ConfigError(f"Invalid URL port: {exc}") from exc
    return value


@dataclass
class Client:
    base_url: str
    api_key: str = ""
    timeout: int = 300

    def __post_init__(self) -> None:
        self.base_url = validate_base_url(self.base_url)

    def request(self, endpoint: str, body: dict[str, Any] | None = None) -> Any:
        url = self.base_url[:-3] + "/health" if endpoint == "/health" else self.base_url + endpoint
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return Request(
            url, data=json.dumps(body).encode() if body is not None else None, headers=headers
        )

    def json(self, endpoint: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            with urlopen(self.request(endpoint, body), timeout=self.timeout) as response:
                result = json.load(response)
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object")
            return result
        except HTTPError as exc:
            raise APIError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
        except (URLError, OSError) as exc:
            raise APIError(
                f"Connection failed: {exc}. Check the SSH tunnel and Slurm job."
            ) from exc
        except ValueError as exc:
            raise APIError(f"Unexpected API response: {exc}") from exc

    def model_id(self, requested: str | None = None) -> str:
        if requested:
            return requested
        try:
            models = self.json("/models")["data"]
            if len(models) == 1:
                return models[0]["id"]
        except (KeyError, IndexError, TypeError) as exc:
            raise APIError(f"Unexpected model list: {exc}") from exc
        raise ConfigError("Endpoint does not serve exactly one model; pass --model or --profile")

    def stream(self, endpoint: str, body: dict[str, Any]) -> Iterator[str]:
        completed = False
        try:
            with urlopen(
                self.request(endpoint, body | {"stream": True}), timeout=self.timeout
            ) as response:
                for line in response:
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        return
                    if not data:
                        continue
                    event = json.loads(data)
                    if not isinstance(event, dict):
                        raise ValueError("Expected a JSON event object")
                    if event.get("error") or event.get("type") in {"error", "response.failed"}:
                        raise APIError(f"Stream failed: {event}")
                    if endpoint == "/responses":
                        if event.get("type") in {"response.completed", "response.incomplete"}:
                            completed = True
                        if event.get("type") == "response.output_text.delta":
                            yield event["delta"]
                    else:
                        choices = event.get("choices", [])
                        if choices:
                            if choices[0].get("finish_reason"):
                                completed = True
                            delta = choices[0].get("delta", {})
                            yield delta.get("content") or delta.get("reasoning_content") or ""
            if not completed:
                raise APIError("Stream ended before a completion event")
        except HTTPError as exc:
            raise APIError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
        except (URLError, OSError) as exc:
            raise APIError(f"Streaming connection failed: {exc}") from exc
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise APIError(f"Unexpected stream event: {exc}") from exc

    def health(self, wait: int = 0) -> None:
        deadline = time.monotonic() + wait
        while True:
            try:
                with urlopen(self.request("/health"), timeout=min(self.timeout, 10)):
                    return
            except HTTPError as exc:
                if exc.code != 503:
                    raise APIError(f"Health check failed: HTTP {exc.code}") from exc
                detail = "Model is still loading (HTTP 503)"
            except (URLError, OSError) as exc:
                detail = f"Endpoint not reachable: {exc}"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise APIError(detail)
            time.sleep(min(2, remaining))


def chat_body(
    model: str,
    prompt: str,
    max_tokens: int,
    *,
    responses: bool = False,
    reasoning_effort: str | None = None,
) -> tuple[str, dict[str, Any]]:
    if responses:
        body: dict[str, Any] = {
            "model": model,
            "input": prompt,
            "max_output_tokens": max_tokens,
            "store": False,
        }
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
        return "/responses", body
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    return "/chat/completions", body


def response_text(result: dict[str, Any], responses: bool = False) -> str:
    try:
        if responses:
            if result.get("error") or result.get("status") == "failed":
                raise APIError(f"Response failed: {result.get('error')}")
            return "".join(
                part["text"]
                for item in result["output"]
                if item.get("type") == "message"
                for part in item.get("content", [])
                if part.get("type") == "output_text"
            )
        message = result["choices"][0]["message"]
        return message.get("content") or message.get("reasoning_content") or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise APIError(f"Unexpected API response: {exc}") from exc

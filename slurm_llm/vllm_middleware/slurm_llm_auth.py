"""Require the server key on every vLLM route except /health.

vLLM's built-in VLLM_API_KEY check only covers /v1, /v2, and /inference, leaving
routes such as /invocations open. vLLM's own interpreter imports this module
through --middleware, so it may use only the standard library.
"""

import hmac
import json
import os

OPEN_PATHS = frozenset({"/health"})


class BearerAuth:
    def __init__(self, app):
        key = os.environ.get("VLLM_API_KEY", "").strip()
        if not key:
            raise RuntimeError("VLLM_API_KEY is not set")
        self.app = app
        self.expected = f"Bearer {key}".encode()

    def authorized(self, scope):
        # CORS preflights never carry credentials and cannot reach a handler.
        if scope["type"] == "http" and (scope["path"] in OPEN_PATHS or scope.get("method") == "OPTIONS"):
            return True
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                return hmac.compare_digest(value.strip(), self.expected)
        return False

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or self.authorized(scope):
            await self.app(scope, receive, send)
        elif scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
        else:
            body = json.dumps({"error": {"message": "Invalid or missing API key", "code": 401}}).encode()
            headers = [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"www-authenticate", b"Bearer"),
            ]
            await send({"type": "http.response.start", "status": 401, "headers": headers})
            await send({"type": "http.response.body", "body": body})

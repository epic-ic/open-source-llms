"""Authenticate vLLM routes using only its interpreter's standard library."""

import hmac
import json
import os


class BearerAuth:
    def __init__(self, app):
        key = os.environ.get("VLLM_API_KEY", "").strip()
        if not key:
            raise RuntimeError("VLLM_API_KEY is not set")
        self.app = app
        self.expected = f"Bearer {key}".encode()

    def authorized(self, scope):
        headers = dict(scope.get("headers", []))
        preflight = (
            scope.get("method") == "OPTIONS" and b"origin" in headers and b"access-control-request-method" in headers
        )
        if scope["type"] == "http" and (scope["path"] == "/health" or preflight):
            return True
        return hmac.compare_digest(headers.get(b"authorization", b"").strip(), self.expected)

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

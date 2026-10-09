"""Small, dependency-free client for the cluster's OpenAI-compatible API."""

import argparse
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="prompt to send to the model")
    parser.add_argument(
        "--base-url", default=os.getenv("QWEN_BASE_URL", "http://127.0.0.1:8000/v1")
    )
    parser.add_argument("--model", default=os.getenv("QWEN_MODEL", "qwen3.8-27b"))
    parser.add_argument("--api-key-file", type=Path, help="alternative to QWEN_API_KEY")
    parser.add_argument("--models", action="store_true", help="list served models")
    parser.add_argument("--responses", action="store_true", help="use /v1/responses")
    parser.add_argument("--reasoning", action="store_true", help="enable reasoning")
    parser.add_argument("--max-tokens", type=positive_int, default=1024)
    parser.add_argument("--timeout", type=positive_int, default=300)
    args = parser.parse_args()

    if args.models and (args.prompt is not None or args.responses):
        parser.error("--models cannot be combined with a prompt or --responses")
    if not args.models and not args.prompt:
        parser.error("provide a prompt or use --models")

    base_url = args.base_url.rstrip("/")
    parsed_url = urlsplit(base_url)
    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.netloc
        or parsed_url.query
        or parsed_url.fragment
        or not parsed_url.path.endswith("/v1")
    ):
        parser.error("--base-url must be an HTTP(S) API URL ending in /v1")

    try:
        api_key = (
            args.api_key_file.read_text().strip()
            if args.api_key_file
            else os.getenv("QWEN_API_KEY", "").strip()
        )
    except OSError as exc:
        parser.error(f"cannot read API key file: {exc}")
    if not api_key or any(character.isspace() for character in api_key):
        parser.error("set QWEN_API_KEY or provide a file containing a single API key")

    body = None
    if args.models:
        endpoint = "/models"
    elif args.responses:
        endpoint = "/responses"
        body = {
            "model": args.model,
            "input": args.prompt,
            "max_output_tokens": args.max_tokens,
            "store": False,
        }
    else:
        endpoint = "/chat/completions"
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens,
        }

    request = Request(
        base_url + endpoint,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
        if args.models:
            print(json.dumps(result, indent=2))
        elif args.responses:
            print("".join(
                part["text"]
                for item in result["output"]
                if item.get("type") == "message"
                for part in item.get("content", [])
                if part.get("type") == "output_text"
            ))
        else:
            message = result["choices"][0]["message"]
            print(message.get("content") or message.get("reasoning_content") or "")
        return 0
    except HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        print(f"API error {exc.code}: {detail}", file=sys.stderr)
    except (URLError, TimeoutError, OSError) as exc:
        print(f"Connection failed: {exc}. Check the SSH tunnel and Slurm job.", file=sys.stderr)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        print(f"Unexpected API response: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

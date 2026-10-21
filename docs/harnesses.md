# Harnesses and API clients

Run the agent on your local code project, with an SSH tunnel to the Slurm server.
All integrations use `LLM_API_KEY`. Copy the server key to
`.secrets/llm-api-key` in your local checkout as shown in the README; that directory
is ignored by Git. Load it before starting the harness:

```bash
export LLM_API_KEY_FILE="$PWD/.secrets/llm-api-key"
export LLM_API_KEY="$(cat "$LLM_API_KEY_FILE")"
```

Generate environment commands if you also want the alias and base URL:

```bash
uv run llm-config env my-model --api-key-file "$LLM_API_KEY_FILE"
# Review and evaluate that output, or save it with --output and source the file.
```

No credentials are embedded in generated configs or printed by these commands.
Any OpenAI-compatible client can use the same base URL, bearer key, and model ID.

## OpenCode

Only OpenCode 2.x is supported. Check `opencode --version` and install or upgrade
to 2.x before using the generated config:

```bash
# Run from your local checkout; target the code project the agent should edit.
uv run llm-config opencode my-model \
  --output /PATH/TO/CODE/PROJECT/opencode.json

cd /PATH/TO/CODE/PROJECT
opencode
```

If `opencode.json` exists, generate to a temporary file and merge its provider
and model settings. Generated configs use `providers`/`package`/`settings` and
the Chat Completions compatible runtime. The config selects the default model
and reads its credential from `LLM_API_KEY`, so exporting that variable before
launching `opencode` is sufficient; no `/connect` step is needed.
The provider ID is `slurm-PROFILE_ID`; the model ID after it is `model.alias`.
See the official [OpenCode 2.x provider documentation](https://opencode.ai/v2/docs/providers).

## Codex

Generate a named provider/profile fragment:

```bash
uv run llm-config codex my-model --output config/generated/my-model-codex.toml
# Step 1: save the top-level profile keys as ~/.codex/slurm_my_model.config.toml.
# Step 2: merge the [model_providers.slurm_my_model] table into ~/.codex/config.toml.
codex --profile slurm_my_model
```

The profile ID is `slurm_` plus the model profile's ID with hyphens replaced by
underscores. Generation does not edit your global Codex configuration or change
its default model. Codex 0.134 and later read `--profile` settings from
`~/.codex/PROFILE.config.toml`, not from a `[profiles]` table in `config.toml`.
It requires both `capabilities.tools = true` and
`capabilities.responses = true`.

Codex uses the Responses wire API; generated providers select HTTP streaming,
reference `LLM_API_KEY`, and set the model context and compaction budget. See
[official OpenAI documentation for custom providers](https://learn.chatgpt.com/docs/config-file/config-advanced)
and the [configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).

The pinned llama.cpp server exposes `/v1/responses` by translating requests to
Chat Completions internally. That implementation is documented in the
[server API reference](https://github.com/ggml-org/llama.cpp/blob/b11472/tools/server/README.md).
vLLM implements `/v1/responses` natively, with tool calls handled by the profile's
`vllm.tool_call_parser` and reasoning by `vllm.reasoning_parser`.
Capabilities in a profile declare the intended API features; they do not certify
an actual model's tool behavior or every client feature. Check streamed Responses
and a real tool call/result loop before relying on a Codex workflow.

## Use the exact submitted settings

If you override context or port during submission, generate configs with those
same flags. Alternatively use the job snapshot path printed during submission:

```bash
uv run llm-config opencode --job-config .tools/jobs/my-model-SNAPSHOT_ID.json \
  --base-url http://127.0.0.1:9000/v1 --output config/generated/opencode.json
uv run llm-config codex --job-config .tools/jobs/my-model-SNAPSHOT_ID.json
```

Snapshots include the model/resource settings and credential file path, never
the key itself. When using a snapshot copied to your laptop, override
`--api-key-file` for `uv run llm-config env` or API client commands to point to the local key.

## Python CLI and curl

The client can discover the model when the endpoint serves exactly one model:

```bash
export LLM_BASE_URL=http://127.0.0.1:8000/v1
uv run llm-models
uv run llm-chat "Say hello."
uv run llm-chat --model my-model --stream "Write a short Python function."
uv run llm-chat --responses --stream "Explain this server in one sentence."
printf '%s\n' 'Summarize this text.' | uv run llm-chat -
```

Use `--reasoning-effort LEVEL` with `llm-chat` to set effort for that request.
Choose a level supported by the model; this flag does not change server defaults.

`--profile my-model` selects its port, alias, and output budget. `--model` overrides
the alias; `--base-url` overrides `LLM_BASE_URL`. `LLM_MODEL` is another alias
override. Key selection is explicit `--api-key-file`, then `LLM_API_KEY`, then
`LLM_API_KEY_FILE` or `.secrets/llm-api-key`. Health needs no authentication:

```bash
uv run llm-health --profile my-model --wait 300
curl --fail-with-body "$LLM_BASE_URL/models" -H "Authorization: Bearer $LLM_API_KEY"
curl -N --fail-with-body "$LLM_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $LLM_API_KEY" -H 'Content-Type: application/json' \
  --data '{"model":"my-model","messages":[{"role":"user","content":"Hello"}],"stream":true,"max_tokens":512}'
```

For an installed OpenAI-compatible SDK:

```python
import os
from openai import OpenAI

client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
response = client.chat.completions.create(
    model=os.environ["LLM_MODEL"],
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

The SDK is optional; the repository CLI uses only the standard library.

## Validate an agent setup

After readiness and model discovery succeed, make one streamed chat request.
For Codex, also test `uv run llm-chat --profile my-model --responses --stream "Say ready"`.
Then ask the harness to read a small local file and report its contents. Confirm
that the tool call executes and that the model consumes the tool result. API
availability alone does not establish tool-call compatibility. Model templates,
parser settings, context limits, and server/client versions all matter.

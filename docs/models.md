# Adding models and serving backends

A model is a file in `config/models/ID.toml`. Its filename is the profile ID;
`model.alias` is the identifier sent over the API. IDs use lowercase letters,
digits, underscores, and hyphens. Commands also accept a TOML path, such as
`uv run llm-prepare examples/gguf-model.toml` after editing that template.

## GGUF models

Create a profile with `uv run llm-init`, or copy [the GGUF template](../examples/gguf-model.toml).
The runtime handles any supported GGUF architecture, regardless of model family.

```bash
uv run llm-init my-model --repo ORG/MODEL-GGUF --file QUANT/model-00001-of-00004.gguf
```

The generated download pattern includes all four shards. Preparation validates
that every shard is present and nonempty before recording the first shard as the
entry point. Missing or ambiguous entry points fail before submission.

Alternatively, use a local file:

```bash
uv run llm-init my-model --model-file /SHARED/STORAGE/model.gguf
```

The minimal local profile is:

```toml
[model]
alias = "my-model"
file = "models/my-model.gguf"
```

Choose `server.context_size` and `server.max_output` and adjust `[slurm]` for
larger models. The defaults are one GPU, eight CPUs, 64G host RAM, eight hours,
8192 total context, one request slot, and 2048 output tokens.

A download profile uses:

```toml
[model]
name = "My model"
alias = "my-model"
quant = "Q4_K_M"

[download]
repo = "ORG/MODEL-GGUF"
revision = "main"
directory = "models/my-model"
include = ["{quant}/*.gguf"]
entrypoint = "{quant}/*-00001-of-*.gguf"
```

`directory` is relative to the repository; `include` and `entrypoint` are relative
to that directory. `include` selects files on Hugging Face and `entrypoint` must
match exactly one local GGUF. For a single file, use its filename in both fields.
`{quant}` is optional; `uv run llm-prepare my-model --quant Q5_K_M` changes the recorded
weights for the next submission. A failed preparation leaves the previous
recorded entry point intact. Use a commit ID in `revision` for reproducible weights.

Hugging Face CLI downloads support cache reuse and authentication via its normal
login/token mechanism. Choose the quant explicitly so preparation does not fetch
every quantization in a repository.

## Server and resource settings

| Section | Settings |
| --- | --- |
| `[model]` | `name`, `alias`, `backend` (`llama.cpp` or `script`), local `file`, script `repo`, download `quant` |
| `[server]` | `host`, `port`, `context_size`, `parallel`, `max_output`, `extra_args` |
| `[server]`, llama.cpp | `gpu_layers`, `reasoning`, `reasoning_effort`, `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repeat_penalty`, `split_mode`, `tensor_split`, `lazy_mode`, `cache_type_k`, `cache_type_v`, `flash_attn` |
| `[slurm]` | `gpus`, `cpus_per_task`, `mem`, `time`, `partition`, `account`, `nodelist`, `constraint`, `gpu_type`, `qos` |
| `[capabilities]` | `tools`, `responses` (both default to `false`), optional `reasoning_efforts` |
| `[scripts]`, script backend | `serve`, optional `prepare` |

`reasoning` accepts `on`, `off`, or `auto`. Reasoning effort, sampling, cache types,
and special flags depend on the selected model/runtime. Unspecified sampling
settings use the engine defaults; no Qwen-specific sampling is applied globally.
Changing reasoning does not automatically change sampling—put appropriate values
in the profile. Backend-specific additions go in `extra_args`, as separate argv
items, e.g. `["--chat-template-file", "templates/my-model.jinja"]`.

`reasoning_efforts` lists the per-request effort levels the model accepts, e.g.
`["low", "medium", "high"]`. Generated OpenCode configs offer exactly these as
model variants; the default variant sends no effort, so `server.reasoning_effort`
applies. Omit it to keep OpenCode's built-in variants, or set `[]` to offer none.

For llama.cpp, `context_size` is the total context shared across `parallel` slots.
Clients are configured with `floor(context_size / parallel)` and output must be
smaller than that value. For custom backends this is the repository's context
budgeting convention; translate it to the engine's per-request setting using
`LLM_CONTEXT_PER_REQUEST`. The vLLM example does this.

llama.cpp receives an equal tensor split based on the actual Slurm allocation
when multiple GPUs are assigned. An explicit `tensor_split` overrides that
calculation. No physical device IDs or `CUDA_VISIBLE_DEVICES` values are forced.
Resources currently support one node with one serving task.

Cluster settings override profile resource suggestions, and submission flags
have highest priority:

```bash
uv run llm-submit my-model --gpus 2 --cpus-per-task 16 --mem 128G \
  --partition gpu --account project --time 04:00:00
uv run llm-submit my-model --port 9000 --context-size 16384 --parallel 2 --max-output 4096
```

Supported server environment overrides are `LLM_HOST`, `LLM_PORT`, `LLM_CTX_SIZE`,
`LLM_PARALLEL`, `LLM_MAX_OUTPUT`, `LLM_GPU_LAYERS`, `LLM_REASONING`,
`LLM_REASONING_EFFORT`, `LLM_TEMPERATURE`, `LLM_TOP_P`, `LLM_TOP_K`, `LLM_MIN_P`,
`LLM_PRESENCE_PENALTY`, `LLM_SPLIT_MODE`, `LLM_TENSOR_SPLIT`, `LLM_LAZY_MODE`,
`LLM_MODEL_ALIAS`, and `LLM_MODEL_FILE`. CLI flags override those variables.
`LLM_REASONING=true/false` is accepted for migration. Snapshots ignore later
server environment changes.

## Other serving engines and custom scripts

Copy [the script template](../examples/script-model.toml) or run:

```bash
uv run llm-init my-model --backend script --repo ORG/MODEL \
  --serve-script scripts/models/serve-my-model.sh \
  --prepare-script scripts/models/prepare-my-model.sh
```

Both scripts are Bash files relative to the repository root. Preparation runs on
the login node; serving runs through `srun` on the allocated compute node. Set up
large downloads or Python environments in the preparation script, avoiding
repeated setup in every allocation. `scripts.prepare` is optional; `uv run llm-prepare`
always ensures the key file exists after preparation. GGUF-specific preparation
flags do not apply to script profiles.

The scripts receive these variables, plus the normal Slurm environment:

| Variable | Value |
| --- | --- |
| `LLM_ROOT`, `LLM_PROFILE` | Repository root and profile ID |
| `LLM_MODEL_REPO`, `LLM_MODEL_FILE`, `LLM_MODEL_ALIAS` | Model source and API alias |
| `LLM_API_KEY_FILE` | Absolute path to the authentication key; never log its contents |
| `LLM_HOST`, `LLM_PORT` | Listening address and port |
| `LLM_CTX_SIZE`, `LLM_CONTEXT_PER_REQUEST` | Total budget and context per request |
| `LLM_PARALLEL`, `LLM_MAX_OUTPUT` | Concurrency and advertised output budget |
| `LLM_GPUS`, `LLM_CPUS` | Resolved resource requests |
| `HF_HOME` | Existing HF cache setting, or the repository's `.cache/huggingface` |

`server.extra_args` are passed as positional arguments to the serving script.
Use `exec` to replace the script with the server, preserving signals and exit
codes. The server must enforce authentication and expose `/v1/models` and
`/v1/chat/completions`; `/health` enables the readiness command. Advertise the
same alias, context, output budget, and capabilities used in the profile.

[The vLLM adapter](../examples/backends/vllm.sh) reads the key into the documented
`VLLM_API_KEY` environment variable and passes the common model/host/port/resource
settings to `vllm serve`. It requires a separately installed, model-compatible
vLLM environment; set `VLLM_BIN=/SHARED/VENV/bin/vllm`. Configure tool/reasoning
parsers in `extra_args` for the selected model. See the official
[vLLM serving options](https://docs.vllm.ai/en/latest/cli/serve/) and
[environment variables](https://docs.vllm.ai/en/latest/configuration/env_vars/).
The adapter is an extension example; the repository does not install vLLM.

For another agent harness, add a pure config renderer in
`slurm_llm/integrations.py` and expose it in `slurm_llm/commands/config.py`. Keep aliases,
endpoints, token budgets, and credentials derived from the same profile.

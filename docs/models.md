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
| `[model]` | `name`, `alias`, `backend` (`llama.cpp`, `vllm`, or `script`), local `file`, Hub `repo` (vLLM/script), download `quant` |
| `[server]` | `host`, `port`, `context_size`, `parallel`, `max_output`, `extra_args` |
| `[server]`, llama.cpp | `gpu_layers`, `reasoning`, `reasoning_effort`, `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repeat_penalty`, `split_mode`, `tensor_split`, `lazy_mode`, `cache_type_k`, `cache_type_v`, `flash_attn` |
| `[server]`, vLLM | `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty` (server-side defaults) |
| `[vllm]`, vLLM | See [vLLM models](#vllm-models) |
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
smaller than that value. For vLLM, `context_size` is the per-request limit
(`--max-model-len`) because all sequences share one paged KV cache; see below.
For custom backends the llama.cpp convention applies; translate it to the
engine's per-request setting using `LLM_CONTEXT_PER_REQUEST`.

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

## vLLM models

`backend = "vllm"` serves a Hugging Face checkpoint (safetensors) with
[vLLM](https://docs.vllm.ai/). Use it for FP8/NVFP4/AWQ checkpoints, high
concurrency, multimodal models, or models without a good GGUF. Create a profile
with `llm-init` or copy [the vLLM template](../examples/vllm-model.toml):

```bash
uv run llm-init my-model --backend vllm --repo ORG/MODEL \
  --reasoning-parser qwen3 --tool-call-parser hermes --tools
uv run llm-prepare my-model
```

`llm-prepare` does two things for vLLM profiles:

1. **Runtime.** Installs the vLLM version pinned in `config/runtime.toml`
   (`[vllm] version`) with uv into `.tools/vllm/VERSION`, using a uv-managed
   Python under `.tools/python`, so compute nodes can run the same interpreter.
   This runs once per version (several GB); later runs reuse it. Change the pin
   to upgrade; the old environment is left in place. `packages` adds extra pins
   (e.g. a newer `transformers`), `overrides` replaces vLLM's own pins via uv's
   `--overrides` (e.g. `flashinfer-python==0.7.1`), and `torch_backend` passes
   uv's `--torch-backend` (e.g. `cu129`) when the default PyPI torch build does
   not suit your driver. Changing `packages`, `overrides` or `torch_backend`
   reinstalls into the same environment on the next prepare.
   Set `VLLM_BIN=/PATH/bin/vllm` to use your own install instead.
2. **Weights.** Downloads the `[download]` repository on the login node into
   `directory`, then checks `config.json` and every shard listed in
   `model.safetensors.index.json`. `include` defaults to configs, tokenizers,
   chat templates, and `*.safetensors`, which skips `.bin` and GGUF duplicates;
   narrow it if a repository carries a second safetensors copy (e.g. `original/`).
   `entrypoint` is not used: vLLM loads the directory.

Model sources, in priority order: `model.file` (an existing local model
directory), `[download]` (prepared locally), or `model.repo` alone (vLLM
downloads into `HF_HOME` when the job starts, which needs compute-node internet
and delays startup). With local weights, jobs set `HF_HUB_OFFLINE=1`.
`--quant` substitutes `{quant}` in download fields as for GGUF profiles.

### Context, concurrency, and output

| Profile | vLLM flag | Meaning |
| --- | --- | --- |
| `server.context_size` | `--max-model-len` | Maximum prompt + output tokens per request |
| `server.parallel` | `--max-num-seqs` | Maximum concurrently running requests |
| `server.max_output` | `max_new_tokens` default | `max_tokens` for requests that omit it |
| `slurm.gpus` | `--tensor-parallel-size` | Unless `vllm.tensor_parallel_size` is set |

All sequences share one KV-cache pool sized by `gpu_memory_utilization`, so
`context_size` is not divided by `parallel`: harness configs advertise the full
`context_size`. vLLM refuses to start if one `context_size` sequence cannot fit
in the pool. When many long requests run together, it queues or preempts them.
The startup log reports the pool size ("GPU KV cache size: N tokens") and the
maximum concurrency at full length; tune `context_size`, `parallel`,
`kv_cache_dtype`, and `gpu_memory_utilization` from that line. Unlike
llama.cpp's `--n-predict`, `max_output` is a default, not a cap: clients may ask
for up to the remaining context.

Sampling keys become `--override-generation-config` defaults on top of the
checkpoint's `generation_config.json`; requests can still override them.
`repeat_penalty` maps to vLLM's `repetition_penalty`. vLLM cannot set a default
`presence_penalty`, so that key is rejected; send it per request.

### Engine options

| `[vllm]` key | vLLM flag / effect |
| --- | --- |
| `tensor_parallel_size`, `pipeline_parallel_size` | GPU split; tensor x pipeline must fit the allocation. Tensor defaults to allocated GPUs / pipeline |
| `enable_expert_parallel` | MoE models: spread experts across the GPUs instead of slicing each expert |
| `gpu_memory_utilization` | Fraction of each GPU for weights + KV cache (vLLM default 0.9) |
| `max_num_batched_tokens` | Prefill chunk budget per step |
| `dtype`, `quantization`, `kv_cache_dtype` | Usually auto-detected; `kv_cache_dtype = "fp8"` roughly doubles KV capacity |
| `enable_prefix_caching` | `true`/`false`: `--enable-prefix-caching` / `--no-enable-prefix-caching` |
| `enforce_eager`, `trust_remote_code`, `language_model_only` | Switches; `language_model_only` skips a multimodal model's vision tower |
| `reasoning_parser` | Splits thinking into `reasoning_content` (e.g. `qwen3`, `deepseek_r1`) |
| `tool_call_parser` | Enables `--enable-auto-tool-choice` with this parser; **required** by `capabilities.tools` |
| `chat_template` | Template path, relative to the repository or absolute; checked before launch |
| `chat_template_kwargs` | Server-wide `--default-chat-template-kwargs`, e.g. `{ enable_thinking = false }` |
| `generation_config` | `auto` (checkpoint), `vllm` (ignore checkpoint), or a directory |
| `speculative_config`, `hf_overrides` | Tables passed as JSON, e.g. `{ method = "mtp", num_speculative_tokens = 3 }` |
| `env` | Extra environment variables for the server, e.g. `{ VLLM_ATTENTION_BACKEND = "FLASHINFER" }` |

Any other `vllm serve` flag goes in `server.extra_args`. Use the
[parser names and flags for your pinned version](https://docs.vllm.ai/en/latest/cli/serve/);
model cards and [vLLM recipes](https://recipes.vllm.ai/) list the right parsers.
llama.cpp-only `[server]` keys (`reasoning`, `cache_type_k`, `tensor_split`, ...)
fail validation with the vLLM equivalent. Matching `LLM_*` environment
overrides are ignored for vLLM profiles. Declared `reasoning_efforts` must be
values vLLM accepts (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`).
vLLM passes the request's effort to the chat template.

### Authentication

The key is handed to vLLM through `VLLM_API_KEY`, never on the command line.
vLLM's own check only covers `/v1`, `/v2`, and `/inference`, leaving routes such
as `/invocations` open
([details](https://docs.vllm.ai/en/latest/usage/security.html#api-key-authentication-limitations)).
Every vLLM job therefore also loads a stdlib ASGI middleware
(`slurm_llm/vllm_middleware/slurm_llm_auth.py`) that requires the bearer key on all
routes except `/health`, matching llama.cpp. Usage statistics are disabled with
`VLLM_NO_USAGE_STATS=1` unless you set it.

### Runtime overrides

```bash
VLLM_BIN=/SHARED/venv/bin/vllm uv run llm-prepare my-model   # validate, skip install
uv run llm-submit my-model --vllm /SHARED/venv/bin/vllm       # one submission
uv run llm-serve my-model --dry-run                           # print the full command
```

Submission records the resolved `vllm` executable and the prepared checkpoint
directory in the job snapshot.

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

[The script template](../examples/backends/custom.sh) shows the shape of an
adapter. vLLM profiles that used a script with `examples/backends/vllm.sh` should
switch to `backend = "vllm"`: remove `[scripts]`, move parser flags from
`extra_args` into `[vllm]`, and use the per-request `context_size` described above.

For another agent harness, add a pure config renderer in
`slurm_llm/integrations.py` and expose it in `slurm_llm/commands/config.py`. Keep aliases,
endpoints, token budgets, and credentials derived from the same profile.

# Slurm LLM

Prepare and serve open source models on a Slurm GPU cluster, then use their
OpenAI-compatible endpoints from Python, OpenCode, Codex, or another agent harness.

Models live in small TOML profiles. Downloads, authentication, Slurm submission,
SSH tunnels, and client configuration share one implementation. Two engines are
built in: llama.cpp serves GGUF models and vLLM serves Hugging Face (safetensors)
checkpoints. Both are installed and pinned by this repository. A script backend
supports any other engine.

## Quick start

Run these commands from a checkout on storage shared by the login and compute
nodes. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
ensure `curl` is available for preparation. Run `uv sync` in the checkout to
create `.venv` and install the command entry points and development tools.
Standalone checkouts pin Python 3.12; uv can download it if unavailable locally.
Weights use the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli),
launched through `uvx` when `hf` is unavailable.

```bash
uv sync

# 1. Configure this cluster once; edit the local copy for your partition/account.
if [ ! -f config/cluster.local.toml ]; then
  cp config/cluster.example.toml config/cluster.local.toml
fi
# Edit config/cluster.local.toml for this cluster.

# 2. Choose a profile and prepare it.
uv run llm-list
uv run llm-prepare qwen-27b

# 3. Preview, then submit.
uv run llm-submit qwen-27b --dry-run
job_id=$(uv run llm-submit qwen-27b)
squeue -j "$job_id"
tail -f "logs/qwen-27b-$job_id.log"
```

The job log reports the compute node's DNS hostname and server port. On your
laptop, use a checkout to create a tunnel (replace the placeholders):

```bash
uv run llm-tunnel qwen-27b \
  --login USER@HEADNODE --compute-host COMPUTE_HOST --execute
```

Keep that terminal open. In your local checkout, copy the cluster key to
`.secrets/llm-api-key`, which this repository ignores in Git. Then use it
from another local terminal:

```bash
mkdir -p .secrets
chmod 700 .secrets
scp USER@HEADNODE:/PATH/TO/llms/.secrets/llm-api-key .secrets/llm-api-key
chmod 600 .secrets/llm-api-key
export LLM_API_KEY_FILE="$PWD/.secrets/llm-api-key"
export LLM_API_KEY="$(cat "$LLM_API_KEY_FILE")"

uv run llm-health --profile qwen-27b --wait 300
uv run llm-models --profile qwen-27b
uv run llm-chat --profile qwen-27b "Explain Slurm in three sentences."
```

Stop the allocation on the login node when finished: `scancel "$job_id"`.
Preparation retains existing API keys and downloads. No model download or GPU
allocation happens when listing profiles or generating configs.

## Independent commands

Each operation has its own executable and Python module. Use uv to install and
run the commands in the project environment:

```bash
uv sync
uv run llm-list
uv run llm-prepare glm-flash
uv run llm-submit glm-flash --test-only
```

| Command              | Purpose                                                       |
| -------------------- | ------------------------------------------------------------- |
| `uv run llm-list`    | List model profiles in the checkout                           |
| `uv run llm-init`    | Create a model profile                                        |
| `uv run llm-prepare` | Prepare runtime, weights, and authentication                  |
| `uv run llm-submit`  | Preview, validate, or submit a Slurm job                      |
| `uv run llm-serve`   | Serve within an allocation or directly on a compute node      |
| `uv run llm-tunnel`  | Print or start an SSH tunnel                                  |
| `uv run llm-config`  | Generate OpenCode/Codex configuration or environment commands |
| `uv run llm-health`  | Check endpoint readiness                                      |
| `uv run llm-models`  | List models served by the endpoint                            |
| `uv run llm-chat`    | Send a prompt, optionally streaming                           |

`uv run` uses the project's environment and keeps it synchronized with `uv.lock`.
You do not need to activate the environment. Use `uv run llm-chat --help` (or any
other command) for its options. See [uv's project workflow](https://docs.astral.sh/uv/guides/projects/).

From another directory, select both the uv project and the model checkout:

```bash
uv run --project /PATH/TO/llms llm-submit \
  --root /PATH/TO/llms my-model --dry-run
```

`--project` selects uv's environment; `--root` selects profiles and model assets.
You can set `LLM_ROOT` instead of passing `--root`. Keep the checkout and `.venv`
visible to compute nodes. Submission records the uv environment's Python
interpreter, which the batch wrapper invokes directly with the frozen job settings.
Sync the environment on the login node before submission.

If the checkout is a member of a parent uv workspace, uv uses that workspace's
environment, lockfile, and Python constraints. See
[uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/).

## Add a model

For any GGUF supported by your llama.cpp version:

```bash
uv run llm-init my-model --repo ORG/MODEL-GGUF --file model-Q4_K_M.gguf
# Edit config/models/my-model.toml: resources, context, sampling, capabilities.
uv run llm-prepare my-model
uv run llm-submit my-model
```

For a split GGUF, give `--file` the first shard; initialization selects all shards
for download. For existing weights, use `uv run llm-init my-model --model-file /PATH/model.gguf`.
No model-specific Python or Slurm script is needed.

For a safetensors checkpoint with vLLM:

```bash
uv run llm-init my-model --backend vllm --repo ORG/MODEL
# Edit config/models/my-model.toml: [server] context/concurrency, [vllm] engine options.
uv run llm-prepare my-model   # installs the pinned vLLM once, then downloads weights
uv run llm-submit my-model
```

The included vLLM profiles show model-specific parser and engine settings. Choose
resources for your hardware; see [vLLM models](docs/models.md#vllm-models).

For another engine, write a Bash serve script (`--backend script --serve-script PATH`).
See [adding models and backends](docs/models.md) for the script contract.

## Connect agent harnesses

Configs are generated from your model's alias, endpoint, capabilities, and context.
OpenCode integration supports only version 2.x; check `opencode --version` first.

```bash
# OpenCode 2.x.
uv run llm-config opencode my-model --output /PATH/TO/YOUR/PROJECT/opencode.json

# Codex: merge the provider table into ~/.codex/config.toml and save the
# remaining profile keys as ~/.codex/slurm_my_model.config.toml.
uv run llm-config codex my-model --output config/generated/my-model-codex.toml
codex --profile slurm_my_model
```

Export `LLM_API_KEY` in the shell that starts the harness. Config generation
refuses to overwrite existing files; merge with an existing configuration when
needed. New model profiles default to disabled tool/Responses capabilities.
Enable them after validating the model and runtime. Codex config generation
requires both. See [harness setup and API examples](docs/harnesses.md).

## Included profiles

Resource sizes are starting configurations; choose weights and context that fit
your hardware. Profiles leave cluster node, account, and partition unset. The GLM
NVFP4 profile selects a Blackwell GPU type; match `gpu_type` to your scheduler's
names in `config/cluster.local.toml` or pass `--gpu-type` when submitting.

| Profile          | Backend   | API model            |
| ---------------- | --------- | -------------------- |
| `qwen-27b`       | llama.cpp | `qwen3.8-27b`        |
| `qwen-27b-vllm`  | vLLM      | `qwen3.8-27b`        |
| `qwen-next`      | llama.cpp | `qwen3.8-flash-next` |
| `glm-flash`      | llama.cpp | `glm-5.3-flash`      |
| `glm-flash-vllm` | vLLM      | `glm-5.3-flash`      |

Shared runtime downloads remain pinned to llama.cpp `b11472` with CUDA 12.8 and
verified SHA-256 hashes. Those binaries require Linux x86_64 and a compatible
NVIDIA driver. Set `LLAMA_SERVER=/PATH/llama-server` to use your own build.
vLLM is pinned to the version in `config/runtime.toml` and installed with uv into
`.tools/vllm/VERSION-HASH`, where the hash identifies the install settings.
Set `VLLM_BIN=/PATH/bin/vllm` to use your own install.

## Layout

```text
config/models/         Model profiles: weights, resources, server options, capabilities
config/runtime.toml    Pinned llama.cpp release/checksums and vLLM version
uv.lock, .python-version  Locked project dependencies and Python version
config/cluster.local.toml  Your cluster settings (ignored by Git)
slurm_llm/             Configuration, preparation, backends, Slurm, clients, integrations
slurm_llm/vllm.py      vLLM installation and command construction
slurm_llm/vllm_middleware/  Authentication loaded by vLLM's interpreter
slurm_llm/commands/    Independent command parsers, handlers, and entry points
slurm/serve.sbatch      One batch entry point for every model
examples/              GGUF, vLLM, and script profile templates; a script backend template
scripts/prepare.sh     Optional Bash wrapper for `uv run llm-prepare PROFILE`
docs/                  Model extension, harness setup, operations
tests/                 Unit and local HTTP integration tests
models/, .tools/       Downloaded weights, runtime, prepared paths, job snapshots (ignored)
.secrets/, logs/       Credentials and job logs (ignored)
```

Development checks:

```bash
uv run python -m unittest discover -s tests -v
uv run ruff check .
uv run ruff format --check .
for script in scripts/*.sh slurm/*.sbatch examples/backends/*.sh; do
  bash -n "$script" || exit
done
```

The tests cover profile validation, shard completeness, credential retention,
submission snapshots, resource overrides, generated configs, and HTTP/streaming
clients against a local test server. GitHub Actions runs the same checks on pushes
and pull requests. Live GPU inference and actual agent tool
loops must be checked against the chosen model. See
[operations](docs/operations.md) for troubleshooting.

# Slurm LLM

Prepare and serve open source models on a Slurm GPU cluster, then use their
OpenAI-compatible endpoints from Python, OpenCode, Codex, or another agent harness.

Models live in small TOML profiles. Downloads, authentication, Slurm submission,
SSH tunnels, and client configuration share one implementation. The built-in
backend serves GGUF models with llama.cpp; the script backend supports other
engines, with a vLLM example included.

## Quick start

Run these commands from a checkout on storage shared by the login and compute
nodes. Python 3.12+ is required. Preparation also needs `curl` and either the
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) or `uv`
(the command automatically uses `uvx` when `hf` is unavailable).

```bash
# 1. Configure this cluster once; edit the local copy for your partition/account.
if [ ! -f config/cluster.local.toml ]; then
  cp config/cluster.example.toml config/cluster.local.toml
fi
# Edit config/cluster.local.toml for this cluster.

# 2. Choose a profile and prepare it.
python3 main.py list
python3 main.py prepare qwen-27b

# 3. Preview, then submit.
python3 main.py submit qwen-27b --dry-run
job_id=$(python3 main.py submit qwen-27b)
squeue -j "$job_id"
tail -f "logs/qwen-27b-$job_id.log"
```

The job log reports the compute node's DNS hostname and server port. On your
laptop, use a checkout to create a tunnel (replace the placeholders):

```bash
python3 main.py tunnel qwen-27b \
  --login USER@HEADNODE --compute-host COMPUTE_HOST --execute
```

Keep that terminal open. Copy `.secrets/llm-api-key` from the cluster to a private
local file, then use it from another local terminal:

```bash
mkdir -p "$HOME/.config/slurm-llm"
chmod 700 "$HOME/.config/slurm-llm"
scp USER@HEADNODE:/PATH/TO/llms/.secrets/llm-api-key "$HOME/.config/slurm-llm/api-key"
chmod 600 "$HOME/.config/slurm-llm/api-key"
export LLM_API_KEY_FILE="$HOME/.config/slurm-llm/api-key"
export LLM_API_KEY="$(cat "$LLM_API_KEY_FILE")"

python3 main.py health --profile qwen-27b --wait 300
python3 main.py models --profile qwen-27b
python3 main.py chat --profile qwen-27b "Explain Slurm in three sentences."
```

Stop the allocation on the login node when finished: `scancel "$job_id"`.
Preparation retains existing API keys and downloads. No model download or GPU
allocation happens when listing profiles or generating configs.

## Use the shorter `llm` command

The CLI itself has no third-party Python dependencies. Install it in a shared
virtual environment if you prefer `llm` to `python3 main.py`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
llm list
llm prepare glm-flash
llm submit glm-flash --test-only
```

Every example using `llm` also works with `python3 main.py` or
`python3 -m slurm_llm`. From another directory, pass `llm --root /PATH/TO/llms …`
or set `LLM_ROOT`. Keep the checkout and the chosen Python interpreter visible
to compute nodes; submissions record that interpreter's absolute path.

## Add a model

For any GGUF supported by your llama.cpp version:

```bash
llm init my-model --repo ORG/MODEL-GGUF --file model-Q4_K_M.gguf
# Edit config/models/my-model.toml: resources, context, sampling, capabilities.
llm prepare my-model
llm submit my-model
```

For a split GGUF, give `--file` the first shard; initialization selects all shards
for download. For existing weights, use `llm init my-model --model-file /PATH/model.gguf`.
No model-specific Python or Slurm script is needed.

For another serving engine:

```bash
llm init my-model --backend script --repo ORG/MODEL \
  --serve-script examples/backends/vllm.sh
# Install a suitable vLLM environment and set VLLM_BIN to its executable.
# Edit the profile and add model-specific engine options.
llm prepare my-model
llm submit my-model
```

You can supply your own Bash preparation and serving scripts. See
[adding models and backends](docs/models.md) for the script contract and examples.

## Connect agent harnesses

Configs are generated from your model's alias, endpoint, capabilities, and context:

```bash
# OpenCode 2.x; use --opencode-version 1 for 1.x.
llm config opencode my-model --output /PATH/TO/YOUR/PROJECT/opencode.json

# Codex: generate a fragment, then merge it into your local ~/.codex/config.toml.
llm config codex my-model --output config/generated/my-model-codex.toml
codex --profile slurm_my_model
```

Export `LLM_API_KEY` in the shell that starts the harness. Config generation
refuses to overwrite existing files; merge with an existing configuration when
needed. New model profiles default to disabled tool/Responses capabilities.
Enable them after validating the model and runtime. Codex config generation
requires both. See [harness setup and API examples](docs/harnesses.md).

## Included profiles

These preserve the original model choices and resource requests. Resource sizes
are starting configurations; choose weights and context that fit your hardware.
Tracked profiles have no fixed cluster node, account, partition, or GPU type.

| Profile | API model | GPUs | CPUs | Host RAM | Context / output | Port |
| --- | --- | ---: | ---: | --- | --- | ---: |
| `qwen-27b` | `qwen3.8-27b` | 1 | 8 | 96G | 32768 / 8192 | 8000 |
| `qwen-next` | `qwen3.8-flash-next` | 4 | 16 | 128G | 32768 / 8192 | 8000 |
| `glm-flash` | `glm-5.3-flash` | 5 | 64 | 256G | 131072 / 16384 | 8001 |

Shared runtime downloads remain pinned to llama.cpp `b11472` with CUDA 12.8 and
verified SHA-256 hashes. Those binaries require Linux x86_64 and a compatible
NVIDIA driver. Set `LLAMA_SERVER=/PATH/llama-server` to use your own build.

## Layout

```text
config/models/         Model profiles: weights, resources, server options, capabilities
config/runtime.toml    Shared runtime release and checksums
config/cluster.local.toml  Your cluster settings (ignored by Git)
slurm_llm/             Configuration, preparation, backends, Slurm, clients, integrations
slurm/serve.sbatch      One batch entry point for every model
examples/              GGUF/script profile templates and a vLLM adapter
scripts/prepare.sh     Optional Bash wrapper for `llm prepare PROFILE`
docs/                  Model extension, harness setup, operations, and migration
tests/                 Unit and local HTTP integration tests
models/, .tools/       Downloaded weights, runtime, prepared paths, job snapshots (ignored)
.secrets/, logs/       Credentials and job logs (ignored)
```

Development checks:

```bash
python3 -m unittest discover -s tests -v
uvx ruff check .
uvx ruff format --check .
bash -n scripts/prepare.sh slurm/serve.sbatch examples/backends/vllm.sh
```

The tests cover profile validation, shard completeness, credential retention,
submission snapshots, resource overrides, generated configs, and HTTP/streaming
clients against a local test server. GitHub Actions runs the same checks on pushes
and pull requests. Live GPU inference and actual agent tool
loops must be checked against the chosen model. See
[operations and migration](docs/operations.md) for troubleshooting and the
replacement commands for the old model-specific scripts.

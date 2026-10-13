# Operations and migration

## Slurm lifecycle

Submit from the login node using storage visible to compute nodes:

```bash
uv run llm-submit my-model --dry-run     # Preview settings without downloading models or calling Slurm.
uv run llm-submit my-model --test-only   # Validate with Slurm; do not allocate resources.
job_id=$(uv run llm-submit my-model)
squeue -j "$job_id"
tail -f "logs/my-model-$job_id.log"
scancel "$job_id"
```

Submission checks the model/shards, executable or serving script, and key before
calling Slurm. It writes a private JSON snapshot under `.tools/jobs/`. The job
loads that snapshot, so later model-path selections or profile/environment
changes do not alter a queued job. Backend scripts, their environments, and runtime
binaries must still remain available; their contents are not copied into the
snapshot. Keep snapshots while their jobs are queued/running. Snapshots and logs
can be removed manually after the jobs finish.

The server stays in the foreground and is launched with `exec srun`, allowing
Slurm cancellation and exit codes to propagate. For an existing allocation:

```bash
# Within an allocation, run the shared server through srun:
uv run llm-serve my-model
# On an already allocated compute node, run the backend directly:
uv run llm-serve my-model --local
# Inspect the resolved server command without starting it:
uv run llm-serve my-model --dry-run
```

`uv run llm-serve --dry-run` validates existing model/runtime/key assets. `uv run llm-submit --dry-run`
can preview an unprepared profile. Its printed snapshot path is illustrative;
the command creates no snapshot until an actual submission/check. The `uv run`
wrapper may synchronize the project environment before the preview.

## Cluster and SSH configuration

Copy `config/cluster.example.toml` to `config/cluster.local.toml` and edit it once.
Only the local copy is ignored by Git. All `[slurm]` fields available in a model
profile can also be set there; local cluster values override portable profile
suggestions. Flags override both. Use `--cluster path.toml` with `uv run llm-submit`/`uv run llm-tunnel`
for a different cluster. Omitting partition/account/node/type lets Slurm choose
its defaults. Requests always use one node and one serving task.

By default the server listens on `0.0.0.0` and requires its key. The tunnel binds
the laptop listener to `127.0.0.1`, forwarding through the login node to the
compute node's DNS hostname. Read that hostname from the job log; Slurm node
names may differ from DNS names. This mode needs login-to-compute TCP access.

If compute-node SSH is allowed, bind the server to loopback and use ProxyJump:

```bash
uv run llm-submit my-model --host 127.0.0.1
# Laptop:
uv run llm-tunnel my-model --host 127.0.0.1 \
  --login USER@HEADNODE --compute-host COMPUTE_HOST --execute
```

`--jump` explicitly chooses compute-node SSH for a server listening on other
interfaces. Use `--local-port 9000` when a laptop port is busy, then set client
and harness `--base-url http://127.0.0.1:9000/v1`. Changing the server port at
submission requires the same `--port` for the tunnel; a job snapshot can also
supply the original server host/port.

## Preparation and storage

`uv run llm-prepare PROFILE` installs the profile's engine once (llama.cpp
binaries or the pinned vLLM environment), downloads selected model files with the
HF CLI, records the validated GGUF entry point or vLLM checkpoint directory, and
ensures a private key file. It resumes/reuses downloads through the downloader's cache.

```bash
uv run llm-prepare my-model --runtime-only
uv run llm-prepare my-model --model-only
LLAMA_SERVER=/SHARED/bin/llama-server uv run llm-prepare my-model
uv run llm-prepare qwen-next --quant UD-Q5_K_XL
uv run llm-prepare qwen-27b-vllm --runtime-only   # vLLM into .tools/vllm/VERSION
VLLM_BIN=/SHARED/venv/bin/vllm uv run llm-prepare qwen-27b-vllm
```

Runtime archive checksums and extraction filtering are enforced when installing.
Existing matching runtime installs are reused. To change a release, update
`config/runtime.toml`, including both archive hashes and the install directory.
A custom `LLAMA_SERVER` uses your own library environment; matching prepared
binaries retain their recorded library paths, including through job snapshots.
The vLLM environment, its uv-managed Python (`.tools/python`), and the uv cache
(`.cache/uv`) all live in the checkout so compute nodes see them. Changing
`[vllm] version` installs a new environment next to the old one; queued jobs keep
the executable recorded in their snapshot. vLLM also compiles CUDA graphs on first
start and caches them under `~/.cache/vllm` (`VLLM_CACHE_ROOT`), so the second
start of the same profile is faster.

Leave every shard of a split model together. Allow enough shared storage for
weights, partial downloads, and runtime archives. GPU memory also needs room
for KV caches and compute buffers; reduce context/concurrency or choose a
smaller quant when necessary. This repository does not delete cached weights
or interrupt existing allocations during preparation.

## Migration from the single CLI

The former `llm COMMAND ...`, `python3 main.py COMMAND ...`, and
`python3 -m slurm_llm COMMAND ...` forms have been replaced by independent commands:
`uv run llm-COMMAND ...`. Run `uv sync` to install the entry points.
Each command accepts
its own `--root` option; no subcommand dispatch is used.

The shared Slurm wrapper runs the serve module directly; job snapshots and model
profiles keep the same format. Existing queued jobs using that wrapper can still
load their snapshots.

## Migration from the original repository

The old duplicated model-specific download scripts, Slurm files, and static
agent configs have been replaced by profiles and generated configs:

| Old command/config | Replacement |
| --- | --- |
| `bash scripts/prepare.sh` | `uv run llm-prepare qwen-27b` or `bash scripts/prepare.sh qwen-27b` |
| `bash scripts/prepare-next.sh [QUANT]` | `uv run llm-prepare qwen-next [--quant QUANT]` |
| `bash scripts/prepare-glm.sh [QUANT]` | `uv run llm-prepare glm-flash [--quant QUANT]` |
| `sbatch slurm/serve.sbatch` | `uv run llm-submit qwen-27b` |
| `sbatch slurm/serve-next.sbatch` | `uv run llm-submit qwen-next` |
| `sbatch slurm/serve-glm.sbatch` | `uv run llm-submit glm-flash` |
| `config/opencode-*.json` | `uv run llm-config opencode PROFILE` (OpenCode 2.x only) |
| `config/codex.toml` | `uv run llm-config codex PROFILE` |
| `python3 main.py --models` | `uv run llm-models --profile qwen-27b` |
| `python3 main.py "PROMPT"` | `uv run llm-chat --profile qwen-27b "PROMPT"` |
| `QWEN_API_KEY` / `GLM_API_KEY` | `LLM_API_KEY` |
| `QWEN_BASE_URL` / `QWEN_MODEL` | `LLM_BASE_URL` / `LLM_MODEL` |

Existing files in `models/`, `.tools/llama-cpp/`, and `.secrets/llm-api-key` remain
usable. The old default downloaded paths are recognized without preparation.
New quant selections are recorded under `.tools/profiles/PROFILE/model.path`.
If the old `.tools/qwen-next/model.path` or `.tools/glm-flash/model.path` selected
a non-default quant, run the new `uv run llm-prepare --quant` command to record that choice,
or pass `--model-file` explicitly. API aliases remain the same; generated
provider/profile IDs are now based on profile filenames.

The existing checkout's gpu8/Blackwell partition settings have been retained in
an ignored `config/cluster.local.toml`. New clones start from the portable example.
GLM's original scripted settings are now authoritative: port 8001, 32 CPUs,
256G host RAM, and high reasoning effort. The old README's conflicting port/CPU/
reasoning descriptions are no longer used.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Job pending | `squeue -j JOB_ID`; inspect partition/account and requested resources |
| Submission fails | `uv run llm-submit PROFILE --test-only`; confirm prepared weights/runtime/key |
| Missing shard | Rerun preparation for the selected quant; retain every GGUF shard |
| Local connection refused | Tunnel foreground process, actual compute hostname, server port, job log |
| HTTP 503 | Model still loading; `uv run llm-health --profile PROFILE --wait 300` |
| vLLM: endpoint not reachable for minutes | Normal while vLLM loads weights and captures CUDA graphs; the port opens only when ready. Follow the job log |
| HTTP 401 | Load the same key that the server's key file contains |
| Local port busy | `uv run llm-tunnel --local-port` and client/config `--base-url` |
| Server port busy | `uv run llm-submit --port`, then regenerate the tunnel and configs |
| CUDA errors | Compatible NVIDIA driver/runtime and correct `LLAMA_SERVER` library environment; for vLLM, `[vllm] torch_backend` |
| vLLM: "max_model_len ... larger than KV cache" | Lower `context_size`, set `kv_cache_dtype = "fp8"`, raise `gpu_memory_utilization`, or add GPUs |
| vLLM: OOM during CUDA graph capture | `enforce_eager = true`, or lower `gpu_memory_utilization`/`parallel` |
| vLLM: tool calls arrive as plain text | `vllm.tool_call_parser` matches the model's tool format |
| Out of GPU memory | Smaller quant, context, or concurrency; inspect per-GPU usage |
| Tool/Responses errors | Model template, parser, declared capabilities, actual harness tool loop |

## Checks

Run `uv run python -m unittest discover -s tests -v`, `uv run ruff check .`,
`uv run ruff format --check .`, and Bash syntax validation from the README.
Tests use temporary files and a localhost HTTP fixture; they do not download
weights, require GPUs, or submit jobs. Slurm's `--test-only` validates scheduling
parameters separately. Live inference, laptop forwarding, and agent tool loops
remain environment-specific checks.

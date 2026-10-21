# Operations

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
uv run llm-prepare qwen-27b-vllm --runtime-only
VLLM_BIN=/SHARED/venv/bin/vllm uv run llm-prepare qwen-27b-vllm
```

llama.cpp archive checksums and extraction filtering are enforced when installing.
Existing matching installs are reused. To change its release, update
`config/runtime.toml`, including both archive hashes and the install directory.
A custom `LLAMA_SERVER` uses your own library environment; matching prepared
binaries retain their recorded library paths, including through job snapshots.
The vLLM environment, its uv-managed Python (`.tools/python`), and the uv cache
(`.cache/uv`) default to shared storage in the checkout. Set
`UV_PYTHON_INSTALL_DIR` to override the interpreter location or `UV_CACHE_DIR`
to override the cache location.
Changing any vLLM install setting selects a new `.tools/vllm/VERSION-HASH`
environment; queued jobs retain their previous executable. The prepared runtime
exposes a wheel-provided CUDA compiler and linkable CUDA libraries for JIT kernels.
vLLM's compilation cache location can be set with `VLLM_CACHE_ROOT`; CUDA graph
capture still happens during startup.

Leave every shard of a split model together. Allow enough shared storage for
weights, partial downloads, and runtime archives. GPU memory also needs room
for KV caches and compute buffers; reduce context/concurrency or choose a
smaller quant when necessary. This repository does not delete cached weights
or interrupt existing allocations during preparation.

## Troubleshooting

| Symptom                                        | Check                                                                                                             |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Job pending                                    | `squeue -j JOB_ID`; inspect partition/account and requested resources                                             |
| Submission fails                               | `uv run llm-submit PROFILE --test-only`; confirm prepared weights/runtime/key                                     |
| Missing shard                                  | Rerun preparation for the selected quant; retain every GGUF shard                                                 |
| Local connection refused                       | Tunnel foreground process, actual compute hostname, server port, job log                                          |
| HTTP 503                                       | Model still loading; `uv run llm-health --profile PROFILE --wait 300`                                             |
| vLLM: endpoint not reachable for minutes       | Normal while vLLM loads weights and captures CUDA graphs; the port opens only when ready. Follow the job log      |
| HTTP 401                                       | Load the same key that the server's key file contains                                                             |
| Local port busy                                | `uv run llm-tunnel --local-port` and client/config `--base-url`                                                   |
| Server port busy                               | `uv run llm-submit --port`, then regenerate the tunnel and configs                                                |
| CUDA errors                                    | Compatible NVIDIA driver/runtime and correct `LLAMA_SERVER` library environment; for vLLM, `[vllm] torch_backend` |
| vLLM: "max_model_len ... larger than KV cache" | Lower `context_size`, set `kv_cache_dtype = "fp8"`, raise `gpu_memory_utilization`, or add GPUs                   |
| vLLM: OOM during CUDA graph capture            | `enforce_eager = true`, or lower `gpu_memory_utilization`/`parallel`                                              |
| vLLM: tool calls arrive as plain text          | `vllm.tool_call_parser` matches the model's tool format                                                           |
| Out of GPU memory                              | Smaller quant, context, or concurrency; inspect per-GPU usage                                                     |
| Tool/Responses errors                          | Model template, parser, declared capabilities, actual harness tool loop                                           |

## Checks

Run `uv run python -m unittest discover -s tests -v`, `uv run ruff check .`,
`uv run ruff format --check .`, and Bash syntax validation from the README.
Tests use temporary files and a localhost HTTP fixture; they do not download
weights, require GPUs, or submit jobs. Slurm's `--test-only` validates scheduling
parameters separately. Live inference, laptop forwarding, and agent tool loops
remain environment-specific checks.

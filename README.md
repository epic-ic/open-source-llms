# Qwen3.8 on Slurm, accessed from your laptop

Yes: run an API server in a GPU Slurm job, then use an SSH tunnel from your
local machine. Curl, Python, OpenCode, and Codex CLI can use that endpoint.
Agent tools and file edits run wherever you start the coding client; only model
inference runs on the GPU node.

```text
Laptop: curl / Python / OpenCode / Codex CLI
  http://127.0.0.1:8000/v1
          |
      SSH tunnel
          |
Headnode: wpia-lxhn.hpc.dide.ic.ac.uk
          |
Compute node: gpu7 (DNS: wpia-gpu-07.dide.ic.ac.uk)
  Slurm job -> llama-server -> Unsloth Qwen3.8-27B GGUF
```

This project uses the model you selected,
[unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF),
with the `UD-Q4_K_XL` quant (about 17.6 GB of weights). One of gpu7's L40S GPUs
should have sufficient VRAM for this quant and the initial 32K context; runtime
buffers and context also use memory. You can leave the other three GPUs available.
See [Unsloth's Qwen3.8 guide](https://unsloth.ai/docs/models/qwen3.8).

Unsloth supplies the quantized model; `llama.cpp` supplies the GPU runtime and
HTTP API. Unsloth documents this serving route for its GGUFs. You do not need the
Unsloth Python training package for this setup.

## 1. Prepare on the headnode

```bash
cd /mnt/gpu-storage/homes/athapar/code/llms
bash scripts/prepare.sh
```

Preparation downloads the chosen GGUF and the official `llama.cpp` CUDA 12.8
binaries/runtime from [release b11472](https://github.com/ggml-org/llama.cpp/releases/tag/b11472).
It verifies the release archive hashes and creates `.secrets/llm-api-key` with a
random key. Existing keys are retained. Downloads can resume after interruption.
Allow about 25-30 GB of free shared storage and download time for the model.
All prepared assets stay in this project, which must be visible from gpu7.

Requires Bash, curl, Python 3.12+, and sha256sum; no root, CUDA compiler, or Docker
is needed. The GPU node still needs a compatible NVIDIA driver. If you already
have a suitable `llama-server`, you can set `LLAMA_SERVER` and `LLM_MODEL_FILE`
instead; provide an API key file with `LLM_API_KEY_FILE` too.

## 2. Submit the server on the headnode

```bash
cd /mnt/gpu-storage/homes/athapar/code/llms
mkdir -p logs
job_id=$(sbatch --parsable slurm/serve.sbatch)
squeue -j "$job_id"
tail -f "logs/qwen-$job_id.log"
```

The job requests partition `main`, node `gpu7`, one L40S, eight CPU threads,
64 GB of host RAM, and eight hours. It starts the server on port 8000 with model
alias `qwen3.8-27b`, one request slot, a 32K context, and Jinja tool support.
Thinking is disabled initially for quick API tests. Slurm selects the GPU via its
environment; the script does not hard-code a physical GPU index.

If your Slurm account requires an explicit allocation, add `--account=YOUR_ACCOUNT`
to `sbatch`. The endpoint exists only while the job runs. Stop it with:

```bash
scancel "$job_id"
```

## 3. Connect from your local machine

Use your normal SSH login address for the headnode; the address below is the
hostname detected in this workspace. Connect to the cluster VPN first if that is
required for SSH. Keep the tunnel running in a separate terminal:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8000:wpia-gpu-07.dide.ic.ac.uk:8000 \
  athapar@wpia-lxhn.hpc.dide.ic.ac.uk
```

The forwarding destination uses gpu7's DNS hostname. `gpu7` is its Slurm name
and did not resolve as a DNS name on the headnode during inspection.
This tunnel needs headnode-to-compute TCP connectivity, but does not need SSH
access to the compute node. The server binds to cluster interfaces and requires
your API key; the laptop listener binds to localhost. The SSH connection encrypts
the laptop-to-headnode leg; the headnode-to-compute HTTP leg uses the cluster
network. For SSH encryption through to the compute node, see the alternative below.

In another **local** terminal, copy the server's key once and load it without
printing it:

```bash
mkdir -p "$HOME/.config/qwen-slurm"
chmod 700 "$HOME/.config/qwen-slurm"
scp athapar@wpia-lxhn.hpc.dide.ic.ac.uk:/mnt/gpu-storage/homes/athapar/code/llms/.secrets/llm-api-key \
  "$HOME/.config/qwen-slurm/api-key"
chmod 600 "$HOME/.config/qwen-slurm/api-key"
export QWEN_API_KEY="$(cat "$HOME/.config/qwen-slurm/api-key")"
export QWEN_BASE_URL=http://127.0.0.1:8000/v1
```

## 4. Test and use the API with curl

Health returns 503 while loading and 200 when ready; it is unauthenticated.
The model list should contain `qwen3.8-27b`:

```bash
curl --fail-with-body http://127.0.0.1:8000/health

curl --fail-with-body "$QWEN_BASE_URL/models" \
  -H "Authorization: Bearer $QWEN_API_KEY"
```

Chat Completions:

```bash
curl --fail-with-body "$QWEN_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $QWEN_API_KEY" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Write a Python function that adds two numbers."}],
    "max_tokens": 512
  }'
```

Streaming output (`-N` prevents curl from buffering server-sent events):

```bash
curl -N --fail-with-body "$QWEN_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $QWEN_API_KEY" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Explain Slurm in three sentences."}],
    "max_tokens": 512,
    "stream": true
  }'
```

Responses API, which current Codex requires:

```bash
curl -N --fail-with-body "$QWEN_BASE_URL/responses" \
  -H "Authorization: Bearer $QWEN_API_KEY" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "qwen3.8-27b",
    "input": "Reply with: cluster API is working",
    "max_output_tokens": 256,
    "store": false,
    "stream": true
  }'
```

`llama-server` implements both API formats. Its Responses endpoint translates
requests to Chat Completions internally, so this is a compatibility implementation.
See the [llama-server API reference](https://github.com/ggml-org/llama.cpp/blob/b11472/tools/server/README.md).

## 5. Use from Python

Copy `main.py` to your laptop, or run it in a local checkout of this project.
It uses Python's standard library and the same environment variables as curl:

```bash
python3 main.py --models
python3 main.py "Explain how a Slurm GPU job works."
python3 main.py --responses "Write a short Python hello-world program."
python3 main.py --api-key-file "$HOME/.config/qwen-slurm/api-key" "Say hello."
```

Any OpenAI-compatible SDK can instead use base URL
`http://127.0.0.1:8000/v1`, this API key, and model `qwen3.8-27b`.

## 6. Use OpenCode locally

Check `opencode --version`. Copy the matching config from the cluster to the
**local code project** you want OpenCode to edit. V1 uses `provider`/`npm`/`options`;
V2 uses `providers`/`package`/`settings`:

```bash
# Run in your local code project. Merge manually if opencode.json already exists.
# For OpenCode 1.x:
scp athapar@wpia-lxhn.hpc.dide.ic.ac.uk:/mnt/gpu-storage/homes/athapar/code/llms/config/opencode-v1.json \
  ./opencode.json

# For OpenCode 2.x, use config/opencode-v2.json instead of opencode-v1.json above.
opencode --model slurm-qwen/qwen3.8-27b
```

The configs use the Chat Completions compatible provider. Export `QWEN_API_KEY`
in the shell that launches OpenCode. See the official
[V1 custom provider guide](https://opencode.ai/docs/providers#custom-provider)
and [V2 provider guide](https://opencode.ai/v2/docs/providers).

## 7. Use Codex CLI locally

Merge `config/codex.toml` into your local `~/.codex/config.toml`; it adds a named
profile without changing your default provider. With the tunnel running and
`QWEN_API_KEY` exported, start Codex in your local code project:

```bash
codex --profile slurm_qwen
```

Current Codex supports `wire_api = "responses"`. `wire_api = "chat"` examples in
older guides are obsolete. See the official
[configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
and [custom provider configuration](https://learn.chatgpt.com/docs/config-file/config-advanced).

Codex needs working streamed Responses events and a full tool call/result loop;
a successful curl chat request alone does not establish agent compatibility.
After the Responses curl test, try an actual Codex task that reads a local file
and reports what it contains. Server/client versions can affect tool compatibility.
The provided Codex profile is a starting configuration; this project has not yet
been tested against a running GPU model. See OpenAI's
[gateway compatibility requirements](https://learn.chatgpt.com/docs/enterprise/gateway-compatibility).

## Optional adjustments

Set these variables on the **headnode before submitting a new job**:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_CTX_SIZE` | `32768` | Total server context; increase after checking VRAM |
| `LLM_PARALLEL` | `1` | Concurrent request slots; context is shared across slots |
| `LLM_REASONING` | `false` | Set `true` for thinking with medium effort |
| `LLM_REASONING_EFFORT` | `medium` | Template reasoning level; GLM defaults to `max` |
| `LLM_TEMPERATURE`, `LLM_TOP_P` | Depends on reasoning mode | Override sampling for another model |
| `LLM_TOP_K`, `LLM_MIN_P` | `20`, `0` | Additional sampling filters |
| `LLM_PRESENCE_PENALTY` | Depends on reasoning mode | Override the presence penalty |
| `LLM_PORT` | `8000` | Change if another job already uses that port on gpu7 |
| `LLM_HOST` | `0.0.0.0` | Use `127.0.0.1` with compute-node SSH below |
| `LLM_GPU_LAYERS` | `99` | Maximum GPU offload; lower if necessary |
| `LLM_MODEL_FILE` | Prepared GGUF | Use another locally downloaded quant |
| `LLM_API_KEY_FILE` | `.secrets/llm-api-key` | Read a different key file |
| `LLAMA_SERVER` | Prepared executable | Use your own CUDA build |

For example:

```bash
LLM_REASONING=true sbatch slurm/serve.sbatch
```

If you change context, update the limits in your local OpenCode/Codex config too.
With multiple slots, budget context per slot. Requesting all four GPUs does not
guarantee better latency for a model that fits on one; start with one and benchmark
before using a larger allocation. This job serves text; image requests also need
the model's `mmproj` file and appropriate server/client configuration.

If compute-node SSH is allowed while your job runs, you can bind the server to
loopback and tunnel all the way to the GPU node:

```bash
# Headnode: stop the old server first, then submit this mode.
LLM_HOST=127.0.0.1 sbatch slurm/serve.sbatch

# Laptop:
ssh -N -o ExitOnForwardFailure=yes \
  -J athapar@wpia-lxhn.hpc.dide.ic.ac.uk \
  -L 127.0.0.1:8000:127.0.0.1:8000 \
  athapar@wpia-gpu-07.dide.ic.ac.uk
```

## Qwen3.8-Flash-Next on all four L40S GPUs

For the larger model in [Unsloth's Next guide](https://unsloth.ai/docs/models/qwen3.8-next),
use `scripts/prepare-next.sh` and `slurm/serve-next.sbatch`. The default
`UD-Q4_K_XL` quant consists of four GGUF shards totaling about 111.3 GB.
Allow at least 120 GB of additional shared storage for this download.
All four shards must remain together; pass shard `00001` to the server and it
opens the other three automatically. The file layout was checked against the
[actual quant directory](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/UD-Q4_K_XL).

Your four roughly 46 GiB GPUs provide about 184 GiB of aggregate VRAM, but each
device has its own memory. The new job requests all four L40S GPUs on gpu7,
16 CPU threads, and 256 GB of host RAM. It explicitly sets `--split-mode layer`
and `--tensor-split 1,1,1,1` to distribute offloaded layers across the four GPUs.
It starts with one request slot and a 32K context.

Flash-Next has large per-layer embedding lookup tables. The job keeps llama.cpp's
`--lazy-mode auto`, which uses CPU-backed memory mappings for large eligible
tensors. Therefore some weights may use host RAM/file cache even when all model
layers are offloaded. The larger RAM allocation leaves room for this cache;
shared-storage speed and cache warmup can affect latency. The GPU allocation is
a starting configuration, not a guarantee of fourfold speed or fully GPU-resident
weights. See [llama.cpp's memory and GPU splitting options](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).

With the existing CUDA runtime already prepared, run on the headnode:

```bash
cd /mnt/gpu-storage/homes/athapar/code/llms
bash scripts/prepare-next.sh
job_id=$(sbatch --parsable slurm/serve-next.sbatch)
squeue -j "$job_id"
tail -f "logs/qwen-next-$job_id.log"
```

The download command selects only this quant and records the first shard in
`.tools/qwen-next/model.path`. Existing model files are reusable. The pinned
runtime includes the model's `qwen4exp` architecture; no MTP branch is needed for
this initial run. This configuration serves text and does not enable MTP or vision.

The port and API key remain the same as the 27B setup, but the API model alias
becomes `qwen3.8-flash-next`. An earlier server using port 8000 or one of these
GPUs must finish before the four-GPU server can use those resources. From the
headnode, after loading completes:

```bash
export QWEN_API_KEY="$(cat .secrets/llm-api-key)"
curl --fail-with-body http://wpia-gpu-07.dide.ic.ac.uk:8000/health
curl --fail-with-body http://wpia-gpu-07.dide.ic.ac.uk:8000/v1/chat/completions \
  -H "Authorization: Bearer $QWEN_API_KEY" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "qwen3.8-flash-next",
    "messages": [{"role": "user", "content": "Reply with: Flash Next is running."}],
    "max_tokens": 128
  }'
```

The existing laptop SSH tunnel works with this server too. For OpenCode, use
`config/opencode-next-v1.json` or `config/opencode-next-v2.json` as the local
project config and select `slurm-qwen/qwen3.8-flash-next`. The V1 config also sets
the auxiliary model to the new alias. For Codex, merge the updated
`config/codex.toml` and start `codex --profile slurm_qwen_next`. For the Python
client, use `--model qwen3.8-flash-next` or export `QWEN_MODEL` to that value.

Once the default quant works, another quant can be selected when downloading:

```bash
# Optional: about 158.3 GB of weights; profile memory before increasing context.
bash scripts/prepare-next.sh UD-Q5_K_XL
```

This changes the recorded model path for the next submission. Start with Q4
and inspect the job log and actual per-GPU usage before choosing larger quants
or increasing context/concurrency. While the job runs, a headnode check is:

```bash
srun --jobid="$job_id" --overlap --ntasks=1 \
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
```

## GLM-5.3-Flash with 128K context on RTX PRO 5000 GPUs

The GLM run uses [Unsloth's GLM-5.3-Flash GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF),
defaulting to `UD-Q4_K_XL`: six shards totaling about 199.7 GB (186 GiB).
Allow at least 210 GB of additional storage. The existing pinned `b11472` runtime
includes the model's `glm5-next` architecture, as checked in the installed library
and [release source](https://github.com/ggml-org/llama.cpp/blob/b11472/src/models/glm5-next.cpp).
It can be reused without compiling the older branch mentioned in the model guide.

`slurm/serve-glm.sbatch` requests **five 48 GB RTX PRO 5000 Blackwell GPUs on gpu8**,
16 CPU threads, 256 GB host RAM, and eight hours. It distributes layers equally,
with one **131,072-token** request slot. Five GPUs leave roughly 50 GiB beyond
the weight-file size for caches, compute buffers, and uneven layer placement.
Four GPUs would leave very little headroom with this Q4 quant. This is a memory
budget estimate; validate actual GPU consumption with a representative long prompt.

The model alias is `glm-5.3-flash`, and its separate endpoint uses **port 8001**.
It reuses `.secrets/llm-api-key`. Reasoning is enabled with `max` effort; sampling
defaults to temperature 1.0 and top-p 0.95 following the
[Unsloth GLM guide](https://unsloth.ai/docs/models/glm-5.3-flash).
This setup serves text and tool calls; vision and MTP are not enabled.

On the headnode, with the existing CUDA runtime prepared:

```bash
cd /mnt/gpu-storage/homes/athapar/code/llms
bash scripts/prepare-glm.sh
job_id=$(sbatch --parsable slurm/serve-glm.sbatch)
squeue -j "$job_id"
tail -f "logs/glm-flash-$job_id.log"
```

Preparation downloads only the selected quant, validates all six shards, and
records the entry point in `.tools/glm-flash/model.path`. Downloads resume through
the Hugging Face CLI. The node needs five GPUs, 16 CPUs and 256 GB RAM free for
the job to start. Shared NFS reads of this large model can make startup slow.
To select a different quant, pass its directory name to `prepare-glm.sh` and
adjust the GPU request according to the model and context memory requirements.

For shorter reasoning, submit with `LLM_REASONING_EFFORT=low` or `high`:

```bash
LLM_REASONING_EFFORT=high sbatch slurm/serve-glm.sbatch
```

Test from the headnode after the log says the model is loaded:

```bash
export GLM_API_KEY="$(cat .secrets/llm-api-key)"
curl --fail-with-body http://wpia-gpu-08.dide.ic.ac.uk:8001/health
curl --fail-with-body http://wpia-gpu-08.dide.ic.ac.uk:8001/v1/chat/completions \
  -H "Authorization: Bearer $GLM_API_KEY" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "glm-5.3-flash",
    "messages": [{"role": "user", "content": "Reply briefly: GLM is running."}],
    "reasoning_effort": "low",
    "max_tokens": 1024
  }'
```

On your laptop, keep this tunnel open:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8001:wpia-gpu-08.dide.ic.ac.uk:8001 \
  athapar@wpia-lxhn.hpc.dide.ic.ac.uk
```

Use `http://127.0.0.1:8001/v1` locally, load the same API key as before, and set
`GLM_API_KEY` to that key. For example, if you already copied it for Qwen:

```bash
export GLM_API_KEY="$(cat "$HOME/.config/qwen-slurm/api-key")"
```

Copy `config/opencode-glm-v2.json` (or `opencode-glm-v1.json` for OpenCode 1.x)
to your local project as `opencode.json`, merging if that file already exists.
Select `slurm-glm/glm-5.3-flash`. Both profiles use the 128K context limit with
a 16K maximum output, leaving input room for prompts and tool history. Context
includes both input and output tokens. Validate an actual tool call/result loop
after the curl test before relying on agent workflows.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Job pending | `squeue -j JOB_ID`; gpu7 must have the requested resources free |
| Local connection refused | Keep the SSH tunnel open; check the job log and server readiness |
| HTTP 401 | Load the same API key that the cluster server uses |
| HTTP 503 | Wait for model loading to complete; inspect the job log |
| Local port in use | Forward local port 8001 to compute port 8000 and update client base URLs |
| Server port in use | Set `LLM_PORT` before submission and update the SSH destination port |
| Missing CUDA libraries or driver error | Inspect the job's NVIDIA driver output; use a compatible CUDA runtime/build |
| Out of VRAM | Reduce context/concurrency or choose a smaller quant |
| Codex tool/API errors | Verify Responses streaming and tool support; test OpenCode's Chat Completions route |

Files were checked for shell/Python syntax, config parsing, and Slurm submission
validity using `sbatch --test-only`. Model downloads, live inference, SSH forwarding
from your laptop, and the agents' tool loops still require running the setup.

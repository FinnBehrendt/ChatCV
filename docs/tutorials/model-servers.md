# Optional tutorial: prepare a model server

This is background guidance, not an integrated deployment feature. ChatCV only
needs a reachable API URL, model name and optional key. You install, start,
secure and stop the server independently. Do not install model-server packages
into ChatCV's Pixi environment just to call their HTTP APIs.

## 1. Ollama

Install [Ollama](https://docs.ollama.com/quickstart) for your OS. Start its app or,
if no server is already running, use a separate terminal:

```bash
ollama serve
```

In another terminal, prepare a small example model:

```bash
ollama pull qwen2.5:0.5b
```

For ChatCV running locally with Pixi, configure `.env`:

```dotenv
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:0.5b
LLM_API_KEY=
```

For ChatCV inside Docker Desktop, use `host.docker.internal` instead of localhost.
Make sure host networking permits that connection; do not automatically expose
an unauthenticated server on all interfaces to fix reachability.

Ollama also provides an [official Docker image](https://docs.ollama.com/docker).
Its guide distinguishes CPU-only, NVIDIA and AMD setups. On an Apple Silicon
Mac, native Ollama can use Metal; the ordinary Linux container is not a way to
pass through the Apple GPU. On Linux, GPU use requires the appropriate host
driver/runtime and explicit device access, not merely installing Docker.

## 2. vLLM

Install a hardware-appropriate vLLM environment following its
[GPU](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) or
[CPU](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/) instructions.
On a compatible installation, an example server is:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --served-model-name chatcv-model --port 8001 --host 127.0.0.1
```

For local Pixi ChatCV:

```dotenv
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://localhost:8001/v1
LLM_MODEL=chatcv-model
LLM_API_KEY=
```

The loopback example is deliberately local. When the app is containerized or on
another machine, choose a protected interface/network reachable from that app.
Use the reachable hostname in `LLM_BASE_URL`; localhost inside the app container
is the app container itself. Shared-container-network DNS names are another
option when you manage both services yourself.

The [official vLLM Docker guide](https://docs.vllm.ai/en/latest/deployment/docker/)
covers NVIDIA/CUDA images. CPU inference has separate images and requirements;
removing GPU flags from a CUDA image is not a CPU setup. Pin a tested image
version/digest for reproducibility and check model, driver and VRAM compatibility.

## 3. Mac Metal and Docker-managed alternatives

For native vLLM on Apple Silicon, consult
[vLLM-Metal](https://github.com/vllm-project/vllm-metal). It uses its own compatible
environment and supported model formats; CUDA/Hugging Face and MLX identifiers
are not automatically interchangeable.

[Docker Model Runner](https://docs.docker.com/ai/model-runner/) offers another
API-compatible workflow. On macOS the inference engine runs as a native sandboxed
process, not inside an ordinary Metal-enabled Linux container. Its available
backends depend on the installed version; check the current backend documentation
and [vLLM-Metal announcement](https://www.docker.com/blog/docker-model-runner-vllm-metal-macos/).
Configure ChatCV with the API URL and model that your runner actually advertises.

## 4. Connect and verify

1. Start the chosen server and confirm its model is available through its API.
2. Set `LLM_BASE_URL`, `LLM_MODEL` and that endpoint's own `LLM_API_KEY` in `.env`.
3. Keep embedding settings independent; local chat does not mean fully offline.
4. Start/recreate only ChatCV with `docker compose up --build -d` from the
   repository root (or use `pixi run dev` without Docker).
5. Test a synthetic question before sending private document content.

ChatCV's Compose file never manages model downloads, GPU selection or server cleanup.
`docker compose stop` stops only ChatCV. Stop the server using its own CLI/service
manager. Keep model APIs private; remote deployments need authentication, TLS
and network access controls. Model-server keys/tokens are not ChatCV role keys
unless that API explicitly uses the same key.

Small models here are setup examples, not recommendations for answer quality.
These commands illustrate documented interfaces; not every hardware/backend
combination has been live-tested by this project. Startup readiness, answer
quality and performance are separate checks. See [endpoint details](../model-endpoints.md)
for the exact client contract.

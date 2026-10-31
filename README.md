# ChatCV - Your Resume, Now Conversational

Ever wished recruiters could just *ask* your resume questions instead of skimming through bullet points? That's exactly what ChatCV does.

[![CI](https://github.com/FinnBehrendt/ChatCV/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/FinnBehrendt/ChatCV/actions/workflows/ci.yml)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](./Dockerfile)

## Demo

<p align="center">
  <img src="ChatCV_Demo.gif" alt="ChatCV Demo" width="500"/>
</p>

## What it does
ChatCV lets you stand out by turning your resume into a conversation. Instead of hoping recruiters notice the right details, you give them a tool to dig deeper when something catches their interest.

---

## Who's This For?

**Job Seekers & Professionals**
- Stand out from the pile of identical PDFs
- Let interested recruiters explore your background in depth
- Show off your technical skills by adpating (and enhancing) this solution for your CV
- Provide a memorable, interactive experience
- Be available to answer questions 24/7

**Recruiters & Hiring Managers**
- When a candidate catches your interest, dig deeper instantly
- Get specific answers instead of guessing from bullet points
- AI-powered job fit analysis for promising candidates
- **Coming soon**: A tool to compare multiple candidate's CVs for a certain job description

---

## What You Get

**Chat with Resume**: Natural conversation about any aspect of a candidate's background
**Smart Job Matching**: Upload a job description, get detailed fit analysis
**Optional Security**: Deploy publicly or lock it down with invite codes
**Built-in Safety**: Keeps conversations professional and on-topic
**Docker Ready**: Deploy anywhere in minutes


---

## Local development with Pixi

Use [Pixi](https://pixi.sh/) 0.64.0 or newer to manage the Python 3.12 environment.
The manifest and lockfile cover Apple silicon macOS, Linux x86-64 and Linux ARM64. Pixi
installs Python and dependencies into the ignored `.pixi/` directory; your
system Python is not used.

From the repository root:

```bash
pixi install --locked
pixi run test
```

The default environment includes development tools. Tests disable external
LangSmith tracing and do not require model-server access. To run the app, copy
`.env.example` to `.env`, configure your model credentials and documents, then run:

```bash
pixi run dev
```

| Command | Purpose |
| --- | --- |
| `pixi run test` | Run the tests |
| `pixi run test-cov` | Run tests with coverage |
| `pixi run lint` | Check with Ruff |
| `pixi run format-check` | Check formatting without editing files |
| `pixi run format` | Apply Black formatting |
| `pixi run typecheck` | Run Mypy |
| `pixi run security` | Run Bandit |
| `pixi run check` | Run tests, lint, formatting, type and security checks |
| `pixi run dev` | Start the app with automatic reload |
| `pixi run start` | Start the app without reload |

For an individual test, use
`pixi run python -m pytest tests/test_guardrails.py -v`.
`pixi shell` opens an interactive shell inside the environment.

The `runtime` environment omits development tools:

```bash
pixi install --locked -e runtime
pixi run -e runtime start
```

Update dependencies through Pixi and review `pixi.toml` and `pixi.lock` together.
Keep both files in version control; never commit `.pixi/`. The LangChain
dependencies remain on the 0.3-compatible API family used by this application.
Starlette stays below 1.0 while the app uses the older `TemplateResponse`
calling convention. These compatibility bounds avoid an API migration as part
of environment setup.

## Continuous Integration

Pull requests targeting `main` and pushes to `main` or `develop` run Ruff,
Black, Mypy, Bandit and pytest with coverage using the locked Pixi environment.

Pushes to `main` additionally build the Docker image and verify that its
`/health` endpoint responds successfully. The Docker job is skipped in pull
requests. Images are not automatically published or deployed.

Run the same code-quality checks locally before opening a pull request:

```bash
pixi run --locked check
```

## Getting Started with Docker

**What you need:**
- Docker & Docker Compose
- A model endpoint: OpenAI, Ollama, or an OpenAI-compatible server/API

### 1. Get the Code
```bash
git clone https://github.com/FinnBehrendt/ChatCV.git
cd ChatCV
cp .env.example .env
```

### 2. Configure Your Model Endpoints

The copied `.env.example` uses `openai_compatible` for both chat and embeddings,
initially pointing to OpenAI's hosted API. Add the endpoint-specific keys:

```bash
# For the default hosted endpoints, both can contain the same OpenAI API key.
LLM_API_KEY="your_openai_api_key_here"
EMBEDDING_API_KEY="your_openai_api_key_here"

# Optional extras (leave empty if you don't need them)
LANGSMITH_API_KEY="your_langsmith_key"     # For advanced analytics
LANGSMITH_TRACING=false

GUARDRAILS_TOKEN="your_guardrails_token"   # Extra safety (slower build)
```

For a local model server, change the corresponding base URL and model instead.
Clear that role's API key if the local server does not require authentication.
See [Ollama via the shared API](#ollama-via-the-shared-api) for the Docker setup.
`OPENAI_API_KEY` alone does not authenticate the compatible provider.
Keep comments on a separate line for empty values: use `LLM_API_KEY=`, not
`LLM_API_KEY= # comment`. Compose can interpret that comment as the actual key.

**Pro tip**: The app works fine without the optional stuff. Only add them if you actually need the features.

### 3. Make It Yours
- Swap out `data/CV_Demo.pdf` with your actual resume
- Edit `data/about_me.md` with extra info not in your CV (or delete it)
- Edit `config/base.yml` if you want to change file paths or naming
- replace `/static/default-avatar.png` with your profile picture

Set `data.cv_path` in `config/base.yml` to your PDF's filesystem path, for example
`data/my_cv.pdf`. The configured CV is publicly downloadable at `/resume` through
the **Download CV** link, including when invite-code authentication is enabled.
When using Docker, rebuild the image after replacing the PDF.

### 4. Start the App
```bash
docker compose up --build
```
Or build and run the image directly, without Compose:

```bash
docker build -t chatcv:local .
docker run --rm --env-file .env -p 8000:8000 chatcv:local
```

The multi-stage Dockerfile uses Pixi 0.64.0 and
`pixi install --locked -e runtime`. Only the installed runtime environment and
application files are copied into the final stage: no development tools,
installer caches, local planning notes or `.env` files. The app runs as a
non-root user and starts without installing or resolving dependencies.

Both `linux/amd64` and `linux/arm64` are covered by the lockfile. Docker uses the
host architecture by default. For an x86-64 cloud server, build explicitly with
`docker build --platform linux/amd64 -t chatcv:local .`.

Optional Guardrails Hub validators can still be installed using a build secret:

```bash
# GUARDRAILS_TOKEN must already be exported in your shell.
docker build --secret id=guardrails_token,env=GUARDRAILS_TOKEN -t chatcv:local .
```

These extra Hub packages and their model downloads are **not** locked by Pixi.
The token is mounted only during installation; the temporary configuration is
not copied into the runtime image. Downloaded model caches are not copied either,
so validators may need to download models again at runtime. Without this secret,
the app uses its existing basic validation fallback.

For development without Docker, use `pixi run dev` as described above.

### 5. Try It Out
Head to `http://localhost:8000` and start chatting with your resume!

## Model providers: local servers and APIs

Use **`openai_compatible` as the preferred integration** for Ollama, vLLM and
compatible hosted APIs, including OpenAI itself. The server runs independently;
ChatCV selects it through its URL, model name and endpoint-specific key.

Choose chat and embeddings **independently**. Model names are sent unchanged to
the selected endpoint. Existing `openai` and native `ollama` configurations remain
supported; they are not required for the shared text-chat/embedding API contract.

| Provider | Chat | Embeddings | Endpoint |
| --- | --- | --- | --- |
| `openai_compatible` | Chat Completions API | Text-input embeddings API | Separate `LLM_BASE_URL` / `EMBEDDING_BASE_URL` |
| `openai` (existing adapter) | OpenAI chat client | OpenAI embeddings client | Official OpenAI API |
| `ollama` (existing adapter) | Native Ollama chat | Native Ollama embeddings | `OLLAMA_ENDPOINT` |

The compatible provider is the common integration point for local servers such
as [vLLM](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/) and
hosted APIs implementing these endpoints. It does not launch or provision those
servers. There is no backend-specific code in the retrieval pipelines.

### OpenAI-compatible chat

For example, with a separately running local server:

```dotenv
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://localhost:8001/v1
LLM_MODEL=your-served-model-name
LLM_API_KEY=
LLM_TEMPERATURE=0.1
LLM_TIMEOUT=120
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=your-openai-key
```

This endpoint is for a **local Pixi process**. From a Compose container use the
server's service name, or `host.docker.internal` for a host server on Docker
Desktop. Inside a container, `localhost` refers to that container, not your host.
Use the API root including its prefix (typically `/v1`), not `/chat/completions`.
For a hosted API, replace the URL and set `LLM_API_KEY` to that service's key.
Use HTTPS for remote services. Keep `.env` out of version control.
When switching endpoints, update or clear the corresponding API key too: an
explicitly configured key is sent to that role's endpoint.

An unauthenticated local endpoint receives a harmless `not-required` bearer token
because the SDK requires a key. Compatible endpoints **never inherit
`OPENAI_API_KEY`**. Chat and embeddings each have their own optional key; for the
official `openai` provider only, these override the shared `OPENAI_API_KEY`.

For compatible embeddings, configure the other role separately:

```dotenv
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_BASE_URL=http://localhost:8002/v1
EMBEDDING_MODEL=your-served-embedding-model
EMBEDDING_API_KEY=
EMBEDDING_TIMEOUT=60
```

This sends text to `/embeddings`, not OpenAI-specific token IDs. Keep document
chunks within the selected model's input limit; no provider-specific tokenizer
is assumed. Switching embedding models requires a fresh vector index (see below).

### Compatibility boundaries

The shared contract is text chat through `/chat/completions` and embeddings
through `/embeddings`. A server must implement the endpoint for the selected
role. Unknown providers and missing compatible base URLs fail explicitly; there
is no automatic switch to another service. API errors propagate to the existing
application error handling instead of silently changing providers.

Set `LLM_TEMPERATURE=null` to omit temperature when a model rejects that option.
Provider-specific features such as Responses API, extended thinking, custom
headers, special tool schemas or multimodal inputs are not part of this shared
contract. Azure-specific authentication and API-version handling are not a native
adapter here; a deployment must expose the compatible contract to use this path.

Anthropic offers an [OpenAI SDK compatibility layer](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk).
For basic text chat it can use `LLM_PROVIDER=openai_compatible`,
`LLM_BASE_URL=https://api.anthropic.com/v1`, a supported Claude model name,
`LLM_API_KEY` containing a Claude key and `LLM_TEMPERATURE=null`. Use a
workspace-scoped key: keys requiring the additional `anthropic-workspace-id`
header need an adapter not provided here. This is **not a native Anthropic
integration**; compatibility-layer restrictions still apply, including differences
in system messages and unsupported features. Embeddings remain a separate service.

Automated tests exercise real SDK request/response handling with an in-memory
HTTP transport, without cloud credentials or model downloads. The compatible
protocol is not a claim that every model or vendor-specific feature has been
tested against each live server.

## Ollama via the shared API

The base `docker-compose.yml` starts only ChatCV and connects to the endpoint
configured in `.env`. The optional `docker-compose.ollama.yml` adds the entire
local model stack: Ollama, model preparation, model storage and the app's local
connection settings. No Compose profiles are needed.

ChatCV and Ollama run independently. To start the app, server and model preparation
with one command, use the local Compose override:

```bash
docker compose -f docker-compose.yml -f docker-compose.ollama.yml up --build -d
```

This selects `openai_compatible`, sets the internal Ollama URL, clears the cloud
chat key and uses `OLLAMA_CHAT_MODEL` (default `llama3.2:1b`). You do not need to
replace the cloud chat settings in `.env` to use this local override.

Startup order is **healthy Ollama server -> model preparation -> ChatCV**.
Preparation reuses an installed model; otherwise it downloads it. A failed model
download prevents dependent startup. The first download requires internet and
disk space. Set `OLLAMA_CHAT_MODEL` to choose another model; changing `LLM_MODEL`
alone does not change the override's chat model. Embedding settings remain as
configured in `.env` and are not automatically changed to a local model.

For the default OpenAI embeddings, set a valid `EMBEDDING_API_KEY` in `.env`.
Document chunks and search queries still go to that service. A local chat model
alone does not make the app offline.

Check setup progress (with the same Compose files):

```bash
docker compose -f docker-compose.yml -f docker-compose.ollama.yml logs -f ollama ollama-init
```

Use the same Compose files when updating or stopping this local stack. To return
to your cloud chat configuration without deleting models or the vector index:

```bash
docker compose -f docker-compose.yml -f docker-compose.ollama.yml stop chatcv ollama-init ollama
docker compose up -d --force-recreate chatcv
```

For a separately managed model server, use the base file only and configure its
reachable URL in `.env` as shown in the provider examples above.

The `ollama_data` volume keeps downloaded models between restarts. Its name and
mount path are unchanged, so an existing Compose model volume is reused.
The API port is published only on the host's loopback interface. Do not expose
the model server directly to the public internet.

Use the API base URL appropriate to where **ChatCV** runs. Set `LLM_BASE_URL`
for chat and, if using Ollama embeddings too, `EMBEDDING_BASE_URL`:

| ChatCV | Ollama | Compatible API base URL |
| --- | --- | --- |
| Compose container | Compose service | `http://ollama:11434/v1` |
| Local Pixi process | Local server or published Docker service | `http://localhost:11434/v1` |
| Docker Desktop container | Native host server | `http://host.docker.internal:11434/v1` |

The local Compose stack is a portable CPU baseline, not a GPU configuration.
For Apple GPU acceleration run Ollama natively; Docker Desktop on macOS does not
provide that GPU access. Linux GPU deployments need a suitable GPU runtime.
See the [official Ollama Docker guide](https://docs.ollama.com/docker) and
[GPU FAQ](https://docs.ollama.com/faq#how-do-i-use-ollama-with-gpu-acceleration-in-docker).
The server image can be pinned with `OLLAMA_IMAGE`; use a fixed version or digest
for reproducible deployments and benchmarks.

### Optional local embeddings

To use Ollama for embeddings as well, set:

```dotenv
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_BASE_URL=http://ollama:11434/v1
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_API_KEY=
```

With the local Compose override, preparation also downloads a missing embedding
model when `EMBEDDING_PROVIDER=openai_compatible` and `EMBEDDING_BASE_URL` is
`http://ollama:11434/v1` (an optional trailing slash is accepted). For a separately
managed server, use its CLI to download the embedding model.

When changing embedding models, configure a **new** `data.vector_db_path` in
`config/base.yml` so ChatCV creates a fresh index. Existing embeddings cannot be
reused with another embedding model, even if the vector dimensions match. Keep
the old index until you have verified the new one; no automatic deletion occurs.
Chat and embedding providers can be selected independently.

## Model configuration

### Existing adapter configurations

Existing installations may continue to use `LLM_PROVIDER=ollama` or
`EMBEDDING_PROVIDER=ollama` with `OLLAMA_ENDPOINT=http://ollama:11434` (no `/v1`)
and `OLLAMA_TIMEOUT`. These settings apply only to the native adapter.

Likewise, `openai` keeps the official endpoint and can use the shared
`OPENAI_API_KEY`. Role-specific keys override that shared key for this adapter.
The Python/YAML defaults retain this older configuration for compatibility;
the new `.env.example` explicitly selects the preferred compatible path.

### Configuration precedence

For model settings: process environment overrides `.env`, which overrides the
merged environment-specific and base YAML settings. Blank model variables leave
the YAML/default value unchanged. Supported variables are `LLM_PROVIDER`,
`LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_TIMEOUT`, `LLM_BASE_URL`, `OLLAMA_ENDPOINT`,
`OLLAMA_TIMEOUT`, `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_BASE_URL`
and `EMBEDDING_TIMEOUT`. Configure keys (`OPENAI_API_KEY`, `LLM_API_KEY`,
`EMBEDDING_API_KEY`) through process environment or `.env`, not public YAML files.

`LLM_TIMEOUT` configures OpenAI/compatible chat requests; `EMBEDDING_TIMEOUT`
configures OpenAI/compatible embeddings. `OLLAMA_TIMEOUT` configures native Ollama
chat and embedding requests. `OLLAMA_MODELS` is an Ollama server storage-directory
setting, **not** the name of the chat model. Use `LLM_MODEL` for model selection.

### Troubleshooting

Inspect the application without changing its files:

```bash
docker compose ps -a
docker compose logs -f --tail=100 chatcv
docker compose exec chatcv sh
```

Inside that shell, the application is in `/app`. Leave with `exit`. Changes inside
the container are not changes to your checkout and may be lost on recreation.
Do not share full `env`, `printenv`, `docker inspect` or resolved Compose output:
they can contain API keys. Do not install debugging packages into the runtime;
prefer its existing Python interpreter or debug locally with Pixi.

After editing environment variables, recreate the app with the same Compose files
you used to start it. `docker compose restart` does not reload `.env`.
An HTTP 401 from a model API indicates an authentication/configuration problem;
check the correct role key and ensure it is not an inline comment. Reaching
`/health` only verifies the web process, not a successful model request.

For the local Ollama stack, use both Compose files with the logs command above
and check the endpoint table. If a model is missing, inspect the `ollama-init`
logs; preparation must finish successfully before the app starts.
Increasing the timeout may help with cold model loads, but does not fix an
incorrect endpoint or a missing model.

---

## Deploy to the Cloud

The Docker container runs anywhere. Deploy it to your favorite cloud platform - AWS, Azure, GCP, whatever works for you.

**Coming soon**: Step-by-step Azure deployment guide (because that's what I use).

---

## Access Control


**Option 1: Keep it public**
```bash
INVITE_CODES={}
```
Perfect for personal websites or internal demos.

**Option 2: Lock it down**
```bash
INVITE_CODES={
  "RECRUITER1": {"company": "TechCorp", "recruiter": "Jane Smith", "active": true},
  "DEMO2024": {"company": "Demo Access", "recruiter": "Public Demo", "active": true}
}
```
Give specific people access with custom invite codes.

---

## How It Works

**The Stack:**
- FastAPI backend for the API
- Retrieval Augmented Generation (RAG):
  - ChromaDB as Database
  - OpenAI-compatible API for local or hosted chat and embedding models
- LangChain connecting the dots
- Guardrails to keep things professional and save

**The Flow:**
1. Your resume gets chopped up and embedded into a vector database
2. When someone asks a question, we find the most relevant parts
3. The LLM writes a natural response using that context (and chat history)
4. Safety filters make sure everything stays professional

Simple, but it works really well.

---

## Want to Contribute?

Got ideas? Found bugs? Want to add features?

Just open an issue or send a PR. I'm always looking to make this better.

### Local commit checks

After installing the Pixi environment, activate the Git hook once per clone so the CI doesn't yell at you:

```bash
pixi run --locked pre-commit install
```
To run all hooks manually:

```bash
pixi run --locked pre-commit run --all-files
```

Pixi must be available on the `PATH` of the terminal or Git client creating the
commit.

---

## Thanks To

- [LangChain](https://langchain.com/) for LLM magic
- [ChromaDB](https://www.trychroma.com/) for vector search
- [Guardrails AI](https://www.guardrailsai.com/) for keeping it safe

---

**Like this project? Give it a ⭐ - it helps more people find it!**

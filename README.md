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
See [Model endpoints](#model-endpoints) for short Ollama and vLLM examples.
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
docker compose up --build -d
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

## Model endpoints

The default `.env.example` uses OpenAI's hosted API. For local servers, select
one case below. `HOST` means `host.docker.internal` for ChatCV in Docker Desktop,
or `localhost` for local Pixi. Replace it before starting.

| Case | `LLM_PROVIDER` | URL setting | Model / timeout |
| --- | --- | --- | --- |
| Ollama compatible | `openai_compatible` | `LLM_BASE_URL=http://HOST:11434/v1` | `LLM_MODEL=qwen2.5:0.5b`, `LLM_TIMEOUT=120` |
| Ollama native | `ollama` | `OLLAMA_ENDPOINT=http://HOST:11434` | `LLM_MODEL=qwen2.5:0.5b`, `OLLAMA_TIMEOUT=120` |
| vLLM compatible | `openai_compatible` | `LLM_BASE_URL=http://HOST:8001/v1` | `LLM_MODEL=chatcv-model`, `LLM_TIMEOUT=120` |

Native Ollama **ignores `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_TIMEOUT`**.
Compatible APIs use `/v1` (API version), not `/api/chat`; the client adds the
request path. Clear `LLM_API_KEY` for keyless local servers; otherwise set that
endpoint's own key. Compatible clients never inherit `OPENAI_API_KEY`.

For **Ollama**, install it and run `ollama serve` if it is not already running.
In another terminal, download a model with `ollama pull qwen2.5:0.5b`.

For **vLLM**, install it in a suitable separate environment and start a server:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --served-model-name chatcv-model --port 8001 --host 127.0.0.1
```

These small models are setup examples, not quality recommendations. ChatCV starts
neither server. Compose has no `ollama` service: `http://ollama:11434` will not
resolve, and Ollama logs are in its own terminal, not `docker compose logs ollama`.
For remote servers use a reachable, protected hostname/interface. Never expose
an unauthenticated model API publicly.

Embeddings stay independent (`EMBEDDING_PROVIDER`, `EMBEDDING_BASE_URL`,
`EMBEDDING_MODEL`, `EMBEDDING_API_KEY`). Local chat alone does not make the app
offline. Apply `.env` changes with `docker compose up -d`, not `restart`.

CPU/GPU choices, Docker and native Mac options belong to the optional
[model-server tutorial](docs/tutorials/model-servers.md). The client contract is
text Chat Completions and text-input embeddings; see
[endpoint details](docs/model-endpoints.md) for compatibility limits.

## Starting ChatCV

Start Docker, then run Compose from the repository root. It automatically reads
`docker-compose.yml` and `.env`; Pixi is not required on the host:

```bash
docker compose up --build -d
docker compose ps -a
docker compose logs --tail 100 --follow chatcv
docker compose stop
```

Compose manages only ChatCV, not model servers or their hardware configuration.
The project name remains `chatcv` for existing deployments. `stop` does not remove
containers or volumes. Use `docker compose config --quiet` to validate the
configuration without printing secrets.

## Model configuration

### Existing adapter configurations

`LLM_PROVIDER=ollama` or `EMBEDDING_PROVIDER=ollama` uses `OLLAMA_ENDPOINT`
(no `/v1`) and `OLLAMA_TIMEOUT`. Neither uses the compatible URL settings.

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
docker compose logs --tail 100 chatcv
docker compose exec chatcv sh
```

Inside that shell, the application is in `/app`. Leave with `exit`. Changes inside
the container are not changes to your checkout and may be lost on recreation.
Do not share full `env`, `printenv`, `docker inspect` or resolved Compose output:
they can contain API keys. Do not install debugging packages into the runtime;
prefer its existing Python interpreter or debug locally with Pixi.

After editing environment variables, run `docker compose up -d` to recreate the
app with its updated configuration. A container restart alone does not reload `.env`.
An HTTP 401 from a model API indicates an authentication/configuration problem;
check the correct role key and ensure it is not an inline comment. Reaching
`/health` only verifies the web process, not a successful model request.

`docker compose logs` shows app logs only. Inspect model-server logs separately when needed.
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

The test suite includes Chromium UI tests against a temporary local app with
synthetic documents and model responses (no API keys or model server needed).
Install its headless browser once after installing the Pixi environment:

```bash
pixi run --locked playwright install --only-shell chromium
```

On Linux, add `--with-deps` to install required browser system libraries. CI
performs this setup automatically. Playwright is a development dependency only;
the application Docker image does not include it.

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

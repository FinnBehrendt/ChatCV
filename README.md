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
- OpenAI API key (local Ollama support coming soon)

### 1. Get the Code
```bash
git clone https://github.com/FinnBehrendt/ChatCV.git
cd ChatCV
cp .env.example .env
```

### 2. Add Your Secrets
Edit the `.env` file:
```bash
# Required
OPENAI_API_KEY="your_openai_api_key_here"

# Optional extras (leave empty if you don't need them)
LANGSMITH_API_KEY="your_langsmith_key"     # For advanced analytics
LANGSMITH_TRACING=false

GUARDRAILS_TOKEN="your_guardrails_token"   # Extra safety (slower build)
```

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
  - OpenAI API (or Ollama locally) as LLM
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

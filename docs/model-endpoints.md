# Model endpoint contract

ChatCV is an API client, not a model-server manager. Configure chat and embeddings
independently; their endpoints can be hosted APIs or separately managed servers.

| Provider | Chat | Embeddings | Configuration |
| --- | --- | --- | --- |
| `openai_compatible` (recommended) | Chat Completions | Text-input embeddings | Separate role URLs, models, keys and timeouts |
| `openai` | Official OpenAI client | Official OpenAI embeddings | Official endpoint; role key or shared `OPENAI_API_KEY` |
| `ollama` (existing adapter) | Native Ollama API | Native Ollama embeddings | `OLLAMA_ENDPOINT` without `/v1`, `OLLAMA_TIMEOUT` |

`.env.example` uses the compatible client with OpenAI's hosted endpoints. Models
are sent unchanged to the selected server; use its advertised model name or alias.
The API root normally ends in `/v1`, not `/chat/completions` or `/embeddings`.

## Example configurations: OpenAI, Ollama and vLLM

Copy `.env.example` to `.env`, then replace its model settings with one of the
examples below. All three use `openai_compatible`: the server changes, not the
application's client interface. The provider names `openai` and `ollama` in the
table above are separate, existing adapters, not required for these examples.

The local URLs below assume ChatCV runs with `pixi run dev` on the same machine
as the model server. For ChatCV in Docker Desktop, replace `localhost` with
`host.docker.internal`. On other deployments use a protected hostname/interface
reachable from the app container. A loopback-only server may need its networking
adjusted; do not publish an unauthenticated API to the internet.

### 1. OpenAI: hosted chat and embeddings

Set both role keys, even if they contain the same OpenAI API key. With the
compatible adapter, setting only `OPENAI_API_KEY` is not sufficient.

```dotenv
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=your-openai-api-key
LLM_TEMPERATURE=0.1
LLM_TIMEOUT=60

EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=your-openai-api-key
EMBEDDING_TIMEOUT=60
```

Both chat context and text submitted for embedding leave your machine. API usage
may incur charges; do not commit `.env`.

### 2. Ollama: local chat, hosted embeddings

Start Ollama separately and prepare the chat model with
`ollama pull qwen2.5:0.5b`. Its compatible API uses port 11434 and the `/v1` root.

```dotenv
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:0.5b
LLM_API_KEY=
LLM_TEMPERATURE=0.1
LLM_TIMEOUT=120

EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=your-openai-api-key
EMBEDDING_TIMEOUT=60
```

This assumes a local Ollama server without API authentication; clear any previous
chat key. Chat runs locally, but document chunks and search queries still go to
OpenAI for embeddings. For fully local operation, configure a separate local
embedding model/endpoint as described under [Independent embeddings](#independent-embeddings).
The Qwen chat model is not an embedding model.

### 3. vLLM: separately hosted chat, OpenAI embeddings

In a hardware-compatible, separate vLLM environment, start:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --served-model-name chatcv-model --port 8001 --host 127.0.0.1
```

Use the advertised alias `chatcv-model`, not the Hugging Face path, in ChatCV:

```dotenv
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://localhost:8001/v1
LLM_MODEL=chatcv-model
LLM_API_KEY=
LLM_TEMPERATURE=0.1
LLM_TIMEOUT=120

EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=your-openai-api-key
EMBEDDING_TIMEOUT=60
```

The example server is local and keyless. For a remote or authenticated deployment,
replace the URL and set `LLM_API_KEY` to that server's own key. The chat server
does not automatically provide embeddings; these settings deliberately keep
them on OpenAI. Ollama and vLLM use the same small Qwen model family here for
setup checks, not as an answer-quality recommendation.

ChatCV does not start either model server. See the optional
[server tutorial](tutorials/model-servers.md) for installation and hardware choices.
After editing `.env`, run `docker compose up -d` to recreate the app container,
or restart the local Pixi app process.

## Keys and network boundaries

- `LLM_API_KEY` belongs to the chat endpoint; `EMBEDDING_API_KEY` belongs to the
  embedding endpoint. Compatible endpoints never inherit `OPENAI_API_KEY`.
- Clear a role key when switching from a cloud API to an unauthenticated local
  server. Explicit role keys are sent to that role's configured endpoint.
- A keyless compatible endpoint receives the SDK placeholder bearer token
  `not-required`; this is not authentication or a security mechanism.
- Use HTTPS and access controls for remote APIs. Never publish resolved Compose
  output or complete environments, which can contain credentials.
- Keep comments on separate lines after empty assignments: `LLM_API_KEY=`.
  Compose may interpret an inline comment after an empty assignment as its value.

## Independent embeddings

For a separately managed embedding server:

```dotenv
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_BASE_URL=http://YOUR_EMBEDDING_SERVER:8002/v1
EMBEDDING_MODEL=your-served-embedding-model
EMBEDDING_API_KEY=
EMBEDDING_TIMEOUT=60
```

The compatible client sends text, not OpenAI-specific token IDs, and requests
float vectors. Keep chunks within the selected model's input limit. Configuring
local chat does not change where document chunks and search queries are embedded.

When changing embedding models, use a fresh `data.vector_db_path` in YAML and
rebuild the app image. Existing embeddings cannot be reused with a different
model, even if dimensions match. No automatic index invalidation/deletion occurs.

## Configuration and compatibility

Named process model variables override `.env`, which overrides merged YAML.
Blank model fields retain YAML defaults; optional empty role keys stay empty.
`LLM_TIMEOUT` and `EMBEDDING_TIMEOUT` configure their compatible/OpenAI clients.
Set `LLM_TEMPERATURE=null` to omit temperature for a model that rejects it.

The shared contract covers text chat through `/chat/completions` and text-input
embeddings through `/embeddings`. Responses API, multimodal inputs, extended
thinking, provider-specific tools/headers and native Azure authentication/version
handling are not promised. A compatible endpoint must implement the selected
role; a chat server does not necessarily provide embeddings. For full support,
specialized adapters like the native ollama API might be used.

Unknown providers or missing/invalid compatible URLs fail explicitly. API errors
propagate to existing app error handling; there is no automatic provider fallback.
Tests exercise actual SDK serialization, key isolation, streaming and errors with
an in-memory HTTP transport. They do not certify every server/model combination
against a live deployment. See the optional [server tutorial](tutorials/model-servers.md)
for setup choices outside the application's responsibility.

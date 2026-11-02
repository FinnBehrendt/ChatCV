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
Compatible roots end in `/v1` (API version); the client adds `/chat/completions`
or `/embeddings`. Native Ollama uses its own `/api/...` routes without `/v1`.

## Copyable configurations

Copy `.env.example` to `.env`. Choose **one chat block**, keeping embeddings
separate. Local examples assume ChatCV in Docker Desktop and a native host server.
For local Pixi replace `host.docker.internal` with `localhost`; for remote servers
use their reachable, protected hostname/interface.

### Default: OpenAI-compatible hosted chat

The default endpoint is OpenAI. Its compatible client needs `LLM_API_KEY`, not
just `OPENAI_API_KEY`.

```dotenv
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=your-openai-api-key
LLM_TEMPERATURE=0.1
LLM_TIMEOUT=60
```

Chat context leaves your machine; API usage may incur charges.

### Case 1: Ollama through the compatible API

Start `ollama serve` if needed, then `ollama pull qwen2.5:0.5b` in another terminal.
Both Ollama cases use this same server and model; only the client adapter changes.

```dotenv
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_MODEL=qwen2.5:0.5b
LLM_API_KEY=
LLM_TEMPERATURE=0.1
LLM_TIMEOUT=120
```

Clear any previous cloud chat key for this keyless local endpoint.

### Case 2: Ollama through the native adapter

```dotenv
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:0.5b
LLM_TEMPERATURE=0.1
OLLAMA_ENDPOINT=http://host.docker.internal:11434
OLLAMA_TIMEOUT=120
```

**No `/v1`.** Native Ollama ignores `LLM_BASE_URL`, `LLM_API_KEY` and
`LLM_TIMEOUT`: changing those will not change its server or timeout.

### Case 3: vLLM through the compatible API

In a hardware-compatible, separate vLLM environment, start:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --served-model-name chatcv-model --port 8001 --host 127.0.0.1
```

Use the advertised alias `chatcv-model`, not the Hugging Face path, in ChatCV:

```dotenv
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://host.docker.internal:8001/v1
LLM_MODEL=chatcv-model
LLM_API_KEY=
LLM_TEMPERATURE=0.1
LLM_TIMEOUT=120
```

Adjust the server's loopback binding if the app cannot reach it; protect access.
For authenticated servers, set their own `LLM_API_KEY`. These small Qwen models
are setup examples, not quality recommendations.

### Embeddings: independent of all chat cases

Keep the default hosted embeddings, or configure a separate local embedding model:

```dotenv
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=your-openai-api-key
EMBEDDING_TIMEOUT=60
```

These settings send document chunks and search queries to OpenAI even with local
chat. Qwen above is not an embedding model. See [Independent embeddings](#independent-embeddings).

ChatCV manages neither server. Compose has no `ollama` service: its old hostname
does not resolve, and `docker compose logs ollama` cannot show native-server logs.
See the [server tutorial](tutorials/model-servers.md). After `.env` changes run
`docker compose up -d` (not `restart`), or restart the local Pixi process.

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

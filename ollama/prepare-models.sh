#!/bin/sh
set -eu

ensure_model() {
    if ollama show "$1" >/dev/null 2>&1; then
        printf 'Model already available: %s\n' "$1"
    else
        printf 'Downloading model: %s\n' "$1"
        ollama pull "$1"
    fi
}

ensure_model "${OLLAMA_CHAT_MODEL:?Set OLLAMA_CHAT_MODEL}"

# Only provision embeddings served by this stack, never an external endpoint.
embedding_base_url="${EMBEDDING_BASE_URL:-}"
if [ "${EMBEDDING_PROVIDER:-}" = "openai_compatible" ] &&
    [ "${embedding_base_url%/}" = "http://ollama:11434/v1" ]; then
    ensure_model "${EMBEDDING_MODEL:?Set EMBEDDING_MODEL for local embeddings}"
fi

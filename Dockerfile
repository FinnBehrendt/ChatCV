# syntax=docker/dockerfile:1.7
FROM ghcr.io/prefix-dev/pixi:0.64.0 AS build

WORKDIR /app
COPY pixi.toml pixi.lock ./
RUN --mount=type=cache,target=/root/.cache/rattler \
    pixi install --locked -e runtime

# Use the locked interpreter for the optional Hub installer as well.
ENV PATH="/app/.pixi/envs/runtime/bin:${PATH}"

# Hub validators are optional additions outside the Pixi lockfile. Keep their
# credentials and installer caches in the build stage, not the runtime image.
RUN --mount=type=secret,id=guardrails_token \
    --mount=type=tmpfs,target=/root \
    if [ -f /run/secrets/guardrails_token ] && [ -s /run/secrets/guardrails_token ]; then \
        echo "Installing Guardrails AI with provided token..." && \
        guardrails configure --token "$(cat /run/secrets/guardrails_token)" --disable-metrics --disable-remote-inferencing && \
        guardrails hub install hub://guardrails/toxic_language --quiet && \
        guardrails hub install hub://guardrails/reading_time --quiet && \
        guardrails hub install hub://guardrails/profanity_free --quiet; \
    else \
        echo "No Guardrails token provided - skipping Guardrails AI installation"; \
    fi

FROM ghcr.io/prefix-dev/pixi:0.64.0 AS runtime

WORKDIR /app
# Keep the prefix identical: Conda environments are not generally relocatable.
COPY --from=build /app/.pixi/envs/runtime /app/.pixi/envs/runtime
COPY pixi.toml pixi.lock ./
ENV PATH="/app/.pixi/envs/runtime/bin:${PATH}"

RUN useradd --uid 10001 --create-home --shell /bin/bash app

COPY app/ ./app/
COPY config/ ./config/
COPY static/ ./static/
COPY templates/ ./templates/
COPY main.py ./
COPY --chown=app:app data/ ./data/

RUN mkdir -p /app/logs /app/data/vector_db && \
    chown -R app:app /app/logs /app/data

ENV LOG_DIR=/app/logs
ENV LOG_LEVEL=INFO
ENV DOCKER_CONTAINER=true

VOLUME ["/app/logs", "/app/data/vector_db"]

USER app
EXPOSE 8000
# No dependency resolution or installation on container startup.
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

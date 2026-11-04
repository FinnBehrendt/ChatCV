"""Fail fast on known worker settings incompatible with process-local state."""

import os
import shlex
import sys


def require_single_worker() -> None:
    settings: list[tuple[str, str]] = []
    for variable in ("WEB_CONCURRENCY", "UVICORN_WORKERS"):
        value = os.getenv(variable)
        if value is not None:
            settings.append((variable, value))
    try:
        gunicorn_arguments = shlex.split(os.getenv("GUNICORN_CMD_ARGS", ""))
    except ValueError as error:
        raise RuntimeError(
            "ChatCV requires a single worker; invalid GUNICORN_CMD_ARGS."
        ) from error
    for source, arguments in (
        ("command line", sys.argv[1:]),
        ("GUNICORN_CMD_ARGS", gunicorn_arguments),
    ):
        options = iter(arguments)
        for option in options:
            if option == "--":
                break
            if option in {"--workers", "-w"}:
                settings.append((source, next(options, "")))
            elif option.startswith("--workers="):
                settings.append((source, option.split("=", 1)[1]))
            elif option.startswith("-w") and len(option) > 2:
                settings.append((source, option[2:]))
    for source, value in settings:
        try:
            supported = int(value) == 1
        except ValueError:
            supported = False
        if not supported:
            raise RuntimeError(
                f"ChatCV requires a single worker; worker count in {source} must be 1. "
                "Conversation and authentication state are process-local."
            )

# Standard library imports
import logging
import os
import sys
from pathlib import Path


def setup_logging(
    log_dir: str | None = None,
    log_level: str | None = None,
    enable_file_logging: bool = True,
) -> dict[str, logging.Logger]:
    """
    Configure logging for the ChatCV application with persistent storage.

    Args:
        log_dir: Directory for log files (default: logs/ or LOG_DIR env var)
        log_level: Logging level (default: INFO or LOG_LEVEL env var)
        enable_file_logging: Whether to enable file logging (default: True)

    Returns:
        Dict[str, logging.Logger]: Dictionary of configured loggers
    """

    # Determine log directory - priority: parameter > env var > default
    if log_dir is None:
        log_dir = os.getenv("LOG_DIR", "logs")

    log_path = Path(log_dir)

    # Create logs directory if it doesn't exist (important for Docker volumes)
    try:
        log_path.mkdir(parents=True, exist_ok=True)
        print(f"Log directory created/verified: {log_path.absolute()}")
    except PermissionError:
        print(
            f"Warning: Cannot create log directory {log_path}. Falling back to console logging only."
        )
        enable_file_logging = False

    # Determine log level
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Configure formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    simple_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Configure handlers
    handlers: list[logging.Handler] = []

    # Console handler (always enabled)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(simple_formatter)
    handlers.append(console_handler)

    # File handlers (if enabled and possible)
    if enable_file_logging:
        try:
            # Main application log
            file_handler = logging.FileHandler(
                log_path / "chatcv.log", encoding="utf-8"
            )
            file_handler.setFormatter(detailed_formatter)
            handlers.append(file_handler)

            # Error log (only errors and critical)
            error_handler = logging.FileHandler(
                log_path / "chatcv_errors.log", encoding="utf-8"
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(detailed_formatter)
            handlers.append(error_handler)

            print(f"File logging enabled: {log_path.absolute()}")

        except Exception as e:
            print(f"Warning: File logging failed ({e}). Using console logging only.")

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Configure specific loggers
    logger_names = [
        "auth",
        "rag",
        "summary",
        "job_matching",
        "analytics",
        "api",
        "main",
        "config",
        "docker",
    ]

    loggers: dict[str, logging.Logger] = {}
    for name in logger_names:
        logger = logging.getLogger(f"chatcv.{name}")
        logger.setLevel(getattr(logging, log_level, logging.INFO))
        loggers[name] = logger

    # Log startup information
    main_logger = loggers["main"]
    main_logger.info("ChatCV logging initialized")
    main_logger.info(f"Log level: {log_level}")
    main_logger.info(f"Log directory: {log_path.absolute()}")
    main_logger.info(
        f"File logging: {'enabled' if enable_file_logging else 'disabled'}"
    )

    # Docker-specific logging
    if os.getenv("DOCKER_CONTAINER"):
        docker_logger = loggers["docker"]
        docker_logger.info("Running in Docker container")
        docker_logger.info(f"Log volume mounted: {log_path.absolute()}")

    return loggers


# Initialize loggers with environment-aware configuration
LOGGERS: dict[str, logging.Logger] = setup_logging()

# Export loggers for easy import
auth_logger: logging.Logger = LOGGERS["auth"]
rag_logger: logging.Logger = LOGGERS["rag"]
summary_logger: logging.Logger = LOGGERS["summary"]
job_matching_logger: logging.Logger = LOGGERS["job_matching"]
analytics_logger: logging.Logger = LOGGERS["analytics"]
api_logger: logging.Logger = LOGGERS["api"]
main_logger: logging.Logger = LOGGERS["main"]
config_logger: logging.Logger = LOGGERS["config"]
docker_logger: logging.Logger = LOGGERS["docker"]

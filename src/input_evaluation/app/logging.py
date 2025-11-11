from __future__ import annotations
import logging
from typing import Optional

import structlog


def setup_logging(level: int = logging.INFO) -> structlog.BoundLogger:
    """Configure the stdlib logging and structlog processors and return a logger.

    Call this early in application startup (for example from `main.py`).

    Args:
        level: logging level from the `logging` module (e.g. logging.INFO).

    Returns:
        A structlog BoundLogger instance.
    """
    logging.basicConfig(level=level, format="%(message)s")

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
    )

    return structlog.get_logger()


def get_logger() -> structlog.BoundLogger:
    """Return a structlog logger. If you need to ensure configuration has
    already happened, call `setup_logging()` first.
    """
    return structlog.get_logger()

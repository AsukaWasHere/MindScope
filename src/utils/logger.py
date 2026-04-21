"""
logger.py — Centralized logging setup for the MindScope pipeline.

Every module in src/ imports `logger` from here instead of using print().
Log format: [TIMESTAMP] [LEVEL] [MODULE] message

Usage in any module:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured logger for the given module name.

    The logger writes to stdout with a consistent format. If a logger
    for `name` already exists (e.g., called twice in same session),
    the existing one is returned without adding duplicate handlers.

    Args:
        name (str): Typically passed as __name__ from the calling module.
                    This makes log lines show the originating module path.

    Returns:
        logging.Logger: Configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Pipeline started")
        [2024-01-15 10:30:00] [INFO] [src.ingestion.loader] Pipeline started
    """
    logger = logging.getLogger(name)

    # Avoid adding multiple handlers if get_logger is called more than once
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler — writes to stdout so it works in notebooks and terminals
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    # Format: timestamp | level | module name | message
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent log messages from bubbling up to the root logger (avoids duplicates)
    logger.propagate = False

    return logger
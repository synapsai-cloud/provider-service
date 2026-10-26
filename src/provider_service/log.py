"""Process loggers for the provider service."""

from __future__ import annotations

import logging


def get_logger(logger_name: str = "provider") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

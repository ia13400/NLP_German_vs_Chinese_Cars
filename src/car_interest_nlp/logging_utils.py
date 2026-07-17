from __future__ import annotations

import logging

from .paths import REPORTS_DIR


def configure_logging(log_file_name: str = "project.log") -> logging.Logger:
    """Configure a reusable logger with a local report file."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("car_interest_nlp")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(REPORTS_DIR / log_file_name, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

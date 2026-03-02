"""Utility functions for the project."""
from __future__ import annotations
import logging
from pathlib import Path

def get_logger(name: str = "patentrisk") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)
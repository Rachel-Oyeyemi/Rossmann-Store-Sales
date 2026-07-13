"""Shared utilities for logging, paths, and JSON serialization."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any


def configure_logging(level: int = logging.INFO) -> None:
    """Configure consistent console logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_parent(path: str | Path) -> Path:
    """Create a file's parent directory and return the normalized path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_json(payload: Any, path: str | Path) -> None:
    """Persist JSON using readable formatting and safe type conversion."""
    target = ensure_parent(path)
    target.write_text(
        json.dumps(payload, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)),
        encoding="utf-8",
    )

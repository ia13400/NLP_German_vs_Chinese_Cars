from __future__ import annotations

import hashlib
import re
from typing import Any


def stable_hash(value: str) -> str:
    """Return a deterministic SHA-256 hash for inputs used in caching."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ensure_utf8_text(value: Any) -> str:
    """Normalize values to a UTF-8 text string."""
    if value is None:
        return ""
    return str(value)


def slugify(value: str) -> str:
    """Create a filesystem-safe slug from a label."""
    cleaned = re.sub(r"[^A-Za-z0-9äöüÄÖÜß\-_ ]+", "", value)
    return "_".join(cleaned.split()).lower()

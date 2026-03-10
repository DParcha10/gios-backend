"""
GIOS — Disk-based caching layer.

Wraps `diskcache` to provide TTL-aware caching for expensive operations
such as STAC queries, computed index arrays, and zonal statistics.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

import diskcache

from app.config import settings

logger = logging.getLogger(__name__)

# ── Module-level cache instance ─────────────────────────────────────────────

_cache: Optional[diskcache.Cache] = None


def get_cache() -> diskcache.Cache:
    """Return (and lazily initialise) the global disk cache."""
    global _cache
    if _cache is None:
        cache_path = Path(settings.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        _cache = diskcache.Cache(str(cache_path), size_limit=2**30)  # 1 GiB
        logger.info("Disk cache initialised at %s", cache_path)
    return _cache


def close_cache() -> None:
    """Flush and close the disk cache (call on app shutdown)."""
    global _cache
    if _cache is not None:
        _cache.close()
        _cache = None
        logger.info("Disk cache closed.")


# ── Key generation ──────────────────────────────────────────────────────────


def make_key(*parts: Any) -> str:
    """
    Build a deterministic cache key from an arbitrary sequence of
    serialisable values.
    """
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ── Public helpers ──────────────────────────────────────────────────────────


def cache_get(key: str) -> Any | None:
    """Retrieve a value from cache, or ``None`` if missing / expired."""
    cache = get_cache()
    value = cache.get(key)
    if value is not None:
        logger.debug("Cache HIT for key %s", key[:12])
    return value


def cache_set(key: str, value: Any, ttl: Optional[int] = None) -> None:
    """
    Store a value in cache.

    Parameters
    ----------
    key : str
        Cache key (use :func:`make_key` to generate).
    value : Any
        Picklable value to store.
    ttl : int, optional
        Time-to-live in seconds. Falls back to ``settings.cache_ttl_seconds``.
    """
    cache = get_cache()
    expire = ttl if ttl is not None else settings.cache_ttl_seconds
    cache.set(key, value, expire=expire)
    logger.debug("Cache SET for key %s (ttl=%ss)", key[:12], expire)


def cache_delete(key: str) -> bool:
    """Remove a single entry. Returns ``True`` if the key existed."""
    cache = get_cache()
    return cache.delete(key)


def cache_clear() -> None:
    """Wipe the entire cache."""
    cache = get_cache()
    cache.clear()
    logger.info("Cache cleared.")

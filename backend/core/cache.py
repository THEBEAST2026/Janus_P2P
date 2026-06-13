"""
core/cache.py
Redis wrapper — falls back to in-memory dict if Redis unavailable.
System never crashes on missing Redis.
"""

import json
import time
import asyncio
from typing import Any
import logging

logger = logging.getLogger("janus.cache")

# ─── In-memory fallback ───────────────────────────────────────────────────────
_mem: dict[str, tuple[Any, float | None]] = {}  # key → (value, expires_at)


def _mem_get(key: str) -> Any | None:
    entry = _mem.get(key)
    if entry is None:
        return None
    value, expires_at = entry
    if expires_at and time.time() > expires_at:
        del _mem[key]
        return None
    return value


def _mem_set(key: str, value: Any, ttl: int | None = None) -> None:
    expires_at = time.time() + ttl if ttl else None
    _mem[key] = (value, expires_at)


def _mem_delete(key: str) -> None:
    _mem.pop(key, None)


def _mem_publish(channel: str, message: str) -> None:
    # In-memory pub/sub just logs — real Redis handles the actual bus
    logger.debug(f"[mem-pubsub] {channel}: {message[:80]}")


# ─── Redis client ─────────────────────────────────────────────────────────────
_redis = None


async def init_redis(url: str = "redis://localhost:6379") -> bool:
    global _redis
    try:
        import redis.asyncio as aioredis
        client = aioredis.from_url(url, decode_responses=True)
        await client.ping()
        _redis = client
        logger.info("✅ Redis connected")
        return True
    except Exception as e:
        logger.warning(f"⚠️  Redis unavailable ({e}) — using in-memory fallback")
        _redis = None
        return False


async def cache_get(key: str) -> Any | None:
    if _redis:
        try:
            raw = await _redis.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            pass
    return _mem_get(key)


async def cache_set(key: str, value: Any, ttl: int | None = None) -> None:
    serialized = json.dumps(value)
    if _redis:
        try:
            if ttl:
                await _redis.setex(key, ttl, serialized)
            else:
                await _redis.set(key, serialized)
            return
        except Exception:
            pass
    _mem_set(key, value, ttl)


async def cache_delete(key: str) -> None:
    if _redis:
        try:
            await _redis.delete(key)
            return
        except Exception:
            pass
    _mem_delete(key)


async def cache_lpush(key: str, value: Any, maxlen: int = 1000) -> None:
    """Push to ring buffer list."""
    serialized = json.dumps(value)
    if _redis:
        try:
            await _redis.lpush(key, serialized)
            await _redis.ltrim(key, 0, maxlen - 1)
            return
        except Exception:
            pass
    existing = _mem_get(key) or []
    existing.insert(0, value)
    _mem_set(key, existing[:maxlen])


async def cache_lrange(key: str, start: int = 0, end: int = -1) -> list:
    if _redis:
        try:
            raw_list = await _redis.lrange(key, start, end)
            return [json.loads(r) for r in raw_list]
        except Exception:
            pass
    existing = _mem_get(key) or []
    if end == -1:
        return existing[start:]
    return existing[start:end + 1]


async def publish(channel: str, message: dict) -> None:
    serialized = json.dumps(message)
    if _redis:
        try:
            await _redis.publish(channel, serialized)
            return
        except Exception:
            pass
    _mem_publish(channel, serialized)


async def redis_health() -> dict:
    if _redis:
        try:
            info = await _redis.info("server")
            return {
                "status": "connected",
                "version": info.get("redis_version", "unknown"),
                "mode": "redis"
            }
        except Exception:
            pass
    return {"status": "fallback", "mode": "in-memory"}

import json
import logging
import os
from typing import Any, Optional
import redis

logger = logging.getLogger("uvicorn.error")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
USER_CACHE_TTL_SECONDS = 300  # 5 minutes

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=1,
    socket_timeout=1,
)


def build_user_cache_key(user_id: int) -> str:
    return f"user:profile:{user_id}"


def get_cached_user(user_id: int) -> Optional[dict[str, Any]]:
    key = build_user_cache_key(user_id)
    try:
        cached_payload = redis_client.get(key)
        if cached_payload:
            return json.loads(cached_payload)
    except redis.RedisError as exc:
        logger.warning(f"Redis GET failed for {key} (falling back to DB): {exc}")
    return None


def set_cached_user(
    user_id: int, user_data: dict[str, Any], ttl: int = USER_CACHE_TTL_SECONDS
) -> None:
    key = build_user_cache_key(user_id)
    try:
        redis_client.setex(key, ttl, json.dumps(user_data, default=str))
    except redis.RedisError as exc:
        logger.warning(f"Redis SETEX failed for {key}: {exc}")


def invalidate_user_cache(user_id: int) -> None:
    key = build_user_cache_key(user_id)
    try:
        redis_client.delete(key)
    except redis.RedisError as exc:
        logger.warning(f"Redis DEL failed for {key}: {exc}")
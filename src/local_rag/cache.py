from __future__ import annotations

import json
import logging
from typing import Sequence

from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.exceptions import RedisError


logger = logging.getLogger(__name__)


class RedisKeyValueCache:
    def __init__(
        self,
        url: str,
        key_prefix: str = "local_rag",
        default_ttl_seconds: int | None = None,
        sync_client: Redis | None = None,
        async_client: AsyncRedis | None = None,
    ) -> None:
        self.url = url
        self.key_prefix = key_prefix.strip(":") or "local_rag"
        self.default_ttl_seconds = default_ttl_seconds
        self._sync_client = sync_client or Redis.from_url(url, decode_responses=True)
        self._async_client = async_client or AsyncRedis.from_url(
            url,
            decode_responses=True,
        )
        self._warned_sync = False
        self._warned_async = False

    def build_key(self, *parts: object) -> str:
        serialized_parts = [str(part).replace(" ", "_") for part in parts]
        return ":".join([self.key_prefix, *serialized_parts])

    def get_many(self, keys: Sequence[str]) -> dict[str, str]:
        if not keys:
            return {}

        try:
            values = self._sync_client.mget(list(keys))
        except RedisError as exc:
            self._warn_sync(exc)
            return {}

        return {
            key: value
            for key, value in zip(keys, values, strict=False)
            if value is not None
        }

    async def get_many_async(self, keys: Sequence[str]) -> dict[str, str]:
        if not keys:
            return {}

        try:
            values = await self._async_client.mget(list(keys))
        except RedisError as exc:
            self._warn_async(exc)
            return {}

        return {
            key: value
            for key, value in zip(keys, values, strict=False)
            if value is not None
        }

    def set_many(
        self,
        mapping: dict[str, str],
        ttl_seconds: int | None = None,
    ) -> None:
        if not mapping:
            return

        expiration = (
            ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        )
        try:
            pipeline = self._sync_client.pipeline()
            for key, value in mapping.items():
                if expiration is not None:
                    pipeline.setex(key, expiration, value)
                else:
                    pipeline.set(key, value)
            pipeline.execute()
        except RedisError as exc:
            self._warn_sync(exc)

    async def set_many_async(
        self,
        mapping: dict[str, str],
        ttl_seconds: int | None = None,
    ) -> None:
        if not mapping:
            return

        expiration = (
            ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        )
        try:
            pipeline = self._async_client.pipeline()
            for key, value in mapping.items():
                if expiration is not None:
                    pipeline.setex(key, expiration, value)
                else:
                    pipeline.set(key, value)
            await pipeline.execute()
        except RedisError as exc:
            self._warn_async(exc)

    def get_json_many(self, keys: Sequence[str]) -> dict[str, object]:
        raw_values = self.get_many(keys)
        return {key: json.loads(value) for key, value in raw_values.items()}

    async def get_json_many_async(self, keys: Sequence[str]) -> dict[str, object]:
        raw_values = await self.get_many_async(keys)
        return {key: json.loads(value) for key, value in raw_values.items()}

    def set_json_many(
        self,
        mapping: dict[str, object],
        ttl_seconds: int | None = None,
    ) -> None:
        self.set_many(
            {
                key: json.dumps(value, ensure_ascii=False)
                for key, value in mapping.items()
            },
            ttl_seconds=ttl_seconds,
        )

    async def set_json_many_async(
        self,
        mapping: dict[str, object],
        ttl_seconds: int | None = None,
    ) -> None:
        await self.set_many_async(
            {
                key: json.dumps(value, ensure_ascii=False)
                for key, value in mapping.items()
            },
            ttl_seconds=ttl_seconds,
        )

    def _warn_sync(self, exc: Exception) -> None:
        if self._warned_sync:
            return
        self._warned_sync = True
        logger.warning("Redis sync cache unavailable: %s", exc)

    def _warn_async(self, exc: Exception) -> None:
        if self._warned_async:
            return
        self._warned_async = True
        logger.warning("Redis async cache unavailable: %s", exc)


class InMemoryKeyValueCache:
    def __init__(self, key_prefix: str = "local_rag") -> None:
        self.key_prefix = key_prefix.strip(":") or "local_rag"
        self._store: dict[str, str] = {}

    def build_key(self, *parts: object) -> str:
        serialized_parts = [str(part).replace(" ", "_") for part in parts]
        return ":".join([self.key_prefix, *serialized_parts])

    def get_many(self, keys: Sequence[str]) -> dict[str, str]:
        return {key: self._store[key] for key in keys if key in self._store}

    async def get_many_async(self, keys: Sequence[str]) -> dict[str, str]:
        return self.get_many(keys)

    def set_many(self, mapping: dict[str, str], ttl_seconds: int | None = None) -> None:
        self._store.update(mapping)

    async def set_many_async(
        self,
        mapping: dict[str, str],
        ttl_seconds: int | None = None,
    ) -> None:
        self.set_many(mapping, ttl_seconds=ttl_seconds)

    def get_json_many(self, keys: Sequence[str]) -> dict[str, object]:
        raw_values = self.get_many(keys)
        return {key: json.loads(value) for key, value in raw_values.items()}

    async def get_json_many_async(self, keys: Sequence[str]) -> dict[str, object]:
        raw_values = await self.get_many_async(keys)
        return {key: json.loads(value) for key, value in raw_values.items()}

    def set_json_many(
        self,
        mapping: dict[str, object],
        ttl_seconds: int | None = None,
    ) -> None:
        self.set_many(
            {
                key: json.dumps(value, ensure_ascii=False)
                for key, value in mapping.items()
            },
            ttl_seconds=ttl_seconds,
        )

    async def set_json_many_async(
        self,
        mapping: dict[str, object],
        ttl_seconds: int | None = None,
    ) -> None:
        await self.set_many_async(
            {
                key: json.dumps(value, ensure_ascii=False)
                for key, value in mapping.items()
            },
            ttl_seconds=ttl_seconds,
        )

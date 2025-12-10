"""
Tests for caching functionality
"""
import pytest
import asyncio
import time
from app.core.cache import MemoryCache, CacheEntry, _safe_serialize, _safe_deserialize


class TestCacheEntry:
    """Tests for CacheEntry dataclass"""

    def test_not_expired_without_expiry(self):
        """Entry without expiry time should never expire"""
        entry = CacheEntry(value="test", created_at=time.time())
        assert not entry.is_expired()

    def test_not_expired_before_expiry(self):
        """Entry should not be expired before expiry time"""
        entry = CacheEntry(
            value="test",
            created_at=time.time(),
            expires_at=time.time() + 3600
        )
        assert not entry.is_expired()

    def test_expired_after_expiry(self):
        """Entry should be expired after expiry time"""
        entry = CacheEntry(
            value="test",
            created_at=time.time() - 10,
            expires_at=time.time() - 5
        )
        assert entry.is_expired()


class TestSerialization:
    """Tests for serialization functions"""

    def test_serialize_string(self):
        """Should serialize strings"""
        result = _safe_serialize("hello")
        assert result == '"hello"'

    def test_serialize_dict(self):
        """Should serialize dictionaries"""
        data = {"key": "value", "number": 42}
        result = _safe_serialize(data)
        assert "key" in result
        assert "value" in result

    def test_serialize_list(self):
        """Should serialize lists"""
        data = [1, 2, 3]
        result = _safe_serialize(data)
        deserialized = _safe_deserialize(result)
        assert deserialized == [1, 2, 3]

    def test_deserialize_json(self):
        """Should deserialize valid JSON"""
        result = _safe_deserialize('{"key": "value"}')
        assert result == {"key": "value"}

    def test_deserialize_invalid_returns_none(self):
        """Should return None for invalid JSON"""
        result = _safe_deserialize("not valid json {")
        assert result is None


class TestMemoryCache:
    """Tests for MemoryCache backend"""

    @pytest.mark.asyncio
    async def test_set_and_get(self, memory_cache):
        """Should store and retrieve values"""
        await memory_cache.set("test_key", "test_value")
        result = await memory_cache.get("test_key")
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, memory_cache):
        """Should return None for nonexistent keys"""
        result = await memory_cache.get("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_removes_key(self, memory_cache):
        """Should delete keys"""
        await memory_cache.set("delete_me", "value")
        deleted = await memory_cache.delete("delete_me")
        assert deleted
        result = await memory_cache.get("delete_me")
        assert result is None

    @pytest.mark.asyncio
    async def test_exists_returns_correct_value(self, memory_cache):
        """Should correctly check key existence"""
        await memory_cache.set("exists_key", "value")
        assert await memory_cache.exists("exists_key")
        assert not await memory_cache.exists("not_exists_key")

    @pytest.mark.asyncio
    async def test_clear_removes_all(self, memory_cache):
        """Should clear all entries"""
        await memory_cache.set("key1", "value1")
        await memory_cache.set("key2", "value2")
        await memory_cache.clear()
        assert await memory_cache.get("key1") is None
        assert await memory_cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Should expire entries after TTL"""
        cache = MemoryCache(max_size=10, default_ttl=1)
        try:
            await cache.set("expire_key", "value", ttl=1)
            # Should exist immediately
            assert await cache.get("expire_key") == "value"
            # Wait for expiration
            await asyncio.sleep(1.5)
            # Should be expired
            assert await cache.get("expire_key") is None
        finally:
            await cache.shutdown()

    @pytest.mark.asyncio
    async def test_max_size_eviction(self):
        """Should evict old entries when max size reached"""
        cache = MemoryCache(max_size=3, default_ttl=3600)
        try:
            await cache.set("key1", "value1")
            await cache.set("key2", "value2")
            await cache.set("key3", "value3")
            await cache.set("key4", "value4")  # Should trigger eviction
            # One of the first keys should be evicted
            keys = await cache.keys()
            assert len(keys) <= 3
        finally:
            await cache.shutdown()

    @pytest.mark.asyncio
    async def test_keys_with_pattern(self, memory_cache):
        """Should filter keys by pattern"""
        await memory_cache.set("search:query1", "value1")
        await memory_cache.set("search:query2", "value2")
        await memory_cache.set("page:url1", "value3")

        search_keys = await memory_cache.keys("search:*")
        assert len(search_keys) == 2
        assert all(k.startswith("search:") for k in search_keys)

    @pytest.mark.asyncio
    async def test_complex_values(self, memory_cache):
        """Should handle complex data structures"""
        complex_data = {
            "list": [1, 2, 3],
            "nested": {"a": "b"},
            "number": 42,
            "string": "hello"
        }
        await memory_cache.set("complex", complex_data)
        result = await memory_cache.get("complex")
        assert result == complex_data

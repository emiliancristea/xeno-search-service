import json
import hashlib
import asyncio
import pickle
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Union
from pathlib import Path
import aioredis
from cachetools import TTLCache, LRUCache
import structlog
from datetime import datetime, timedelta

from app.core.config import get_config

logger = structlog.get_logger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern"""
        pass


class RedisCache(CacheBackend):
    """Redis-based cache backend"""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client: Optional[aioredis.Redis] = None
        self.connected = False
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,
                socket_connect_timeout=10,
                socket_timeout=10,
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test connection
            await self.redis_client.ping()
            self.connected = True
            logger.info("Connected to Redis cache", redis_url=self.redis_url)
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            self.connected = False
            raise
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
            self.connected = False
            logger.info("Disconnected from Redis cache")
    
    async def _ensure_connected(self):
        """Ensure Redis connection is active"""
        if not self.connected or not self.redis_client:
            await self.connect()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        try:
            await self._ensure_connected()
            data = await self.redis_client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error("Redis get error", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache"""
        try:
            await self._ensure_connected()
            data = pickle.dumps(value)
            if ttl:
                await self.redis_client.setex(key, ttl, data)
            else:
                await self.redis_client.set(key, data)
            return True
        except Exception as e:
            logger.error("Redis set error", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache"""
        try:
            await self._ensure_connected()
            result = await self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error("Redis delete error", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache"""
        try:
            await self._ensure_connected()
            result = await self.redis_client.exists(key)
            return result > 0
        except Exception as e:
            logger.error("Redis exists error", key=key, error=str(e))
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            await self._ensure_connected()
            await self.redis_client.flushdb()
            return True
        except Exception as e:
            logger.error("Redis clear error", error=str(e))
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern"""
        try:
            await self._ensure_connected()
            keys = await self.redis_client.keys(pattern)
            return [key.decode() if isinstance(key, bytes) else key for key in keys]
        except Exception as e:
            logger.error("Redis keys error", pattern=pattern, error=str(e))
            return []


class MemoryCache(CacheBackend):
    """In-memory cache backend using cachetools"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.cache = TTLCache(maxsize=max_size, ttl=default_ttl)
        self.lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        async with self.lock:
            return self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache"""
        try:
            async with self.lock:
                if ttl:
                    # Create a new TTL cache entry with custom TTL
                    self.cache[key] = value
                else:
                    self.cache[key] = value
            return True
        except Exception as e:
            logger.error("Memory cache set error", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from memory cache"""
        try:
            async with self.lock:
                if key in self.cache:
                    del self.cache[key]
                    return True
                return False
        except Exception as e:
            logger.error("Memory cache delete error", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in memory cache"""
        async with self.lock:
            return key in self.cache
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            async with self.lock:
                self.cache.clear()
            return True
        except Exception as e:
            logger.error("Memory cache clear error", error=str(e))
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern (simplified pattern matching)"""
        async with self.lock:
            if pattern == "*":
                return list(self.cache.keys())
            else:
                # Simple pattern matching
                import fnmatch
                return [key for key in self.cache.keys() if fnmatch.fnmatch(key, pattern)]


class FileCache(CacheBackend):
    """File-based cache backend"""
    
    def __init__(self, cache_dir: str = ".cache", max_size: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self.lock = asyncio.Lock()
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key"""
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.cache"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from file cache"""
        try:
            file_path = self._get_file_path(key)
            if not file_path.exists():
                return None
            
            async with self.lock:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                    # Check if expired
                    if data.get("expires_at") and time.time() > data["expires_at"]:
                        file_path.unlink()
                        return None
                    return data["value"]
        except Exception as e:
            logger.error("File cache get error", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in file cache"""
        try:
            file_path = self._get_file_path(key)
            expires_at = time.time() + ttl if ttl else None
            
            data = {
                "value": value,
                "created_at": time.time(),
                "expires_at": expires_at
            }
            
            async with self.lock:
                with open(file_path, "wb") as f:
                    pickle.dump(data, f)
            return True
        except Exception as e:
            logger.error("File cache set error", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from file cache"""
        try:
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            logger.error("File cache delete error", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in file cache"""
        file_path = self._get_file_path(key)
        return file_path.exists()
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            async with self.lock:
                for file_path in self.cache_dir.glob("*.cache"):
                    file_path.unlink()
            return True
        except Exception as e:
            logger.error("File cache clear error", error=str(e))
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern"""
        # This is complex for file cache as we'd need to read all files
        # For now, return empty list
        return []


class EnhancedCacheManager:
    """Enhanced cache manager with multiple backends and advanced features"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.backend: Optional[CacheBackend] = None
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
        self.content_hashes: Dict[str, str] = {}  # For deduplication
        
    async def initialize(self):
        """Initialize cache backend"""
        cache_config = self.config.get_cache_config()
        
        if not cache_config["enabled"]:
            logger.info("Cache is disabled")
            return
        
        cache_type = cache_config["type"]
        
        try:
            if cache_type == "redis":
                self.backend = RedisCache(cache_config["redis_url"])
                await self.backend.connect()
            elif cache_type == "memory":
                self.backend = MemoryCache()
            elif cache_type == "file":
                self.backend = FileCache()
            else:
                logger.warning("Unknown cache type, using memory cache", cache_type=cache_type)
                self.backend = MemoryCache()
                
            logger.info("Cache initialized", cache_type=cache_type)
        except Exception as e:
            logger.error("Failed to initialize cache", error=str(e))
            # Fallback to memory cache
            self.backend = MemoryCache()
    
    async def shutdown(self):
        """Shutdown cache backend"""
        if self.backend and hasattr(self.backend, 'disconnect'):
            await self.backend.disconnect()
    
    def _generate_cache_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from prefix and parameters"""
        key_data = json.dumps(kwargs, sort_keys=True)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate content hash for deduplication"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def get_search_results(self, query: str, search_type: str, num_results: int) -> Optional[Any]:
        """Get cached search results"""
        if not self.backend:
            return None
        
        try:
            key = self._generate_cache_key(
                "search_results",
                query=query,
                search_type=search_type,
                num_results=num_results
            )
            
            result = await self.backend.get(key)
            if result:
                self.cache_stats["hits"] += 1
                logger.debug("Cache hit for search results", query=query)
                return result
            else:
                self.cache_stats["misses"] += 1
                logger.debug("Cache miss for search results", query=query)
                return None
        except Exception as e:
            self.cache_stats["errors"] += 1
            logger.error("Cache get error for search results", error=str(e))
            return None
    
    async def set_search_results(self, query: str, search_type: str, num_results: int, results: Any) -> bool:
        """Cache search results"""
        if not self.backend:
            return False
        
        try:
            key = self._generate_cache_key(
                "search_results",
                query=query,
                search_type=search_type,
                num_results=num_results
            )
            
            ttl = self.config.cache_ttl_search_results
            success = await self.backend.set(key, results, ttl)
            
            if success:
                self.cache_stats["sets"] += 1
                logger.debug("Cached search results", query=query)
            
            return success
        except Exception as e:
            self.cache_stats["errors"] += 1
            logger.error("Cache set error for search results", error=str(e))
            return False
    
    async def get_page_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached page content"""
        if not self.backend:
            return None
        
        try:
            key = self._generate_cache_key("page_content", url=url)
            
            result = await self.backend.get(key)
            if result:
                self.cache_stats["hits"] += 1
                logger.debug("Cache hit for page content", url=url)
                return result
            else:
                self.cache_stats["misses"] += 1
                logger.debug("Cache miss for page content", url=url)
                return None
        except Exception as e:
            self.cache_stats["errors"] += 1
            logger.error("Cache get error for page content", error=str(e))
            return None
    
    async def set_page_content(self, url: str, content: Dict[str, Any]) -> bool:
        """Cache page content with deduplication"""
        if not self.backend:
            return False
        
        try:
            # Check for content deduplication
            if self.config.content_deduplication and content.get("raw_text"):
                content_hash = self._generate_content_hash(content["raw_text"])
                
                # Check if we've seen this content before
                if content_hash in self.content_hashes:
                    logger.debug("Duplicate content detected", url=url, original_url=self.content_hashes[content_hash])
                    # Still cache but mark as duplicate
                    content["is_duplicate"] = True
                    content["original_url"] = self.content_hashes[content_hash]
                else:
                    self.content_hashes[content_hash] = url
                    content["is_duplicate"] = False
            
            key = self._generate_cache_key("page_content", url=url)
            ttl = self.config.cache_ttl_page_content
            
            success = await self.backend.set(key, content, ttl)
            
            if success:
                self.cache_stats["sets"] += 1
                logger.debug("Cached page content", url=url)
            
            return success
        except Exception as e:
            self.cache_stats["errors"] += 1
            logger.error("Cache set error for page content", error=str(e))
            return False
    
    async def get_summary(self, content_hash: str) -> Optional[str]:
        """Get cached summary"""
        if not self.backend:
            return None
        
        try:
            key = self._generate_cache_key("summary", content_hash=content_hash)
            
            result = await self.backend.get(key)
            if result:
                self.cache_stats["hits"] += 1
                logger.debug("Cache hit for summary", content_hash=content_hash[:16])
                return result
            else:
                self.cache_stats["misses"] += 1
                logger.debug("Cache miss for summary", content_hash=content_hash[:16])
                return None
        except Exception as e:
            self.cache_stats["errors"] += 1
            logger.error("Cache get error for summary", error=str(e))
            return None
    
    async def set_summary(self, content_hash: str, summary: str) -> bool:
        """Cache summary"""
        if not self.backend:
            return False
        
        try:
            key = self._generate_cache_key("summary", content_hash=content_hash)
            ttl = self.config.cache_ttl_summaries
            
            success = await self.backend.set(key, summary, ttl)
            
            if success:
                self.cache_stats["sets"] += 1
                logger.debug("Cached summary", content_hash=content_hash[:16])
            
            return success
        except Exception as e:
            self.cache_stats["errors"] += 1
            logger.error("Cache set error for summary", error=str(e))
            return False
    
    async def clear_cache(self, pattern: str = "*") -> bool:
        """Clear cache entries matching pattern"""
        if not self.backend:
            return False
        
        try:
            if pattern == "*":
                success = await self.backend.clear()
            else:
                keys = await self.backend.keys(pattern)
                for key in keys:
                    await self.backend.delete(key)
                success = True
            
            if success:
                logger.info("Cache cleared", pattern=pattern)
            
            return success
        except Exception as e:
            logger.error("Cache clear error", error=str(e))
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }


# Global cache manager instance
cache_manager = EnhancedCacheManager()


async def get_cache_manager() -> EnhancedCacheManager:
    """Get global cache manager instance"""
    return cache_manager


async def initialize_cache():
    """Initialize global cache manager"""
    await cache_manager.initialize()


async def shutdown_cache():
    """Shutdown global cache manager"""
    await cache_manager.shutdown()
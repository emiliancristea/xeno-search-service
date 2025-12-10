"""
Pytest configuration and fixtures for Xeno Search Service tests
"""
import pytest
import asyncio
import os
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock, AsyncMock

# Set test environment variables before importing app modules
os.environ.setdefault('ENVIRONMENT', 'test')
os.environ.setdefault('LOGGING_LEVEL', 'DEBUG')
os.environ.setdefault('CACHE_ENABLED', 'true')
os.environ.setdefault('CACHE_TYPE', 'memory')
os.environ.setdefault('AUTH_ENABLED', 'false')
os.environ.setdefault('RATE_LIMITING_ENABLED', 'false')


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for testing HTTP calls"""
    client = AsyncMock()
    response = MagicMock()
    response.status_code = 200
    response.text = "<html><body><h1>Test</h1><p>Test content</p></body></html>"
    response.raise_for_status = MagicMock()
    response.url = "https://example.com"
    response.is_redirect = False
    client.get.return_value = response
    return client


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return [
        {
            'title': 'Test Result 1',
            'url': 'https://example.com/result1',
            'snippet': 'This is a test result snippet'
        },
        {
            'title': 'Test Result 2',
            'url': 'https://example.org/result2',
            'snippet': 'Another test result snippet'
        }
    ]


@pytest.fixture
def sample_source():
    """Sample Source object for testing"""
    from app.models.search_models import Source
    return Source(
        url="https://example.com/test",
        title="Test Title",
        snippet="Test snippet content",
        raw_text="This is the full raw text content for testing purposes.",
        summary="Test summary"
    )


@pytest.fixture
async def memory_cache():
    """In-memory cache for testing"""
    from app.core.cache import MemoryCache
    cache = MemoryCache(max_size=100, default_ttl=60)
    yield cache
    await cache.shutdown()


@pytest.fixture
def valid_urls():
    """List of valid URLs that should pass SSRF validation"""
    return [
        "https://example.com",
        "https://www.google.com/search?q=test",
        "http://example.org/path/to/page",
        "https://api.github.com/repos",
    ]


@pytest.fixture
def blocked_urls():
    """List of URLs that should be blocked by SSRF protection"""
    return [
        "http://localhost/admin",
        "http://127.0.0.1:8080",
        "http://192.168.1.1/",
        "http://10.0.0.1/internal",
        "http://169.254.169.254/metadata",  # Cloud metadata
        "http://[::1]/",  # IPv6 loopback
    ]

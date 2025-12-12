"""
Xeno Search Engine - Web Crawler
A production-grade web crawler for building your own search index
"""

import asyncio
import hashlib
import time
import random
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass, field

import aiohttp
import redis.asyncio as redis
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CrawlResult:
    """Result of crawling a single page"""
    url: str
    title: str
    content: str
    description: str
    links: List[str]
    crawled_at: datetime
    status_code: int
    content_type: str
    word_count: int
    language: str = "en"
    domain: str = ""
    path: str = ""
    meta_keywords: List[str] = field(default_factory=list)
    headings: Dict[str, List[str]] = field(default_factory=dict)
    images: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "description": self.description,
            "links": self.links,
            "crawled_at": self.crawled_at.isoformat(),
            "status_code": self.status_code,
            "content_type": self.content_type,
            "word_count": self.word_count,
            "language": self.language,
            "domain": self.domain,
            "path": self.path,
            "meta_keywords": self.meta_keywords,
            "headings": self.headings,
            "images": self.images,
        }


@dataclass
class CrawlConfig:
    """Configuration for the crawler"""
    max_pages_per_domain: int = 1000
    max_depth: int = 5
    delay_between_requests: float = 1.0  # seconds
    request_timeout: int = 30
    max_concurrent_requests: int = 10
    respect_robots_txt: bool = True
    follow_external_links: bool = False
    allowed_content_types: List[str] = field(default_factory=lambda: ["text/html"])
    excluded_extensions: List[str] = field(default_factory=lambda: [
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".zip", ".rar", ".tar", ".gz", ".7z",
        ".mp3", ".mp4", ".avi", ".mkv", ".mov",
        ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".webp",
        ".css", ".js", ".json", ".xml", ".rss"
    ])
    user_agents: List[str] = field(default_factory=lambda: [
        "XenoSearchBot/1.0 (+https://xenolabs.ai/bot)",
        "Mozilla/5.0 (compatible; XenoSearchBot/1.0; +https://xenolabs.ai/bot)",
    ])


class XenoCrawler:
    """
    Xeno Search Engine Web Crawler

    Features:
    - Async crawling with rate limiting
    - Respects robots.txt
    - URL deduplication via Redis
    - Content extraction with BeautifulSoup
    - Link discovery and queueing
    - Domain-level politeness
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/2",
        config: Optional[CrawlConfig] = None
    ):
        self.config = config or CrawlConfig()
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.robots_cache: Dict[str, Tuple[RobotFileParser, float]] = {}
        self.domain_last_request: Dict[str, float] = {}
        self.session: Optional[aiohttp.ClientSession] = None

        # Queue keys in Redis
        self.QUEUE_KEY = "xeno:crawler:queue"
        self.VISITED_KEY = "xeno:crawler:visited"
        self.DOMAIN_COUNT_KEY = "xeno:crawler:domain_count"
        self.STATS_KEY = "xeno:crawler:stats"

    async def initialize(self):
        """Initialize Redis connection and HTTP session"""
        self.redis = redis.from_url(self.redis_url, decode_responses=True)
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
            headers={"User-Agent": random.choice(self.config.user_agents)}
        )
        logger.info("Crawler initialized", redis_url=self.redis_url)

    async def shutdown(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
        if self.redis:
            await self.redis.close()
        logger.info("Crawler shutdown complete")

    def _get_url_hash(self, url: str) -> str:
        """Get unique hash for URL"""
        normalized = url.lower().rstrip("/")
        return hashlib.md5(normalized.encode()).hexdigest()

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        parsed = urlparse(url)
        return parsed.netloc

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid for crawling"""
        try:
            parsed = urlparse(url)

            # Must have scheme and netloc
            if not parsed.scheme or not parsed.netloc:
                return False

            # Only http/https
            if parsed.scheme not in ["http", "https"]:
                return False

            # Check excluded extensions
            path_lower = parsed.path.lower()
            for ext in self.config.excluded_extensions:
                if path_lower.endswith(ext):
                    return False

            return True
        except Exception:
            return False

    async def _check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt"""
        if not self.config.respect_robots_txt:
            return True

        domain = self._get_domain(url)
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{domain}/robots.txt"

        # Check cache (cache for 1 hour)
        if domain in self.robots_cache:
            parser, cached_time = self.robots_cache[domain]
            if time.time() - cached_time < 3600:
                return parser.can_fetch(self.config.user_agents[0], url)

        # Fetch robots.txt
        try:
            async with self.session.get(robots_url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    parser = RobotFileParser()
                    parser.parse(content.splitlines())
                else:
                    # No robots.txt, allow all
                    parser = RobotFileParser()
                    parser.allow_all = True

                self.robots_cache[domain] = (parser, time.time())
                return parser.can_fetch(self.config.user_agents[0], url)
        except Exception as e:
            logger.warning("Failed to fetch robots.txt", domain=domain, error=str(e))
            return True  # Allow if can't fetch

    async def _apply_rate_limit(self, domain: str):
        """Apply rate limiting per domain"""
        now = time.time()
        last_request = self.domain_last_request.get(domain, 0)
        wait_time = self.config.delay_between_requests - (now - last_request)

        if wait_time > 0:
            await asyncio.sleep(wait_time)

        self.domain_last_request[domain] = time.time()

    async def add_to_queue(self, urls: List[str], depth: int = 0, priority: int = 0):
        """Add URLs to the crawl queue"""
        for url in urls:
            if not self._is_valid_url(url):
                continue

            url_hash = self._get_url_hash(url)

            # Check if already visited
            if await self.redis.sismember(self.VISITED_KEY, url_hash):
                continue

            # Check domain limit
            domain = self._get_domain(url)
            domain_count = await self.redis.hget(self.DOMAIN_COUNT_KEY, domain) or 0
            if int(domain_count) >= self.config.max_pages_per_domain:
                continue

            # Add to queue with priority (lower = higher priority)
            score = priority * 1000 + depth
            await self.redis.zadd(self.QUEUE_KEY, {f"{url}|{depth}": score})

        queue_size = await self.redis.zcard(self.QUEUE_KEY)
        logger.info("Added URLs to queue", count=len(urls), queue_size=queue_size)

    async def get_next_url(self) -> Optional[Tuple[str, int]]:
        """Get next URL from queue"""
        result = await self.redis.zpopmin(self.QUEUE_KEY)
        if not result:
            return None

        item, score = result[0]
        url, depth = item.rsplit("|", 1)
        return url, int(depth)

    async def mark_visited(self, url: str):
        """Mark URL as visited"""
        url_hash = self._get_url_hash(url)
        await self.redis.sadd(self.VISITED_KEY, url_hash)

        # Increment domain count
        domain = self._get_domain(url)
        await self.redis.hincrby(self.DOMAIN_COUNT_KEY, domain, 1)

    def _extract_content(self, html: str, url: str) -> CrawlResult:
        """Extract content from HTML"""
        soup = BeautifulSoup(html, "lxml")
        parsed_url = urlparse(url)

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()

        # Extract title
        title = ""
        if soup.title:
            title = soup.title.string or ""
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)

        # Extract meta description
        description = ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc:
            description = meta_desc.get("content", "")

        # Extract meta keywords
        meta_keywords = []
        meta_kw = soup.find("meta", attrs={"name": "keywords"})
        if meta_kw:
            keywords_str = meta_kw.get("content", "")
            meta_keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]

        # Extract headings
        headings = {"h1": [], "h2": [], "h3": []}
        for level in ["h1", "h2", "h3"]:
            for heading in soup.find_all(level):
                text = heading.get_text(strip=True)
                if text:
                    headings[level].append(text)

        # Extract main content
        main_content = soup.find("main") or soup.find("article") or soup.find("body")
        if main_content:
            content = main_content.get_text(separator=" ", strip=True)
        else:
            content = soup.get_text(separator=" ", strip=True)

        # Clean content
        content = re.sub(r'\s+', ' ', content).strip()

        # Extract links
        links = []
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            absolute_url = urljoin(url, href)
            if self._is_valid_url(absolute_url):
                links.append(absolute_url)

        # Extract images
        images = []
        for img in soup.find_all("img", src=True)[:10]:  # Limit to 10 images
            images.append({
                "src": urljoin(url, img.get("src", "")),
                "alt": img.get("alt", ""),
            })

        # Detect language (simple heuristic)
        lang = soup.find("html").get("lang", "en") if soup.find("html") else "en"

        return CrawlResult(
            url=url,
            title=title[:500],
            content=content[:50000],  # Limit content size
            description=description[:1000],
            links=links[:100],  # Limit links
            crawled_at=datetime.utcnow(),
            status_code=200,
            content_type="text/html",
            word_count=len(content.split()),
            language=lang[:10],
            domain=parsed_url.netloc,
            path=parsed_url.path,
            meta_keywords=meta_keywords[:20],
            headings=headings,
            images=images,
        )

    async def crawl_url(self, url: str) -> Optional[CrawlResult]:
        """Crawl a single URL"""
        domain = self._get_domain(url)

        # Check robots.txt
        if not await self._check_robots_txt(url):
            logger.info("URL blocked by robots.txt", url=url)
            return None

        # Apply rate limiting
        await self._apply_rate_limit(domain)

        try:
            # Rotate user agent
            headers = {"User-Agent": random.choice(self.config.user_agents)}

            async with self.session.get(url, headers=headers, allow_redirects=True) as response:
                # Check content type
                content_type = response.headers.get("Content-Type", "")
                if not any(ct in content_type for ct in self.config.allowed_content_types):
                    logger.debug("Skipping non-HTML content", url=url, content_type=content_type)
                    return None

                if response.status != 200:
                    logger.warning("Non-200 response", url=url, status=response.status)
                    return None

                html = await response.text()
                result = self._extract_content(html, str(response.url))
                result.status_code = response.status
                result.content_type = content_type

                # Update stats
                await self.redis.hincrby(self.STATS_KEY, "pages_crawled", 1)
                await self.redis.hincrby(self.STATS_KEY, "total_words", result.word_count)

                logger.info("Crawled page",
                           url=url,
                           title=result.title[:50],
                           words=result.word_count,
                           links=len(result.links))

                return result

        except asyncio.TimeoutError:
            logger.warning("Timeout crawling URL", url=url)
            await self.redis.hincrby(self.STATS_KEY, "timeouts", 1)
        except Exception as e:
            logger.error("Error crawling URL", url=url, error=str(e))
            await self.redis.hincrby(self.STATS_KEY, "errors", 1)

        return None

    async def crawl(
        self,
        seed_urls: List[str],
        max_pages: int = 100,
        callback=None
    ) -> List[CrawlResult]:
        """
        Crawl starting from seed URLs

        Args:
            seed_urls: Starting URLs
            max_pages: Maximum pages to crawl
            callback: Async function called with each CrawlResult

        Returns:
            List of CrawlResults
        """
        await self.initialize()

        try:
            # Add seed URLs to queue
            await self.add_to_queue(seed_urls, depth=0, priority=0)

            results = []
            pages_crawled = 0

            while pages_crawled < max_pages:
                # Get next URL
                next_item = await self.get_next_url()
                if not next_item:
                    logger.info("Queue empty, stopping crawl")
                    break

                url, depth = next_item

                # Check depth limit
                if depth > self.config.max_depth:
                    continue

                # Crawl the URL
                result = await self.crawl_url(url)

                if result:
                    # Mark as visited
                    await self.mark_visited(url)

                    results.append(result)
                    pages_crawled += 1

                    # Callback for real-time processing
                    if callback:
                        await callback(result)

                    # Add discovered links to queue
                    if depth < self.config.max_depth:
                        # Filter links based on config
                        if not self.config.follow_external_links:
                            domain = self._get_domain(url)
                            links = [l for l in result.links if self._get_domain(l) == domain]
                        else:
                            links = result.links

                        await self.add_to_queue(links, depth=depth + 1, priority=1)

            # Final stats
            stats = await self.redis.hgetall(self.STATS_KEY)
            logger.info("Crawl completed",
                       pages_crawled=pages_crawled,
                       stats=stats)

            return results

        finally:
            await self.shutdown()

    async def get_stats(self) -> Dict[str, Any]:
        """Get crawler statistics"""
        if not self.redis:
            self.redis = redis.from_url(self.redis_url, decode_responses=True)

        stats = await self.redis.hgetall(self.STATS_KEY)
        queue_size = await self.redis.zcard(self.QUEUE_KEY)
        visited_count = await self.redis.scard(self.VISITED_KEY)

        return {
            "queue_size": queue_size,
            "visited_count": visited_count,
            "pages_crawled": int(stats.get("pages_crawled", 0)),
            "total_words": int(stats.get("total_words", 0)),
            "errors": int(stats.get("errors", 0)),
            "timeouts": int(stats.get("timeouts", 0)),
        }

    async def clear_queue(self):
        """Clear the crawl queue and visited set"""
        if not self.redis:
            self.redis = redis.from_url(self.redis_url, decode_responses=True)

        await self.redis.delete(self.QUEUE_KEY)
        await self.redis.delete(self.VISITED_KEY)
        await self.redis.delete(self.DOMAIN_COUNT_KEY)
        await self.redis.delete(self.STATS_KEY)
        logger.info("Crawler queue cleared")

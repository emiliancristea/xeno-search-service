"""
Xeno Search Engine - Dynamic Smart Crawler

Finds exact pages with answers by:
1. Using site's internal search
2. Searching sitemaps for relevant URLs
3. Following and scoring relevant links
"""

import asyncio
import re
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse, quote_plus
from dataclasses import dataclass

import aiohttp
import structlog
from bs4 import BeautifulSoup

logger = structlog.get_logger(__name__)


@dataclass
class DiscoveredPage:
    """A discovered page with relevance info"""
    url: str
    title: str
    snippet: str
    relevance_score: float
    source: str  # "site_search", "sitemap", "link_follow"


# Common site search URL patterns
SITE_SEARCH_PATTERNS = [
    "/search?q={query}",
    "/search/?q={query}",
    "/search?s={query}",
    "/?s={query}",
    "/search/{query}",
    "/search?query={query}",
    "/search?search={query}",
    "/find?q={query}",
    "/results?q={query}",
    "/api/search?q={query}",
]

# Topic to domain mapping (expanded)
TOPIC_DOMAINS = {
    "technology": [
        "techcrunch.com", "theverge.com", "arstechnica.com",
        "wired.com", "engadget.com", "cnet.com", "zdnet.com"
    ],
    "apple": [
        "apple.com", "9to5mac.com", "macrumors.com", "appleinsider.com"
    ],
    "android": [
        "android.com", "9to5google.com", "androidcentral.com", "androidpolice.com"
    ],
    "programming": [
        "stackoverflow.com", "github.com", "dev.to", "medium.com",
        "docs.python.org", "developer.mozilla.org", "w3schools.com"
    ],
    "python": [
        "docs.python.org", "realpython.com", "python.org", "pypi.org"
    ],
    "javascript": [
        "developer.mozilla.org", "javascript.info", "nodejs.org", "npmjs.com"
    ],
    "ai": [
        "openai.com", "anthropic.com", "huggingface.co", "arxiv.org",
        "ai.google", "deepmind.com"
    ],
    "news": [
        "reuters.com", "bbc.com", "cnn.com", "apnews.com", "npr.org"
    ],
    "finance": [
        "bloomberg.com", "reuters.com", "wsj.com", "cnbc.com", "yahoo.com/finance"
    ],
    "science": [
        "nature.com", "science.org", "scientificamerican.com", "arxiv.org"
    ],
    "health": [
        "nih.gov", "webmd.com", "mayoclinic.org", "who.int", "cdc.gov"
    ],
    "gaming": [
        "ign.com", "gamespot.com", "kotaku.com", "polygon.com"
    ],
    "sports": [
        "espn.com", "sports.yahoo.com", "bleacherreport.com", "cbssports.com"
    ],
}

# Keywords for topic detection
TOPIC_KEYWORDS = {
    "apple": ["iphone", "ipad", "mac", "macbook", "apple", "ios", "macos", "airpods", "apple watch"],
    "android": ["android", "samsung", "pixel", "google phone", "oneplus", "xiaomi"],
    "programming": ["code", "programming", "developer", "software", "api", "function", "class", "debug"],
    "python": ["python", "pip", "django", "flask", "pandas", "numpy", "pytest"],
    "javascript": ["javascript", "js", "node", "npm", "react", "vue", "angular", "typescript"],
    "ai": ["ai", "artificial intelligence", "machine learning", "ml", "gpt", "llm", "neural", "deep learning", "chatgpt", "claude"],
    "technology": ["tech", "technology", "gadget", "device", "app", "software", "hardware"],
    "news": ["news", "breaking", "latest", "today", "yesterday", "announced", "report"],
    "finance": ["stock", "price", "market", "invest", "crypto", "bitcoin", "trading", "finance", "money"],
    "science": ["research", "study", "scientist", "experiment", "discovery", "physics", "chemistry", "biology"],
    "health": ["health", "medical", "doctor", "disease", "symptom", "treatment", "medicine", "drug"],
    "gaming": ["game", "gaming", "playstation", "xbox", "nintendo", "steam", "esports"],
    "sports": ["sports", "football", "basketball", "soccer", "nfl", "nba", "score", "team", "player"],
}


def detect_topics(query: str) -> List[str]:
    """Detect topics from a search query"""
    query_lower = query.lower()
    topics = []
    scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[topic] = score

    # Sort by score and return top topics
    sorted_topics = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    topics = [t[0] for t in sorted_topics[:3]]

    # Always include "technology" and "news" as fallbacks
    if not topics:
        topics = ["technology", "news"]

    return topics


def get_domains_for_topics(topics: List[str]) -> List[str]:
    """Get relevant domains for detected topics"""
    domains = []
    seen = set()

    for topic in topics:
        if topic in TOPIC_DOMAINS:
            for domain in TOPIC_DOMAINS[topic]:
                if domain not in seen:
                    domains.append(domain)
                    seen.add(domain)

    return domains[:10]  # Limit to 10 domains


def extract_keywords(query: str) -> List[str]:
    """Extract important keywords from query"""
    # Remove common stop words
    stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'to', 'do', 'does',
                  'can', 'could', 'would', 'should', 'why', 'when', 'where',
                  'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                  'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
                  'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our'}

    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    return keywords


def score_url_relevance(url: str, title: str, snippet: str, keywords: List[str]) -> float:
    """Score how relevant a URL is to the query keywords"""
    score = 0.0
    url_lower = url.lower()
    title_lower = (title or "").lower()
    snippet_lower = (snippet or "").lower()

    for keyword in keywords:
        # URL contains keyword (high weight)
        if keyword in url_lower:
            score += 3.0
        # Title contains keyword (high weight)
        if keyword in title_lower:
            score += 2.5
        # Snippet contains keyword
        if keyword in snippet_lower:
            score += 1.0

    # Normalize by number of keywords
    if keywords:
        score = score / len(keywords)

    return min(score, 10.0)  # Cap at 10


async def fetch_page(session: aiohttp.ClientSession, url: str, timeout: int = 10) -> Optional[str]:
    """Fetch a page's HTML content"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout), ssl=False) as resp:
            if resp.status == 200:
                return await resp.text()
    except Exception as e:
        logger.debug("Failed to fetch page", url=url, error=str(e))
    return None


async def try_site_search(session: aiohttp.ClientSession, domain: str, query: str) -> List[DiscoveredPage]:
    """Try to use a site's internal search to find relevant pages"""
    discovered = []
    encoded_query = quote_plus(query)
    keywords = extract_keywords(query)

    for pattern in SITE_SEARCH_PATTERNS[:5]:  # Try first 5 patterns
        search_url = f"https://{domain}{pattern.format(query=encoded_query)}"

        html = await fetch_page(session, search_url, timeout=8)
        if not html:
            continue

        soup = BeautifulSoup(html, 'html.parser')

        # Look for search result links
        links_found = []

        # Common search result patterns
        for selector in ['article a', '.search-result a', '.result a',
                        '.entry-title a', 'h2 a', 'h3 a', '.post-title a',
                        '[class*="result"] a', '[class*="search"] a']:
            for link in soup.select(selector)[:10]:
                href = link.get('href', '')
                if href and not href.startswith('#'):
                    full_url = urljoin(search_url, href)
                    # Skip if it's the same domain search page
                    if '/search' not in full_url and domain in full_url:
                        title = link.get_text(strip=True)
                        # Get snippet from parent
                        parent = link.find_parent(['article', 'div', 'li'])
                        snippet = ""
                        if parent:
                            snippet = parent.get_text(strip=True)[:200]

                        links_found.append({
                            'url': full_url,
                            'title': title,
                            'snippet': snippet
                        })

        if links_found:
            logger.info("Found results via site search", domain=domain, count=len(links_found))
            for link in links_found[:5]:  # Top 5 from this search
                score = score_url_relevance(link['url'], link['title'], link['snippet'], keywords)
                if score > 0.5:  # Minimum relevance threshold
                    discovered.append(DiscoveredPage(
                        url=link['url'],
                        title=link['title'],
                        snippet=link['snippet'],
                        relevance_score=score,
                        source="site_search"
                    ))
            break  # Found results, stop trying patterns

    return discovered


async def search_sitemap(session: aiohttp.ClientSession, domain: str, keywords: List[str]) -> List[DiscoveredPage]:
    """Search a site's sitemap for relevant URLs"""
    discovered = []
    sitemap_urls = [
        f"https://{domain}/sitemap.xml",
        f"https://{domain}/sitemap_index.xml",
        f"https://www.{domain}/sitemap.xml",
    ]

    for sitemap_url in sitemap_urls:
        xml = await fetch_page(session, sitemap_url, timeout=10)
        if not xml:
            continue

        # Extract URLs from sitemap
        urls = re.findall(r'<loc>(https?://[^<]+)</loc>', xml)

        # Score and filter URLs by keyword relevance
        scored_urls = []
        for url in urls[:500]:  # Check first 500 URLs
            score = 0
            url_lower = url.lower()
            for kw in keywords:
                if kw in url_lower:
                    score += 2
            if score > 0:
                scored_urls.append((url, score))

        # Sort by score and take top results
        scored_urls.sort(key=lambda x: x[1], reverse=True)

        for url, score in scored_urls[:5]:
            discovered.append(DiscoveredPage(
                url=url,
                title="",  # Will be filled when crawled
                snippet="",
                relevance_score=score,
                source="sitemap"
            ))

        if discovered:
            logger.info("Found URLs via sitemap", domain=domain, count=len(discovered))
            break

    return discovered


async def crawl_and_extract(session: aiohttp.ClientSession, url: str, keywords: List[str]) -> Optional[Dict]:
    """Crawl a page and extract its content"""
    html = await fetch_page(session, url, timeout=15)
    if not html:
        return None

    soup = BeautifulSoup(html, 'html.parser')

    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()

    # Get title
    title = ""
    if soup.title:
        title = soup.title.get_text(strip=True)
    elif soup.find('h1'):
        title = soup.find('h1').get_text(strip=True)

    # Get main content
    content = ""
    main = soup.find('main') or soup.find('article') or soup.find('[role="main"]') or soup.body
    if main:
        content = main.get_text(separator=' ', strip=True)

    # Get meta description
    description = ""
    meta_desc = soup.find('meta', {'name': 'description'})
    if meta_desc:
        description = meta_desc.get('content', '')

    # Score relevance
    relevance = score_url_relevance(url, title, content[:500], keywords)

    return {
        'url': url,
        'title': title,
        'description': description,
        'content': content[:10000],  # Limit content size
        'relevance_score': relevance,
        'word_count': len(content.split())
    }


async def dynamic_crawl_for_query(query: str, max_pages: int = 10) -> List[Dict]:
    """
    Main function: Dynamically crawl to find pages that answer a query

    1. Detect topics from query
    2. Get relevant domains
    3. Try site search on each domain
    4. Search sitemaps for relevant URLs
    5. Crawl discovered pages
    6. Return ranked results
    """
    logger.info("Starting dynamic crawl", query=query)

    # Step 1: Detect topics and get domains
    topics = detect_topics(query)
    domains = get_domains_for_topics(topics)
    keywords = extract_keywords(query)

    logger.info("Detected topics and domains",
               topics=topics,
               domains=domains[:5],
               keywords=keywords)

    discovered_pages: List[DiscoveredPage] = []

    async with aiohttp.ClientSession() as session:
        # Step 2: Try site search on each domain (parallel)
        search_tasks = [try_site_search(session, domain, query) for domain in domains[:5]]
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        for result in search_results:
            if isinstance(result, list):
                discovered_pages.extend(result)

        # Step 3: If not enough results, try sitemaps
        if len(discovered_pages) < 5:
            sitemap_tasks = [search_sitemap(session, domain, keywords) for domain in domains[:5]]
            sitemap_results = await asyncio.gather(*sitemap_tasks, return_exceptions=True)

            for result in sitemap_results:
                if isinstance(result, list):
                    discovered_pages.extend(result)

        # Step 4: Deduplicate and sort by relevance
        seen_urls = set()
        unique_pages = []
        for page in discovered_pages:
            if page.url not in seen_urls:
                seen_urls.add(page.url)
                unique_pages.append(page)

        unique_pages.sort(key=lambda x: x.relevance_score, reverse=True)
        top_pages = unique_pages[:max_pages]

        logger.info("Discovered pages to crawl", count=len(top_pages))

        # Step 5: Crawl the discovered pages
        crawl_tasks = [crawl_and_extract(session, page.url, keywords) for page in top_pages]
        crawl_results = await asyncio.gather(*crawl_tasks, return_exceptions=True)

        # Step 6: Filter and return successful crawls
        final_results = []
        for result in crawl_results:
            if isinstance(result, dict) and result.get('content'):
                final_results.append(result)

        # Sort by relevance
        final_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        logger.info("Dynamic crawl complete",
                   query=query,
                   pages_crawled=len(final_results))

        return final_results

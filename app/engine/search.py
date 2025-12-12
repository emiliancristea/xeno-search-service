"""
Xeno Search Engine - Search API
Query your own search index built from crawled pages
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import meilisearch
from meilisearch.errors import MeilisearchApiError
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    """A single search result"""
    url: str
    title: str
    description: str
    snippet: str
    domain: str
    score: float
    page_rank: float
    word_count: int
    crawled_at: str
    highlights: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "snippet": self.snippet,
            "domain": self.domain,
            "score": self.score,
            "page_rank": self.page_rank,
            "word_count": self.word_count,
            "crawled_at": self.crawled_at,
            "highlights": self.highlights,
        }


@dataclass
class SearchResponse:
    """Response from a search query"""
    query: str
    results: List[SearchResult]
    total_hits: int
    processing_time_ms: int
    page: int
    per_page: int
    has_more: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_hits": self.total_hits,
            "processing_time_ms": self.processing_time_ms,
            "page": self.page,
            "per_page": self.per_page,
            "has_more": self.has_more,
        }


@dataclass
class SearchConfig:
    """Configuration for the search engine"""
    host: str = "http://localhost:7700"
    api_key: str = "xeno_search_master_key_change_me"
    index_name: str = "xeno_pages"
    default_per_page: int = 10
    max_per_page: int = 50
    highlight_pre_tag: str = "<mark>"
    highlight_post_tag: str = "</mark>"


class XenoSearchEngine:
    """
    Xeno Search Engine - Query Interface

    Features:
    - Full-text search with typo tolerance
    - Filtering by domain, language, date
    - Highlighted snippets
    - Pagination
    - Relevance ranking with PageRank boost
    """

    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        self.client = meilisearch.Client(self.config.host, self.config.api_key)
        self.index = self.client.index(self.config.index_name)

    def _extract_snippet(self, hit: Dict, formatted: Dict, max_length: int = 200) -> str:
        """Extract a relevant snippet from the search result"""
        # Try to use highlighted content first
        if "_formatted" in hit:
            formatted = hit["_formatted"]

            # Prefer content with highlights
            if "content" in formatted and self.config.highlight_pre_tag in formatted["content"]:
                content = formatted["content"]
                # Find position of first highlight
                pos = content.find(self.config.highlight_pre_tag)
                start = max(0, pos - 50)
                end = min(len(content), pos + max_length)
                snippet = content[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(content):
                    snippet = snippet + "..."
                return snippet

        # Fall back to description or content start
        if hit.get("description"):
            return hit["description"][:max_length]

        if hit.get("content"):
            return hit["content"][:max_length] + "..."

        return ""

    async def search(
        self,
        query: str,
        page: int = 1,
        per_page: int = 10,
        domain: Optional[str] = None,
        language: Optional[str] = None,
        min_word_count: Optional[int] = None,
        sort_by: Optional[str] = None,
    ) -> SearchResponse:
        """
        Search the index

        Args:
            query: Search query
            page: Page number (1-indexed)
            per_page: Results per page
            domain: Filter by domain
            language: Filter by language
            min_word_count: Minimum word count filter
            sort_by: Sort field (e.g., "crawled_at:desc")

        Returns:
            SearchResponse with results
        """
        per_page = min(per_page, self.config.max_per_page)
        offset = (page - 1) * per_page

        # Build search options
        search_params = {
            "offset": offset,
            "limit": per_page,
            "attributesToHighlight": ["title", "content", "description"],
            "highlightPreTag": self.config.highlight_pre_tag,
            "highlightPostTag": self.config.highlight_post_tag,
            "attributesToRetrieve": [
                "url", "title", "description", "content", "domain",
                "language", "word_count", "page_rank", "crawled_at"
            ],
        }

        # Build filters
        filters = []
        if domain:
            filters.append(f"domain = '{domain}'")
        if language:
            filters.append(f"language = '{language}'")
        if min_word_count:
            filters.append(f"word_count >= {min_word_count}")

        if filters:
            search_params["filter"] = " AND ".join(filters)

        # Add sorting
        if sort_by:
            search_params["sort"] = [sort_by]

        try:
            # Execute search
            start_time = datetime.now()
            response = self.index.search(query, search_params)
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

            # Process results
            results = []
            for hit in response.get("hits", []):
                formatted = hit.get("_formatted", {})

                result = SearchResult(
                    url=hit.get("url", ""),
                    title=formatted.get("title", hit.get("title", "")),
                    description=hit.get("description", ""),
                    snippet=self._extract_snippet(hit, formatted),
                    domain=hit.get("domain", ""),
                    score=hit.get("_score", 0),
                    page_rank=hit.get("page_rank", 1.0),
                    word_count=hit.get("word_count", 0),
                    crawled_at=hit.get("crawled_at", ""),
                    highlights={
                        "title": [formatted.get("title", "")] if formatted.get("title") else [],
                        "content": [formatted.get("content", "")[:500]] if formatted.get("content") else [],
                    }
                )
                results.append(result)

            total_hits = response.get("estimatedTotalHits", len(results))
            has_more = offset + len(results) < total_hits

            logger.info("Search completed",
                       query=query,
                       results=len(results),
                       total_hits=total_hits,
                       processing_time_ms=processing_time)

            return SearchResponse(
                query=query,
                results=results,
                total_hits=total_hits,
                processing_time_ms=processing_time,
                page=page,
                per_page=per_page,
                has_more=has_more,
            )

        except MeilisearchApiError as e:
            logger.error("Meilisearch error", query=query, error=str(e))
            return SearchResponse(
                query=query,
                results=[],
                total_hits=0,
                processing_time_ms=0,
                page=page,
                per_page=per_page,
                has_more=False,
            )

    async def suggest(self, query: str, limit: int = 5) -> List[str]:
        """
        Get search suggestions based on indexed content

        Args:
            query: Partial query
            limit: Maximum suggestions

        Returns:
            List of suggested queries
        """
        try:
            # Use search to find matching titles/content
            response = self.index.search(query, {
                "limit": limit,
                "attributesToRetrieve": ["title"],
            })

            suggestions = []
            for hit in response.get("hits", []):
                title = hit.get("title", "")
                if title and title.lower() != query.lower():
                    suggestions.append(title)

            return suggestions[:limit]

        except Exception as e:
            logger.error("Suggestion error", query=query, error=str(e))
            return []

    async def get_domains(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get list of indexed domains with document counts"""
        try:
            # Get facet distribution
            response = self.index.search("", {
                "limit": 0,
                "facets": ["domain"],
            })

            facets = response.get("facetDistribution", {}).get("domain", {})

            domains = [
                {"domain": domain, "count": count}
                for domain, count in sorted(
                    facets.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:limit]
            ]

            return domains

        except Exception as e:
            logger.error("Get domains error", error=str(e))
            return []

    async def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        try:
            stats = self.index.get_stats()
            return {
                "total_documents": stats.number_of_documents,
                "is_indexing": stats.is_indexing,
                "index_name": self.config.index_name,
            }
        except Exception as e:
            logger.error("Get stats error", error=str(e))
            return {"error": str(e)}

    async def health_check(self) -> bool:
        """Check if search engine is healthy"""
        try:
            self.client.health()
            return True
        except Exception:
            return False


# Convenience function for quick searches
async def quick_search(query: str, num_results: int = 10) -> List[Dict[str, Any]]:
    """
    Quick search function for integration with existing code

    Args:
        query: Search query
        num_results: Number of results

    Returns:
        List of search results as dictionaries
    """
    engine = XenoSearchEngine()
    response = await engine.search(query, per_page=num_results)

    return [
        {
            "url": r.url,
            "title": r.title,
            "snippet": r.snippet or r.description,
            "domain": r.domain,
        }
        for r in response.results
    ]

"""
Xeno Search Engine - Indexer
Indexes crawled content into Meilisearch for fast full-text search
"""

import hashlib
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import meilisearch
from meilisearch.errors import MeilisearchApiError
import structlog

from .crawler import CrawlResult

logger = structlog.get_logger(__name__)


@dataclass
class IndexConfig:
    """Configuration for the indexer"""
    host: str = "http://localhost:7700"
    api_key: str = "xeno_search_master_key_change_me"
    index_name: str = "xeno_pages"
    batch_size: int = 100


class XenoIndexer:
    """
    Xeno Search Engine Indexer

    Features:
    - Index crawled pages to Meilisearch
    - Configurable ranking rules
    - Batch indexing support
    - Document updates and deletions
    """

    def __init__(self, config: Optional[IndexConfig] = None):
        self.config = config or IndexConfig()
        self.client = meilisearch.Client(self.config.host, self.config.api_key)
        self.index = None

    async def initialize(self):
        """Initialize the index with proper settings"""
        try:
            # Create index if it doesn't exist
            try:
                self.client.create_index(
                    self.config.index_name,
                    {"primaryKey": "id"}
                )
                logger.info("Created new index", index=self.config.index_name)
            except MeilisearchApiError as e:
                if "index_already_exists" not in str(e):
                    raise
                logger.info("Index already exists", index=self.config.index_name)

            self.index = self.client.index(self.config.index_name)

            # Configure searchable attributes
            self.index.update_searchable_attributes([
                "title",
                "description",
                "content",
                "headings_text",
                "domain",
                "meta_keywords",
            ])

            # Configure filterable attributes
            self.index.update_filterable_attributes([
                "domain",
                "language",
                "crawled_at",
                "word_count",
            ])

            # Configure sortable attributes
            self.index.update_sortable_attributes([
                "crawled_at",
                "word_count",
                "page_rank",
            ])

            # Configure ranking rules
            self.index.update_ranking_rules([
                "words",
                "typo",
                "proximity",
                "attribute",
                "sort",
                "exactness",
                "page_rank:desc",
                "word_count:desc",
            ])

            # Configure typo tolerance
            self.index.update_typo_tolerance({
                "enabled": True,
                "minWordSizeForTypos": {
                    "oneTypo": 4,
                    "twoTypos": 8
                }
            })

            logger.info("Index configured successfully", index=self.config.index_name)

        except Exception as e:
            logger.error("Failed to initialize index", error=str(e))
            raise

    def _generate_document_id(self, url: str) -> str:
        """Generate unique document ID from URL"""
        return hashlib.md5(url.lower().encode()).hexdigest()

    def _crawl_result_to_document(self, result: CrawlResult, page_rank: float = 1.0) -> Dict[str, Any]:
        """Convert CrawlResult to Meilisearch document"""
        # Combine headings into searchable text
        headings_text = " ".join(
            result.headings.get("h1", []) +
            result.headings.get("h2", []) +
            result.headings.get("h3", [])
        )

        return {
            "id": self._generate_document_id(result.url),
            "url": result.url,
            "title": result.title,
            "description": result.description,
            "content": result.content[:30000],  # Limit content for indexing
            "headings_text": headings_text,
            "domain": result.domain,
            "path": result.path,
            "language": result.language,
            "word_count": result.word_count,
            "meta_keywords": result.meta_keywords,
            "crawled_at": result.crawled_at.isoformat(),
            "page_rank": page_rank,
            "links_count": len(result.links),
        }

    async def index_page(self, result: CrawlResult, page_rank: float = 1.0) -> bool:
        """Index a single crawled page"""
        try:
            document = self._crawl_result_to_document(result, page_rank)
            task = self.index.add_documents([document])

            logger.debug("Indexed page",
                        url=result.url,
                        title=result.title[:50],
                        task_uid=task.task_uid)
            return True

        except Exception as e:
            logger.error("Failed to index page", url=result.url, error=str(e))
            return False

    async def index_pages(self, results: List[CrawlResult], page_ranks: Optional[Dict[str, float]] = None) -> int:
        """Index multiple pages in batches"""
        if not results:
            return 0

        page_ranks = page_ranks or {}
        indexed = 0

        # Process in batches
        for i in range(0, len(results), self.config.batch_size):
            batch = results[i:i + self.config.batch_size]
            documents = []

            for result in batch:
                rank = page_ranks.get(result.url, 1.0)
                doc = self._crawl_result_to_document(result, rank)
                documents.append(doc)

            try:
                task = self.index.add_documents(documents)
                indexed += len(documents)

                logger.info("Indexed batch",
                           batch_size=len(documents),
                           total_indexed=indexed,
                           task_uid=task.task_uid)

            except Exception as e:
                logger.error("Failed to index batch", error=str(e))

        return indexed

    async def delete_page(self, url: str) -> bool:
        """Delete a page from the index"""
        try:
            doc_id = self._generate_document_id(url)
            self.index.delete_document(doc_id)
            logger.info("Deleted page from index", url=url)
            return True
        except Exception as e:
            logger.error("Failed to delete page", url=url, error=str(e))
            return False

    async def delete_domain(self, domain: str) -> int:
        """Delete all pages from a domain"""
        try:
            # Use filter to delete
            task = self.index.delete_documents({
                "filter": f"domain = '{domain}'"
            })
            logger.info("Deleted domain from index", domain=domain, task_uid=task.task_uid)
            return task.task_uid
        except Exception as e:
            logger.error("Failed to delete domain", domain=domain, error=str(e))
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            stats = self.index.get_stats()
            return {
                "number_of_documents": stats.number_of_documents,
                "is_indexing": stats.is_indexing,
                "field_distribution": stats.field_distribution,
            }
        except Exception as e:
            logger.error("Failed to get stats", error=str(e))
            return {}

    async def clear_index(self):
        """Clear all documents from the index"""
        try:
            task = self.index.delete_all_documents()
            logger.info("Cleared index", task_uid=task.task_uid)
            return task.task_uid
        except Exception as e:
            logger.error("Failed to clear index", error=str(e))
            return None


class PageRankCalculator:
    """
    Simple PageRank implementation for ranking pages

    Based on link analysis - pages with more inbound links rank higher
    """

    def __init__(self, damping_factor: float = 0.85, iterations: int = 20):
        self.damping_factor = damping_factor
        self.iterations = iterations

    def calculate(self, pages: List[CrawlResult]) -> Dict[str, float]:
        """
        Calculate PageRank scores for crawled pages

        Args:
            pages: List of crawled pages with links

        Returns:
            Dictionary mapping URL to PageRank score
        """
        if not pages:
            return {}

        # Build URL to index mapping
        url_to_idx = {page.url: i for i, page in enumerate(pages)}
        n = len(pages)

        # Initialize scores
        scores = {page.url: 1.0 / n for page in pages}

        # Build link graph
        outbound_links = {}
        inbound_links = {page.url: [] for page in pages}

        for page in pages:
            # Filter to only include links to pages we've crawled
            valid_links = [l for l in page.links if l in url_to_idx]
            outbound_links[page.url] = valid_links

            for link in valid_links:
                inbound_links[link].append(page.url)

        # Iterate
        for iteration in range(self.iterations):
            new_scores = {}

            for page in pages:
                url = page.url

                # Calculate new score
                rank_sum = 0
                for inbound_url in inbound_links[url]:
                    out_count = len(outbound_links[inbound_url])
                    if out_count > 0:
                        rank_sum += scores[inbound_url] / out_count

                new_score = (1 - self.damping_factor) / n + self.damping_factor * rank_sum
                new_scores[url] = new_score

            scores = new_scores

        # Normalize scores to 0-10 range
        max_score = max(scores.values()) if scores else 1
        min_score = min(scores.values()) if scores else 0
        score_range = max_score - min_score if max_score != min_score else 1

        normalized = {
            url: 1 + 9 * (score - min_score) / score_range
            for url, score in scores.items()
        }

        logger.info("PageRank calculated",
                   pages=len(pages),
                   iterations=self.iterations,
                   max_score=max(normalized.values()),
                   min_score=min(normalized.values()))

        return normalized

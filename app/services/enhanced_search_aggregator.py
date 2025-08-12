import asyncio
import random
import aiohttp
import httpx
import json
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass
from urllib.parse import quote_plus, urljoin, urlparse, parse_qsl, urlunparse, urlencode
from bs4 import BeautifulSoup
import structlog
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import get_config
from app.core.monitoring import track_operation, get_monitor
from app.core.cache import get_cache_manager
from app.models.search_models import Source

logger = structlog.get_logger(__name__)


@dataclass
class SearchEngineResult:
    """Standardized search engine result"""
    title: str
    url: str
    snippet: str
    engine: str
    rank: int
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SearchEngineBase:
    """Base class for search engines"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.timeout = config.get('timeout', 30)
        self.enabled = config.get('enabled', True)
        self.weight = config.get('weight', 1.0)  # For result ranking
        
    async def search(self, query: str, num_results: int = 10) -> List[SearchEngineResult]:
        """Perform search - must be implemented by subclasses"""
        raise NotImplementedError
        
    def is_enabled(self) -> bool:
        """Check if this search engine is enabled"""
        return self.enabled


class SearXNGEngine(SearchEngineBase):
    """SearXNG search engine implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("searxng", config)
        self.instances = config.get('instances', [])
        self.current_instance_index = 0
        
    def _get_next_instance(self) -> str:
        """Get next SearXNG instance (round-robin)"""
        if not self.instances:
            raise ValueError("No SearXNG instances configured")
            
        instance = self.instances[self.current_instance_index]
        self.current_instance_index = (self.current_instance_index + 1) % len(self.instances)
        return instance
    
    @track_operation("searxng_search")
    async def search(self, query: str, num_results: int = 10) -> List[SearchEngineResult]:
        """Search using SearXNG"""
        results = []
        max_retries = min(3, len(self.instances))
        
        for attempt in range(max_retries):
            try:
                instance = self._get_next_instance()
                search_url = f"{instance}/search"
                
                params = {
                    'q': query,
                    'format': 'json',
                    'engines': 'google,bing,duckduckgo,startpage',
                    'categories': 'general',
                    # language and time_range can be overridden at call-site via query formatting if needed
                    'language': self.config.get('language', 'en'),
                    'time_range': self.config.get('time_range', ''),
                    'safesearch': '1'
                }
                
                headers = {
                    'User-Agent': 'Xeno-Search-Service/2.0',
                    'Accept': 'application/json',
                    'Accept-Language': 'en-US,en;q=0.9'
                }
                
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    logger.info("Querying SearXNG", instance=instance, query=query)
                    
                    response = await client.get(search_url, params=params, headers=headers)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    if 'results' not in data:
                        logger.warning("No results in SearXNG response", instance=instance)
                        continue
                    
                    for i, result in enumerate(data['results'][:num_results]):
                        if 'url' not in result or 'title' not in result:
                            continue
                            
                        search_result = SearchEngineResult(
                            title=result.get('title', '').strip(),
                            url=result.get('url', '').strip(),
                            snippet=result.get('content', '').strip(),
                            engine=self.name,
                            rank=i + 1,
                            confidence_score=0.8,  # SearXNG aggregates multiple engines
                            metadata={
                                'engine_list': result.get('engines', []),
                                'score': result.get('score', 0) if 'score' in result else None,
                                'instance': instance
                            }
                        )
                        results.append(search_result)
                    
                    logger.info("SearXNG search successful", 
                              instance=instance, 
                              results_count=len(results))
                    return results
                    
            except Exception as e:
                logger.warning("SearXNG search failed", 
                             instance=instance if 'instance' in locals() else "unknown",
                             error=str(e), 
                             attempt=attempt + 1)
                if attempt == max_retries - 1:
                    raise
                
        return results


class DuckDuckGoEngine(SearchEngineBase):
    """DuckDuckGo search engine implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("duckduckgo", config)
        self.use_html = config.get('use_html', True)
        
    @track_operation("duckduckgo_search")
    async def search(self, query: str, num_results: int = 10) -> List[SearchEngineResult]:
        """Search using DuckDuckGo"""
        if self.use_html:
            return await self._search_html(query, num_results)
        else:
            return await self._search_api(query, num_results)
    
    async def _search_html(self, query: str, num_results: int) -> List[SearchEngineResult]:
        """Search using DuckDuckGo HTML interface"""
        results = []
        search_url = f"https://html.duckduckgo.com/html/"
        
        params = {'q': query}
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info("Querying DuckDuckGo HTML", query=query)
                
                response = await client.get(search_url, params=params, headers=headers)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find search result containers
                result_containers = soup.find_all('div', class_='web-result') or \
                                  soup.find_all('div', class_='result')
                
                if not result_containers:
                    # Try alternative selectors
                    result_containers = soup.select('div.result, table tr')
                
                for i, container in enumerate(result_containers[:num_results]):
                    try:
                        # Extract title and URL
                        title_link = container.find('a', class_='result__a') or \
                                   container.find('a', href=True)
                        
                        if not title_link:
                            continue
                            
                        title = title_link.get_text(strip=True)
                        url = title_link.get('href', '')
                        
                        # Handle DuckDuckGo redirects
                        if url.startswith('/l/?uddg='):
                            from urllib.parse import parse_qs, urlparse
                            parsed = urlparse(url)
                            qs = parse_qs(parsed.query)
                            if 'uddg' in qs:
                                url = qs['uddg'][0]
                        
                        # Extract snippet
                        snippet_element = container.find(class_='result__snippet') or \
                                        container.find('td', class_='result-snippet')
                        snippet = snippet_element.get_text(strip=True) if snippet_element else ""
                        
                        if title and url and url.startswith('http'):
                            search_result = SearchEngineResult(
                                title=title,
                                url=url,
                                snippet=snippet,
                                engine=self.name,
                                rank=i + 1,
                                confidence_score=0.7,
                                metadata={'method': 'html'}
                            )
                            results.append(search_result)
                            
                    except Exception as e:
                        logger.warning("Error parsing DuckDuckGo result", error=str(e))
                        continue
                
                logger.info("DuckDuckGo HTML search successful", results_count=len(results))
                return results
                
        except Exception as e:
            logger.error("DuckDuckGo HTML search failed", error=str(e))
            raise
    
    async def _search_api(self, query: str, num_results: int) -> List[SearchEngineResult]:
        """Search using DuckDuckGo Instant Answer API (limited results)"""
        # Note: DDG's Instant Answer API doesn't provide web search results
        # This is kept for potential future use or alternative endpoints
        results = []
        api_url = "https://api.duckduckgo.com/"
        
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(api_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                # DDG API mainly provides instant answers, not web results
                # This would need to be adapted based on available endpoints
                
        except Exception as e:
            logger.warning("DuckDuckGo API search failed", error=str(e))
            
        return results


class BraveSearchEngine(SearchEngineBase):
    """Brave Search API implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("brave", config)
        self.api_key = config.get('api_key')
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        
    def is_enabled(self) -> bool:
        """Brave Search requires API key"""
        return self.enabled and bool(self.api_key)
    
    @track_operation("brave_search")
    async def search(self, query: str, num_results: int = 10) -> List[SearchEngineResult]:
        """Search using Brave Search API"""
        if not self.api_key:
            logger.warning("Brave Search API key not configured")
            return []
            
        results = []
        
        params = {
            'q': query,
            'count': min(num_results, 20),  # Brave API limit
            'offset': 0,
            'country': self.config.get('country', 'US'),
            'search_lang': self.config.get('language', 'en'),
            'ui_lang': self.config.get('ui_lang', 'en-US'),
            'text_decorations': False,
            'result_filter': 'web'
        }
        
        headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': self.api_key
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info("Querying Brave Search API", query=query)
                
                response = await client.get(self.base_url, params=params, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                
                if 'web' not in data or 'results' not in data['web']:
                    logger.warning("No web results in Brave Search response")
                    return results
                
                for i, result in enumerate(data['web']['results'][:num_results]):
                    search_result = SearchEngineResult(
                        title=result.get('title', '').strip(),
                        url=result.get('url', '').strip(),
                        snippet=result.get('description', '').strip(),
                        engine=self.name,
                        rank=i + 1,
                        confidence_score=0.9,  # Brave has good quality results
                        metadata={
                            'age': result.get('age'),
                            'language': result.get('language'),
                            'location': result.get('location')
                        }
                    )
                    results.append(search_result)
                
                logger.info("Brave Search successful", results_count=len(results))
                return results
                
        except Exception as e:
            logger.error("Brave Search failed", error=str(e))
            raise


class EnhancedSearchAggregator:
    """Enhanced search aggregator with multiple engines and intelligent ranking"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.engines: Dict[str, SearchEngineBase] = {}
        self.semantic_model: Optional[SentenceTransformer] = None
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.monitor = get_monitor()
        self.cross_encoder: Optional[Any] = None
        
        # Initialize search engines
        self._initialize_engines()
        
        # Initialize semantic model if enabled
        if self.config.semantic_search_enabled:
            self._initialize_semantic_model()
        # Cross-encoder is optional
        if getattr(self.config, 'enable_cross_encoder', False):
            self._initialize_cross_encoder()
    
    def _initialize_engines(self):
        """Initialize all configured search engines"""
        search_config = self.config.get_search_engine_config()
        
        # SearXNG
        if "searxng" in search_config['engines']:
            self.engines["searxng"] = SearXNGEngine({
                'instances': search_config['searxng_instances'],
                'timeout': search_config['timeout'],
                'enabled': True,
                'weight': 1.0
            })
        
        # DuckDuckGo
        if "duckduckgo" in search_config['engines'] and search_config['duckduckgo_enabled']:
            self.engines["duckduckgo"] = DuckDuckGoEngine({
                'timeout': search_config['timeout'],
                'enabled': True,
                'weight': 0.8,
                'use_html': True
            })
        
        # Brave Search
        if "brave" in search_config['engines'] and search_config['brave_api_key']:
            self.engines["brave"] = BraveSearchEngine({
                'api_key': search_config['brave_api_key'],
                'timeout': search_config['timeout'],
                'enabled': True,
                'weight': 1.2
            })
        
        enabled_engines = [name for name, engine in self.engines.items() if engine.is_enabled()]
        logger.info("Search engines initialized", engines=enabled_engines)
    
    def _initialize_semantic_model(self):
        """Initialize semantic similarity model"""
        try:
            model_name = self.config.embedding_model
            logger.info("Loading semantic model", model=model_name)
            self.semantic_model = SentenceTransformer(model_name)
            logger.info("Semantic model loaded successfully")
        except Exception as e:
            logger.error("Failed to load semantic model", error=str(e))
            self.semantic_model = None
    
    def _initialize_cross_encoder(self):
        """Initialize optional cross-encoder for reranking"""
        try:
            from sentence_transformers import CrossEncoder
            model_name = getattr(self.config, 'cross_encoder_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Loading cross-encoder model", model=model_name)
            self.cross_encoder = CrossEncoder(model_name)
            logger.info("Cross-encoder model loaded successfully")
        except Exception as e:
            logger.warning("Failed to load cross-encoder model", error=str(e))
            self.cross_encoder = None
    
    async def search_all_engines(self, query: str, num_results_per_engine: int = 15, language: Optional[str] = None, time_range: Optional[str] = None) -> Dict[str, List[SearchEngineResult]]:
        """Search all enabled engines concurrently"""
        tasks = []
        engine_names = []
        
        for name, engine in self.engines.items():
            if engine.is_enabled():
                # Thread optional language/time_range to engine-local config for this call
                engine.config['language'] = language or getattr(self.config, 'language', 'en')
                engine.config['time_range'] = time_range or getattr(self.config, 'time_range', '')
                tasks.append(engine.search(query, num_results_per_engine))
                engine_names.append(name)
        
        if not tasks:
            logger.warning("No search engines enabled")
            return {}
        
        logger.info("Searching all engines", engines=engine_names, query=query)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            engine_results = {}
            for i, result in enumerate(results):
                engine_name = engine_names[i]
                if isinstance(result, Exception):
                    logger.error("Search engine failed", engine=engine_name, error=str(result))
                    engine_results[engine_name] = []
                    self.monitor.record_error(f"search_engine_{engine_name}", str(result))
                else:
                    engine_results[engine_name] = result
                    logger.info("Search engine completed", 
                              engine=engine_name, 
                              results_count=len(result))
            
            return engine_results
            
        except Exception as e:
            logger.error("Error in concurrent search", error=str(e))
            raise
    
    def _deduplicate_results(self, all_results: List[SearchEngineResult], 
                           similarity_threshold: float = 0.8) -> List[SearchEngineResult]:
        """Remove duplicate results based on URL and content similarity"""
        if not all_results:
            return []
        
        # Step 1: Remove exact URL duplicates
        seen_urls: Set[str] = set()
        url_deduped = []
        
        for result in all_results:
            normalized_url = self._normalize_url(result.url)
            if normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                url_deduped.append(result)
        
        # Step 2: Remove content similarity duplicates
        if len(url_deduped) <= 1 or not self.config.content_deduplication:
            return url_deduped
        
        try:
            # Use title + snippet for similarity comparison
            contents = [f"{result.title} {result.snippet}" for result in url_deduped]
            
            # Calculate TF-IDF similarity
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(contents)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Keep track of which results to keep
            keep_indices = set(range(len(url_deduped)))
            
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    if similarity_matrix[i][j] > similarity_threshold:
                        # Remove the one with lower confidence score
                        if url_deduped[i].confidence_score < url_deduped[j].confidence_score:
                            keep_indices.discard(i)
                        else:
                            keep_indices.discard(j)
            
            content_deduped = [url_deduped[i] for i in sorted(keep_indices)]
            
            logger.info("Deduplication completed", 
                       original_count=len(all_results),
                       url_deduped_count=len(url_deduped),
                       final_count=len(content_deduped))
            
            return content_deduped
            
        except Exception as e:
            logger.warning("Content deduplication failed", error=str(e))
            return url_deduped
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication"""
        try:
            parsed = urlparse(url.lower())
            # Remove tracking parameters while preserving canonical ones
            tracking_prefixes = ("utm_", "fbclid", "gclid", "ref", "ref_src")
            query_params = [(k, v) for k, v in parse_qsl(parsed.query) if not k.startswith(tracking_prefixes)]
            clean_query = urlencode(query_params)
            clean = parsed._replace(query=clean_query, fragment="")
            clean_url = urlunparse(clean)
            # Remove trailing slash
            return clean_url.rstrip('/')
        except:
            return url.lower()
    
    def _calculate_semantic_relevance(self, query: str, results: List[SearchEngineResult]) -> List[SearchEngineResult]:
        """Calculate semantic relevance scores using sentence transformers"""
        if not self.semantic_model or not results:
            return results
        
        try:
            # Encode query
            query_embedding = self.semantic_model.encode([query])
            
            # Encode result contents (title + snippet)
            result_texts = [f"{result.title} {result.snippet}" for result in results]
            result_embeddings = self.semantic_model.encode(result_texts)
            
            # Calculate semantic similarities
            similarities = cosine_similarity(query_embedding, result_embeddings)[0]
            
            # Update confidence scores with semantic relevance
            for i, result in enumerate(results):
                semantic_score = similarities[i]
                # Combine original confidence with semantic relevance
                result.confidence_score = (result.confidence_score * 0.6) + (semantic_score * 0.4)
                result.metadata['semantic_score'] = float(semantic_score)
            
            logger.info("Semantic relevance calculated", results_count=len(results))
            return results
            
        except Exception as e:
            logger.warning("Semantic relevance calculation failed", error=str(e))
            return results
    
    def _rank_and_merge_results(self, engine_results: Dict[str, List[SearchEngineResult]], 
                               target_count: int) -> List[SearchEngineResult]:
        """Intelligently rank and merge results from multiple engines"""
        all_results = []
        
        # Collect all results with engine weights
        for engine_name, results in engine_results.items():
            engine = self.engines.get(engine_name)
            weight = engine.weight if engine else 1.0
            
            for result in results:
                # Apply engine weight to confidence score
                result.confidence_score *= weight
                all_results.append(result)
        
        if not all_results:
            return []
        
        # Deduplicate results
        deduplicated = self._deduplicate_results(all_results)
        
        # Sort by confidence score first
        prelim_ranked = sorted(deduplicated, key=lambda x: x.confidence_score, reverse=True)
        
        # Apply domain diversification cap before final trim
        max_per_domain = getattr(self.config, 'max_results_per_domain', 2)
        domain_counts: Dict[str, int] = {}
        diversified: List[SearchEngineResult] = []
        for item in prelim_ranked:
            domain = urlparse(item.url).netloc.lower() if item.url else ""
            if domain_counts.get(domain, 0) < max_per_domain:
                diversified.append(item)
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Return a larger pool; caller will apply semantic and/or cross-encoder rerank then trim
        return diversified[: max(target_count * getattr(self.config, 'candidate_multiplier_before_trim', 3), target_count)]
    
    @track_operation("enhanced_search_aggregation")
    async def search(self, query: str, num_results: int = 10, language: Optional[str] = None, time_range: Optional[str] = None) -> List[Source]:
        """Perform enhanced search across multiple engines"""
        logger.info("Starting enhanced search", query=query, num_results=num_results)
        
        # Check cache first
        cache_manager = await get_cache_manager()
        cached_results = await cache_manager.get_search_results(query, "enhanced", num_results, language=language, time_range=time_range)
        if cached_results:
            logger.info("Returning cached search results", query=query)
            return cached_results
        
        try:
            # Temporarily override config fields for cache key consistency
            prev_lang = getattr(self.config, 'language', None)
            prev_tr = getattr(self.config, 'time_range', None)
            if language is not None:
                setattr(self.config, 'language', language)
            if time_range is not None:
                setattr(self.config, 'time_range', time_range)
            # Search all engines
            pool_multiplier = max(2, getattr(self.config, 'candidate_multiplier_before_trim', 3))
            engine_results = await self.search_all_engines(query, num_results * pool_multiplier, language=language, time_range=time_range)
            
            if not engine_results or not any(engine_results.values()):
                logger.warning("No results from any search engine", query=query)
                return []
            
            # Rank and merge results
            merged_results = self._rank_and_merge_results(engine_results, num_results)
            
            # Calculate semantic relevance if enabled (before final trim)
            if self.config.semantic_search_enabled and merged_results:
                merged_results = self._calculate_semantic_relevance(query, merged_results)
                merged_results.sort(key=lambda x: x.confidence_score, reverse=True)
            
            # Optional cross-encoder rerank on top-K candidates
            if getattr(self.config, 'enable_cross_encoder', False) and self.cross_encoder and merged_results:
                top_k = min(len(merged_results), getattr(self.config, 'max_candidates_for_rerank', 50))
                pairs = [(query, f"{r.title} {r.snippet}") for r in merged_results[:top_k]]
                try:
                    scores = self.cross_encoder.predict(pairs)
                    for i, s in enumerate(scores):
                        merged_results[i].confidence_score = float(0.5 * merged_results[i].confidence_score + 0.5 * s)
                    merged_results.sort(key=lambda x: x.confidence_score, reverse=True)
                except Exception as e:
                    logger.warning("Cross-encoder rerank failed", error=str(e))
            
            # Convert to Source objects (final top N)
            final_ranked = merged_results[:num_results]
            sources = []
            for result in final_ranked:
                source = Source(
                    url=result.url,
                    title=result.title,
                    snippet=result.snippet,
                    raw_text=None,  # Will be populated during scraping
                    summary=None,   # Will be populated during scraping
                    metadata={
                        'search_engine': result.engine,
                        'engine_rank': result.rank,
                        'confidence_score': result.confidence_score,
                        **result.metadata
                    }
                )
                sources.append(source)
            
            # Cache results
            await cache_manager.set_search_results(query, "enhanced", num_results, sources, language=language, time_range=time_range)
            
            logger.info("Enhanced search completed", 
                       query=query, 
                       total_results=len(sources),
                       engines_used=list(engine_results.keys()))
            
            return sources
            
        except Exception as e:
            logger.error("Enhanced search failed", query=query, error=str(e))
            self.monitor.record_error("enhanced_search", str(e))
            raise
        finally:
            # Restore previous config overrides
            if language is not None:
                setattr(self.config, 'language', prev_lang)
            if time_range is not None:
                setattr(self.config, 'time_range', prev_tr)


# Global instance
enhanced_search_aggregator = EnhancedSearchAggregator()


async def get_enhanced_search_aggregator() -> EnhancedSearchAggregator:
    """Get global enhanced search aggregator instance"""
    return enhanced_search_aggregator


async def perform_enhanced_search(query: str, num_results: int = 10, language: Optional[str] = None, time_range: Optional[str] = None) -> List[Source]:
    """Convenience function to perform enhanced search"""
    aggregator = await get_enhanced_search_aggregator()
    return await aggregator.search(query, num_results, language=language, time_range=time_range)
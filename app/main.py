import time
import uuid
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Request, Response, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import structlog

# Import our enhanced services
from app.core.config import get_config
from app.core.cache import initialize_cache, shutdown_cache, get_cache_manager
from app.core.monitoring import get_monitor, EnhancedMonitor
from app.models.search_models import (
    SearchRequest, SearchResponse, Source, SearchStats, RelatedQuery,
    SearchType, ContentCategory, SourceMetadata
)
# External search engines removed - using only Xeno Engine
from app.services.enhanced_nlp_service import get_enhanced_nlp_service
from app.services.scraping_service import _scrape_page_content

# Import Xeno Search Engine components
from app.engine.crawler import XenoCrawler, CrawlConfig, CrawlResult
from app.engine.indexer import XenoIndexer, IndexConfig, PageRankCalculator
from app.engine.search import XenoSearchEngine, SearchConfig
from app.engine.dynamic_crawler import dynamic_crawl_for_query, detect_topics, get_domains_for_topics

logger = structlog.get_logger(__name__)

# Global configuration
config = get_config()
monitor = get_monitor()


async def require_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Optional API key authentication"""
    if not config.api_key_required:
        return

    if not x_api_key or x_api_key not in config.api_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Xeno Search Service", version=config.service_version)
    
    # Initialize core services
    await initialize_cache()
    
    # Pre-load critical models if configured
    if config.semantic_search_enabled:
        logger.info("Pre-loading NLP models...")
        try:
            nlp_service = await get_enhanced_nlp_service()
            # Trigger model loading
            await nlp_service.summarize_text_advanced("test", max_length=50)
        except Exception as e:
            logger.warning("Failed to pre-load NLP models", error=str(e))
    
    logger.info("Xeno Search Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Xeno Search Service")
    await shutdown_cache()
    logger.info("Xeno Search Service stopped")


# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="Xeno Search Service",
    description="Production-ready AI-powered search and analysis service that rivals Perplexity",
    version=config.service_version,
    docs_url="/docs" if config.debug else None,
    redoc_url="/redoc" if config.debug else None,
    lifespan=lifespan
)

# Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if config.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )


# Request/Response middleware for monitoring
@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    """Monitor all HTTP requests"""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    monitor.record_http_request(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
        duration=duration
    )
    
    # Add response headers
    response.headers["X-Processing-Time"] = f"{duration:.3f}s"
    response.headers["X-Service-Version"] = config.service_version
    
    return response


class RateLimitMiddleware:
    """In-memory IP-based rate limiting with automatic cleanup"""

    def __init__(self, app: FastAPI):
        self.app = app
        self.ip_requests: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # Cleanup every 5 minutes
        self._max_ips = 10000  # Maximum IPs to track before forced cleanup

    async def _cleanup_old_entries(self, now: float):
        """Remove entries older than 1 hour and limit total IPs tracked"""
        cutoff = now - 3600

        # Remove old timestamps and empty entries
        ips_to_remove = []
        for ip, timestamps in self.ip_requests.items():
            # Filter to only keep timestamps within the last hour
            fresh_timestamps = [t for t in timestamps if t > cutoff]
            if fresh_timestamps:
                self.ip_requests[ip] = fresh_timestamps
            else:
                ips_to_remove.append(ip)

        for ip in ips_to_remove:
            del self.ip_requests[ip]

        # If still too many IPs, remove oldest entries
        if len(self.ip_requests) > self._max_ips:
            # Sort by most recent request and keep only max_ips
            sorted_ips = sorted(
                self.ip_requests.items(),
                key=lambda x: max(x[1]) if x[1] else 0,
                reverse=True
            )[:self._max_ips]
            self.ip_requests = dict(sorted_ips)

        self._last_cleanup = now
        logger.debug("Rate limiter cleanup completed",
                    tracked_ips=len(self.ip_requests),
                    removed_ips=len(ips_to_remove))

    async def __call__(self, request: Request, call_next):
        if not config.rate_limiting_enabled:
            return await call_next(request)

        ip = request.client.host if request.client else "anonymous"
        now = time.time()

        async with self._lock:
            # Periodic cleanup
            if now - self._last_cleanup > self._cleanup_interval or len(self.ip_requests) > self._max_ips:
                await self._cleanup_old_entries(now)

            minute_limit = config.rate_limit_requests_per_minute
            hour_limit = config.rate_limit_requests_per_hour

            timestamps = self.ip_requests.get(ip, [])
            # Filter to only timestamps within the last hour
            timestamps = [t for t in timestamps if now - t < 3600]

            minute_count = len([t for t in timestamps if now - t < 60])

            if minute_count >= minute_limit or len(timestamps) >= hour_limit:
                logger.warning("Rate limit exceeded", ip=ip, minute_count=minute_count, hour_count=len(timestamps))
                return Response(status_code=429, content="Rate limit exceeded")

            timestamps.append(now)
            self.ip_requests[ip] = timestamps

        return await call_next(request)


# Only add rate limiting middleware if enabled
if config.rate_limiting_enabled:
    from starlette.middleware.base import BaseHTTPMiddleware
    app.add_middleware(BaseHTTPMiddleware, dispatch=RateLimitMiddleware(app).__call__)


# WebSocket connection manager for real-time updates
class EnhancedWebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.search_connections: Dict[str, str] = {}  # search_id -> connection_id
    
    async def connect(self, websocket: WebSocket, connection_id: str, search_id: str = None):
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        if search_id:
            self.search_connections[search_id] = connection_id
        monitor.set_active_connections(len(self.active_connections))
        logger.info("WebSocket connected", connection_id=connection_id, search_id=search_id)
    
    def disconnect(self, connection_id: str, search_id: str = None):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        if search_id and search_id in self.search_connections:
            del self.search_connections[search_id]
        monitor.set_active_connections(len(self.active_connections))
        logger.info("WebSocket disconnected", connection_id=connection_id)
    
    async def send_progress_update(self, search_id: str, phase: str, progress: int,
                                 message: str, data: dict = None):
        connection_id = self.search_connections.get(search_id)
        if not connection_id or connection_id not in self.active_connections:
            return

        try:
            update = {
                "search_id": search_id,
                "type": "progress_update",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "phase": phase,
                    "progress": progress,
                    "message": message,
                    "data": data or {}
                }
            }

            websocket = self.active_connections.get(connection_id)
            if websocket is None:
                # Connection was removed between check and access
                return

            await websocket.send_json(update)

        except Exception as e:
            logger.warning("Failed to send WebSocket update",
                         search_id=search_id, error=str(e))
            # Clean up dead connection
            self.disconnect(connection_id, search_id)


websocket_manager = EnhancedWebSocketManager()


# Enhanced search endpoint with full feature set
@app.post("/api/v2/search", response_model=SearchResponse)
async def enhanced_search_endpoint(request: SearchRequest, api_key: None = Depends(require_api_key)):
    """
    Enhanced search endpoint with comprehensive features
    """
    start_time = time.time()
    
    async with monitor.track_operation("enhanced_search_request", 
                                     query=request.query, 
                                     search_type=request.search_type.value) as metrics:
        
        logger.info("Enhanced search request received", 
                   query=request.query, 
                   search_type=request.search_type,
                   num_results=request.num_results)
        
        try:
            # Validate request
            if len(request.query.strip()) < 2:
                raise HTTPException(status_code=400, detail="Query too short")
            
            # Check rate limiting (if enabled)
            # TODO: Implement rate limiting logic
            
            # Initialize response
            response_id = f"search-{uuid.uuid4().hex[:12]}"
            sources = []
            search_stats = SearchStats(
                total_results=0,
                processing_time_ms=0,
                engines_used=[],
                cache_hits=0,
                cache_misses=0
            )
            
            # All search types now use Xeno Engine (custom index)
            # External search engines (DuckDuckGo, SearXNG, Brave) have been removed
            sources = await perform_xeno_search(request.query, request.num_results)
            search_stats.engines_used = ["xeno_engine"]
            
            # Enhanced content processing
            if sources and request.include_summaries:
                await _process_sources_with_enhanced_nlp(sources, request.query)
            
            # Apply content filtering
            if request.min_quality_score:
                sources = [s for s in sources if s.quality_score and s.quality_score >= request.min_quality_score]
            
            if request.include_categories or request.exclude_categories:
                sources = _filter_sources_by_category(sources, request.include_categories, request.exclude_categories)
            
            # Generate comprehensive summary
            summary = await _generate_enhanced_summary(request.query, sources, request.search_type)
            
            # Generate related queries
            related_queries = await _generate_related_queries(request.query, sources[:5])
            
            # Calculate statistics
            processing_time_ms = int((time.time() - start_time) * 1000)
            search_stats.total_results = len(sources)
            search_stats.processing_time_ms = processing_time_ms
            
            if sources:
                confidence_scores = [s.confidence_score for s in sources if s.confidence_score]
                if confidence_scores:
                    search_stats.avg_confidence_score = sum(confidence_scores) / len(confidence_scores)
                
                quality_scores = [s.quality_score for s in sources if s.quality_score]
                if quality_scores:
                    search_stats.avg_quality_score = sum(quality_scores) / len(quality_scores)
                
                # Category distribution
                for source in sources:
                    if source.content_category:
                        search_stats.category_distribution[source.content_category] = \
                            search_stats.category_distribution.get(source.content_category, 0) + 1
            
            # Create enhanced response
            response = SearchResponse(
                query=request.query,
                search_type=request.search_type,
                sources=sources,
                summary=summary,
                related_queries=related_queries,
                stats=search_stats,
                response_id=response_id,
                processing_time_ms=processing_time_ms
            )
            
            # Calculate additional metrics
            response.overall_confidence = response.calculate_overall_confidence()
            response.result_diversity_score = response.calculate_result_diversity()
            
            # Record metrics
            monitor.record_search_request(
                request.search_type.value,
                processing_time_ms / 1000.0,
                len(sources)
            )
            
            # Update metrics metadata
            metrics.metadata.update({
                "results_count": len(sources),
                "overall_confidence": response.overall_confidence,
                "diversity_score": response.result_diversity_score
            })
            
            logger.info("Enhanced search completed successfully",
                       query=request.query,
                       results_count=len(sources),
                       processing_time_ms=processing_time_ms,
                       overall_confidence=response.overall_confidence)
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Enhanced search failed", query=request.query, error=str(e))
            monitor.record_error("enhanced_search", str(e))
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


async def _process_sources_with_enhanced_nlp(sources: List[Source], query: str):
    """Process sources with enhanced NLP capabilities"""
    nlp_service = await get_enhanced_nlp_service()
    
    # Process sources in batches for efficiency
    batch_size = 5
    for i in range(0, len(sources), batch_size):
        batch = sources[i:i + batch_size]
        tasks = []
        
        for source in batch:
            if source.raw_text:
                task = nlp_service.analyze_text_comprehensively(
                    source.raw_text,
                    source.title or "",
                    include_summary=True,
                    include_classification=True,
                    include_entities=False,  # Skip entities for performance
                    include_keywords=True
                )
                tasks.append((source, task))
        
        # Execute batch
        for source, task in tasks:
            try:
                analysis = await task
                
                # Update source with analysis results
                if analysis.get('summary'):
                    summary_result = analysis['summary']
                    if hasattr(summary_result, 'summary'):
                        source.summary = summary_result.summary
                        if not source.metadata:
                            source.metadata = SourceMetadata()
                        source.metadata.content_quality_score = summary_result.confidence_score
                
                if analysis.get('classification'):
                    classification = analysis['classification']
                    if classification:
                        source.content_category = classification.category
                        if not source.confidence_score:
                            source.confidence_score = classification.confidence
                
                if analysis.get('keywords'):
                    source.content_tags = analysis['keywords'][:5]  # Top 5 keywords
                
                # Calculate quality score
                source.quality_score = source.calculate_quality_score()
                
            except Exception as e:
                logger.warning("Failed to process source with NLP", 
                             url=source.url, error=str(e))


def _filter_sources_by_category(sources: List[Source], 
                               include_categories: List[ContentCategory],
                               exclude_categories: List[ContentCategory]) -> List[Source]:
    """Filter sources by content category"""
    filtered = []
    
    for source in sources:
        # Skip if category should be excluded
        if exclude_categories and source.content_category in exclude_categories:
            continue
        
        # Include if no specific categories requested or category matches
        if not include_categories or source.content_category in include_categories:
            filtered.append(source)
    
    return filtered


async def _generate_enhanced_summary(query: str, sources: List[Source], 
                                   search_type: SearchType) -> Optional[str]:
    """Generate enhanced summary using advanced NLP"""
    if not sources:
        return f"No results found for query: '{query}'"
    
    try:
        nlp_service = await get_enhanced_nlp_service()
        
        # Collect summaries and key content
        summaries = []
        for source in sources[:10]:  # Limit to top 10 for summary
            if source.summary:
                summaries.append(f"From {source.get_domain()}: {source.summary}")
            elif source.snippet:
                summaries.append(f"From {source.get_domain()}: {source.snippet}")
        
        if not summaries:
            return f"Found {len(sources)} sources for '{query}' but unable to generate summary."
        
        # Combine summaries
        combined_text = "\n\n".join(summaries)
        
        # Generate meta-summary
        summary_result = await nlp_service.summarize_text_advanced(
            combined_text,
            max_length=300,
            min_length=100,
            strategy="balanced"
        )
        
        if summary_result and summary_result.summary:
            return f"Search Summary for '{query}':\n\n{summary_result.summary}"
        
        # Fallback summary
        return f"Found {len(sources)} relevant sources for '{query}'. Key insights from multiple sources have been aggregated."
        
    except Exception as e:
        logger.warning("Enhanced summary generation failed", error=str(e))
        return f"Found {len(sources)} sources for '{query}'. Summary generation temporarily unavailable."


async def _generate_related_queries(query: str, sources: List[Source]) -> List[RelatedQuery]:
    """Generate related query suggestions"""
    try:
        # Extract keywords from top sources
        all_keywords = set()
        for source in sources:
            if source.content_tags:
                all_keywords.update(source.content_tags)
        
        # Generate simple related queries (in production, use more sophisticated methods)
        related = []
        keywords = list(all_keywords)[:10]
        
        for keyword in keywords:
            if keyword.lower() not in query.lower():
                related_query = f"{query} {keyword}"
                related.append(RelatedQuery(
                    query=related_query,
                    confidence=0.7,
                    category="keyword_expansion"
                ))
        
        return related[:5]  # Top 5 related queries
        
    except Exception as e:
        logger.warning("Related query generation failed", error=str(e))
        return []


# Legacy endpoint for backward compatibility
@app.post("/api/xeno-search-internal", response_model=SearchResponse)
async def legacy_search_endpoint(request: SearchRequest, api_key: None = Depends(require_api_key)):
    """Legacy search endpoint for backward compatibility"""
    # Convert to new format and delegate
    return await enhanced_search_endpoint(request)


# WebSocket endpoint for real-time search updates
@app.websocket("/ws/search/{search_id}")
async def search_websocket(websocket: WebSocket, search_id: str):
    """WebSocket endpoint for real-time search progress"""
    connection_id = f"ws-{uuid.uuid4().hex[:8]}"
    
    await websocket_manager.connect(websocket, connection_id, search_id)
    
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back to keep connection active
            await websocket.send_json({"type": "ping", "data": "pong"})
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(connection_id, search_id)


# Advanced search endpoint with real-time updates
@app.post("/api/v2/search/streaming")
async def streaming_search_endpoint(request: SearchRequest, api_key: None = Depends(require_api_key)):
    """Start a streaming search with WebSocket updates"""
    search_id = f"search-{uuid.uuid4().hex[:12]}"
    
    # Start search in background
    asyncio.create_task(perform_streaming_search(search_id, request))
    
    return {"search_id": search_id, "websocket_url": f"/ws/search/{search_id}"}


async def perform_streaming_search(search_id: str, request: SearchRequest):
    """Perform search with real-time WebSocket updates"""
    try:
        await websocket_manager.send_progress_update(
            search_id, "initializing", 5, "🔄 Initializing Xeno Engine search..."
        )

        await websocket_manager.send_progress_update(
            search_id, "searching", 20, "🔍 Searching Xeno Engine index..."
        )

        # Perform the actual search using Xeno Engine only
        sources = await perform_xeno_search(request.query, request.num_results)
        
        await websocket_manager.send_progress_update(
            search_id, "processing", 60, f"📊 Processing {len(sources)} sources..."
        )
        
        # Enhanced processing
        if sources:
            await _process_sources_with_enhanced_nlp(sources, request.query)
        
        await websocket_manager.send_progress_update(
            search_id, "finalizing", 90, "🤖 Generating AI analysis..."
        )
        
        # Generate summary and related queries
        summary = await _generate_enhanced_summary(request.query, sources, request.search_type)
        related_queries = await _generate_related_queries(request.query, sources[:5])
        
        # Send final results
        response = SearchResponse(
            query=request.query,
            search_type=request.search_type,
            sources=sources,
            summary=summary,
            related_queries=related_queries,
            response_id=search_id
        )
        
        await websocket_manager.send_progress_update(
            search_id, "completed", 100, f"✅ Search completed! Found {len(sources)} sources.",
            {"results": response.dict()}
        )
        
    except Exception as e:
        logger.error("Streaming search failed", search_id=search_id, error=str(e))
        await websocket_manager.send_progress_update(
            search_id, "error", 0, f"❌ Search failed: {str(e)}"
        )


# Content analysis endpoint
@app.post("/api/v2/analyze")
async def analyze_content_endpoint(
    url: Optional[str] = None,
    text: Optional[str] = None,
    include_summary: bool = True,
    include_classification: bool = True,
    include_entities: bool = True,
    include_keywords: bool = True,
    api_key: None = Depends(require_api_key)
):
    """Analyze content from URL or text"""
    if not url and not text:
        raise HTTPException(status_code=400, detail="Either URL or text must be provided")
    
    try:
        content_text = text
        title = ""
        
        if url and not text:
            # Scrape content from URL
            scraped_data = await _scrape_page_content(url)
            content_text = scraped_data.get('raw_text')
            title = scraped_data.get('title', '')
            
            if not content_text:
                raise HTTPException(status_code=400, detail="Failed to extract content from URL")
        
        # Perform comprehensive analysis
        nlp_service = await get_enhanced_nlp_service()
        analysis = await nlp_service.analyze_text_comprehensively(
            content_text,
            title,
            include_summary=include_summary,
            include_classification=include_classification,
            include_entities=include_entities,
            include_keywords=include_keywords
        )
        
        return {
            "url": url,
            "analysis": analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Content analysis failed", url=url, error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = await monitor.get_health_status()
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    
    return JSONResponse(
        content=health_status,
        status_code=status_code
    )


# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    metrics_data = monitor.get_prometheus_metrics()
    return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)


# Performance analytics endpoint
@app.get("/api/v2/analytics/performance")
async def performance_analytics(
    operation: Optional[str] = None,
    hours: int = 24
):
    """Get performance analytics"""
    summary = monitor.get_performance_summary(operation, hours)
    
    # Add cache statistics
    cache_manager = await get_cache_manager()
    cache_stats = cache_manager.get_cache_stats()
    
    return {
        "performance_summary": summary,
        "cache_statistics": cache_stats,
        "timestamp": datetime.utcnow().isoformat()
    }


# Cache management endpoints
@app.post("/api/v2/admin/cache/clear")
async def clear_cache_endpoint(pattern: str = "*"):
    """Clear cache entries"""
    cache_manager = await get_cache_manager()
    success = await cache_manager.clear_cache(pattern)
    
    return {
        "success": success,
        "pattern": pattern,
        "timestamp": datetime.utcnow().isoformat()
    }


# Configuration endpoint
@app.get("/api/v2/config")
async def get_configuration():
    """Get current service configuration (non-sensitive fields only)"""
    return {
        "service_name": config.service_name,
        "service_version": config.service_version,
        "environment": config.environment,
        "search_engines": config.search_engines,
        "features": {
            "deep_search": config.deep_search_enabled,
            "semantic_search": config.semantic_search_enabled,
            "caching": config.cache_enabled,
            "monitoring": config.metrics_enabled,
            "websockets": config.enable_websocket_updates
        },
        "limits": {
            "max_results_per_request": config.max_results_per_request,
            "default_num_results": config.default_num_results
        }
    }


# Root endpoint
@app.get("/")
async def root():
    """Service information"""
    return {
        "service": "Xeno Search Service",
        "version": config.service_version,
        "description": "Production-ready AI-powered search and analysis service with custom search engine",
        "status": "operational",
        "features": [
            "Multi-engine search aggregation",
            "Advanced AI summarization",
            "Semantic search and ranking",
            "Deep link analysis",
            "Real-time progress updates",
            "Comprehensive content analysis",
            "Redis caching with deduplication",
            "Prometheus metrics and monitoring",
            "🔍 Xeno Search Engine - Custom web crawler",
            "🗃️ Xeno Search Engine - Meilisearch indexing",
            "📊 Xeno Search Engine - PageRank algorithm"
        ],
        "xeno_engine": {
            "search": "/api/v2/engine/search",
            "crawl": "/api/v2/engine/crawl",
            "stats": "/api/v2/engine/stats",
            "domains": "/api/v2/engine/domains"
        },
        "external_search": {
            "google": "/api/v2/engine/google-search",
            "brave": "/api/v2/engine/brave-search"
        },
        "docs_url": "/docs" if config.debug else None,
        "health_url": "/health",
        "metrics_url": "/metrics"
    }

# ============================================================================
# XENO SEARCH ENGINE - Custom Crawler & Index API
# ============================================================================

from pydantic import BaseModel, Field
from typing import Optional as OptionalType
import aiohttp

# Pydantic models for Xeno Search Engine
class XenoEngineSearchRequest(BaseModel):
    """Request model for searching the custom index"""
    query: str = Field(..., min_length=1, max_length=500)
    page: int = Field(default=1, ge=1)
    per_page: int = Field(default=10, ge=1, le=50)
    domain: OptionalType[str] = None
    language: OptionalType[str] = None
    min_word_count: OptionalType[int] = None
    sort_by: OptionalType[str] = None


class XenoCrawlRequest(BaseModel):
    """Request model for starting a crawl job"""
    seed_urls: List[str] = Field(..., min_items=1, max_items=10)
    max_pages: int = Field(default=100, ge=1, le=10000)
    max_depth: int = Field(default=5, ge=1, le=10)
    follow_external: bool = False


# Global crawl job storage (in production, use Redis or database)
crawl_jobs: Dict[str, Dict[str, Any]] = {}

# Global Xeno Search Engine instance
xeno_search_engine: OptionalType[XenoSearchEngine] = None


async def get_xeno_search_engine() -> XenoSearchEngine:
    """Get or create Xeno Search Engine instance"""
    global xeno_search_engine
    if xeno_search_engine is None:
        # Use Meilisearch configuration from environment/config
        search_config = SearchConfig(
            host=config.meili_host,
            api_key=config.meili_master_key,
            index_name=config.meili_index_name
        )
        xeno_search_engine = XenoSearchEngine(search_config)
        logger.info("Xeno Search Engine initialized",
                   meili_host=config.meili_host,
                   index_name=config.meili_index_name)
    return xeno_search_engine


def convert_xeno_results_to_sources(xeno_results: List[Any], query: str) -> List[Source]:
    """Convert Xeno Engine search results to Source objects for API response"""
    sources = []
    for i, result in enumerate(xeno_results):
        # Convert SearchResult dict or object to Source model
        if isinstance(result, dict):
            url = result.get("url", "")
            title = result.get("title", "")
            description = result.get("description", "")
            snippet = result.get("snippet", description)
            domain = result.get("domain", "")
            score = result.get("score", 0.0)
            page_rank = result.get("page_rank", 0.0)
            word_count = result.get("word_count", 0)
        else:
            # Handle SearchResult dataclass
            url = getattr(result, "url", "")
            title = getattr(result, "title", "")
            description = getattr(result, "description", "")
            snippet = getattr(result, "snippet", description)
            domain = getattr(result, "domain", "")
            score = getattr(result, "score", 0.0)
            page_rank = getattr(result, "page_rank", 0.0)
            word_count = getattr(result, "word_count", 0)

        # Calculate confidence and quality scores from Xeno Engine scores
        # Normalize score (Meilisearch returns 0-1 or higher depending on ranking)
        confidence = min(1.0, score) if score <= 1.0 else min(1.0, score / 100)
        quality = min(1.0, (page_rank + 0.5) / 1.5)  # Normalize PageRank

        metadata = SourceMetadata(
            search_engine="xeno_engine",
            engine_rank=i + 1,
            confidence_score=confidence,
            domain=domain,
            word_count=word_count,
            content_quality_score=quality,
        )

        source = Source(
            url=url,
            title=title,
            snippet=snippet,
            summary=description,
            metadata=metadata,
            confidence_score=confidence,
            quality_score=quality,
            relevance_score=confidence,
        )
        sources.append(source)

    return sources


async def perform_xeno_search(query: str, num_results: int = 10) -> List[Source]:
    """Perform search using only Xeno Engine (custom index)"""
    try:
        engine = await get_xeno_search_engine()
        response = await engine.search(
            query=query,
            per_page=num_results,
            page=1
        )

        # Convert results to Source objects
        results = response.results if hasattr(response, 'results') else []
        sources = convert_xeno_results_to_sources(results, query)

        logger.info("Xeno Engine search completed",
                   query=query,
                   results_count=len(sources))

        return sources

    except Exception as e:
        logger.error("Xeno Engine search failed", query=query, error=str(e))
        return []


# Search using our custom index
@app.post("/api/v2/engine/search")
async def xeno_engine_search(request: XenoEngineSearchRequest):
    """
    Search using Xeno Search Engine's custom index

    This searches your own crawled and indexed content, not external search engines.
    """
    try:
        engine = await get_xeno_search_engine()

        response = await engine.search(
            query=request.query,
            page=request.page,
            per_page=request.per_page,
            domain=request.domain,
            language=request.language,
            min_word_count=request.min_word_count,
            sort_by=request.sort_by
        )

        logger.info("Xeno engine search completed",
                   query=request.query,
                   results=len(response.results),
                   total_hits=response.total_hits)

        return response.to_dict()

    except Exception as e:
        logger.error("Xeno engine search failed", query=request.query, error=str(e))
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


class DynamicSearchRequest(BaseModel):
    """Request for dynamic real-time search"""
    query: str = Field(..., min_length=2, max_length=500)
    max_pages: int = Field(default=10, ge=1, le=20)
    index_results: bool = Field(default=True, description="Also index crawled pages for future searches")


@app.post("/api/v2/engine/dynamic-search")
async def dynamic_search_endpoint(request: DynamicSearchRequest):
    """
    Dynamic Real-Time Search

    This endpoint:
    1. Detects topics from your query
    2. Finds relevant domains (apple.com, techcrunch.com, etc.)
    3. Uses site search to find exact pages with answers
    4. Crawls those pages in real-time
    5. Optionally indexes them for future searches
    6. Returns fresh, relevant results

    No external search APIs used - only our curated domain database.
    """
    start_time = time.time()

    try:
        # Step 1: Detect topics
        topics = detect_topics(request.query)
        domains = get_domains_for_topics(topics)

        logger.info("Dynamic search started",
                   query=request.query,
                   topics=topics,
                   domains=domains[:5])

        # Step 2: Dynamic crawl to find and fetch relevant pages
        crawl_results = await dynamic_crawl_for_query(
            query=request.query,
            max_pages=request.max_pages
        )

        # Step 3: Optionally index the results
        indexed_count = 0
        if request.index_results and crawl_results:
            try:
                engine = await get_xeno_search_engine()
                indexer = XenoIndexer(IndexConfig(
                    host=config.meili_host,
                    api_key=config.meili_master_key
                ))

                for page in crawl_results:
                    if page.get('content') and page.get('url'):
                        await indexer.index_page({
                            'url': page['url'],
                            'title': page.get('title', ''),
                            'content': page.get('content', ''),
                            'description': page.get('description', ''),
                            'domain': urlparse(page['url']).netloc,
                            'word_count': page.get('word_count', 0),
                        })
                        indexed_count += 1

                logger.info("Indexed dynamic search results", count=indexed_count)
            except Exception as e:
                logger.warning("Failed to index dynamic results", error=str(e))

        # Step 4: Format response
        processing_time_ms = int((time.time() - start_time) * 1000)

        results = []
        for page in crawl_results:
            results.append({
                'url': page.get('url'),
                'title': page.get('title'),
                'description': page.get('description'),
                'snippet': page.get('content', '')[:300] + '...' if page.get('content') else '',
                'relevance_score': page.get('relevance_score', 0),
                'word_count': page.get('word_count', 0),
                'source': 'dynamic_crawl'
            })

        return {
            'query': request.query,
            'topics_detected': topics,
            'domains_searched': domains[:5],
            'results': results,
            'total_results': len(results),
            'indexed_count': indexed_count,
            'processing_time_ms': processing_time_ms,
            'method': 'dynamic_crawl',
            'note': 'Results crawled in real-time from authoritative sources'
        }

    except Exception as e:
        logger.error("Dynamic search failed", query=request.query, error=str(e))
        raise HTTPException(status_code=500, detail=f"Dynamic search failed: {str(e)}")


# Get search suggestions
@app.get("/api/v2/engine/suggest")
async def xeno_engine_suggest(query: str, limit: int = 5):
    """Get search suggestions based on indexed content"""
    try:
        engine = await get_xeno_search_engine()
        suggestions = await engine.suggest(query, limit)
        return {"query": query, "suggestions": suggestions}
    except Exception as e:
        logger.error("Xeno engine suggest failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Get indexed domains
@app.get("/api/v2/engine/domains")
async def xeno_engine_domains(limit: int = 20):
    """Get list of indexed domains with document counts"""
    try:
        engine = await get_xeno_search_engine()
        domains = await engine.get_domains(limit)
        return {"domains": domains, "count": len(domains)}
    except Exception as e:
        logger.error("Xeno engine get domains failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Get index statistics
@app.get("/api/v2/engine/stats")
async def xeno_engine_stats():
    """Get search engine index statistics"""
    try:
        engine = await get_xeno_search_engine()
        stats = await engine.get_stats()
        health = await engine.health_check()

        return {
            "index_stats": stats,
            "healthy": health,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("Xeno engine stats failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Start a crawl job
@app.post("/api/v2/engine/crawl")
async def start_crawl_job(request: XenoCrawlRequest):
    """
    Start a new web crawling job

    The crawler will:
    1. Start from the seed URLs
    2. Crawl up to max_pages following links
    3. Index all crawled content into Meilisearch
    """
    job_id = f"crawl-{uuid.uuid4().hex[:12]}"

    # Store job info
    crawl_jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "seed_urls": request.seed_urls,
        "max_pages": request.max_pages,
        "max_depth": request.max_depth,
        "follow_external": request.follow_external,
        "pages_crawled": 0,
        "pages_indexed": 0,
        "errors": 0,
        "started_at": None,
        "completed_at": None,
        "created_at": datetime.utcnow().isoformat()
    }

    # Start crawl in background
    asyncio.create_task(run_crawl_job(job_id, request))

    logger.info("Crawl job started", job_id=job_id, seed_urls=request.seed_urls)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Crawl job started with {len(request.seed_urls)} seed URLs"
    }


async def run_crawl_job(job_id: str, request: XenoCrawlRequest):
    """Background task to run a crawl job"""
    try:
        crawl_jobs[job_id]["status"] = "running"
        crawl_jobs[job_id]["started_at"] = datetime.utcnow().isoformat()

        # Configure crawler
        meili_host = "http://meilisearch:7700" if config.environment == "production" else "http://localhost:7700"
        redis_url = config.redis_url if hasattr(config, 'redis_url') else "redis://localhost:6379/2"

        crawl_config = CrawlConfig(
            max_pages_per_domain=request.max_pages,
            max_depth=request.max_depth,
            follow_external_links=request.follow_external,
            delay_between_requests=1.0,
            respect_robots_txt=True
        )

        crawler = XenoCrawler(redis_url=redis_url, config=crawl_config)

        # Initialize indexer
        index_config = IndexConfig(
            host=meili_host,
            api_key=config.meili_master_key if hasattr(config, 'meili_master_key') else "xeno_search_master_key_change_me"
        )
        indexer = XenoIndexer(index_config)
        await indexer.initialize()

        # Crawl with real-time indexing callback
        crawled_results: List[CrawlResult] = []

        async def on_page_crawled(result: CrawlResult):
            """Index each page as it's crawled"""
            crawled_results.append(result)
            crawl_jobs[job_id]["pages_crawled"] = len(crawled_results)

            # Index immediately
            success = await indexer.index_page(result)
            if success:
                crawl_jobs[job_id]["pages_indexed"] += 1

        # Run the crawl
        results = await crawler.crawl(
            seed_urls=request.seed_urls,
            max_pages=request.max_pages,
            callback=on_page_crawled
        )

        # Calculate PageRank and re-index with scores
        if results:
            logger.info("Calculating PageRank scores", pages=len(results))
            calculator = PageRankCalculator()
            page_ranks = calculator.calculate(results)

            # Re-index with PageRank scores
            indexed = await indexer.index_pages(results, page_ranks)
            crawl_jobs[job_id]["pages_indexed"] = indexed

        crawl_jobs[job_id]["status"] = "completed"
        crawl_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()

        logger.info("Crawl job completed",
                   job_id=job_id,
                   pages_crawled=len(results),
                   pages_indexed=crawl_jobs[job_id]["pages_indexed"])

    except Exception as e:
        logger.error("Crawl job failed", job_id=job_id, error=str(e))
        crawl_jobs[job_id]["status"] = "failed"
        crawl_jobs[job_id]["error"] = str(e)
        crawl_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()


# Get crawl job status
@app.get("/api/v2/engine/crawl/{job_id}")
async def get_crawl_job(job_id: str):
    """Get the status of a crawl job"""
    if job_id not in crawl_jobs:
        raise HTTPException(status_code=404, detail="Crawl job not found")

    return crawl_jobs[job_id]


# List all crawl jobs
@app.get("/api/v2/engine/crawl")
async def list_crawl_jobs():
    """List all crawl jobs"""
    return {
        "jobs": list(crawl_jobs.values()),
        "total": len(crawl_jobs)
    }


# Delete pages by domain
@app.delete("/api/v2/engine/domain/{domain}")
async def delete_domain(domain: str):
    """Delete all indexed pages from a domain"""
    try:
        meili_host = "http://meilisearch:7700" if config.environment == "production" else "http://localhost:7700"
        index_config = IndexConfig(
            host=meili_host,
            api_key=config.meili_master_key if hasattr(config, 'meili_master_key') else "xeno_search_master_key_change_me"
        )
        indexer = XenoIndexer(index_config)
        await indexer.initialize()

        task_uid = await indexer.delete_domain(domain)

        return {
            "success": True,
            "domain": domain,
            "task_uid": task_uid,
            "message": f"Deletion task queued for domain: {domain}"
        }
    except Exception as e:
        logger.error("Delete domain failed", domain=domain, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Clear entire index
@app.delete("/api/v2/engine/index")
async def clear_index():
    """Clear all documents from the search index (DANGEROUS!)"""
    try:
        meili_host = "http://meilisearch:7700" if config.environment == "production" else "http://localhost:7700"
        index_config = IndexConfig(
            host=meili_host,
            api_key=config.meili_master_key if hasattr(config, 'meili_master_key') else "xeno_search_master_key_change_me"
        )
        indexer = XenoIndexer(index_config)
        await indexer.initialize()

        task_uid = await indexer.clear_index()

        return {
            "success": True,
            "task_uid": task_uid,
            "message": "Index clear task queued"
        }
    except Exception as e:
        logger.error("Clear index failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SMART AUTO-CRAWL SEARCH
# ============================================================================

class SmartSearchRequest(BaseModel):
    """Request for smart search with auto-crawl"""
    query: str = Field(..., min_length=1, max_length=500)
    auto_crawl: bool = Field(default=True, description="Auto-crawl if no results found")
    max_crawl_pages: int = Field(default=5, ge=1, le=20)
    min_results_threshold: int = Field(default=2, description="Trigger crawl if fewer results")


async def find_urls_to_crawl(query: str, num_urls: int = 5) -> List[str]:
    """
    Find relevant URLs to crawl based on a search query
    Uses the Xeno Engine index and topic seed database to discover URLs

    Note: External search engines have been removed. URLs are discovered from:
    1. Existing Xeno Engine index
    2. Topic seed database (curated authoritative sources)
    """
    urls = []
    skip_domains = ['youtube.com', 'facebook.com', 'twitter.com', 'instagram.com',
                    'tiktok.com', 'reddit.com', 'linkedin.com', 'pinterest.com',
                    'google.com', 'bing.com', 'yahoo.com']

    try:
        # Use Xeno Engine to find URLs from existing index
        sources = await perform_xeno_search(query, num_results=num_urls * 2)

        for source in sources:
            url = source.url if hasattr(source, 'url') else str(source)
            if url and url.startswith('http'):
                if not any(domain in url for domain in skip_domains):
                    if url not in urls:
                        urls.append(url)
                        if len(urls) >= num_urls:
                            break

        if urls:
            logger.info("Found URLs from Xeno Engine index", count=len(urls))

    except Exception as e:
        logger.warning("Xeno Engine search failed for URL discovery", error=str(e))

    # If no URLs found, try topic seed database
    if len(urls) < 2:
        try:
            from app.engine.seeds import TOPIC_SEEDS, detect_topics
            detected_topics = detect_topics(query)

            for topic in detected_topics:
                if topic in TOPIC_SEEDS:
                    for seed_url in TOPIC_SEEDS[topic]:
                        if seed_url not in urls:
                            urls.append(seed_url)
                            if len(urls) >= num_urls:
                                break
                if len(urls) >= num_urls:
                    break

            if urls:
                logger.info("Found seed URLs from topic database", count=len(urls), topics=detected_topics[:3])
        except Exception as e:
            logger.warning("Topic seed database lookup failed", error=str(e))

    logger.info("Total URLs found to crawl", query=query, count=len(urls), urls=urls[:3])
    return urls[:num_urls]


@app.post("/api/v2/engine/smart-search")
async def smart_search(request: SmartSearchRequest):
    """
    Smart search with automatic crawling

    1. Search existing index
    2. If not enough results and auto_crawl=True:
       - Find relevant URLs via external search
       - Crawl and index them
       - Return combined results
    """
    start_time = time.time()

    try:
        engine = await get_xeno_search_engine()

        # First, search existing index
        initial_response = await engine.search(
            query=request.query,
            per_page=10
        )

        initial_results = len(initial_response.results)
        crawled_new = False
        crawl_info = None

        # If not enough results and auto-crawl is enabled
        if initial_results < request.min_results_threshold and request.auto_crawl:
            logger.info("Not enough results, triggering auto-crawl",
                       query=request.query,
                       initial_results=initial_results,
                       threshold=request.min_results_threshold)

            # Find URLs to crawl
            urls_to_crawl = await find_urls_to_crawl(request.query, num_urls=request.max_crawl_pages)

            if urls_to_crawl:
                # Configure crawler and indexer
                meili_host = "http://meilisearch:7700" if config.environment == "production" else "http://localhost:7700"
                redis_url = config.redis_url if hasattr(config, 'redis_url') else "redis://localhost:6379/2"

                crawl_config = CrawlConfig(
                    max_pages_per_domain=3,  # Limit per domain for speed
                    max_depth=1,  # Don't go too deep
                    follow_external_links=False,
                    delay_between_requests=0.5,  # Faster for auto-crawl
                    respect_robots_txt=True
                )

                crawler = XenoCrawler(redis_url=redis_url, config=crawl_config)

                index_config = IndexConfig(
                    host=meili_host,
                    api_key=config.meili_master_key if hasattr(config, 'meili_master_key') else "xeno_search_master_key_change_me"
                )
                indexer = XenoIndexer(index_config)
                await indexer.initialize()

                # Crawl the URLs
                try:
                    results = await crawler.crawl(
                        seed_urls=urls_to_crawl,
                        max_pages=request.max_crawl_pages
                    )

                    if results:
                        # Calculate PageRank and index
                        calculator = PageRankCalculator()
                        page_ranks = calculator.calculate(results)
                        indexed = await indexer.index_pages(results, page_ranks)

                        crawled_new = True
                        crawl_info = {
                            "urls_found": len(urls_to_crawl),
                            "pages_crawled": len(results),
                            "pages_indexed": indexed
                        }

                        logger.info("Auto-crawl completed",
                                   query=request.query,
                                   pages_indexed=indexed)

                        # Wait a moment for Meilisearch to index
                        await asyncio.sleep(0.5)

                except Exception as e:
                    logger.error("Auto-crawl failed", query=request.query, error=str(e))

        # Search again (with new content if crawled)
        final_response = await engine.search(
            query=request.query,
            per_page=10
        )

        processing_time = int((time.time() - start_time) * 1000)

        return {
            "query": request.query,
            "results": [r.to_dict() for r in final_response.results],
            "total_hits": final_response.total_hits,
            "processing_time_ms": processing_time,
            "auto_crawled": crawled_new,
            "crawl_info": crawl_info,
            "message": f"Found {final_response.total_hits} results" +
                      (f" (crawled {crawl_info['pages_indexed']} new pages)" if crawl_info else "")
        }

    except Exception as e:
        logger.error("Smart search failed", query=request.query, error=str(e))
        raise HTTPException(status_code=500, detail=f"Smart search failed: {str(e)}")


# ============================================================================
# SEED DATABASE & CATEGORY-BASED DYNAMIC CRAWLING
# ============================================================================

from app.engine.seed_database import (
    SEED_DATABASE,
    find_matching_categories,
    get_sites_for_query,
    get_all_categories,
    get_category_sites,
    get_database_stats
)


@app.get("/api/v2/engine/seed/stats")
async def get_seed_database_stats():
    """Get statistics about the seed database"""
    return get_database_stats()


@app.get("/api/v2/engine/seed/categories")
async def list_seed_categories():
    """List all available seed categories"""
    return {"categories": get_all_categories()}


@app.get("/api/v2/engine/seed/category/{category_name}")
async def get_seed_category(category_name: str):
    """Get sites for a specific category"""
    sites = get_category_sites(category_name)
    if not sites:
        raise HTTPException(status_code=404, detail=f"Category '{category_name}' not found")
    return {"category": category_name, "sites": sites}


@app.get("/api/v2/engine/seed/match")
async def match_query_to_categories(query: str, max_categories: int = 3):
    """Find matching categories for a query"""
    categories = find_matching_categories(query, max_categories)
    sites = get_sites_for_query(query, max_sites=10)

    return {
        "query": query,
        "matching_categories": categories,
        "suggested_sites": [
            {"url": s.url, "name": s.name, "description": s.description}
            for s in sites
        ]
    }


class DynamicCrawlRequest(BaseModel):
    """Request for dynamic category-based crawling"""
    query: str = Field(..., min_length=1, max_length=500)
    max_sites: int = Field(default=5, ge=1, le=20)
    max_pages_per_site: int = Field(default=50, ge=10, le=500)
    background: bool = Field(default=True, description="Run crawl in background")


@app.post("/api/v2/engine/crawl/dynamic")
async def dynamic_crawl_by_query(request: DynamicCrawlRequest):
    """
    Dynamic crawling based on query topic detection

    1. Analyzes the query to detect topic/category
    2. Selects appropriate seed sites from the database
    3. Starts crawling those sites
    4. Returns job info or results
    """
    # Find matching categories and sites
    categories = find_matching_categories(request.query, max_categories=3)
    sites = get_sites_for_query(request.query, max_sites=request.max_sites)

    if not sites:
        raise HTTPException(
            status_code=400,
            detail=f"No relevant sites found for query: '{request.query}'. Try a more specific topic."
        )

    # Prepare seed URLs
    seed_urls = [site.url for site in sites]

    job_id = f"dynamic-{uuid.uuid4().hex[:12]}"

    # Store job info
    crawl_jobs[job_id] = {
        "id": job_id,
        "type": "dynamic",
        "status": "queued",
        "query": request.query,
        "categories": categories,
        "seed_urls": seed_urls,
        "sites": [{"name": s.name, "url": s.url} for s in sites],
        "max_pages_per_site": request.max_pages_per_site,
        "pages_crawled": 0,
        "pages_indexed": 0,
        "errors": 0,
        "started_at": None,
        "completed_at": None,
        "created_at": datetime.utcnow().isoformat()
    }

    if request.background:
        # Start in background
        asyncio.create_task(run_dynamic_crawl_job(job_id, sites, request.max_pages_per_site))

        return {
            "job_id": job_id,
            "status": "started",
            "query": request.query,
            "categories": categories,
            "sites_to_crawl": len(sites),
            "message": f"Dynamic crawl started for {len(sites)} sites matching '{request.query}'"
        }
    else:
        # Run synchronously (blocking)
        await run_dynamic_crawl_job(job_id, sites, request.max_pages_per_site)
        return crawl_jobs[job_id]


async def run_dynamic_crawl_job(job_id: str, sites, max_pages_per_site: int):
    """Background task for dynamic crawl job"""
    try:
        crawl_jobs[job_id]["status"] = "running"
        crawl_jobs[job_id]["started_at"] = datetime.utcnow().isoformat()

        meili_host = "http://meilisearch:7700" if config.environment == "production" else "http://localhost:7700"
        redis_url = config.redis_url if hasattr(config, 'redis_url') else "redis://localhost:6379/2"

        index_config = IndexConfig(
            host=meili_host,
            api_key=config.meili_master_key if hasattr(config, 'meili_master_key') else "xeno_search_master_key_change_me"
        )
        indexer = XenoIndexer(index_config)
        await indexer.initialize()

        all_results = []

        # Crawl each site
        for site in sites:
            try:
                logger.info("Dynamic crawl: starting site", site=site.name, url=site.url)

                crawl_config = CrawlConfig(
                    max_pages_per_domain=min(site.max_pages, max_pages_per_site),
                    max_depth=5,
                    follow_external_links=False,
                    delay_between_requests=1.0,
                    respect_robots_txt=True
                )

                crawler = XenoCrawler(redis_url=redis_url, config=crawl_config)

                results = await crawler.crawl(
                    seed_urls=[site.url],
                    max_pages=min(site.max_pages, max_pages_per_site)
                )

                if results:
                    all_results.extend(results)
                    crawl_jobs[job_id]["pages_crawled"] += len(results)

                    # Index immediately
                    calculator = PageRankCalculator()
                    page_ranks = calculator.calculate(results)
                    indexed = await indexer.index_pages(results, page_ranks)
                    crawl_jobs[job_id]["pages_indexed"] += indexed

                logger.info("Dynamic crawl: site completed",
                           site=site.name,
                           pages=len(results) if results else 0)

            except Exception as e:
                logger.error("Dynamic crawl: site failed", site=site.name, error=str(e))
                crawl_jobs[job_id]["errors"] += 1

        crawl_jobs[job_id]["status"] = "completed"
        crawl_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()

        logger.info("Dynamic crawl job completed",
                   job_id=job_id,
                   total_pages=crawl_jobs[job_id]["pages_crawled"],
                   total_indexed=crawl_jobs[job_id]["pages_indexed"])

    except Exception as e:
        logger.error("Dynamic crawl job failed", job_id=job_id, error=str(e))
        crawl_jobs[job_id]["status"] = "failed"
        crawl_jobs[job_id]["error"] = str(e)
        crawl_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()


class TopicSearchRequest(BaseModel):
    """Request for topic-aware search with auto-crawl"""
    query: str = Field(..., min_length=1, max_length=500)
    auto_crawl_if_empty: bool = Field(default=True)
    min_results_threshold: int = Field(default=3)
    max_crawl_sites: int = Field(default=3)
    max_pages_per_site: int = Field(default=30)


def check_result_relevance(query: str, results: list, min_score: float = 0.3) -> tuple[bool, float]:
    """
    Check if search results are actually relevant to the query.
    Returns (is_relevant, avg_score).

    A result is considered relevant if:
    1. Average score is above threshold
    2. Query terms appear meaningfully in results (not just incidental mentions)
    """
    if not results:
        return False, 0.0

    query_words = set(query.lower().split())
    # Remove common words that might match incidentally
    stop_words = {'what', 'is', 'the', 'a', 'an', 'in', 'are', 'we', 'how', 'why', 'when', 'where', 'do', 'does', 'can', 'will', 'be', 'to', 'of', 'for', 'on', 'with', 'at', 'by', 'from'}
    meaningful_words = query_words - stop_words

    total_score = 0.0
    meaningful_matches = 0

    for result in results:
        # Get the score from the result
        score = getattr(result, 'score', 0) or 0
        total_score += score

        # Check if meaningful query words appear in title or content
        title = (getattr(result, 'title', '') or '').lower()
        content = (getattr(result, 'content', '') or getattr(result, 'snippet', '') or '').lower()
        combined = title + ' ' + content

        # Check for meaningful word presence
        for word in meaningful_words:
            if len(word) > 2 and word in combined:
                meaningful_matches += 1
                break

    avg_score = total_score / len(results) if results else 0.0
    relevance_ratio = meaningful_matches / len(results) if results else 0.0

    # Results are relevant if average score is good AND meaningful words are found
    is_relevant = avg_score >= min_score and relevance_ratio >= 0.3

    return is_relevant, avg_score


@app.post("/api/v2/engine/topic-search")
async def topic_aware_search(request: TopicSearchRequest):
    """
    Topic-aware search using ONLY the Xeno Search Engine

    1. Search existing index
    2. Check result RELEVANCE (not just count)
    3. If results are irrelevant, detect topic from query
    4. Find relevant seed sites from our database
    5. Crawl and index them
    6. Return results from newly indexed content

    NO external search APIs - only our own crawler and seed database!
    """
    start_time = time.time()

    try:
        engine = await get_xeno_search_engine()

        # First, search existing index
        initial_response = await engine.search(
            query=request.query,
            per_page=10
        )

        initial_results = len(initial_response.results)
        crawl_info = None
        categories_matched = []

        # Check result relevance, not just count
        is_relevant, avg_score = check_result_relevance(
            request.query,
            initial_response.results,
            min_score=0.3
        )

        # Trigger crawl if: not enough results OR results are not relevant
        needs_crawl = (
            initial_results < request.min_results_threshold or
            (initial_results > 0 and not is_relevant)
        )

        if needs_crawl and request.auto_crawl_if_empty:
            logger.info("Topic search: triggering auto-crawl",
                       query=request.query,
                       initial_results=initial_results,
                       is_relevant=is_relevant,
                       avg_score=avg_score,
                       reason="low_count" if initial_results < request.min_results_threshold else "low_relevance")

            # Find matching categories and sites
            categories_matched = find_matching_categories(request.query, max_categories=2)
            sites = get_sites_for_query(request.query, max_sites=request.max_crawl_sites)

            if sites:
                meili_host = "http://meilisearch:7700" if config.environment == "production" else "http://localhost:7700"
                redis_url = config.redis_url if hasattr(config, 'redis_url') else "redis://localhost:6379/2"

                index_config = IndexConfig(
                    host=meili_host,
                    api_key=config.meili_master_key if hasattr(config, 'meili_master_key') else "xeno_search_master_key_change_me"
                )
                indexer = XenoIndexer(index_config)
                await indexer.initialize()

                total_crawled = 0
                total_indexed = 0
                sites_crawled = []

                for site in sites:
                    try:
                        crawl_config = CrawlConfig(
                            max_pages_per_domain=request.max_pages_per_site,
                            max_depth=3,
                            follow_external_links=False,
                            delay_between_requests=0.5,  # Faster for topic search
                            respect_robots_txt=True
                        )

                        crawler = XenoCrawler(redis_url=redis_url, config=crawl_config)

                        results = await crawler.crawl(
                            seed_urls=[site.url],
                            max_pages=request.max_pages_per_site
                        )

                        if results:
                            calculator = PageRankCalculator()
                            page_ranks = calculator.calculate(results)
                            indexed = await indexer.index_pages(results, page_ranks)

                            total_crawled += len(results)
                            total_indexed += indexed
                            sites_crawled.append({
                                "name": site.name,
                                "url": site.url,
                                "pages": len(results)
                            })

                    except Exception as e:
                        logger.warning("Topic search: site crawl failed",
                                      site=site.name, error=str(e))

                if total_indexed > 0:
                    crawl_info = {
                        "categories": categories_matched,
                        "sites_crawled": sites_crawled,
                        "total_pages_crawled": total_crawled,
                        "total_pages_indexed": total_indexed
                    }

                    # Wait for indexing
                    await asyncio.sleep(0.5)

                    logger.info("Topic search: crawl completed",
                               query=request.query,
                               pages_indexed=total_indexed)

        # Search again with new content
        final_response = await engine.search(
            query=request.query,
            per_page=10
        )

        processing_time = int((time.time() - start_time) * 1000)

        return {
            "query": request.query,
            "results": [r.to_dict() for r in final_response.results],
            "total_hits": final_response.total_hits,
            "processing_time_ms": processing_time,
            "categories_detected": categories_matched,
            "crawl_info": crawl_info,
            "message": f"Found {final_response.total_hits} results" +
                      (f" (auto-crawled {crawl_info['total_pages_indexed']} pages from {len(crawl_info['sites_crawled'])} sites)"
                       if crawl_info else " from existing index")
        }

    except Exception as e:
        logger.error("Topic search failed", query=request.query, error=str(e))
        raise HTTPException(status_code=500, detail=f"Topic search failed: {str(e)}")


# ============================================================================
# STREAMING TOPIC SEARCH WITH WEBSOCKET PROGRESS
# ============================================================================

@app.websocket("/ws/engine/topic-search/{search_id}")
async def topic_search_websocket(websocket: WebSocket, search_id: str):
    """WebSocket endpoint for real-time topic search progress"""
    connection_id = f"ws-topic-{uuid.uuid4().hex[:8]}"

    await websocket_manager.connect(websocket, connection_id, search_id)

    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"type": "ping", "data": "pong"})
    except WebSocketDisconnect:
        websocket_manager.disconnect(connection_id, search_id)


@app.post("/api/v2/engine/topic-search/streaming")
async def streaming_topic_search(request: TopicSearchRequest):
    """Start a streaming topic search with WebSocket progress updates"""
    search_id = f"topic-{uuid.uuid4().hex[:12]}"

    # Start search in background
    asyncio.create_task(perform_streaming_topic_search(search_id, request))

    return {
        "search_id": search_id,
        "websocket_url": f"/ws/engine/topic-search/{search_id}",
        "status": "started"
    }


async def perform_streaming_topic_search(search_id: str, request: TopicSearchRequest):
    """Perform topic search with real-time WebSocket updates"""
    start_time = time.time()

    try:
        await websocket_manager.send_progress_update(
            search_id, "initializing", 5, "🔍 Searching existing index..."
        )

        engine = await get_xeno_search_engine()

        # First search existing index
        initial_response = await engine.search(
            query=request.query,
            per_page=10
        )

        initial_results = len(initial_response.results)
        crawl_info = None
        categories_matched = []

        await websocket_manager.send_progress_update(
            search_id, "checking", 15,
            f"📊 Found {initial_results} existing results, checking relevance..."
        )

        # Check result relevance
        is_relevant, avg_score = check_result_relevance(
            request.query,
            initial_response.results,
            min_score=0.3
        )

        needs_crawl = (
            initial_results < request.min_results_threshold or
            (initial_results > 0 and not is_relevant)
        )

        if needs_crawl and request.auto_crawl_if_empty:
            await websocket_manager.send_progress_update(
                search_id, "detecting", 20,
                "🎯 Results not relevant enough, detecting topic..."
            )

            categories_matched = find_matching_categories(request.query, max_categories=2)
            sites = get_sites_for_query(request.query, max_sites=request.max_crawl_sites)

            if categories_matched:
                await websocket_manager.send_progress_update(
                    search_id, "topic_found", 25,
                    f"📂 Detected topic: {', '.join(categories_matched)}"
                )

            if sites:
                meili_host = "http://meilisearch:7700" if config.environment == "production" else "http://localhost:7700"
                redis_url = config.redis_url if hasattr(config, 'redis_url') else "redis://localhost:6379/2"

                index_config = IndexConfig(
                    host=meili_host,
                    api_key=config.meili_master_key if hasattr(config, 'meili_master_key') else "xeno_search_master_key_change_me"
                )
                indexer = XenoIndexer(index_config)
                await indexer.initialize()

                total_crawled = 0
                total_indexed = 0
                sites_crawled = []

                for i, site in enumerate(sites):
                    progress = 30 + int((i / len(sites)) * 50)  # 30-80%

                    await websocket_manager.send_progress_update(
                        search_id, "crawling", progress,
                        f"🌐 Crawling {site.name} ({i+1}/{len(sites)})..."
                    )

                    try:
                        crawl_config = CrawlConfig(
                            max_pages_per_domain=request.max_pages_per_site,
                            max_depth=2,  # Reduced depth for speed
                            follow_external_links=False,
                            delay_between_requests=0.3,  # Faster
                            respect_robots_txt=True
                        )

                        crawler = XenoCrawler(redis_url=redis_url, config=crawl_config)

                        results = await crawler.crawl(
                            seed_urls=[site.url],
                            max_pages=request.max_pages_per_site
                        )

                        if results:
                            calculator = PageRankCalculator()
                            page_ranks = calculator.calculate(results)
                            indexed = await indexer.index_pages(results, page_ranks)

                            total_crawled += len(results)
                            total_indexed += indexed
                            sites_crawled.append({
                                "name": site.name,
                                "url": site.url,
                                "pages": len(results)
                            })

                            await websocket_manager.send_progress_update(
                                search_id, "indexed", progress + 5,
                                f"✅ Indexed {indexed} pages from {site.name}"
                            )

                    except Exception as e:
                        logger.warning("Streaming topic search: site crawl failed",
                                      site=site.name, error=str(e))
                        await websocket_manager.send_progress_update(
                            search_id, "crawl_error", progress,
                            f"⚠️ Could not crawl {site.name}, continuing..."
                        )

                if total_indexed > 0:
                    crawl_info = {
                        "categories": categories_matched,
                        "sites_crawled": sites_crawled,
                        "total_pages_crawled": total_crawled,
                        "total_pages_indexed": total_indexed
                    }
                    await asyncio.sleep(0.5)  # Wait for indexing

        await websocket_manager.send_progress_update(
            search_id, "finalizing", 90, "🔎 Fetching final results..."
        )

        # Final search with new content
        final_response = await engine.search(
            query=request.query,
            per_page=10
        )

        processing_time = int((time.time() - start_time) * 1000)

        result = {
            "query": request.query,
            "results": [r.to_dict() for r in final_response.results],
            "total_hits": final_response.total_hits,
            "processing_time_ms": processing_time,
            "categories_detected": categories_matched,
            "crawl_info": crawl_info,
            "message": f"Found {final_response.total_hits} results" +
                      (f" (auto-crawled {crawl_info['total_pages_indexed']} pages from {len(crawl_info['sites_crawled'])} sites)"
                       if crawl_info else " from existing index")
        }

        await websocket_manager.send_progress_update(
            search_id, "completed", 100,
            f"✅ Search completed in {processing_time}ms!",
            {"results": result}
        )

    except Exception as e:
        logger.error("Streaming topic search failed", query=request.query, error=str(e))
        await websocket_manager.send_progress_update(
            search_id, "error", 0,
            f"❌ Search failed: {str(e)}"
        )


# ============================================================================
# EXTERNAL SEARCH ENGINES (Google & Brave)
# ============================================================================

class GoogleSearchRequest(BaseModel):
    """Request model for Google Custom Search"""
    query: str = Field(..., min_length=1, max_length=500)
    num_results: int = Field(default=10, ge=1, le=50)


class BraveSearchRequest(BaseModel):
    """Request model for Brave Search"""
    query: str = Field(..., min_length=1, max_length=500)
    count: int = Field(default=10, ge=1, le=50)


@app.post("/api/v2/engine/google-search")
async def google_search_endpoint(request: GoogleSearchRequest):
    """
    Search using Google Custom Search API

    Requires GOOGLE_API_KEY and GOOGLE_CX environment variables to be set.
    """
    start_time = time.time()

    if not config.google_api_key:
        raise HTTPException(
            status_code=500,
            detail="Google API key not configured. Set GOOGLE_API_KEY environment variable."
        )

    if not config.google_cx:
        raise HTTPException(
            status_code=500,
            detail="Google Custom Search Engine ID not configured. Set GOOGLE_CX environment variable."
        )

    try:
        # Google Custom Search API endpoint
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": config.google_api_key,
            "cx": config.google_cx,
            "q": request.query,
            "num": min(request.num_results, 10),  # Google CSE allows max 10 per request
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error("Google Search API error", status=response.status, error=error_text)
                    raise HTTPException(status_code=response.status, detail=f"Google API error: {error_text}")

                data = await response.json()

        # Parse Google Search results
        results = []
        items = data.get("items", [])

        for i, item in enumerate(items):
            results.append({
                "url": item.get("link", ""),
                "title": item.get("title", ""),
                "description": item.get("snippet", ""),
                "snippet": item.get("snippet", ""),
                "domain": urlparse(item.get("link", "")).netloc,
                "source": "google",
                "rank": i + 1,
            })

        processing_time_ms = int((time.time() - start_time) * 1000)

        logger.info("Google search completed",
                   query=request.query,
                   results_count=len(results),
                   processing_time_ms=processing_time_ms)

        return {
            "query": request.query,
            "results": results,
            "total_results": int(data.get("searchInformation", {}).get("totalResults", 0)),
            "processing_time_ms": processing_time_ms,
            "source": "google_custom_search"
        }

    except aiohttp.ClientError as e:
        logger.error("Google search network error", query=request.query, error=str(e))
        raise HTTPException(status_code=502, detail=f"Google search failed: {str(e)}")
    except Exception as e:
        logger.error("Google search failed", query=request.query, error=str(e))
        raise HTTPException(status_code=500, detail=f"Google search failed: {str(e)}")


@app.post("/api/v2/engine/brave-search")
async def brave_search_endpoint(request: BraveSearchRequest):
    """
    Search using Brave Search API

    Requires BRAVE_SEARCH_API_KEY environment variable to be set.
    """
    start_time = time.time()

    if not config.brave_search_api_key:
        raise HTTPException(
            status_code=500,
            detail="Brave Search API key not configured. Set BRAVE_SEARCH_API_KEY environment variable."
        )

    try:
        # Brave Search API endpoint
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": config.brave_search_api_key,
        }
        params = {
            "q": request.query,
            "count": request.count,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error("Brave Search API error", status=response.status, error=error_text)
                    raise HTTPException(status_code=response.status, detail=f"Brave API error: {error_text}")

                data = await response.json()

        # Parse Brave Search results
        results = []
        web_results = data.get("web", {}).get("results", [])

        for i, item in enumerate(web_results):
            results.append({
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "snippet": item.get("description", ""),
                "domain": urlparse(item.get("url", "")).netloc,
                "source": "brave",
                "rank": i + 1,
                "age": item.get("age"),  # Brave includes page age
                "language": item.get("language"),
            })

        processing_time_ms = int((time.time() - start_time) * 1000)

        logger.info("Brave search completed",
                   query=request.query,
                   results_count=len(results),
                   processing_time_ms=processing_time_ms)

        return {
            "query": request.query,
            "results": results,
            "total_results": len(results),
            "processing_time_ms": processing_time_ms,
            "source": "brave_search"
        }

    except aiohttp.ClientError as e:
        logger.error("Brave search network error", query=request.query, error=str(e))
        raise HTTPException(status_code=502, detail=f"Brave search failed: {str(e)}")
    except Exception as e:
        logger.error("Brave search failed", query=request.query, error=str(e))
        raise HTTPException(status_code=500, detail=f"Brave search failed: {str(e)}")


# ============================================================================
# END OF XENO SEARCH ENGINE API
# ============================================================================
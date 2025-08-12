import time
import uuid
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

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
from app.services.enhanced_search_aggregator import perform_enhanced_search
from app.services.enhanced_nlp_service import get_enhanced_nlp_service
from app.services.scraping_service import perform_search as legacy_search, _scrape_page_content
from app.services.deep_search_service import perform_deep_search_analysis

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
    """Simple in-memory IP-based rate limiting"""

    def __init__(self, app: FastAPI):
        self.app = app
        self.ip_requests: Dict[str, List[float]] = {}

    async def __call__(self, request: Request, call_next):
        if not config.rate_limiting_enabled:
            return await call_next(request)

        ip = request.client.host if request.client else "anonymous"
        now = time.time()

        minute_limit = config.rate_limit_requests_per_minute
        hour_limit = config.rate_limit_requests_per_hour

        timestamps = self.ip_requests.get(ip, [])
        timestamps = [t for t in timestamps if now - t < 3600]

        minute_count = len([t for t in timestamps if now - t < 60])

        if minute_count >= minute_limit or len(timestamps) >= hour_limit:
            return Response(status_code=429, content="Rate limit exceeded")

        timestamps.append(now)
        self.ip_requests[ip] = timestamps

        return await call_next(request)


app.add_middleware(RateLimitMiddleware)


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
            
            websocket = self.active_connections[connection_id]
            await websocket.send_json(update)
            
        except Exception as e:
            logger.warning("Failed to send WebSocket update", 
                         search_id=search_id, error=str(e))


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
            
            # Perform search based on type
            if request.search_type == SearchType.ENHANCED:
                sources = await perform_enhanced_search(request.query, request.num_results, language=request.language, time_range=request.time_range)
                search_stats.engines_used = list(config.search_engines)
                
            elif request.search_type == SearchType.DEEP:
                # First get initial results
                initial_sources = await perform_enhanced_search(request.query, request.num_results, language=request.language, time_range=request.time_range)
                # Then perform deep search
                if initial_sources:
                    all_sources, comprehensive_summary = await perform_deep_search_analysis(
                        request.query, initial_sources
                    )
                    sources = all_sources
                search_stats.engines_used = list(config.search_engines)
                
            elif request.search_type == SearchType.SEMANTIC:
                # Enhanced semantic search
                sources = await perform_enhanced_search(request.query, request.num_results, language=request.language, time_range=request.time_range)
                # Additional semantic ranking will be applied in the aggregator
                search_stats.engines_used = list(config.search_engines)
                
            else:
                # Fallback to legacy search for NORMAL type
                sources = await legacy_search(request.query, request.search_type.value, request.num_results)
                search_stats.engines_used = ["duckduckgo"]
            
            # Enhanced content processing
            if sources and request.include_summaries:
                await _process_sources_with_enhanced_nlp(sources, request.query)
                # Final content-aware rerank
                sources = await _final_rerank_with_content(request.query, sources)
            
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

async def _final_rerank_with_content(query: str, sources: List[Source]) -> List[Source]:
    """After NLP, rerank using semantic similarity on summaries/raw text and apply domain diversification."""
    try:
        if not sources:
            return sources
        nlp_service = await get_enhanced_nlp_service()
        texts = []
        for s in sources:
            if s.summary and s.summary.strip():
                texts.append(s.summary)
            elif s.snippet:
                texts.append(s.snippet)
            elif s.raw_text:
                texts.append(s.raw_text[:2000])
            else:
                texts.append("")
        sims = await nlp_service.calculate_semantic_similarity(query, texts)
        # Attach scores
        for s, sc in zip(sources, sims):
            if s.confidence_score is None:
                s.confidence_score = float(sc)
            else:
                s.confidence_score = float(0.6 * s.confidence_score + 0.4 * sc)
        # Simple domain diversification
        max_per_domain = getattr(config, 'max_results_per_domain', 2)
        domain_counts: Dict[str, int] = {}
        diversified: List[Source] = []
        for s in sorted(sources, key=lambda x: (x.confidence_score or 0), reverse=True):
            domain = s.get_domain() or ""
            if domain_counts.get(domain, 0) < max_per_domain:
                diversified.append(s)
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        return diversified
    except Exception as e:
        logger.warning("Final content-based rerank failed", error=str(e))
        return sources


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
            search_id, "initializing", 5, "🔄 Initializing enhanced search..."
        )
        
        await websocket_manager.send_progress_update(
            search_id, "searching", 20, "🔍 Searching multiple engines..."
        )
        
        # Perform the actual search
        sources = await perform_enhanced_search(request.query, request.num_results, language=request.language, time_range=request.time_range)
        
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
        "description": "Production-ready AI-powered search and analysis service",
        "status": "operational",
        "features": [
            "Multi-engine search aggregation",
            "Advanced AI summarization", 
            "Semantic search and ranking",
            "Deep link analysis",
            "Real-time progress updates",
            "Comprehensive content analysis",
            "Redis caching with deduplication",
            "Prometheus metrics and monitoring"
        ],
        "docs_url": "/docs" if config.debug else None,
        "health_url": "/health",
        "metrics_url": "/metrics"
    }

# To run this (after creating the directory structure and installing dependencies):
# Ensure you are in the 'xeno-search-service' directory.
# 1. Create and activate virtual environment (e.g., python -m venv venv; source venv/bin/activate or venv\Scripts\activate on Windows)
# 2. Install FastAPI and Uvicorn: pip install fastapi uvicorn[standard]
# 3. Run the server: uvicorn app.main:app --reload --port 8000
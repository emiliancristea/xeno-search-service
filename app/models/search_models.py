from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, HttpUrl, Field, validator
from datetime import datetime
from enum import Enum

class SearchType(str, Enum):
    """Enumeration of search types"""
    NORMAL = "normal"
    DEEP = "deep"
    ENHANCED = "enhanced"
    SEMANTIC = "semantic"

class ContentCategory(str, Enum):
    """Content categories for classification"""
    NEWS = "news"
    ACADEMIC = "academic"
    BLOG = "blog"
    DOCUMENTATION = "documentation"
    FORUM = "forum"
    SOCIAL = "social"
    COMMERCIAL = "commercial"
    GOVERNMENT = "government"
    REFERENCE = "reference"
    OTHER = "other"

class SourceMetadata(BaseModel):
    """Enhanced metadata for sources"""
    search_engine: Optional[str] = None
    engine_rank: Optional[int] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    semantic_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    content_category: Optional[ContentCategory] = None
    language: Optional[str] = None
    domain: Optional[str] = None
    publish_date: Optional[datetime] = None
    author: Optional[str] = None
    word_count: Optional[int] = None
    reading_time_minutes: Optional[int] = None
    is_duplicate: Optional[bool] = False
    original_url: Optional[str] = None  # For duplicate detection
    content_hash: Optional[str] = None
    extraction_method: Optional[str] = None  # e.g., "readability", "boilerplate"
    scraping_success: Optional[bool] = None
    scraping_error: Optional[str] = None
    cache_hit: Optional[bool] = None
    processing_time_ms: Optional[int] = None
    
    # Search engine specific metadata
    engine_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Quality indicators
    content_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    relevance_indicators: Dict[str, Any] = Field(default_factory=dict)

class Source(BaseModel):
    """Enhanced source model with comprehensive metadata"""
    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    raw_text: Optional[str] = None
    summary: Optional[str] = None
    
    # Enhanced metadata
    metadata: Optional[SourceMetadata] = None
    
    # Legacy support for existing code
    legacy_metadata: Dict[str, Any] = Field(default_factory=dict, alias="metadata_dict")
    
    # Confidence and quality scores
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Content classification
    content_category: Optional[ContentCategory] = None
    content_tags: List[str] = Field(default_factory=list)
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    scraped_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        
    @validator('metadata', pre=True, always=True)
    def ensure_metadata(cls, v):
        """Ensure metadata is always a SourceMetadata instance"""
        if v is None:
            return SourceMetadata()
        if isinstance(v, dict):
            return SourceMetadata(**v)
        return v
    
    @validator('confidence_score', 'quality_score', 'relevance_score')
    def validate_scores(cls, v):
        """Validate score values"""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Scores must be between 0.0 and 1.0")
        return v
    
    def get_domain(self) -> Optional[str]:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.url)
            return parsed.netloc.lower()
        except:
            return None
    
    def get_reading_time(self) -> Optional[int]:
        """Estimate reading time in minutes"""
        if not self.raw_text:
            return None
        
        # Average reading speed: 200 words per minute
        word_count = len(self.raw_text.split())
        return max(1, round(word_count / 200))
    
    def calculate_quality_score(self) -> float:
        """Calculate a composite quality score"""
        scores = []
        
        # Title quality
        if self.title:
            title_score = min(1.0, len(self.title) / 60)  # Optimal title length ~60 chars
            scores.append(title_score * 0.2)
        
        # Content length quality
        if self.raw_text:
            content_score = min(1.0, len(self.raw_text) / 2000)  # Good content ~2000 chars+
            scores.append(content_score * 0.3)
        
        # Summary quality
        if self.summary:
            summary_score = min(1.0, len(self.summary) / 150)  # Good summary ~150 chars
            scores.append(summary_score * 0.2)
        
        # Confidence score contribution
        if self.confidence_score:
            scores.append(self.confidence_score * 0.3)
        
        return sum(scores) if scores else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with all fields"""
        return self.dict(exclude_none=True)

class SearchRequest(BaseModel):
    """Enhanced search request model"""
    query: str = Field(..., min_length=1, max_length=500)
    search_type: SearchType = SearchType.NORMAL
    num_results: int = Field(default=10, ge=1, le=100)
    
    # Advanced search options
    language: Optional[str] = Field(None, regex=r'^[a-z]{2}$')  # ISO language code
    region: Optional[str] = None
    time_range: Optional[str] = None  # e.g., "past_day", "past_week", "past_month"
    safe_search: bool = True
    
    # Content filtering
    include_categories: List[ContentCategory] = Field(default_factory=list)
    exclude_categories: List[ContentCategory] = Field(default_factory=list)
    min_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Engine preferences
    preferred_engines: List[str] = Field(default_factory=list)
    exclude_engines: List[str] = Field(default_factory=list)
    
    # Deep search options
    deep_search_max_depth: Optional[int] = Field(None, ge=1, le=5)
    deep_search_max_links: Optional[int] = Field(None, ge=1, le=50)
    
    # Response options
    include_raw_text: bool = True
    include_summaries: bool = True
    include_metadata: bool = True
    include_confidence_scores: bool = True
    
    # Caching options
    use_cache: bool = True
    cache_ttl: Optional[int] = None  # Override default cache TTL
    
    @validator('query')
    def validate_query(cls, v):
        """Validate search query"""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class RelatedQuery(BaseModel):
    """Related query suggestion"""
    query: str
    confidence: float = Field(ge=0.0, le=1.0)
    category: Optional[str] = None

class SearchStats(BaseModel):
    """Search statistics and metrics"""
    total_results: int
    processing_time_ms: int
    engines_used: List[str]
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Quality metrics
    avg_confidence_score: Optional[float] = None
    avg_quality_score: Optional[float] = None
    
    # Content breakdown
    category_distribution: Dict[ContentCategory, int] = Field(default_factory=dict)
    language_distribution: Dict[str, int] = Field(default_factory=dict)
    domain_distribution: Dict[str, int] = Field(default_factory=dict)
    
    # Performance metrics
    search_engine_performance: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    scraping_success_rate: Optional[float] = None
    summarization_success_rate: Optional[float] = None

class SearchResponse(BaseModel):
    """Enhanced search response model"""
    query: str
    search_type: SearchType
    sources: List[Source]
    
    # Enhanced summary
    summary: Optional[str] = None
    summary_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Related queries and suggestions
    related_queries: List[RelatedQuery] = Field(default_factory=list)
    query_suggestions: List[str] = Field(default_factory=list)
    
    # Search statistics
    stats: Optional[SearchStats] = None
    
    # Response metadata
    response_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[int] = None
    
    # Pagination (for future use)
    page: int = 1
    per_page: int = 10
    total_pages: Optional[int] = None
    has_more: bool = False
    
    # Quality indicators
    overall_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    result_diversity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
    
    def calculate_overall_confidence(self) -> float:
        """Calculate overall confidence score"""
        if not self.sources:
            return 0.0
        
        confidence_scores = [
            s.confidence_score for s in self.sources 
            if s.confidence_score is not None
        ]
        
        if not confidence_scores:
            return 0.0
        
        # Weighted average (higher weight for top results)
        weights = [1.0 / (i + 1) for i in range(len(confidence_scores))]
        weighted_sum = sum(score * weight for score, weight in zip(confidence_scores, weights))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    def calculate_result_diversity(self) -> float:
        """Calculate diversity score based on domains and categories"""
        if not self.sources:
            return 0.0
        
        # Domain diversity
        domains = set()
        categories = set()
        
        for source in self.sources:
            if source.get_domain():
                domains.add(source.get_domain())
            if source.content_category:
                categories.add(source.content_category)
        
        # Normalize by total results
        domain_diversity = len(domains) / len(self.sources)
        category_diversity = len(categories) / len(self.sources) if categories else 0.0
        
        return (domain_diversity + category_diversity) / 2.0
    
    def get_top_domains(self, limit: int = 5) -> List[tuple]:
        """Get top domains by frequency"""
        domain_counts = {}
        
        for source in self.sources:
            domain = source.get_domain()
            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        return sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def filter_by_confidence(self, min_confidence: float) -> 'SearchResponse':
        """Return filtered response with sources above confidence threshold"""
        filtered_sources = [
            s for s in self.sources 
            if s.confidence_score is None or s.confidence_score >= min_confidence
        ]
        
        # Create new response with filtered sources
        return SearchResponse(
            query=self.query,
            search_type=self.search_type,
            sources=filtered_sources,
            summary=self.summary,
            summary_confidence=self.summary_confidence,
            related_queries=self.related_queries,
            query_suggestions=self.query_suggestions,
            stats=self.stats,
            response_id=self.response_id,
            timestamp=self.timestamp,
            processing_time_ms=self.processing_time_ms
        )
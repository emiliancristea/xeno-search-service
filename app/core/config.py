import os
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import json


class XenoConfig(BaseSettings):
    """
    Enhanced configuration for Xeno Search Service
    Supports all advanced features mentioned in the enhancement requirements
    """
    
    # === Basic Service Configuration ===
    service_name: str = Field(default="xeno-search-service", env="SERVICE_NAME")
    service_version: str = Field(default="2.0.0", env="SERVICE_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # === Server Configuration ===
    host: str = Field(default="127.0.0.1", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # === Search Engine Configuration ===  
    search_engines: List[str] = Field(
        default=["searxng", "duckduckgo", "brave"], 
        env="SEARCH_ENGINES"
    )
    searxng_instances: List[str] = Field(
        default=[
            "https://searx.tiekoetter.com",
            "https://searx.be",
            "https://searx.ninja"
        ],
        env="SEARXNG_INSTANCES"
    )
    duckduckgo_enabled: bool = Field(default=True, env="DUCKDUCKGO_ENABLED")
    brave_search_api_key: Optional[str] = Field(default=None, env="BRAVE_SEARCH_API_KEY")
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    google_cx: Optional[str] = Field(default=None, env="GOOGLE_CX")  # Custom Search Engine ID
    bing_api_key: Optional[str] = Field(default=None, env="BING_API_KEY")

    # === Search Configuration ===
    default_num_results: int = Field(default=10, env="DEFAULT_NUM_RESULTS")
    max_results_per_request: int = Field(default=50, env="MAX_RESULTS_PER_REQUEST")
    search_timeout: int = Field(default=30, env="SEARCH_TIMEOUT")
    
    # === Deep Search Configuration ===
    deep_search_enabled: bool = Field(default=True, env="DEEP_SEARCH_ENABLED")
    deep_search_max_depth: int = Field(default=3, env="DEEP_SEARCH_MAX_DEPTH")
    deep_search_max_links_per_level: int = Field(default=8, env="DEEP_SEARCH_MAX_LINKS_PER_LEVEL")
    deep_search_min_relevance_score: float = Field(default=0.25, env="DEEP_SEARCH_MIN_RELEVANCE_SCORE")
    deep_search_max_additional_sources: int = Field(default=15, env="DEEP_SEARCH_MAX_ADDITIONAL_SOURCES")
    
    # === NLP and AI Configuration ===
    summarization_model: str = Field(
        default="facebook/bart-large-cnn", 
        env="SUMMARIZATION_MODEL"
    )
    summarization_model_device: str = Field(default="auto", env="SUMMARIZATION_MODEL_DEVICE")
    summarization_max_length: int = Field(default=200, env="SUMMARIZATION_MAX_LENGTH")
    summarization_min_length: int = Field(default=50, env="SUMMARIZATION_MIN_LENGTH")
    
    # Semantic embedding model for enhanced link ranking
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )
    semantic_search_enabled: bool = Field(default=True, env="SEMANTIC_SEARCH_ENABLED")
    semantic_similarity_threshold: float = Field(default=0.3, env="SEMANTIC_SIMILARITY_THRESHOLD")
    
    # === Caching Configuration ===
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    cache_type: str = Field(default="redis", env="CACHE_TYPE")  # redis, memory, file
    
    # Redis configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_ssl: bool = Field(default=False, env="REDIS_SSL")
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    
    # Cache TTL settings (in seconds)
    cache_ttl_search_results: int = Field(default=3600, env="CACHE_TTL_SEARCH_RESULTS")  # 1 hour
    cache_ttl_page_content: int = Field(default=86400, env="CACHE_TTL_PAGE_CONTENT")  # 24 hours
    cache_ttl_summaries: int = Field(default=604800, env="CACHE_TTL_SUMMARIES")  # 7 days
    
    # === Scraping Configuration ===
    scraping_enabled: bool = Field(default=True, env="SCRAPING_ENABLED")
    scraping_timeout: int = Field(default=15, env="SCRAPING_TIMEOUT")
    scraping_max_retries: int = Field(default=3, env="SCRAPING_MAX_RETRIES")
    scraping_delay: float = Field(default=0.5, env="SCRAPING_DELAY")
    
    # User agent rotation
    user_agent_rotation: bool = Field(default=True, env="USER_AGENT_ROTATION")
    user_agents: List[str] = Field(
        default=[
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:122.0) Gecko/20100101 Firefox/122.0"
        ],
        env="USER_AGENTS"
    )
    
    # Proxy configuration
    proxy_enabled: bool = Field(default=False, env="PROXY_ENABLED")
    proxy_list: List[str] = Field(default=[], env="PROXY_LIST")
    proxy_rotation: bool = Field(default=True, env="PROXY_ROTATION")
    
    # === Content Processing Configuration ===
    content_deduplication: bool = Field(default=True, env="CONTENT_DEDUPLICATION")
    content_similarity_threshold: float = Field(default=0.8, env="CONTENT_SIMILARITY_THRESHOLD")
    content_min_length: int = Field(default=100, env="CONTENT_MIN_LENGTH")
    content_max_length: int = Field(default=50000, env="CONTENT_MAX_LENGTH")
    
    # === Response Enhancement Configuration ===
    include_confidence_scores: bool = Field(default=True, env="INCLUDE_CONFIDENCE_SCORES")
    include_content_categories: bool = Field(default=True, env="INCLUDE_CONTENT_CATEGORIES")
    include_source_metadata: bool = Field(default=True, env="INCLUDE_SOURCE_METADATA")
    include_related_queries: bool = Field(default=True, env="INCLUDE_RELATED_QUERIES")
    
    # === Monitoring and Logging Configuration ===
    logging_level: str = Field(default="INFO", env="LOGGING_LEVEL")
    logging_format: str = Field(default="json", env="LOGGING_FORMAT")  # json, text
    logging_file: Optional[str] = Field(default=None, env="LOGGING_FILE")
    
    # Metrics and monitoring
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    health_check_enabled: bool = Field(default=True, env="HEALTH_CHECK_ENABLED")
    
    # === Rate Limiting Configuration ===
    rate_limiting_enabled: bool = Field(default=True, env="RATE_LIMITING_ENABLED")
    rate_limit_requests_per_minute: int = Field(default=100, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    rate_limit_requests_per_hour: int = Field(default=1000, env="RATE_LIMIT_REQUESTS_PER_HOUR")
    
    # === Security Configuration ===
    api_key_required: bool = Field(default=False, env="API_KEY_REQUIRED")
    api_keys: List[str] = Field(default=[], env="API_KEYS")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # === Advanced Features ===
    enable_websocket_updates: bool = Field(default=True, env="ENABLE_WEBSOCKET_UPDATES")
    enable_streaming_responses: bool = Field(default=True, env="ENABLE_STREAMING_RESPONSES")
    enable_result_ranking: bool = Field(default=True, env="ENABLE_RESULT_RANKING")
    enable_query_expansion: bool = Field(default=True, env="ENABLE_QUERY_EXPANSION")
    
    # === Database Configuration (for persistent storage) ===
    database_enabled: bool = Field(default=False, env="DATABASE_ENABLED")
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")

    # === Meilisearch Configuration (for Xeno Search Engine) ===
    meili_host: str = Field(default="http://localhost:7700", env="MEILI_HOST")
    meili_master_key: str = Field(default="xeno_search_master_key_change_me", env="MEILI_MASTER_KEY")
    meili_index_name: str = Field(default="xeno_pages", env="MEILI_INDEX_NAME")
    
    @validator("search_engines", pre=True)
    def parse_search_engines(cls, v):
        if isinstance(v, str):
            return [engine.strip() for engine in v.split(",")]
        return v
    
    @validator("searxng_instances", pre=True)
    def parse_searxng_instances(cls, v):
        if isinstance(v, str):
            return [instance.strip() for instance in v.split(",")]
        return v
    
    @validator("user_agents", pre=True)
    def parse_user_agents(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [ua.strip() for ua in v.split("||")]
        return v
    
    @validator("proxy_list", pre=True)
    def parse_proxy_list(cls, v):
        if isinstance(v, str):
            return [proxy.strip() for proxy in v.split(",")]
        return v
    
    @validator("api_keys", pre=True)
    def parse_api_keys(cls, v):
        if isinstance(v, str):
            if not v or v.strip() == '' or v.strip() == '[]':
                return []
            return [key.strip() for key in v.split(",") if key.strip()]
        return v or []
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            if not v or v.strip() == '':
                return ["*"]
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v or ["*"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        if self.redis_url:
            return self.redis_url
        
        scheme = "rediss" if self.redis_ssl else "redis"
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"{scheme}://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    def get_search_engine_config(self) -> Dict[str, Any]:
        """Get search engine configuration"""
        return {
            "engines": self.search_engines,
            "searxng_instances": self.searxng_instances,
            "duckduckgo_enabled": self.duckduckgo_enabled,
            "brave_api_key": self.brave_search_api_key,
            "google_api_key": self.google_api_key,
            "google_cx": self.google_cx,
            "bing_api_key": self.bing_api_key,
            "timeout": self.search_timeout
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration"""
        return {
            "enabled": self.cache_enabled,
            "type": self.cache_type,
            "redis_url": self.get_redis_url(),
            "ttl": {
                "search_results": self.cache_ttl_search_results,
                "page_content": self.cache_ttl_page_content,
                "summaries": self.cache_ttl_summaries
            }
        }
    
    def get_nlp_config(self) -> Dict[str, Any]:
        """Get NLP configuration"""
        return {
            "summarization_model": self.summarization_model,
            "embedding_model": self.embedding_model,
            "device": self.summarization_model_device,
            "max_length": self.summarization_max_length,
            "min_length": self.summarization_min_length,
            "semantic_search": self.semantic_search_enabled,
            "similarity_threshold": self.semantic_similarity_threshold
        }


# Global configuration instance
config = XenoConfig()


def get_config() -> XenoConfig:
    """Get global configuration instance"""
    return config


def reload_config():
    """Reload configuration from environment"""
    global config
    config = XenoConfig()
    return config
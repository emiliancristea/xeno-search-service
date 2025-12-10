<div align="center">

# Xeno Search Service v2.0

### Enterprise-Grade AI-Powered Search & Analysis Platform

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Redis](https://img.shields.io/badge/Redis-7.0+-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)

*A production-ready, scalable search aggregation service featuring multi-engine orchestration, transformer-based NLP, real-time WebSocket streaming, and comprehensive observability — rivaling commercial solutions like Perplexity AI.*

[Features](#-key-features) • [Architecture](#-system-architecture) • [Quick Start](#-quick-start) • [API Reference](#-api-reference) • [Configuration](#%EF%B8%8F-configuration-reference) • [Deployment](#-production-deployment)

</div>

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [System Architecture](#-system-architecture)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Data Models](#-data-models)
- [Services & Components](#-services--components)
- [Configuration Reference](#%EF%B8%8F-configuration-reference)
- [Security Implementation](#-security-implementation)
- [Monitoring & Observability](#-monitoring--observability)
- [Production Deployment](#-production-deployment)
- [Performance Optimization](#-performance-optimization)
- [Testing](#-testing)
- [Technical Decisions](#-technical-decisions)
- [Roadmap](#-roadmap)

---

## ✨ Key Features

### Multi-Engine Search Aggregation
- **Concurrent Multi-Source Querying**: Parallel requests to SearXNG, DuckDuckGo, and Brave Search APIs
- **Intelligent Result Merging**: Weighted confidence scoring with engine-specific quality factors
- **Semantic Deduplication**: TF-IDF + cosine similarity-based duplicate detection (configurable threshold)
- **Result Diversification**: Domain and content category distribution optimization

### AI-Powered Content Analysis
- **Abstractive Summarization**: BART/Pegasus transformer models with GPU acceleration
- **Content Classification**: 10-category taxonomy (news, academic, blog, documentation, etc.)
- **Named Entity Recognition**: spaCy-powered extraction (persons, organizations, locations, dates)
- **Keyword Extraction**: TF-IDF vectorization with n-gram support
- **Semantic Similarity Scoring**: Sentence-transformer embeddings for relevance ranking

### Real-Time Capabilities
- **WebSocket Streaming**: Live progress updates during search operations
- **Progressive Result Delivery**: Incremental result streaming as sources are processed
- **Connection Management**: Automatic reconnection and dead connection cleanup

### Enterprise-Grade Infrastructure
- **Multi-Tier Caching**: Redis/Memory/File backends with per-key TTL support
- **SSRF Protection**: Comprehensive URL validation blocking private IPs and cloud metadata
- **Rate Limiting**: Per-IP throttling with automatic cleanup and memory bounds
- **Prometheus Metrics**: 15+ custom metrics for HTTP, search, NLP, and cache operations
- **Structured Logging**: JSON-formatted logs with correlation IDs via structlog

### Production Readiness
- **Docker & Docker Compose**: Multi-stage builds with non-root user execution
- **Health Checks**: Multi-component health monitoring with automatic degradation
- **Graceful Shutdown**: Proper cleanup of background tasks and connections
- **100+ Configuration Options**: Fine-grained control over every system aspect

---

## 🛠 Technology Stack

### Core Framework
| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.11+ | Runtime environment |
| **FastAPI** | 0.104+ | Async REST API framework with automatic OpenAPI docs |
| **Uvicorn** | 0.24+ | ASGI server with HTTP/2 and WebSocket support |
| **Pydantic** | 2.5+ | Data validation with JSON Schema generation |

### Search & Web Scraping
| Technology | Purpose |
|------------|---------|
| **httpx** | Async HTTP client with connection pooling |
| **aiohttp** | Alternative async HTTP for specific integrations |
| **BeautifulSoup4** | HTML/XML parsing and content extraction |
| **lxml** | High-performance XML/HTML processing |
| **readability-lxml** | Article extraction algorithm (Mozilla Readability port) |

### Machine Learning & NLP
| Technology | Version | Purpose |
|------------|---------|---------|
| **PyTorch** | 2.0+ | Deep learning framework with CUDA/MPS support |
| **Transformers** | 4.35+ | BART, Pegasus, and custom model loading |
| **Sentence-Transformers** | 2.2+ | Semantic similarity embeddings |
| **spaCy** | 3.7+ | Named entity recognition and linguistic analysis |
| **scikit-learn** | 1.3+ | TF-IDF vectorization and clustering |
| **langdetect** | 1.0+ | Language identification |

### Data Processing
| Technology | Purpose |
|------------|---------|
| **pandas** | Structured data manipulation |
| **numpy** | Numerical computations for ML pipelines |

### Caching & Storage
| Technology | Version | Purpose |
|------------|---------|---------|
| **Redis** | 5.0+ | Distributed caching with pub/sub |
| **redis-py** | 5.0+ | Async Redis client (replaces deprecated aioredis) |
| **cachetools** | 5.3+ | In-memory LRU/TTL cache implementations |

### Monitoring & Observability
| Technology | Purpose |
|------------|---------|
| **structlog** | Structured JSON logging with context propagation |
| **prometheus-client** | Metrics exposition in Prometheus format |
| **OpenTelemetry** | Distributed tracing (prepared for integration) |
| **psutil** | System resource monitoring |

### Development & Testing
| Technology | Purpose |
|------------|---------|
| **pytest** | Test framework with async support |
| **pytest-asyncio** | Async test execution |
| **pytest-cov** | Coverage reporting |

### DevOps & Deployment
| Technology | Purpose |
|------------|---------|
| **Docker** | Containerization with multi-stage builds |
| **Docker Compose** | Multi-service orchestration |
| **Prometheus** | Metrics collection and alerting |
| **Grafana** | Metrics visualization dashboards |

---

## 🏗 System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                   CLIENT LAYER                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   REST API   │  │  WebSocket   │  │  Prometheus  │  │   Health     │         │
│  │   Clients    │  │   Clients    │  │   Scraper    │  │   Probes     │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
└─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┘
          │                 │                 │                 │
          ▼                 ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 API GATEWAY LAYER                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                           FastAPI Application                            │    │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │    │
│  │  │    CORS     │ │    Rate     │ │    Auth     │ │  Monitoring │       │    │
│  │  │ Middleware  │ │  Limiter    │ │ Middleware  │ │  Middleware │       │    │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘       │    │
│  │                                                                         │    │
│  │  ┌─────────────────────────────────────────────────────────────┐       │    │
│  │  │                    Route Handlers                            │       │    │
│  │  │  • POST /api/v2/search      • GET /health                   │       │    │
│  │  │  • POST /api/v2/analyze     • GET /metrics                  │       │    │
│  │  │  • WS /ws/search/{id}       • GET /api/v2/analytics         │       │    │
│  │  └─────────────────────────────────────────────────────────────┘       │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
          │                                                     │
          ▼                                                     ▼
┌─────────────────────────────────────────────┐  ┌─────────────────────────────────┐
│           SERVICE LAYER                      │  │      INFRASTRUCTURE LAYER       │
│  ┌───────────────────────────────────────┐  │  │  ┌─────────────────────────┐    │
│  │      Enhanced Search Aggregator       │  │  │  │    Cache Manager        │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ │  │  │  │  ┌─────┐ ┌─────┐ ┌────┐│    │
│  │  │ SearXNG │ │  DDG    │ │  Brave  │ │  │  │  │  │Redis│ │Memory│ │File││    │
│  │  │ Engine  │ │ Engine  │ │ Engine  │ │  │  │  │  └─────┘ └─────┘ └────┘│    │
│  │  └─────────┘ └─────────┘ └─────────┘ │  │  │  └─────────────────────────┘    │
│  │  ┌───────────────────────────────────┐│  │  │  ┌─────────────────────────┐    │
│  │  │   Result Merger & Deduplicator   ││  │  │  │   Security Module       │    │
│  │  └───────────────────────────────────┘│  │  │  │  • SSRF Protection     │    │
│  └───────────────────────────────────────┘  │  │  │  • URL Validation      │    │
│  ┌───────────────────────────────────────┐  │  │  │  • Input Sanitization  │    │
│  │        Enhanced NLP Service           │  │  │  └─────────────────────────┘    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ │  │  │  ┌─────────────────────────┐    │
│  │  │Summary- │ │Classifi-│ │Semantic │ │  │  │  │   Monitoring System     │    │
│  │  │ ization │ │ cation  │ │Similarity│ │  │  │  │  • Prometheus Metrics  │    │
│  │  └─────────┘ └─────────┘ └─────────┘ │  │  │  │  • Health Checks       │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ │  │  │  │  • Performance Logs    │    │
│  │  │ Entity  │ │ Keyword │ │Language │ │  │  │  └─────────────────────────┘    │
│  │  │  NER    │ │TF-IDF   │ │Detection│ │  │  │                                 │
│  │  └─────────┘ └─────────┘ └─────────┘ │  │  │                                 │
│  └───────────────────────────────────────┘  │  │                                 │
│  ┌───────────────────────────────────────┐  │  │                                 │
│  │         Deep Search Service           │  │  │                                 │
│  │  • Recursive Link Following           │  │  │                                 │
│  │  • Relevance Scoring                  │  │  │                                 │
│  │  • Content Aggregation                │  │  │                                 │
│  └───────────────────────────────────────┘  │  │                                 │
└─────────────────────────────────────────────┘  └─────────────────────────────────┘
          │                                                     │
          ▼                                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              EXTERNAL SERVICES                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   SearXNG    │  │  DuckDuckGo  │  │ Brave Search │  │    Redis     │         │
│  │  Instances   │  │   HTML API   │  │     API      │  │   Cluster    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Request Flow: Search Operation

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           SEARCH REQUEST FLOW                                 │
└──────────────────────────────────────────────────────────────────────────────┘

    Client                API              Aggregator           NLP Service
       │                   │                   │                    │
       │  POST /search     │                   │                    │
       │──────────────────>│                   │                    │
       │                   │                   │                    │
       │                   │ Check Cache       │                    │
       │                   │───────────────────│                    │
       │                   │                   │                    │
       │                   │ Cache Miss        │                    │
       │                   │<──────────────────│                    │
       │                   │                   │                    │
       │                   │ Concurrent Search │                    │
       │                   │──────────────────>│                    │
       │                   │                   │                    │
       │                   │                   │ ┌────────────────┐ │
       │                   │                   │ │ SearXNG Query  │ │
       │                   │                   │ │ DDG Query      │ │
       │                   │                   │ │ Brave Query    │ │
       │                   │                   │ └────────────────┘ │
       │                   │                   │                    │
       │                   │                   │ Deduplicate        │
       │                   │                   │────────────────────│
       │                   │                   │                    │
       │                   │                   │ Semantic Ranking   │
       │                   │                   │───────────────────>│
       │                   │                   │                    │
       │                   │                   │    Embeddings      │
       │                   │                   │<───────────────────│
       │                   │                   │                    │
       │                   │                   │ Summarize Content  │
       │                   │                   │───────────────────>│
       │                   │                   │                    │
       │                   │                   │    Summaries       │
       │                   │                   │<───────────────────│
       │                   │                   │                    │
       │                   │  Ranked Results   │                    │
       │                   │<──────────────────│                    │
       │                   │                   │                    │
       │                   │ Cache Results     │                    │
       │                   │───────────────────│                    │
       │                   │                   │                    │
       │  JSON Response    │                   │                    │
       │<──────────────────│                   │                    │
       │                   │                   │                    │
```

### Caching Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CACHING ARCHITECTURE                               │
└─────────────────────────────────────────────────────────────────────────────┘

                        ┌─────────────────────────┐
                        │    Cache Manager        │
                        │  (EnhancedCacheManager) │
                        └───────────┬─────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │  Redis    │   │  Memory   │   │   File    │
            │  Backend  │   │  Backend  │   │  Backend  │
            └───────────┘   └───────────┘   └───────────┘
                 │               │               │
                 │               │               │
    ┌────────────┴────────────┐  │    ┌─────────┴─────────┐
    │ • Distributed           │  │    │ • Persistent      │
    │ • Connection Pooling    │  │    │ • SHA-256 Keys    │
    │ • Auto-reconnect        │  │    │ • Async I/O       │
    │ • TTL per key           │  │    │ • JSON Storage    │
    └─────────────────────────┘  │    └───────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ • LRU Eviction          │
                    │ • Per-key TTL           │
                    │ • Background Cleanup    │
                    │ • Max Size Enforcement  │
                    └─────────────────────────┘

    Cache Types & TTLs:
    ┌──────────────────────┬─────────────┬────────────────────┐
    │      Cache Type      │   Default   │     Key Pattern    │
    ├──────────────────────┼─────────────┼────────────────────┤
    │ Search Results       │   1 hour    │ search_results:*   │
    │ Page Content         │  24 hours   │ page_content:*     │
    │ AI Summaries         │   7 days    │ summary:*          │
    └──────────────────────┴─────────────┴────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (3.12 recommended)
- **Redis 7.0+** (optional but recommended for production)
- **8GB+ RAM** (for ML models)
- **NVIDIA GPU** (optional, for faster inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/emiliancristea/xeno-search-service.git
cd xeno-search-service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings
```

### Development Server

```bash
# Start Redis (Docker)
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Run development server
uvicorn app.main:app --reload --port 8000

# Access API documentation
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

### Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f xeno-search

# With monitoring stack
docker-compose --profile monitoring up -d

# With self-hosted SearXNG
docker-compose --profile searxng up -d
```

### First API Call

```bash
curl -X POST http://localhost:8000/api/v2/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest developments in artificial intelligence",
    "search_type": "enhanced",
    "num_results": 5,
    "include_summaries": true
  }'
```

---

## 📚 API Reference

### Core Endpoints

#### `POST /api/v2/search` - Enhanced Search

Performs multi-engine search with AI-powered analysis.

**Request Body:**
```json
{
  "query": "string (1-500 chars, required)",
  "search_type": "normal | enhanced | deep | semantic",
  "num_results": "integer (1-100, default: 10)",
  "language": "string (ISO 639-1, optional)",
  "region": "string (optional)",
  "time_range": "past_day | past_week | past_month | past_year (optional)",
  "safe_search": "boolean (default: true)",
  "include_categories": ["news", "academic", "blog", "..."],
  "exclude_categories": ["commercial", "..."],
  "min_quality_score": "float (0.0-1.0, optional)",
  "preferred_engines": ["searxng", "brave"],
  "exclude_engines": ["duckduckgo"],
  "deep_search_max_depth": "integer (1-5, optional)",
  "deep_search_max_links": "integer (1-50, optional)",
  "include_raw_text": "boolean (default: true)",
  "include_summaries": "boolean (default: true)",
  "include_metadata": "boolean (default: true)",
  "include_confidence_scores": "boolean (default: true)",
  "use_cache": "boolean (default: true)",
  "cache_ttl": "integer (seconds, optional override)"
}
```

**Response:**
```json
{
  "query": "latest developments in artificial intelligence",
  "search_type": "enhanced",
  "sources": [
    {
      "url": "https://example.com/ai-news",
      "title": "AI Breakthrough: New Model Achieves...",
      "snippet": "Researchers have developed...",
      "raw_text": "Full article content...",
      "summary": "AI-generated summary of the article...",
      "confidence_score": 0.92,
      "quality_score": 0.88,
      "relevance_score": 0.95,
      "content_category": "news",
      "content_tags": ["artificial intelligence", "machine learning", "research"],
      "metadata": {
        "search_engine": "brave",
        "engine_rank": 1,
        "semantic_score": 0.91,
        "language": "en",
        "domain": "example.com",
        "word_count": 1523,
        "reading_time_minutes": 6,
        "processing_time_ms": 234
      }
    }
  ],
  "summary": "Comprehensive summary of all search results...",
  "summary_confidence": 0.87,
  "related_queries": [
    {"query": "machine learning advances 2024", "confidence": 0.82},
    {"query": "GPT-5 release date", "confidence": 0.75}
  ],
  "query_suggestions": ["AI research papers", "deep learning trends"],
  "stats": {
    "total_results": 5,
    "processing_time_ms": 2847,
    "engines_used": ["searxng", "brave"],
    "cache_hits": 0,
    "cache_misses": 1,
    "avg_confidence_score": 0.89,
    "category_distribution": {"news": 3, "academic": 2},
    "scraping_success_rate": 1.0,
    "summarization_success_rate": 1.0
  },
  "response_id": "uuid-v4-identifier",
  "timestamp": "2024-01-15T10:30:00Z",
  "processing_time_ms": 2847,
  "overall_confidence": 0.89,
  "result_diversity_score": 0.73
}
```

**Status Codes:**
| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Invalid request parameters |
| 401 | API key required but not provided |
| 422 | Validation error |
| 429 | Rate limit exceeded |
| 500 | Internal server error |
| 503 | Service temporarily unavailable |

---

#### `POST /api/v2/search/streaming` - Real-Time Search

Initiates a search with WebSocket progress updates.

**Response:**
```json
{
  "search_id": "uuid-search-identifier",
  "websocket_url": "ws://localhost:8000/ws/search/{search_id}"
}
```

**WebSocket Messages:**
```json
{
  "search_id": "uuid",
  "type": "progress_update",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "phase": "searching | processing | finalizing | completed | error",
    "progress": 45,
    "message": "Processing source 3 of 10...",
    "data": {}
  }
}
```

---

#### `POST /api/v2/analyze` - Content Analysis

Analyzes URL content or raw text with full NLP pipeline.

**Query Parameters:**
- `url` (string, optional): URL to analyze
- `text` (string, optional): Raw text to analyze
- `include_summary` (boolean, default: true)
- `include_classification` (boolean, default: true)
- `include_entities` (boolean, default: true)
- `include_keywords` (boolean, default: true)

**Response:**
```json
{
  "url": "https://example.com/article",
  "analysis": {
    "text_length": 5234,
    "language": "en",
    "summary": {
      "text": "Article summary...",
      "confidence": 0.89,
      "compression_ratio": 0.12
    },
    "classification": {
      "category": "news",
      "confidence": 0.92,
      "all_scores": {
        "news": 0.92,
        "blog": 0.45,
        "academic": 0.23
      }
    },
    "entities": {
      "PERSON": ["Elon Musk", "Sam Altman"],
      "ORG": ["OpenAI", "Tesla"],
      "GPE": ["San Francisco", "United States"]
    },
    "keywords": ["artificial intelligence", "neural network", "transformer"]
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

#### `GET /health` - Health Check

Returns system health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "checks": {
    "memory": {
      "status": "healthy",
      "details": {"percent_used": 45.2}
    },
    "cache": {
      "status": "healthy",
      "details": {"connected": true}
    }
  }
}
```

---

#### `GET /metrics` - Prometheus Metrics

Returns metrics in Prometheus text format.

**Sample Metrics:**
```prometheus
# HELP xeno_http_requests_total Total HTTP requests
# TYPE xeno_http_requests_total counter
xeno_http_requests_total{method="POST",endpoint="/api/v2/search",status_code="200"} 1523

# HELP xeno_search_duration_seconds Search request duration
# TYPE xeno_search_duration_seconds histogram
xeno_search_duration_seconds_bucket{search_type="enhanced",le="1.0"} 234
xeno_search_duration_seconds_bucket{search_type="enhanced",le="5.0"} 1489

# HELP xeno_cache_hit_rate Cache hit rate
# TYPE xeno_cache_hit_rate gauge
xeno_cache_hit_rate 0.78

# HELP xeno_nlp_operations_total Total NLP operations
# TYPE xeno_nlp_operations_total counter
xeno_nlp_operations_total{operation="summarization",status="success"} 4521
```

---

#### `GET /api/v2/analytics/performance` - Performance Analytics

Returns detailed performance metrics.

**Query Parameters:**
- `operation` (string, optional): Filter by operation type
- `hours` (integer, default: 24): Time window

**Response:**
```json
{
  "total_operations": 15234,
  "successful_operations": 14892,
  "failed_operations": 342,
  "success_rate": 0.978,
  "performance": {
    "avg_duration": 2.34,
    "min_duration": 0.45,
    "max_duration": 12.3,
    "p50_duration": 2.1,
    "p95_duration": 4.8,
    "p99_duration": 8.2
  },
  "errors": {
    "TimeoutError": 234,
    "ConnectionError": 108
  }
}
```

---

## 📦 Data Models

### SearchType Enum
```python
class SearchType(str, Enum):
    NORMAL = "normal"      # Basic single-engine search
    ENHANCED = "enhanced"  # Multi-engine with ranking
    DEEP = "deep"          # Recursive link following
    SEMANTIC = "semantic"  # Embedding-based ranking
```

### ContentCategory Enum
```python
class ContentCategory(str, Enum):
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
```

### Source Model
```python
class Source(BaseModel):
    url: str                                    # Required
    title: Optional[str]
    snippet: Optional[str]                      # Brief excerpt
    raw_text: Optional[str]                     # Full content
    summary: Optional[str]                      # AI summary
    metadata: Optional[SourceMetadata]
    confidence_score: Optional[float]           # 0.0-1.0
    quality_score: Optional[float]              # 0.0-1.0
    relevance_score: Optional[float]            # 0.0-1.0
    content_category: Optional[ContentCategory]
    content_tags: List[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    scraped_at: Optional[datetime]
```

### SourceMetadata Model
```python
class SourceMetadata(BaseModel):
    search_engine: Optional[str]        # Which engine found it
    engine_rank: Optional[int]          # Position in results
    confidence_score: Optional[float]
    semantic_score: Optional[float]
    content_category: Optional[ContentCategory]
    language: Optional[str]             # ISO code
    domain: Optional[str]
    publish_date: Optional[datetime]
    author: Optional[str]
    word_count: Optional[int]
    reading_time_minutes: Optional[int]
    is_duplicate: bool = False
    original_url: Optional[str]         # If duplicate
    content_hash: Optional[str]         # For dedup
    extraction_method: Optional[str]    # readability, etc.
    scraping_success: Optional[bool]
    scraping_error: Optional[str]
    cache_hit: Optional[bool]
    processing_time_ms: Optional[int]
    engine_metadata: Dict[str, Any]
    content_quality_score: Optional[float]
    relevance_indicators: Dict[str, Any]
```

---

## 🔧 Services & Components

### Enhanced Search Aggregator

**Location:** `app/services/enhanced_search_aggregator.py`

Multi-engine search orchestration with intelligent result merging.

**Search Engines:**

| Engine | Method | Confidence | Weight | API Key |
|--------|--------|------------|--------|---------|
| SearXNG | Metasearch API | 0.8 | 1.0x | No |
| DuckDuckGo | HTML Scraping | 0.7 | 0.8x | No |
| Brave Search | Official API | 0.9 | 1.2x | Yes |

**Deduplication Pipeline:**
1. **URL Normalization**: Strip fragments, normalize case, remove trailing slashes
2. **Exact Match Removal**: Identical URLs after normalization
3. **Content Similarity**: TF-IDF + cosine similarity (threshold: 0.8)
4. **Keep Higher Confidence**: When duplicates found, retain best-scored result

**Ranking Algorithm:**
```
final_score = (engine_confidence × engine_weight × 0.6) + (semantic_similarity × 0.4)
```

---

### Enhanced NLP Service

**Location:** `app/services/enhanced_nlp_service.py`

Comprehensive NLP pipeline with transformer models.

**Capabilities:**

| Feature | Model/Method | Output |
|---------|--------------|--------|
| Summarization | BART-large-CNN / Pegasus | Text + confidence + compression ratio |
| Classification | Rule-based + ML | Category + confidence + all scores |
| Semantic Similarity | all-MiniLM-L6-v2 | Cosine similarity scores |
| Entity Extraction | spaCy en_core_web_sm | Named entities by type |
| Keyword Extraction | TF-IDF Vectorizer | Top keywords by score |
| Language Detection | langdetect | ISO 639-1 code |

**Device Support:**
- **CUDA**: NVIDIA GPU acceleration
- **MPS**: Apple Silicon (M1/M2/M3)
- **CPU**: Fallback for all systems

---

### Deep Search Service

**Location:** `app/services/deep_search_service.py`

Recursive link following with relevance scoring.

**Algorithm:**
1. Extract links from initial results
2. Score links by query term overlap (anchor text + context)
3. Follow top-ranked links (max depth: 3, max links: 8 per level)
4. Apply SSRF protection to all URLs
5. Aggregate and deduplicate all sources

---

### Cache Manager

**Location:** `app/core/cache.py`

Multi-backend caching with per-key TTL support.

**Backends:**

| Backend | Use Case | Features |
|---------|----------|----------|
| Redis | Production | Distributed, persistent, pub/sub |
| Memory | Development | Fast, per-key TTL, LRU eviction |
| File | Offline | JSON storage, persistent |

**Serialization:**
- **JSON-based** (replaced insecure pickle)
- Handles Pydantic models, datetime, complex types
- Graceful fallback for edge cases

---

### Security Module

**Location:** `app/core/security.py`

Comprehensive SSRF protection and input validation.

**Blocked Ranges:**
- Private IPv4: 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
- Loopback: 127.0.0.0/8, ::1/128
- Link-local: 169.254.0.0/16
- Cloud metadata: 169.254.169.254
- Reserved: 224.0.0.0/4 (multicast), 240.0.0.0/4

**Blocked Hostnames:**
- localhost, *.local, *.internal
- metadata.google.internal

---

### Monitoring System

**Location:** `app/core/monitoring.py`

Prometheus metrics and structured logging.

**Metrics Categories:**

| Category | Metrics |
|----------|---------|
| HTTP | requests_total, request_duration_seconds |
| Search | search_requests_total, search_duration_seconds, search_results_count |
| Scraping | scraping_requests_total, scraping_duration_seconds |
| NLP | nlp_operations_total, nlp_operation_duration_seconds |
| Cache | cache_operations_total, cache_hit_rate |
| System | active_connections, memory_usage_bytes, errors_total |

---

## ⚙️ Configuration Reference

### Environment Variables

<details>
<summary><b>Service Configuration</b></summary>

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SERVICE_NAME` | string | xeno-search-service | Service identifier |
| `SERVICE_VERSION` | string | 2.0.0 | Version number |
| `ENVIRONMENT` | string | development | development / production |
| `DEBUG` | boolean | false | Enable debug mode |
| `HOST` | string | 127.0.0.1 | Bind address |
| `PORT` | integer | 8000 | Server port |
| `WORKERS` | integer | 1 | Number of workers |

</details>

<details>
<summary><b>Search Engine Configuration</b></summary>

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SEARCH_ENGINES` | list | searxng,duckduckgo | Enabled engines |
| `SEARXNG_INSTANCES` | list | (multiple) | SearXNG instance URLs |
| `DUCKDUCKGO_ENABLED` | boolean | true | Enable DDG |
| `BRAVE_SEARCH_API_KEY` | string | - | Brave API key |
| `DEFAULT_NUM_RESULTS` | integer | 10 | Default result count |
| `MAX_RESULTS_PER_REQUEST` | integer | 50 | Maximum results |
| `SEARCH_TIMEOUT` | integer | 30 | Timeout seconds |

</details>

<details>
<summary><b>Deep Search Configuration</b></summary>

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DEEP_SEARCH_ENABLED` | boolean | true | Enable deep search |
| `DEEP_SEARCH_MAX_DEPTH` | integer | 3 | Maximum recursion depth |
| `DEEP_SEARCH_MAX_LINKS_PER_LEVEL` | integer | 8 | Links to follow per level |
| `DEEP_SEARCH_MIN_RELEVANCE_SCORE` | float | 0.25 | Minimum relevance threshold |
| `DEEP_SEARCH_MAX_ADDITIONAL_SOURCES` | integer | 15 | Max additional sources |

</details>

<details>
<summary><b>NLP & AI Configuration</b></summary>

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SUMMARIZATION_MODEL` | string | facebook/bart-large-cnn | HuggingFace model ID |
| `SUMMARIZATION_MODEL_DEVICE` | string | auto | auto / cpu / cuda / mps |
| `SUMMARIZATION_MAX_LENGTH` | integer | 200 | Max summary length |
| `SUMMARIZATION_MIN_LENGTH` | integer | 50 | Min summary length |
| `EMBEDDING_MODEL` | string | all-MiniLM-L6-v2 | Sentence transformer model |
| `SEMANTIC_SEARCH_ENABLED` | boolean | true | Enable semantic ranking |
| `SEMANTIC_SIMILARITY_THRESHOLD` | float | 0.3 | Similarity threshold |

</details>

<details>
<summary><b>Caching Configuration</b></summary>

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CACHE_ENABLED` | boolean | true | Enable caching |
| `CACHE_TYPE` | string | redis | redis / memory / file |
| `REDIS_HOST` | string | localhost | Redis hostname |
| `REDIS_PORT` | integer | 6379 | Redis port |
| `REDIS_DB` | integer | 0 | Redis database |
| `REDIS_PASSWORD` | string | - | Redis password |
| `REDIS_SSL` | boolean | false | Use SSL |
| `REDIS_URL` | string | - | Complete Redis URL |
| `CACHE_TTL_SEARCH_RESULTS` | integer | 3600 | Search cache TTL (1hr) |
| `CACHE_TTL_PAGE_CONTENT` | integer | 86400 | Content cache TTL (24hr) |
| `CACHE_TTL_SUMMARIES` | integer | 604800 | Summary cache TTL (7d) |

</details>

<details>
<summary><b>Security Configuration</b></summary>

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `API_KEY_REQUIRED` | boolean | false | Require API key |
| `API_KEYS` | list | - | Valid API keys (comma-separated) |
| `CORS_ORIGINS` | string | * | Allowed origins |
| `RATE_LIMITING_ENABLED` | boolean | true | Enable rate limiting |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | integer | 100 | Per-minute limit |
| `RATE_LIMIT_REQUESTS_PER_HOUR` | integer | 1000 | Per-hour limit |

</details>

<details>
<summary><b>Logging & Monitoring</b></summary>

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LOGGING_LEVEL` | string | INFO | DEBUG / INFO / WARNING / ERROR |
| `LOGGING_FORMAT` | string | json | json / text |
| `LOGGING_FILE` | string | - | Log file path |
| `METRICS_ENABLED` | boolean | true | Enable Prometheus metrics |
| `HEALTH_CHECK_ENABLED` | boolean | true | Enable health endpoint |

</details>

---

## 🛡 Security Implementation

### SSRF Protection

All URL operations are validated through the security module:

```python
from app.core.security import validate_url_for_ssrf

is_safe, error = validate_url_for_ssrf(url)
if not is_safe:
    raise ValueError(f"Blocked: {error}")
```

**Protection Layers:**
1. **Scheme Validation**: Only HTTP/HTTPS allowed
2. **IP Range Blocking**: Private, loopback, link-local, multicast
3. **Hostname Blocking**: localhost, *.local, *.internal
4. **DNS Resolution Check**: Resolves hostname and validates resulting IP
5. **Redirect Validation**: Each redirect URL is validated

### Rate Limiting

Per-IP rate limiting with automatic cleanup:

```python
class RateLimitMiddleware:
    # Per-minute and per-hour limits
    # Automatic old entry cleanup (1 hour window)
    # Max 10,000 tracked IPs
    # Thread-safe with asyncio.Lock
```

### API Authentication

Optional API key authentication via `X-API-Key` header:

```python
# .env
API_KEY_REQUIRED=true
API_KEYS=key1,key2,key3
```

### Input Sanitization

All user inputs are sanitized:

```python
from app.core.security import sanitize_search_query

clean_query = sanitize_search_query(user_input, max_length=500)
# Removes null bytes, control characters
# Enforces length limits
# Strips whitespace
```

---

## 📊 Monitoring & Observability

### Prometheus Metrics

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'xeno-search'
    static_configs:
      - targets: ['xeno-search:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Available Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `xeno_http_requests_total` | Counter | method, endpoint, status_code | Total HTTP requests |
| `xeno_http_request_duration_seconds` | Histogram | method, endpoint | Request latency |
| `xeno_search_requests_total` | Counter | search_type | Search operations |
| `xeno_search_duration_seconds` | Histogram | search_type | Search latency |
| `xeno_search_results_count` | Histogram | search_type | Results per search |
| `xeno_scraping_requests_total` | Counter | status | Page scrapes |
| `xeno_scraping_duration_seconds` | Histogram | - | Scrape latency |
| `xeno_nlp_operations_total` | Counter | operation, status | NLP operations |
| `xeno_nlp_operation_duration_seconds` | Histogram | operation | NLP latency |
| `xeno_cache_operations_total` | Counter | operation, status | Cache operations |
| `xeno_cache_hit_rate` | Gauge | - | Cache efficiency |
| `xeno_active_connections` | Gauge | - | WebSocket connections |
| `xeno_memory_usage_bytes` | Gauge | - | Memory consumption |
| `xeno_errors_total` | Counter | error_type, component | Error count |

### Structured Logging

JSON-formatted logs with context:

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "info",
  "logger": "app.services.enhanced_search_aggregator",
  "event": "Search completed",
  "query": "artificial intelligence",
  "sources_found": 10,
  "processing_time_ms": 2847,
  "cache_hit": false
}
```

### Grafana Dashboard

Import the included dashboard for visualization:
- Request rates and latencies
- Cache hit rates
- Error rates by type
- Search engine performance comparison
- NLP operation metrics

---

## 🚀 Production Deployment

### Docker Compose (Recommended)

```bash
# Production deployment with all services
docker-compose up -d

# Scale search service
docker-compose up -d --scale xeno-search=3

# With monitoring stack
docker-compose --profile monitoring up -d

# View service status
docker-compose ps
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: xeno-search
  labels:
    app: xeno-search
spec:
  replicas: 3
  selector:
    matchLabels:
      app: xeno-search
  template:
    metadata:
      labels:
        app: xeno-search
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: xeno-search
        image: xeno-search:v2.0.0
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: xeno-secrets
              key: redis-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: xeno-search-service
spec:
  selector:
    app: xeno-search
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: xeno-search-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: xeno-search
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Production Checklist

- [ ] Set `ENVIRONMENT=production`
- [ ] Configure Redis for distributed caching
- [ ] Set `LOGGING_LEVEL=WARNING`
- [ ] Enable `RATE_LIMITING_ENABLED=true`
- [ ] Configure `API_KEY_REQUIRED=true` if public
- [ ] Set specific `CORS_ORIGINS`
- [ ] Configure health check probes
- [ ] Set up Prometheus scraping
- [ ] Configure log aggregation
- [ ] Set resource limits
- [ ] Enable TLS termination at load balancer

---

## ⚡ Performance Optimization

### GPU Acceleration

```bash
# NVIDIA GPU
export SUMMARIZATION_MODEL_DEVICE=cuda

# Apple Silicon
export SUMMARIZATION_MODEL_DEVICE=mps
```

### Worker Configuration

```bash
# Production with multiple workers
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --loop uvloop \
  --http h11
```

### Caching Strategy

```env
# Aggressive caching for high-traffic
CACHE_ENABLED=true
CACHE_TYPE=redis
CACHE_TTL_SEARCH_RESULTS=7200    # 2 hours
CACHE_TTL_PAGE_CONTENT=172800    # 48 hours
CACHE_TTL_SUMMARIES=604800       # 7 days
CONTENT_DEDUPLICATION=true
```

### Resource Allocation

| Component | Recommended Memory | CPU |
|-----------|-------------------|-----|
| API Server | 512MB - 1GB | 0.5 - 1 core |
| ML Models | 2GB - 4GB | 1 - 2 cores |
| Redis | 512MB - 2GB | 0.5 core |
| Total (single instance) | 4GB - 8GB | 2 - 4 cores |

---

## 🧪 Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html

# Specific test file
pytest tests/test_security.py -v

# Specific test
pytest tests/test_models.py::TestSourceModel::test_create_minimal_source -v
```

### Test Structure

```
tests/
├── conftest.py          # Shared fixtures
├── test_security.py     # SSRF, URL validation
├── test_cache_new.py    # Cache backends
├── test_models.py       # Pydantic models
└── test_api.py          # API endpoints
```

### Manual Testing

```bash
# Health check
curl http://localhost:8000/health

# Basic search
curl -X POST http://localhost:8000/api/v2/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "num_results": 3}'

# Content analysis
curl -X POST "http://localhost:8000/api/v2/analyze?url=https://example.com"

# Metrics
curl http://localhost:8000/metrics
```

---

## 💡 Technical Decisions

### Why FastAPI?

- **Async-first**: Native async/await for concurrent I/O operations
- **Type hints**: Automatic request/response validation via Pydantic
- **Auto-documentation**: OpenAPI/Swagger UI out of the box
- **Performance**: One of the fastest Python frameworks available

### Why Redis for Caching?

- **Speed**: Sub-millisecond latency for cached results
- **Distributed**: Shared cache across multiple service instances
- **TTL Support**: Native key expiration
- **Persistence**: Optional AOF/RDB for cache survival

### Why JSON over Pickle?

- **Security**: Pickle allows arbitrary code execution
- **Interoperability**: JSON readable by any language
- **Debugging**: Human-readable cache contents
- **Compatibility**: No Python version dependency

### Why Sentence Transformers?

- **Quality**: State-of-the-art semantic similarity
- **Speed**: Optimized for inference
- **Size**: all-MiniLM-L6-v2 is only 90MB
- **Accuracy**: Excellent for search relevance ranking

### Why Multiple Search Engines?

- **Redundancy**: No single point of failure
- **Coverage**: Different engines surface different results
- **Quality**: Consensus ranking improves accuracy
- **Availability**: Continue operating if one engine fails

---

## 🗺 Roadmap

### v2.1 - Enhanced Intelligence
- [ ] Custom model fine-tuning pipeline
- [ ] Multi-language support (10+ languages)
- [ ] Query understanding and expansion
- [ ] Personalized ranking

### v2.2 - Scale & Performance
- [ ] Vector database integration (Pinecone/Weaviate)
- [ ] Distributed search architecture
- [ ] Result prefetching
- [ ] Advanced analytics dashboard

### v3.0 - Enterprise Features
- [ ] Multi-tenancy support
- [ ] SSO integration (SAML/OIDC)
- [ ] Custom search indices
- [ ] A/B testing framework
- [ ] GraphQL API

---

## 📁 Project Structure

```
xeno-search-service/
├── app/
│   ├── __init__.py
│   ├── main.py                           # FastAPI application, routes, middleware
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                     # Pydantic settings (100+ options)
│   │   ├── cache.py                      # Multi-backend cache system
│   │   ├── monitoring.py                 # Prometheus metrics, health checks
│   │   └── security.py                   # SSRF protection, URL validation
│   ├── models/
│   │   ├── __init__.py
│   │   └── search_models.py              # Pydantic request/response models
│   └── services/
│       ├── __init__.py
│       ├── enhanced_search_aggregator.py # Multi-engine search orchestration
│       ├── enhanced_nlp_service.py       # Transformer-based NLP pipeline
│       ├── scraping_service.py           # Web content extraction
│       ├── deep_search_service.py        # Recursive link analysis
│       └── nlp_service.py                # Legacy summarization (deprecated)
├── tests/
│   ├── conftest.py                       # Pytest fixtures
│   ├── test_security.py                  # Security module tests
│   ├── test_cache_new.py                 # Cache backend tests
│   ├── test_models.py                    # Model validation tests
│   └── test_api.py                       # API endpoint tests
├── .env.example                          # Configuration template
├── requirements.txt                      # Python dependencies
├── Dockerfile                            # Multi-stage production build
├── docker-compose.yml                    # Full stack deployment
├── .dockerignore                         # Build optimization
├── pytest.ini                            # Test configuration
└── README.md                             # This file
```

---

## 📄 License

This project is proprietary software. All rights reserved.

---

## 👤 Author

**Emilian Cristea**

- Portfolio: [emiliancristea.com](https://emiliancristea.com)
- LinkedIn: [linkedin.com/in/emiliancristea](https://linkedin.com/in/emiliancristea)
- GitHub: [@emiliancristea](https://github.com/emiliancristea)

---

<div align="center">

**Xeno Search Service v2.0**

*Enterprise-grade search intelligence, built with modern Python*

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type hints: mypy](https://img.shields.io/badge/type%20hints-mypy-blue.svg)](http://mypy-lang.org/)

</div>

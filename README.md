# 🔍 Xeno Search Service v2.0

**Production-ready AI-powered search and analysis service that rivals Perplexity**

A comprehensive, scalable search service featuring multi-engine aggregation, advanced AI summarization, semantic search, real-time analysis, and enterprise-grade monitoring.

## ✨ Key Features

### 🚀 **Advanced Search Capabilities**
- **Multi-Engine Aggregation**: SearXNG, DuckDuckGo, Brave Search with intelligent result merging
- **Semantic Search**: AI-powered relevance ranking using sentence transformers
- **Deep Search**: Recursive link following with intelligent content discovery
- **Real-time Updates**: WebSocket-based progress tracking for long-running searches

### 🤖 **AI-Powered Analysis**
- **Advanced Summarization**: Multiple summarization strategies with BART/Pegasus models
- **Content Classification**: Automatic categorization (news, academic, blog, etc.)
- **Entity Extraction**: Named entity recognition with spaCy
- **Keyword Analysis**: TF-IDF based keyword extraction
- **Semantic Similarity**: Vector-based content relevance scoring

### ⚡ **Performance & Scalability**
- **Redis Caching**: Intelligent caching with deduplication and TTL management
- **Concurrent Processing**: Parallel search engine queries and content processing
- **Rate Limiting**: Configurable request throttling
- **Load Balancing**: Multi-worker deployment support

### 📊 **Enterprise Monitoring**
- **Prometheus Metrics**: Comprehensive performance and usage metrics
- **Structured Logging**: JSON-formatted logs with context
- **Health Checks**: Multi-component health monitoring
- **Performance Analytics**: Detailed operation timing and success rates

### 🛡️ **Production Ready**
- **Comprehensive Configuration**: 100+ configurable parameters
- **Security Features**: Optional API key authentication and rate limiting
- **Error Handling**: Graceful degradation and detailed error reporting
- **Docker Support**: Container-ready with optimized builds

## 🏆 **Competitive Advantages vs Perplexity**

| Feature | Xeno Search v2.0 | Perplexity |
|---------|------------------|------------|
| **Search Engines** | Multiple (SearXNG, DDG, Brave) | Limited transparency |
| **Real-time Updates** | ✅ WebSocket streaming | ❌ |
| **Caching Layer** | ✅ Redis with deduplication | Unknown |
| **Content Analysis** | ✅ Full NLP pipeline | Limited |
| **Semantic Ranking** | ✅ Configurable models | ✅ |
| **Deep Search** | ✅ Recursive link following | Partial |
| **Monitoring** | ✅ Prometheus + metrics | ❌ |
| **Self-Hostable** | ✅ Complete control | ❌ |
| **API Transparency** | ✅ Open source | ❌ |
| **Custom Models** | ✅ Configurable AI models | ❌ |

## 🚀 **Quick Start**

### Prerequisites
- Python 3.9+
- Redis (optional, for caching)
- CUDA GPU (optional, for faster AI processing)

### Installation

1. **Clone and Setup**
```bash
git clone <repository-url>
cd xeno-search-service
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt

# Optional: Install spaCy language model for entity extraction
python -m spacy download en_core_web_sm
```

3. **Configuration**
```bash
# Copy example configuration
cp .env.example .env

# Edit configuration (see Configuration section below)
nano .env
```

4. **Start Redis (Optional but Recommended)**
```bash
# Using Docker
docker run -d --name redis -p 6379:6379 redis:alpine

# Or install locally
# Ubuntu/Debian: sudo apt install redis-server
# macOS: brew install redis
```

5. **Run the Service**
```bash
# Development
uvicorn app.main:app --reload --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment

```bash
# Build image
docker build -t xeno-search:v2 .

# Run with Redis
docker-compose up -d
```

## 📋 **API Documentation**

### **Enhanced Search Endpoint**
```http
POST /api/v2/search
Content-Type: application/json

{
  "query": "latest AI developments",
  "search_type": "enhanced",
  "num_results": 10,
  "include_summaries": true,
  "include_metadata": true,
  "min_quality_score": 0.7,
  "preferred_engines": ["searxng", "brave"]
}
```

**Response:**
```json
{
  "query": "latest AI developments",
  "search_type": "enhanced",
  "sources": [
    {
      "url": "https://example.com/ai-news",
      "title": "Latest AI Breakthroughs",
      "snippet": "Recent developments in AI...",
      "summary": "AI-generated summary of content...",
      "confidence_score": 0.95,
      "quality_score": 0.87,
      "content_category": "news",
      "content_tags": ["artificial intelligence", "machine learning"],
      "metadata": {
        "search_engine": "searxng",
        "semantic_score": 0.89,
        "word_count": 1500,
        "reading_time_minutes": 6
      }
    }
  ],
  "summary": "Comprehensive AI summary...",
  "related_queries": [
    {"query": "AI developments machine learning", "confidence": 0.8}
  ],
  "stats": {
    "total_results": 10,
    "processing_time_ms": 2450,
    "engines_used": ["searxng", "brave"],
    "avg_confidence_score": 0.82
  },
  "overall_confidence": 0.85,
  "result_diversity_score": 0.73
}
```

### **Search Types**

- **`normal`**: Basic search using legacy engine
- **`enhanced`**: Multi-engine search with AI ranking
- **`deep`**: Enhanced search + recursive link following
- **`semantic`**: Enhanced search with semantic similarity ranking

### **Real-time Search**
```http
POST /api/v2/search/streaming
```

Connect to WebSocket endpoint `/ws/search/{search_id}` for real-time updates.

### **Content Analysis**
```http
POST /api/v2/analyze
Content-Type: application/json

{
  "url": "https://example.com/article",
  "include_summary": true,
  "include_classification": true,
  "include_entities": true
}
```

### **Health & Monitoring**
```http
GET /health              # Service health check
GET /metrics             # Prometheus metrics
GET /api/v2/analytics/performance  # Performance analytics
```

## ⚙️ **Configuration**

The service is highly configurable via environment variables. Key configurations:

### **Search Engines**
```env
SEARCH_ENGINES=searxng,duckduckgo,brave
SEARXNG_INSTANCES=https://searx.be,https://searx.ninja
BRAVE_SEARCH_API_KEY=your_api_key_here
```

### **AI Models**
```env
SUMMARIZATION_MODEL=facebook/bart-large-cnn
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
SEMANTIC_SEARCH_ENABLED=true
```

### **Caching**
```env
CACHE_ENABLED=true
CACHE_TYPE=redis
REDIS_URL=redis://localhost:6379/0
CACHE_TTL_SEARCH_RESULTS=3600
```

### **Performance**
```env
DEEP_SEARCH_MAX_DEPTH=3
SCRAPING_TIMEOUT=15
CONTENT_DEDUPLICATION=true
```

See `.env.example` for complete configuration options.

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │  Search Engines  │    │   AI Services   │
│                 │    │                  │    │                 │
│ • REST API      │◄──►│ • SearXNG        │    │ • Summarization │
│ • WebSocket     │    │ • DuckDuckGo     │    │ • Classification│
│ • Monitoring    │    │ • Brave Search   │    │ • Embeddings    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Redis Cache    │    │   Aggregation    │    │   Deep Search   │
│                 │    │                  │    │                 │
│ • Search Cache  │    │ • Result Merge   │    │ • Link Following│
│ • Content Cache │    │ • Deduplication  │    │ • Content Anal. │
│ • Summary Cache │    │ • Ranking        │    │ • Semantic Rank │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔧 **Development**

### **Project Structure**
```
xeno-search-service/
├── app/
│   ├── core/                    # Core infrastructure
│   │   ├── config.py           # Configuration management
│   │   ├── cache.py            # Caching layer
│   │   └── monitoring.py       # Metrics & health checks
│   ├── services/               # Business logic
│   │   ├── enhanced_search_aggregator.py
│   │   ├── enhanced_nlp_service.py
│   │   ├── scraping_service.py
│   │   └── deep_search_service.py
│   ├── models/                 # Data models
│   │   └── search_models.py
│   └── main.py                 # FastAPI application
├── requirements.txt            # Dependencies
├── .env.example               # Configuration template
├── docker-compose.yml         # Docker setup
└── README.md                  # Documentation
```

### **Adding New Search Engines**

1. **Implement Engine Class**
```python
class NewSearchEngine(SearchEngineBase):
    async def search(self, query: str, num_results: int) -> List[SearchEngineResult]:
        # Implementation here
        pass
```

2. **Register in Aggregator**
```python
# In enhanced_search_aggregator.py
if "newsearch" in search_config['engines']:
    self.engines["newsearch"] = NewSearchEngine(config)
```

### **Custom AI Models**

Replace the default models in configuration:
```env
SUMMARIZATION_MODEL=your/custom-model
EMBEDDING_MODEL=your/embedding-model
```

### **Testing**

```bash
# Run basic tests
python -m pytest tests/

# Test specific endpoint
curl -X POST http://localhost:8000/api/v2/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "search_type": "enhanced"}'
```

## 🚀 **Production Deployment**

### **Docker Compose (Recommended)**

```yaml
version: '3.8'
services:
  xeno-search:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
  
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
```

### **Kubernetes Deployment**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: xeno-search
spec:
  replicas: 3
  selector:
    matchLabels:
      app: xeno-search
  template:
    metadata:
      labels:
        app: xeno-search
    spec:
      containers:
      - name: xeno-search
        image: xeno-search:v2
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
```

### **Performance Optimization**

1. **GPU Acceleration**
```env
SUMMARIZATION_MODEL_DEVICE=cuda
```

2. **Worker Scaling**
```bash
uvicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

3. **Caching Strategy**
```env
CACHE_TTL_SEARCH_RESULTS=3600
CACHE_TTL_PAGE_CONTENT=86400
CONTENT_DEDUPLICATION=true
```

## 📊 **Monitoring & Analytics**

### **Prometheus Metrics**

Key metrics exposed at `/metrics`:

- `xeno_search_requests_total` - Total search requests
- `xeno_search_duration_seconds` - Search processing time
- `xeno_cache_hit_rate` - Cache efficiency
- `xeno_nlp_operations_total` - AI processing metrics
- `xeno_errors_total` - Error tracking

### **Performance Dashboard**

Access analytics at `/api/v2/analytics/performance`:

```json
{
  "performance_summary": {
    "total_operations": 1500,
    "success_rate": 0.97,
    "avg_duration": 2.4,
    "p95_duration": 4.2
  },
  "cache_statistics": {
    "hit_rate": 0.78,
    "total_requests": 2300
  }
}
```

### **Health Monitoring**

Health check at `/health` validates:
- Memory usage
- Redis connectivity
- AI model availability
- Search engine status

## 🛡️ **Security**

### **API Key Authentication**
```env
API_KEY_REQUIRED=true
API_KEYS=your-secret-key-1,your-secret-key-2
```
When enabled, clients must include the `X-API-Key` header with one of the allowed keys.

### **Rate Limiting**
```env
RATE_LIMITING_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
```
Limits are applied per IP address to protect the service from abuse.

### **CORS Configuration**
```env
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### **Development Guidelines**

- Follow PEP 8 style guidelines
- Add comprehensive tests for new features
- Update documentation for API changes
- Ensure all health checks pass

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 **Support**

- **Documentation**: See `/docs` endpoint when running
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Community discussions via GitHub Discussions

## 🎯 **Roadmap**

### **v2.1 - Q1 2024**
- [ ] GraphQL API support
- [ ] Multi-language support
- [ ] Advanced result filtering
- [ ] Custom AI model fine-tuning

### **v2.2 - Q2 2024**
- [ ] Vector database integration
- [ ] Advanced analytics dashboard
- [ ] A/B testing framework
- [ ] Enterprise SSO integration

### **v3.0 - Q3 2024**
- [ ] Distributed search architecture
- [ ] Real-time indexing
- [ ] Machine learning ranking
- [ ] Advanced personalization

---

**Made with ❤️ for the open-source community**

*Xeno Search Service v2.0 - Bringing enterprise-grade search capabilities to everyone*
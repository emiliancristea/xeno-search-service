# Xeno Search Service Integration Guide

This document provides complete instructions for integrating the Xeno Search service into your existing chat application interface. The Xeno Search service provides Perplexity-like web search capabilities with real-time content scraping and AI-powered summarization.

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Python Service Setup](#python-service-setup)
4. [Node.js Backend Integration](#nodejs-backend-integration)
5. [Frontend Integration (ChatWithLLM.tsx)](#frontend-integration-chatwithllmtsx)
6. [API Specifications](#api-specifications)
7. [Error Handling](#error-handling)
8. [Testing Guidelines](#testing-guidelines)
9. [Configuration Management](#configuration-management)
10. [Production Deployment](#production-deployment)
11. [Troubleshooting](#troubleshooting)

## System Architecture Overview

The Xeno Search system consists of three main components:

```
Frontend (React/TypeScript)
    ↓ HTTP Request
Node.js Backend (API Gateway)
    ↓ HTTP Request
Python Service (Xeno Search Engine)
    ↓ Web Scraping
External Websites
```

**Flow:**
1. User enables "Xeno Search" in the chat interface
2. Frontend sends request to Node.js backend `/api/xeno-search`
3. Node.js backend forwards request to Python service `/api/xeno-search-internal`
4. Python service scrapes web for information and generates summaries
5. Results flow back through the chain to display in frontend
6. LLM uses the search results as context for generating responses

## Prerequisites

### System Requirements
- Python 3.8+ with pip
- Node.js 16+ with npm/yarn
- Windows PowerShell (for Windows development)
- 4GB+ RAM (for AI summarization models)
- GPU support recommended but not required

### Existing Project Structure
Your project should have:
- Node.js backend server (typically in `src/server/` or similar)
- React frontend with `ChatWithLLM.tsx` component
- Package management files (`package.json`, etc.)

## Python Service Setup

### 1. Create Python Service Directory

```powershell
# Navigate to your project root
cd C:\path\to\your\project
mkdir xeno-search-service
cd xeno-search-service
```

### 2. Set Up Python Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Verify activation (should show (venv) in prompt)
```

### 3. Create Directory Structure

```
xeno-search-service/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── core/
│   │   └── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── search_models.py
│   └── services/
│       ├── __init__.py
│       ├── scraping_service.py
│       └── nlp_service.py
├── requirements.txt
├── .gitignore
└── README.md
```

### 4. Install Dependencies

Create `requirements.txt`:
```txt
fastapi
uvicorn[standard]
httpx
beautifulsoup4
lxml
python-dotenv
readability-lxml
transformers
torch
sentencepiece
```

Install dependencies:
```powershell
pip install -r requirements.txt
```

### 5. Implement Core Files

The service consists of several key files that need to be implemented exactly as specified in the attached project files:

- `app/main.py` - FastAPI application with endpoints
- `app/models/search_models.py` - Pydantic models for request/response
- `app/services/scraping_service.py` - Web scraping logic
- `app/services/nlp_service.py` - AI summarization service

### 6. Start Python Service

```powershell
# In xeno-search-service directory with venv activated
uvicorn app.main:app --reload --port 8000
```

**Verification:** Visit `http://127.0.0.1:8000/docs` to see the API documentation.

## Node.js Backend Integration

### 1. Install HTTP Client (if not already present)

```bash
# In your Node.js project directory
npm install axios
# or
yarn add axios
```

### 2. Add Xeno Search Endpoint

In your main server file (typically `src/server/index.js` or similar), add:

```javascript
const axios = require('axios');

// Xeno Search endpoint
app.post('/api/xeno-search', async (req, res) => {
  try {
    const { query, search_type = 'normal', num_results = 5 } = req.body;

    // Validate input
    if (!query || typeof query !== 'string' || query.trim().length === 0) {
      return res.status(400).json({ 
        error: 'Query is required and must be a non-empty string' 
      });
    }

    // Log the request
    console.log(`[Xeno Search] Processing request: "${query}" (type: ${search_type})`);

    // Call Python service
    const pythonServiceUrl = process.env.PYTHON_SERVICE_URL || 'http://127.0.0.1:8000';
    const startTime = Date.now();
    
    const response = await axios.post(
      `${pythonServiceUrl}/api/xeno-search-internal`,
      {
        query: query.trim(),
        search_type,
        num_results: Math.min(Math.max(num_results, 1), 10) // Clamp between 1-10
      },
      {
        timeout: 30000, // 30 second timeout
        headers: {
          'Content-Type': 'application/json'
        }
      }
    );

    const processingTime = Date.now() - startTime;
    console.log(`[Xeno Search] Completed in ${processingTime}ms`);

    // Return the results
    res.json(response.data);

  } catch (error) {
    console.error('[Xeno Search] Error:', error.message);
    
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({ 
        error: 'Search service is currently unavailable. Please try again later.',
        details: 'Python service connection refused'
      });
    }

    if (error.response) {
      // Python service returned an error
      return res.status(error.response.status).json({
        error: 'Search processing failed',
        details: error.response.data?.detail || error.response.data?.error || 'Unknown error'
      });
    }

    if (error.code === 'ENOTFOUND') {
      return res.status(503).json({ 
        error: 'Search service is not accessible',
        details: 'Cannot reach Python service'
      });
    }

    // Generic error
    res.status(500).json({ 
      error: 'Internal server error during search processing',
      details: process.env.NODE_ENV === 'development' ? error.message : 'Internal error'
    });
  }
});
```

### 3. Environment Configuration

Add to your `.env` file:
```env
# Python service configuration
PYTHON_SERVICE_URL=http://127.0.0.1:8000
```

### 4. Restart Node.js Server

After implementing the endpoint, restart your Node.js server to apply changes.

## Frontend Integration (ChatWithLLM.tsx)

### 1. Add Xeno Search State Management

Add these state variables to your `ChatWithLLM.tsx` component:

```typescript
interface XenoSearchResult {
  query: string;
  search_type: string;
  summary?: string;
  sources: Array<{
    url: string;
    title?: string;
    snippet?: string;
    summary?: string;
  }>;
  error?: string;
}

// Add to your component state
const [isXenoSearchEnabled, setIsXenoSearchEnabled] = useState(false);
const [xenoSearchResults, setXenoSearchResults] = useState<XenoSearchResult | null>(null);
const [isSearching, setIsSearching] = useState(false);
```

### 2. Add Xeno Search UI Toggle

Add a toggle component in your chat interface:

```typescript
// Add to your JSX, typically near other chat controls
<div className="xeno-search-controls">
  <label className="flex items-center space-x-2">
    <input
      type="checkbox"
      checked={isXenoSearchEnabled}
      onChange={(e) => setIsXenoSearchEnabled(e.target.checked)}
      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
    />
    <span className="text-sm font-medium text-gray-700">
      Enable Xeno Search
    </span>
  </label>
  {isXenoSearchEnabled && (
    <span className="text-xs text-gray-500">
      🔍 Real-time web search will be performed
    </span>
  )}
</div>
```

### 3. Implement Search Function

Add this function to your component:

```typescript
const performXenoSearch = async (query: string): Promise<XenoSearchResult | null> => {
  try {
    setIsSearching(true);
    
    const response = await fetch('/api/xeno-search', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        search_type: 'normal',
        num_results: 5
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP ${response.status}`);
    }

    const searchResults = await response.json();
    setXenoSearchResults(searchResults);
    return searchResults;

  } catch (error) {
    console.error('Xeno Search failed:', error);
    setXenoSearchResults({
      query,
      search_type: 'normal',
      sources: [],
      error: error instanceof Error ? error.message : 'Search failed'
    });
    return null;
  } finally {
    setIsSearching(false);
  }
};
```

### 4. Modify Message Handling Logic

Update your message generation function:

```typescript
const handleGenerate = async () => {
  if (!message.trim()) return;

  let searchContext = '';
  let searchResults: XenoSearchResult | null = null;

  // Perform Xeno Search if enabled
  if (isXenoSearchEnabled) {
    searchResults = await performXenoSearch(message);
    
    if (searchResults && searchResults.sources.length > 0) {
      // Create context from search results
      const summaries = searchResults.sources
        .map(source => source.summary)
        .filter(Boolean)
        .join('\n\n');
      
      searchContext = `
Based on recent web search results for "${message}":

${searchResults.summary || 'Search Summary: ' + summaries}

Sources:
${searchResults.sources.map((source, index) => 
  `${index + 1}. ${source.title} - ${source.url}`
).join('\n')}

Please provide a comprehensive response using this information as context.

---

User Question: `;
    }
  }

  // Prepare the final prompt
  const finalPrompt = searchContext + message;

  // Continue with your existing LLM generation logic
  // Pass finalPrompt to your LLM instead of just message
  // ... rest of your generation logic
};
```

### 5. Display Search Results

Add search results display component:

```typescript
const XenoSearchResultsDisplay = ({ results }: { results: XenoSearchResult }) => {
  if (!results) return null;

  return (
    <div className="xeno-search-results mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
      <div className="flex items-center mb-2">
        <span className="text-blue-600 font-semibold">🔍 Xeno Search Results</span>
        {results.error && (
          <span className="ml-2 text-red-600 text-sm">⚠️ {results.error}</span>
        )}
      </div>
      
      {results.summary && (
        <div className="mb-3">
          <h4 className="font-medium text-gray-900 mb-1">Summary:</h4>
          <p className="text-gray-700 text-sm">{results.summary}</p>
        </div>
      )}
      
      {results.sources.length > 0 && (
        <div>
          <h4 className="font-medium text-gray-900 mb-2">Sources:</h4>
          <div className="space-y-2">
            {results.sources.map((source, index) => (
              <div key={index} className="border-l-2 border-blue-300 pl-3">
                <a
                  href={source.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-800 font-medium text-sm"
                >
                  {source.title || source.url}
                </a>
                {source.snippet && (
                  <p className="text-gray-600 text-xs mt-1">{source.snippet}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Add to your JSX where you want to display results
{xenoSearchResults && <XenoSearchResultsDisplay results={xenoSearchResults} />}
{isSearching && (
  <div className="flex items-center text-blue-600 mb-2">
    <span className="animate-spin mr-2">🔄</span>
    Performing Xeno Search...
  </div>
)}
```

## API Specifications

### Python Service API

#### POST `/api/xeno-search-internal`

**Request Body:**
```json
{
  "query": "string (required)",
  "search_type": "normal | deep (default: normal)",
  "num_results": "number (default: 5, max: 10)"
}
```

**Response:**
```json
{
  "query": "string",
  "search_type": "string",
  "summary": "string | null",
  "sources": [
    {
      "url": "string",
      "title": "string | null",
      "snippet": "string | null",
      "raw_text": "string | null",
      "summary": "string | null"
    }
  ],
  "error": "string | null"
}
```

### Node.js Backend API

#### POST `/api/xeno-search`

**Request Body:**
```json
{
  "query": "string (required)",
  "search_type": "normal | deep (optional, default: normal)",
  "num_results": "number (optional, default: 5, max: 10)"
}
```

**Response:** Same as Python service response

## Error Handling

### Common Error Scenarios

1. **Python Service Unavailable**
   - Status: 503
   - Message: "Search service is currently unavailable"
   - Action: Check if Python service is running

2. **Invalid Query**
   - Status: 400
   - Message: "Query is required and must be a non-empty string"
   - Action: Validate input on frontend

3. **Search Processing Failed**
   - Status: 500
   - Message: "Search processing failed"
   - Action: Check Python service logs

4. **Timeout**
   - Status: 504
   - Message: "Search request timed out"
   - Action: Reduce num_results or optimize search

### Frontend Error Handling

```typescript
// Add to your error handling in frontend
if (error.message.includes('unavailable')) {
  setErrorMessage('🔍 Search service is temporarily unavailable. Using standard response.');
} else if (error.message.includes('timeout')) {
  setErrorMessage('🔍 Search took too long. Using standard response.');
} else {
  setErrorMessage('🔍 Search failed. Using standard response.');
}
```

## Testing Guidelines

### 1. Python Service Testing

```powershell
# Test health endpoint
curl http://127.0.0.1:8000/health

# Test search endpoint
curl -X POST http://127.0.0.1:8000/api/xeno-search-internal \
  -H "Content-Type: application/json" \
  -d '{"query":"latest AI news","search_type":"normal","num_results":3}'
```

### 2. Node.js Integration Testing

```powershell
# Test Node.js endpoint
Invoke-WebRequest -Uri "http://localhost:3000/api/xeno-search" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"query":"latest technology trends","search_type":"normal","num_results":3}'
```

### 3. Frontend Testing

1. Enable Xeno Search toggle
2. Send a test message
3. Verify search results appear
4. Check that LLM response incorporates search context
5. Test error scenarios (Python service down)

## Configuration Management

### Environment Variables

Create `.env` files for each service:

**Node.js Backend (.env):**
```env
NODE_ENV=development
PYTHON_SERVICE_URL=http://127.0.0.1:8000
XENO_SEARCH_TIMEOUT=30000
```

**Python Service (.env):**
```env
ENVIRONMENT=development
LOG_LEVEL=INFO
SEARXNG_INSTANCE_URL=https://searx.tiekoetter.com/search
SUMMARIZATION_MODEL=sshleifer/distilbart-cnn-12-6
```

## Production Deployment

### 1. Containerization

**Python Service Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker Compose:**
```yaml
version: '3.8'
services:
  xeno-search:
    build: ./xeno-search-service
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
    restart: unless-stopped

  backend:
    # Your existing Node.js service
    environment:
      - PYTHON_SERVICE_URL=http://xeno-search:8000
    depends_on:
      - xeno-search
```

### 2. Security Considerations

- Use HTTPS in production
- Implement rate limiting
- Validate all inputs
- Use environment variables for sensitive config
- Monitor resource usage (AI models are memory-intensive)

## Troubleshooting

### Common Issues

1. **"Module not found" errors**
   - Solution: Ensure virtual environment is activated and dependencies installed

2. **CUDA out of memory**
   - Solution: Reduce batch size or use CPU-only mode

3. **Search service connection refused**
   - Solution: Verify Python service is running on correct port

4. **Empty search results**
   - Solution: Check search engine availability and query validity

5. **Slow performance**
   - Solution: Consider using smaller AI models or implementing caching

### Debug Mode

Enable debug logging in Python service:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks

Both services should implement health check endpoints:
- Python: `GET /health`
- Node.js: Add similar endpoint

## Support and Maintenance

### Regular Tasks

1. **Monitor AI model performance** - Summarization quality may degrade
2. **Update search engine selectors** - Web scraping targets change frequently
3. **Rotate proxy services** - If using proxies for scraping
4. **Monitor resource usage** - AI models consume significant memory
5. **Update dependencies** - Security and performance updates

### Performance Optimization

1. **Implement caching** - Cache search results for repeated queries
2. **Use smaller models** - Trade accuracy for speed if needed
3. **Batch processing** - Process multiple sources simultaneously
4. **Connection pooling** - Reuse HTTP connections

This completes the integration guide for the Xeno Search service. Follow these steps sequentially to implement the search functionality into your existing chat application. 
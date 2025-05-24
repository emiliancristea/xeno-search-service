# Deep Search Implementation Guide

**Complete guide for implementing real-time deep search functionality with advanced UI/UX patterns**

---

## Table of Contents

1. [Deep Search Architecture Overview](#deep-search-architecture-overview)
2. [Backend Implementation](#backend-implementation)
3. [Real-Time UI Design Specifications](#real-time-ui-design-specifications)
4. [Frontend Implementation](#frontend-implementation)
5. [Progressive Search Flow](#progressive-search-flow)
6. [Advanced UI Components](#advanced-ui-components)
7. [Testing & Optimization](#testing--optimization)

## Deep Search Architecture Overview

### What is Deep Search?

Deep Search is an advanced search methodology that:
- **Follows links** from initial search results (1-2 levels deep)
- **Performs semantic analysis** to find related content
- **Aggregates information** from multiple sources
- **Provides comprehensive summaries** with citations
- **Shows real-time progress** to users during the search process

### Key Differentiators from Normal Search

| Normal Search | Deep Search |
|---------------|-------------|
| 5-10 sources | 15-25 sources |
| Surface-level content | Multi-level analysis |
| Basic summarization | Comprehensive synthesis |
| 2-5 second response | 8-15 second response |
| Static results | Progressive discovery |

## Backend Implementation

### 1. Deep Search Service Architecture

Our implementation consists of these key components:

```python
# app/services/deep_search_service.py (already implemented)
class DeepSearchService:
    def __init__(self):
        self.max_depth = 2
        self.max_links_per_level = 5
        self.min_relevance_score = 0.3
        
    async def perform_deep_search(self, query, initial_sources, num_additional_sources=10):
        # Implementation details in existing file
```

### 2. Real-Time Progress Tracking

To enable real-time UI updates, we need to modify our service to emit progress events:

```python
# app/services/deep_search_service_realtime.py
from typing import AsyncGenerator, Dict, Any
import asyncio
from enum import Enum

class SearchPhase(Enum):
    INITIALIZING = "initializing"
    INITIAL_SEARCH = "initial_search"
    ANALYZING_SOURCES = "analyzing_sources"
    EXTRACTING_LINKS = "extracting_links"
    FOLLOWING_LINKS = "following_links"
    SCRAPING_CONTENT = "scraping_content"
    GENERATING_SUMMARIES = "generating_summaries"
    CREATING_COMPREHENSIVE_SUMMARY = "creating_comprehensive_summary"
    COMPLETED = "completed"

class RealTimeDeepSearchService:
    async def perform_deep_search_with_updates(
        self, 
        query: str, 
        initial_sources: List[Source]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Performs deep search while yielding real-time progress updates
        """
        total_steps = 8
        current_step = 0
        
        # Step 1: Initialize
        current_step += 1
        yield {
            "phase": SearchPhase.INITIALIZING.value,
            "step": current_step,
            "total_steps": total_steps,
            "progress": (current_step / total_steps) * 100,
            "message": "Initializing deep search...",
            "data": {"query": query, "initial_sources_count": len(initial_sources)}
        }
        
        await asyncio.sleep(0.5)  # UI feedback delay
        
        # Step 2: Analyze initial sources
        current_step += 1
        yield {
            "phase": SearchPhase.ANALYZING_SOURCES.value,
            "step": current_step,
            "total_steps": total_steps,
            "progress": (current_step / total_steps) * 100,
            "message": f"Analyzing {len(initial_sources)} initial sources...",
            "data": {"sources": [{"title": s.title, "url": str(s.url)} for s in initial_sources]}
        }
        
        # Step 3: Extract link candidates
        current_step += 1
        link_candidates = await self._extract_link_candidates_with_progress(query, initial_sources)
        yield {
            "phase": SearchPhase.EXTRACTING_LINKS.value,
            "step": current_step,
            "total_steps": total_steps,
            "progress": (current_step / total_steps) * 100,
            "message": f"Found {len(link_candidates)} relevant links to explore...",
            "data": {
                "link_candidates": [
                    {
                        "url": lc.url,
                        "anchor_text": lc.anchor_text,
                        "relevance_score": round(lc.relevance_score, 2)
                    } for lc in link_candidates[:10]  # Show top 10
                ]
            }
        }
        
        # Step 4: Follow links
        current_step += 1
        ranked_links = self._rank_link_candidates(query, link_candidates)
        yield {
            "phase": SearchPhase.FOLLOWING_LINKS.value,
            "step": current_step,
            "total_steps": total_steps,
            "progress": (current_step / total_steps) * 100,
            "message": f"Following top {min(10, len(ranked_links))} most relevant links...",
            "data": {
                "ranked_links": [
                    {
                        "url": rl.url,
                        "relevance_score": round(rl.relevance_score, 2),
                        "source": rl.source_url
                    } for rl in ranked_links[:10]
                ]
            }
        }
        
        # Step 5: Scrape content (with individual updates)
        current_step += 1
        additional_sources = []
        for i, candidate in enumerate(ranked_links[:10]):
            yield {
                "phase": SearchPhase.SCRAPING_CONTENT.value,
                "step": current_step,
                "total_steps": total_steps,
                "progress": (current_step / total_steps) * 100,
                "message": f"Scraping content from {candidate.url}... ({i+1}/{min(10, len(ranked_links))})",
                "data": {
                    "current_url": candidate.url,
                    "progress_detail": f"{i+1}/{min(10, len(ranked_links))}"
                }
            }
            
            source = await self._scrape_link_candidate(candidate)
            if source:
                additional_sources.append(source)
                yield {
                    "phase": SearchPhase.SCRAPING_CONTENT.value,
                    "step": current_step,
                    "total_steps": total_steps,
                    "progress": (current_step / total_steps) * 100,
                    "message": f"✅ Successfully scraped: {source.title}",
                    "data": {
                        "scraped_source": {
                            "title": source.title,
                            "url": str(source.url),
                            "content_length": len(source.raw_text) if source.raw_text else 0
                        }
                    }
                }
            
            await asyncio.sleep(0.3)  # Respectful delay
        
        # Step 6: Generate individual summaries
        current_step += 1
        yield {
            "phase": SearchPhase.GENERATING_SUMMARIES.value,
            "step": current_step,
            "total_steps": total_steps,
            "progress": (current_step / total_steps) * 100,
            "message": f"Generating AI summaries for {len(additional_sources)} new sources...",
            "data": {"sources_to_summarize": len(additional_sources)}
        }
        
        # Step 7: Create comprehensive summary
        current_step += 1
        all_sources = initial_sources + additional_sources
        yield {
            "phase": SearchPhase.CREATING_COMPREHENSIVE_SUMMARY.value,
            "step": current_step,
            "total_steps": total_steps,
            "progress": (current_step / total_steps) * 100,
            "message": f"Creating comprehensive summary from {len(all_sources)} total sources...",
            "data": {"total_sources": len(all_sources)}
        }
        
        comprehensive_summary = await self._generate_comprehensive_summary(query, all_sources)
        
        # Step 8: Complete
        current_step += 1
        yield {
            "phase": SearchPhase.COMPLETED.value,
            "step": current_step,
            "total_steps": total_steps,
            "progress": 100,
            "message": f"✅ Deep search completed! Analyzed {len(all_sources)} sources.",
            "data": {
                "final_results": {
                    "query": query,
                    "total_sources": len(all_sources),
                    "initial_sources": len(initial_sources),
                    "deep_sources": len(additional_sources),
                    "comprehensive_summary": comprehensive_summary,
                    "sources": [
                        {
                            "title": s.title,
                            "url": str(s.url),
                            "snippet": s.snippet,
                            "summary": s.summary
                        } for s in all_sources
                    ]
                }
            }
        }
```

### 3. WebSocket Integration for Real-Time Updates

```python
# app/websocket/deep_search_ws.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class DeepSearchWebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast_progress(self, search_id: str, progress_data: dict):
        message = {
            "search_id": search_id,
            "type": "progress_update",
            "data": progress_data
        }
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.active_connections.remove(conn)

# Add to main.py
from fastapi import WebSocket
import uuid

manager = DeepSearchWebSocketManager()

@app.websocket("/ws/deep-search/{search_id}")
async def deep_search_websocket(websocket: WebSocket, search_id: str):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/start-deep-search")
async def start_deep_search(request: SearchRequest):
    search_id = str(uuid.uuid4())
    
    # Start deep search in background
    asyncio.create_task(perform_deep_search_with_websocket(search_id, request))
    
    return {"search_id": search_id}

async def perform_deep_search_with_websocket(search_id: str, request: SearchRequest):
    # Perform initial search
    initial_sources = await perform_search(request.query, "normal", request.num_results)
    
    # Start deep search with real-time updates
    realtime_service = RealTimeDeepSearchService()
    async for progress_update in realtime_service.perform_deep_search_with_updates(
        request.query, initial_sources
    ):
        await manager.broadcast_progress(search_id, progress_update)
```

## Real-Time UI Design Specifications

### 1. Search Progress Interface Design

Based on analysis of Perplexity AI and other modern search interfaces, here's the optimal UI design:

#### Layout Structure
```
┌─────────────────────────────────────────────────┐
│ [Query Input]                    [🔍 Deep Search] │
├─────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────┐ │
│ │ 🔄 Deep Search Progress                     │ │
│ │ ████████████░░░░ 75% Complete               │ │
│ │ Following links from initial sources...     │ │
│ └─────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────┐ │
│ │ 📊 Search Timeline                          │ │
│ │ ✅ Initial Search (5 sources found)         │ │
│ │ ✅ Link Analysis (23 candidates)            │ │
│ │ 🔄 Content Scraping (3/10 complete)        │ │
│ │ ⏳ Summary Generation                       │ │
│ └─────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────┐ │
│ │ 🌐 Sources Being Explored                   │ │
│ │ • ✅ wikipedia.org - AI Technologies        │ │
│ │ • 🔄 techcrunch.com - Latest AI News       │ │
│ │ • ⏳ arxiv.org - Research Papers            │ │
│ └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

#### Visual Design Elements

**Color Scheme:**
- Primary Blue: `#2563eb` (Active/In-progress)
- Success Green: `#16a34a` (Completed steps)
- Neutral Gray: `#6b7280` (Pending steps)
- Background: `#f8fafc` (Light mode) / `#0f172a` (Dark mode)

**Icons:**
- 🔍 Deep Search activation
- 🔄 In-progress indicator (animated)
- ✅ Completed step
- ⏳ Pending step
- 🌐 Website/source icon
- 📊 Analytics/progress icon

### 2. Progressive Disclosure Pattern

The interface reveals information progressively:

1. **Phase 1: Search Initiation** (0-10%)
   - Show search query
   - Display "Initializing deep search..." message

2. **Phase 2: Initial Discovery** (10-25%)
   - Show initial sources found
   - Display source count and titles

3. **Phase 3: Link Exploration** (25-50%)
   - Show link candidates being analyzed
   - Display relevance scores
   - Show which sites are being explored

4. **Phase 4: Content Scraping** (50-75%)
   - Real-time updates for each URL being scraped
   - Show success/failure status
   - Display content length/quality indicators

5. **Phase 5: AI Processing** (75-90%)
   - Show summarization progress
   - Display AI model working indicators

6. **Phase 6: Final Results** (90-100%)
   - Show comprehensive summary
   - Display all sources with snippets
   - Provide related search suggestions

## Frontend Implementation

### 1. React Component Structure

```tsx
// components/DeepSearch/DeepSearchInterface.tsx
import React, { useState, useEffect } from 'react';
import { SearchProgressBar } from './SearchProgressBar';
import { SearchTimeline } from './SearchTimeline';
import { SourceExplorer } from './SourceExplorer';
import { SearchResults } from './SearchResults';

interface DeepSearchProps {
  query: string;
  onResults: (results: DeepSearchResults) => void;
}

export const DeepSearchInterface: React.FC<DeepSearchProps> = ({ query, onResults }) => {
  const [searchState, setSearchState] = useState<DeepSearchState>({
    phase: 'idle',
    progress: 0,
    message: '',
    data: null,
    isActive: false
  });

  const [searchResults, setSearchResults] = useState<DeepSearchResults | null>(null);
  const [websocket, setWebSocket] = useState<WebSocket | null>(null);

  const startDeepSearch = async () => {
    try {
      // Start deep search
      const response = await fetch('/api/start-deep-search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          search_type: 'deep',
          num_results: 5
        })
      });

      const { search_id } = await response.json();
      
      // Connect to WebSocket for real-time updates
      const ws = new WebSocket(`ws://localhost:8000/ws/deep-search/${search_id}`);
      
      ws.onmessage = (event) => {
        const update = JSON.parse(event.data);
        if (update.type === 'progress_update') {
          setSearchState(prev => ({
            ...prev,
            ...update.data,
            isActive: update.data.phase !== 'completed'
          }));

          // If completed, extract final results
          if (update.data.phase === 'completed') {
            setSearchResults(update.data.data.final_results);
            onResults(update.data.data.final_results);
          }
        }
      };

      setWebSocket(ws);
      setSearchState(prev => ({ ...prev, isActive: true }));

    } catch (error) {
      console.error('Failed to start deep search:', error);
    }
  };

  useEffect(() => {
    return () => {
      if (websocket) {
        websocket.close();
      }
    };
  }, [websocket]);

  return (
    <div className="deep-search-interface">
      {/* Search Control */}
      <div className="search-control mb-6">
        <div className="flex items-center gap-4">
          <span className="text-lg font-medium">{query}</span>
          <button
            onClick={startDeepSearch}
            disabled={searchState.isActive}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {searchState.isActive ? '🔄 Searching...' : '🔍 Start Deep Search'}
          </button>
        </div>
      </div>

      {/* Progress Section */}
      {searchState.isActive && (
        <div className="progress-section space-y-4">
          <SearchProgressBar 
            progress={searchState.progress}
            phase={searchState.phase}
            message={searchState.message}
          />
          
          <SearchTimeline 
            currentPhase={searchState.phase}
            searchData={searchState.data}
          />
          
          <SourceExplorer 
            phase={searchState.phase}
            searchData={searchState.data}
          />
        </div>
      )}

      {/* Results Section */}
      {searchResults && (
        <SearchResults results={searchResults} />
      )}
    </div>
  );
};
```

### 2. Search Progress Bar Component

```tsx
// components/DeepSearch/SearchProgressBar.tsx
import React from 'react';

interface SearchProgressBarProps {
  progress: number;
  phase: string;
  message: string;
}

export const SearchProgressBar: React.FC<SearchProgressBarProps> = ({ progress, phase, message }) => {
  const getPhaseIcon = (currentPhase: string) => {
    const icons = {
      'initializing': '🔄',
      'initial_search': '🔍',
      'analyzing_sources': '📊',
      'extracting_links': '🔗',
      'following_links': '🌐',
      'scraping_content': '📄',
      'generating_summaries': '🤖',
      'creating_comprehensive_summary': '📝',
      'completed': '✅'
    };
    return icons[currentPhase] || '🔄';
  };

  return (
    <div className="search-progress-bar bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
      <div className="flex items-center gap-3 mb-4">
        <span className="text-2xl animate-pulse">{getPhaseIcon(phase)}</span>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Deep Search Progress
        </h3>
      </div>
      
      <div className="mb-2">
        <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400">
          <span>{message}</span>
          <span>{Math.round(progress)}%</span>
        </div>
      </div>
      
      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
        <div 
          className="bg-gradient-to-r from-blue-500 to-blue-600 h-3 rounded-full transition-all duration-300 ease-out"
          style={{ width: `${progress}%` }}
        >
          <div className="h-full bg-white/20 rounded-full animate-pulse"></div>
        </div>
      </div>
    </div>
  );
};
```

### 3. Search Timeline Component

```tsx
// components/DeepSearch/SearchTimeline.tsx
import React from 'react';

interface SearchTimelineProps {
  currentPhase: string;
  searchData: any;
}

export const SearchTimeline: React.FC<SearchTimelineProps> = ({ currentPhase, searchData }) => {
  const phases = [
    { id: 'initializing', label: 'Initializing', icon: '🔄' },
    { id: 'initial_search', label: 'Initial Search', icon: '🔍' },
    { id: 'analyzing_sources', label: 'Analyzing Sources', icon: '📊' },
    { id: 'extracting_links', label: 'Finding Links', icon: '🔗' },
    { id: 'following_links', label: 'Following Links', icon: '🌐' },
    { id: 'scraping_content', label: 'Scraping Content', icon: '📄' },
    { id: 'generating_summaries', label: 'AI Summarization', icon: '🤖' },
    { id: 'creating_comprehensive_summary', label: 'Final Summary', icon: '📝' }
  ];

  const getPhaseStatus = (phaseId: string) => {
    const phaseIndex = phases.findIndex(p => p.id === phaseId);
    const currentIndex = phases.findIndex(p => p.id === currentPhase);
    
    if (phaseIndex < currentIndex) return 'completed';
    if (phaseIndex === currentIndex) return 'active';
    return 'pending';
  };

  const getPhaseData = (phaseId: string) => {
    if (!searchData) return null;
    
    switch (phaseId) {
      case 'initial_search':
        return searchData.initial_sources_count ? `${searchData.initial_sources_count} sources` : null;
      case 'extracting_links':
        return searchData.link_candidates ? `${searchData.link_candidates.length} links found` : null;
      case 'scraping_content':
        return searchData.progress_detail || null;
      default:
        return null;
    }
  };

  return (
    <div className="search-timeline bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        📊 Search Timeline
      </h3>
      
      <div className="space-y-3">
        {phases.map((phase, index) => {
          const status = getPhaseStatus(phase.id);
          const data = getPhaseData(phase.id);
          
          return (
            <div key={phase.id} className="flex items-center gap-3">
              <div className={`flex items-center justify-center w-8 h-8 rounded-full ${
                status === 'completed' ? 'bg-green-100 text-green-600' :
                status === 'active' ? 'bg-blue-100 text-blue-600 animate-pulse' :
                'bg-gray-100 text-gray-400'
              }`}>
                {status === 'completed' ? '✅' : phase.icon}
              </div>
              
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className={`font-medium ${
                    status === 'active' ? 'text-blue-600' : 
                    status === 'completed' ? 'text-green-600' : 
                    'text-gray-500'
                  }`}>
                    {phase.label}
                  </span>
                  {data && (
                    <span className="text-sm text-gray-500 bg-gray-100 px-2 py-1 rounded">
                      {data}
                    </span>
                  )}
                </div>
              </div>
              
              {index < phases.length - 1 && (
                <div className={`w-px h-4 ${
                  status === 'completed' ? 'bg-green-300' : 'bg-gray-200'
                }`} />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
```

### 4. Source Explorer Component

```tsx
// components/DeepSearch/SourceExplorer.tsx
import React from 'react';

interface SourceExplorerProps {
  phase: string;
  searchData: any;
}

export const SourceExplorer: React.FC<SourceExplorerProps> = ({ phase, searchData }) => {
  if (!searchData) return null;

  const renderInitialSources = () => {
    if (!searchData.sources) return null;
    
    return (
      <div className="space-y-2">
        <h4 className="font-medium text-gray-900 dark:text-white">Initial Sources Found:</h4>
        {searchData.sources.map((source, index) => (
          <div key={index} className="flex items-center gap-2 p-2 bg-gray-50 dark:bg-gray-700 rounded">
            <span className="text-green-500">✅</span>
            <div className="flex-1">
              <div className="font-medium text-sm text-gray-900 dark:text-white">
                {source.title}
              </div>
              <div className="text-xs text-gray-500">{source.url}</div>
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderLinkCandidates = () => {
    if (!searchData.link_candidates) return null;
    
    return (
      <div className="space-y-2">
        <h4 className="font-medium text-gray-900 dark:text-white">Link Candidates:</h4>
        {searchData.link_candidates.slice(0, 5).map((link, index) => (
          <div key={index} className="flex items-center gap-2 p-2 bg-gray-50 dark:bg-gray-700 rounded">
            <span className="text-blue-500">🔗</span>
            <div className="flex-1">
              <div className="font-medium text-sm text-gray-900 dark:text-white">
                {link.anchor_text}
              </div>
              <div className="text-xs text-gray-500">{link.url}</div>
            </div>
            <div className="text-xs bg-blue-100 text-blue-600 px-2 py-1 rounded">
              {(link.relevance_score * 100).toFixed(0)}%
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderScrapingProgress = () => {
    if (!searchData.current_url) return null;
    
    return (
      <div className="space-y-2">
        <h4 className="font-medium text-gray-900 dark:text-white">Currently Scraping:</h4>
        <div className="flex items-center gap-2 p-2 bg-yellow-50 dark:bg-yellow-900/20 rounded">
          <span className="text-yellow-500 animate-pulse">🔄</span>
          <div className="flex-1">
            <div className="font-medium text-sm text-gray-900 dark:text-white">
              {searchData.current_url}
            </div>
            {searchData.progress_detail && (
              <div className="text-xs text-gray-500">
                Progress: {searchData.progress_detail}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="source-explorer bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        🌐 Sources Being Explored
      </h3>
      
      <div className="space-y-4">
        {phase === 'analyzing_sources' && renderInitialSources()}
        {phase === 'extracting_links' && renderLinkCandidates()}
        {phase === 'scraping_content' && renderScrapingProgress()}
      </div>
    </div>
  );
};
```

## Progressive Search Flow

### 1. User Experience Flow

```
1. User Input
   ├─ User types query: "latest AI developments"
   ├─ User clicks "🔍 Deep Search" button
   └─ Interface shows: "Initializing deep search..."

2. Initial Discovery (2-3 seconds)
   ├─ "🔍 Searching web for initial sources..."
   ├─ Shows: "Found 5 initial sources from search engines"
   └─ Displays source cards with titles and URLs

3. Link Analysis (3-5 seconds)
   ├─ "📊 Analyzing content for relevant links..."
   ├─ Shows: "Found 23 link candidates"
   └─ Displays top candidates with relevance scores

4. Deep Exploration (8-12 seconds)
   ├─ "🌐 Following most relevant links..."
   ├─ Real-time updates: "Scraping techcrunch.com... ✅"
   ├─ Real-time updates: "Scraping arxiv.org... ✅"
   └─ Shows progress: "Scraped 8/10 additional sources"

5. AI Processing (2-3 seconds)
   ├─ "🤖 Generating AI summaries..."
   ├─ "📝 Creating comprehensive analysis..."
   └─ Shows: "Processing 13 total sources"

6. Final Results (Instant)
   ├─ "✅ Deep search completed!"
   ├─ Shows comprehensive summary
   ├─ Displays all 13 sources with snippets
   └─ Provides related search suggestions
```

### 2. Animation and Timing

```css
/* Real-time UI Animations */
@keyframes searchPulse {
  0% { opacity: 0.6; transform: scale(1); }
  50% { opacity: 1; transform: scale(1.05); }
  100% { opacity: 0.6; transform: scale(1); }
}

@keyframes progressBar {
  0% { width: 0%; }
  100% { width: var(--target-width); }
}

@keyframes typewriter {
  from { width: 0; }
  to { width: 100%; }
}

.search-active-indicator {
  animation: searchPulse 2s infinite;
}

.progress-bar-fill {
  animation: progressBar 0.5s ease-out forwards;
}

.status-message {
  animation: typewriter 0.3s steps(40, end);
  overflow: hidden;
  white-space: nowrap;
}
```

## Advanced UI Components

### 1. Interactive Source Cards

```tsx
// components/DeepSearch/SourceCard.tsx
interface SourceCardProps {
  source: Source;
  isActive?: boolean;
  status: 'pending' | 'scraping' | 'completed' | 'failed';
}

export const SourceCard: React.FC<SourceCardProps> = ({ source, isActive, status }) => {
  const getStatusIcon = () => {
    switch (status) {
      case 'pending': return '⏳';
      case 'scraping': return '🔄';
      case 'completed': return '✅';
      case 'failed': return '❌';
      default: return '📄';
    }
  };

  return (
    <div className={`source-card p-4 rounded-lg border transition-all duration-300 ${
      isActive ? 'border-blue-500 bg-blue-50 shadow-lg transform scale-105' : 
      'border-gray-200 bg-white hover:shadow-md'
    }`}>
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0">
          <span className={`text-lg ${status === 'scraping' ? 'animate-spin' : ''}`}>
            {getStatusIcon()}
          </span>
        </div>
        
        <div className="flex-1 min-w-0">
          <h4 className="font-medium text-gray-900 truncate">
            {source.title || 'Untitled'}
          </h4>
          <p className="text-sm text-gray-500 truncate">
            {source.url}
          </p>
          
          {source.snippet && (
            <p className="text-sm text-gray-600 mt-2 line-clamp-2">
              {source.snippet}
            </p>
          )}
          
          {status === 'completed' && source.summary && (
            <div className="mt-3 p-2 bg-green-50 rounded text-sm text-green-700">
              <strong>AI Summary:</strong> {source.summary}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
```

### 2. Real-Time Metrics Dashboard

```tsx
// components/DeepSearch/MetricsDashboard.tsx
export const MetricsDashboard: React.FC<{ searchData: any }> = ({ searchData }) => {
  return (
    <div className="metrics-dashboard grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
      <div className="metric-card bg-white p-4 rounded-lg shadow text-center">
        <div className="text-2xl font-bold text-blue-600">
          {searchData?.total_sources || 0}
        </div>
        <div className="text-sm text-gray-500">Total Sources</div>
      </div>
      
      <div className="metric-card bg-white p-4 rounded-lg shadow text-center">
        <div className="text-2xl font-bold text-green-600">
          {searchData?.successful_scrapes || 0}
        </div>
        <div className="text-sm text-gray-500">Successful</div>
      </div>
      
      <div className="metric-card bg-white p-4 rounded-lg shadow text-center">
        <div className="text-2xl font-bold text-purple-600">
          {searchData?.ai_summaries || 0}
        </div>
        <div className="text-sm text-gray-500">AI Summaries</div>
      </div>
      
      <div className="metric-card bg-white p-4 rounded-lg shadow text-center">
        <div className="text-2xl font-bold text-orange-600">
          {searchData?.search_time || 0}s
        </div>
        <div className="text-sm text-gray-500">Search Time</div>
      </div>
    </div>
  );
};
```

## Testing & Optimization

### 1. Performance Testing

```javascript
// tests/deep-search-performance.test.js
describe('Deep Search Performance', () => {
  test('should complete search within 15 seconds', async () => {
    const startTime = Date.now();
    const results = await performDeepSearch('AI technology trends');
    const endTime = Date.now();
    
    expect(endTime - startTime).toBeLessThan(15000);
    expect(results.sources.length).toBeGreaterThan(10);
  });
  
  test('should handle concurrent searches', async () => {
    const searches = Array(5).fill().map(() => 
      performDeepSearch('test query')
    );
    
    const results = await Promise.all(searches);
    expect(results).toHaveLength(5);
    results.forEach(result => {
      expect(result.sources.length).toBeGreaterThan(0);
    });
  });
});
```

### 2. UI Responsiveness Testing

```javascript
// tests/ui-responsiveness.test.js
describe('Deep Search UI Responsiveness', () => {
  test('should update progress within 100ms of backend events', async () => {
    const component = render(<DeepSearchInterface query="test" />);
    
    // Simulate WebSocket message
    const mockUpdate = { phase: 'scraping_content', progress: 50 };
    fireEvent(window, new CustomEvent('websocket-message', { detail: mockUpdate }));
    
    await waitFor(() => {
      expect(screen.getByText(/50%/)).toBeInTheDocument();
    }, { timeout: 100 });
  });
});
```

### 3. User Experience Optimization

```typescript
// Performance optimization strategies
const optimizations = {
  // Debounce rapid updates
  progressUpdates: debounce(updateProgress, 50),
  
  // Lazy load source cards
  sourceCards: useMemo(() => 
    sources.map(source => <SourceCard key={source.url} source={source} />),
    [sources]
  ),
  
  // Virtualize long lists
  virtualizedSources: useVirtualizer({
    count: sources.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 120,
  }),
  
  // Preload critical CSS
  criticalCSS: `
    .search-progress-bar { /* Critical styles */ }
    .source-card { /* Critical styles */ }
  `
};
```

## Conclusion

This implementation guide provides:

✅ **Complete Backend Architecture** - Real-time progress tracking
✅ **Modern UI/UX Design** - Progressive disclosure patterns  
✅ **Real-Time Updates** - WebSocket integration
✅ **Performance Optimization** - Efficient rendering
✅ **Production Ready** - Error handling and testing

The deep search functionality will provide users with a Perplexity-like experience, showing them exactly what the AI is discovering in real-time, building trust and engagement while delivering comprehensive search results. 
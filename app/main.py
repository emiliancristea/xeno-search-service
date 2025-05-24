from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uuid
import asyncio
import json

from app.models.search_models import SearchRequest, SearchResponse, Source
from app.services.scraping_service import perform_search
from app.services.nlp_service import summarize_text_async  # Import for meta-summary
from app.services.deep_search_service import perform_deep_search_analysis  # Import deep search

# WebSocket connection manager
class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.search_connections: dict = {}
    
    async def connect(self, websocket: WebSocket, search_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.search_connections[search_id] = websocket
        print(f"[WebSocket] Client connected for search {search_id}")
    
    def disconnect(self, websocket: WebSocket, search_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if search_id in self.search_connections:
            del self.search_connections[search_id]
        print(f"[WebSocket] Client disconnected for search {search_id}")
    
    async def send_progress_update(self, search_id: str, phase: str, progress: int, message: str, data: dict = None):
        if search_id in self.search_connections:
            try:
                update = {
                    "search_id": search_id,
                    "type": "progress_update",
                    "data": {
                        "phase": phase,
                        "progress": progress,
                        "message": message,
                        "data": data or {}
                    }
                }
                await self.search_connections[search_id].send_text(json.dumps(update))
                print(f"[WebSocket] Sent update for {search_id}: {phase} - {progress}%")
            except Exception as e:
                print(f"[WebSocket] Error sending update for {search_id}: {e}")

manager = WebSocketManager()

app = FastAPI(
    title="Xeno Search Service",
    description="An advanced web search and processing service.",
    version="0.1.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.post("/api/xeno-search-internal", response_model=SearchResponse)
async def process_search(request: SearchRequest):
    print(f"Received search request: Query='{request.query}', Type='{request.search_type}', NumResults='{request.num_results}'")
    
    if not request.query:
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    try:
        # Perform initial search
        initial_sources: List[Source] = await perform_search(request.query, request.search_type, request.num_results)
        
        # Determine if we need to perform deep search
        if request.search_type == "deep" and initial_sources:
            print(f"🔍 Performing deep search for query: '{request.query}'")
            # Perform deep search analysis
            all_sources, comprehensive_summary = await perform_deep_search_analysis(request.query, initial_sources)
            
            return SearchResponse(
                query=request.query,
                search_type=request.search_type,
                summary=comprehensive_summary,
                sources=all_sources
            )
        else:
            # Standard search processing (existing logic)
            overall_summary_text = None
            if initial_sources:
                # Collect individual summaries
                individual_summaries = [s.summary for s in initial_sources if s.summary and s.summary.strip()]
                
                if individual_summaries:
                    combined_summaries_text = "\n\n".join(individual_summaries)
                    # Summarize the combined summaries to create a meta-summary
                    meta_min_length = max(30, len(individual_summaries) * 10)
                    meta_max_length = max(150, len(individual_summaries) * 50)
                    meta_max_length = min(meta_max_length, 400) 

                    print(f"Generating overall summary from {len(individual_summaries)} individual summaries. Combined length: {len(combined_summaries_text)}")
                    if len(combined_summaries_text) > 50:
                        overall_summary_text = await summarize_text_async(
                            combined_summaries_text,
                            min_length=meta_min_length,
                            max_length=meta_max_length
                        )
                        if overall_summary_text:
                            print(f"Overall summary generated. Length: {len(overall_summary_text)}")
                        else:
                            print("Meta-summarization returned None or failed. Falling back.")
                            overall_summary_text = f"Found {len(initial_sources)} sources. Review individual summaries for details."
                    else:
                        print("Combined individual summaries too short for meta-summarization. Using fallback.")
                        overall_summary_text = f"Found {len(initial_sources)} sources. Review individual summaries for details. Content may be too brief for an overall summary."
                else:
                    overall_summary_text = f"Found {len(initial_sources)} sources, but no individual summaries were generated. Content might be unavailable or too short."
            else:
                overall_summary_text = f"No content found for query: '{request.query}'."

            return SearchResponse(
                query=request.query,
                search_type=request.search_type,
                summary=overall_summary_text,
                sources=initial_sources
            )
            
    except Exception as e:
        # Log the exception for debugging
        print(f"Error during search processing in main.py: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.post("/api/start-deep-search")
async def start_deep_search(request: SearchRequest):
    """Start a deep search with WebSocket progress tracking"""
    search_id = f"search-{uuid.uuid4().hex[:12]}"
    
    # Start deep search in background
    asyncio.create_task(perform_deep_search_with_websocket(search_id, request))
    
    return {"search_id": search_id}

@app.websocket("/ws/deep-search/{search_id}")
async def deep_search_websocket(websocket: WebSocket, search_id: str):
    """WebSocket endpoint for real-time deep search progress updates"""
    await manager.connect(websocket, search_id)
    try:
        while True:
            # Keep connection alive and listen for any client messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, search_id)

async def perform_deep_search_with_websocket(search_id: str, request: SearchRequest):
    """Perform deep search with real-time WebSocket progress updates"""
    try:
        # Phase 1: Initializing (0-10%)
        await manager.send_progress_update(
            search_id, "initializing", 5, 
            "🔄 Initializing deep search...",
            {"query": request.query}
        )
        
        # Phase 2: Initial Search (10-25%)
        await manager.send_progress_update(
            search_id, "initial_search", 15,
            "🔍 Performing initial web search..."
        )
        
        initial_sources = await perform_search(request.query, "normal", request.num_results)
        
        await manager.send_progress_update(
            search_id, "initial_search", 25,
            f"✅ Found {len(initial_sources)} initial sources",
            {"sources_found": len(initial_sources)}
        )
        
        # Phase 3: Analyzing Sources (25-40%)
        await manager.send_progress_update(
            search_id, "analyzing_sources", 30,
            "📊 Analyzing source content for relevance..."
        )
        
        # Phase 4: Deep Search Analysis (40-80%)
        await manager.send_progress_update(
            search_id, "following_links", 45,
            "🌐 Following links and discovering additional content..."
        )
        
        # Perform the actual deep search
        all_sources, comprehensive_summary = await perform_deep_search_analysis(request.query, initial_sources)
        
        await manager.send_progress_update(
            search_id, "scraping_content", 65,
            f"📄 Successfully scraped {len(all_sources)} total sources"
        )
        
        # Phase 5: AI Processing (80-95%)
        await manager.send_progress_update(
            search_id, "generating_summaries", 85,
            "🤖 Generating AI summaries and analysis..."
        )
        
        # Phase 6: Completed (100%)
        final_results = {
            "query": request.query,
            "search_type": "deep",
            "summary": comprehensive_summary,
            "sources": [
                {
                    "url": str(source.url),
                    "title": source.title,
                    "snippet": source.snippet,
                    "summary": source.summary
                } for source in all_sources
            ]
        }
        
        await manager.send_progress_update(
            search_id, "completed", 100,
            f"✅ Deep search completed! Analyzed {len(all_sources)} sources.",
            {"final_results": final_results}
        )
        
    except Exception as e:
        print(f"[WebSocket] Error in deep search {search_id}: {e}")
        await manager.send_progress_update(
            search_id, "error", 0,
            f"❌ Search failed: {str(e)}",
            {"error": str(e)}
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def read_root():
    return {"message": "Welcome to Xeno Search Service"}

# To run this (after creating the directory structure and installing dependencies):
# Ensure you are in the 'xeno-search-service' directory.
# 1. Create and activate virtual environment (e.g., python -m venv venv; source venv/bin/activate or venv\Scripts\activate on Windows)
# 2. Install FastAPI and Uvicorn: pip install fastapi uvicorn[standard]
# 3. Run the server: uvicorn app.main:app --reload --port 8000
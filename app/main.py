from fastapi import FastAPI, HTTPException
from typing import List

from app.models.search_models import SearchRequest, SearchResponse, Source
from app.services.scraping_service import perform_search 

app = FastAPI(
    title="Xeno Search Service",
    description="An advanced web search and processing service.",
    version="0.1.0"
)

@app.post("/api/xeno-search-internal", response_model=SearchResponse)
async def process_search(request: SearchRequest):
    print(f"Received search request: Query='{request.query}', Type='{request.search_type}', NumResults='{request.num_results}'")
    
    if not request.query:
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    try:
        results = await perform_search(request.query, request.search_type, request.num_results)
        
        # For now, we don't have a separate summary generation step.
        # We can construct a basic summary or leave it for a later NLP stage.
        # Placeholder summary for now if results are found:
        summary_text = None
        if results:
            summary_text = f"Found {len(results)} sources for query: '{request.query}'. Content extracted."
        else:
            summary_text = f"No content found for query: '{request.query}'."

        return SearchResponse(
            query=request.query,
            search_type=request.search_type,
            summary=summary_text,
            sources=results
        )
    except Exception as e:
        # Log the exception for debugging
        print(f"Error during search processing in main.py: {e}")
        # Optionally, re-raise or return a more generic error response
        # For now, let's return a 500 error with the exception detail for easier debugging during development.
        # In production, you might want to obscure such details.
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

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
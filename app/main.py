from fastapi import FastAPI, HTTPException
from typing import List

from app.models.search_models import SearchRequest, SearchResponse, Source
from app.services.scraping_service import perform_search
from app.services.nlp_service import summarize_text_async  # Import for meta-summary
from app.services.deep_search_service import perform_deep_search_analysis  # Import deep search

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
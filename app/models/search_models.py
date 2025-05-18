from typing import List, Optional
from pydantic import BaseModel, HttpUrl

class SearchRequest(BaseModel):
    query: str
    search_type: str = "normal"  # "normal" or "deep"
    num_results: int = 5 # Number of initial search results to process

class Source(BaseModel):
    url: HttpUrl
    title: Optional[str] = None
    snippet: Optional[str] = None
    raw_text: Optional[str] = None # Full text if needed for deep analysis
    summary: Optional[str] = None # NLP-generated summary of the content

class SearchResponse(BaseModel):
    query: str
    search_type: str
    summary: Optional[str] = None
    sources: List[Source] = []
    error: Optional[str] = None
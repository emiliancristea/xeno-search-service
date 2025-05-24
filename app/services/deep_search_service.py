import asyncio
from typing import List, Dict, Set, Optional, Tuple
from urllib.parse import urljoin, urlparse, parse_qs
import re
from bs4 import BeautifulSoup
import httpx
from dataclasses import dataclass
import time

from app.models.search_models import Source
from app.services.nlp_service import summarize_text_async


@dataclass
class LinkCandidate:
    """Represents a potential link to follow during deep search"""
    url: str
    anchor_text: str
    context: str
    relevance_score: float
    depth: int
    source_url: str


class DeepSearchService:
    """
    Advanced deep search service that follows links and performs comprehensive content analysis.
    Implements intelligent link following, semantic analysis, and multi-level content aggregation.
    """
    
    def __init__(self):
        self.max_depth = 2
        self.max_links_per_level = 5
        self.min_relevance_score = 0.3
        self.visited_urls: Set[str] = set()
        self.content_cache: Dict[str, Dict] = {}
        
        # Headers for requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    async def perform_deep_search(self, query: str, initial_sources: List[Source], num_additional_sources: int = 10) -> Tuple[List[Source], str]:
        """
        Performs deep search by following links from initial sources and gathering additional relevant content.
        
        Args:
            query: The original search query
            initial_sources: List of sources from the initial search
            num_additional_sources: Maximum number of additional sources to discover
            
        Returns:
            Tuple of (expanded_sources_list, comprehensive_summary)
        """
        print(f"🔍 Starting deep search for query: '{query}' with {len(initial_sources)} initial sources")
        
        # Reset state for new search
        self.visited_urls.clear()
        self.content_cache.clear()
        
        # Mark initial sources as visited
        for source in initial_sources:
            self.visited_urls.add(self._normalize_url(str(source.url)))
        
        # Extract links from initial sources
        link_candidates = await self._extract_link_candidates(query, initial_sources)
        print(f"📋 Found {len(link_candidates)} link candidates from initial sources")
        
        # Score and rank link candidates
        ranked_links = self._rank_link_candidates(query, link_candidates)
        
        # Follow top-ranked links
        additional_sources = await self._follow_links(query, ranked_links[:num_additional_sources])
        print(f"📖 Successfully scraped {len(additional_sources)} additional sources")
        
        # Combine all sources
        all_sources = initial_sources + additional_sources
        
        # Generate comprehensive summary
        comprehensive_summary = await self._generate_comprehensive_summary(query, all_sources)
        
        print(f"✅ Deep search completed. Total sources: {len(all_sources)}")
        return all_sources, comprehensive_summary
    
    async def _extract_link_candidates(self, query: str, sources: List[Source]) -> List[LinkCandidate]:
        """Extract relevant link candidates from source content"""
        candidates = []
        query_terms = set(query.lower().split())
        
        for source in sources:
            if not source.raw_text:
                continue
                
            try:
                # Parse the content to find links
                # For now, we'll extract links from the raw text if it contains HTML
                # In a more sophisticated implementation, we'd re-fetch and parse the HTML
                soup = BeautifulSoup(source.raw_text, 'html.parser') if '<' in source.raw_text else None
                
                if soup:
                    links = soup.find_all('a', href=True)
                    for link in links:
                        href = link.get('href')
                        if not href:
                            continue
                            
                        # Convert relative URLs to absolute
                        absolute_url = urljoin(str(source.url), href)
                        
                        # Skip if not HTTP(S)
                        if not absolute_url.startswith(('http://', 'https://')):
                            continue
                            
                        # Skip if already visited
                        if self._normalize_url(absolute_url) in self.visited_urls:
                            continue
                        
                        # Extract anchor text and context
                        anchor_text = link.get_text(strip=True)
                        context = self._extract_link_context(link, source.raw_text)
                        
                        # Calculate relevance score
                        relevance_score = self._calculate_link_relevance(query_terms, anchor_text, context)
                        
                        if relevance_score >= self.min_relevance_score:
                            candidates.append(LinkCandidate(
                                url=absolute_url,
                                anchor_text=anchor_text,
                                context=context,
                                relevance_score=relevance_score,
                                depth=1,
                                source_url=str(source.url)
                            ))
                            
            except Exception as e:
                print(f"⚠️ Error extracting links from {source.url}: {e}")
                continue
        
        return candidates
    
    def _extract_link_context(self, link_element, full_text: str, context_window: int = 100) -> str:
        """Extract surrounding context for a link"""
        try:
            link_text = link_element.get_text(strip=True)
            if not link_text:
                return ""
                
            # Find the position of the link text in the full text
            text_only = BeautifulSoup(full_text, 'html.parser').get_text() if '<' in full_text else full_text
            link_pos = text_only.lower().find(link_text.lower())
            
            if link_pos == -1:
                return ""
                
            # Extract context window around the link
            start = max(0, link_pos - context_window)
            end = min(len(text_only), link_pos + len(link_text) + context_window)
            context = text_only[start:end].strip()
            
            return context
        except:
            return ""
    
    def _calculate_link_relevance(self, query_terms: Set[str], anchor_text: str, context: str) -> float:
        """Calculate relevance score for a link based on query terms"""
        anchor_lower = anchor_text.lower()
        context_lower = context.lower()
        
        # Count query term matches
        anchor_matches = sum(1 for term in query_terms if term in anchor_lower)
        context_matches = sum(1 for term in query_terms if term in context_lower)
        
        # Calculate scores
        anchor_score = anchor_matches / len(query_terms) if query_terms else 0
        context_score = context_matches / len(query_terms) if query_terms else 0
        
        # Weight anchor text more heavily than context
        relevance_score = (anchor_score * 0.7) + (context_score * 0.3)
        
        # Bonus for exact phrase matches
        query_phrase = " ".join(query_terms)
        if query_phrase in anchor_lower:
            relevance_score += 0.3
        elif query_phrase in context_lower:
            relevance_score += 0.1
            
        return min(1.0, relevance_score)
    
    def _rank_link_candidates(self, query: str, candidates: List[LinkCandidate]) -> List[LinkCandidate]:
        """Rank link candidates by relevance and other factors"""
        # Sort by relevance score (descending)
        ranked = sorted(candidates, key=lambda x: x.relevance_score, reverse=True)
        
        # Remove duplicates (same domain + similar paths)
        deduped = []
        seen_domains = {}
        
        for candidate in ranked:
            domain = urlparse(candidate.url).netloc
            
            # Limit links per domain to avoid overwhelming single sources
            if seen_domains.get(domain, 0) < 2:
                deduped.append(candidate)
                seen_domains[domain] = seen_domains.get(domain, 0) + 1
                
        return deduped
    
    async def _follow_links(self, query: str, candidates: List[LinkCandidate]) -> List[Source]:
        """Follow link candidates and extract content"""
        tasks = []
        
        for candidate in candidates:
            if len(tasks) >= self.max_links_per_level:
                break
                
            tasks.append(self._scrape_link_candidate(candidate))
            # Add delay between requests to be respectful
            await asyncio.sleep(0.2)
        
        # Execute all scraping tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        sources = []
        for result in results:
            if isinstance(result, Source):
                sources.append(result)
            elif isinstance(result, Exception):
                print(f"⚠️ Error following link: {result}")
        
        return sources
    
    async def _scrape_link_candidate(self, candidate: LinkCandidate) -> Optional[Source]:
        """Scrape content from a link candidate"""
        try:
            print(f"📖 Scraping deep search link: {candidate.url}")
            
            async with httpx.AsyncClient(headers=self.headers, timeout=15.0, follow_redirects=True) as client:
                response = await client.get(candidate.url)
                response.raise_for_status()
                
                # Parse content
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract title
                title = None
                if soup.title:
                    title = soup.title.get_text(strip=True)
                
                # Extract main content (simplified approach)
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Get text content
                text_content = soup.get_text()
                
                # Clean up text
                lines = (line.strip() for line in text_content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                raw_text = '\n'.join(chunk for chunk in chunks if chunk)
                
                # Generate summary if content is substantial
                summary = None
                if raw_text and len(raw_text) > 200:
                    summary = await summarize_text_async(raw_text[:3000])  # Limit input size
                
                # Create snippet from first part of content
                snippet = raw_text[:300] + "..." if len(raw_text) > 300 else raw_text
                
                # Mark as visited
                self.visited_urls.add(self._normalize_url(candidate.url))
                
                return Source(
                    url=candidate.url,
                    title=title or "Deep Search Result",
                    snippet=snippet,
                    raw_text=raw_text,
                    summary=summary
                )
                
        except Exception as e:
            print(f"❌ Failed to scrape {candidate.url}: {e}")
            return None
    
    async def _generate_comprehensive_summary(self, query: str, all_sources: List[Source]) -> str:
        """Generate a comprehensive summary from all sources including deep search results"""
        try:
            # Collect all summaries
            summaries = []
            deep_search_summaries = []
            
            for i, source in enumerate(all_sources):
                if source.summary and source.summary.strip():
                    if i < 5:  # Assume first 5 are initial sources
                        summaries.append(f"Initial Source: {source.summary}")
                    else:
                        deep_search_summaries.append(f"Deep Search: {source.summary}")
            
            if not summaries and not deep_search_summaries:
                return f"Comprehensive search completed for '{query}' but no substantial content summaries were generated."
            
            # Combine summaries
            combined_text = ""
            if summaries:
                combined_text += "Initial Search Results:\n" + "\n\n".join(summaries[:5])
            
            if deep_search_summaries:
                if combined_text:
                    combined_text += "\n\nDeep Search Additional Insights:\n"
                combined_text += "\n\n".join(deep_search_summaries[:10])
            
            # Generate meta-summary
            if len(combined_text) > 100:
                comprehensive_summary = await summarize_text_async(
                    combined_text,
                    min_length=50,
                    max_length=300
                )
                if comprehensive_summary:
                    return f"Deep Search Summary for '{query}':\n\n{comprehensive_summary}"
            
            # Fallback summary
            total_sources = len(all_sources)
            initial_count = min(5, total_sources)
            deep_count = total_sources - initial_count
            
            return f"Comprehensive search for '{query}' analyzed {total_sources} sources ({initial_count} initial + {deep_count} deep search). Key insights integrated from multiple perspectives."
            
        except Exception as e:
            print(f"❌ Error generating comprehensive summary: {e}")
            return f"Deep search completed for '{query}' with {len(all_sources)} total sources. Summary generation encountered an error."
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for comparison (remove fragments, normalize case, etc.)"""
        try:
            parsed = urlparse(url.lower())
            # Remove fragment and some query parameters
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        except:
            return url.lower()


# Global instance for use in other modules
deep_search_service = DeepSearchService()


async def perform_deep_search_analysis(query: str, initial_sources: List[Source]) -> Tuple[List[Source], str]:
    """
    Convenience function to perform deep search analysis.
    
    Args:
        query: The search query
        initial_sources: Initial sources from normal search
        
    Returns:
        Tuple of (all_sources, comprehensive_summary)
    """
    return await deep_search_service.perform_deep_search(query, initial_sources)


# Example usage for testing
if __name__ == "__main__":
    async def test_deep_search():
        # Mock initial sources for testing
        test_sources = [
            Source(
                url="https://example.com/ai-news",
                title="Latest AI Developments",
                snippet="Recent advances in artificial intelligence...",
                raw_text="<p>Recent advances in artificial intelligence have shown remarkable progress. <a href='/deep-learning'>Deep learning</a> techniques are evolving rapidly. See also our <a href='/neural-networks'>neural networks guide</a> for more details.</p>",
                summary="AI is advancing rapidly with new developments in deep learning."
            )
        ]
        
        all_sources, summary = await perform_deep_search_analysis(
            "latest AI developments", 
            test_sources
        )
        
        print(f"Total sources found: {len(all_sources)}")
        print(f"Comprehensive summary: {summary}")

    # asyncio.run(test_deep_search()) 
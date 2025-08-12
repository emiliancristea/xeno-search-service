import httpx
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Any
import asyncio
import urllib.parse
from readability import Document # For readability-lxml
from app.core.config import get_config

from app.models.search_models import Source #Pydantic model for structuring results
from app.services.nlp_service import summarize_text_async # Import the summarizer

# Standard headers to mimic a browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Cache-Control': 'max-age=0',
}

# Chosen SearXNG public instance - updated to a different public instance
SEARXNG_INSTANCE_URL = "https://searx.tiekoetter.com/search"

_config = get_config()

def _build_headers() -> dict:
    if _config.user_agent_rotation and _config.user_agents:
        import random
        ua = random.choice(list(_config.user_agents))
        h = dict(HEADERS)
        h['User-Agent'] = ua
        return h
    return HEADERS

async def _get_duckduckgo_html_results(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """
    Fetches search results from DuckDuckGo HTML version using httpx.
    Parses the basic HTML structure of html.duckduckgo.com.
    NOTE: This is fragile and DDG's HTML can change.
    """
    results: List[Dict[str, str]] = []
    search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(query)}"

    try:
        async with httpx.AsyncClient(headers=_build_headers(), timeout=10.0, follow_redirects=True) as client:
            print(f"Querying DuckDuckGo HTML: {search_url}")
            response = await client.get(search_url)
            response.raise_for_status()
            
            content = response.text
            print(f"--- DDG HTML Content Start (Length: {len(content)}) ---")
            print(content[:2000]) # DEBUG: Print first 2000 characters of HTML
            print("--- DDG HTML Content End ---")

            if len(content) < 500: # Arbitrary threshold, adjust as needed
                print(f"Warning: Content from DDG HTML for query '{query}' is very short (len: {len(content)}). May indicate an issue.")
                # print(content[:200]) # Optionally print short content for debugging
            
            soup = BeautifulSoup(content, 'lxml')
            
            # Selector for html.duckduckgo.com results (can change!)
            # Results are typically in divs with class 'result'
            # Link and title are in an <a> tag with class 'result__a'
            # Snippet is in a <td> with class 'result__snippet'
            result_items = soup.find_all('div', class_='web-result') # Updated class based on common DDG HTML structure
            if not result_items:
                # Fallback to older/simpler structure if 'web-result' is not found
                result_items = soup.select('div.result, table tr td + td') # a more general selector used previously
            
            if not result_items:
                print(f"No result items found using selectors in DDG HTML for query: '{query}'. HTML structure might have changed or page did not contain results.")
                return []
            else:
                print(f"Found {len(result_items)} potential result items in DDG HTML for query: '{query}'")

            for item in result_items[:num_results]:
                title_tag = item.find('a', class_='result__a')
                if not title_tag:
                    title_tag = item.find('a') # More general fallback for title link
                
                snippet_tag = item.find(class_='result__snippet')
                if not snippet_tag and title_tag: # if no direct snippet, try parent's next relevant tag
                    # This part is highly heuristic and likely to break
                    parent_td = title_tag.find_parent('td')
                    if parent_td and parent_td.next_sibling and hasattr(parent_td.next_sibling, 'get_text'):
                         snippet_tag = parent_td.next_sibling

                title = title_tag.get_text(strip=True) if title_tag else None
                url = title_tag['href'] if title_tag and title_tag.has_attr('href') else None
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else None
                
                # Decode DDG's redirected URLs (e.g., /l/?uddg=...)
                if url and url.startswith('/l/'):
                    parsed_url = urllib.parse.urlparse(url)
                    qs = urllib.parse.parse_qs(parsed_url.query)
                    if 'uddg' in qs and qs['uddg']:
                        url = qs['uddg'][0]
                    else:
                        print(f"Could not extract actual URL from DDG redirect: {url}")
                        url = None # Mark as None if extraction fails
                elif url and not url.startswith('http'):
                    # If it's a relative URL not starting with /l/, prepend duckduckgo.com (less common for results)
                    # This is a guess, most result URLs from DDG HTML are either /l/ or absolute
                    # url = f"https://duckduckgo.com{url}"
                    print(f"Found relative URL not in expected format: {url}")
                    url = None # Treat unexpected relative URLs as invalid for now

                if title and url:
                    results.append({'title': title, 'url': url, 'snippet': snippet or ""})
                    if len(results) >= num_results:
                        break
            return results
            
    except httpx.HTTPStatusError as e:
        print(f"HTTP error {e.response.status_code} while querying DDG HTML for query '{query}': {e.response.text[:200]}...")
        return []
    except httpx.RequestError as e:
        print(f"Request error while querying DDG HTML for query '{query}': {e}")
        return []
    except Exception as e:
        print(f"Error processing DDG HTML results for query '{query}': {e}")
        return []


async def _scrape_page_content(url: str) -> Dict[str, Optional[str]]:
    """
    Scrapes the title and extracts main textual content from a given URL
    using httpx and readability-lxml (enhanced extraction without Playwright).
    Also generates a summary if content is found.
    """
    print(f"Attempting to scrape with httpx+readability: {url}")
    page_summary: Optional[str] = None # Initialize summary

    try:
        async with httpx.AsyncClient(headers=_build_headers(), follow_redirects=True, timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            html_content = response.text
            final_url = str(response.url)  # Capture final URL after redirects
            
            # Use readability-lxml to extract main content
            doc = Document(html_content)
            title = doc.title()
            main_content_html = doc.summary()

            # Parse the cleaned HTML with BeautifulSoup to get structured text
            soup = BeautifulSoup(main_content_html, 'lxml')
            
            # Extract title from original HTML as fallback if readability doesn't get a good one
            if not title:
                original_soup = BeautifulSoup(html_content, 'lxml')
                title = original_soup.title.string if original_soup.title else None
            
            # Extract text, trying to maintain some structure (paragraphs)
            text_parts = []
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'article']):
                text_parts.append(element.get_text(separator=' ', strip=True))
            
            raw_text = "\n\n".join(text_parts) if text_parts else None

            # Generate summary if raw_text is available
            if raw_text and raw_text.strip():
                print(f"Attempting to summarize content for: {final_url}")
                page_summary = await summarize_text_async(raw_text)
                if page_summary:
                    print(f"Successfully summarized content for: {final_url}")
                else:
                    print(f"Summarization returned None or failed for: {final_url}")
            else:
                print(f"No substantial raw text to summarize for: {final_url}")
            
            print(f"Successfully scraped with httpx+readability: {title} from {final_url}")
            return {'url': final_url, 'title': title, 'raw_text': raw_text, 'summary': page_summary}

    except httpx.TimeoutException:
        print(f"Timeout while scraping URL: {url}")
        return {'url': url, 'title': None, 'raw_text': None, 'summary': None, 'error': 'Timeout'}
    except httpx.RequestError as e:
        print(f"Request error while scraping URL {url}: {e}")
        return {'url': url, 'title': None, 'raw_text': None, 'summary': None, 'error': f'Request error: {e}'}
    except Exception as e:
        # This will catch other errors including Readability issues
        print(f"An unexpected error occurred while scraping {url}: {e}")
        error_type = type(e).__name__
        return {'url': url, 'title': None, 'raw_text': None, 'summary': None, 'error': f'{error_type}: {str(e)[:200]}'}


async def perform_search(query: str, search_type: str, num_results: int) -> List[Source]:
    """
    Orchestrates the web search process.
    1. Gets initial URLs from DuckDuckGo HTML.
    2. Scrapes content from these URLs.
    """
    print(f"Performing '{search_type}' search for query: '{query}' (target results: {num_results})")

    search_engine_results = await _get_duckduckgo_html_results(query, num_results)

    if not search_engine_results:
        print("No results from DuckDuckGo HTML.")
        return []

    print(f"Found {len(search_engine_results)} potential sources from DuckDuckGo HTML.")

    scraped_sources: List[Source] = []
    tasks = []
    for result in search_engine_results:
        print(f"Queueing scrape for: {result['url']}")
        tasks.append(_scrape_page_content(result['url']))
        await asyncio.sleep(0.1)

    scraped_pages_results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, page_data_or_exc in enumerate(scraped_pages_results):
        original_url = search_engine_results[i]['url']
        original_title = search_engine_results[i]['title']
        original_snippet = search_engine_results[i].get('snippet')

        if isinstance(page_data_or_exc, Exception):
            print(f"Exception during scraping {original_url}: {page_data_or_exc}")
            scraped_sources.append(Source(
                url=original_url,
                title=original_title,
                snippet=original_snippet or f"Failed to scrape content: {page_data_or_exc}",
                raw_text=None,
                summary=None
            ))
        elif page_data_or_exc.get('error'):
            print(f"Error scraping {original_url}: {page_data_or_exc.get('error')}")
            scraped_sources.append(Source(
                url=original_url,
                title=page_data_or_exc.get('title') or original_title,
                snippet=original_snippet or f"Failed to scrape content: {page_data_or_exc.get('error')}",
                raw_text=None,
                summary=None
            ))
        elif page_data_or_exc:
            raw_text_content = page_data_or_exc.get('raw_text')
            current_snippet = (raw_text_content[:250] + '...') if raw_text_content else (original_snippet or "No content extracted.")
            print(f"Successfully scraped: {page_data_or_exc.get('title', 'No title')} from {page_data_or_exc.get('url')}")
            scraped_sources.append(Source(
                url=page_data_or_exc.get('url', original_url),
                title=page_data_or_exc.get('title') or original_title,
                raw_text=raw_text_content,
                snippet=current_snippet,
                summary=page_data_or_exc.get('summary')
            ))
        
        if len(scraped_sources) >= num_results:
            break

    print(f"Collected {len(scraped_sources)} sources after scraping.")
    return scraped_sources


# Example usage (for testing this module directly):
if __name__ == '__main__':
    async def main_test():
        test_query = "latest AI advancements"
        print(f"Testing Xeno Search module with query: '{test_query}'")
        results = await perform_search(query=test_query, search_type="normal", num_results=3)
        if results:
            print(f"\n--- Test Search Results for '{test_query}' ---")
            for i, source in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"  Title: {source.title}")
                print(f"  URL: {source.url}")
                print(f"  Snippet: {source.snippet}")
                print(f"  Summary: {source.summary}")
        else:
            print("No results returned from the test search.")

    asyncio.run(main_test())
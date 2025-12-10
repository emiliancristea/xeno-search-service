import httpx
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Any
import asyncio
import urllib.parse
from readability import Document  # For readability-lxml
import structlog

from app.models.search_models import Source  # Pydantic model for structuring results
from app.services.enhanced_nlp_service import get_enhanced_nlp_service
from app.core.security import validate_url_for_ssrf, sanitize_url

logger = structlog.get_logger(__name__)

# Standard headers to mimic a browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
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


async def _summarize_text(text: str) -> Optional[str]:
    """Summarize text using enhanced NLP service"""
    try:
        nlp_service = await get_enhanced_nlp_service()
        result = await nlp_service.summarize_text_advanced(text, max_length=150, min_length=30)
        return result.summary if result else None
    except Exception as e:
        logger.warning("Summarization failed", error=str(e))
        return None

async def _get_duckduckgo_html_results(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """
    Fetches search results from DuckDuckGo HTML version using httpx.
    Parses the basic HTML structure of html.duckduckgo.com.
    NOTE: This is fragile and DDG's HTML can change.
    """
    results: List[Dict[str, str]] = []
    search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(query)}"

    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=10.0, follow_redirects=True) as client:
            logger.info("Querying DuckDuckGo HTML", url=search_url, query=query)
            response = await client.get(search_url)
            response.raise_for_status()

            content = response.text
            logger.debug("DDG HTML response received", content_length=len(content))

            if len(content) < 500:
                logger.warning("DDG HTML content unusually short",
                             query=query, content_length=len(content))

            soup = BeautifulSoup(content, 'lxml')

            # Selector for html.duckduckgo.com results (can change!)
            result_items = soup.find_all('div', class_='web-result')
            if not result_items:
                # Fallback to older/simpler structure
                result_items = soup.select('div.result, table tr td + td')

            if not result_items:
                logger.warning("No result items found in DDG HTML",
                             query=query, hint="HTML structure might have changed")
                return []

            logger.debug("Found DDG result items", count=len(result_items), query=query)

            for item in result_items[:num_results]:
                title_tag = item.find('a', class_='result__a')
                if not title_tag:
                    title_tag = item.find('a')

                snippet_tag = item.find(class_='result__snippet')
                if not snippet_tag and title_tag:
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
                        logger.debug("Could not extract URL from DDG redirect", redirect_url=url)
                        url = None
                elif url and not url.startswith('http'):
                    logger.debug("Found unexpected relative URL", url=url)
                    url = None

                if title and url:
                    # Validate URL for SSRF protection
                    is_safe, _ = validate_url_for_ssrf(url)
                    if is_safe:
                        results.append({'title': title, 'url': url, 'snippet': snippet or ""})
                    else:
                        logger.debug("Skipped unsafe URL from search results", url=url)

                    if len(results) >= num_results:
                        break

            return results

    except httpx.HTTPStatusError as e:
        logger.error("HTTP error querying DDG HTML",
                    status_code=e.response.status_code, query=query)
        return []
    except httpx.RequestError as e:
        logger.error("Request error querying DDG HTML", query=query, error=str(e))
        return []
    except Exception as e:
        logger.error("Error processing DDG HTML results", query=query, error=str(e))
        return []


async def _scrape_page_content(url: str) -> Dict[str, Optional[str]]:
    """
    Scrapes the title and extracts main textual content from a given URL
    using httpx and readability-lxml (enhanced extraction without Playwright).
    Also generates a summary if content is found.
    Includes SSRF protection.
    """
    # Validate URL for SSRF protection
    is_safe, error_msg = validate_url_for_ssrf(url)
    if not is_safe:
        logger.warning("SSRF protection blocked URL", url=url, reason=error_msg)
        return {'url': url, 'title': None, 'raw_text': None, 'summary': None, 'error': f'Blocked: {error_msg}'}

    logger.info("Scraping page content", url=url)
    page_summary: Optional[str] = None

    try:
        async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True, timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()

            html_content = response.text
            final_url = str(response.url)

            # Validate final URL after redirects
            is_safe, error_msg = validate_url_for_ssrf(final_url)
            if not is_safe:
                logger.warning("SSRF protection blocked redirect", original_url=url, final_url=final_url)
                return {'url': url, 'title': None, 'raw_text': None, 'summary': None, 'error': f'Redirect blocked: {error_msg}'}

            # Use readability-lxml to extract main content
            doc = Document(html_content)
            title = doc.title()
            main_content_html = doc.summary()

            # Parse the cleaned HTML with BeautifulSoup to get structured text
            soup = BeautifulSoup(main_content_html, 'lxml')

            # Extract title from original HTML as fallback
            if not title:
                original_soup = BeautifulSoup(html_content, 'lxml')
                title = original_soup.title.string if original_soup.title else None

            # Extract text, maintaining structure
            text_parts = []
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'article']):
                text_parts.append(element.get_text(separator=' ', strip=True))

            raw_text = "\n\n".join(text_parts) if text_parts else None

            # Generate summary if raw_text is available
            if raw_text and raw_text.strip():
                logger.debug("Summarizing scraped content", url=final_url)
                page_summary = await _summarize_text(raw_text)
                if page_summary:
                    logger.debug("Content summarized successfully", url=final_url)
                else:
                    logger.debug("Summarization returned None", url=final_url)
            else:
                logger.debug("No substantial text to summarize", url=final_url)

            logger.info("Page scraped successfully", url=final_url, title=title)
            return {'url': final_url, 'title': title, 'raw_text': raw_text, 'summary': page_summary}

    except httpx.TimeoutException:
        logger.warning("Timeout scraping URL", url=url)
        return {'url': url, 'title': None, 'raw_text': None, 'summary': None, 'error': 'Timeout'}
    except httpx.RequestError as e:
        logger.warning("Request error scraping URL", url=url, error=str(e))
        return {'url': url, 'title': None, 'raw_text': None, 'summary': None, 'error': f'Request error: {e}'}
    except Exception as e:
        logger.error("Unexpected error scraping URL", url=url, error=str(e), error_type=type(e).__name__)
        error_type = type(e).__name__
        return {'url': url, 'title': None, 'raw_text': None, 'summary': None, 'error': f'{error_type}: {str(e)[:200]}'}


async def perform_search(query: str, search_type: str, num_results: int) -> List[Source]:
    """
    Orchestrates the web search process.
    1. Gets initial URLs from DuckDuckGo HTML.
    2. Scrapes content from these URLs.
    """
    logger.info("Performing search", search_type=search_type, query=query, target_results=num_results)

    search_engine_results = await _get_duckduckgo_html_results(query, num_results)

    if not search_engine_results:
        logger.warning("No results from DuckDuckGo HTML", query=query)
        return []

    logger.info("Found potential sources", count=len(search_engine_results), query=query)

    scraped_sources: List[Source] = []
    tasks = []
    for result in search_engine_results:
        logger.debug("Queueing scrape", url=result['url'])
        tasks.append(_scrape_page_content(result['url']))
        await asyncio.sleep(0.1)

    scraped_pages_results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, page_data_or_exc in enumerate(scraped_pages_results):
        original_url = search_engine_results[i]['url']
        original_title = search_engine_results[i]['title']
        original_snippet = search_engine_results[i].get('snippet')

        if isinstance(page_data_or_exc, Exception):
            logger.warning("Exception during scraping", url=original_url, error=str(page_data_or_exc))
            scraped_sources.append(Source(
                url=original_url,
                title=original_title,
                snippet=original_snippet or f"Failed to scrape content: {page_data_or_exc}",
                raw_text=None,
                summary=None
            ))
        elif page_data_or_exc.get('error'):
            logger.warning("Error scraping", url=original_url, error=page_data_or_exc.get('error'))
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
            logger.debug("Successfully scraped", url=page_data_or_exc.get('url'), title=page_data_or_exc.get('title'))
            scraped_sources.append(Source(
                url=page_data_or_exc.get('url', original_url),
                title=page_data_or_exc.get('title') or original_title,
                raw_text=raw_text_content,
                snippet=current_snippet,
                summary=page_data_or_exc.get('summary')
            ))

        if len(scraped_sources) >= num_results:
            break

    logger.info("Search completed", sources_collected=len(scraped_sources), query=query)
    return scraped_sources


# Example usage (for testing this module directly):
if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    async def main_test():
        test_query = "latest AI advancements"
        logger.info("Testing Xeno Search module", query=test_query)
        results = await perform_search(query=test_query, search_type="normal", num_results=3)
        if results:
            logger.info("Test search completed", result_count=len(results))
            for i, source in enumerate(results):
                logger.info(f"Result {i+1}", title=source.title, url=source.url)
        else:
            logger.warning("No results returned from test search")

    asyncio.run(main_test())
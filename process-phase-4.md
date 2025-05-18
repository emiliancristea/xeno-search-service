# Process Log: Phase 4 - Advanced Scraping & NLP Capabilities

**Overall Phase Goal:** Enhance the Python `xeno-search-service` with more robust scraping techniques, advanced content extraction, and Natural Language Processing capabilities like summarization and keyword/entity extraction.

---

**Current Date:** May 19, 2025

## Part 1: Text Summarization for Scraped Content

### Step 1.1: Add NLP Dependencies

*   **Action:** Add `transformers`, `torch` (CPU version), and `sentencepiece` to `requirements.txt`.
*   **Rationale:** These libraries are required to download and use pre-trained summarization models from the Hugging Face Hub.
    *   `transformers`: Provides the high-level API for models and tokenizers.
    *   `torch`: The deep learning framework many Hugging Face models are built on.
    *   `sentencepiece`: A tokenizer library often used by models like T5.
*   **Status:** Done (Added in previous step).

### Step 1.2: Install New Dependencies

*   **Action:** Run `pip install -r requirements.txt` to install the newly added libraries.
*   **Rationale:** Makes the libraries available in the Python environment.
*   **Status:** Done.

### Step 1.3: Create `app/services/nlp_service.py`

*   **Action:** Create a new file `app/services/nlp_service.py`.
*   **Rationale:** This module will encapsulate NLP-related logic, starting with summarization, keeping it separate from scraping and API concerns.
*   **Status:** Done.

### Step 1.4: Implement Summarization Function

*   **File to Tweak:** `app/services/nlp_service.py`.
*   **Action:**
    *   Define an asynchronous function `async def summarize_text(text: str, model_name: str = "facebook/bart-large-cnn", max_length: int = 150, min_length: int = 30) -> str`.
    *   Inside the function, load a pre-trained summarization model and tokenizer (e.g., `facebook/bart-large-cnn` or `t5-small`) using `transformers.pipeline`.
    *   Process the input text to generate a summary.
    *   Handle potential errors (e.g., text too short, model errors).
    *   Consider running the model inference in a separate thread for async compatibility if it's blocking.
*   **Rationale:** Provides the core text summarization capability.
*   **Status:** Done.

### Step 1.5: Update `Source` Pydantic Model

*   **File to Tweak:** `app/models/search_models.py`.
*   **Action:** Add an optional `summary: Optional[str] = None` field to the `Source` Pydantic model.
*   **Rationale:** Allows each scraped source to have its own summary.
*   **Status:** Done.

### Step 1.6: Integrate Summarization into `scraping_service.py`

*   **File to Tweak:** `app/services/scraping_service.py`.
*   **Action:**
    *   Import the `summarize_text_async` function from `app.services.nlp_service`.
    *   In the `_scrape_page_content` function, after successfully extracting `raw_text` and `title`:
        *   If `raw_text` is substantial, call `await summarize_text_async(raw_text)`.
        *   Store the returned summary in the dictionary returned by `_scrape_page_content`.
    *   Modify the `perform_search` function to correctly populate the `Source` model with this new summary from the dictionary.
*   **Rationale:** Enriches the data for each source with a concise summary.
*   **Status:** Done.

### Step 1.7: Update Overall Summary in `SearchResponse` (Optional for now)

*   **File to Tweak:** `app/main.py` (or `scraping_service.py` if summary logic moves there).
*   **Action:** Consider how the individual source summaries contribute to the overall `summary` field in the `SearchResponse`. For now, the existing overall summary logic (e.g., "Found X sources...") can remain, or it could be enhanced to be a meta-summary.
*   **Rationale:** Improves the top-level summary provided by the API.
*   **Status:** Pending (Lower priority for initial summarization implementation).

### Step 1.8: Testing

*   **Action:** Test the search functionality thoroughly.
    *   Verify that summaries are being generated for scraped content.
    *   Check the quality of the summaries.
    *   Observe performance (summarization can be computationally intensive).
*   **Rationale:** Ensures the new summarization feature works correctly and efficiently.
*   **Status:** Pending.

---
(Further parts of Phase 4, like Deep Search, Keyword Extraction, etc., will be added later)

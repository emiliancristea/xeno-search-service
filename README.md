# Xeno Search Service

This service provides advanced web searching and processing capabilities, including web scraping, content extraction, and NLP-based summarization. It is designed to be called by other backend services to augment them with real-time web information.

## Features (Planned)

*   Receives search queries via an API endpoint.
*   Scrapes search engines for initial URLs.
*   Fetches and parses content from target web pages (handles static and dynamic content).
*   Extracts main textual content and metadata.
*   Performs NLP tasks (summarization, keyword extraction).
*   Supports "normal" and "deep" search modes.
*   Built with Python and FastAPI.

## Setup and Running

1.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    # source venv/bin/activate # Linux/macOS
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the FastAPI application:**
    ```bash
    uvicorn app.main:app --reload --port 8000
    ```
    The service will be available at `http://127.0.0.1:8000`.
    API documentation can be found at `http://127.0.0.1:8000/docs`.

## Project Structure

```
xeno-search-service/
├── app/                  # Main application code
│   ├── __init__.py
│   ├── main.py           # FastAPI app definition and endpoints
│   ├── core/             # Core logic (config, etc.)
│   ├── models/           # Pydantic models for request/response
│   └── services/         # Business logic (scraping, NLP)
├── venv/                 # Python virtual environment (ignored by Git)
├── requirements.txt      # Python dependencies
├── .gitignore            # Specifies intentionally untracked files
└── README.md             # This file
``` 
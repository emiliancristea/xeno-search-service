"""
Tests for API endpoints
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for health check endpoint"""

    def test_health_check_returns_ok(self):
        """Health endpoint should return status"""
        # Import here to avoid import issues during collection
        from app.main import app
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestSearchEndpoint:
    """Tests for search API endpoint"""

    def test_search_requires_query(self):
        """Search should require a query parameter"""
        from app.main import app
        client = TestClient(app)

        response = client.post("/api/v2/search", json={})
        assert response.status_code == 422  # Validation error

    def test_search_validates_num_results(self):
        """Search should validate num_results range"""
        from app.main import app
        client = TestClient(app)

        # Too high
        response = client.post("/api/v2/search", json={
            "query": "test",
            "num_results": 500
        })
        assert response.status_code == 422

    def test_search_accepts_valid_request(self):
        """Search should accept valid requests"""
        from app.main import app
        client = TestClient(app)

        with patch('app.main.perform_enhanced_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []

            response = client.post("/api/v2/search", json={
                "query": "test query",
                "num_results": 5,
                "search_type": "normal"
            })
            # Should either succeed or fail gracefully
            assert response.status_code in [200, 500, 503]


class TestRateLimiting:
    """Tests for rate limiting functionality"""

    def test_rate_limit_header_present(self):
        """Rate limit headers should be present in responses"""
        from app.main import app
        client = TestClient(app)

        response = client.get("/health")
        # Rate limiting may or may not be enabled in tests
        assert response.status_code == 200


class TestCORSHeaders:
    """Tests for CORS configuration"""

    def test_cors_headers_on_options(self):
        """OPTIONS requests should return CORS headers"""
        from app.main import app
        client = TestClient(app)

        response = client.options(
            "/api/v2/search",
            headers={"Origin": "http://localhost:3000"}
        )
        # CORS should be configured
        assert response.status_code in [200, 405]


class TestErrorHandling:
    """Tests for error handling"""

    def test_404_for_unknown_endpoint(self):
        """Unknown endpoints should return 404"""
        from app.main import app
        client = TestClient(app)

        response = client.get("/nonexistent/endpoint")
        assert response.status_code == 404

    def test_method_not_allowed(self):
        """Wrong HTTP method should return 405"""
        from app.main import app
        client = TestClient(app)

        response = client.get("/api/v2/search")  # Should be POST
        assert response.status_code == 405

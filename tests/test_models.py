"""
Tests for Pydantic models
"""
import pytest
from datetime import datetime
from app.models.search_models import (
    Source,
    SourceMetadata,
    SearchRequest,
    SearchResponse,
    SearchType,
    ContentCategory,
    SearchStats,
)


class TestSourceModel:
    """Tests for Source model"""

    def test_create_minimal_source(self):
        """Should create source with minimal fields"""
        source = Source(url="https://example.com")
        assert source.url == "https://example.com"
        assert source.metadata is not None  # Default metadata

    def test_create_full_source(self, sample_source):
        """Should create source with all fields"""
        assert sample_source.url == "https://example.com/test"
        assert sample_source.title == "Test Title"
        assert sample_source.snippet == "Test snippet content"
        assert sample_source.raw_text is not None
        assert sample_source.summary == "Test summary"

    def test_get_domain(self, sample_source):
        """Should extract domain from URL"""
        domain = sample_source.get_domain()
        assert domain == "example.com"

    def test_get_reading_time(self):
        """Should calculate reading time"""
        # 200 words = 1 minute at 200 WPM
        text = " ".join(["word"] * 400)
        source = Source(url="https://example.com", raw_text=text)
        reading_time = source.get_reading_time()
        assert reading_time == 2

    def test_validate_scores_in_range(self):
        """Should accept scores in valid range"""
        source = Source(
            url="https://example.com",
            confidence_score=0.5,
            quality_score=0.8,
            relevance_score=1.0
        )
        assert source.confidence_score == 0.5
        assert source.quality_score == 0.8
        assert source.relevance_score == 1.0

    def test_validate_scores_at_boundaries(self):
        """Should accept scores at boundaries"""
        source = Source(
            url="https://example.com",
            confidence_score=0.0,
            quality_score=1.0
        )
        assert source.confidence_score == 0.0
        assert source.quality_score == 1.0

    def test_reject_invalid_scores(self):
        """Should reject scores outside valid range"""
        with pytest.raises(ValueError):
            Source(url="https://example.com", confidence_score=1.5)

        with pytest.raises(ValueError):
            Source(url="https://example.com", quality_score=-0.1)

    def test_to_dict(self, sample_source):
        """Should convert to dictionary"""
        result = sample_source.to_dict()
        assert isinstance(result, dict)
        assert result["url"] == "https://example.com/test"


class TestSearchRequest:
    """Tests for SearchRequest model"""

    def test_create_minimal_request(self):
        """Should create request with minimal fields"""
        request = SearchRequest(query="test query")
        assert request.query == "test query"
        assert request.search_type == SearchType.NORMAL
        assert request.num_results == 10

    def test_validate_query_not_empty(self):
        """Should reject empty query"""
        with pytest.raises(ValueError):
            SearchRequest(query="")

        with pytest.raises(ValueError):
            SearchRequest(query="   ")

    def test_validate_query_strips_whitespace(self):
        """Should strip whitespace from query"""
        request = SearchRequest(query="  test query  ")
        assert request.query == "test query"

    def test_validate_language_format(self):
        """Should accept valid language codes"""
        request = SearchRequest(query="test", language="en")
        assert request.language == "en"

    def test_validate_num_results_range(self):
        """Should accept num_results in valid range"""
        request = SearchRequest(query="test", num_results=50)
        assert request.num_results == 50

    def test_reject_invalid_num_results(self):
        """Should reject num_results outside range"""
        with pytest.raises(ValueError):
            SearchRequest(query="test", num_results=0)

        with pytest.raises(ValueError):
            SearchRequest(query="test", num_results=101)

    def test_all_search_types(self):
        """Should accept all search types"""
        for search_type in SearchType:
            request = SearchRequest(query="test", search_type=search_type)
            assert request.search_type == search_type


class TestSearchResponse:
    """Tests for SearchResponse model"""

    def test_create_response(self):
        """Should create response with required fields"""
        source = Source(url="https://example.com")
        response = SearchResponse(
            query="test",
            search_type=SearchType.NORMAL,
            sources=[source]
        )
        assert response.query == "test"
        assert len(response.sources) == 1

    def test_calculate_overall_confidence(self):
        """Should calculate overall confidence"""
        sources = [
            Source(url="https://example1.com", confidence_score=0.9),
            Source(url="https://example2.com", confidence_score=0.7),
        ]
        response = SearchResponse(
            query="test",
            search_type=SearchType.NORMAL,
            sources=sources
        )
        confidence = response.calculate_overall_confidence()
        assert 0 < confidence < 1

    def test_calculate_result_diversity(self):
        """Should calculate result diversity"""
        sources = [
            Source(url="https://example.com/1", content_category=ContentCategory.NEWS),
            Source(url="https://other.org/2", content_category=ContentCategory.ACADEMIC),
        ]
        response = SearchResponse(
            query="test",
            search_type=SearchType.NORMAL,
            sources=sources
        )
        diversity = response.calculate_result_diversity()
        assert diversity == 1.0  # All different domains and categories

    def test_get_top_domains(self):
        """Should get top domains"""
        sources = [
            Source(url="https://example.com/1"),
            Source(url="https://example.com/2"),
            Source(url="https://other.org/3"),
        ]
        response = SearchResponse(
            query="test",
            search_type=SearchType.NORMAL,
            sources=sources
        )
        top_domains = response.get_top_domains(limit=2)
        assert len(top_domains) == 2
        assert top_domains[0][0] == "example.com"
        assert top_domains[0][1] == 2

    def test_filter_by_confidence(self):
        """Should filter sources by confidence"""
        sources = [
            Source(url="https://high.com", confidence_score=0.9),
            Source(url="https://low.com", confidence_score=0.3),
            Source(url="https://medium.com", confidence_score=0.6),
        ]
        response = SearchResponse(
            query="test",
            search_type=SearchType.NORMAL,
            sources=sources
        )
        filtered = response.filter_by_confidence(0.5)
        assert len(filtered.sources) == 2


class TestSourceMetadata:
    """Tests for SourceMetadata model"""

    def test_create_minimal_metadata(self):
        """Should create metadata with defaults"""
        metadata = SourceMetadata()
        assert metadata.is_duplicate is False
        assert metadata.engine_metadata == {}

    def test_create_full_metadata(self):
        """Should create metadata with all fields"""
        metadata = SourceMetadata(
            search_engine="searxng",
            engine_rank=1,
            confidence_score=0.85,
            content_category=ContentCategory.NEWS,
            language="en",
            domain="example.com"
        )
        assert metadata.search_engine == "searxng"
        assert metadata.engine_rank == 1
        assert metadata.confidence_score == 0.85


class TestSearchStats:
    """Tests for SearchStats model"""

    def test_create_stats(self):
        """Should create search statistics"""
        stats = SearchStats(
            total_results=10,
            processing_time_ms=150,
            engines_used=["searxng", "duckduckgo"]
        )
        assert stats.total_results == 10
        assert stats.processing_time_ms == 150
        assert len(stats.engines_used) == 2

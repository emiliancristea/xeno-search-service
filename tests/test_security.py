"""
Tests for security module - SSRF protection and URL validation
"""
import pytest
from app.core.security import (
    validate_url_for_ssrf,
    is_ip_blocked,
    is_hostname_blocked,
    sanitize_url,
    sanitize_search_query,
)


class TestSSRFProtection:
    """Tests for SSRF protection functionality"""

    def test_valid_urls_pass_validation(self, valid_urls):
        """Valid public URLs should pass SSRF validation"""
        for url in valid_urls:
            is_safe, error = validate_url_for_ssrf(url)
            assert is_safe, f"URL {url} should be valid but got error: {error}"

    def test_blocked_urls_fail_validation(self, blocked_urls):
        """Blocked URLs (localhost, private IPs) should fail validation"""
        for url in blocked_urls:
            is_safe, error = validate_url_for_ssrf(url)
            assert not is_safe, f"URL {url} should be blocked"

    def test_empty_url_fails(self):
        """Empty URL should fail validation"""
        is_safe, error = validate_url_for_ssrf("")
        assert not is_safe
        assert "empty" in error.lower()

    def test_invalid_scheme_fails(self):
        """Non-HTTP(S) schemes should fail"""
        invalid_urls = [
            "ftp://example.com",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
        ]
        for url in invalid_urls:
            is_safe, error = validate_url_for_ssrf(url)
            assert not is_safe, f"URL {url} should be blocked"

    def test_url_too_long_fails(self):
        """URLs exceeding max length should fail"""
        long_url = "https://example.com/" + "a" * 3000
        is_safe, error = validate_url_for_ssrf(long_url)
        assert not is_safe
        assert "length" in error.lower()


class TestIPBlocking:
    """Tests for IP address blocking"""

    def test_private_ipv4_ranges_blocked(self):
        """Private IPv4 ranges should be blocked"""
        private_ips = [
            "10.0.0.1",
            "10.255.255.255",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.0.1",
            "192.168.255.255",
        ]
        for ip in private_ips:
            assert is_ip_blocked(ip), f"Private IP {ip} should be blocked"

    def test_loopback_blocked(self):
        """Loopback addresses should be blocked"""
        loopback_ips = ["127.0.0.1", "127.0.0.2", "127.255.255.255"]
        for ip in loopback_ips:
            assert is_ip_blocked(ip), f"Loopback IP {ip} should be blocked"

    def test_public_ips_allowed(self):
        """Public IP addresses should be allowed"""
        public_ips = ["8.8.8.8", "1.1.1.1", "93.184.216.34"]
        for ip in public_ips:
            assert not is_ip_blocked(ip), f"Public IP {ip} should be allowed"

    def test_cloud_metadata_ip_blocked(self):
        """Cloud metadata endpoint IP should be blocked"""
        assert is_ip_blocked("169.254.169.254")


class TestHostnameBlocking:
    """Tests for hostname blocking"""

    def test_localhost_blocked(self):
        """Localhost variants should be blocked"""
        assert is_hostname_blocked("localhost")
        assert is_hostname_blocked("localhost.localdomain")

    def test_wildcard_patterns_blocked(self):
        """Wildcard blocked patterns should work"""
        assert is_hostname_blocked("something.local")
        assert is_hostname_blocked("service.internal")

    def test_public_hostnames_allowed(self):
        """Public hostnames should be allowed"""
        assert not is_hostname_blocked("example.com")
        assert not is_hostname_blocked("google.com")
        assert not is_hostname_blocked("api.github.com")


class TestURLSanitization:
    """Tests for URL sanitization"""

    def test_adds_https_scheme(self):
        """Should add https:// to URLs without scheme"""
        result = sanitize_url("example.com/path")
        assert result.startswith("https://")

    def test_strips_whitespace(self):
        """Should strip whitespace from URLs"""
        result = sanitize_url("  https://example.com  ")
        assert result == "https://example.com"

    def test_returns_none_for_invalid(self):
        """Should return None for invalid URLs"""
        result = sanitize_url("")
        assert result is None


class TestSearchQuerySanitization:
    """Tests for search query sanitization"""

    def test_removes_control_characters(self):
        """Should remove control characters"""
        result = sanitize_search_query("hello\x00world\x1f")
        assert "\x00" not in result
        assert "\x1f" not in result

    def test_limits_length(self):
        """Should limit query length"""
        long_query = "a" * 1000
        result = sanitize_search_query(long_query, max_length=500)
        assert len(result) == 500

    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace"""
        result = sanitize_search_query("  test query  ")
        assert result == "test query"

    def test_empty_returns_empty(self):
        """Empty input should return empty string"""
        result = sanitize_search_query("")
        assert result == ""

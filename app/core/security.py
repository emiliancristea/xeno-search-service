"""
Security utilities for Xeno Search Service
Includes SSRF protection, URL validation, and input sanitization
"""

import ipaddress
import re
import socket
from typing import Optional, Tuple
from urllib.parse import urlparse
import structlog

logger = structlog.get_logger(__name__)

# Private/reserved IP ranges that should not be accessed
BLOCKED_IP_RANGES = [
    ipaddress.ip_network('0.0.0.0/8'),       # "This" network
    ipaddress.ip_network('10.0.0.0/8'),      # Private-Use
    ipaddress.ip_network('100.64.0.0/10'),   # Shared Address Space
    ipaddress.ip_network('127.0.0.0/8'),     # Loopback
    ipaddress.ip_network('169.254.0.0/16'),  # Link Local
    ipaddress.ip_network('172.16.0.0/12'),   # Private-Use
    ipaddress.ip_network('192.0.0.0/24'),    # IETF Protocol Assignments
    ipaddress.ip_network('192.0.2.0/24'),    # Documentation (TEST-NET-1)
    ipaddress.ip_network('192.88.99.0/24'),  # 6to4 Relay Anycast
    ipaddress.ip_network('192.168.0.0/16'),  # Private-Use
    ipaddress.ip_network('198.18.0.0/15'),   # Benchmarking
    ipaddress.ip_network('198.51.100.0/24'), # Documentation (TEST-NET-2)
    ipaddress.ip_network('203.0.113.0/24'),  # Documentation (TEST-NET-3)
    ipaddress.ip_network('224.0.0.0/4'),     # Multicast
    ipaddress.ip_network('233.252.0.0/24'), # MCAST-TEST-NET
    ipaddress.ip_network('240.0.0.0/4'),     # Reserved
    ipaddress.ip_network('255.255.255.255/32'),  # Limited Broadcast
    # IPv6 private ranges
    ipaddress.ip_network('::1/128'),         # Loopback
    ipaddress.ip_network('fc00::/7'),        # Unique Local
    ipaddress.ip_network('fe80::/10'),       # Link Local
    ipaddress.ip_network('ff00::/8'),        # Multicast
]

# Blocked hostnames
BLOCKED_HOSTNAMES = [
    'localhost',
    'localhost.localdomain',
    'local',
    '*.local',
    '*.internal',
    '*.localhost',
    'metadata.google.internal',  # GCP metadata
    '169.254.169.254',           # Cloud metadata endpoint
]

# Allowed URL schemes
ALLOWED_SCHEMES = ['http', 'https']

# Maximum URL length
MAX_URL_LENGTH = 2048


def is_ip_blocked(ip_str: str) -> bool:
    """Check if an IP address is in a blocked range"""
    try:
        ip = ipaddress.ip_address(ip_str)
        for blocked_range in BLOCKED_IP_RANGES:
            if ip in blocked_range:
                return True
        return False
    except ValueError:
        # Invalid IP address
        return False


def is_hostname_blocked(hostname: str) -> bool:
    """Check if a hostname is blocked"""
    hostname_lower = hostname.lower()

    for blocked in BLOCKED_HOSTNAMES:
        if blocked.startswith('*.'):
            # Wildcard match
            suffix = blocked[1:]  # Remove the *
            if hostname_lower.endswith(suffix):
                return True
        elif hostname_lower == blocked:
            return True

    return False


def resolve_hostname(hostname: str, timeout: float = 2.0) -> Optional[str]:
    """Resolve hostname to IP address with timeout"""
    try:
        socket.setdefaulttimeout(timeout)
        ip = socket.gethostbyname(hostname)
        return ip
    except (socket.gaierror, socket.timeout, OSError) as e:
        logger.warning("Failed to resolve hostname", hostname=hostname, error=str(e))
        return None


def validate_url_for_ssrf(url: str) -> Tuple[bool, str]:
    """
    Validate a URL for SSRF vulnerabilities.

    Returns:
        Tuple of (is_safe, error_message)
    """
    if not url:
        return False, "URL is empty"

    if len(url) > MAX_URL_LENGTH:
        return False, f"URL exceeds maximum length of {MAX_URL_LENGTH}"

    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Invalid URL format: {str(e)}"

    # Check scheme
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return False, f"Invalid URL scheme: {parsed.scheme}. Only {ALLOWED_SCHEMES} are allowed"

    # Check for empty hostname
    if not parsed.netloc:
        return False, "URL has no hostname"

    # Extract hostname (remove port if present)
    hostname = parsed.hostname or parsed.netloc
    if not hostname:
        return False, "Could not extract hostname from URL"

    # Check for blocked hostnames
    if is_hostname_blocked(hostname):
        logger.warning("Blocked hostname access attempt", hostname=hostname, url=url)
        return False, f"Access to hostname '{hostname}' is not allowed"

    # Check if hostname is an IP address
    try:
        ip = ipaddress.ip_address(hostname)
        if is_ip_blocked(str(ip)):
            logger.warning("Blocked IP access attempt", ip=str(ip), url=url)
            return False, f"Access to IP address '{ip}' is not allowed"
    except ValueError:
        # Not an IP address, try to resolve it
        resolved_ip = resolve_hostname(hostname)
        if resolved_ip:
            if is_ip_blocked(resolved_ip):
                logger.warning("Blocked resolved IP access attempt",
                             hostname=hostname, resolved_ip=resolved_ip, url=url)
                return False, f"Hostname '{hostname}' resolves to blocked IP"

    return True, ""


def sanitize_url(url: str) -> Optional[str]:
    """
    Sanitize and normalize a URL.
    Returns None if URL is invalid.
    """
    if not url:
        return None

    url = url.strip()

    # Add scheme if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    try:
        parsed = urlparse(url)
        # Reconstruct URL to normalize it
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        return normalized
    except Exception:
        return None


def sanitize_search_query(query: str, max_length: int = 500) -> str:
    """
    Sanitize a search query to prevent injection attacks.
    """
    if not query:
        return ""

    # Remove null bytes and control characters
    query = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', query)

    # Limit length
    query = query[:max_length]

    # Strip whitespace
    query = query.strip()

    return query


class SSRFProtectedClient:
    """
    A wrapper that provides SSRF protection for HTTP requests.
    Use this instead of direct httpx/aiohttp calls.
    """

    def __init__(self, client, allow_redirects: bool = True, max_redirects: int = 5):
        self.client = client
        self.allow_redirects = allow_redirects
        self.max_redirects = max_redirects

    async def safe_get(self, url: str, **kwargs):
        """Perform a GET request with SSRF protection"""
        is_safe, error = validate_url_for_ssrf(url)
        if not is_safe:
            raise ValueError(f"SSRF protection blocked request: {error}")

        # Disable automatic redirects to validate each redirect
        kwargs['follow_redirects'] = False

        redirect_count = 0
        current_url = url

        while redirect_count < self.max_redirects:
            response = await self.client.get(current_url, **kwargs)

            # Check for redirect
            if response.is_redirect and self.allow_redirects:
                redirect_count += 1
                redirect_url = response.headers.get('location')

                if not redirect_url:
                    break

                # Handle relative redirects
                if not redirect_url.startswith(('http://', 'https://')):
                    from urllib.parse import urljoin
                    redirect_url = urljoin(current_url, redirect_url)

                # Validate redirect URL
                is_safe, error = validate_url_for_ssrf(redirect_url)
                if not is_safe:
                    raise ValueError(f"SSRF protection blocked redirect to: {redirect_url}")

                current_url = redirect_url
            else:
                return response

        raise ValueError(f"Too many redirects (max {self.max_redirects})")

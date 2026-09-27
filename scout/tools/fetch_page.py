"""Safe page fetcher: httpx + trafilatura, with an SSRF guard.

``fetch_page(url, focus)`` returns extracted page text cut to about ``fetch_max_chars``
characters, keeping the chunks most relevant to ``focus``.

Safety (docs/SPEC.md §4): only http/https; localhost and private, loopback, link-local and other
non-global addresses are blocked; every redirect hop is re-validated; the body is capped at
``fetch_max_bytes``; the whole fetch is bounded by ``fetch_timeout_s``. Wrapping the text in
``<untrusted_content>`` is the agent layer's job, not this module's.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
import time
from collections.abc import Callable
from urllib.parse import urljoin, urlsplit

import httpx
import trafilatura

from scout.config import Settings, load_settings

logger = logging.getLogger(__name__)

Resolver = Callable[[str, int], list[str]]

_ALLOWED_SCHEMES = {"http", "https"}
_BLOCKED_HOSTNAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}
_TEXT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
_CHUNK_CHARS = 600
_WORD_RE = re.compile(r"[a-z0-9]{3,}")
_STOPWORDS = frozenset(
    "the and for with from that this what who how are was were their its about into over "
    "have has had not but they them you your our any all can will".split()
)


class FetchError(Exception):
    """The page could not be fetched or yielded no text."""


class BlockedURLError(FetchError):
    """The URL is not allowed (bad scheme, localhost or non-public address)."""


def resolve_host(host: str, port: int) -> list[str]:
    """Resolve ``host`` to its IP address strings via the system resolver."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise FetchError(f"Cannot resolve host {host!r}") from exc
    return sorted({str(info[4][0]) for info in infos})


def is_public_ip(address: str) -> bool:
    """True only for globally routable unicast addresses (IPv4-mapped IPv6 unwrapped)."""
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def validate_url(url: str, resolver: Resolver = resolve_host) -> str:
    """Return ``url`` if it is safe to fetch, else raise :class:`BlockedURLError`.

    Checks the scheme, the hostname, and every address the hostname resolves to.
    """
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError as exc:
        raise BlockedURLError(f"Invalid URL: {url!r}") from exc
    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        raise BlockedURLError(f"Scheme not allowed: {parts.scheme or '<none>'!r}")
    host = (parts.hostname or "").rstrip(".").lower()
    if not host:
        raise BlockedURLError(f"URL has no host: {url!r}")
    if host in _BLOCKED_HOSTNAMES or host.endswith(".localhost"):
        raise BlockedURLError(f"Host not allowed: {host}")
    port = port or (443 if parts.scheme.lower() == "https" else 80)
    try:
        addresses = [str(ipaddress.ip_address(host))]
    except ValueError:
        addresses = resolver(host, port)
    if not addresses:
        raise FetchError(f"Host {host!r} did not resolve")
    blocked = [a for a in addresses if not is_public_ip(a)]
    if blocked:
        raise BlockedURLError(f"Host {host} resolves to non-public address {blocked[0]}")
    return url.strip()


def _read_capped(response: httpx.Response, max_bytes: int, deadline: float) -> bytes:
    """Read at most ``max_bytes`` from a streamed response, enforcing an absolute deadline."""
    buffer = bytearray()
    for chunk in response.iter_bytes():
        if time.monotonic() > deadline:
            raise FetchError("Fetch timed out")
        buffer.extend(chunk)
        if len(buffer) >= max_bytes:
            logger.info("Response truncated at %d bytes", max_bytes)
            return bytes(buffer[:max_bytes])
    return bytes(buffer)


def _download(
    url: str, settings: Settings, client: httpx.Client, resolver: Resolver
) -> tuple[str, str, bytes]:
    """GET ``url`` following redirects manually (each hop re-validated).

    Returns (final_url, content_type, body).
    """
    deadline = time.monotonic() + settings.fetch_timeout_s
    current = url
    for _ in range(settings.fetch_max_redirects + 1):
        validate_url(current, resolver)
        with client.stream("GET", current, follow_redirects=False) as response:
            if response.is_redirect:
                location = response.headers.get("location", "")
                if not location:
                    raise FetchError("Redirect without Location header")
                current = urljoin(current, location)
                continue
            if response.status_code >= 400:
                raise FetchError(f"HTTP {response.status_code} for {current}")
            content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type and not content_type.startswith(_TEXT_TYPES):
                raise FetchError(f"Unsupported content type {content_type!r}")
            body = _read_capped(response, settings.fetch_max_bytes, deadline)
            return current, content_type, body
    raise FetchError(f"Too many redirects (>{settings.fetch_max_redirects})")


def extract_text(body: bytes, url: str, content_type: str) -> str:
    """Extract main text with trafilatura; plain-text bodies pass through."""
    html = body.decode("utf-8", errors="replace")
    if content_type == "text/plain":
        return html.strip()
    text = trafilatura.extract(html, url=url, include_comments=False, include_tables=True)
    return (text or "").strip()


def _keywords(text: str) -> set[str]:
    """Lowercase content words used for focus scoring."""
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def _chunks(text: str, size: int = _CHUNK_CHARS) -> list[str]:
    """Group paragraphs into chunks of roughly ``size`` characters."""
    chunks: list[str] = []
    current = ""
    for para in (p.strip() for p in text.split("\n")):
        if not para:
            continue
        if current and len(current) + len(para) > size:
            chunks.append(current)
            current = ""
        current = f"{current}\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def focus_text(text: str, focus: str, max_chars: int) -> str:
    """Cut ``text`` to ``max_chars``, keeping the chunks most relevant to ``focus`` in order."""
    if len(text) <= max_chars:
        return text
    chunks = _chunks(text)
    terms = _keywords(focus)
    if not terms:
        return text[:max_chars]

    def score(idx: int) -> tuple[int, int]:
        words = _WORD_RE.findall(chunks[idx].lower())
        return (-sum(w in terms for w in words), idx)

    chosen: list[int] = []
    used = 0
    for idx in sorted(range(len(chunks)), key=score):
        cost = len(chunks[idx]) + 2
        if used + cost > max_chars:
            continue
        chosen.append(idx)
        used += cost
    if not chosen:
        return text[:max_chars]
    return "\n\n".join(chunks[i] for i in sorted(chosen))


def fetch_page(
    url: str,
    focus: str = "",
    *,
    settings: Settings | None = None,
    client: httpx.Client | None = None,
    resolver: Resolver = resolve_host,
) -> str:
    """Fetch ``url`` safely and return focus-relevant extracted text (~``fetch_max_chars``).

    Raises :class:`BlockedURLError` for unsafe URLs and :class:`FetchError` for network errors,
    timeouts, bad status codes, unsupported content or pages with no extractable text.
    """
    settings = settings or load_settings()
    own_client = client is None
    http = client or httpx.Client(
        timeout=httpx.Timeout(settings.fetch_timeout_s),
        headers={"User-Agent": settings.fetch_user_agent},
    )
    try:
        final_url, content_type, body = _download(url, settings, http, resolver)
    except httpx.TimeoutException as exc:
        raise FetchError(f"Fetch timed out after {settings.fetch_timeout_s}s") from exc
    except httpx.HTTPError as exc:
        raise FetchError(f"Fetch failed: {type(exc).__name__}") from exc
    finally:
        if own_client:
            http.close()
    text = extract_text(body, final_url, content_type)
    if not text:
        raise FetchError(f"No extractable text at {final_url}")
    return focus_text(text, focus, settings.fetch_max_chars)

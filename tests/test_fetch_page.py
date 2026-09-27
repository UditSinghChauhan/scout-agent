from __future__ import annotations

import itertools
from collections.abc import Callable

import httpx
import pytest

from scout.config import Settings
from scout.tools import fetch_page as fp
from scout.tools.fetch_page import (
    BlockedURLError,
    FetchError,
    fetch_page,
    focus_text,
    is_public_ip,
    validate_url,
)

PUBLIC_IP = "93.184.216.34"
SETTINGS = Settings(fetch_max_chars=4_000)

ARTICLE = """<html><head><title>Acme Corp</title></head><body>
<nav>Home | About | Careers</nav>
<article>
<h1>About Acme Corp</h1>
<p>Acme Corp is a software company headquartered in Chennai, India. It builds business
applications used by thousands of companies around the world, and it has grown steadily.</p>
<p>The company hires hundreds of engineering graduates each year through campus programs,
coding challenges and internships across several Indian universities.</p>
<p>Acme was founded in 1996 and remains privately held, with no outside venture funding.</p>
</article>
<footer>Copyright Acme</footer>
</body></html>"""


def public_resolver(host: str, port: int) -> list[str]:
    return [PUBLIC_IP]


def resolver_for(mapping: dict[str, list[str]]) -> Callable[[str, int], list[str]]:
    return lambda host, port: mapping.get(host, [PUBLIC_IP])


def client_for(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def html_response(body: str = ARTICLE, status: int = 200) -> httpx.Response:
    return httpx.Response(status, text=body, headers={"content-type": "text/html; charset=utf-8"})


# --- SSRF guard -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/",
        "http://LOCALHOST:8080/admin",
        "http://api.localhost/",
        "http://127.0.0.1/",
        "http://0.0.0.0/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
    ],
)
def test_blocks_localhost(url: str) -> None:
    with pytest.raises(BlockedURLError):
        validate_url(url, public_resolver)


@pytest.mark.parametrize(
    "url",
    [
        "http://10.0.0.5/",
        "http://172.16.3.4/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://100.64.0.1/",
        "http://[fd00::1]/",
        "http://[fe80::1]/",
        "http://224.0.0.1/",
    ],
)
def test_blocks_private_and_reserved_ips(url: str) -> None:
    with pytest.raises(BlockedURLError):
        validate_url(url, public_resolver)


def test_blocks_hostname_resolving_to_private_ip() -> None:
    resolver = resolver_for({"intranet.example.com": ["10.1.2.3"]})
    with pytest.raises(BlockedURLError, match="non-public"):
        validate_url("https://intranet.example.com/", resolver)


def test_blocks_when_any_resolved_address_is_private() -> None:
    resolver = resolver_for({"mixed.example.com": [PUBLIC_IP, "192.168.0.10"]})
    with pytest.raises(BlockedURLError):
        validate_url("https://mixed.example.com/", resolver)


@pytest.mark.parametrize(
    "url",
    ["ftp://example.com/file", "file:///etc/passwd", "javascript:alert(1)", "gopher://x/"],
)
def test_blocks_non_http_schemes(url: str) -> None:
    with pytest.raises(BlockedURLError, match="Scheme"):
        validate_url(url, public_resolver)


@pytest.mark.parametrize("url", ["not a url", "http://", "https:///path", "http://host:99999/"])
def test_invalid_urls_fail_cleanly(url: str) -> None:
    with pytest.raises(BlockedURLError):
        validate_url(url, public_resolver)


def test_public_url_allowed() -> None:
    assert validate_url("https://www.example.com/about", public_resolver)
    assert is_public_ip(PUBLIC_IP)
    assert not is_public_ip("not-an-ip")


def test_unresolvable_host_fails_cleanly() -> None:
    with pytest.raises(FetchError):
        validate_url("https://nowhere.example/", lambda host, port: [])


# --- fetching ---------------------------------------------------------------------------------


def test_fetch_extracts_article_text() -> None:
    text = fetch_page(
        "https://acme.example/about",
        "hiring",
        settings=SETTINGS,
        client=client_for(lambda r: html_response()),
        resolver=public_resolver,
    )
    assert "campus programs" in text
    assert "Copyright" not in text


def test_redirect_to_private_address_is_blocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/"})

    with pytest.raises(BlockedURLError):
        fetch_page("https://acme.example/", settings=SETTINGS, client=client_for(handler),
                   resolver=public_resolver)


def test_safe_redirect_is_followed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "/new"})
        return html_response()

    text = fetch_page("https://acme.example/old", settings=SETTINGS, client=client_for(handler),
                      resolver=public_resolver)
    assert "Chennai" in text


def test_redirect_loop_is_capped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "/again"})

    with pytest.raises(FetchError, match="Too many redirects"):
        fetch_page("https://acme.example/", settings=SETTINGS.with_overrides(fetch_max_redirects=2),
                   client=client_for(handler), resolver=public_resolver)


def test_http_error_status_fails_cleanly() -> None:
    with pytest.raises(FetchError, match="HTTP 404"):
        fetch_page("https://acme.example/", settings=SETTINGS,
                   client=client_for(lambda r: html_response("nope", 404)),
                   resolver=public_resolver)


def test_timeout_fails_cleanly() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(FetchError, match="timed out"):
        fetch_page("https://acme.example/", settings=SETTINGS, client=client_for(handler),
                   resolver=public_resolver)


def test_connection_error_fails_cleanly() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(FetchError, match="ConnectError"):
        fetch_page("https://acme.example/", settings=SETTINGS, client=client_for(handler),
                   resolver=public_resolver)


def test_overall_deadline_enforced_while_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = itertools.count(start=0, step=100)
    monkeypatch.setattr(fp.time, "monotonic", lambda: float(next(ticks)))
    with pytest.raises(FetchError, match="timed out"):
        fetch_page("https://acme.example/", settings=SETTINGS,
                   client=client_for(lambda r: html_response()), resolver=public_resolver)


def test_response_size_is_capped() -> None:
    big = "word " * 50_000  # 250 KB
    settings = SETTINGS.with_overrides(fetch_max_bytes=1_000, fetch_max_chars=100_000)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=big, headers={"content-type": "text/plain"})

    text = fetch_page("https://acme.example/big.txt", settings=settings,
                      client=client_for(handler), resolver=public_resolver)
    assert len(text) <= 1_000


def test_binary_content_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"%PDF-1.7", headers={"content-type": "application/pdf"})

    with pytest.raises(FetchError, match="content type"):
        fetch_page("https://acme.example/x.pdf", settings=SETTINGS, client=client_for(handler),
                   resolver=public_resolver)


def test_empty_page_fails_cleanly() -> None:
    with pytest.raises(FetchError, match="No extractable text"):
        fetch_page("https://acme.example/", settings=SETTINGS,
                   client=client_for(lambda r: html_response("<html></html>")),
                   resolver=public_resolver)


def test_blocked_url_never_reaches_transport() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return html_response()

    with pytest.raises(BlockedURLError):
        fetch_page("http://127.0.0.1:8000/", settings=SETTINGS, client=client_for(handler),
                   resolver=public_resolver)
    assert calls == []


# --- focus selection --------------------------------------------------------------------------


def test_focus_text_keeps_relevant_chunks() -> None:
    filler = "\n".join(f"Filler paragraph {i} about office furniture and lunch menus." * 3
                       for i in range(40))
    target = "Pricing: the enterprise plan costs 40 dollars per user per month."
    text = f"{filler}\n{target}\n{filler}"
    out = focus_text(text, "enterprise pricing per user", max_chars=800)
    assert target in out
    assert len(out) <= 800


def test_focus_text_short_text_unchanged() -> None:
    assert focus_text("short", "anything", 100) == "short"


def test_focus_text_without_focus_truncates() -> None:
    assert focus_text("a" * 500, "", 100) == "a" * 100

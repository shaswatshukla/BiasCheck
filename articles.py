"""Bounded public URL extraction, with per-redirect address checks."""
import ipaddress
import socket
import urllib.request
from urllib.parse import urlsplit, urljoin
import trafilatura

MAX_BYTES = 2_000_000

def validate_url(url):
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Use a public HTTP or HTTPS URL without credentials.")
        if parsed.port not in (None, 80, 443):
            raise ValueError("Only standard web ports are supported.")
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
            raise ValueError("Private and local network URLs are not supported.")
    except (OSError, ValueError) as exc:
        raise ValueError("URL must resolve to a public website on a standard web port.") from exc
    return url

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

def extract_article(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect)
    for _ in range(6):
        validate_url(url)
        request = urllib.request.Request(url, headers={"User-Agent": "BiasCheck/1.0 (article reader)"})
        try:
            response = opener.open(request, timeout=15)
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308) and exc.headers.get("Location"):
                url = urljoin(url, exc.headers["Location"])
                exc.close()
                continue
            raise ValueError("The website could not be read. Paste the article text instead.") from exc
        except OSError as exc:
            raise ValueError("The website could not be reached. Paste the article text instead.") from exc
        with response:
            if response.headers.get_content_type() not in ("text/html", "application/xhtml+xml"):
                raise ValueError("URL must be an HTML news page. Paste the article text instead.")
            raw = response.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise ValueError("Page is too large. Paste the article text instead.")
        document = trafilatura.bare_extraction(raw, url=url, with_metadata=True)
        if not document or not document.text or len(document.text.split()) < 60:
            raise ValueError("Not enough article text was found. Paste the full text instead.")
        return {"text": document.text, "author": document.author or "", "outlet": document.sitename or urlsplit(url).hostname, "url": url}
    raise ValueError("Too many redirects. Paste the article text instead.")

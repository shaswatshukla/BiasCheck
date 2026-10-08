from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from analyzer import Assessment, Evidence, analyze, validate_evidence, research_reporter
from articles import validate_url, extract_article, MAX_BYTES

TEXT = "The proposed tax would fund public hospitals. " + "Officials discussed the plan and residents requested further details. " * 10

def assessment(quote="The proposed tax would fund public hospitals.", leaning="Left leaning", direction="Left"):
    return Assessment(leaning=leaning, evidence_strength="Moderate", summary="Example", evidence=[Evidence(dimension="Policy stance", quote=quote, direction=direction, explanation="Example")], alternative_reading="Reporting is not endorsement.", limitations=["Single article"])

def test_quotes_must_be_in_article():
    with pytest.raises(ValueError, match="unverifiable"):
        validate_evidence(assessment("Invented claim"), TEXT)

def test_quote_whitespace_is_normalized():
    assert validate_evidence(assessment("The proposed tax\nwould fund public hospitals."), TEXT)

def test_classification_requires_directional_evidence():
    with pytest.raises(ValueError, match="supporting"):
        validate_evidence(assessment(direction="Unclear"), TEXT)

def test_api_uses_structured_output_and_no_storage():
    client = Mock()
    client.responses.parse.return_value.output_parsed = assessment()
    assert analyze(client, TEXT, "India", "test-model").leaning == "Left leaning"
    assert client.responses.parse.call_args.kwargs["store"] is False
    assert client.responses.parse.call_args.kwargs["text_format"] is Assessment

def test_refusal_is_handled():
    client = Mock()
    client.responses.parse.return_value.output_parsed = None
    with pytest.raises(ValueError, match="complete assessment"):
        analyze(client, TEXT, "India", "test-model")

@pytest.mark.parametrize("text,country", [("short", "India"), ("x " * 31000, "India"), (TEXT, "")], ids=["short", "long", "no-country"])
def test_bad_inputs_do_not_call_api(text, country):
    client = Mock()
    with pytest.raises(ValueError):
        analyze(client, text, country, "test-model")
    client.responses.parse.assert_not_called()

def test_background_preserves_citations():
    annotation = NS(type="url_citation", url="https://example.com/bio", title="Author biography")
    client = Mock()
    client.responses.create.return_value = NS(output_text="A sourced biography.", output=[NS(type="message", content=[NS(type="output_text", annotations=[annotation, annotation])])])
    result = research_reporter(client, "Reporter", "Outlet", "India", "test-model")
    assert len(result["sources"]) == 1

def test_uncited_background_is_rejected():
    client = Mock()
    client.responses.create.return_value = NS(output_text="Unsupported biography", output=[])
    with pytest.raises(ValueError, match="cited"):
        research_reporter(client, "Reporter", "Outlet", "India", "test-model")

@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://user:pass@example.com", "https://example.com:8080", "http://127.0.0.1", "http://[::1]", "http://169.254.169.254"])
def test_unsafe_urls_rejected(url):
    with pytest.raises(ValueError):
        validate_url(url)

def test_private_dns_rejected(monkeypatch):
    monkeypatch.setattr("articles.socket.getaddrinfo", lambda *a, **kw: [(2, 1, 6, "", ("10.0.0.1", 443))])
    with pytest.raises(ValueError):
        validate_url("https://example.com")

def test_extraction_uses_metadata(monkeypatch):
    monkeypatch.setattr("articles.validate_url", lambda url: url)
    response = Mock()
    response.headers.get_content_type.return_value = "text/html"
    response.read.return_value = b"<html>article</html>"
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    opener = Mock()
    opener.open.return_value = response
    monkeypatch.setattr("articles.urllib.request.build_opener", lambda *a: opener)
    monkeypatch.setattr("articles.trafilatura.bare_extraction", lambda *a, **kw: NS(text=TEXT, author="A Reporter", sitename="Example"))
    result = extract_article("https://example.com/news")
    assert result["author"] == "A Reporter"
    assert result["text"] == TEXT

def test_redirect_to_private_url_rejected(monkeypatch):
    import urllib.error
    from email.message import Message
    headers = Message()
    headers["Location"] = "http://127.0.0.1/private"
    opener = Mock()
    opener.open.side_effect = urllib.error.HTTPError("https://example.com", 302, "redirect", headers, None)
    monkeypatch.setattr("articles.urllib.request.build_opener", lambda *a: opener)
    monkeypatch.setattr("articles.socket.getaddrinfo", lambda host, *a, **kw: [(2, 1, 6, "", ("127.0.0.1" if host == "127.0.0.1" else "93.184.216.34", 443))])
    with pytest.raises(ValueError):
        extract_article("https://example.com")
    assert opener.open.call_count == 1

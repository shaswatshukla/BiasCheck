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

def test_api_uses_structured_output():
    client = Mock()
    client.models.generate_content.return_value.text = assessment().model_dump_json()
    assert analyze(client, TEXT, "India", "test-model").leaning == "Left leaning"
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == Assessment.model_json_schema()

def test_refusal_is_handled():
    client = Mock()
    client.models.generate_content.return_value.text = None
    with pytest.raises(ValueError, match="complete assessment"):
        analyze(client, TEXT, "India", "test-model")

@pytest.mark.parametrize("text,country", [("short", "India"), ("x " * 31000, "India"), (TEXT, "")], ids=["short", "long", "no-country"])
def test_bad_inputs_do_not_call_api(text, country):
    client = Mock()
    with pytest.raises(ValueError):
        analyze(client, text, country, "test-model")
    client.models.generate_content.assert_not_called()

def test_background_preserves_citations():
    metadata = NS(grounding_chunks=[NS(web=NS(uri="https://example.com/bio", title="Author biography"))], grounding_supports=[NS(segment=NS(end_index=18), grounding_chunk_indices=[0])], search_entry_point=None)
    client = Mock()
    client.models.generate_content.return_value = NS(text="A sourced biography.", candidates=[NS(grounding_metadata=metadata)])
    result = research_reporter(client, "Reporter", "Outlet", "India", "test-model")
    assert len(result["sources"]) == 1
    assert "[1](https://example.com/bio)" in result["text"]

def test_uncited_background_is_rejected():
    client = Mock()
    client.models.generate_content.return_value = NS(text="Unsupported biography", candidates=[])
    with pytest.raises(ValueError, match="cited"):
        research_reporter(client, "Reporter", "Outlet", "India", "test-model")

def test_malformed_json_is_rejected():
    client = Mock()
    client.models.generate_content.return_value.text = "not JSON"
    with pytest.raises(ValueError):
        analyze(client, TEXT, "India", "test-model")

def test_grounding_citations_use_utf8_offsets():
    text = "भारत में reporter."
    metadata = NS(grounding_chunks=[NS(web=NS(uri="https://example.com/bio", title="Bio"))], grounding_supports=[NS(segment=NS(end_index=len(text.encode('utf-8'))), grounding_chunk_indices=[0])], search_entry_point=NS(rendered_content="<div>Search suggestion</div>"))
    client = Mock()
    client.models.generate_content.return_value = NS(text=text, candidates=[NS(grounding_metadata=metadata)])
    result = research_reporter(client, "Reporter", "Outlet", "India", "test-model")
    assert result["text"] == text + " [1](https://example.com/bio)"
    assert result["search_suggestions"] == "<div>Search suggestion</div>"

def test_missing_claim_citations_is_rejected():
    metadata = NS(grounding_chunks=[NS(web=NS(uri="https://example.com/bio", title="Bio"))], grounding_supports=[], search_entry_point=None)
    client = Mock()
    client.models.generate_content.return_value = NS(text="Biography", candidates=[NS(grounding_metadata=metadata)])
    with pytest.raises(ValueError, match="claim-level"):
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

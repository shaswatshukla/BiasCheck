from streamlit.testing.v1 import AppTest
from pathlib import Path
from analyzer import Assessment, Evidence

TEXT = "The proposed tax would fund public hospitals. " + "Officials discussed the plan and residents requested further details. " * 10
APP = str(Path(__file__).resolve().parents[1] / "app.py")

def test_app_starts_without_key():
    app = AppTest.from_file(APP, default_timeout=15).run()
    assert not app.exception
    assert app.title[0].value == "📰 BiasCheck"
    app.button[0].click().run()
    assert "API key" in app.error[0].value

def test_result_and_export(monkeypatch):
    result = Assessment(leaning="Uncertain", evidence_strength="Low", summary="Insufficient author stance.", evidence=[], alternative_reading="Straight reporting.", limitations=["One article"])
    monkeypatch.setattr("analyzer.analyze", lambda *a: result)
    app = AppTest.from_file(APP, default_timeout=15).run()
    app.sidebar.text_input[0].set_value("test-key")
    app.text_area[0].set_value(TEXT)
    app.button[0].click().run()
    assert not app.exception
    assert app.subheader[0].value == "Uncertain"
    assert app.session_state["report"]["article"]["summary"] == result.summary
    assert "test-key" not in str(app.session_state["report"])

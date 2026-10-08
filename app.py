import json
import hashlib
import os
import streamlit as st
from openai import OpenAI, OpenAIError
from analyzer import analyze, research_reporter
from articles import extract_article

st.set_page_config(page_title="BiasCheck", page_icon="📰", layout="wide")
st.title("📰 BiasCheck")
st.write("Read beyond the headline. Explore political framing and the reporter's public professional background.")
st.caption("An AI assessment, not a fact-check or a judgment of a reporter's personal beliefs.")

def setting(name, default=""):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except st.errors.StreamlitSecretNotFoundError:
        return os.getenv(name, default)

with st.sidebar:
    st.header("Analysis settings")
    key = st.text_input("OpenAI API key", type="password") or setting("OPENAI_API_KEY")
    model = st.text_input("Model", value=setting("OPENAI_MODEL", "gpt-4.1-mini"))
    country = st.selectbox("Political context", ["India", "United States", "United Kingdom", "Other"])
    if country == "Other":
        country = st.text_input("Country / political context")
    st.info("Left and right vary by country. Centrist requires evidence; missing evidence is reported as uncertain.")
    st.caption("Analysis sends article text to OpenAI. Optional reporter research uses web search. API charges apply. Keys are never included in exports.")

mode = st.radio("Article input", ["Paste text", "Article URL"], horizontal=True)
if mode == "Article URL":
    url = st.text_input("Public article URL")
    if st.button("Read article"):
        try:
            with st.spinner("Reading article…"):
                article = extract_article(url)
            st.session_state["article_text"] = article["text"]
            st.session_state["reporter"] = article["author"]
            st.session_state["outlet"] = article["outlet"]
            st.session_state["source_url"] = article["url"]
            st.session_state.pop("report", None)
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    st.caption("Review the extracted text and byline before analyzing. For blocked pages, paste the text.")
text = st.text_area("Article text", key="article_text", height=300)
left, right = st.columns(2)
reporter = left.text_input("Reporter name (optional)", key="reporter")
outlet = right.text_input("Outlet (optional)", key="outlet")
research = st.checkbox("Research the reporter's public professional background", value=False)
signature = hashlib.sha256(json.dumps([text, reporter, outlet, research, country, model, mode]).encode()).hexdigest()
if "report" in st.session_state and st.session_state.get("report_signature") != signature:
    st.session_state.pop("report", None)
if st.button("Analyze article", type="primary"):
    st.session_state.pop("report", None)
    try:
        if not key:
            raise ValueError("Add an OpenAI API key in the sidebar to analyze.")
        if research and not (reporter.strip() and outlet.strip()):
            raise ValueError("Reporter research needs a name and outlet to distinguish namesakes.")
        client = OpenAI(api_key=key, timeout=90, max_retries=1)
        with st.spinner("Assessing framing and checking evidence…"):
            report = analyze(client, text, country, model)
            background = None
            research_error = None
            if research:
                try:
                    background = research_reporter(client, reporter, outlet, country, model)
                except (ValueError, OpenAIError):
                    research_error = "Reporter research was unavailable. The article assessment is still shown; try again later."
        st.session_state["report"] = {"article": report.model_dump(), "background": background, "research_error": research_error, "context": country, "model": model, "source_url": st.session_state.get("source_url", "") if mode == "Article URL" else ""}
        st.session_state["report_signature"] = signature
    except ValueError as exc:
        st.error(str(exc))
    except OpenAIError:
        st.error("Analysis could not complete. Check your API key, model access, account credit and connection, then retry.")

if "report" in st.session_state:
    saved = st.session_state["report"]
    result = saved["article"]
    st.subheader(result["leaning"])
    st.caption(f"Evidence strength: {result['evidence_strength']} · Context: {saved['context']}")
    st.write(result["summary"])
    tab1, tab2, tab3 = st.tabs(["Article evidence", "Reporter background", "Limits & export"])
    with tab1:
        for evidence in result["evidence"]:
            st.markdown(f"**{evidence['dimension']}** · {evidence['direction']}")
            st.text(evidence["quote"])
            st.write(evidence["explanation"])
        st.markdown("**Alternative reading**")
        st.write(result["alternative_reading"])
    with tab2:
        if saved["research_error"]:
            st.warning(saved["research_error"])
        if saved["background"]:
            st.markdown(saved["background"]["text"])
            st.markdown("**Sources**")
            for source in saved["background"]["sources"]:
                st.link_button(source["title"], source["url"])
        elif not saved["research_error"]:
            st.info("Enable reporter research to check public biographies, employment, published work and explicitly disclosed affiliations.")
        st.caption("Background does not change the article classification. Identity and source claims need human review.")
    with tab3:
        for limitation in result["limitations"]:
            st.write("• " + limitation)
        st.write("Political framing and factual accuracy are different. Model judgments are not calibrated probabilities. Personal ideology is not inferred from identity or employer.")
        st.download_button("Download report", json.dumps(saved, indent=2, ensure_ascii=False), "biascheck-report.json", "application/json")

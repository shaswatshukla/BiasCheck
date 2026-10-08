# BiasCheck

A Streamlit app that assesses a news article's political framing in a chosen country context and optionally researches the reporter's public professional background.

## Features

- Paste article text or extract a public news URL, then review the text and byline.
- Left leaning, right leaning, centrist, mixed or uncertain assessment.
- Exact article quotes, framing explanations, evidence strength and an alternative reading.
- Optional web research of reporter employment, beats, published work and explicitly disclosed affiliations, with inline citations and source links.
- Download a JSON report. Keys are never exported.
- India, US, UK and custom country context.

Reporter background is displayed separately and does **not** influence the content classification. The app does not infer a person's political beliefs from their identity or employer. Source claims and reporter identity need human review. Missing evidence does not mean centrist. There are no fabricated percentage scores or pretrained dummy models.

## Run locally

Requires Python 3.11 or newer.

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Enter your OpenAI API key in the sidebar, or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in your key. `OPENAI_API_KEY` and `OPENAI_MODEL` environment variables are also supported. Never commit keys. The default model is `gpt-4.1-mini`; select an accessible model supporting Responses structured outputs and web search for optional research.

Article text is sent to OpenAI when you click Analyze. Reporter name, outlet and country are sent when research is enabled. API usage and web search can incur charges. Responses are requested with `store=False`; this is not a claim of zero provider retention. Reports remain in the browser session until it ends and can be downloaded. The app does not write reports to a database.

## Deploy on Streamlit Community Cloud

Connect this repository, select the reviewed branch and `app.py` as the entry point. Configure `OPENAI_API_KEY` in the deployment's Secrets settings or let each visitor enter their own key. A shared server key pays for all visitor usage; use private access or add authentication and usage controls before operating a public shared-key service.

URL extraction rejects local/private IP addresses, unusual ports and credential-bearing URLs, validates each redirect, and limits download size. DNS validation is a best-effort defense, not protection against DNS rebinding: use an egress proxy/firewall that blocks private networks for an untrusted public deployment. Some websites block extraction; paste text instead. Extraction and bylines may be incomplete.

## Tests

```sh
pip install pytest
python -m pytest -q
```

Tests cover evidence validation, malformed inputs, model refusals, citations, URL checks, redirect checks, extraction, app startup and report rendering. API responses are mocked; they verify plumbing, not real-world classifier accuracy. GitHub Actions runs the suite on pushes and pull requests.

## Method and limitations

This is an LLM-assisted reading tool, not a trained or benchmarked political classifier. It examines author voice, policy stance, framing, loaded language and source selection; quoted speakers are not automatically the author's stance. It distinguishes country contexts and economic/social dimensions. Quote verification confirms a quotation exists in the article; it cannot establish the correctness of the model's interpretation. It does not establish factual accuracy, motives, omitted facts or a reporter's personal beliefs.

For production classification, assemble licensed, independently annotated, country-specific articles; split evaluation by outlet, author and time; compare against a baseline and measure per-label precision/recall, inter-annotator agreement and abstention. Publish those results before making accuracy claims.

API integration follows the [Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs) and [Web Search guide](https://developers.openai.com/api/docs/guides/tools-web-search).

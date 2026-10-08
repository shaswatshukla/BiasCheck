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

Enter your Gemini API key in the sidebar, or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in your key. `GEMINI_API_KEY` and `GEMINI_MODEL` environment variables are also supported. Never commit keys. The default model is `gemini-2.5-flash`; select an accessible model supporting structured outputs and Google Search grounding for optional research.

Article text is sent to Google when you click Analyze. Reporter name, outlet and country are sent when research is enabled. Gemini 2.5 Flash currently offers limited free-tier text generation and Google Search grounding. Keep your project on the free tier for free usage; account quotas apply and may change. Paid-tier projects can incur charges. Google may use free-tier inputs to improve products. Reports remain in the browser session until it ends and can be downloaded. The app does not write reports to a database.

## Deploy on Streamlit Community Cloud

Connect this repository, select the reviewed branch and `app.py` as the entry point. Configure `GEMINI_API_KEY` in the deployment's Secrets settings or let each visitor enter their own key. A shared server key shares your project quota; use private access or add authentication and usage controls before operating a public shared-key service.

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

Get a free-tier key from [Google AI Studio](https://aistudio.google.com/apikey). API integration follows Google's [structured outputs](https://ai.google.dev/gemini-api/docs/generate-content/structured-output), [search grounding](https://ai.google.dev/gemini-api/docs/generate-content/google-search) and [pricing](https://ai.google.dev/gemini-api/docs/pricing) documentation.

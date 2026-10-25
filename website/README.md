# BiasCheck · Netlify edition

An interactive news-framing reader with Gemini free-tier analysis, exact evidence quotes, an original-article reader, sourced reporter context, device-local saved reports, JSON export and print/PDF support.

## Run locally

Requires Node.js 22 or newer.

```sh
npm ci
npm run dev
```

Open the printed local URL. The development server runs a Vite interface and the same API handler used by Netlify. `npm test` verifies input and quotation validation, extraction, private-network blocking and API failure paths. `npm run build` checks types and produces the static frontend.

## Deploy on Netlify

The tracked `netlify.toml` sets the build command (`npm run build`), frontend directory (`dist`) and functions directory (`netlify/functions`). Netlify Functions provide server-side Gemini requests and bounded article reading. No Oracle server or database is required.

```sh
npx netlify login
npx netlify deploy --build --prod
```

Select your free-tier team and create or link the intended project. Do not enable paid add-ons or auto recharge for a free deployment. Netlify Free has a monthly credit limit; check your dashboard because account plans and limits vary. Repository deployment can instead use these same build settings.

## Gemini key and privacy

Visitors enter their own Google AI Studio key in the Connect Gemini dialog. It stays only in page memory, is sent over HTTPS through the site to Google, and disappears on refresh. There is no embedded shared key and no key storage in reports or local storage. Never put credentials in `VITE_` variables, source, Git, or chat. A previously shared key should be revoked and replaced.

The current model is `gemini-3.5-flash-lite`. Google lists a limited free tier; keep your Google project on its free tier for free usage. Paid-tier projects can be charged by Google. Reporter background uses user-supplied public biography or author-page URLs; no paid Google Search grounding is called. Free-tier inputs may be used by Google for product improvement.

Report saving is explicit, browser-local and limited to the latest 20 saved reports. It stores article text and report content on that device. Nothing is synced between users. Removing a report offers undo; JSON export provides a separate copy. Browser storage can be cleared or fail, so export important reports.

## Method and limits

Labels are Left leaning, Right leaning, Centrist, Mixed and Uncertain. The assessment uses country context and distinguishes the author’s voice from quotations. A directional label must contain a supporting evidence item, and every displayed quote is checked against submitted text. This validates quotation existence, not interpretation. Reporter identity is checked against the supplied name/outlet and source text. Professional context does not change the article’s label or establish personal political beliefs.

This is an AI-assisted reading tool, not a fact-checker. Classification accuracy has not been benchmarked. Sample reports are explicitly illustrative and never presented as a live model result. Evidence strength is qualitative, not a calibrated probability.

Public URL reading rejects IP literals, credentials, custom ports and private/reserved addresses. DNS is checked inside the network connection lookup, pinning the selected public address; redirects are validated again. Responses are size-limited and timed out. Some pages block extraction or require browser rendering; paste their full text instead. Content parsing does not execute scripts.

Security headers restrict browser connections to this site's own API. Fonts are self-hosted. API responses use no-store, and errors avoid echoing credentials or raw provider responses. No logs of request content or keys are deliberately written by this application. Hosting/provider logs and retention follow their own policies.

## References

- [Google Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Netlify pricing](https://www.netlify.com/pricing/)
- [Netlify Functions](https://docs.netlify.com/build/functions/overview/)

No deployment is complete until the Netlify production URL and API response have been verified.

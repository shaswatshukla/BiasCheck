# BiasCheck

The complete interactive Netlify website is in [`website/`](website/README.md). It includes Gemini free-tier article framing analysis, exact-quote evidence, an article reader, sourced reporter context, device-local saved reports, JSON export, and print/PDF.

## Netlify setup

Import this repository and use branch `netlify-website-v2` (or main after merging). The root `netlify.toml` sets base directory `website`, build command `npm run build`, publish directory `dist`, and the Netlify Functions directory. No Oracle server or database is needed for this version.

See [deployment instructions](website/DEPLOY.md) and [privacy and methodology](website/README.md).

The root Python files are the original prototype; the new website does not load its dummy pickle models. Visitors supply a session-only Gemini free-tier key. Never commit keys.

Netlify hosting is pending account sign-in and production verification; this source does not imply a new Netlify URL is already live.

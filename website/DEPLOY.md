# Publish BiasCheck on Netlify Free

The website and API are ready for Netlify; no Oracle server is required.

1. Sign in to Netlify. Stay on the Free plan; do not enable paid add-ons or automatic credit purchases.
2. Import the BiasCheck GitHub repository and select branch `netlify-website-v2` (or `main` after merging its pull request).
3. The repository's root `netlify.toml` sets base directory `website`, command `npm run build`, publish directory `dist`, and functions directory `netlify/functions`.
4. Deploy and open the assigned Netlify URL. For a fresh analysis, enter a replacement Gemini free-tier key through **Connect Gemini**. Do not add it to source code, a `VITE_` variable, or this document.
5. Verify that **Try a sample** displays a sample report, then run one real article after connecting the key. The article reader can be tested with a public page that allows extraction.

For the standalone source ZIP, extract it and run `npm ci`, `netlify login`, then `netlify deploy --build --prod` from the extracted directory. The standalone configuration uses the project root as its base.

Netlify Free and Gemini free-tier quotas both apply. Google Search grounding is not used; reporter context uses the source URLs you provide. Browser report saving is device-local, not account-based.

This package does not claim a Netlify deployment has happened. Publication is complete only after a Netlify production URL and its API have been checked.

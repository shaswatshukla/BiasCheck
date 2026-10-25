# Free ABP research model

`newsbias.py` preserves the user's supplied engine. `load_abp.py` adapts the supplied loader and streams the Git snapshot (or reads ordinary JSON files), keeps 60+ word articles, caps each outlet at 40 articles / each article at 400 words, and removes exact duplicate prefixes globally. Dataset text stays under ignored `.local/` and is never published.

## Reproduce

Install numpy, pandas, scipy and scikit-learn in an isolated environment, then:

```
git clone --depth 1 https://github.com/ramybaly/Article-Bias-Prediction.git .local/abp-research
python training/train_abp.py .local/abp-research
```

`train_abp.py` writes `public/models/abp-us-v1.json` (vocabulary, IDF, linear weights, scaler and temperature), `public/models/abp-model-card.json` (audit, exact source splits, metrics and limitations), and synthetic Python/TypeScript parity fixtures. It does not export article text.

The new training path corrects the supplied code's evaluation/calibration design. It uses three outer source-grouped folds. Each fold holds out complete calibration outlets before training the text and stacked models. Meta features use source-cross-fitted text probabilities, centered logs, lexicon balance and entity tone. Source priors and peer features remain zero; LLM calls are disabled. A temperature is fitted on entirely separate calibration outlets. The final exported model also retains separate calibration outlets rather than refitting on them. Accuracy/F1/ECE/coverage use equal outlet weighting; random-vs-grouped text-only F1 is a leakage diagnostic, not a deployment claim.

This is a compact TF-IDF/logistic baseline, not the paper's neural model. Labels can largely act as outlet ratings. Source-held-out performance does not establish objective article ideology, time/topic transfer or India/UK validity. Exact duplicate prefixes are removed, but near duplicates/same-event coverage can remain. No validated reliability score is claimed.

The model is an optional experimental method, enabled only in United States context; the free heuristic mode remains the default. The held-out accuracy (~43%) and answer coverage (~7%) do not support using this as a dependable article classifier. Other countries retain the existing heuristic mode. The trained path retains the original Python side-feature substring semantics for inference parity; the separate browser heuristic panel uses whole-word matching and is not identical to the learned feature vector. Sensationalism and quoted voice counts remain descriptive and do not affect the trained stack. Gemini is an independent optional contextual path.

The browser ports TF-IDF, multinomial logits, scaler, stacked logits, temperature and abstention to TypeScript. Synthetic fixtures compare Python probabilities with browser inference. The website does not need Python or a key.

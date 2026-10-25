# Supplied newsbias.py

The user's Python source is preserved in `newsbias.py`. The live free mode ports cleaning, lexicon density, entity-tone, voice counts and sensationalism to TypeScript in `src/multi-signal.ts`. It runs locally without keys.

No dataset or fitted weights were supplied. The deployed mode is NOT the trained BiasEngine. It averages available lexical balance `(right-left)/(right+left)` and `tanh(entity_tone*2)` equally; display shares are `(1-score)/2` and its complement. These fixed heuristic shares are not calibrated probabilities. Source priors, TF-IDF/LR, learned stacking and peer framing are inactive. Sensationalism retains the original sigmoid formula and is explicitly not a reliability score.

The India seeds describe opposition/government associations, not ideological left/right, and are provisional. US seeds are also hand-authored. Word matches use boundaries instead of substring counts. Quoted-voice counts are descriptive only. Prior policy-rule evidence remains visible separately.

Before enabling the trained model, supply licensed English data with `text`, `source`, `label` (`left`, `center`, `right`), sufficient sources and examples of all classes in every nested training/calibration fold. Do not use toy labels as accuracy evidence. The supplied implementation needs auditing: the inner text calibration uses ordinary folds rather than source groups; the meta-classifier is not explicitly calibrated; peer features differ between training/inference; source-prior provenance and event/date grouping require validation; and `evaluate` needs guards for no answered predictions. Run source-held-out evaluation and calibration checks before exposing model probabilities.

Gemini remains an independent, optional contextual analysis path, not part of the free heuristic fusion.

"""
newsbias.py: multi-signal news political-lean engine.

Signals (fused by a stacked meta-learner, then calibrated + abstention):
  1. Text classifier        TF-IDF + LR (out-of-source predictions only)
  2. Source prior           external outlet ratings (dropout-trained, so unseen outlets work)
  3. Loaded-language lexicon partisan wording per 1000 words
  4. Entity tone            sentiment toward left-/right-coded actors
  5. Relative framing       lean vs. other outlets covering the SAME event
  6. Optional LLM judge     any callable text -> [p_left, p_center, p_right]
Separate axis: sensationalism (0..1), so "center" never absorbs "loud/sloppy".

Input df columns: text, source, label in {left, center, right}.
Usage:
    eng = BiasEngine(INDIA).fit(df)
    eng.analyze(article_text, source="outlet", peers=[same_event_texts_from_other_outlets])
    evaluate(df, INDIA)           # leave-sources-out evaluation + leakage gap
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

LABELS = ["left", "center", "right"]


# --------------------------------------------------------------------------
# Context: the axis is culture-specific. Define it explicitly, don't assume it.
# --------------------------------------------------------------------------
@dataclass
class Context:
    name: str
    left_terms: list = field(default_factory=list)     # left-coded loaded words
    right_terms: list = field(default_factory=list)    # right-coded loaded words
    left_actors: list = field(default_factory=list)    # entities of the "left" pole
    right_actors: list = field(default_factory=list)   # entities of the "right" pole
    source_prior: dict = field(default_factory=dict)   # {"outlet": [pL, pC, pR]} from external raters


US = Context(
    "US",
    left_terms=["far-right", "extremist", "gutted", "dark money", "voter suppression", "climate denier"],
    right_terms=["radical left", "woke", "mainstream media", "illegal aliens", "open borders", "elites"],
    left_actors=["democrat", "biden", "harris", "progressive"],
    right_actors=["republican", "trump", "gop", "conservative"],
)

# India: poles are NOT US left/right. Here left = opposition-aligned / right = government-aligned
# framing. These seeds are placeholders: curate them with domain experts before trusting output.
INDIA = Context(
    "India (opposition-aligned <-> government-aligned)",
    left_terms=["authoritarian", "majoritarian", "crony capitalism", "undeclared emergency", "witch-hunt"],
    right_terms=["appeasement", "tukde", "anti-national", "urban naxal", "vote bank politics"],
    left_actors=["congress", "rahul gandhi", "aap", "kejriwal", "opposition"],
    right_actors=["bjp", "modi", "amit shah", "nda", "government"],
)


# --------------------------------------------------------------------------
# Preprocessing: strip everything that identifies the PUBLISHER (leakage)
# --------------------------------------------------------------------------
def clean(t: str, source_names=()) -> str:
    for s in source_names:
        if s:
            t = re.sub(re.escape(s), " ", t, flags=re.I)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"^\s*[A-Z][A-Z ,.'-]{2,40}\s*(\([^)]*\))?\s*[-\u2014\u2013:]\s*", " ", t)  # datelines
    t = re.sub(r"\b(photo|file photo|getty images?|ap photo|reuters)\b[^.\n]*", " ", t, flags=re.I)
    t = re.sub(r"(follow us on|subscribe|sign up for|read more)[^.\n]*", " ", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


def _sents(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", t) if s]


# --------------------------------------------------------------------------
# Hand-built signals
# --------------------------------------------------------------------------
POS = set("good great success win praised welcomed strong boost growth relief hailed secure fair".split())
NEG = set("bad crisis failure fail slammed attack corrupt scandal weak crackdown fraud harm threat unfair".split())


def _tone(s: str) -> float:
    w = re.findall(r"[a-z']+", s.lower())
    p, n = sum(x in POS for x in w), sum(x in NEG for x in w)
    return (p - n) / max(p + n, 1)


def side_features(text: str, ctx: Context) -> np.ndarray:
    """[left-coded words/1000, right-coded words/1000, entity tone (+ = favours right actors)]"""
    tl, n = text.lower(), max(len(text.split()), 1)
    lex_l = sum(tl.count(w) for w in ctx.left_terms) / n * 1000
    lex_r = sum(tl.count(w) for w in ctx.right_terms) / n * 1000
    sl, sr = [], []
    for s in _sents(text):
        x = s.lower()
        a, b = any(k in x for k in ctx.left_actors), any(k in x for k in ctx.right_actors)
        if a and not b:
            sl.append(_tone(s))
        elif b and not a:
            sr.append(_tone(s))
    m = lambda v: float(np.mean(v)) if v else 0.0
    return np.array([lex_l, lex_r, m(sr) - m(sl)])


def voice_balance(text: str, ctx: Context) -> dict:
    """Evidence only (not a lean feature): who gets quoted."""
    q = {"left_actors": 0, "right_actors": 0}
    for s in _sents(text):
        x = s.lower()
        if '"' in s or "\u201c" in s or " said" in x or " says" in x:
            q["left_actors"] += any(k in x for k in ctx.left_actors)
            q["right_actors"] += any(k in x for k in ctx.right_actors)
    return q


def sensationalism(text: str) -> float:
    """Heuristic 0..1. Calibrate against a reliability-labelled set (e.g. Ad Fontes) before trusting."""
    w, n = text.split(), max(len(text.split()), 1)
    caps = sum(1 for x in w if len(x) > 3 and x.isupper()) / n
    excl = text.count("!") / max(len(_sents(text)), 1)
    sup = len(re.findall(r"\b(shocking|devastating|explosive|bombshell|slams|destroys|stunning|outrage|"
                         r"meltdown|unbelievable|massive|historic)\b", text, re.I)) / n * 100
    click = 1 if re.search(r"(you won't believe|what happens next|here's why|this is why)", text, re.I) else 0
    attrib = len(re.findall(r"\b(said|according to|told|reported|data|study|figures)\b", text, re.I)) / n * 100
    raw = 4 * caps + 0.5 * excl + 0.3 * sup + 0.5 * click - 0.1 * attrib
    return float(1 / (1 + np.exp(-(raw - 0.3) * 3)))


# --------------------------------------------------------------------------
# Model pieces
# --------------------------------------------------------------------------
def _text_model():
    base = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True, stop_words="english"),
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
    )
    return CalibratedClassifierCV(base, cv=3, method="sigmoid")


def _proba(model, X) -> np.ndarray:
    """predict_proba reordered to LABELS (missing classes -> 0)."""
    P = model.predict_proba(X)
    out = np.zeros((len(X), 3))
    for j, c in enumerate(model.classes_):
        out[:, LABELS.index(c)] = P[:, j]
    return out


def _centered_log(p):
    l = np.log(np.asarray(p) + 1e-4)
    return l - l.mean(axis=-1, keepdims=True)


def relative_framing(texts, sources, scores, thr=0.75) -> np.ndarray:
    """Lean of each article minus the mean lean of other-outlet articles on the same event.
    Clusters by TF-IDF cosine distance. For >5k articles swap in embeddings + HDBSCAN/FAISS
    and add a publication-date window."""
    n = len(texts)
    rel = np.zeros(n)
    if n < 4 or n > 5000:
        return rel
    X = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True).fit_transform(texts)
    D = np.clip(1 - cosine_similarity(X), 0, None)
    lab = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                  distance_threshold=thr).fit_predict(D)
    s, sc = np.asarray(sources), np.asarray(scores)
    for c in np.unique(lab):
        idx = np.where(lab == c)[0]
        if len(idx) >= 2 and len(set(s[idx])) >= 2:
            rel[idx] = sc[idx] - sc[idx].mean()
    return rel


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------
class BiasEngine:
    def __init__(self, ctx: Context, llm_fn=None, use_source_prior=True, src_dropout=0.3,
                 abstain_below=0.50, max_disagreement=0.9, seed=0):
        self.ctx, self.llm_fn, self.use_src = ctx, llm_fn, use_source_prior
        self.src_dropout, self.abstain_below, self.max_dis = src_dropout, abstain_below, max_disagreement
        self.rng = np.random.default_rng(seed)

    # ---- feature plumbing ----
    def _src_vec(self, src, train=False):
        p = self.ctx.source_prior.get((src or "").lower())
        if p is None or not self.use_src or (train and self.rng.random() < self.src_dropout):
            return np.zeros(3)  # unknown / masked -> no information
        return _centered_log(p)

    def _meta_row(self, p_text, src, side, rel, llm_p, train=False):
        f = list(_centered_log(p_text)) + list(self._src_vec(src, train))
        f += [side[1] - side[0], side[2], rel]
        if self.llm_fn is not None:
            f += list(_centered_log(llm_p))
        return f

    # ---- training ----
    def fit(self, df: pd.DataFrame):
        d = df.reset_index(drop=True)
        texts = [clean(t, [s]) for t, s in zip(d.text, d.source)]
        y = d.label.values
        groups = d.source.values
        k = min(5, len(set(groups)))
        assert k >= 3, "Need >= 3 distinct sources for source-grouped training"

        # out-of-SOURCE text probabilities: the meta-learner never sees in-source leakage
        P = np.zeros((len(d), 3))
        for tr, te in GroupKFold(n_splits=k).split(texts, y, groups):
            m = _text_model().fit([texts[i] for i in tr], y[tr])
            P[te] = _proba(m, [texts[i] for i in te])

        rel = relative_framing(texts, groups, P[:, 2] - P[:, 0])
        side = np.array([side_features(t, self.ctx) for t in texts])
        llm = np.array([self.llm_fn(t) for t in texts]) if self.llm_fn else [None] * len(d)
        M = np.array([self._meta_row(P[i], groups[i], side[i], rel[i], llm[i], train=True)
                      for i in range(len(d))])
        self.meta = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)).fit(M, y)
        self.text_model = _text_model().fit(texts, y)
        self.explainer = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True, stop_words="english"),
            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")).fit(texts, y)
        return self

    # ---- inference ----
    def _evidence(self, t, cls, k=6):
        vec, clf = self.explainer.steps[0][1], self.explainer.steps[1][1]
        ci = list(clf.classes_).index(LABELS[cls])
        x = vec.transform([t]).tocoo()
        names = vec.get_feature_names_out()
        contrib = sorted(((x.data[j] * clf.coef_[ci][x.col[j]], names[x.col[j]]) for j in range(x.nnz)),
                         reverse=True)[:k]
        return [w for _, w in contrib]

    def analyze(self, text: str, source: str = "", peers=()) -> dict:
        """peers: texts about the SAME event from OTHER outlets (enables relative framing)."""
        t = clean(text, [source])
        p_text = _proba(self.text_model, [t])[0]
        rel = 0.0
        if peers:
            pp = _proba(self.text_model, [clean(x) for x in peers])
            rel = (p_text[2] - p_text[0]) - float(np.mean(pp[:, 2] - pp[:, 0]))
        side = side_features(t, self.ctx)
        llm_p = np.asarray(self.llm_fn(t)) if self.llm_fn else None
        row = self._meta_row(p_text, source, side, rel, llm_p)
        P = np.zeros(3)
        for j, c in enumerate(self.meta.classes_):
            P[LABELS.index(c)] = self.meta.predict_proba([row])[0][j]
        i = int(P.argmax())

        # disagreement: spread between independent lean estimates (each in about [-1, 1])
        src_p = self.ctx.source_prior.get((source or "").lower())
        votes = [p_text[2] - p_text[0], np.tanh(side[2] * 2)]
        if src_p is not None and self.use_src:
            votes.append(src_p[2] - src_p[0])
        if llm_p is not None:
            votes.append(llm_p[2] - llm_p[0])
        disagreement = float(np.std(votes)) if len(votes) > 1 else 0.0

        abstain = P[i] < self.abstain_below or disagreement > self.max_dis
        return {
            "label": "uncertain" if abstain else LABELS[i],
            "probs": dict(zip(LABELS, P.round(3))),
            "lean_score": round(float(P[2] - P[0]), 3),          # -1 left ... +1 right
            "confidence": round(float(P[i]), 3),
            "disagreement": round(disagreement, 3),
            "sensationalism": round(sensationalism(t), 3),       # separate axis
            "evidence": {
                "top_terms": self._evidence(t, i),
                "left_coded_per_1k / right_coded_per_1k": side[:2].round(2).tolist(),
                "entity_tone (+ favours right actors)": round(float(side[2]), 3),
                "relative_to_peers": round(float(rel), 3) if peers else None,
                "quoted_voices": voice_balance(t, self.ctx),
            },
        }


# --------------------------------------------------------------------------
# Evaluation: leave-sources-out, plus the leakage gap
# --------------------------------------------------------------------------
def ece(P, y, bins=10):
    conf, acc = P.max(1), (P.argmax(1) == y).astype(float)
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def evaluate(df: pd.DataFrame, ctx: Context, n_splits=5, **kw):
    d = df.reset_index(drop=True)
    texts = np.array([clean(t, [s]) for t, s in zip(d.text, d.source)], dtype=object)
    y = d.label.values

    def cv_text(splitter, groups=None):
        pred = np.empty(len(d), dtype=object)
        for tr, te in splitter.split(texts, y, groups):
            pred[te] = np.array(LABELS)[_proba(_text_model().fit(texts[tr], y[tr]), texts[te]).argmax(1)]
        return f1_score(y, pred, average="macro", labels=LABELS)

    f1_rand = cv_text(StratifiedKFold(n_splits, shuffle=True, random_state=0))
    f1_grp = cv_text(GroupKFold(n_splits), d.source.values)
    print(f"[text only] macro-F1 random split: {f1_rand:.3f} | source-grouped: {f1_grp:.3f} | "
          f"LEAKAGE GAP: {f1_rand - f1_grp:.3f}  (>0.20 = model detects publishers)")

    pred, probs = [], np.zeros((len(d), 3))
    for tr, te in GroupKFold(n_splits).split(texts, y, d.source.values):
        eng = BiasEngine(ctx, **kw).fit(d.iloc[tr])
        for i in te:
            r = eng.analyze(d.text[i], d.source[i])
            probs[i] = [r["probs"][l] for l in LABELS]
            pred.append((i, r["label"]))
    pred = dict(pred)
    lab = np.array([pred[i] for i in range(len(d))])
    ans = lab != "uncertain"
    print(f"[full engine, unseen outlets] coverage: {ans.mean():.2%} | "
          f"accuracy on answered: {accuracy_score(y[ans], lab[ans]):.3f} | "
          f"macro-F1 on answered: {f1_score(y[ans], lab[ans], average='macro', labels=LABELS):.3f} | "
          f"ECE: {ece(probs, np.array([LABELS.index(v) for v in y])):.3f}")
    # Also run with use_source_prior=False to see how much the prior is carrying.
"""Lightweight text-cleaning helpers used across the feature extractors."""

import re
import ssl
import nltk
from nltk.corpus import stopwords

# Allow NLTK download on first run even with self-signed certs (common on macOS).
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

try:
    _STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    _STOPWORDS = set(stopwords.words("english"))


_PUNCT_RE = re.compile(r"[^a-zA-Z0-9\s]")
_WS_RE    = re.compile(r"\s+")


def clean(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    if text is None:
        return ""
    text = str(text).lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def tokens(text: str, drop_stop: bool = True) -> list:
    cleaned = clean(text)
    toks = cleaned.split()
    if drop_stop:
        toks = [t for t in toks if t not in _STOPWORDS and len(t) > 1]
    return toks


def label_to_score(label) -> float:
    """SciEntsBank label (0..3) -> normalised score (1.0 .. 0.0).
    Defensive: handles strings/floats and clamps out-of-range values."""
    try:
        lbl = int(float(label))
    except (TypeError, ValueError):
        return 0.0
    lbl = max(0, min(3, lbl))
    return 1.0 - (lbl / 3.0)

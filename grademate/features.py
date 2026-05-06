"""Feature extractors used by both the ML branch and the DL branch."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .preprocess import clean, tokens


# ── Lexical features ────────────────────────────────────────────────────────

def keyword_overlap(reference: str, student: str) -> float:
    """Jaccard-style overlap of content tokens between reference and student answer."""
    ref_toks = set(tokens(reference, drop_stop=True))
    stu_toks = set(tokens(student,   drop_stop=True))
    if not ref_toks:
        return 0.0
    return len(ref_toks & stu_toks) / len(ref_toks)


def length_ratio(reference: str, student: str) -> float:
    """Ratio of student-length to reference-length, clamped to [0, 2]."""
    r = max(1, len(clean(reference).split()))
    s = len(clean(student).split())
    return min(2.0, s / r)


def shared_bigrams(reference: str, student: str) -> float:
    """Fraction of reference bigrams that appear in the student's answer."""
    def bigrams(t):
        toks = clean(t).split()
        return {f"{a} {b}" for a, b in zip(toks, toks[1:])}
    ref_b, stu_b = bigrams(reference), bigrams(student)
    if not ref_b:
        return 0.0
    return len(ref_b & stu_b) / len(ref_b)


# ── TF-IDF cosine similarity ────────────────────────────────────────────────

class TfidfSimilarity:
    """Fits one global TF-IDF vocab over the full training corpus, then exposes
    per-pair cosine similarity at inference time."""

    def __init__(self, ngram_range=(1, 2), max_features=20000):
        self.vec = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words="english",
        )
        self._fitted = False

    def fit(self, corpus):
        self.vec.fit([clean(t) for t in corpus])
        self._fitted = True
        return self

    def similarity(self, a: str, b: str) -> float:
        if not self._fitted:
            raise RuntimeError("TfidfSimilarity not fitted yet.")
        a_clean, b_clean = clean(a), clean(b)
        if not a_clean or not b_clean:
            return 0.0
        m = self.vec.transform([a_clean, b_clean])
        return float(cosine_similarity(m[0], m[1])[0, 0])


# ── Sentence-BERT semantic similarity ───────────────────────────────────────

class SemanticEncoder:
    """Lazy wrapper around sentence-transformers. Encodes text → 384-dim vector."""

    _model = None
    _device = None

    @classmethod
    def _ensure(cls):
        if cls._model is not None:
            return
        import torch
        from sentence_transformers import SentenceTransformer
        if torch.cuda.is_available():
            cls._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            cls._device = "mps"
        else:
            cls._device = "cpu"
        cls._model = SentenceTransformer(config.SBERT_MODEL_NAME, device=cls._device)

    @classmethod
    def encode(cls, texts, batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        cls._ensure()
        if isinstance(texts, str):
            texts = [texts]
        emb = cls._model.encode(
            [clean(t) or " " for t in texts],
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb.astype(np.float32)

    @classmethod
    def cosine(cls, a_vec: np.ndarray, b_vec: np.ndarray) -> float:
        # Vectors are L2-normalised so dot product == cosine.
        return float(np.dot(a_vec, b_vec))


# ── Combined classical-feature vector ───────────────────────────────────────

CLASSICAL_FEATURE_NAMES = [
    "keyword_overlap",
    "tfidf_cosine",
    "semantic_cosine",
    "length_ratio",
    "bigram_overlap",
]


def classical_feature_vector(
    question: str,
    reference: str,
    student: str,
    tfidf: TfidfSimilarity,
    ref_embed: np.ndarray | None = None,
    stu_embed: np.ndarray | None = None,
) -> np.ndarray:
    """Build the 5-dimensional engineered-feature vector used by RF/GB."""
    if ref_embed is None:
        ref_embed = SemanticEncoder.encode(reference)[0]
    if stu_embed is None:
        stu_embed = SemanticEncoder.encode(student)[0]

    return np.array([
        keyword_overlap(reference, student),
        tfidf.similarity(reference, student),
        SemanticEncoder.cosine(ref_embed, stu_embed),
        length_ratio(reference, student),
        shared_bigrams(reference, student),
    ], dtype=np.float32)

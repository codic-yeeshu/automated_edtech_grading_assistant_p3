"""End-to-end inference pipeline used by the FastAPI backend."""

from __future__ import annotations

import json
import joblib
import numpy as np
from pathlib import Path

from . import config
from . import dl_model
from .features import (
    SemanticEncoder, TfidfSimilarity,
    keyword_overlap, length_ratio, shared_bigrams,
    classical_feature_vector,
)
from .stacking import stack_predictions
from .preprocess import clean


# ── Cached singletons ───────────────────────────────────────────────────────

_state: dict = {"loaded": False}


def is_ready() -> bool:
    """Return True iff all model artifacts exist on disk."""
    paths = [
        config.TFIDF_PATH, config.RF_PATH, config.GB_PATH,
        config.DL_PATH, config.META_PATH,
    ]
    return all(Path(p).exists() for p in paths)


def _load():
    if _state.get("loaded"):
        return

    if not is_ready():
        raise RuntimeError(
            "GradeMate models not found. Run `python -m scripts.train` first."
        )

    _state["tfidf"]    = joblib.load(config.TFIDF_PATH)
    _state["rf"]       = joblib.load(config.RF_PATH)
    _state["gb"]       = joblib.load(config.GB_PATH)
    _state["meta"]     = joblib.load(config.META_PATH)
    _state["dl"]       = dl_model.load(config.DL_PATH)

    if config.METRICS_PATH.exists():
        with open(config.METRICS_PATH) as f:
            _state["metrics"] = json.load(f)
    else:
        _state["metrics"] = {}

    # Warm up the embedder
    SemanticEncoder.encode("warmup")
    _state["loaded"] = True


def get_metrics() -> dict:
    """Return metrics produced by the training script (if any)."""
    if not _state.get("loaded"):
        try:
            _load()
        except RuntimeError:
            return {}
    return _state.get("metrics", {})


# ── Grading core ────────────────────────────────────────────────────────────

def grade(
    question: str,
    reference: str,
    student: str,
    max_marks: float = 10.0,
) -> dict:
    """Run the full ML+DL stacking pipeline on one (question, reference, student)
    triple and return a dict suitable for direct JSON serialisation."""
    _load()

    # If the student answer is empty, short-circuit.
    if not clean(student):
        return _empty_result(max_marks, reason="Empty student answer.")

    # 1) Embed reference + student once and reuse.
    ref_emb = SemanticEncoder.encode(reference)[0]
    stu_emb = SemanticEncoder.encode(student)[0]

    # 2) Classical engineered features (5-d vector).
    feat = classical_feature_vector(
        question=question,
        reference=reference,
        student=student,
        tfidf=_state["tfidf"],
        ref_embed=ref_emb,
        stu_embed=stu_emb,
    )
    feat_2d = feat.reshape(1, -1)

    rf_pred = float(np.clip(_state["rf"].predict(feat_2d)[0], 0.0, 1.0))
    gb_pred = float(np.clip(_state["gb"].predict(feat_2d)[0], 0.0, 1.0))

    # 3) Deep-learning prediction over pairwise embedding interactions.
    dl_pred = float(np.clip(
        dl_model.predict_score(_state["dl"], ref_emb, stu_emb), 0.0, 1.0
    ))

    # 4) Stacking meta-regressor.
    meta_in   = stack_predictions(
        np.array([rf_pred]), np.array([gb_pred]), np.array([dl_pred])
    )
    final_norm = float(np.clip(_state["meta"].predict(meta_in)[0], 0.0, 1.0))
    final_score = round(final_norm * max_marks, 2)

    return {
        "score":        final_score,
        "max_marks":    max_marks,
        "percentage":   round(final_norm * 100, 1),
        "verdict":      _verdict(final_norm),
        "details": {
            "features": {
                "keyword_overlap": round(float(feat[0]), 4),
                "tfidf_cosine":    round(float(feat[1]), 4),
                "semantic_cosine": round(float(feat[2]), 4),
                "length_ratio":    round(float(feat[3]), 4),
                "bigram_overlap":  round(float(feat[4]), 4),
            },
            "base_models": {
                "random_forest":     round(rf_pred, 4),
                "gradient_boost":    round(gb_pred, 4),
                "deep_regressor":    round(dl_pred, 4),
            },
            "meta_score_normalised": round(final_norm, 4),
        },
    }


def grade_image(
    image_path: str,
    question: str,
    reference: str,
    max_marks: float = 10.0,
) -> dict:
    """OCR the image and feed the recognised text into `grade()`."""
    from .ocr import extract_text, is_available
    if not is_available():
        raise RuntimeError(
            "Tesseract OCR is not available. Install it via `brew install tesseract` "
            "(macOS) and `pip install pytesseract`."
        )
    student_text = extract_text(image_path)
    result       = grade(question, reference, student_text, max_marks)
    result["extracted_text"] = student_text
    return result


# ── Helpers ─────────────────────────────────────────────────────────────────

def _verdict(norm_score: float) -> str:
    if norm_score >= 0.85: return "Excellent"
    if norm_score >= 0.70: return "Good"
    if norm_score >= 0.50: return "Partial"
    if norm_score >= 0.25: return "Weak"
    return "Incorrect"


def _empty_result(max_marks: float, reason: str) -> dict:
    return {
        "score":        0.0,
        "max_marks":    max_marks,
        "percentage":   0.0,
        "verdict":      "Incorrect",
        "details":      {"reason": reason},
    }

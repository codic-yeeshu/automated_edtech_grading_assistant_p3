"""End-to-end training script.

Pipeline
────────
1. Load SciEntsBank train.csv + test.csv.
2. Fit a global TF-IDF vectoriser over all text in the corpus.
3. Encode (reference, student) pairs with sentence-transformers/all-MiniLM-L6-v2.
4. Build the 5-d engineered classical-feature matrix.
5. Train RandomForestRegressor and GradientBoostingRegressor.
6. Train a small PyTorch MLP regression head over pairwise embedding interactions.
7. Get out-of-fold predictions from each base learner and fit a Ridge stacking
   meta-regressor on top.
8. Evaluate the full stack on the held-out test split and dump metrics.json.

Run from the project root:

    python -m scripts.train               # full training (~2-4 min on M2 Pro)
    python -m scripts.train --limit 1500  # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold

# Ensure project root is on path even if invoked as a plain script.
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from grademate import config
from grademate import dl_model
from grademate.features import (
    SemanticEncoder, TfidfSimilarity,
    keyword_overlap, length_ratio, shared_bigrams,
)
from grademate.ml_models import (
    build_random_forest, build_gradient_boost, regression_metrics,
)
from grademate.stacking import build_meta, stack_predictions
from grademate.preprocess import label_to_score, clean


# ────────────────────────────────────────────────────────────────────────────

def load_split(csv_path, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["question", "reference_answer", "student_answer", "label"])
    df["question"]         = df["question"].astype(str)
    df["reference_answer"] = df["reference_answer"].astype(str)
    df["student_answer"]   = df["student_answer"].astype(str)
    df["target"]           = df["label"].apply(label_to_score).astype(np.float32)
    if limit:
        df = df.head(limit)
    return df.reset_index(drop=True)


def encode_corpus(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (ref_embeddings, student_embeddings) both shape (N, 384)."""
    print(f"[encode] embedding {len(df)} reference + {len(df)} student answers …")
    t0 = time.time()
    ref_emb = SemanticEncoder.encode(df["reference_answer"].tolist(), batch_size=64,  show_progress=True)
    stu_emb = SemanticEncoder.encode(df["student_answer"].tolist(),   batch_size=64,  show_progress=True)
    print(f"[encode] done in {time.time() - t0:.1f}s  (ref={ref_emb.shape}, stu={stu_emb.shape})")
    return ref_emb, stu_emb


def build_classical_matrix(df: pd.DataFrame, ref_emb: np.ndarray, stu_emb: np.ndarray, tfidf: TfidfSimilarity) -> np.ndarray:
    """Build the (N, 5) engineered-feature matrix used by the ML branch."""
    rows = []
    # Pre-compute cosine using L2-normalised vectors.
    sem_cos = (ref_emb * stu_emb).sum(axis=1)

    for i, row in df.iterrows():
        rows.append([
            keyword_overlap(row["reference_answer"], row["student_answer"]),
            tfidf.similarity(row["reference_answer"], row["student_answer"]),
            float(sem_cos[i]),
            length_ratio(row["reference_answer"], row["student_answer"]),
            shared_bigrams(row["reference_answer"], row["student_answer"]),
        ])
    return np.asarray(rows, dtype=np.float32)


# ────────────────────────────────────────────────────────────────────────────

def main(limit: int | None = None, dl_epochs: int | None = None):
    print("=" * 60)
    print("GradeMate — Hybrid ML+DL Training")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    train_df = load_split(config.TRAIN_CSV, limit=limit)
    test_df  = load_split(config.TEST_CSV)
    print(f"[data] train={len(train_df)}  test={len(test_df)}")

    # ── 2. Fit global TF-IDF ─────────────────────────────────────────────────
    print("[tfidf] fitting global vocab …")
    tfidf = TfidfSimilarity()
    corpus = (
        train_df["reference_answer"].tolist()
        + train_df["student_answer"].tolist()
        + train_df["question"].tolist()
    )
    tfidf.fit(corpus)
    joblib.dump(tfidf, config.TFIDF_PATH)
    print(f"[tfidf] saved → {config.TFIDF_PATH}")

    # ── 3. Embeddings ────────────────────────────────────────────────────────
    train_ref_emb, train_stu_emb = encode_corpus(train_df)
    test_ref_emb,  test_stu_emb  = encode_corpus(test_df)

    # ── 4. Classical feature matrix ──────────────────────────────────────────
    print("[features] building engineered feature matrix …")
    X_cls_train = build_classical_matrix(train_df, train_ref_emb, train_stu_emb, tfidf)
    X_cls_test  = build_classical_matrix(test_df,  test_ref_emb,  test_stu_emb,  tfidf)
    y_train     = train_df["target"].to_numpy(dtype=np.float32)
    y_test      = test_df["target"].to_numpy(dtype=np.float32)

    # ── 5. Train Random Forest + Gradient Boosting ───────────────────────────
    print("[ml] training Random Forest …")
    t0 = time.time()
    rf = build_random_forest()
    rf.fit(X_cls_train, y_train)
    print(f"[ml] RF trained in {time.time() - t0:.1f}s")

    print("[ml] training Gradient Boosting …")
    t0 = time.time()
    gb = build_gradient_boost()
    gb.fit(X_cls_train, y_train)
    print(f"[ml] GB trained in {time.time() - t0:.1f}s")

    joblib.dump(rf, config.RF_PATH)
    joblib.dump(gb, config.GB_PATH)
    print(f"[ml] saved → {config.RF_PATH.name}, {config.GB_PATH.name}")

    # ── 6. Train DL regression head ──────────────────────────────────────────
    print("[dl] building pair-features for DL branch …")
    X_dl_train = dl_model.build_pair_features(train_ref_emb, train_stu_emb)
    X_dl_test  = dl_model.build_pair_features(test_ref_emb,  test_stu_emb)

    dl_model_obj, dl_history = dl_model.train_dl_regressor(
        X_train=X_dl_train, y_train=y_train,
        X_val=X_dl_test, y_val=y_test,
        epochs=dl_epochs or config.DL_EPOCHS,
    )
    dl_model.save(dl_model_obj, config.DL_PATH)
    print(f"[dl] saved → {config.DL_PATH.name}")

    # ── 7. Out-of-fold base predictions for stacking ─────────────────────────
    print("[stack] generating out-of-fold base-learner predictions (5-fold KFold) …")
    rf_oof = np.zeros(len(train_df), dtype=np.float32)
    gb_oof = np.zeros(len(train_df), dtype=np.float32)
    dl_oof = np.zeros(len(train_df), dtype=np.float32)

    kf = KFold(n_splits=5, shuffle=True, random_state=config.SEED)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cls_train), start=1):
        print(f"[stack] fold {fold}/5  train={len(tr_idx)}  val={len(va_idx)}")
        # Fold-level RF & GB
        rf_f = build_random_forest().fit(X_cls_train[tr_idx], y_train[tr_idx])
        gb_f = build_gradient_boost().fit(X_cls_train[tr_idx], y_train[tr_idx])
        rf_oof[va_idx] = rf_f.predict(X_cls_train[va_idx])
        gb_oof[va_idx] = gb_f.predict(X_cls_train[va_idx])

        # Fold-level DL (fewer epochs to keep total time bounded)
        dl_f, _ = dl_model.train_dl_regressor(
            X_train=X_dl_train[tr_idx], y_train=y_train[tr_idx],
            epochs=max(2, (dl_epochs or config.DL_EPOCHS) // 2),
            verbose=False,
        )
        # Predict on val fold
        import torch
        device = next(dl_f.parameters()).device
        with torch.no_grad():
            dl_oof[va_idx] = dl_f(
                torch.from_numpy(X_dl_train[va_idx]).to(device)
            ).cpu().numpy()

    rf_oof = np.clip(rf_oof, 0.0, 1.0)
    gb_oof = np.clip(gb_oof, 0.0, 1.0)
    dl_oof = np.clip(dl_oof, 0.0, 1.0)

    print("[stack] training Ridge meta-regressor …")
    meta_X  = stack_predictions(rf_oof, gb_oof, dl_oof)
    meta    = build_meta()
    meta.fit(meta_X, y_train)
    joblib.dump(meta, config.META_PATH)
    print(f"[stack] saved → {config.META_PATH.name}")
    print(f"[stack] meta weights: rf={meta.coef_[0]:.3f}  gb={meta.coef_[1]:.3f}  dl={meta.coef_[2]:.3f}  bias={meta.intercept_:.3f}")

    # ── 8. Evaluate on held-out test split ───────────────────────────────────
    print("[eval] evaluating on held-out test split …")
    rf_test = np.clip(rf.predict(X_cls_test), 0.0, 1.0)
    gb_test = np.clip(gb.predict(X_cls_test), 0.0, 1.0)
    import torch
    device = next(dl_model_obj.parameters()).device
    with torch.no_grad():
        dl_test = dl_model_obj(torch.from_numpy(X_dl_test).to(device)).cpu().numpy()
    dl_test = np.clip(dl_test, 0.0, 1.0)

    meta_test_X  = stack_predictions(rf_test, gb_test, dl_test)
    final_test   = np.clip(meta.predict(meta_test_X), 0.0, 1.0)

    metrics = {
        "dataset": {
            "train_size": int(len(train_df)),
            "test_size":  int(len(test_df)),
            "name":       "SciEntsBank",
        },
        "random_forest":   regression_metrics(y_test, rf_test),
        "gradient_boost":  regression_metrics(y_test, gb_test),
        "deep_regressor":  regression_metrics(y_test, dl_test),
        "stacking_ensemble": regression_metrics(y_test, final_test),
        "meta_weights": {
            "rf":   float(meta.coef_[0]),
            "gb":   float(meta.coef_[1]),
            "dl":   float(meta.coef_[2]),
            "bias": float(meta.intercept_),
        },
        "dl_history": dl_history,
    }
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("Final test-set performance (lower MAE / RMSE = better)")
    print("=" * 60)
    for name, m in [
        ("Random Forest",      metrics["random_forest"]),
        ("Gradient Boosting",  metrics["gradient_boost"]),
        ("Deep Regressor",     metrics["deep_regressor"]),
        ("Stacking Ensemble",  metrics["stacking_ensemble"]),
    ]:
        print(f"  {name:<22}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}")
    print(f"\nMetrics JSON → {config.METRICS_PATH}")
    print("Training complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",     type=int, default=None,
                        help="Cap training rows (for quick smoke runs)")
    parser.add_argument("--dl-epochs", type=int, default=None,
                        help="Override DL training epochs")
    args = parser.parse_args()
    main(limit=args.limit, dl_epochs=args.dl_epochs)

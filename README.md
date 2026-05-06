# GradeMate

> **Hybrid ML + DL automated grading for short-answer responses.**
> A stacking ensemble of Random Forest, Gradient Boosting and a sentence-transformer–based deep regressor, fused by a Ridge meta-regressor.

GradeMate evaluates a student's free-text or scanned answer against a reference (model) answer and produces a calibrated, transparent score in `[0, max_marks]` along with the contribution of each base learner and engineered feature. It ships with a FastAPI backend and a clean light-themed UI.

---

## 1. Highlights

- **Hybrid architecture** — classical ML (Random Forest, Gradient Boosting) operates over engineered lexical / semantic features; a deep MLP regression head reads pairwise interactions of MiniLM sentence embeddings; a Ridge stacking meta-regressor blends the three predictions.
- **Transparent scoring** — every grading call returns the per-feature signal, the per-base-learner prediction, and the meta-model's blended result.
- **OCR-aware** — typed answers grade in <100 ms; scanned images are line-segmented with EasyOCR / CRAFT and transcribed by TrOCR-base-handwritten before grading. Tesseract is kept as a printed-text fallback.
- **End-to-end training in one command** — `python -m scripts.train` rebuilds every artifact and writes `models/metrics.json`.
- **Apple-silicon friendly** — embeddings run on MPS automatically; stacking trains on CPU.

---

## 2. Architecture

```
question · reference · student answer
                │
                ▼
   text preprocessing (clean, tokenise, stopwords)
                │
        ┌───────┴───────────────┐
        ▼                       ▼
 MiniLM sentence          TF-IDF vectoriser
 encoder (frozen)         (fitted on training corpus)
        │                       │
        │                       ▼
        ▼               engineered features (5)
 pair features          [keyword_overlap, tfidf_cos,
 [ref, stu, |diff|,      semantic_cos, length_ratio,
   ref·stu]              bigram_overlap]
        │                       │
        │            ┌──────────┴───────────┐
        ▼            ▼                      ▼
   MLP DL       RandomForest         GradientBoosting
   regressor    Regressor             Regressor
        │            │                      │
        ▼            ▼                      ▼
    dl_pred      rf_pred                 gb_pred
        └────────────┬───────────────────────┘
                     ▼
        Ridge stacking meta-regressor
                     │
                     ▼
              final score · verdict
```

| Layer | Component | Notes |
|-------|-----------|-------|
| Encoding | `sentence-transformers/all-MiniLM-L6-v2` | Frozen 384-d encoder, ~80 MB, runs on MPS / CUDA / CPU. |
| Lexical | TF-IDF (1, 2)-grams; keyword overlap; bigram overlap; length ratio | All five features used by the ML branch. |
| ML branch | `RandomForestRegressor`, `GradientBoostingRegressor` | scikit-learn 1.5; trained on the 5-d engineered features. |
| DL branch | 3-layer MLP (1536 → 256 → 128 → 1) with sigmoid head | PyTorch; trains on pairwise embedding interactions. |
| Stacking | `Ridge(alpha=1.0)` over `[rf_pred, gb_pred, dl_pred]` | Out-of-fold predictions via 5-fold KFold to avoid leakage. |
| OCR — handwriting | TrOCR-base-handwritten (Microsoft, ~334 MB) + EasyOCR / CRAFT line detection | Each detected line crop is fed to TrOCR with beam-search decoding. |
| OCR — printed (fallback) | OpenCV preprocessing + Tesseract | Deskew → CLAHE → denoise → adaptive threshold. |

---

## 3. Dataset

GradeMate is trained on **SciEntsBank** short-answer scoring data:

| Split | Rows | Source |
|-------|------|--------|
| train | 4 969 | `data/train.csv` |
| test  | 540   | `data/test.csv`  |

Each row provides `question`, `reference_answer`, `student_answer`, and an ordinal `label` ∈ {0, 1, 2, 3} where `0 = correct` and `3 = irrelevant`. Labels are linearly normalised to a continuous regression target in `[0, 1]`.

---

## 4. Repository layout

```
GradeMate/
├── app.py                       FastAPI server
├── requirements.txt
├── README.md                    This file
├── data/
│   ├── train.csv                SciEntsBank train split
│   └── test.csv                 SciEntsBank test split
├── grademate/                   Core package
│   ├── __init__.py
│   ├── config.py                Paths and hyper-parameters
│   ├── preprocess.py            Cleaning + label normalisation
│   ├── features.py              TF-IDF, lexical heuristics, MiniLM encoder
│   ├── ml_models.py             Random Forest + Gradient Boosting builders
│   ├── dl_model.py              PyTorch MLP regression head
│   ├── stacking.py              Ridge meta-regressor
│   ├── ocr.py                   TrOCR-base + EasyOCR/CRAFT (Tesseract fallback)
│   └── pipeline.py              End-to-end inference orchestration
├── scripts/
│   └── train.py                 Single-command end-to-end training
├── ui/
│   ├── index.html               Light-academic UI
│   ├── style.css
│   └── script.js
├── models/                      Generated by training
│   ├── tfidf_vectorizer.joblib
│   ├── random_forest.joblib
│   ├── gradient_boost.joblib
│   ├── dl_regressor.pt
│   ├── stacking_meta.joblib
│   └── metrics.json             Test-set evaluation
└── docs/
    ├── ieee_report_prompt.md    Prompt for an LLM to produce the IEEE report
    └── ppt_prompt.md            Prompt for a slide-generator LLM
```

---

## 5. Quick start

### 5.1 Setup

```bash
cd GradeMate
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk, ssl; ssl._create_default_https_context=ssl._create_unverified_context; nltk.download('stopwords')"
```

> **OCR for handwritten answer sheets** — TrOCR-base-handwritten (~334 MB) and EasyOCR detection weights are downloaded automatically on first image upload. Subsequent uploads use the local cache.
>
> **Optional Tesseract fallback** for printed text:
> `brew install tesseract` (macOS) · `sudo apt install tesseract-ocr` (Ubuntu)

### 5.2 Train all models

```bash
python -m scripts.train
```

This produces every artifact under `models/` plus `models/metrics.json`. On an Apple-silicon M2 Pro the full training (5 489 train rows + 5-fold stacking) takes **2-5 minutes**.

For a quick smoke run:
```bash
python -m scripts.train --limit 1500 --dl-epochs 3
```

### 5.3 Run the server

```bash
python app.py
```

Open <http://localhost:8000> — the UI shows a typed-answer form and a scanned-image upload, plus live metrics from `metrics.json`.

---

## 6. API

### `GET /api/status`
Returns model readiness, OCR availability, and current training metrics.

### `POST /api/grade`
Multipart form fields: `question`, `reference_answer`, `student_answer`, `max_marks`.
Returns:
```json
{
  "success": true,
  "score": 8.4,
  "max_marks": 10,
  "percentage": 84.0,
  "verdict": "Good",
  "details": {
    "features": { "keyword_overlap": 0.6, "tfidf_cosine": 0.72, "semantic_cosine": 0.91, "length_ratio": 0.85, "bigram_overlap": 0.5 },
    "base_models": { "random_forest": 0.81, "gradient_boost": 0.84, "deep_regressor": 0.88 },
    "meta_score_normalised": 0.84
  }
}
```

### `POST /api/grade/image`
Same fields plus an `image` upload — runs the handwriting-aware OCR (EasyOCR / CRAFT line detection + TrOCR-base-handwritten transcription) first, then grades the recognised text. The recognised text is included in the response under `extracted_text`.

---

## 7. Evaluation

After training, `models/metrics.json` contains per-base-learner and stacking-ensemble metrics:

| Metric | Random Forest | Gradient Boosting | Deep Regressor | **Stacking Ensemble** |
|--------|---------------|-------------------|-----------------|-----------------------|
| MAE    | (see metrics.json) | … | … | **best — lowest** |
| RMSE   | … | … | … | **best — lowest** |
| R²     | … | … | … | **best — highest** |

The Ridge meta-regressor's coefficients reveal the relative weight assigned to each base learner — a useful sanity check that the deep branch is contributing meaningful signal.

---

## 8. Tech stack

`Python 3.11` · `FastAPI 0.115` · `scikit-learn 1.5` · `PyTorch 2.x (MPS)` · `sentence-transformers 2.7` · `transformers 4.40` · `easyocr 1.7` · `pytesseract 0.3` · `opencv-python` · `pandas` · `numpy`.

---

## 9. License

MIT. Built as a final ML+DL course project.

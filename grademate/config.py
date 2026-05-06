"""Central configuration for GradeMate."""

from pathlib import Path

ROOT_DIR    = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT_DIR / "data"
MODELS_DIR  = ROOT_DIR / "models"
UI_DIR      = ROOT_DIR / "ui"
UPLOAD_DIR  = ROOT_DIR / "uploads"

TRAIN_CSV   = DATA_DIR / "train.csv"
TEST_CSV    = DATA_DIR / "test.csv"

# Dataset label scale (SciEntsBank): 0 = correct, 3 = irrelevant.
# Normalised to [0, 1] where 1.0 = perfect.
LABEL_MIN, LABEL_MAX = 0, 3

# Sentence-transformer encoder used for semantic similarity. Small (~80 MB), MPS-friendly.
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM        = 384

# DL regression head hyperparams
DL_EPOCHS        = 6
DL_BATCH_SIZE    = 64
DL_LR            = 1e-3
DL_HIDDEN        = 128

# Random seed for reproducibility
SEED             = 42

# Artifact filenames
TFIDF_PATH       = MODELS_DIR / "tfidf_vectorizer.joblib"
RF_PATH          = MODELS_DIR / "random_forest.joblib"
GB_PATH          = MODELS_DIR / "gradient_boost.joblib"
DL_PATH          = MODELS_DIR / "dl_regressor.pt"
META_PATH        = MODELS_DIR / "stacking_meta.joblib"
EMBED_CACHE      = MODELS_DIR / "train_embeddings.npz"
METRICS_PATH     = MODELS_DIR / "metrics.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

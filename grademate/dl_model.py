"""Deep-learning regression head.

Architecture
────────────
input  : concat[ ref_embed (384) , stu_embed (384) , |ref-stu| (384) , ref*stu (384) ]
         shape = (1536,)
hidden : Linear -> ReLU -> Dropout
output : Linear -> Sigmoid -> score in [0, 1]

This sits on top of frozen MiniLM embeddings so training takes <1 minute on CPU
and a few seconds on Apple-Silicon MPS.
"""

from __future__ import annotations

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import config


# ── Model ───────────────────────────────────────────────────────────────────

class GradeRegressor(nn.Module):
    def __init__(self, embed_dim: int = config.EMBED_DIM, hidden: int = config.DL_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Feature builder for the DL branch ───────────────────────────────────────

def build_pair_features(ref_emb: np.ndarray, stu_emb: np.ndarray) -> np.ndarray:
    """Pairwise embedding interactions (concat + |diff| + elementwise-product).
    Accepts (D,) or (N, D) and broadcasts correctly."""
    if ref_emb.ndim == 1:
        ref_emb = ref_emb[None, :]
        stu_emb = stu_emb[None, :]
    diff = np.abs(ref_emb - stu_emb)
    prod = ref_emb * stu_emb
    return np.concatenate([ref_emb, stu_emb, diff, prod], axis=1).astype(np.float32)


# ── Training ────────────────────────────────────────────────────────────────

def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_dl_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = config.DL_EPOCHS,
    batch_size: int = config.DL_BATCH_SIZE,
    lr: float = config.DL_LR,
    verbose: bool = True,
) -> tuple[GradeRegressor, dict]:
    """Train the regression head. Returns (model, history)."""
    device = _pick_device()
    if verbose:
        print(f"[DL] device = {device}, train_n={len(X_train)}, dim={X_train.shape[1]}")

    model = GradeRegressor(embed_dim=config.EMBED_DIM).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32)))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        train_loss = total / len(train_ds)
        history["train_loss"].append(train_loss)

        val_loss = None
        if X_val is not None and y_val is not None:
            val_loss = evaluate(model, X_val, y_val, device)
            history["val_loss"].append(val_loss)

        if verbose:
            msg = f"[DL] epoch {epoch}/{epochs}  train_mse={train_loss:.4f}"
            if val_loss is not None:
                msg += f"  val_mse={val_loss:.4f}"
            print(msg)

    return model, history


def evaluate(model: GradeRegressor, X: np.ndarray, y: np.ndarray, device=None) -> float:
    if device is None:
        device = _pick_device()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X).to(device)).cpu().numpy()
    return float(np.mean((pred - y) ** 2))


# ── Persistence ─────────────────────────────────────────────────────────────

def save(model: GradeRegressor, path):
    torch.save({"state_dict": model.state_dict(), "embed_dim": config.EMBED_DIM}, path)


def load(path) -> GradeRegressor:
    ckpt   = torch.load(path, map_location="cpu")
    model  = GradeRegressor(embed_dim=ckpt["embed_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ── Inference ───────────────────────────────────────────────────────────────

def predict_score(model: GradeRegressor, ref_emb: np.ndarray, stu_emb: np.ndarray) -> float:
    feats = build_pair_features(ref_emb, stu_emb)
    device = _pick_device()
    model.to(device).eval()
    with torch.no_grad():
        out = model(torch.from_numpy(feats).to(device))
    return float(out.cpu().item())

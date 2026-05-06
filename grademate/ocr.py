"""OCR layer for GradeMate.

Two engines, picked at runtime:

* `tesseract` — fast, OK for typed/printed text, poor on cursive handwriting.
* `trocr` (default) — TrOCR-base-handwritten (Microsoft, ~334 MB) for
  handwritten answer sheets. Lines are detected with EasyOCR's CRAFT detector
  and each line crop is then fed to TrOCR for recognition.

The default engine for the public `extract_text()` entry-point is auto-selected
based on availability — TrOCR if its dependencies are installed, otherwise
Tesseract. The FastAPI layer always passes `engine="trocr"` so the demo gets
the better recogniser.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 · Pre-processing — shared by both engines
# ─────────────────────────────────────────────────────────────────────────────

def _deskew(gray: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(gray < 200))
    if len(coords) == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


def _binarize(image_path: str) -> np.ndarray:
    """Deskew → CLAHE → denoise → adaptive threshold → morphological close."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = _deskew(gray)

    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray     = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(gray, h=10, searchWindowSize=21)

    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=10,
    )
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2A · Tesseract engine (typed/printed text, fast fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _tesseract_extract(image_path: str) -> str:
    try:
        import pytesseract
    except ImportError as e:
        raise RuntimeError("pytesseract not installed.") from e

    processed = _binarize(image_path)
    cfg = (
        "--psm 6 --oem 3 "
        "-c tessedit_char_whitelist="
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        "0123456789.,;:?!()/\\-+= "
    )
    return _post_clean(pytesseract.image_to_string(processed, config=cfg))


def _tesseract_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2B · TrOCR-base-handwritten engine (handwriting)
# ─────────────────────────────────────────────────────────────────────────────

_TROCR_MODEL_NAME = "microsoft/trocr-base-handwritten"

_trocr_state: dict = {"model": None, "proc": None, "device": None}
_easyocr_reader = None


def _trocr_load() -> None:
    if _trocr_state["model"] is not None:
        return

    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"[OCR] loading TrOCR-base-handwritten on {device} …")
    proc  = TrOCRProcessor.from_pretrained(_TROCR_MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(_TROCR_MODEL_NAME).to(device)
    model.eval()
    _trocr_state.update(model=model, proc=proc, device=device)


def _easyocr_load():
    global _easyocr_reader
    if _easyocr_reader is not None:
        return _easyocr_reader
    import easyocr
    logger.info("[OCR] loading EasyOCR (CRAFT detector) …")
    try:
        _easyocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    except Exception:
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _easyocr_reader


def _detect_lines(image_path: str) -> List[Tuple[int, int, int, int]]:
    """Run EasyOCR's CRAFT detector to find word boxes, then group into lines.
    Returns axis-aligned line bboxes (x, y, w, h) sorted top-to-bottom."""
    reader  = _easyocr_load()
    results = reader.readtext(image_path)

    if not results:
        img = cv2.imread(image_path)
        return [(0, 0, img.shape[1], img.shape[0])]

    word_boxes = []
    for bbox, _, _ in results:
        xs = [int(p[0]) for p in bbox]
        ys = [int(p[1]) for p in bbox]
        word_boxes.append({
            "x0": min(xs), "x1": max(xs),
            "y0": min(ys), "y1": max(ys),
            "yc": (min(ys) + max(ys)) // 2,
        })
    word_boxes.sort(key=lambda b: b["yc"])

    lines = []
    current = []
    Y_THRESH = 25
    for box in word_boxes:
        if not current:
            current.append(box)
            continue
        avg_yc = sum(b["yc"] for b in current) / len(current)
        if abs(box["yc"] - avg_yc) < Y_THRESH:
            current.append(box)
        else:
            lines.append(current); current = [box]
    if current:
        lines.append(current)

    out = []
    for line in lines:
        x0 = min(b["x0"] for b in line)
        x1 = max(b["x1"] for b in line)
        y0 = min(b["y0"] for b in line)
        y1 = max(b["y1"] for b in line)
        out.append((x0, y0, x1 - x0, y1 - y0))
    return out


def _trocr_extract(image_path: str) -> str:
    """Detect lines with EasyOCR/CRAFT, then transcribe each with TrOCR."""
    import torch
    from PIL import Image

    _trocr_load()
    proc, model, device = _trocr_state["proc"], _trocr_state["model"], _trocr_state["device"]

    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(image_path)
    gray   = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    binary = _binarize(image_path)
    h, w   = gray.shape[:2]

    boxes = _detect_lines(image_path)
    if not boxes:
        boxes = [(0, 0, w, h)]

    transcripts = []
    for (x, y, bw, bh) in boxes:
        # Generous vertical padding so ascenders/descenders don't get clipped.
        pad_x = max(4, int(bw * 0.05))
        pad_y = max(8, int(bh * 0.40))
        x1, y1 = max(0, x - pad_x),       max(0, y - pad_y)
        x2, y2 = min(w, x + bw + pad_x),  min(h, y + bh + pad_y)

        # Use the binarised crop (bg ruled lines erased) but soften edges so
        # the strokes look continuous to TrOCR.
        crop = binary[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop = cv2.GaussianBlur(crop, (3, 3), 0)

        pil_img = Image.fromarray(crop).convert("RGB")
        pixel_values = proc(images=pil_img, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            ids = model.generate(
                pixel_values,
                max_new_tokens=128,
                num_beams=4,
                early_stopping=True,
            )
        line_text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if line_text:
            transcripts.append(line_text)

    return _post_clean(" ".join(transcripts))


def _trocr_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import easyocr  # noqa: F401
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def _post_clean(raw: str) -> str:
    import re
    text = re.sub(r"[^\x20-\x7E\n]", "", raw)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_text(image_path: str, engine: str = "auto") -> str:
    """OCR an image and return the recognised text.

    Args:
        image_path: filesystem path to the image.
        engine: "trocr" (handwriting), "tesseract" (printed), or "auto".

    The auto mode prefers TrOCR when available, otherwise falls back to
    Tesseract.
    """
    image_path = str(image_path)
    if not Path(image_path).exists():
        raise FileNotFoundError(image_path)

    if engine == "auto":
        engine = "trocr" if _trocr_available() else "tesseract"

    if engine == "trocr":
        try:
            return _trocr_extract(image_path)
        except Exception as e:
            logger.warning(f"[OCR] TrOCR failed ({e!r}) — falling back to Tesseract.")
            return _tesseract_extract(image_path)

    if engine == "tesseract":
        return _tesseract_extract(image_path)

    raise ValueError(f"Unknown OCR engine: {engine!r}")


def is_available() -> bool:
    """True iff at least one OCR engine is usable."""
    return _trocr_available() or _tesseract_available()


def best_engine() -> str:
    """Name of the engine `extract_text(engine='auto')` will pick."""
    return "trocr" if _trocr_available() else ("tesseract" if _tesseract_available() else "none")

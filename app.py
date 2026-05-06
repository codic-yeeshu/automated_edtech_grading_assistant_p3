"""GradeMate FastAPI backend.

Endpoints
─────────
GET  /                  → serves the UI (ui/index.html)
GET  /api/status        → readiness + dataset metrics
POST /api/grade         → grade a typed answer
POST /api/grade/image   → OCR an uploaded image, then grade
"""

from __future__ import annotations

import os
import shutil
import logging
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from grademate import config
from grademate import pipeline as grading_pipeline
from grademate.ocr import is_available as ocr_available, best_engine as ocr_best_engine


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("grademate")


app = FastAPI(
    title="GradeMate API",
    version="1.0.0",
    description="Hybrid ML+DL automated short-answer grading.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=config.UI_DIR), name="static")


# ── UI ──────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_ui():
    return FileResponse(config.UI_DIR / "index.html")


# ── Status ──────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def status():
    ready = grading_pipeline.is_ready()
    metrics = grading_pipeline.get_metrics() if ready else {}
    return JSONResponse({
        "ready":      ready,
        "ocr_ready":  ocr_available(),
        "ocr_engine": ocr_best_engine(),
        "metrics":    metrics,
        "models": {
            "ml":  ["RandomForestRegressor", "GradientBoostingRegressor"],
            "dl":  ["MiniLM-Encoder + MLP-Regressor"],
            "meta": "RidgeRegression (Stacking)",
            "ocr": "TrOCR-base-handwritten + EasyOCR/CRAFT" if ocr_best_engine() == "trocr" else "Tesseract",
        },
    })


# ── Grade typed answer ──────────────────────────────────────────────────────

@app.post("/api/grade")
async def grade(
    question: str = Form(...),
    reference_answer: str = Form(...),
    student_answer: str = Form(...),
    max_marks: float = Form(10.0),
):
    if not grading_pipeline.is_ready():
        raise HTTPException(503, "Models not trained. Run `python -m scripts.train` first.")

    try:
        result = grading_pipeline.grade(
            question=question,
            reference=reference_answer,
            student=student_answer,
            max_marks=max_marks,
        )
        return {"success": True, **result}
    except Exception as e:
        logger.exception("grade failed")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ── Grade scanned image (OCR + grading) ─────────────────────────────────────

@app.post("/api/grade/image")
async def grade_image(
    image: UploadFile = File(...),
    question: str = Form(...),
    reference_answer: str = Form(...),
    max_marks: float = Form(10.0),
):
    if not grading_pipeline.is_ready():
        raise HTTPException(503, "Models not trained. Run `python -m scripts.train` first.")
    if not ocr_available():
        raise HTTPException(
            503,
            "Tesseract OCR not available. Install via `brew install tesseract` or use the typed-answer mode.",
        )

    import time, secrets
    suffix   = Path(image.filename or "upload.jpg").suffix or ".jpg"
    tmp_path = config.UPLOAD_DIR / f"tmp_{os.getpid()}_{int(time.time()*1000)}_{secrets.token_hex(3)}{suffix}"
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        result = grading_pipeline.grade_image(
            image_path=str(tmp_path),
            question=question,
            reference=reference_answer,
            max_marks=max_marks,
        )
        return {"success": True, **result}
    except Exception as e:
        logger.exception("grade_image failed")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

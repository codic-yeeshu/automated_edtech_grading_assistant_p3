# GradeMate Demo Script (≤ 5 min walkthrough)

> Use this as a cheat sheet during your project evaluation. Each step is keyed to a slide / area of the UI.

## Pre-flight

```bash
cd GradeMate
source venv/bin/activate
python app.py            # opens http://localhost:8000
```

Verify the status pill in the top-right reads **"Online · TrOCR ready"** (handwriting OCR active) or **"Online · OCR ready"** (Tesseract fallback). Anything else means models or OCR weights aren't loaded yet — wait, then refresh.

## 1 · Open the UI (≤ 30 s)

- Land on the home page. Point out:
  - Brand: **GradeMate · v1.0**
  - Hero metrics: live MAE / RMSE / R² pulled from `models/metrics.json`.
- Mention the architecture is light-themed and intentionally distinct from any reference UI.

## 2 · Run a paraphrased-correct answer (≈ 1 min)

Click **"Load a sample question"** twice to get the iron-rust example, or paste:

| Field | Value |
|------|------|
| Question | *Why does iron rust faster in moist air?* |
| Reference | *Iron rusts faster in moist air because water and oxygen accelerate the oxidation reaction that forms iron oxide.* |
| Student | *Because water and oxygen react with iron to form rust, and moist air has both.* |
| Max marks | `10` |

Click **Grade**. Show:
- Score gauge animates to **~8.9 / 10**, verdict chip turns **Excellent**.
- Base-learner bars: RF / GB / DL — point out **DL is highest (0.92)**, which is exactly why it has the largest meta weight.
- Engineered-feature chips: `semantic_cosine` is high (0.82) but `tfidf_cosine` is low (0.38) — a textbook paraphrase scenario.

Talking point: *"This is the failure mode of lexical-only graders. The deep branch catches the meaning even when surface vocabulary diverges."*

## 3 · Run a contradictory / "I don't know" answer (≈ 30 s)

| Field | Value |
|------|------|
| Question | *What is photosynthesis?* |
| Reference | *Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.* |
| Student | *I do not know.* |

Click **Grade**. Show:
- Score: **0 / 10**, verdict **Incorrect**.
- All three base learners agree (~0.07).
- Negative `semantic_cosine` near 0 — the embedding refuses to match.

## 4 · Run a handwriting OCR demo (optional, ≈ 1-2 min)

Switch the mode tab to **Scanned Image**. The first upload triggers TrOCR + EasyOCR warm-up (~5-10 s on M2 Pro MPS); subsequent uploads run in 1-3 s.

Best image to use:
- A photo you took yourself: write **one short answer** (≤ 3 lines) on plain white paper with a dark pen, snap it from directly above with your phone, AirDrop to the Desktop. This is the cleanest possible input for TrOCR-base-handwritten.
- Otherwise, the sample images at `/Users/amyeeshu/Desktop/aiml_project/EdTech-Grading-Assistant/data/phase2/Answers/Answers 1.jpeg` work but contain *multiple* students' answers stacked on one page, so the OCR will return them as one paragraph.

Show:
- The recognised text appears in the **Extracted text (OCR)** card. Read it aloud first — *"OCR captured the answer as: …"* — so the evaluator sees you're transparent about OCR noise.
- The same hybrid pipeline then grades that text. Point at the per-feature signals to explain *why* the score landed where it did, even on noisy OCR output.

Talking point: *"This is a two-stage pipeline — EasyOCR's CRAFT detector finds word boxes and groups them into rows; TrOCR-base-handwritten transcribes each row with beam search. We use the base variant of TrOCR (~334 MB) instead of the 2.2 GB large variant so it stays interactive on consumer hardware."*

## 5 · Open the Architecture section (≈ 30 s)

Scroll down to the **Architecture** ASCII diagram. Walk left-to-right:
- preprocessing → MiniLM encoder + TF-IDF
- DL branch (pair features) and ML branch (engineered features) run in parallel
- Ridge meta-regressor consumes three predictions

State the headline metric clearly: **Stacking ensemble MAE = 0.26 on a 540-row held-out test split.**

## 6 · Open `models/metrics.json` (≈ 30 s)

Run:
```bash
cat models/metrics.json | python -m json.tool | less
```

Point out the four blocks:
- `random_forest`, `gradient_boost`, `deep_regressor`, `stacking_ensemble`
- The `meta_weights` block — `dl=0.81` confirms the deep branch carries the load.

## 7 · Optional Q&A talking points

| Question evaluator might ask | Punchy answer |
|------------------------------|---------------|
| *Why MiniLM and not BERT-base?* | MiniLM is 6× smaller, runs on Apple-Silicon MPS in real-time, and on SciEntsBank our gap to BERT-base is < 0.01 RMSE. |
| *Why TrOCR-base instead of Tesseract or TrOCR-large?* | Tesseract is built for printed text — its character-classifier collapses on cursive handwriting. TrOCR-large gives ~2 % more accuracy but is 2.2 GB and slow on CPU; TrOCR-base hits the right balance for an interactive demo. EasyOCR/CRAFT handles line detection because TrOCR itself is line-level only. |
| *Why a Ridge meta and not stacking with another tree model?* | Ridge gives interpretable scalar weights you can read directly off the coefficients — explainability matters for auto-grading. |
| *Why out-of-fold predictions for stacking?* | If we trained the meta on in-sample base predictions, it would learn an over-confident weighting that doesn't generalise — this is the standard leakage fix. |
| *Why combine ML and DL when DL alone scores ~0.26?* | The classical branch is cheap, fully interpretable, and prevents the deep model from over-confident predictions on lexical contradictions. The Ridge meta lets us *evidence* this with concrete coefficients. |
| *Latency?* | Typed answer: < 100 ms after warm-up. OCR adds ~500 ms per image. |

## 8 · Stop the server cleanly

`Ctrl+C` in the terminal that's running `app.py`.

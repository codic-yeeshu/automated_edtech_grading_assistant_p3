// ─────────────────────────────────────────────────────────────────────
// GradeMate · Frontend
// ─────────────────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

const state = {
    mode: "text",                 // "text" | "image"
    imageFile: null,
    sampleIdx: 0,
};

const SAMPLES = [
    {
        question: "Why does iron rust faster in moist air?",
        reference: "Iron rusts faster in moist air because the presence of water and oxygen accelerates the oxidation reaction that forms iron oxide.",
        student: "Because water and oxygen react with iron to form rust, and moist air has both.",
        max_marks: 10,
        subject: "Chemistry · Class 9",
    },
    {
        question: "What is photosynthesis?",
        reference: "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.",
        student: "Plants use sun, CO2 and water to make food and release oxygen.",
        max_marks: 10,
        subject: "Biology · Class 8",
    },
    {
        question: "Explain Newton's third law of motion.",
        reference: "Newton's third law states that for every action there is an equal and opposite reaction.",
        student: "If you push something it pushes you back equally.",
        max_marks: 5,
        subject: "Physics · Class 9",
    },
    {
        question: "How do you separate salt from water?",
        reference: "The water can be evaporated, leaving the salt behind.",
        student: "By evaporation — the water turns into vapour and the salt is left.",
        max_marks: 5,
        subject: "Science · Class 6",
    },
];


// ── INITIALISATION ─────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    bindModeTabs();
    bindForm();
    bindSampleButton();
    bindFileInput();
    refreshStatus();
});


// ── STATUS POLLING ─────────────────────────────────────────────────
async function refreshStatus() {
    const pill = $("status-pill");
    const txt  = $("status-text");
    pill.classList.remove("is-ready", "is-pending", "is-error");
    pill.classList.add("is-pending");
    txt.textContent = "Checking…";

    try {
        const r = await fetch("/api/status");
        const j = await r.json();
        if (j.ready) {
            pill.classList.remove("is-pending");
            pill.classList.add("is-ready");
            const ocrLabel = j.ocr_ready
                ? (j.ocr_engine === "trocr" ? "TrOCR ready" : "OCR ready")
                : "text-only";
            txt.textContent = `Online · ${ocrLabel}`;
            $("not-trained").hidden = true;

            // Inject metrics into hero
            const m = j.metrics?.stacking_ensemble;
            if (m) {
                $("stat-mae-value").textContent  = m.mae?.toFixed(3) ?? "–";
                $("stat-rmse-value").textContent = m.rmse?.toFixed(3) ?? "–";
                $("stat-r2-value").textContent   = (m.r2 != null) ? m.r2.toFixed(3) : "–";
            }
        } else {
            pill.classList.remove("is-pending");
            pill.classList.add("is-error");
            txt.textContent = "Models not trained";
            $("not-trained").hidden = false;
        }
    } catch (e) {
        pill.classList.remove("is-pending");
        pill.classList.add("is-error");
        txt.textContent = "Offline";
    }
}


// ── MODE SWITCHER ──────────────────────────────────────────────────
function bindModeTabs() {
    document.querySelectorAll(".mode-tab").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".mode-tab").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            state.mode = btn.dataset.mode;
            $("mode-text-field").hidden  = state.mode !== "text";
            $("mode-image-field").hidden = state.mode !== "image";
        });
    });
}


// ── FILE PICKER + DRAG-DROP ────────────────────────────────────────
function bindFileInput() {
    const input  = $("f-image");
    const drop   = document.querySelector(".dropzone");
    const prev   = $("f-image-preview");

    input.addEventListener("change", e => handleFiles(e.target.files));

    drop.addEventListener("dragover", e => { e.preventDefault(); drop.style.background = "var(--indigo-50)"; drop.style.borderColor = "var(--indigo)"; });
    drop.addEventListener("dragleave", () => { drop.style.background = ""; drop.style.borderColor = ""; });
    drop.addEventListener("drop", e => {
        e.preventDefault();
        drop.style.background = ""; drop.style.borderColor = "";
        if (e.dataTransfer?.files?.length) handleFiles(e.dataTransfer.files);
    });

    function handleFiles(fileList) {
        const f = fileList?.[0];
        if (!f) return;
        state.imageFile = f;
        const url = URL.createObjectURL(f);
        prev.src = url;
        prev.hidden = false;
    }
}


// ── SAMPLE BUTTON ──────────────────────────────────────────────────
function bindSampleButton() {
    $("btn-sample").addEventListener("click", () => {
        const s = SAMPLES[state.sampleIdx % SAMPLES.length];
        state.sampleIdx += 1;
        $("f-question").value  = s.question;
        $("f-reference").value = s.reference;
        $("f-student").value   = s.student;
        $("f-maxmarks").value  = s.max_marks;
        $("f-subject").value   = s.subject;
        // Switch to text mode
        document.querySelector('.mode-tab[data-mode="text"]').click();
        toast("Sample loaded — click Grade to run the pipeline.");
    });
}


// ── FORM SUBMIT ────────────────────────────────────────────────────
function bindForm() {
    $("grade-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const question  = $("f-question").value.trim();
        const reference = $("f-reference").value.trim();
        const maxMarks  = parseFloat($("f-maxmarks").value || "10");

        if (!question || !reference) return toast("Question and reference answer are required.", true);

        let endpoint = "/api/grade";
        let body;

        if (state.mode === "text") {
            const studentText = $("f-student").value.trim();
            if (!studentText) return toast("Enter the student's typed answer or switch to image mode.", true);

            body = new FormData();
            body.append("question",         question);
            body.append("reference_answer", reference);
            body.append("student_answer",   studentText);
            body.append("max_marks",        maxMarks);
        } else {
            if (!state.imageFile) return toast("Pick an image first.", true);
            endpoint = "/api/grade/image";
            body = new FormData();
            body.append("image",            state.imageFile);
            body.append("question",         question);
            body.append("reference_answer", reference);
            body.append("max_marks",        maxMarks);
        }

        setBusy(true);
        try {
            const r = await fetch(endpoint, { method: "POST", body });
            const j = await r.json();
            if (!j.success) throw new Error(j.error || j.detail || "Grading failed.");
            renderResult(j);
        } catch (err) {
            toast(err.message || String(err), true);
        } finally {
            setBusy(false);
        }
    });
}

function setBusy(busy) {
    $("btn-grade").disabled = busy;
    $("btn-spinner").hidden = !busy;
    $("btn-grade").querySelector(".btn-arrow").style.display = busy ? "none" : "";
}


// ── RESULT RENDERING ───────────────────────────────────────────────
function renderResult(r) {
    $("result-empty").hidden = true;
    $("result-body").hidden  = false;

    const norm = (r.percentage ?? 0) / 100;
    animateGauge(norm);

    $("score-num").textContent = formatScore(r.score);
    $("score-den").textContent = r.max_marks;
    $("score-pct").textContent = `${r.percentage?.toFixed(1) ?? "0"}%`;

    // Verdict chip
    const v = (r.verdict || "—").toLowerCase();
    const chip = $("verdict-chip");
    chip.textContent = r.verdict || "—";
    chip.className = "verdict-chip v-" + v;

    // Base learner bars
    const base = r.details?.base_models || {};
    setBar("rf", base.random_forest);
    setBar("gb", base.gradient_boost);
    setBar("dl", base.deep_regressor);

    // Engineered features as chips
    const feats = r.details?.features || {};
    const order = [
        ["keyword_overlap", "Keyword overlap"],
        ["tfidf_cosine",    "TF-IDF cosine"],
        ["semantic_cosine", "Semantic cosine"],
        ["bigram_overlap",  "Bigram overlap"],
        ["length_ratio",    "Length ratio"],
    ];
    $("feat-chips").innerHTML = order
        .filter(([k]) => feats[k] !== undefined)
        .map(([k, label]) => `<span class="chip">${label} <b>${(+feats[k]).toFixed(3)}</b></span>`)
        .join("");

    // OCR text (image mode only)
    if (r.extracted_text) {
        $("ocr-block").hidden = false;
        $("ocr-text").textContent = r.extracted_text;
    } else {
        $("ocr-block").hidden = true;
    }

    // Smooth scroll into view on small screens
    if (window.innerWidth < 980) {
        $("result-body").scrollIntoView({ behavior: "smooth", block: "start" });
    }
}

function setBar(suffix, value) {
    const v = Math.max(0, Math.min(1, +value || 0));
    $("bar-" + suffix).style.width = (v * 100) + "%";
    $("val-" + suffix).textContent = v.toFixed(2);
}

function animateGauge(norm) {
    // Half-circle path is ~251 length (PI*r where r=80 → ~251)
    const ARC = 251;
    const off = ARC * (1 - Math.max(0, Math.min(1, norm)));
    const fg = $("gauge-fg");
    fg.style.strokeDashoffset = off;
    // Colour graduation by score
    const stops = [
        [0.85, "#10B981"], // emerald
        [0.65, "#4F46E5"], // indigo
        [0.45, "#F59E0B"], // amber
        [0.0,  "#F43F5E"], // rose
    ];
    const colour = stops.find(([t]) => norm >= t)?.[1] ?? "#F43F5E";
    fg.style.stroke = colour;
}

function formatScore(s) {
    if (Number.isInteger(s)) return s;
    return (+s).toFixed(2);
}


// ── TOAST ──────────────────────────────────────────────────────────
let _toastTimer = null;
function toast(msg, isError = false) {
    const t = $("toast");
    t.textContent = msg;
    t.classList.toggle("error", isError);
    t.hidden = false;
    requestAnimationFrame(() => t.classList.add("show"));
    clearTimeout(_toastTimer);
    _toastTimer = setTimeout(() => {
        t.classList.remove("show");
        setTimeout(() => { t.hidden = true; }, 250);
    }, 3500);
}

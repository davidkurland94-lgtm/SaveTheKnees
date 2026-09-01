"""The Save the Knees API.

Run from the repo root so `functions` imports:

    uvicorn serving.my_api:app --reload --port 8000

Then open http://localhost:8000/docs

Every route is a thin wrapper: parse the URL, call something in functions/,
turn the answer into JSON. No modelling logic lives here.
"""

import io
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from functions import catalog
from functions.catalog import DataUnavailable
from functions.labels import LABELS
from functions.sequence_to_tensor import AXIS_PLANE, sequence_to_tensor

app = FastAPI(
    title="Save the Knees",
    description="Knee MRI in, twelve finding probabilities out.",
)

# The front end is served from a different origin, so without this the browser
# blocks every call. Comma-separated list; "*" is fine until the front end has
# a real domain, after which pin it -- there is no auth on this API.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    """Cloud Run health checks and curious humans both land here."""
    return {"service": "save-the-knees", "docs": "/docs", "health": "/health"}

# Uploads and hand-written reports are ours, not the dataset's, so they go in a
# scratch directory rather than anywhere near train.csv.
#
# WRITE_ROOT is separate from DATA_ROOT on purpose. In Cloud Run the dataset is
# a read-only bucket mount, so writing under it fails; this defaults to the
# container's own /tmp, which is writable but EPHEMERAL and PER-INSTANCE --
# anything written here is gone when the instance is recycled and invisible to
# every other instance. Point it at a writable mount (or replace the JSON store
# with a real database) before anyone depends on an upload surviving.
WRITE_ROOT = Path(os.environ.get("WRITE_ROOT", catalog.DATA_ROOT / "_tmp"))
UPLOAD_ROOT = WRITE_ROOT / "uploads"
REPORT_STORE = WRITE_ROOT / "my_reports.json"

# Where <study>/<series>/*.dcm actually sit, best first. catalog.series_dir
# points at data/train_series/, which is the Kaggle layout and is NOT what is
# checked out locally -- the gold images live under their own directory, and
# anything I POST lands in _tmp/uploads. So look in all three.
IMAGE_ROOTS = [
    catalog.DATA_ROOT / "train_series",
    catalog.DATA_ROOT / "savetheknees_gold_images",
    UPLOAD_ROOT,
]


def series_path(study_uid: str, series_uid: str) -> Path | None:
    """First image root that actually holds this series, or None."""
    for root in IMAGE_ROOTS:
        candidate = root / study_uid / series_uid
        if candidate.is_dir() and any(candidate.glob("*.dcm")):
            return candidate
    return None


def guard(fn, *args, **kwargs):
    """A missing dataset is a 503, not a 500 traceback."""
    try:
        return fn(*args, **kwargs)
    except DataUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


def optional(fn, *args, **kwargs):
    """For the extras -- the pilkwang labels, the English translations -- that
    are not always checked out. Missing means "no data", not a broken request,
    so it degrades to None instead of failing the whole route."""
    try:
        return fn(*args, **kwargs)
    except DataUnavailable:
        return None


def require_study(study_uid: str):
    """404 unless the study exists. Most routes start with this."""
    study = guard(catalog.get_study, study_uid)
    if study is None:
        raise HTTPException(404, f"Unknown study: {study_uid}")
    return study


# ---------------------------------------------------------------------------
# Request / response shapes
# ---------------------------------------------------------------------------

class ReportIn(BaseModel):
    text: str = Field(min_length=1, description="The report body.")
    author: str = "david"


class StoredReport(BaseModel):
    study_uid: str
    text: str
    author: str
    created_at: str
    updated_at: str


class Prediction(BaseModel):
    study_uid: str
    model_status: str
    probabilities: dict[str, float]


class UploadPrediction(BaseModel):
    model_status: str
    n_slices_received: int
    predictions: dict[str, float]


class ReportTextIn(BaseModel):
    text: str


class ReportPrediction(BaseModel):
    model_status: str
    predictions: dict[str, float]


# ---------------------------------------------------------------------------
# The JSON file standing in for a database
# ---------------------------------------------------------------------------

def load_reports() -> dict:
    if not REPORT_STORE.exists():
        return {}
    return json.loads(REPORT_STORE.read_text())


def save_reports(reports: dict) -> None:
    REPORT_STORE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_STORE.write_text(json.dumps(reports, indent=2))


_model_cache = {}


def get_model():
    """Load the checkpoint once and keep it. Loading per request would add
    seconds of graph-building to every call."""
    if "model" not in _model_cache:
        from functions.predict import load_predictor
        _model_cache["model"], _model_cache["status"] = load_predictor()
    return _model_cache["model"], _model_cache["status"]


def get_report_model():
    """The text model + its vectorizer, loaded once; None when not on disk."""
    if "report" not in _model_cache:
        try:
            from models.report_model import load_report_predictor
            _model_cache["report"] = load_report_predictor()
        except Exception:
            _model_cache["report"] = None
    return _model_cache["report"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Is the process up, and can it see the dataset?"""
    try:
        n_studies = len(catalog.studies())
    except DataUnavailable as exc:
        return {"status": "ok", "dataset": "unavailable", "detail": str(exc)}
    return {"status": "ok", "dataset": "ready", "n_studies": n_studies}


# ---------------------------------------------------------------------------
# Dataset -- read-only routes over the corpus (migrated from serving/studies.py)
#
# Route order matters: /studies/golden is declared before /studies/{study_uid},
# otherwise FastAPI matches the literal "golden" as a study UID.
# ---------------------------------------------------------------------------

@app.get("/studies", tags=["dataset"])
def list_studies(limit: int = Query(100, ge=1, le=1000),
                 offset: int = Query(0, ge=0)):
    """Every study in the corpus, paginated (4,407 rows -- never dumped whole).

    golden=True marks the 58 hand-labelled studies; drill into one with
    /studies/{study_uid}.
    """
    df = guard(catalog.studies)
    page = df.iloc[offset:offset + limit]
    golden = page[LABELS].notna().all(axis=1)
    return {"total": int(len(df)), "offset": offset, "limit": limit,
            "studies": [{"study_uid": uid,
                         "golden": bool(golden.loc[uid]),
                         "has_report": bool(isinstance(row.Report, str) and row.Report.strip())}
                        for uid, row in page.iterrows()]}


@app.get("/studies/golden", tags=["dataset"])
def list_golden():
    """The 58 hand-labelled studies -- the only ground truth in the dataset."""
    rows = guard(catalog.golden_studies)
    return {"count": len(rows), "labels": LABELS, "studies": rows}


@app.get("/studies/{study_uid}", tags=["dataset"])
def get_study(study_uid: str):
    """One study's record, its series inline."""
    return require_study(study_uid)


@app.get("/studies/{study_uid}/report", tags=["dataset"])
def get_dataset_report(study_uid: str,
                       lang: str = Query("original", pattern="^(original|en)$")):
    """The radiologist's report from the dataset (NOT a user-written one).

    Mixed language corpus (half English, the rest es/tr/hr/bg/el/nl/de).
    Default: exactly as the dataset ships it. lang=en serves the machine
    translation the report model trains on (data/meta/reports_en.csv);
    404 when that cache is not mounted.
    """
    report = guard(catalog.get_report, study_uid, lang)
    if report is None:
        detail = (f"No English translation available for {study_uid}" if lang == "en"
                  else f"No report for study: {study_uid}")
        raise HTTPException(404, detail)
    return report


@app.get("/studies/{study_uid}/labels", tags=["dataset"])
def get_labels(study_uid: str):
    """All twelve pilkwang labels: probability, confidence, YES/NO/UNK verdict."""
    labels = guard(catalog.get_labels, study_uid)
    if labels is None:
        raise HTTPException(404, f"No pilkwang labels for study: {study_uid}")
    return labels


@app.get("/studies/{study_uid}/labels/{label}", tags=["dataset"])
def get_label(study_uid: str, label: str):
    """One label. Accepts the slug ("medial-meniscus", "bakers") or exact name."""
    if catalog.resolve_label(label) is None:
        raise HTTPException(404, f"Unknown label: {label}. Known: {sorted(catalog.SLUGS)}")
    result = guard(catalog.get_label, study_uid, label)
    if result is None:
        raise HTTPException(404, f"No pilkwang labels for study: {study_uid}")
    return result


@app.get("/studies/{study_uid}/predict", tags=["dataset"])
def predict_with_model(study_uid: str,
                       model: str = Query("fusion", pattern="^(sagittal|multiplane|fusion)$")):
    """Run one of OUR trained models on a study by UID.

    sagittal    the single-plane 3D CNN (also behind GET /predict/{uid})
    multiplane  the 3-seed sagittal+coronal+axial ensemble
    fusion      images + report, jointly trained -- the project's best (default)

    Needs the tensor cache (and, for fusion, the translated reports) under
    DATA_ROOT, which is why this is study-keyed rather than an upload.
    """
    from functions.predict import predict_study_with
    try:
        result = predict_study_with(study_uid, model)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    if result is None:
        raise HTTPException(404, f"{study_uid}: not cached for model '{model}' "
                                 "(needs the tensor cache; fusion also needs reports_en.csv)")
    return {"study_uid": study_uid, "model": model, "predictions": result}


@app.get("/studies/{study_uid}/series", tags=["dataset"])
def list_series(study_uid: str,
                plane: str | None = Query(None, description="Sagittal | Coronal | Axial")):
    """Every sequence of a study."""
    if plane is not None and plane not in AXIS_PLANE.values():
        raise HTTPException(400, f"plane must be one of {sorted(AXIS_PLANE.values())}")
    require_study(study_uid)
    rows = guard(catalog.list_series, study_uid, plane)
    return {"study_uid": study_uid, "count": len(rows), "series": rows}


@app.get("/studies/{study_uid}/series/{series_uid}", tags=["dataset"])
def get_series(study_uid: str, series_uid: str):
    """One sequence: plane, acquisition flags, slice count, whether files exist."""
    record = guard(catalog.get_series, study_uid, series_uid)
    if record is None:
        raise HTTPException(404, f"Series {series_uid} not found in study {study_uid}")
    return record


@app.get("/studies/{study_uid}/series/{series_uid}/tensor", tags=["dataset"])
def get_tensor(study_uid: str, series_uid: str):
    """The model-ready tensor as a .npy download: (24, 224, 224, 1) float32.

    Same function the model is served by, so what you download here is exactly
    what the network sees. Resolved through series_path, so uploaded and gold
    series work too, not only train_series/.
    """
    directory = series_path(study_uid, series_uid)
    if directory is None:
        raise HTTPException(404, f"No DICOM files for series {series_uid}")

    x = sequence_to_tensor(directory)
    if x is None:
        raise HTTPException(422, f"Could not build a tensor from {series_uid}")

    buffer = io.BytesIO()
    np.save(buffer, x)
    return Response(
        content=buffer.getvalue(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{series_uid}.npy"'},
    )


@app.get("/studies/{study_uid}/series/{series_uid}/preview.png", tags=["dataset"])
def get_preview(study_uid: str, series_uid: str,
                columns: int = Query(6, ge=1, le=24, description="montage width")):
    """A PNG contact sheet of the 24 sampled slices, for eyeballing a series."""
    from PIL import Image

    directory = series_path(study_uid, series_uid)
    if directory is None:
        raise HTTPException(404, f"No DICOM files for series {series_uid}")

    x = sequence_to_tensor(directory)
    if x is None:
        raise HTTPException(422, f"Could not build a tensor from {series_uid}")

    k, size = x.shape[0], x.shape[1]
    rows = -(-k // columns)
    sheet = np.zeros((rows * size, columns * size), dtype=np.uint8)
    for i in range(k):
        r, c = divmod(i, columns)
        sheet[r * size:(r + 1) * size, c * size:(c + 1) * size] = (x[i, :, :, 0] * 255).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(sheet).save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@app.post("/upload/{study_uid}/image_sequence")
async def upload_new_sequence(
    study_uid: str,
    series_uid: str = Query(..., description="Folder name to store the slices under."),
    files: list[UploadFile] = File(..., description="The .dcm slices."),
):
    """Save an uploaded sequence to data/_tmp/uploads/{study_uid}/{series_uid}/.

    async, unlike the rest of this file, because reading an upload off the
    wire is genuinely awaited.
    """
    if not files:
        raise HTTPException(400, "No files uploaded.")

    # A UID arriving from the network must never escape the upload root --
    # "../../etc" in a path segment would otherwise write outside it.
    destination = (UPLOAD_ROOT / study_uid / series_uid).resolve()
    if not destination.is_relative_to(UPLOAD_ROOT.resolve()):
        raise HTTPException(400, "Bad study_uid or series_uid.")
    destination.mkdir(parents=True, exist_ok=True)

    written = []
    for upload in files:
        name = Path(upload.filename or "").name
        if not name.lower().endswith(".dcm"):
            raise HTTPException(400, f"Not a DICOM file: {upload.filename}")
        with open(destination / name, "wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        written.append(name)

    return {
        "study_uid": study_uid,
        "series_uid": series_uid,
        "n_slices": len(written),
        "stored_at": str(destination),
        "files": sorted(written),
    }


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def resolve_series(study_uid: str, series_uid: str | None, plane: str | None):
    """Pick which series to render: the one asked for, or the first of a plane,
    or just the study's first available one."""
    candidates = guard(catalog.list_series, study_uid, plane)
    # catalog computes "available" against train_series/, so recompute it
    # against the roots that exist here.
    available = []
    for record in candidates:
        found = series_path(study_uid, record["series_uid"])
        if found is not None:
            available.append({**record, "available": True, "path": str(found)})
    if not available:
        raise HTTPException(
            404, f"No series files on disk for {study_uid}"
                 + (f" in plane {plane}" if plane else ""))
    if series_uid is None:
        return available[0]
    match = next((s for s in available if s["series_uid"] == series_uid), None)
    if match is None:
        raise HTTPException(404, f"Unknown or unavailable series: {series_uid}")
    return match


@app.get("/view/{study_uid}/3d_image_sequence")
def view_3d_sequence(
    study_uid: str,
    series_uid: str | None = None,
    plane: str | None = Query(None, pattern="^(Sagittal|Coronal|Axial)$"),
    downsample: int = Query(2, ge=1, le=8),
):
    """A marching-cubes isosurface of the series, as raw mesh JSON.

    Returns vertices and triangle faces so a front end can draw it with
    three.js / plotly. Deliberately not an HTML page: the mesh is tens of MB
    and the client should decide how to render it.
    """
    require_study(study_uid)
    chosen = resolve_series(study_uid, series_uid, plane)

    # scikit-image and scipy are dev extras, not in requirements.txt, so this
    # import stays inside the route -- the rest of the API boots without them.
    try:
        from functions.dicom_3d import build_surface, get_volume_with_spacing
        from functions.dicom_utils import load_series
    except ImportError as exc:
        raise HTTPException(
            503, f"3D extras not installed (scikit-image, scipy, plotly): {exc}")

    directory = Path(chosen["path"])
    datasets = load_series(str(directory))
    if not datasets:
        raise HTTPException(404, f"No readable slices in {directory}")

    volume, spacing = get_volume_with_spacing(datasets)
    verts, faces = build_surface(volume, spacing, downsample=downsample)

    return {
        "study_uid": study_uid,
        "series_uid": chosen["series_uid"],
        "plane": chosen["plane"],
        "spacing_mm": list(spacing),
        "n_vertices": int(len(verts)),
        "n_faces": int(len(faces)),
        "vertices": np.asarray(verts, dtype=float).round(2).tolist(),
        "faces": np.asarray(faces, dtype=int).tolist(),
    }


@app.get("/view/{study_uid}/2d_image_sequence",
         responses={200: {"content": {"image/png": {}}}})
def view_2d_sequence(
    study_uid: str,
    series_uid: str | None = None,
    plane: str | None = Query(None, pattern="^(Sagittal|Coronal|Axial)$"),
    columns: int = Query(6, ge=1, le=12),
):
    """A contact sheet PNG: the 24 slices the model sees, tiled into a grid.

    Built from sequence_to_tensor, so what you look at is exactly what gets
    fed to the network -- same crop, same window, same resize.
    """
    require_study(study_uid)
    chosen = resolve_series(study_uid, series_uid, plane)

    directory = Path(chosen["path"])
    tensor = sequence_to_tensor(directory)
    if tensor is None:
        raise HTTPException(404, f"No readable slices in {directory}")

    from PIL import Image

    slices = (tensor[..., 0] * 255).clip(0, 255).astype(np.uint8)
    k, height, width = slices.shape
    rows = -(-k // columns)                       # ceiling division
    sheet = np.zeros((rows * height, columns * width), np.uint8)
    for i, image in enumerate(slices):
        r, c = divmod(i, columns)
        sheet[r * height:(r + 1) * height, c * width:(c + 1) * width] = image

    buffer = io.BytesIO()
    Image.fromarray(sheet).save(buffer, format="PNG")
    return Response(buffer.getvalue(), media_type="image/png")


@app.get("/view/{study_uid}/information")
def view_patient_information(study_uid: str):
    """Everything the dataset knows about the study: series, planes, whether it
    is one of the 58 gold-labelled ones, and any report I have stored."""
    study = require_study(study_uid)
    report = optional(catalog.get_report, study_uid)
    labels = optional(catalog.get_labels, study_uid)
    return {
        **study,
        "report": report,
        "pilkwang_labels": labels["labels"] if labels else None,
        "my_report": load_reports().get(study_uid),
    }


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

@app.get("/predict/{study_uid}", response_model=Prediction)
def predict(
    study_uid: str,
    axis: str = Query("X", pattern="^[XYZ]$",
                      description="X=Sagittal, Y=Coronal, Z=Axial."),
):
    """Run the checkpointed image model over the study.

    functions.predict.predict_study would be the one-liner here, but it builds
    its tensor through study_to_tensor, which hardcodes data/train_series/.
    So the two halves are composed directly instead -- same series pick, same
    sequence_to_tensor, just resolved against the roots that exist locally.

    model_status is "untrained" when no weights were on disk. The response
    shape is identical either way, so check it before believing a number.
    """
    require_study(study_uid)
    chosen = resolve_series(study_uid, None, AXIS_PLANE[axis])

    # TensorFlow is slow to import and only needed here, so it is pulled in
    # lazily rather than at module import.
    from functions.predict import predict_tensor

    model, status = get_model()
    tensor = sequence_to_tensor(Path(chosen["path"]))
    if tensor is None:
        raise HTTPException(404, f"No readable slices for {study_uid}")

    probabilities = predict_tensor(model, tensor)
    return Prediction(
        study_uid=study_uid,
        model_status=status,
        probabilities={k: float(v) for k, v in probabilities.items()},
    )


@app.post("/predict", response_model=UploadPrediction, tags=["predict"])
async def predict_upload(files: list[UploadFile] = File(...)):
    """Upload every .dcm of ONE series (one plane of one study) -- for data
    that is NOT in the dataset yet.

    The whole series is needed: intensity is normalised with one percentile
    window PER SERIES, and 24 slices are sampled across it.
    """
    from functions.predict import predict_series
    dicoms = [f for f in files if (f.filename or "").lower().endswith(".dcm")]
    if not dicoms:
        raise HTTPException(400, "No .dcm files in the upload.")
    model, status = get_model()
    tmp = Path(tempfile.mkdtemp(prefix="stk_"))
    try:
        for upload in dicoms:
            with open(tmp / Path(upload.filename).name, "wb") as out:
                shutil.copyfileobj(upload.file, out)
        result = predict_series(model, tmp)
        if result is None:
            raise HTTPException(422, "Could not read a series from those files.")
        return UploadPrediction(model_status=status,
                                n_slices_received=len(dicoms), predictions=result)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/predict/report", response_model=ReportPrediction, tags=["predict"])
def predict_report_text(body: ReportTextIn):
    """English report text in, twelve finding probabilities out."""
    pair = get_report_model()
    if pair is None:
        raise HTTPException(503, "report model not available on this deployment")
    model, vectorizer = pair
    x = vectorizer.transform([body.text]).toarray().astype("float32")
    probs = model.predict(x, verbose=0)[0]
    return ReportPrediction(model_status="trained",
                            predictions={l: float(p) for l, p in zip(LABELS, probs)})


# ---------------------------------------------------------------------------
# Evaluation -- the referee's tables and model cards, what the notebook's
# remote mode reads. Thin JSON views over models/evaluate_labels.py and
# models/report_quality.py; a missing artifact answers 503, never a traceback.
# ---------------------------------------------------------------------------

def _records(table):
    """DataFrame (label index) -> JSON-safe list of rows."""
    out = table.reset_index()
    # NaN is not JSON, and pandas keeps NaN in float columns even after
    # where(..., None); go through object dtype so None actually lands.
    out = out.astype(object).where(out.notna(), None)
    return out.to_dict(orient="records")


def _guarded(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except FileNotFoundError as exc:
        raise HTTPException(503, f"evaluation artifact missing on the server: {exc}") from exc


@app.get("/report/table", tags=["evaluation"])
def report_table():
    """The full sheet: every reader vs the 58 gold studies (final_table)."""
    from models.evaluate_labels import final_table
    return {"rows": _records(_guarded(final_table))}


@app.get("/report/compare/{which}", tags=["evaluation"])
def report_compare(which: str):
    """image = David vs LLM . report = Kevin vs LLM . showdown = David vs best reader."""
    from models import evaluate_labels as ev
    fn = {"image": ev.image_model_vs_llm, "report": ev.report_model_vs_llm,
          "showdown": ev.david_vs_best_reader}.get(which)
    if fn is None:
        raise HTTPException(404, "which must be image | report | showdown")
    table = _guarded(fn)
    if table is None:
        raise HTTPException(503, "image model gold predictions not available yet")
    return {"rows": _records(table)}


@app.get("/report/verdicts", tags=["evaluation"])
def report_verdicts(top: int = Query(20, ge=1, le=500)):
    """Worst reports first: what the report said vs what the images think."""
    from models import report_quality as rq
    if not rq.OUT.exists():
        _guarded(rq.run, top=0)              # score the corpus once, quietly
    return {"rows": _guarded(rq.verdict_sheet, top).to_dict(orient="records")}


@app.get("/models/{name}/summary", tags=["evaluation"])
def model_summary(name: str):
    """Keras layer table + parameter count of one trained model, as text."""
    from functions.predict import STUDY_MODELS, _load_study_models
    if name == "report":
        pair = get_report_model()
        if pair is None:
            raise HTTPException(503, "report model not available on this deployment")
        model = pair[0]
    elif name in STUDY_MODELS:
        models = _load_study_models(name)
        if not models:
            raise HTTPException(503, f"no checkpoint for '{name}' on the server")
        model = models[0]
    else:
        raise HTTPException(404, "name must be sagittal | multiplane | fusion | report")
    lines = []
    model.summary(line_length=100, print_fn=lines.append)
    return {"name": name, "parameters": int(model.count_params()), "summary": "\n".join(lines)}


# ---------------------------------------------------------------------------
# My reports -- create / update / delete, backed by the JSON file above
# ---------------------------------------------------------------------------

@app.post("/create/{study_uid}/sequence_report",
          response_model=StoredReport, status_code=201)
def create_report(study_uid: str, body: ReportIn):
    """Write a new report. 409 if one already exists -- overwriting is PUT's
    job, and silently clobbering on POST is how people lose work."""
    require_study(study_uid)
    reports = load_reports()
    if study_uid in reports:
        raise HTTPException(409, f"A report for {study_uid} already exists; use PUT.")

    timestamp = now()
    record = {"study_uid": study_uid, "text": body.text, "author": body.author,
              "created_at": timestamp, "updated_at": timestamp}
    reports[study_uid] = record
    save_reports(reports)
    return record


@app.put("/update/{study_uid}/sequence_report", response_model=StoredReport)
def update_report(study_uid: str, body: ReportIn):
    """Replace an existing report. 404 if there is nothing to replace."""
    reports = load_reports()
    existing = reports.get(study_uid)
    if existing is None:
        raise HTTPException(404, f"No stored report for {study_uid}; use POST.")

    existing.update(text=body.text, author=body.author, updated_at=now())
    save_reports(reports)
    return existing


@app.delete("/delete/{study_uid}/sequence_report")
def delete_report(study_uid: str):
    reports = load_reports()
    if reports.pop(study_uid, None) is None:
        raise HTTPException(404, f"No stored report for {study_uid}.")
    save_reports(reports)
    return {"deleted": study_uid}

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

from functions import catalog, uploads
from functions.catalog import DataUnavailable
from functions.labels import LABELS
from functions.sequence_to_tensor import AXIS_PLANE, sequence_to_tensor, sorted_slice_paths

OPENAPI_TAGS = [
    {"name": "health", "description": "Liveness and dataset visibility."},
    {"name": "studies", "description": "Read-only browsing of the corpus: "
     "studies, reports, labels, series, tensors, previews."},
    {"name": "predict", "description": "The trained models. Three entry points, three cases: "
     "**a study already in the dataset** -> GET /studies/{uid}/predict?model= (sagittal | multiplane | fusion); "
     "**a NEW series you upload** -> POST /predict; "
     "**raw report text** -> POST /predict/report."},
    {"name": "evaluation", "description": "The referee: every reader scored on the 58 gold "
     "studies, verdict sheets, model cards. What models_pipe.ipynb reads remotely."},
    {"name": "upload", "description": "Add new DICOM series (stored under WRITE_ROOT)."},
    {"name": "views", "description": "Rendering: 3D mesh, 2D contact sheet, study information."},
    {"name": "reports", "description": "User-WRITTEN reports (create/update/delete) -- not the "
     "dataset's radiology reports, those live under /studies/{uid}/report. Saving one translates "
     "it to English on the way in, which is what unlocks the report model and, with the images, "
     "the fusion model."},
]

app = FastAPI(
    title="Save the Knees",
    description="Knee MRI in, twelve finding probabilities out. "
                "Sections read top to bottom: check health, browse studies, "
                "run a predict, audit the models in evaluation.",
    openapi_tags=OPENAPI_TAGS,
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

@app.get("/", tags=["health"])
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
    """404 unless the study exists, in the corpus OR in the upload store.

    Uploads are checked first and without `guard`: an uploaded study must stay
    reachable on a machine where the 17 GB dataset is not mounted, which is
    exactly the case a bare catalog lookup turns into a 503.
    """
    uploaded = uploads.get(study_uid)
    if uploaded is not None:
        # The record is written at upload time, when there is by definition no
        # report; the doctor's arrives later through /create. Read it fresh
        # rather than leaving the flag stuck on False and the report tab greyed.
        return {**uploaded, "has_report": bool(load_reports().get(study_uid))}
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
    # Filled at save time, not at read time. Both the fusion model and
    # POST /predict/report only understand English, and translating on every
    # prediction would put a network call to Google in front of every score --
    # so the English is written once, when the doctor saves.
    #
    # Defaulted rather than required: reports stored before this existed have
    # neither field, and a record written last week is not a validation error.
    language: str = "unknown"
    report_en: str = ""


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


def to_english(text: str) -> tuple[str, str]:
    """(English text, detected source language) for a report a doctor typed.

    The corpus reports were translated once, offline, into
    data/meta/reports_en.csv; a report written in the app a second ago is in no
    cache, so it is translated here on the way into the store.

    Degrades rather than fails, twice over. deep-translator and langdetect are
    small but they are still two more packages and one more outbound network
    call than the rest of this API needs, so a deployment without them -- or a
    translator that is down -- gets the ORIGINAL text back under the language
    "unknown". Scoring a Spanish report with an English model is a worse answer
    than scoring an English one, but it is a much better answer than refusing to
    save the doctor's work.
    """
    try:
        from functions.report_translation import translate_report
    except ImportError:
        return text, "unknown"
    try:
        return translate_report(text)
    except Exception:                    # noqa: BLE001 -- any translator failure
        return text, "unknown"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["health"])
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

@app.get("/studies", tags=["studies"])
def list_studies(limit: int = Query(100, ge=1, le=1000),
                 offset: int = Query(0, ge=0)):
    """Every study in the corpus, paginated (4,407 rows -- never dumped whole).

    golden=True marks the 58 hand-labelled studies; drill into one with
    /studies/{study_uid}.
    """
    # Uploaded studies come first and are paginated together with the corpus:
    # someone who has just uploaded one expects to find it, and hunting for it
    # on page 221 is not finding it.
    mine = [{"study_uid": r["study_uid"], "golden": False,
             "has_report": bool(load_reports().get(r["study_uid"])), "source": "upload"}
            for r in uploads.list_all()]

    # A missing corpus is not an error when there are uploads to show: the
    # dataset is a 17 GB mount that a laptop legitimately does not have, and an
    # uploaded study lives entirely in the write store.
    try:
        df = catalog.studies()
    except DataUnavailable:
        if not mine:
            raise
        df = None

    total = len(mine) + (0 if df is None else int(len(df)))

    rows = mine[offset:offset + limit]
    # Once the uploads are exhausted the corpus continues where they left off.
    corpus_offset = max(offset - len(mine), 0)
    if df is not None and len(rows) < limit:
        page = df.iloc[corpus_offset:corpus_offset + (limit - len(rows))]
        golden = page[LABELS].notna().all(axis=1)
        rows += [{"study_uid": uid,
                  "golden": bool(golden.loc[uid]),
                  "has_report": bool(isinstance(row.Report, str) and row.Report.strip()),
                  "source": "dataset"}
                 for uid, row in page.iterrows()]

    return {"total": total, "offset": offset, "limit": limit, "studies": rows}


@app.get("/studies/golden", tags=["studies"])
def list_golden():
    """The 58 hand-labelled studies -- the only ground truth in the dataset."""
    rows = guard(catalog.golden_studies)
    return {"count": len(rows), "labels": LABELS, "studies": rows}


@app.get("/studies/{study_uid}", tags=["studies"])
def get_study(study_uid: str):
    """One study's record, its series inline."""
    return require_study(study_uid)


@app.get("/studies/{study_uid}/report", tags=["studies"])
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


@app.get("/studies/{study_uid}/labels", tags=["studies"])
def get_labels(study_uid: str):
    """All twelve pilkwang labels: probability, confidence, YES/NO/UNK verdict."""
    labels = guard(catalog.get_labels, study_uid)
    if labels is None:
        raise HTTPException(404, f"No pilkwang labels for study: {study_uid}")
    return labels


@app.get("/studies/{study_uid}/labels/{label}", tags=["studies"])
def get_label(study_uid: str, label: str):
    """One label. Accepts the slug ("medial-meniscus", "bakers") or exact name."""
    if catalog.resolve_label(label) is None:
        raise HTTPException(404, f"Unknown label: {label}. Known: {sorted(catalog.SLUGS)}")
    result = guard(catalog.get_label, study_uid, label)
    if result is None:
        raise HTTPException(404, f"No pilkwang labels for study: {study_uid}")
    return result


@app.get("/studies/{study_uid}/predict", tags=["studies"])
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
        # An uploaded study has no cached tensor -- the cache was built once,
        # offline, for the corpus -- so it is scored straight off its DICOM.
        record = uploads.get(study_uid)
        if record is not None:
            # The doctor's report is the fusion model's fourth input. Read at
            # request time rather than at upload time, because it is written
            # after the upload -- that is the whole order of the workflow.
            stored = load_reports().get(study_uid) or {}
            predictions, used = predict_uploaded(record, model,
                                                 report_text=stored.get("report_en")
                                                 or stored.get("text"))
            if predictions is None:
                raise HTTPException(422, f"Could not build tensors for {study_uid}")
            return {"study_uid": study_uid, "model": used, "predictions": predictions}

        result = predict_study_with(study_uid, model)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    if result is None:
        raise HTTPException(404, f"{study_uid}: not cached for model '{model}' "
                                 "(needs the tensor cache; fusion also needs reports_en.csv)")
    return {"study_uid": study_uid, "model": model, "predictions": result}


@app.get("/studies/{study_uid}/series", tags=["studies"])
def list_series(study_uid: str,
                plane: str | None = Query(None, description="Sagittal | Coronal | Axial")):
    """Every sequence of a study."""
    if plane is not None and plane not in AXIS_PLANE.values():
        raise HTTPException(400, f"plane must be one of {sorted(AXIS_PLANE.values())}")
    study = require_study(study_uid)
    # An uploaded study carries its series in its own record; the catalog has
    # never heard of it.
    if study.get("source") == "upload":
        rows = [s for s in study["series"] if plane is None or s["plane"] == plane]
    else:
        rows = guard(catalog.list_series, study_uid, plane)
    return {"study_uid": study_uid, "count": len(rows), "series": rows}


@app.get("/studies/{study_uid}/series/{series_uid}", tags=["studies"])
def get_series(study_uid: str, series_uid: str):
    """One sequence: plane, acquisition flags, slice count, whether files exist."""
    record = guard(catalog.get_series, study_uid, series_uid)
    if record is None:
        raise HTTPException(404, f"Series {series_uid} not found in study {study_uid}")
    return record


@app.get("/studies/{study_uid}/series/{series_uid}/tensor", tags=["studies"])
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


@app.get("/studies/{study_uid}/series/{series_uid}/preview.png", tags=["studies"])
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


@app.get("/studies/{study_uid}/series/{series_uid}/instances", tags=["studies"])
def list_instances(study_uid: str, series_uid: str):
    """File names of the raw DICOM slices, in acquisition order.

    The order is the one sorted_slice_paths computes -- ImagePositionPatient
    projected onto the scan axis -- so it is the same order the model is fed.
    That matters more here than anywhere else: a browser-side viewer reads
    geometry from only the first, middle and last instance and trusts the rest
    of the order as given, so this route is what decides whether a volume comes
    out the right way round.
    """
    directory = series_path(study_uid, series_uid)
    if directory is None:
        raise HTTPException(404, f"No DICOM files for series {series_uid}")

    names = [path.name for path in sorted_slice_paths(directory)]
    return {"study_uid": study_uid, "series_uid": series_uid,
            "count": len(names), "instances": names}


@app.get("/studies/{study_uid}/series/{series_uid}/instances/{instance}",
         tags=["studies"], responses={200: {"content": {"application/dicom": {}}}})
def get_instance(study_uid: str, series_uid: str, instance: str):
    """One slice as the raw DICOM file, straight off disk.

    Everything else under /studies serves cooked pixels -- the (24, 224, 224, 1)
    tensor, the contact sheet PNG -- which is right for the model and wrong for
    a viewer: it has been sampled to 24 slices, resized to 224 and window/levelled
    already, and it carries no geometry. This serves the bytes themselves, so a
    client-side DICOM reader gets full resolution, real millimetres and its own
    windowing.
    """
    directory = series_path(study_uid, series_uid)
    if directory is None:
        raise HTTPException(404, f"No DICOM files for series {series_uid}")

    # A name arriving from the network must not walk out of the series folder;
    # .name drops any directory part before the path is ever joined.
    path = (directory / Path(instance).name).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise HTTPException(404, f"No such instance: {instance}")

    return Response(
        path.read_bytes(),
        media_type="application/dicom",
        # A stored slice never changes, and a viewer asks for every one of them
        # on each visit, so let the browser keep them.
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@app.post("/upload/{study_uid}/image_sequence", tags=["upload"])
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


def predict_uploaded(record, model="multiplane", report_text=None):
    """(predictions, model actually used) for an uploaded study.

    Substitutions happen here, and the caller is told about them through the
    returned name rather than being left to assume:

    - fusion needs a report, and a freshly uploaded study has none by design --
      writing it is the whole point of the doctor's next step. Until then the
      images are all there is, which is exactly what multiplane scores. Once a
      report is stored, `report_text` carries it and fusion runs for real.
    - multiplane needs all three planes. A folder holding only a sagittal run
      falls back to the single-plane model instead of failing.

    report_text must be ENGLISH: it goes through the report model's vectorizer,
    which was fitted on the translation cache. The stored report already carries
    it -- POST /create translates on the way in -- so the caller passes
    `report_en`, not `text`.
    """
    from functions.predict import predict_from_dirs

    # One series per axis, fluid-sensitive first: most of the twelve findings
    # are fluid, and several are invisible on a structural sequence. Same rule
    # sequence_to_tensor.ranked_series applies to the corpus.
    directories = {}
    for entry in sorted(record["series"], key=lambda s: not s.get("fluid_sensitive")):
        axis = entry.get("axis")
        if axis and axis not in directories:
            found = series_path(record["study_uid"], entry["series_uid"])
            if found is not None:
                directories[axis] = found

    text = str(report_text or "").strip()
    wanted = model if (model != "fusion" or text) else "multiplane"
    # Fusion degrades to the images alone before it degrades to one plane.
    fallbacks = ["multiplane", "sagittal"] if wanted == "fusion" else ["sagittal"]
    for name in dict.fromkeys([wanted, *fallbacks]):
        try:
            result = predict_from_dirs(directories, name, report_text=text or None)
        except FileNotFoundError:
            continue                     # that checkpoint is not on this box
        if result is not None:
            return result, name
    return None, None


@app.post("/upload/study", tags=["upload"])
async def upload_study(
    files: list[UploadFile] = File(..., description="Every .dcm of one study."),
    model: str = Query("multiplane", pattern="^(sagittal|multiplane)$",
                       description="Which model pre-fills the labels."),
):
    """Store a new study from a folder of DICOM, and pre-fill its labels.

    What comes back is a real study: it appears in GET /studies, opens in the
    viewer, and can be scored again later. What it deliberately does NOT get:

    - a report. That is the doctor's, written from these images through
      POST /create/{study_uid}/sequence_report.
    - golden labels. The 58 hand-labelled studies are the only ground truth in
      this project and nothing uploaded joins them. The twelve numbers here come
      from a model and are filed under `predicted_labels`, with the name of the
      model beside them.

    The UIDs are generated the way the corpus's own were -- pydicom's
    generate_uid() -- so an uploaded study is indistinguishable in form from a
    dataset one and cannot collide with it. Series are split out by the
    SeriesInstanceUID in the headers, so one dropped folder becomes the sagittal,
    coronal and axial runs it actually contains.
    """
    if not files:
        raise HTTPException(400, "No files uploaded.")

    study_uid = uploads.new_study_uid()
    destination = uploads.study_dir(study_uid)

    with tempfile.TemporaryDirectory() as staging:
        staged = []
        for upload in files:
            # Only the extension is taken from the client; the stored name is
            # ours. A name off the network never reaches the filesystem.
            name = Path(upload.filename or "").name
            if not name.lower().endswith(".dcm"):
                continue
            path = Path(staging) / f"{len(staged):05d}.dcm"
            with open(path, "wb") as handle:
                shutil.copyfileobj(upload.file, handle)
            staged.append(path)

        if not staged:
            raise HTTPException(400, "No .dcm files in the upload.")

        groups = uploads.group_by_series(staged)
        if not groups:
            raise HTTPException(422, "None of the uploaded files could be read as DICOM.")

        records = []
        for paths in groups.values():
            series_uid = uploads.new_series_uid()
            folder = destination / series_uid
            folder.mkdir(parents=True, exist_ok=True)
            for position, source in enumerate(paths):
                shutil.copyfile(source, folder / f"{position:05d}.dcm")

            import pydicom
            head = pydicom.dcmread(str(paths[0]), stop_before_pixels=True)
            records.append(uploads.series_record(study_uid, series_uid, head, len(paths)))

    record = uploads.build_record(study_uid, records)
    # Written before scoring, so a model that is missing or slow leaves a study
    # that is complete and browsable rather than a half-created one.
    uploads.write_record(study_uid, record)

    predictions, used = predict_uploaded(record, model)
    record = uploads.build_record(study_uid, records, labels=predictions, label_model=used)
    uploads.write_record(study_uid, record)

    return {**record, "n_files": sum(len(p) for p in groups.values()),
            "skipped": len(files) - sum(len(p) for p in groups.values())}


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def resolve_series(study_uid: str, series_uid: str | None, plane: str | None):
    """Pick which series to render: the one asked for, or the first of a plane,
    or just the study's first available one."""
    record = uploads.get(study_uid)
    candidates = (record["series"] if plane is None
                  else [s for s in record["series"] if s["plane"] == plane]) \
        if record is not None else guard(catalog.list_series, study_uid, plane)
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


@app.get("/view/{study_uid}/3d_image_sequence", tags=["views"])
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


@app.get("/view/{study_uid}/2d_image_sequence", tags=["views"],
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


@app.get("/view/{study_uid}/information", tags=["views"])
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

@app.get("/predict/{study_uid}", response_model=Prediction, tags=["predict"],
         deprecated=True)
def predict(
    study_uid: str,
    axis: str = Query("X", pattern="^[XYZ]$",
                      description="X=Sagittal, Y=Coronal, Z=Axial."),
):
    """DEPRECATED -- use GET /studies/{study_uid}/predict?model=sagittal.

    Kept for the front end's transition. It builds its tensor from raw DICOM
    on the server, so it only works where train_series/ (or an upload) exists
    -- NOT on Cloud Run, whose bucket deliberately carries no raw DICOM.

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


@app.get("/predict/report/terms", tags=["predict"])
def report_terms():
    """The medical dictionary POST /predict/report counts as features.

    The same list the report branch of the fusion model sees, served so a
    reader can be shown which of these terms a report actually uses. An empty
    list is a real answer, not an error: the dictionary is data, and a
    deployment without the file simply scores reports on TF-IDF alone.
    """
    from models.report_model import medical_terms
    terms = [t.strip() for t in medical_terms()]
    return {"count": len(terms), "terms": terms}


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

@app.post("/create/{study_uid}/sequence_report", tags=["reports"],
          response_model=StoredReport, status_code=201)
def create_report(study_uid: str, body: ReportIn):
    """Write a new report. 409 if one already exists -- overwriting is PUT's
    job, and silently clobbering on POST is how people lose work.

    The stored record carries the English rendering beside the text as written
    (`report_en`, `language`): everything downstream of a report -- the report
    model behind POST /predict/report, the report branch of the fusion model --
    was trained on English, and this is the one moment the text changes.
    """
    require_study(study_uid)
    reports = load_reports()
    if study_uid in reports:
        raise HTTPException(409, f"A report for {study_uid} already exists; use PUT.")

    timestamp = now()
    english, language = to_english(body.text)
    record = {"study_uid": study_uid, "text": body.text, "author": body.author,
              "created_at": timestamp, "updated_at": timestamp,
              "language": language, "report_en": english}
    reports[study_uid] = record
    save_reports(reports)
    return record


@app.put("/update/{study_uid}/sequence_report", tags=["reports"],
         response_model=StoredReport)
def update_report(study_uid: str, body: ReportIn):
    """Replace an existing report. 404 if there is nothing to replace.

    Re-translates: the English is a rendering of the text, so leaving a stale
    one behind would score the doctor's previous draft.
    """
    reports = load_reports()
    existing = reports.get(study_uid)
    if existing is None:
        raise HTTPException(404, f"No stored report for {study_uid}; use POST.")

    english, language = to_english(body.text)
    existing.update(text=body.text, author=body.author, updated_at=now(),
                    language=language, report_en=english)
    save_reports(reports)
    return existing


@app.delete("/delete/{study_uid}/sequence_report", tags=["reports"])
def delete_report(study_uid: str):
    reports = load_reports()
    if reports.pop(study_uid, None) is None:
        raise HTTPException(404, f"No stored report for {study_uid}.")
    save_reports(reports)
    return {"deleted": study_uid}

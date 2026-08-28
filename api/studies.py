"""
Read-only endpoints over the dataset. Thin: every lookup lives in
functions.catalog, so this file only maps HTTP to that.

Route order matters -- /studies/golden is declared before /studies/{study_uid},
otherwise FastAPI matches the literal "golden" as a study UID and returns 404.
"""

import io

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Response

from functions import catalog
from functions.catalog import DataUnavailable
from functions.model import LABELS
from functions.sequence_to_tensor import AXIS_PLANE, sequence_to_tensor

router = APIRouter(prefix="/studies", tags=["dataset"])


def _guard(fn, *args, **kwargs):
    """Turn a missing dataset into 503 rather than a 500 traceback."""
    try:
        return fn(*args, **kwargs)
    except DataUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/golden")
def list_golden():
    """The 58 hand-labelled studies -- the only ground truth in the dataset."""
    rows = _guard(catalog.golden_studies)
    return {"count": len(rows), "labels": LABELS, "studies": rows}


@router.get("/{study_uid}")
def get_study(study_uid: str):
    study = _guard(catalog.get_study, study_uid)
    if study is None:
        raise HTTPException(404, f"Unknown study: {study_uid}")
    return study


@router.get("/{study_uid}/report")
def get_report(study_uid: str):
    """The radiology report, as the dataset ships it.

    Mixed language by design: about half the corpus is already English, the
    rest is Spanish, Greek, Bulgarian, Turkish, Croatian, Dutch, German or
    French. Nothing translates them, and no translated copy exists in the
    dataset -- see functions.catalog.get_report.
    """
    report = _guard(catalog.get_report, study_uid)
    if report is None:
        raise HTTPException(404, f"No report for study: {study_uid}")
    return report


@router.get("/{study_uid}/labels")
def get_labels(study_uid: str):
    """All twelve pilkwang labels: probability, confidence, YES/NO/UNK verdict."""
    labels = _guard(catalog.get_labels, study_uid)
    if labels is None:
        raise HTTPException(404, f"No pilkwang labels for study: {study_uid}")
    return labels


@router.get("/{study_uid}/labels/{label}")
def get_label(study_uid: str, label: str):
    """One label. Accepts the slug ("medial-meniscus", "bakers") or exact name."""
    if catalog.resolve_label(label) is None:
        raise HTTPException(404, f"Unknown label: {label}. Known: {sorted(catalog.SLUGS)}")
    result = _guard(catalog.get_label, study_uid, label)
    if result is None:
        raise HTTPException(404, f"No pilkwang labels for study: {study_uid}")
    return result


@router.get("/{study_uid}/series")
def list_series(study_uid: str,
                plane: str | None = Query(None, description="Sagittal | Coronal | Axial")):
    """Every sequence of a study."""
    if plane is not None and plane not in AXIS_PLANE.values():
        raise HTTPException(400, f"plane must be one of {sorted(AXIS_PLANE.values())}")
    if _guard(catalog.get_study, study_uid) is None:
        raise HTTPException(404, f"Unknown study: {study_uid}")
    rows = _guard(catalog.list_series, study_uid, plane)
    return {"study_uid": study_uid, "count": len(rows), "series": rows}


@router.get("/{study_uid}/series/{series_uid}")
def get_series(study_uid: str, series_uid: str):
    """One sequence: plane, acquisition flags, slice count, whether files exist."""
    record = _guard(catalog.get_series, study_uid, series_uid)
    if record is None:
        raise HTTPException(404, f"Series {series_uid} not found in study {study_uid}")
    return record


@router.get("/{study_uid}/series/{series_uid}/tensor")
def get_tensor(study_uid: str, series_uid: str):
    """The model-ready tensor as a .npy download: (24, 224, 224, 1) float32.

    Same function the model is served by, so what you download here is exactly
    what the network sees.
    """
    directory = catalog.series_dir(study_uid, series_uid)
    if not directory.is_dir():
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


@router.get("/{study_uid}/series/{series_uid}/preview.png")
def get_preview(study_uid: str, series_uid: str,
                columns: int = Query(6, ge=1, le=24, description="montage width")):
    """A PNG contact sheet of the 24 sampled slices, for eyeballing a series."""
    from PIL import Image

    directory = catalog.series_dir(study_uid, series_uid)
    if not directory.is_dir():
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

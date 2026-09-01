"""Studies uploaded through the API, as opposed to the dataset ones.

catalog.py is the READ-ONLY seam over the corpus: train.csv, train_series.csv
and the pilkwang labels. This is the writable one. The two are deliberately
separate, because a study a doctor uploads is not a corpus row:

  - train.csv is what the models train on and what the benchmark scores
    against, so appending to it would move the ground truth under the
    experiments. It also lives on a read-only mount by design.
  - an uploaded study has no report until someone writes one, and its twelve
    labels come from a model rather than from a radiologist. Mixing those in
    with the 58 hand-labelled studies would destroy the only ground truth in
    the project.

LAYOUT

    {UPLOAD_ROOT}/{study_uid}/study.json        the record
    {UPLOAD_ROOT}/{study_uid}/{series_uid}/*.dcm

One JSON per study rather than a single index file: uploads land on a bucket
mount shared by several Cloud Run instances, and a shared index would need a
read-modify-write from all of them at once. A per-study file is only ever
written by the request that creates it.

UIDS

Built on pydicom's generate_uid(), which is what the corpus itself used: every
one of the 4,407 study UIDs is PYDICOM_ROOT_UID + 38 digits, 64 characters, and
that width never varies. generate_uid() on its own misses it about one draw in
ten, so _uid() below redraws -- see its docstring. New studies are then
indistinguishable in form from dataset ones, and cannot collide with them.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import generate_uid

from functions.sequence_to_tensor import AXIS_PLANE

# Mirrors serving/my_api.py: the container's writable mount, or data/_tmp when
# running from a checkout.
WRITE_ROOT = Path(os.environ.get("WRITE_ROOT",
                                 Path(__file__).resolve().parent.parent / "data" / "_tmp"))
UPLOAD_ROOT = WRITE_ROOT / "uploads"

RECORD_NAME = "study.json"

# Which plane each axis letter means, inverted from sequence_to_tensor's map so
# a record can carry both the way the dataset spells it and the way the models
# index it.
PLANE_AXIS = {plane: axis for axis, plane in AXIS_PLANE.items()}

# SeriesDescription substrings that mark a sequence fluid-sensitive or
# fat-suppressed. The corpus ships these as columns; an upload has only what the
# scanner wrote in the header, so this is a heuristic and is recorded as one.
# It matters because a fluid-sensitive series is the one worth scoring: most of
# the twelve findings are fluid, and several are invisible without it.
FLUID_MARKERS = ("t2", "pd", "stir", "dp", "densidad")
FATSAT_MARKERS = ("fs", "fatsat", "fat_sat", "fat sat", "spair", "spir", "stir", "sat")


# Every one of the corpus's 4,407 study UIDs is exactly this long: the pydicom
# root plus 38 digits. Sampled at four points across the bucket, the width never
# varies.
UID_LENGTH = 64


def _uid() -> str:
    """A UID of the exact shape the bucket uses.

    generate_uid() alone is not enough. It truncates the decimal expansion of a
    hash to fill out 64 characters, and that decimal is occasionally a digit or
    two short -- roughly one draw in ten comes back 62 or 63 characters. The
    corpus contains no such UID, so a short one would be the one visible mark
    that a study did not come from the dataset. Draw again instead.
    """
    while True:
        uid = str(generate_uid())
        if len(uid) == UID_LENGTH:
            return uid


def new_study_uid() -> str:
    """A study UID in the corpus's own form."""
    return _uid()


def new_series_uid() -> str:
    return _uid()


def study_dir(study_uid: str) -> Path:
    return UPLOAD_ROOT / study_uid


def plane_of(ds) -> str:
    """Sagittal | Coronal | Axial for one slice.

    Uses the dominant axis of cross(row, column) -- the scan axis, the same
    vector sorted_slice_paths orders a series along -- rather than matching the
    orientation cosines against the three exact standard planes. Real knee MRI
    is rarely axis-aligned (a gold study in this corpus reads
    [-0.07, 0.988, 0.139, -0.109, 0.131, -0.985]), and dicom_utils'
    get_slice_view_from_dataset answers "Unknown/Oblique" for those, which is not
    one of the three values Anatomical_Plane is allowed to take.

    In DICOM's LPS frame the normal points along x for sagittal, y for coronal
    and z for axial, so the largest component names the plane.
    """
    orientation = getattr(ds, "ImageOrientationPatient", None)
    if orientation is None or len(orientation) != 6:
        return "Sagittal"          # the corpus's most common plane
    cosines = np.asarray(orientation, dtype=float)
    normal = np.cross(cosines[:3], cosines[3:])
    return ("Sagittal", "Coronal", "Axial")[int(np.argmax(np.abs(normal)))]


def _flagged(description: str, markers) -> bool:
    text = description.lower()
    return any(marker in text for marker in markers)


def describe_sequence(ds) -> dict:
    """The two acquisition flags train_series.csv carries, guessed from the
    series description. Absent a description both come back False, which makes
    the series merely un-preferred rather than unusable."""
    description = str(getattr(ds, "SeriesDescription", "") or "")
    return {
        "fluid_sensitive": _flagged(description, FLUID_MARKERS),
        "fat_suppression": _flagged(description, FATSAT_MARKERS),
        "description": description,
    }


def group_by_series(paths):
    """[(name, bytes-on-disk path)] -> {SeriesInstanceUID: [paths]}.

    Headers only: grouping a 300-slice study should not decode 300 images.
    Files without a SeriesInstanceUID land under one "unknown" key rather than
    being dropped, so a folder from a scanner that omits it still uploads as a
    single series.
    """
    groups = {}
    for path in sorted(paths):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            continue                          # not DICOM; the caller counts it
        uid = str(getattr(ds, "SeriesInstanceUID", "") or "unknown")
        groups.setdefault(uid, []).append(path)
    return groups


def write_record(study_uid: str, record: dict) -> dict:
    directory = study_dir(study_uid)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / RECORD_NAME).write_text(json.dumps(record, indent=2))
    return record


def build_record(study_uid: str, series: list, labels=None, label_model=None) -> dict:
    """The study record, shaped like catalog.get_study so the API can serve
    either from one code path."""
    planes = {}
    for entry in series:
        planes[entry["plane"]] = planes.get(entry["plane"], 0) + 1
    return {
        "study_uid": study_uid,
        "n_series": len(series),
        "planes": planes,
        # Never golden: the 58 hand-labelled studies are the only ground truth
        # and nothing uploaded can join them.
        "is_golden": False,
        "has_report": False,
        "golden_labels": None,
        "series": series,
        "source": "upload",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Model prefill, not ground truth. Kept under its own key, with the
        # model that produced it, so nothing downstream can mistake it for a
        # radiologist's read.
        "predicted_labels": labels,
        "label_model": label_model,
    }


def get(study_uid: str):
    """One uploaded study's record, or None."""
    path = study_dir(study_uid) / RECORD_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def list_all() -> list:
    """Every uploaded study, newest first. Empty when nothing was ever
    uploaded, or when WRITE_ROOT is not mounted -- an upload store that is not
    there is not an error, it just means no uploads."""
    if not UPLOAD_ROOT.is_dir():
        return []
    records = []
    for directory in UPLOAD_ROOT.iterdir():
        if not directory.is_dir():
            continue
        record = get(directory.name)
        if record is not None:
            records.append(record)
    records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return records


def series_record(study_uid: str, series_uid: str, ds, n_slices: int) -> dict:
    """One row of the study's series list, in catalog's own shape."""
    plane = plane_of(ds)
    flags = describe_sequence(ds)
    return {
        "study_uid": study_uid,
        "series_uid": series_uid,
        "plane": plane,
        "axis": PLANE_AXIS.get(plane),
        "fluid_sensitive": flags["fluid_sensitive"],
        "fat_suppression": flags["fat_suppression"],
        "description": flags["description"],
        "n_slices": n_slices,
        "available": True,
    }

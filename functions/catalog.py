"""
Read-only access to the dataset: studies, series, reports, labels.

Separate from predict.py on purpose. predict.py is the model seam; this is the
data seam. The API is a thin layer over both, so a front end never touches a
CSV or a DICOM path directly.

Everything loads lazily and tolerates a missing data/ directory: the API must
still boot on Cloud Run, where the 17 GB dataset is not mounted. Endpoints that
need absent data return 503 rather than crashing at startup.

Sources
-------
data/train.csv         one row per study: the Report, plus the 12 label columns
                       which are filled for the 58 GOLDEN studies and NaN
                       everywhere else
data/train_series.csv  one row per series: plane, fluid-sensitivity, fat-sat
data/meta/rsna-knee-llm-labels-pilkwang/report_labels_v2.csv
                       pilkwang's LLM read of the reports: per finding a
                       probability, a __conf and a YES/NO/UNK __verdict
"""

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

from functions.labels import LABELS
from functions.sequence_to_tensor import AXIS_PLANE

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO_ROOT / "data"))

PILKWANG_CSV = "meta/rsna-knee-llm-labels-pilkwang/report_labels_v2.csv"

# Label names carry spaces and an apostrophe ("Baker's"), which make ugly URLs.
# Endpoints accept either the slug or the exact name.
SLUGS = {label.lower().replace(" ", "-").replace("'", ""): label for label in LABELS}


class DataUnavailable(RuntimeError):
    """Raised when a source file is not mounted. The API turns this into a 503."""


def _read(relative_path):
    path = DATA_ROOT / relative_path
    if not path.exists():
        raise DataUnavailable(
            f"{relative_path} not found under {DATA_ROOT}. "
            "Mount the dataset or set DATA_ROOT."
        )
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def studies():
    """One row per study: Report + the 12 golden label columns (mostly NaN)."""
    return _read("train.csv").set_index("StudyInstanceUID")


@lru_cache(maxsize=1)
def series():
    """One row per series: plane, fluid-sensitivity, fat-suppression."""
    return _read("train_series.csv")


@lru_cache(maxsize=1)
def pilkwang():
    """The LLM read of the reports, one row per study."""
    return _read(PILKWANG_CSV).set_index("StudyInstanceUID")


def resolve_label(name):
    """Slug or exact name -> the exact label. None if unknown."""
    if name in LABELS:
        return name
    return SLUGS.get(name.lower().replace(" ", "-").replace("'", ""))


def series_dir(study_uid, series_uid, split="train"):
    return DATA_ROOT / f"{split}_series" / study_uid / series_uid


def get_study(study_uid):
    """Study-level summary: how many series, which planes, is it golden."""
    df = studies()
    if study_uid not in df.index:
        return None
    row = df.loc[study_uid]
    own = series()[series().StudyInstanceUID == study_uid]
    golden_labels = {label: row[label] for label in LABELS if pd.notna(row[label])}
    return {
        "study_uid": study_uid,
        "n_series": int(len(own)),
        "planes": own.Anatomical_Plane.value_counts().to_dict(),
        "is_golden": len(golden_labels) == len(LABELS),
        "has_report": bool(pd.notna(row.Report)),
        "golden_labels": {k: int(v) for k, v in golden_labels.items()} or None,
    }


def _series_record(row, split="train"):
    directory = series_dir(row.StudyInstanceUID, row.SeriesInstanceUID, split)
    return {
        "study_uid": row.StudyInstanceUID,
        "series_uid": row.SeriesInstanceUID,
        "plane": row.Anatomical_Plane,
        "axis": next((a for a, p in AXIS_PLANE.items() if p == row.Anatomical_Plane), None),
        "fluid_sensitive": bool(row.Fluid_Sensitive),
        "fat_suppression": bool(row.Fat_Suppression),
        "n_slices": len(list(directory.glob("*.dcm"))) if directory.is_dir() else 0,
        "available": directory.is_dir(),
    }


def list_series(study_uid, plane=None):
    """Every sequence of a study, optionally filtered to one plane."""
    own = series()[series().StudyInstanceUID == study_uid]
    if plane is not None:
        own = own[own.Anatomical_Plane == plane]
    return [_series_record(row) for row in own.itertuples(index=False)]


def get_series(study_uid, series_uid):
    """One sequence. None if that series does not belong to that study."""
    own = series()[(series().StudyInstanceUID == study_uid)
                   & (series().SeriesInstanceUID == series_uid)]
    if own.empty:
        return None
    return _series_record(next(own.itertuples(index=False)))


@lru_cache(maxsize=1)
def _translations():
    """The English translation cache, built once by functions.report_translation
    for the report model. None when the file has not been built locally."""
    path = DATA_ROOT / "meta" / "reports_en.csv"
    if not path.exists():
        return None
    return pd.read_csv(path).set_index("StudyInstanceUID")


def get_report(study_uid, lang="original"):
    """The radiology report.

    The corpus is MIXED LANGUAGE (half English; the rest es/tr/hr/bg/el/nl/de).
    lang="original" (default) returns it exactly as the dataset ships it;
    lang="en" serves the machine translation from data/meta/reports_en.csv --
    the same cache the report model trains on -- and returns None when that
    file is absent or the study is not in it.
    """
    df = studies()
    if study_uid not in df.index:
        return None

    if lang == "en":
        translations = _translations()
        if translations is None or study_uid not in translations.index:
            return None
        row = translations.loc[study_uid]
        return {"study_uid": study_uid, "language": "en",
                "source_language": str(row.language), "report": str(row.report_en)}

    report = df.loc[study_uid, "Report"]
    if pd.isna(report):
        return None
    return {"study_uid": study_uid, "language": "as_written", "report": str(report)}


def get_labels(study_uid):
    """All twelve pilkwang labels for a study: probability, confidence, verdict."""
    df = pilkwang()
    if study_uid not in df.index:
        return None
    row = df.loc[study_uid]
    return {
        "study_uid": study_uid,
        "source": "pilkwang/report_labels_v2",
        "labels": {
            label: {
                "probability": float(row[label]),
                "confidence": float(row[f"{label}__conf"]),
                "verdict": str(row[f"{label}__verdict"]),   # YES | NO | UNK
            }
            for label in LABELS
        },
    }


def get_label(study_uid, label_name):
    """One pilkwang label. Returns None for an unknown study OR unknown label."""
    label = resolve_label(label_name)
    if label is None:
        return None
    everything = get_labels(study_uid)
    if everything is None:
        return None
    return {"study_uid": study_uid, "source": everything["source"],
            "label": label, **everything["labels"][label]}


def golden_studies():
    """The 58 studies Kaggle labelled by hand -- every one of the 12 filled in.

    These are the only ground truth in the dataset. Everything else is an LLM
    read of a report, so this is the set to evaluate against.
    """
    df = studies()
    complete = df[df[LABELS].notna().all(axis=1)]
    return [
        {"study_uid": uid,
         "labels": {label: int(row[label]) for label in LABELS}}
        for uid, row in complete.iterrows()
    ]

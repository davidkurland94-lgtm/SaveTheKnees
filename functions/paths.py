"""Default filesystem locations, resolved from the repo root.

WHEN THIS IS USED
    Only by entry-point scripts (`models/train_model.py`), to supply DEFAULTS
    for their command-line arguments. Library modules never import this - they
    take paths as arguments so a test or a notebook can point them anywhere.

"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_ROOT", REPO_ROOT / "data"))

# --- point these at the real data when it lands -------------------------------
TRAIN_IMAGES = DATA / "train_series"                   # <study>/<series>/*.dcm
GOLD_IMAGES = DATA / "savetheknees_gold_images"
DERIVED_LABELS = DATA / "meta" / "derived_labels.csv"  # StudyInstanceUID + the 12 columns
# ------------------------------------------------------------------------------

TRAIN_CSV = DATA / "train.csv"
SERIES_CSV = DATA / "train_series.csv"
CACHE = DATA / "tensor_cache"                          # one .npy per study, written once

"""
The seam between the model and everything built around it.

The API, the container and the front end depend on THIS contract only:

    predict_series(model, series_dir) -> {label: probability}

and nothing about how the model was trained. That is deliberate. A predictor
loaded with untrained weights returns the same shape as a good one, so the
Docker/Cloud Run/front-end work can proceed in parallel with training instead
of queuing behind it. Swapping in real weights is a file copy.
"""

import os
from pathlib import Path

import keras
import numpy as np

from functions.model import LABELS, build_model
from functions.sequence_to_tensor import sequence_to_tensor

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = Path(os.environ.get(
    "MODEL_PATH", REPO_ROOT / "models" / "knee_resnet50.keras"))


def load_predictor(model_path=DEFAULT_MODEL_PATH):
    """Load a saved model, or fall back to an untrained one.

    Returns (model, status). status is "trained" when weights were loaded from
    disk and "untrained" when the fallback was used -- the API reports it on
    every response so nobody mistakes random output for a prediction.

    Loading a saved .keras file also skips the ImageNet weight download, which
    matters for a Cloud Run cold start.
    """
    model_path = Path(model_path)
    if model_path.exists():
        return keras.saving.load_model(model_path), "trained"
    return build_model(), "untrained"


def predict_tensor(model, x):
    """(k, size, size, 1) or a batch of them -> {label: probability}."""
    if x.ndim == 4:
        x = x[None, ...]
    probs = model.predict(x, verbose=0)[0]
    return {label: float(p) for label, p in zip(LABELS, probs)}


def predict_series(model, series_dir):
    """A folder of .dcm files -> {label: probability}, or None if unreadable."""
    x = sequence_to_tensor(series_dir)
    if x is None:
        return None
    return predict_tensor(model, x)


if __name__ == "__main__":
    import pandas as pd

    model, status = load_predictor()
    print(f"model: {status}")

    series_df = pd.read_csv(REPO_ROOT / "data" / "train_series.csv")
    row = series_df.iloc[0]
    d = REPO_ROOT / "data" / "train_series" / row.StudyInstanceUID / row.SeriesInstanceUID

    result = predict_series(model, d)
    if result is None:
        raise SystemExit(f"No readable .dcm files in {d}")
    for label, p in result.items():
        print(f"  {label:<18} {p:.3f}")

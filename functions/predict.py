"""
The seam between the model and everything built around it.

The API, the container and the front end depend on THIS contract only:

    predict_series(model, series_dir) -> {label: probability}

and nothing about how the model was trained. That is deliberate. A predictor
loaded with untrained weights returns the same shape as a good one, so the
Docker/Cloud Run/front-end work can proceed in parallel with training instead
of queuing behind it. Swapping in real weights is a file copy.

The model is whatever models/train_model.py checkpointed -- 3D or 2.5D, this
module reads the loaded model's input shape and hands it the layout it wants,
so a --as-channels training run needs no serving change.
"""

import os
from pathlib import Path

import keras

from functions.labels import LABELS
from functions.sequence_to_tensor import IMG_SIZE, K, sequence_to_tensor, slices_to_channels

REPO_ROOT = Path(__file__).resolve().parent.parent

# Where models/train_model.py --checkpoint should point for the API to pick the
# weights up. Overridable so a container can mount weights anywhere.
DEFAULT_MODEL_PATH = Path(os.environ.get(
    "MODEL_PATH", REPO_ROOT / "models" / "knee_findings.keras"))


def load_predictor(model_path=DEFAULT_MODEL_PATH):
    """Load the checkpoint, or fall back to an untrained model.

    Returns (model, status). status is "trained" when weights were loaded from
    disk and "untrained" when the fallback was used -- the API reports it on
    every response so nobody mistakes random output for a prediction.

    compile=False: serving only ever calls predict(), so the optimizer state
    and the custom training metrics are dead weight here -- skipping them makes
    loading faster and independent of the training code's metric classes.
    """
    model_path = Path(model_path)
    if model_path.exists():
        return keras.saving.load_model(model_path, compile=False), "trained"

    from models.architectures import build_model
    return build_model((K, IMG_SIZE, IMG_SIZE, 1)), "untrained"


def _wants_channels(model):
    """True when the loaded model is the 2.5D variant: input (H, W, K)."""
    return len(model.input_shape) == 4      # (None, H, W, K) vs (None, K, H, W, 1)


def predict_tensor(model, x):
    """One sample from sequence_to_tensor -> {label: probability}.

    Accepts the (K, H, W, 1) sequence layout and converts to (H, W, K) itself
    when the model is the 2.5D variant, so callers never need to know which
    architecture was trained.
    """
    if _wants_channels(model) and x.ndim == 4:
        x = slices_to_channels(x)
    probs = model.predict(x[None, ...], verbose=0)[0]
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
    layout = "2.5D (H, W, K)" if _wants_channels(model) else "3D (K, H, W, 1)"
    print(f"model: {status} | layout: {layout}")

    series_df = pd.read_csv(REPO_ROOT / "data" / "train_series.csv")
    row = series_df.iloc[0]
    d = REPO_ROOT / "data" / "train_series" / row.StudyInstanceUID / row.SeriesInstanceUID

    result = predict_series(model, d)
    if result is None:
        raise SystemExit(f"No readable .dcm files in {d}")
    for label, p in result.items():
        print(f"  {label:<18} {p:.3f}")

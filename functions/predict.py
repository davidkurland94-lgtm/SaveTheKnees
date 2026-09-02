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


# ---------------------------------------------------------------------------
# Notebook convenience -- the one-liner for models_pipe.ipynb
# ---------------------------------------------------------------------------

_loaded = None


def predict_study(study_uid, axis="X", data_root=None):
    """StudyInstanceUID -> {label: probability} via the checkpointed image
    model. Loads the model and the series table once, on first call.

    Serving-consistent by construction: picks the same series pick_series
    picks, builds the tensor with sequence_to_tensor.
    """
    global _loaded
    import pandas as pd
    from functions.sequence_to_tensor import study_to_tensor

    data_root = _data_root(data_root)
    if _loaded is None:
        model, status = load_predictor()
        series_df = pd.read_csv(data_root / "train_series.csv")
        _loaded = (model, status, series_df)
    model, status, series_df = _loaded

    x = study_to_tensor(study_uid, series_df, data_root=data_root, axis=axis)
    if x is None:
        return None
    return {"model_status": status, **predict_tensor(model, x)}


# ---------------------------------------------------------------------------
# Study-keyed prediction with ANY trained model -- the way to reach the
# multi-plane ensemble and the fusion model, whose inputs (3 volumes, plus the
# report for fusion) only exist server-side, in the tensor + translation caches.
# ---------------------------------------------------------------------------

def model_file(fname):
    """One checkpoint: the repo's models/ first, else the Cloud Run mount.

    The serving image ships only the checkpoints that are in the repo (the
    image diet keeps the rest out); everything else arrives through the bucket
    mounted at /mnt/models, overridable via MODELS_ROOT.

    Public, and the ONE way any serving path should name a checkpoint. It used
    to be private to this module, which is how models/report_model.py came to
    hardcode `REPO_ROOT / "models"` instead -- a path that does not exist in the
    container, so /predict/report answered 503 on every deployment while the
    weights sat in the mount beside the ones this function found.
    """
    local = REPO_ROOT / "models" / fname
    if local.exists():
        return local
    return Path(os.environ.get("MODELS_ROOT", "/mnt/models")) / fname


def _data_root(override=None):
    """The repo data/ locally, /mnt/data (env DATA_ROOT) on Cloud Run."""
    if override:
        return Path(override)
    return Path(os.environ.get("DATA_ROOT", REPO_ROOT / "data"))


STUDY_MODELS = {
    "sagittal":   ["knee_findings.keras"],
    "multiplane": ["knee_multiplane.keras", "knee_multiplane_s1.keras", "knee_multiplane_s2.keras"],
    "fusion":     ["knee_fusion_v2_s0.keras", "knee_fusion_v2_s1.keras"],
}
_study_models = {}


def _load_study_models(name):
    """Load (and cache) the checkpoints behind one STUDY_MODELS entry;
    returns [] when none is on disk so the caller can 503 honestly."""
    if name not in _study_models:
        models = []
        for fname in STUDY_MODELS[name]:
            path = model_file(fname)
            if path.exists():
                models.append(keras.saving.load_model(path, compile=False))
        _study_models[name] = models
    return _study_models[name]


def predict_study_with(study_uid, model="fusion", data_root=None, report_text=None):
    """StudyInstanceUID -> {label: probability} using the chosen trained model
    ("sagittal" | "multiplane" | "fusion"). Ensembles average their seeds.

    Reads the tensor cache (all three axes for multiplane/fusion) and, for
    fusion, the translated report through the report model's vectorizer.
    Returns None when the study is not cached for that model's inputs, or
    raises FileNotFoundError when the checkpoints are absent.

    report_text overrides the report the fusion model reads, and must be
    ENGLISH. Without it fusion falls back to reports_en.csv -- the translation
    cache built once, offline, over the corpus. That cache is a snapshot of what
    the study shipped with, so a report a doctor has since edited in the app
    would never reach the model: same study, same images, and an answer computed
    from text nobody can see any more. The caller passes the stored report when
    there is one.
    """
    import numpy as np
    import pandas as pd
    from functions.tensor_cache import cache_path, load_cached

    data_root = _data_root(data_root)
    cache_dir = data_root / "tensor_cache"
    models = _load_study_models(model)
    if not models:
        raise FileNotFoundError(f"no checkpoint for '{model}' under models/")

    axes = ("X",) if model == "sagittal" else ("X", "Y", "Z")
    if not all(cache_path(study_uid, cache_dir, a).exists() for a in axes):
        return None
    inputs = [load_cached(study_uid, cache_dir, a)[None, ...] for a in axes]

    if model == "fusion":
        import joblib
        text = report_text
        if text is None:
            texts = pd.read_csv(data_root / "meta" / "reports_en.csv").set_index("StudyInstanceUID")
            if study_uid not in texts.index:
                return None
            text = str(texts.loc[study_uid, "report_en"])
        vectorizer = joblib.load(model_file("report_vectorizer.joblib"))
        inputs.append(vectorizer.transform([text]).toarray().astype("float32"))

    feed = inputs if len(inputs) > 1 else inputs[0]
    probs = np.mean([m.predict(feed, verbose=0)[0] for m in models], axis=0)
    return {label: float(p) for label, p in zip(LABELS, probs)}


def predict_from_dirs(dirs_by_axis, model="multiplane", report_text=None):
    """{axis: series directory} -> {label: probability}, straight off the DICOM.

    predict_study_with reads the tensor cache, which only exists for the corpus:
    it was built once, offline, for the 4,407 dataset studies. A study uploaded
    a second ago is in no cache, so this builds its tensors on the spot with the
    same sequence_to_tensor the cache was made from -- same crop, same window,
    same resize -- and feeds them to the same checkpoints.

    axis is "X" sagittal, "Y" coronal, "Z" axial, as everywhere else.
    Returns None when a plane the model needs has no readable series, so the
    caller can fall back rather than serve numbers from a half-empty input.

    report_text is the fourth input the fusion model takes, and it is ENGLISH:
    the vectorizer was fitted on the translation cache, so a report in another
    language goes through functions.report_translation first. Fusion without a
    report returns None -- the missing branch is half the model, and zeroing it
    would serve a number that no training run ever produced.
    """
    import numpy as np

    models = _load_study_models(model)
    if not models:
        raise FileNotFoundError(f"no checkpoint for '{model}' under models/")

    axes = ("X",) if model == "sagittal" else ("X", "Y", "Z")
    if not all(axis in dirs_by_axis for axis in axes):
        return None
    if model == "fusion" and not str(report_text or "").strip():
        return None

    tensors = [sequence_to_tensor(dirs_by_axis[axis]) for axis in axes]
    if any(x is None for x in tensors):
        return None

    inputs = [x[None, ...] for x in tensors]
    if model == "fusion":
        # The report model's OWN fitted vectorizer, the same one
        # predict_study_with and models/train_fusion.py load, so the text
        # features here are identical to the ones the model trained on.
        import joblib
        vectorizer = joblib.load(model_file("report_vectorizer.joblib"))
        inputs.append(vectorizer.transform([str(report_text)]).toarray().astype("float32"))

    feed = inputs if len(inputs) > 1 else inputs[0]
    probs = np.mean([m.predict(feed, verbose=0)[0] for m in models], axis=0)
    return {label: float(p) for label, p in zip(LABELS, probs)}


if __name__ == "__main__":
    import pandas as pd

    model, status = load_predictor()
    layout = "2.5D (H, W, K)" if _wants_channels(model) else "3D (K, H, W, 1)"
    print(f"model: {status} | layout: {layout}")

    series_df = pd.read_csv(_data_root() / "train_series.csv")
    row = series_df.iloc[0]
    d = _data_root() / "train_series" / row.StudyInstanceUID / row.SeriesInstanceUID

    result = predict_series(model, d)
    if result is None:
        raise SystemExit(f"No readable .dcm files in {d}")
    for label, p in result.items():
        print(f"  {label:<18} {p:.3f}")

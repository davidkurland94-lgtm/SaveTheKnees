"""
FastAPI wrapper around functions.predict.

The model is loaded ONCE at startup, not per request. On Cloud Run a per-request
load would add seconds to every call and re-download ImageNet weights on every
cold start.

Run locally:
    uvicorn api.main:app --reload --port 8000
Then open http://localhost:8000/docs
"""

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from api.studies import router as studies_router
from functions.labels import LABELS
from functions.predict import load_predictor, predict_series

# Reading a server-side path is handy in local dev and a liability on a public
# URL, so it is off unless explicitly switched on.
ALLOW_LOCAL_PATHS = os.environ.get("ALLOW_LOCAL_PATHS", "0") == "1"

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"], state["status"] = load_predictor()


    import numpy as np
    state["model"].predict(
        np.zeros((1, *state["model"].input_shape[1:]), np.float32), verbose=0)

    yield
    state.clear()


app = FastAPI(
    title="Save the Knees",
    description="Knee MRI series in, twelve finding probabilities out.",
    version="0.1.0",
    lifespan=lifespan,
)

# Read-only dataset endpoints: studies, series, reports, pilkwang labels.
app.include_router(studies_router)


class Prediction(BaseModel):
    model_status: str          # "trained" | "untrained"
    n_slices_received: int
    predictions: dict[str, float]


@app.get("/")
def root():
    return {"service": "save-the-knees", "labels": LABELS, "docs": "/docs"}


@app.get("/health")
def health():
    """Cloud Run's readiness probe. Cheap on purpose -- no inference here."""
    return {"status": "ok", "model": state.get("status", "not_loaded")}


@app.post("/predict", response_model=Prediction)
async def predict(files: list[UploadFile] = File(...)):
    """Upload every .dcm of ONE series (one plane of one study).

    The whole series is needed, not a single slice: intensity is normalised
    with one percentile window per series, and 24 slices are sampled across it.
    """
    dicoms = [f for f in files if (f.filename or "").lower().endswith(".dcm")]
    if not dicoms:
        raise HTTPException(400, "No .dcm files in the upload.")

    tmp = Path(tempfile.mkdtemp(prefix="stk_"))
    try:
        for upload in dicoms:
            with open(tmp / Path(upload.filename).name, "wb") as out:
                shutil.copyfileobj(upload.file, out)

        result = predict_series(state["model"], tmp)
        if result is None:
            raise HTTPException(422, "Could not read a series from those files.")

        return Prediction(model_status=state["status"],
                          n_slices_received=len(dicoms),
                          predictions=result)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/predict/path", response_model=Prediction)
def predict_path(series_dir: str):
    """Predict from a directory on the server. Local development only."""
    if not ALLOW_LOCAL_PATHS:
        raise HTTPException(403, "Disabled. Set ALLOW_LOCAL_PATHS=1 for local use.")

    path = Path(series_dir)
    if not path.is_dir():
        raise HTTPException(404, f"Not a directory: {series_dir}")

    result = predict_series(state["model"], path)
    if result is None:
        raise HTTPException(422, f"No readable .dcm files in {series_dir}")

    return Prediction(model_status=state["status"],
                      n_slices_received=len(list(path.glob("*.dcm"))),
                      predictions=result)

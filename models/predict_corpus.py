import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from functions import paths
from functions.labels import LABELS
from functions.tensor_cache import cache_path, load_cached

DEFAULT_OUT = paths.DATA / "meta" / "image_model_all.csv"


def predict_corpus(uids=None, batch_size=16, axis="X", out_csv=DEFAULT_OUT,
                   model_path=None, verbose=True):
    """Every cached study through the trained image model -> DataFrame + CSV."""
    from functions.predict import DEFAULT_MODEL_PATH, load_predictor

    model, status = load_predictor(model_path or DEFAULT_MODEL_PATH)
    if status != "trained":
        raise SystemExit("no trained checkpoint -- train first, then predict the corpus")

    if uids is None:
        index = pd.read_csv(paths.DATA / "meta" / "train_index.csv")
        uids = sorted(index.StudyInstanceUID.unique())
    uids = [u for u in uids if cache_path(u, paths.CACHE, axis).exists()]
    if verbose:
        print(f"predicting {len(uids)} cached studies (batch {batch_size})")

    rows = []
    for start in range(0, len(uids), batch_size):
        chunk = uids[start:start + batch_size]
        x = np.stack([load_cached(u, paths.CACHE, axis) for u in chunk])
        probs = model.predict(x, verbose=0)
        rows.extend({"StudyInstanceUID": u,
                     **{l: float(p) for l, p in zip(LABELS, prob)}}
                    for u, prob in zip(chunk, probs))
        if verbose and (start // batch_size) % 25 == 0:
            print(f"  {start + len(chunk)}/{len(uids)}", flush=True)

    table = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    if verbose:
        print(f"{len(table)} predictions -> {out_csv}")
    return table


MP_CHECKPOINTS = ["models/knee_multiplane.keras",
                  "models/knee_multiplane_s1.keras",
                  "models/knee_multiplane_s2.keras"]


def predict_corpus_multiplane(checkpoints=None, batch_size=8,
                              out_csv=None, verbose=True):
    """The 3-seed multi-plane ensemble over every study cached on all 3 axes.

    This is the corpus counterpart of the image_multiplane reader in the gold
    table: the strongest pure-image judge, averaged over seeds, for the
    report-quality analysis. GPU recommended (3 volumes per study).
    """
    import keras
    from functions.datasets import make_multiplane_dataset
    from functions.labels import gold_labels

    out_csv = Path(out_csv) if out_csv else paths.DATA / "meta" / "image_ens_all.csv"
    checkpoints = checkpoints or MP_CHECKPOINTS

    index = pd.read_csv(paths.DATA / "meta" / "train_index.csv")
    uids = sorted(u for u in index.StudyInstanceUID.unique()
                  if all(cache_path(u, paths.CACHE, a).exists() for a in "XYZ"))
    if verbose:
        print(f"{len(uids)} studies cached on all 3 axes")

    dummy = np.zeros((len(uids), len(LABELS)), dtype="float32")
    ds = make_multiplane_dataset(uids, dummy, paths.CACHE, batch_size=batch_size)
    preds = []
    for ckpt in checkpoints:
        model = keras.saving.load_model(ckpt, compile=False)
        preds.append(model.predict(ds, verbose=1 if verbose else 0))
        if verbose:
            print(f"  {ckpt} done", flush=True)
    ens = np.mean(preds, axis=0)

    table = pd.DataFrame(ens, columns=LABELS)
    table.insert(0, "StudyInstanceUID", uids)
    table.to_csv(out_csv, index=False)
    if verbose:
        print(f"{len(table)} ensemble predictions -> {out_csv}")
    return table


def main(argv=None):
    p = argparse.ArgumentParser(description="Trained image model over the full corpus")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--axis", default="X", choices=["X", "Y", "Z"])
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--multi-plane", action="store_true",
                   help="3-seed multi-plane ensemble instead of the single-plane model")
    a = p.parse_args(argv)
    if a.multi_plane:
        predict_corpus_multiplane(batch_size=a.batch_size, out_csv=a.out)
    else:
        predict_corpus(batch_size=a.batch_size, axis=a.axis,
                       out_csv=a.out, model_path=a.model)


if __name__ == "__main__":
    main()

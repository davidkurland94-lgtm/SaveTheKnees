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


def main(argv=None):
    p = argparse.ArgumentParser(description="Trained image model over the full corpus")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--axis", default="X", choices=["X", "Y", "Z"])
    p.add_argument("--model", type=Path, default=None)
    a = p.parse_args(argv)
    predict_corpus(batch_size=a.batch_size, axis=a.axis,
                   out_csv=a.out, model_path=a.model)


if __name__ == "__main__":
    main()

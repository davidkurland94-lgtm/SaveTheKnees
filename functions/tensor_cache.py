"""Decode every study once, store it on disk, read it back fast.

PIPELINE POSITION
    Step 2 of 5. Runs after `labels.py` has produced a study list, and before
    `datasets.py` streams the results into TensorFlow.

        labels.py  ->  TENSOR_CACHE.PY  ->  datasets.py  ->  architectures.py
                                                          -> train_model.py

WHEN TO USE IT
    Once per dataset, before the first training run. `build_cache` is the slow
    step in the whole pipeline - roughly 15 minutes for ~4350 studies - and it is
    safe to re-run: anything already on disk is skipped, so an interrupted run
    resumes rather than starting over.

WHY THE CACHE EXISTS
    Reading one study means opening ~24 DICOM files, decoding the pixels,
    resizing and normalising them. That takes about 0.2 s, so ~4350 training
    studies take roughly 15 minutes. An epoch is one pass over the training
    data, and training takes many epochs. If the decoding happened inside the
    training loop, those 15 minutes would be paid again every epoch, every time
    redoing byte-for-byte identical work. So it is done once, here, and saved as
    one small .npy file per study. Training then reads those.

WHY uint8
    The cache stores whole numbers 0-255 rather than float32 decimals.
    `sequence_to_tensor` has already normalised every image to [0, 1], so storing
    0-255 loses about 1/255 of the precision and takes a quarter of the disk:
    roughly 5 GB instead of 21 GB. `load_cached` divides by 255 to get the
    decimals back.

THE ALIGNMENT CONTRACT
    `build_cache` skips studies it cannot read, so afterwards the label table has
    rows the cache has no image for. Always pipe one into the other:

        gold = cached_subset(gold, build_cache(gold, series_df, images_root, cache))

    Skipping `cached_subset` does not raise - study A just gets trained against
    study B's labels and the model quietly learns nonsense.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from functions.sequence_to_tensor import pick_series, sequence_to_tensor


def load_study(study_uid, series_df, images_root, axis="X", as_channels=False):
    """One StudyInstanceUID -> one tensor, or None if there is no usable series.

    `sequence_to_tensor` takes a SERIES directory and a study holds several, so
    `pick_series` chooses one (fluid-sensitive first). The module's own
    study_to_tensor hardcodes data_root/train_series/, which is not where the gold
    images live, so the two are composed directly instead.

    Returns:
        float32 array in [0, 1], (K, H, W, 1) or (H, W, K), or None.
    """
    series_uid = pick_series(study_uid, series_df, axis)
    if series_uid is None:
        return None

    series_dir = Path(images_root) / study_uid / series_uid
    if not series_dir.is_dir():
        return None

    return sequence_to_tensor(series_dir, as_channels=as_channels)


def cache_path(study_uid, cache_dir, axis="X"):
    """Where one study's cached tensor lives. The axis is in the filename so
    caches for different orientations cannot collide."""
    return Path(cache_dir) / f"{study_uid}_{axis}.npy"


def build_cache(index, series_df, images_root, cache_dir, axis="X", verbose=True):
    """Decode every study once, store it as uint8, return the UIDs that worked.

    Safe to re-run: a study already on disk is skipped, so an interrupted run
    resumes rather than starting over. Studies with no readable series are simply
    absent from the returned list, which is what keeps labels aligned later.

    Args:
        index: DataFrame with a StudyInstanceUID column (a label table).
        series_df: the `train_series.csv` table, for `pick_series`.
        images_root: directory holding <study>/<series>/*.dcm.
        cache_dir: where the .npy files go; created if absent.

    Returns:
        list of the StudyInstanceUIDs that are now in the cache, in index order.
        Feed it straight to `cached_subset`.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    ok = []
    for n, study_uid in enumerate(index.StudyInstanceUID, start=1):
        out = cache_path(study_uid, cache_dir, axis)

        if not out.exists():
            x = load_study(study_uid, series_df, images_root, axis)   # float32 in [0, 1]
            if x is None:
                continue
            np.save(out, (x * 255).round().astype("uint8"))

        ok.append(study_uid)
        if verbose and n % 250 == 0:
            print(f"cached {n}/{len(index)} studies")

    return ok


def cached_subset(index, uids):
    """Cut the label table down to the studies that actually made it into the cache.

    Why this is needed: build_cache skips studies it cannot read, so afterwards the
    label table has rows the cache has no image for. If the two ever drift apart,
    study A gets trained against study B's labels and nothing raises an error - the
    model just quietly learns nonsense.

    The merge rebuilds the table in cache order, so row i of the labels and study i
    of the cache are the same knee, by construction.
    """
    order = pd.DataFrame({"StudyInstanceUID": list(uids)})
    return order.merge(index, on="StudyInstanceUID", how="left").reset_index(drop=True)


def load_cached(study_uid, cache_dir, axis="X", as_channels=False):
    """One cached study back to the exact contract sequence_to_tensor produces:
    float32 in [0, 1], (24, 224, 224, 1) or (224, 224, 24).

    Called once per study per epoch, from inside the tf.data pipeline in
    `datasets.make_dataset`. Keep it cheap.
    """
    x = np.load(cache_path(study_uid, cache_dir, axis)).astype("float32") / 255.0
    return np.ascontiguousarray(np.transpose(x[..., 0], (1, 2, 0))) if as_channels else x

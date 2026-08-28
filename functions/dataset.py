"""
Training input: studies in, (tensor, twelve labels) out.

Reads the data/volumes/ mirror by default, NOT the DICOM files. Measured on
this repo, sequence_to_tensor takes ~0.45s per series against ~0.024s to load
the npz -- about 20x. Over 4,407 studies that is roughly 33 minutes of pure I/O
per epoch versus under two.

That speed is not free. The mirror and sequence_to_tensor DISAGREE: median
correlation 0.806 over 60 sagittal studies, and the slice direction is
inconsistent (32 studies run one way, 28 the other). load_study documents the
measurement and what it costs per model type, and takes source="dicom" to read
through sequence_to_tensor instead when consistency matters more than speed.

The 0.973-0.996 figure quoted in sequence_to_tensor.py's own docstring does not
reproduce as a per-study mirror-vs-function correlation; only series that
already hold exactly 24 slices land in that range (median 0.970), because there
is nothing to resample.

One sample is one STUDY, not one series, because the labels are per study. A
study holds 1-7 series per plane, so pick_series' rule is reproduced here:
fluid-sensitive wins, UID breaks ties. Using a different rule at training time
than at serving time is the kind of skew that never raises an error.

Labels are the twelve soft_* columns of data/meta/train_index.csv. Only 58
studies carry hand labels; the soft ones cover all 4,407, which is the only way
to train on more than a rounding error of the data.

Usage:
    train_ds, val_ds = make_datasets(val_fold=0, batch_size=4)
"""

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from functions.model import LABELS
from functions.sequence_to_tensor import AXIS_PLANE, IMG_SIZE, K, sequence_to_tensor

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO_ROOT / "data"))
VOLUMES = DATA_ROOT / "volumes"

SOFT = [f"soft_{label}" for label in LABELS]


@lru_cache(maxsize=1)
def index():
    """One row per series, with folds and the soft labels already attached."""
    return pd.read_csv(DATA_ROOT / "meta" / "train_index.csv")


def studies_for_fold(fold, exclude=False):
    """StudyInstanceUIDs in (or outside) one fold, deduplicated and sorted."""
    df = index().drop_duplicates("StudyInstanceUID")
    rows = df[df.fold != fold] if exclude else df[df.fold == fold]
    return sorted(rows.StudyInstanceUID)


@lru_cache(maxsize=4)
def chosen_series(axis="X"):
    """study_uid -> (npz filename, twelve soft labels), one row per study.

    Built ONCE per axis. Doing the filter-and-sort per sample would re-scan a
    24,371-row frame 4,407 times an epoch, which costs more than reading the
    pixels does.

    The selection rule is functions.sequence_to_tensor.pick_series, reproduced:
    right plane, fluid-sensitive wins, UID breaks ties. Choosing a different
    series at training time than at serving time is the kind of skew that never
    raises an error.
    """
    df = index()
    df = df[df.Anatomical_Plane == AXIS_PLANE[axis]]
    df = df.sort_values(["Fluid_Sensitive", "SeriesInstanceUID"],
                        ascending=[False, True])
    df = df.drop_duplicates("StudyInstanceUID", keep="first")

    labels = df[SOFT].to_numpy(dtype=np.float32)
    return {uid: (npz, labels[i])
            for i, (uid, npz) in enumerate(zip(df.StudyInstanceUID, df.npz))}


def load_study(study_uid, axis="X", source="mirror"):
    """study_uid -> ((K, IMG_SIZE, IMG_SIZE, 1) float32 [0,1], (12,) float32).

    source="mirror" reads data/volumes/*.npz  -- ~0.024s, but see below
    source="dicom"  runs sequence_to_tensor   -- ~0.45s, matches what the API serves

    MEASURED SKEW BETWEEN THE TWO (60 studies, sagittal):

        forward correlation          median 0.806   (2/60 above 0.95)
        allowing a reversed stack    median 0.901   (5/60 above 0.95)
        mirror matches forward       32 studies
        mirror matches reversed      28 studies

    Two separate disagreements. First, slice DIRECTION is inconsistent -- the
    mirror runs the same way as sequence_to_tensor in barely half the studies.
    Second, even at best alignment, series that do not already hold exactly 24
    slices reach only 0.888, so the two pick different subsets of slices
    (series that hold exactly 24 reach 0.970, with nothing to resample).

    What that means depends entirely on the model:

      - Pooling over slices (the current model.py: TimeDistributed then
        GlobalAveragePooling1D) is permutation-invariant, so direction does not
        matter and only the 0.888 pixel difference does. Tolerable.
      - Stacking slices as CHANNELS (the 2.5D layout) is not. Channel 5 would
        be a different anatomical slice at training time than at serving time,
        mirrored in roughly half of studies. Do NOT train 2.5D off the mirror.

    Returns None when the study has no series in that plane, or the volume is
    missing. Callers must filter: not every study is imaged in every plane.
    """
    entry = chosen_series(axis).get(study_uid)
    if entry is None:
        return None
    npz_name, y = entry
    series_uid = Path(str(npz_name)).stem

    if source == "dicom":
        x = sequence_to_tensor(DATA_ROOT / "train_series" / study_uid / series_uid)
        return None if x is None else (x, y)

    path = VOLUMES / f"{series_uid}.npz"
    if not path.exists():
        return None

    volume = np.load(path)["data"]                       # (K, IMG, IMG) uint8
    x = volume[..., None].astype(np.float32) / 255.0     # the mirror is uint8
    return x, y


def make_dataset(study_uids, batch_size=4, axis="X", shuffle=False, limit=None,
                 source="mirror"):
    """A tf.data pipeline over study UIDs. See load_study for source semantics."""
    uids = list(study_uids)[:limit] if limit else list(study_uids)

    def generator():
        for uid in uids:
            sample = load_study(uid, axis, source)
            if sample is not None:
                yield sample

    signature = (
        tf.TensorSpec(shape=(K, IMG_SIZE, IMG_SIZE, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(len(LABELS),), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(generator, output_signature=signature)
    if shuffle:
        # Studies are read in UID order, so without this every batch would be
        # neighbours in the same arbitrary ordering.
        ds = ds.shuffle(min(len(uids), 256), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_datasets(val_fold=0, batch_size=4, axis="X", limit=None, source="mirror"):
    """The usual split: one fold held out, the other four for training."""
    train = make_dataset(studies_for_fold(val_fold, exclude=True),
                         batch_size, axis, shuffle=True, limit=limit, source=source)
    val = make_dataset(studies_for_fold(val_fold),
                       batch_size, axis, shuffle=False, limit=limit, source=source)
    return train, val


if __name__ == "__main__":
    train_uids = studies_for_fold(0, exclude=True)
    val_uids = studies_for_fold(0)
    print(f"fold 0 held out: {len(train_uids)} train studies, {len(val_uids)} val")

    sample = load_study(train_uids[0])
    if sample is None:
        raise SystemExit(f"No sagittal volume for {train_uids[0]}")
    x, y = sample
    print(f"one sample: x {x.shape} {x.dtype} [{x.min():.2f}, {x.max():.2f}]  y {y.shape}")
    print("  labels:", ", ".join(f"{l}={v:.2f}" for l, v in zip(LABELS, y)))

    ds = make_dataset(train_uids, batch_size=4, limit=8)
    for xb, yb in ds.take(1):
        print(f"one batch : x {tuple(xb.shape)}  y {tuple(yb.shape)}")

"""
The whole point of this module is that a model needs the SAME shape for every
sample, while the data does not cooperate:

  - the number of images per series runs from 11 to 320 (median 30)
  - raw intensity has no absolute meaning at all (a 38x spread between series)
  - the physical field of view runs from 81 mm to 270 mm (a 3.3x spread)

The output contract never changes:

    (k, img_size, img_size, 1) float32, values in [0, 1]

The output matches the existing data/volumes/ mirror to a correlation of
0.973-0.996, so a model can train on the mirror and be served by this function
without a train/inference mismatch.

Typical use:

    from sequence_to_tensor import sequence_to_tensor, study_to_tensor

    x = sequence_to_tensor("data/train_series/<study_uid>/<series_uid>")
    x = study_to_tensor(study_uid, series_df, data_root="data", axis="X")
    # 2.5D: 24 slices as 24 channels
    x = sequence_to_tensor(series_dir, as_channels=True)      # (224, 224, 24)

Labels are not produced here. They live in data/meta/train_index.csv, one row
per series, in the twelve "soft_*" columns.
"""

from pathlib import Path

import numpy as np
import pydicom

K = 24            # images per sequence
IMG_SIZE = 224    # pixels per side
PCT = (1, 99.5)   # percentile window used for intensity normalisation

# Physical crop, in millimetres, or None for "resize the whole field of view".
#
#     no crop     ->  correlation 0.973 - 0.996
#     crop 130 mm ->  correlation 0.369 - 0.507

CROP_MM: float | None = None
AXIS_PLANE = {"X": "Sagittal", "Y": "Coronal", "Z": "Axial"}


def sorted_slice_paths(series_dir):
    """
    Only headers are read here so ordering a 320-image
    series is cheap: pixels are decoded later, for the k images we keep.
    """
    positions = []
    for path in sorted(Path(series_dir).glob("*.dcm")):
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        try:
            iop = np.array(ds.ImageOrientationPatient, dtype=float)
            ipp = np.array(ds.ImagePositionPatient, dtype=float)
            pos = float(np.dot(np.cross(iop[:3], iop[3:]), ipp))
        except AttributeError:
            pos = float(getattr(ds, "InstanceNumber", 0))
        positions.append((pos, path))
    return [path for _, path in sorted(positions)]


def pick_slice_indices(n_available, k=K):
    """Choose k slice indices, spread evenly over a series of n_available.

    Handles both directions with one rule:
      - more images than k  -> keep k of them, evenly spaced
      - fewer images than k -> repeat real images to reach k
    """
    if n_available < 1:
        return []
    return np.linspace(0, n_available - 1, k).round().astype(int)


def raw_slice(ds):
    """Decode one slice to real-valued pixels, before any normalisation.
    """
    arr = ds.pixel_array.astype(np.float32)
    arr = arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


def crop_to_mm(arr, pixel_spacing, crop_mm=CROP_MM):
    """Centre-crop one slice to a constant PHYSICAL size, in millimetres.
    OFF by default (CROP_MM is None), in which case the slice is returned
    untouched -- the caller's guard is not the only thing keeping this safe.
    """
    if crop_mm is None:
        return arr
    want_r = int(round(crop_mm / float(pixel_spacing[0])))
    want_c = int(round(crop_mm / float(pixel_spacing[1])))
    rows, cols = arr.shape

    # Crop the axes that are large enough.
    r0 = max((rows - want_r) // 2, 0)
    c0 = max((cols - want_c) // 2, 0)
    arr = arr[r0:r0 + min(want_r, rows), c0:c0 + min(want_c, cols)]

    # Pad the axes that were too small (the rare narrow-FOV series).
    pad_r = max(want_r - arr.shape[0], 0)
    pad_c = max(want_c - arr.shape[1], 0)
    if pad_r or pad_c:
        arr = np.pad(arr,
                     ((pad_r // 2, pad_r - pad_r // 2), (pad_c // 2, pad_c - pad_c // 2)),
                     mode="edge")
    return arr


def resize_slice(arr, size=IMG_SIZE):
    """Bilinear resize of one 2D array, pure numpy, half-pixel-center convention.
    (mode="bilinear", align_corners=False)
    """
    in_h, in_w = arr.shape
    if in_h == size and in_w == size:
        return arr.astype(np.float32)

    row = np.clip((np.arange(size) + 0.5) * (in_h / size) - 0.5, 0, in_h - 1)
    col = np.clip((np.arange(size) + 0.5) * (in_w / size) - 0.5, 0, in_w - 1)
    r0, c0 = np.floor(row).astype(int), np.floor(col).astype(int)
    r1, c1 = np.clip(r0 + 1, 0, in_h - 1), np.clip(c0 + 1, 0, in_w - 1)
    rf, cf = (row - r0)[:, None], (col - c0)[None, :]

    top = arr[r0][:, c0] * (1 - cf) + arr[r0][:, c1] * cf
    bottom = arr[r1][:, c0] * (1 - cf) + arr[r1][:, c1] * cf
    return (top * (1 - rf) + bottom * rf).astype(np.float32)


def slices_to_channels(x): #fix on format expectation
    """Sequence layout -> 2.5D layout.

        (k, img_size, img_size, 1)  ->  (img_size, img_size, k)

    Slice i becomes channel i, so channel order IS anatomical order along the
    plane. Nothing is resampled, rescaled or copied in value terms; this is a
    transpose plus dropping the length-1 channel axis.
    """
    if x is None:
        return None
    return np.ascontiguousarray(np.transpose(x[..., 0], (1, 2, 0)))


def sequence_to_tensor(series_dir, k=K, img_size=IMG_SIZE, crop_mm=CROP_MM, pct=PCT,
                       as_channels=False):
    """
    One channel, not three: the images are monochrome, so three copies would
    only cost 3x the memory.

    Intensity is normalised with ONE window per SERIES
    so normalising is not optional
    """
    paths = sorted_slice_paths(series_dir)
    if not paths:
        return None

    idx = pick_slice_indices(len(paths), k)
    datasets = [pydicom.dcmread(paths[i]) for i in idx]
    slices = [raw_slice(ds) for ds in datasets]

    cropped = []
    for ds, arr in zip(datasets, slices):
        spacing = getattr(ds, "PixelSpacing", None)
        if crop_mm is not None and spacing is not None:
            arr = crop_to_mm(arr, spacing, crop_mm)
        cropped.append(arr)

    lo, hi = np.percentile(np.concatenate([a.ravel() for a in cropped]), pct)
    out = np.stack([resize_slice(np.clip(a, lo, hi), img_size) for a in cropped])
    out = (out - lo) / (hi - lo) if hi > lo else np.zeros_like(out)
    out = out[..., None].astype(np.float32)
    return slices_to_channels(out) if as_channels else out


def ranked_series(study_uid, series_df, axis="X"):
    """ALL of a study's series for one plane, best first (fluid-sensitive wins,
    UID breaks ties). pick_series takes the first; callers that must survive a
    corrupt file walk the list until one decodes."""
    rows = series_df[(series_df.StudyInstanceUID == study_uid)
                     & (series_df.Anatomical_Plane == AXIS_PLANE[axis])]
    rows = rows.sort_values(["Fluid_Sensitive", "SeriesInstanceUID"],
                            ascending=[False, True])
    return list(rows.SeriesInstanceUID)


def pick_series(study_uid, series_df, axis="X"):
    """Choose ONE series of a study for the given axis.

    A study holds 1-7 series per plane, so a rule is needed. Fluid-sensitive
    (PD/T2 fat-sat) wins, because most of the twelve findings are fluid, and
    several of them are invisible on a structural sequence. Ties break on UID
    so the choice is stable across runs.

    Returns the SeriesInstanceUID, or None if the study has no series in that
    plane.
    """
    rows = series_df[(series_df.StudyInstanceUID == study_uid)
                     & (series_df.Anatomical_Plane == AXIS_PLANE[axis])]
    if rows.empty:
        return None
    rows = rows.sort_values(["Fluid_Sensitive", "SeriesInstanceUID"], ascending=[False, True])
    return rows.iloc[0].SeriesInstanceUID


def study_to_tensor(study_uid, series_df, data_root="data", axis="X", **kwargs):
    """A StudyInstanceUID -> the tensor for one plane ("X" sagittal by default).
    Extra keywords go straight to sequence_to_tensor, so as_channels=True gives
    the 2.5D (img_size, img_size, k) layout here too.
    """
    series_uid = pick_series(study_uid, series_df, axis)
    if series_uid is None:
        return None
    return sequence_to_tensor(Path(data_root) / "train_series" / study_uid / series_uid, **kwargs)


if __name__ == "__main__":

    import pandas as pd

    root = Path(__file__).resolve().parent.parent
    series_df = pd.read_csv(root / "data" / "train_series.csv")

    uid = series_df.StudyInstanceUID.iloc[0]
    x = study_to_tensor(uid, series_df, data_root=root / "data", axis="X")
    if x is None:
        raise SystemExit(f"No sagittal series with readable .dcm files for {uid}")
    print(f"study_to_tensor  -> {x.shape} {x.dtype} [{x.min():.2f}, {x.max():.2f}]")
    print(f"  as_channels=True -> {c.shape}   batched for Keras: {(1,) + c.shape}")
    print(f"  same pixels as the sequence layout: {np.array_equal(slices_to_channels(x), c)}")

    for series_uid in series_df.SeriesInstanceUID.head(3):
        study_uid = series_df.loc[series_df.SeriesInstanceUID == series_uid,
                                  "StudyInstanceUID"].iloc[0]
        d = root / "data" / "train_series" / study_uid / series_uid
        n = len(list(d.glob("*.dcm")))
        t = sequence_to_tensor(d)
        print(f"  {n:>3} images -> {'(unreadable)' if t is None else t.shape}")

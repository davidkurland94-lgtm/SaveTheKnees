"""
dicom_utils.py — standalone helpers for opening and working with DICOM files.

No dependency on the rest of the RSNA project scaffold — just pydicom,
numpy, and (optionally) matplotlib for viewing. Use this to explore real
competition data once you have it.

Install requirements:
    pip install pydicom numpy matplotlib --break-system-packages
    # if you hit "compressed pixel data" errors, also:
    pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg --break-system-packages

No PyTorch or Pillow required anywhere in this file -- resizing is done with
a small pure-numpy bilinear implementation (see resize_slice_array).
"""

from pathlib import Path
import numpy as np
import pydicom


# ---------------------------------------------------------------------------
# 1. Opening a single file
# ---------------------------------------------------------------------------

def load_dicom(path):
    """Read a .dcm file and return the pydicom Dataset object (metadata + pixel data)."""
    return pydicom.dcmread(str(path))


def print_metadata(ds, tags=None):
    """
    Print the most commonly-needed tags. Pass a custom list of tag names
    in `tags` to print only those, otherwise a sensible default set is used.
    """
    default_tags = [
        "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
        "PatientID", "Modality", "SeriesDescription",
        "Rows", "Columns", "PixelSpacing", "SliceThickness",
        "ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber",
        "BitsAllocated", "PixelRepresentation", "PhotometricInterpretation",
        "RescaleSlope", "RescaleIntercept", "WindowCenter", "WindowWidth",
    ]
    for tag in (tags or default_tags):
        print(f"{tag:28s}: {getattr(ds, tag, '(not present)')}")
    print(f"{'View (derived)':28s}: {get_slice_view_from_dataset(ds)}")


# ---------------------------------------------------------------------------
# 2. Identifying the scan plane (axial / sagittal / coronal)
# ---------------------------------------------------------------------------

def get_slice_view_from_dataset(ds) -> str:
    """
    Determine the anatomical plane (Axial / Sagittal / Coronal) from an
    already-loaded pydicom Dataset's ImageOrientationPatient tag.

    ImageOrientationPatient is 6 numbers: the row direction cosine (3) then
    the column direction cosine (3). Rounding each to the nearest whole
    number identifies which of the 3 standard planes the scan is aligned to:
        Axial:    row=[1,0,0]  col=[0,1,0]
        Coronal:  row=[1,0,0]  col=[0,0,-1]
        Sagittal: row=[0,1,0]  col=[0,0,-1]

    NOTE: a genuinely oblique scan (not aligned to any standard plane) can
    still get bucketed into "Sagittal" here due to the loose matching
    condition below -- rare for knee MRI, but worth knowing if you see
    unexpected plane labels on real data.
    """
    if "ImageOrientationPatient" not in ds:
        return "Unknown/Oblique"

    orient = [round(x) for x in ds.ImageOrientationPatient]

    if orient[0] == 1 and orient[4] == 1:
        return "Axial"
    elif orient[1] == 1 or orient[2] == 1 or orient[0] == 0:
        # Sagittal scans typically start with row vector 0 (perpendicular to X-axis)
        if orient[0] == 0:
            return "Sagittal"
        return "Sagittal"
    elif orient[0] == 1 and orient[5] == -1:
        return "Coronal"

    return "Unknown/Oblique"


def get_slice_view(dcm_path) -> str:
    """Same as get_slice_view_from_dataset, but takes a file path directly."""
    ds = pydicom.dcmread(str(dcm_path))
    return get_slice_view_from_dataset(ds)


# ---------------------------------------------------------------------------
# 3. Getting a usable pixel array (handles the gotchas from before)
# ---------------------------------------------------------------------------

def get_pixel_array(ds, normalize="percentile", pct=(1, 99)):
    """
    Decode pixel data and return a normalized float32 array in [0, 1],
    ready to feed into a model or matplotlib.

    normalize:
        "percentile" -> clip to the given percentile range, then min-max
                         (robust to outlier bright/dark pixels)
        "window"     -> use the DICOM's own WindowCenter/WindowWidth tags
                         if present, falling back to "percentile" otherwise
        "none"       -> just min-max the raw (rescaled) array, no clipping
    """
    arr = ds.pixel_array.astype(np.float32)

    # Apply rescale slope/intercept if present (raw -> "true" values)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    # MONOCHROME1 means higher value = darker -- invert so higher = brighter,
    # consistent with the MONOCHROME2 convention most tooling assumes.
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    if normalize == "window":
        center = getattr(ds, "WindowCenter", None)
        width = getattr(ds, "WindowWidth", None)
        # these tags can be multi-valued (a list) on some files -- take the first
        if isinstance(center, (list, pydicom.multival.MultiValue)):
            center = center[0]
        if isinstance(width, (list, pydicom.multival.MultiValue)):
            width = width[0]
        if center is not None and width is not None:
            lo, hi = float(center) - float(width) / 2, float(center) + float(width) / 2
        else:
            lo, hi = np.percentile(arr, pct)  # fallback if tags missing
    elif normalize == "percentile":
        lo, hi = np.percentile(arr, pct)
    else:  # "none"
        lo, hi = arr.min(), arr.max()

    arr = np.clip(arr, lo, hi)
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)
    return arr


# ---------------------------------------------------------------------------
# 4. Working with a whole series (folder of slices)
# ---------------------------------------------------------------------------

def load_series(folder):
    """
    Read every .dcm file in `folder`, sort them into correct spatial order,
    and return a list of pydicom Datasets (still holding their pixel data).

    Sorting priority:
      1. Position along the slice-normal vector (computed from
         ImageOrientationPatient + ImagePositionPatient) -- the geometrically
         correct method, robust even if InstanceNumber is missing/wrong.
      2. Falls back to InstanceNumber if orientation tags are missing.
    """
    folder = Path(folder)
    paths = sorted(folder.glob("*.dcm"))
    datasets = [load_dicom(p) for p in paths]
    if not datasets:
        return []

    def slice_position(ds):
        try:
            iop = np.array(ds.ImageOrientationPatient, dtype=float)
            ipp = np.array(ds.ImagePositionPatient, dtype=float)
            row_vec, col_vec = iop[:3], iop[3:]
            normal = np.cross(row_vec, col_vec)
            return float(np.dot(normal, ipp))  # projection onto slice-normal axis
        except AttributeError:
            return float(getattr(ds, "InstanceNumber", 0))

    datasets.sort(key=slice_position)
    return datasets


def series_to_volume(datasets, normalize="percentile"):
    """Stack a sorted list of Datasets (from load_series) into one 3D numpy array (num_slices, H, W)."""
    slices = [get_pixel_array(ds, normalize=normalize) for ds in datasets]
    return np.stack(slices, axis=0)


def describe_series(folder) -> dict:
    """
    Quick summary of a series folder: which plane it is (Axial/Sagittal/
    Coronal), how many slices, and the series description if present.
    Handy for sanity-checking a study's series before deciding which plane
    to feed the model for a given finding (e.g. ACL reads best sagittal).
    """
    datasets = load_series(folder)
    if not datasets:
        return {"view": "Unknown/Oblique", "n_slices": 0, "series_description": None}
    return {
        "view": get_slice_view_from_dataset(datasets[0]),
        "n_slices": len(datasets),
        "series_description": getattr(datasets[0], "SeriesDescription", None),
    }


# ---------------------------------------------------------------------------
# 5. Grouping FLAT files by series, and looping over patients/studies
#
# Some datasets don't give you one subfolder per series -- instead every
# slice for a study sits in one flat folder, and only the SeriesInstanceUID
# tag inside each file tells you which series it belongs to. The functions
# below handle that layout.
# ---------------------------------------------------------------------------

def group_dicoms_by_series(study_dir) -> dict:
    """
    Read every .dcm file directly inside `study_dir` (flat, not organized
    into series subfolders) and group them by SeriesInstanceUID.

    Uses stop_before_pixels=True while grouping -- we only need header tags
    here, not pixel data, so this is much faster over large studies.

    Returns: {series_uid: [pydicom Dataset, ...]} -- each list is sorted
    into correct spatial order (same sorting logic as load_series).
    """
    study_dir = Path(study_dir)
    paths = sorted(study_dir.glob("*.dcm"))

    groups = {}
    for p in paths:
        ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        uid = getattr(ds, "SeriesInstanceUID", "UNKNOWN_SERIES")
        groups.setdefault(uid, []).append((p, ds))

    def slice_position(item):
        path, ds = item
        try:
            iop = np.array(ds.ImageOrientationPatient, dtype=float)
            ipp = np.array(ds.ImagePositionPatient, dtype=float)
            normal = np.cross(iop[:3], iop[3:])
            return float(np.dot(normal, ipp))
        except AttributeError:
            return float(getattr(ds, "InstanceNumber", 0))

    # Sort each group spatially, then re-read those files WITH pixel data
    # (only now, since this is what callers will actually want to use).
    sorted_groups = {}
    for uid, items in groups.items():
        items.sort(key=slice_position)
        sorted_groups[uid] = [load_dicom(p) for p, _ in items]
    return sorted_groups


def get_study_series_summary(study_dir) -> list:
    """
    Summarize every series found (flat) in a study folder: view, slice
    count, series number/description, sorted by SeriesNumber (falling back
    to SeriesInstanceUID if SeriesNumber is missing) so "middle" has a
    stable, meaningful order rather than dict/UID order.

    Returns a list of dicts:
        [{"series_uid", "view", "n_slices", "series_number",
          "series_description", "datasets"}, ...]
    """
    groups = group_dicoms_by_series(study_dir)

    summaries = []
    for uid, datasets in groups.items():
        first = datasets[0]
        summaries.append({
            "series_uid": uid,
            "view": get_slice_view_from_dataset(first),
            "n_slices": len(datasets),
            "series_number": getattr(first, "SeriesNumber", None),
            "series_description": getattr(first, "SeriesDescription", None),
            "datasets": datasets,
        })

    summaries.sort(key=lambda s: (s["series_number"] is None, s["series_number"], s["series_uid"]))
    return summaries


def select_middle_series(summaries: list) -> dict:
    """
    Given the output of get_study_series_summary(), pick the middle series
    by position in that (already meaningfully-ordered) list -- a simple,
    content-agnostic way to get one representative sequence per study.
    """
    if not summaries:
        return None
    return summaries[len(summaries) // 2]


def iter_patient_studies(patients_root):
    """
    Yield (patient_id, study_id, study_dir) for every study folder under a
    patients_root/<patient_id>/<study_id>/ hierarchy.
    """
    patients_root = Path(patients_root)
    for patient_dir in sorted(patients_root.iterdir()):
        if not patient_dir.is_dir():
            continue
        for study_dir in sorted(patient_dir.iterdir()):
            if not study_dir.is_dir():
                continue
            yield patient_dir.name, study_dir.name, study_dir


def collect_middle_sequences(patients_root) -> dict:
    """
    Walk a patients_root/<patient_id>/<study_id>/*.dcm hierarchy and return,
    for every study, the middle series (by select_middle_series) grouped by
    patient:

        {patient_id: {study_id: middle_series_summary_dict, ...}, ...}

    Studies that fail to read (corrupt files, no series found, missing
    orientation tags, etc.) are skipped with a warning printed rather than
    crashing the whole loop -- real datasets are messy, and one bad study
    shouldn't stop you from processing the rest.
    """
    results = {}
    for patient_id, study_id, study_dir in iter_patient_studies(patients_root):
        try:
            summaries = get_study_series_summary(study_dir)
            middle = select_middle_series(summaries)
            if middle is None:
                print(f"  [skip] {patient_id}/{study_id}: no series found")
                continue
            results.setdefault(patient_id, {})[study_id] = middle
        except Exception as e:
            print(f"  [skip] {patient_id}/{study_id}: failed to read ({e})")
    return results


# ---------------------------------------------------------------------------
# 6. Turning a sequence into a model-ready array
# ---------------------------------------------------------------------------

def resize_slice_array(arr: np.ndarray, img_size: int) -> np.ndarray:
    """
    Resize a single 2D array to (img_size, img_size) using bilinear
    interpolation, implemented in pure numpy (no PyTorch, no Pillow).
    Matches torch.nn.functional.interpolate(..., mode="bilinear",
    align_corners=False) to within floating-point rounding, verified
    against it directly -- so results are consistent with the rest of
    this project even though this file has no torch dependency.
    """
    in_h, in_w = arr.shape
    arr = arr.astype(np.float32)

    if in_h == img_size and in_w == img_size:
        return arr

    # For each output pixel, find the corresponding fractional coordinate
    # in the input image (the "half-pixel center" convention, same as
    # align_corners=False), then blend the 4 nearest input pixels.
    row_idx = (np.arange(img_size) + 0.5) * (in_h / img_size) - 0.5
    col_idx = (np.arange(img_size) + 0.5) * (in_w / img_size) - 0.5
    row_idx = np.clip(row_idx, 0, in_h - 1)
    col_idx = np.clip(col_idx, 0, in_w - 1)

    row0 = np.floor(row_idx).astype(int)
    row1 = np.clip(row0 + 1, 0, in_h - 1)
    col0 = np.floor(col_idx).astype(int)
    col1 = np.clip(col0 + 1, 0, in_w - 1)

    row_frac = (row_idx - row0)[:, None]
    col_frac = (col_idx - col0)[None, :]

    top = arr[row0][:, col0] * (1 - col_frac) + arr[row0][:, col1] * col_frac
    bottom = arr[row1][:, col0] * (1 - col_frac) + arr[row1][:, col1] * col_frac
    return (top * (1 - row_frac) + bottom * row_frac).astype(np.float32)


def sample_datasets(datasets: list, max_slices: int) -> list:
    """Evenly sample up to max_slices Datasets from an already-sorted list."""
    if len(datasets) == 0:
        return []
    if len(datasets) <= max_slices:
        return datasets
    idx = np.linspace(0, len(datasets) - 1, max_slices).astype(int)
    return [datasets[i] for i in idx]


def sequence_to_array(datasets: list, img_size: int = 224, max_slices: int = 24,
                       normalize: str = "percentile") -> np.ndarray:
    """
    Convert an already-loaded, sorted list of pydicom Datasets (e.g. from
    load_series(), or the "datasets" field of a middle_sequence summary from
    select_middle_series / collect_middle_sequences) into one fixed-size,
    normalized numpy array ready to feed a model.

    - Evenly samples down to max_slices if the series has more slices than that.
    - Zero-pads if the series has fewer slices than max_slices (so every
      output array has the same shape, which is required for batching).

    Returns: float32 array of shape (max_slices, img_size, img_size), values in [0, 1].
    """
    sampled = sample_datasets(datasets, max_slices)
    slices = [resize_slice_array(get_pixel_array(ds, normalize=normalize), img_size)
              for ds in sampled]
    while len(slices) < max_slices:
        slices.append(np.zeros((img_size, img_size), dtype=np.float32))
    return np.stack(slices, axis=0)


def get_middle_slice(datasets: list):
    """
    Return the single pydicom Dataset sitting at the middle index of an
    already spatially-sorted slice list (e.g. datasets from load_series(),
    or middle_summary["datasets"] from select_middle_series()).

    This is "the one image in the middle of the sequence" -- as opposed to
    sequence_to_array(), which returns the whole (sampled/padded) stack.
    """
    if not datasets:
        return None
    return datasets[len(datasets) // 2]


def middle_slice_to_array(datasets: list, img_size: int = 224,
                           normalize: str = "percentile") -> np.ndarray:
    """
    Get just the middle slice of a sequence as ONE (img_size, img_size)
    normalized array -- a single representative 2D image per study, rather
    than a whole padded volume. Useful for a simpler single-image-per-study
    baseline, or for a quick visual sanity check of a study.

    Returns: float32 array of shape (img_size, img_size), values in [0, 1].
              None if datasets is empty.
    """
    ds = get_middle_slice(datasets)
    if ds is None:
        return None
    arr = get_pixel_array(ds, normalize=normalize)
    return resize_slice_array(arr, img_size)


# ---------------------------------------------------------------------------
# 7. Quick visualization
# ---------------------------------------------------------------------------

def show_slice(ds_or_array, normalize="percentile", title=None):
    """Display a single slice. Accepts either a pydicom Dataset or a raw numpy array."""
    import matplotlib.pyplot as plt

    if isinstance(ds_or_array, np.ndarray):
        img = ds_or_array
    else:
        img = get_pixel_array(ds_or_array, normalize=normalize)

    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray")
    plt.title(title or "")
    plt.axis("off")
    plt.show()


def show_grid(datasets, normalize="percentile", max_slices=12, cols=4, suptitle=None):
    """Display a grid of slices from a series -- handy for scanning a whole volume quickly."""
    import matplotlib.pyplot as plt

    datasets = datasets[:max_slices]
    rows = int(np.ceil(len(datasets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i < len(datasets):
            img = get_pixel_array(datasets[i], normalize=normalize)
            ax.imshow(img, cmap="gray")
            ax.set_title(f"slice {i}", fontsize=8)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    plt.show()


def show_middle_sequence(middle_summary: dict, normalize="percentile", max_slices_to_show=8):
    """
    Display a grid of slices from a middle-sequence summary dict (the output
    of select_middle_series(), or one value from collect_middle_sequences()).
    """
    if middle_summary is None:
        print("No middle sequence to display.")
        return

    desc = middle_summary.get("series_description") or middle_summary["series_uid"]
    title = (f"{desc}  |  view={middle_summary['view']}  |  "
             f"{middle_summary['n_slices']} slices")
    show_grid(middle_summary["datasets"], normalize=normalize,
              max_slices=max_slices_to_show, suptitle=title)


def show_middle_slice(middle_summary: dict, normalize="percentile"):
    """
    Display just the ONE slice sitting in the middle of a middle-sequence
    summary dict (the output of select_middle_series(), or one value from
    collect_middle_sequences()) -- a single representative image, not a grid.
    """
    if middle_summary is None:
        print("No middle sequence to display.")
        return

    datasets = middle_summary["datasets"]
    ds = get_middle_slice(datasets)
    desc = middle_summary.get("series_description") or middle_summary["series_uid"]
    title = (f"{desc}  |  view={middle_summary['view']}  |  "
             f"slice {len(datasets)//2 + 1}/{middle_summary['n_slices']}")
    show_slice(ds, normalize=normalize, title=title)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Demo with pydicom's bundled sample file (swap for a real path)
    from pydicom.data import get_testdata_file

    path = get_testdata_file("MR_small.dcm")
    ds = load_dicom(path)

    print("=== Metadata ===")
    print_metadata(ds)

    arr = get_pixel_array(ds, normalize="percentile")
    print("\n=== Pixel array ===")
    print("shape:", arr.shape, "dtype:", arr.dtype, "range:", (arr.min(), arr.max()))

    # show_slice(ds)  # uncomment to view interactively (needs a display)

    # For a whole series folder:
    # datasets = load_series("/path/to/StudyUID/SeriesUID")
    # volume = series_to_volume(datasets)      # shape (num_slices, H, W)
    # show_grid(datasets)

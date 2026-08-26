"""
dicom_utils.py — standalone helpers for opening and working with DICOM files.

No dependency on the rest of the RSNA project scaffold — just pydicom,
numpy, and (optionally) matplotlib for viewing. Use this to explore real
competition data once you have it.

Install requirements:
    pip install pydicom numpy matplotlib --break-system-packages
    # if you hit "compressed pixel data" errors, also:
    pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg --break-system-packages
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


# ---------------------------------------------------------------------------
# 2. Getting a usable pixel array (handles the gotchas from before)
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
# 3. Working with a whole series (folder of slices)
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


# ---------------------------------------------------------------------------
# 4. Quick visualization
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


def show_grid(datasets, normalize="percentile", max_slices=12, cols=4):
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
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# 5. Get type of view
# ---------------------------------------------------------------------------

def get_slice_view(dcm_path):
    dcm = pydicom.dcmread(dcm_path)

    # Check if the mandatory orientation tag is present
    if "ImageOrientationPatient" in dcm:
        orient = [round(x) for x in dcm.ImageOrientationPatient]

        # Analyze the row and column directional cosine signatures
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

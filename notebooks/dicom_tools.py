"""Reading knee MRI DICOM series into standardized volumes.

A study on disk looks like this:

    C:\\datasets\\Knees\\<StudyInstanceUID>\\<SeriesInstanceUID>\\<SOPInstanceUID>.dcm

One .dcm file is one 2D slice. A series folder is one acquisition (one plane,
one pulse sequence) and its slices stack into a 3D volume. A study folder is
all the series for one knee - typically 5 here, e.g. sagittal PD, sagittal PD
fat-sat, axial STIR, coronal T1, coronal PD fat-sat.

The three things that make DICOM reading non-obvious, all handled below:

1. Slice order in the folder is meaningless. Filenames are random UIDs and
   InstanceNumber is not guaranteed to follow anatomy. Slices are ordered by
   their physical position, projected onto the slice normal.
2. Plane (sagittal / axial / coronal) is not stored as a field. It is derived
   from the direction cosines in ImageOrientationPatient.
3. MR pixel values have no absolute units. Unlike CT Hounsfield units, a value
   of 800 means nothing across scanners, sequences, or even two series of the
   same knee. Intensity has to be normalized per volume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
from scipy import ndimage

# WSL sees the Windows drive under /mnt/c. Override if your data lives elsewhere.
DATA_ROOT = Path("/mnt/c/datasets/Knees")

# DICOM patient coordinates are LPS: +x -> patient Left, +y -> Posterior,
# +z -> Superior. The slice normal's dominant axis names the plane: a sagittal
# stack advances along x (left-right), coronal along y, axial along z.
PLANE_BY_AXIS = {0: "Sagittal", 1: "Coronal", 2: "Axial"}

# Target direction for each volume axis, as an LPS vector, per plane. Chosen so
# that plt.imshow (row 0 at top) shows the knee the way a radiologist expects:
# superior up, anterior left for sagittal; patient-left on the image right for
# coronal and axial. Slices advance R->L, A->P, S->I respectively.
CANONICAL_DIRS = {
    #            slice axis          row axis (down)     col axis (right)
    "Sagittal": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
    "Coronal": ((0, 1, 0), (0, 0, -1), (1, 0, 0)),
    "Axial": ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
}


@dataclass
class Series:
    """One loaded series: the voxels plus the geometry needed to interpret them."""

    volume: np.ndarray  # (n_slices, rows, cols), float32
    plane: str  # 'Sagittal' | 'Coronal' | 'Axial'
    spacing: tuple[float, float, float]  # mm per voxel, (slice, row, col)
    series_uid: str
    study_uid: str
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.volume.shape

    @property
    def extent_mm(self):
        """Physical size of the volume, (slice, row, col) in mm."""
        return tuple(s * n for s, n in zip(self.spacing, self.volume.shape))

    def __repr__(self):
        mm = " x ".join(f"{v:.0f}" for v in self.extent_mm)
        sp = " x ".join(f"{v:.2f}" for v in self.spacing)
        return (
            f"<Series {self.plane} {self.volume.shape} "
            f"spacing=({sp})mm extent=({mm})mm '{self.meta.get('series_description', '')}'>"
        )


# --------------------------------------------------------------------------
# 1. Finding files
# --------------------------------------------------------------------------

def list_studies(root: Path | str = DATA_ROOT) -> list[Path]:
    return sorted(p for p in Path(root).iterdir() if p.is_dir())


def list_series(study_dir: Path | str) -> list[Path]:
    return sorted(p for p in Path(study_dir).iterdir() if p.is_dir())


def read_headers(series_dir: Path | str) -> list[pydicom.Dataset]:
    """Every slice header in a series, no pixel data.

    stop_before_pixels=True skips the (7FE0,0010) PixelData element and its
    decode. On these files that is roughly 2x on wall clock and far more on
    memory - a header is a few KB against 512KB of pixels - which is what
    makes a sweep of all 4,407 studies practical.
    """
    files = sorted(Path(series_dir).glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"no .dcm files in {series_dir}")
    return [pydicom.dcmread(f, stop_before_pixels=True) for f in files]


# --------------------------------------------------------------------------
# 2. Geometry
# --------------------------------------------------------------------------

def slice_normal(ds: pydicom.Dataset) -> np.ndarray:
    """Unit vector perpendicular to the image plane, in LPS patient coordinates.

    ImageOrientationPatient (0020,0037) holds six numbers: the first three are
    the direction in which the *column index* increases (moving right along a
    row), the last three the direction in which the *row index* increases
    (moving down a column). Their cross product is the stacking direction.
    """
    iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
    return np.cross(iop[:3], iop[3:])


def detect_plane(ds: pydicom.Dataset) -> str:
    """'Sagittal' | 'Coronal' | 'Axial', from the slice normal's dominant axis.

    Scanners tilt the plane a few degrees off the anatomical axes, so this is
    argmax(|normal|) rather than an exact match. SeriesDescription often says
    'SAG' too, but it is free text - 'T1_TSE_CS' names no plane at all - so
    geometry is the only reliable source.
    """
    return PLANE_BY_AXIS[int(np.argmax(np.abs(slice_normal(ds))))]


def sort_by_position(headers: list[pydicom.Dataset]) -> tuple[list[pydicom.Dataset], np.ndarray]:
    """Order slices along the normal and return their positions in mm.

    ImagePositionPatient (0020,0032) is the LPS coordinate of the center of the
    first voxel. Its projection onto the normal is where the slice sits in the
    stack - monotonic and physically meaningful, unlike InstanceNumber.
    """
    normal = slice_normal(headers[0])
    proj = np.array([float(np.dot(np.asarray(h.ImagePositionPatient, float), normal)) for h in headers])
    order = np.argsort(proj)
    return [headers[i] for i in order], proj[order]


def slice_spacing(positions: np.ndarray, headers: list[pydicom.Dataset]) -> float:
    """Center-to-center distance between slices, in mm.

    Measured from the sorted positions rather than trusted from a tag, because
    SliceThickness (3.4mm here) is the slab thickness and ignores any gap,
    while the true step is 4.635mm. Falls back to the tags for a 1-slice series.
    """
    if len(positions) > 1:
        return float(np.median(np.diff(positions)))
    h = headers[0]
    return float(getattr(h, "SpacingBetweenSlices", None) or h.SliceThickness)


def volume_center(headers: list[pydicom.Dataset]) -> np.ndarray:
    """LPS coordinate of the middle of the imaged volume, in mm.

    ImagePositionPatient marks the centre of the *first* voxel - the corner of
    the image - so it has to be walked half a row and half a column inward
    along the direction cosines to reach the middle. PixelSpacing is
    [row spacing, column spacing], so moving along a row uses the second value.
    """
    pts = []
    for h in headers:
        ipp = np.asarray(h.ImagePositionPatient, dtype=float)
        iop = np.asarray(h.ImageOrientationPatient, dtype=float)
        row_mm, col_mm = (float(v) for v in h.PixelSpacing)
        pts.append(
            ipp
            + iop[:3] * col_mm * (int(h.Columns) - 1) / 2
            + iop[3:] * row_mm * (int(h.Rows) - 1) / 2
        )
    return np.mean(pts, axis=0)


def guess_laterality(headers: list[pydicom.Dataset]) -> str:
    """'R' | 'L' | 'unknown' - a heuristic, see the caveat below.

    The Laterality (0020,0060) tag is blank in this dataset, so the only signal
    left is which side of scanner isocenter the knee sits on: in LPS, negative
    x is the patient's right. That holds when only one knee is in the coil, but
    it is table positioning, not anatomy. Treat it as a hint, not a label.
    """
    x = volume_center(headers)[0]
    if x < -20:
        return "R"
    if x > 20:
        return "L"
    return "unknown"


# --------------------------------------------------------------------------
# 3. Loading pixels
# --------------------------------------------------------------------------

def load_series(series_dir: Path | str) -> Series:
    """Read a series folder into a geometrically ordered, canonically oriented volume.

    Pixel values get RescaleSlope/RescaleIntercept applied (as the standard
    requires) but are still in arbitrary MR units afterwards - normalize before
    training. See standardize().
    """
    series_dir = Path(series_dir)
    headers = read_headers(series_dir)
    headers, positions = sort_by_position(headers)
    first = headers[0]

    gaps = np.diff(positions)
    if len(gaps) and np.ptp(gaps) > 0.1 * np.median(gaps):
        print(f"[warn] uneven slice spacing in {series_dir.name[-12:]}: {np.round(gaps, 2)}")
    if len(gaps) and np.any(gaps < 1e-3):
        print(f"[warn] duplicate slice positions in {series_dir.name[-12:]} - multi-echo series?")

    slices = []
    for h in headers:
        ds = pydicom.dcmread(series_dir / Path(h.filename).name)
        px = ds.pixel_array.astype(np.float32)
        # Stored integers are a compressed form of the real values; the standard
        # says to recover them as slope * stored + intercept. Philips varies the
        # slope per series (2.56 / 2.41 / 3.30 across this study's five), so
        # skipping it makes series silently non-comparable.
        px = px * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        slices.append(px)

    volume = np.stack(slices).astype(np.float32)
    plane = detect_plane(first)
    row_mm, col_mm = (float(v) for v in first.PixelSpacing)
    spacing = (slice_spacing(positions, headers), row_mm, col_mm)

    volume, spacing, flips = _to_canonical(volume, first, plane, spacing)

    meta = {
        "series_description": str(getattr(first, "SeriesDescription", "")),
        "series_number": int(getattr(first, "SeriesNumber", -1)),
        "n_slices": len(headers),
        "slice_thickness": float(first.SliceThickness),
        "spacing_between_slices": float(getattr(first, "SpacingBetweenSlices", 0) or 0),
        "manufacturer": str(getattr(first, "Manufacturer", "")),
        "model": str(getattr(first, "ManufacturerModelName", "")),
        "field_strength": float(getattr(first, "MagneticFieldStrength", 0) or 0),
        "tr": float(getattr(first, "RepetitionTime", 0) or 0),
        "te": float(getattr(first, "EchoTime", 0) or 0),
        "scan_options": str(getattr(first, "ScanOptions", "")),
        "patient_position": str(getattr(first, "PatientPosition", "")),
        "laterality_guess": guess_laterality(headers),
        "flips_applied": flips,
        "normal": np.round(slice_normal(first), 3).tolist(),
    }
    return Series(volume, plane, spacing, str(first.SeriesInstanceUID), str(first.StudyInstanceUID), meta)


def _to_canonical(volume, ds, plane, spacing):
    """Flip axes so every volume of a given plane faces the same way.

    Two scanners can produce the same sagittal knee with the slice order
    reversed or the image mirrored, and nothing in the array shape reveals it.
    Each of the three axes is compared against the target direction for its
    plane and flipped when it points the opposite way. Only flips are used, so
    the array layout stays (slice, row, col) and the spacing tuple still lines up.
    """
    iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
    actual = np.stack([slice_normal(ds), iop[3:], iop[:3]])  # slice, row, col
    target = np.asarray(CANONICAL_DIRS[plane], dtype=float)

    flips = []
    for axis in range(3):
        if np.dot(actual[axis], target[axis]) < 0:
            volume = np.flip(volume, axis=axis)
            flips.append(axis)
    return np.ascontiguousarray(volume), spacing, flips


# --------------------------------------------------------------------------
# 4. Standardizing
# --------------------------------------------------------------------------

def normalize_intensity(volume: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    """Map a volume to roughly [0, 1] using its own percentiles.

    Deliberately not using WindowCenter/WindowWidth: those are display hints
    that vary slice to slice within one series (WC 833..1160 across the 22
    sagittal slices here), so applying them would make neighbouring slices
    inconsistent. Percentiles are computed over the whole volume for the same
    reason, and clipped at 1/99 so one bright vessel or artifact cannot
    compress the tissue range.
    """
    lo, hi = np.percentile(volume, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - lo) / (hi - lo), 0, 1).astype(np.float32)


def resample_in_plane(volume, spacing, target_mm: float, order=1):
    """Rescale rows and columns to a fixed millimetres-per-pixel, slices untouched."""
    _, row_mm, col_mm = spacing
    zoom = (1.0, row_mm / target_mm, col_mm / target_mm)
    if np.allclose(zoom, 1.0):
        return volume, spacing
    out = ndimage.zoom(volume, zoom, order=order, mode="nearest")
    return out.astype(np.float32), (spacing[0], target_mm, target_mm)


def resample_slices(volume, spacing, n_slices: int, order=1):
    """Force a fixed slice count by interpolating along the stack axis."""
    n = volume.shape[0]
    if n == n_slices:
        return volume, spacing
    out = ndimage.zoom(volume, (n_slices / n, 1.0, 1.0), order=order, mode="nearest")
    return out.astype(np.float32), (spacing[0] * n / n_slices, spacing[1], spacing[2])


def center_crop_pad(volume, out_hw: tuple[int, int], pad_value=0.0):
    """Crop or pad rows/cols around the centre to an exact pixel size.

    Used after resample_in_plane: that call fixes the physical scale but leaves
    a different pixel count per series, and this makes the arrays stackable
    without rescaling anything a second time. The knee sits near the centre of
    the coil, so a centre crop keeps it.
    """
    out = volume
    for axis, want in zip((1, 2), out_hw):
        have = out.shape[axis]
        if have > want:
            start = (have - want) // 2
            out = out.take(range(start, start + want), axis=axis)
        elif have < want:
            before = (want - have) // 2
            pad = [(0, 0)] * 3
            pad[axis] = (before, want - have - before)
            out = np.pad(out, pad, mode="constant", constant_values=pad_value)
    return out


def standardize(series: Series, target_mm=0.5, out_hw=(320, 320), n_slices=None) -> Series:
    """Put any incoming series onto one common grid and intensity scale.

    Order matters: resample to physical units first, then fix the pixel grid,
    then normalize intensity last so the percentiles are taken on the voxels
    that actually reach the model.

    target_mm resamples to a fixed millimetres-per-pixel rather than resizing
    every image to a fixed pixel count. A plain resize would silently rescale
    anatomy, because in-plane spacing runs 0.264-0.331 mm/px across this study
    alone - a 25% size difference for the same knee. Since several labels are
    about size (osteophytes, effusion volume, tear extent), keeping millimetres
    constant is worth the crop.

    The default grid is 320px at 0.5mm = 160mm of field of view, which holds
    the whole acquired FOV here (169mm) minus a few mm of air at the edges.
    Dropping to (256, 256) = 128mm still contains the joint but shaves off
    femoral and tibial shaft, where contusion and fracture signal can sit -
    a reasonable trade for memory, but make it deliberately.

    n_slices is left None by default: slice counts vary (16-22 here) and
    interpolating along a 4.6mm axis invents data. Prefer a model that accepts
    a variable stack, and set this only if the architecture demands it.
    """
    vol, spacing = resample_in_plane(series.volume, series.spacing, target_mm)
    vol = center_crop_pad(vol, out_hw)
    if n_slices is not None:
        vol, spacing = resample_slices(vol, spacing, n_slices)
    vol = normalize_intensity(vol)

    meta = dict(series.meta, standardized={"target_mm": target_mm, "out_hw": out_hw, "n_slices": n_slices},
                original_shape=series.volume.shape, original_spacing=series.spacing)
    return Series(vol, series.plane, spacing, series.series_uid, series.study_uid, meta)


# --------------------------------------------------------------------------
# 5. Surveying the dataset
# --------------------------------------------------------------------------

def series_summary(series_dir: Path | str) -> dict:
    """One row of geometry per series, read from headers only (no pixels)."""
    series_dir = Path(series_dir)
    headers = read_headers(series_dir)
    headers, positions = sort_by_position(headers)
    h = headers[0]
    row_mm, col_mm = (float(v) for v in h.PixelSpacing)
    return {
        "StudyInstanceUID": str(h.StudyInstanceUID),
        "SeriesInstanceUID": str(h.SeriesInstanceUID),
        "plane": detect_plane(h),
        "n_slices": len(headers),
        "rows": int(h.Rows),
        "cols": int(h.Columns),
        "row_mm": row_mm,
        "col_mm": col_mm,
        "slice_mm": slice_spacing(positions, headers),
        "thickness": float(h.SliceThickness),
        "series_description": str(getattr(h, "SeriesDescription", "")),
        "tr": float(getattr(h, "RepetitionTime", 0) or 0),
        "te": float(getattr(h, "EchoTime", 0) or 0),
        "scan_options": str(getattr(h, "ScanOptions", "")),
        "manufacturer": str(getattr(h, "Manufacturer", "")),
        "laterality_guess": guess_laterality(headers),
    }


def build_manifest(root: Path | str = DATA_ROOT):
    """Geometry table for every series on disk. Header-only, so it scales."""
    import pandas as pd

    rows = []
    for study in list_studies(root):
        for series in list_series(study):
            try:
                rows.append(series_summary(series))
            except Exception as exc:  # a corrupt series should not stop the sweep
                print(f"[skip] {study.name[-12:]}/{series.name[-12:]}: {exc}")
    return pd.DataFrame(rows)

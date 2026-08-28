"""
dicom_3d.py — build and render a 3D volume from a knee MRI series.

Depends on dicom_utils.py being importable (same folder / sys.path).
Install requirements (on top of dicom_utils.py's requirements):
    pip install scikit-image plotly --break-system-packages

Rendering note: plot_3d_volume writes a standalone .html file by default
rather than displaying inline. A marching-cubes mesh serializes to tens of
MB of JSON, and an inline Plotly figure stores every byte of that *inside*
the .ipynb -- a handful of 3D plots is enough to take a 3 KB notebook past
300 MB. Pass inline=True if you really want it embedded.
"""

from pathlib import Path
import warnings
import webbrowser

import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology
import plotly.graph_objects as go

from dicom_utils3 import (
    iter_patient_studies,
    get_study_series_summary,
    load_series,
    series_to_volume,
)


# ---------------------------------------------------------------------------
# 1. Series selection for 3D reconstruction
# ---------------------------------------------------------------------------

def select_largest_series(summaries: list, view: str = None) -> dict:
    """
    Pick the series with the most slices -- the best candidate for a 3D
    reconstruction (as opposed to select_middle_series, which just grabs a
    single representative sequence).

    Pass view="Sagittal" / "Axial" / "Coronal" to restrict the candidates to
    one plane first (useful since e.g. ACL reads best sagittal).
    """
    candidates = summaries
    if view is not None:
        candidates = [s for s in summaries if s["view"] == view]
    if not candidates:
        return None
    return max(candidates, key=lambda s: s["n_slices"])


# ---------------------------------------------------------------------------
# 2. Volume + physical spacing
# ---------------------------------------------------------------------------

def get_volume_with_spacing(datasets, normalize="none"):
    """
    Stack a sorted list of Datasets into a volume + physical voxel spacing
    (dz, dy, dx), in mm.

    normalize="none" by default here (unlike dicom_utils' own default of
    "percentile") -- it skips the percentile clipping, so faint structure at
    the intensity extremes survives into the isosurface. Note that
    get_pixel_array still min-maxes *each slice independently* into [0, 1]
    whichever mode you pass, so no fixed `level` is comparable across slices
    or patients; that is why plot_3d_volume picks its threshold from the
    volume's own intensity distribution instead of taking a constant.

    Slice spacing is computed geometrically from the first two slices'
    ImagePositionPatient (like dicom_utils' own sorting logic), since
    SliceThickness is frequently wrong or missing on real data. Falls back
    to SliceThickness if there's only one slice.
    """
    volume = series_to_volume(datasets, normalize=normalize)

    row_spacing, col_spacing = [float(x) for x in datasets[0].PixelSpacing]

    if len(datasets) > 1:
        p0 = np.array(datasets[0].ImagePositionPatient, dtype=float)
        p1 = np.array(datasets[1].ImagePositionPatient, dtype=float)
        slice_spacing = float(np.linalg.norm(p1 - p0))
    else:
        slice_spacing = float(getattr(datasets[0], "SliceThickness", 1.0))

    return volume, (slice_spacing, row_spacing, col_spacing)


# ---------------------------------------------------------------------------
# 3. Surface extraction
# ---------------------------------------------------------------------------

def build_surface(volume, spacing, level=None, downsample=2, step_size=2,
                  smooth=1.0, min_component_voxels=None):
    """
    Extract a cleaned isosurface and return (verts, faces).

    Thresholding raw MRI intensity directly traces the noise floor as well as
    the anatomy, which is what makes a naive marching_cubes mesh explode into
    millions of tiny disconnected triangles. So instead:

        blur -> threshold -> drop small components -> fill holes -> blur mask
        -> marching cubes on the smooth 0..1 mask at level 0.5

    downsample: in-plane downsampling factor (rows/cols only, not slices --
    knee series are already coarse along z).

    step_size: marching-cubes stride. 2 is ~4x fewer triangles than 1 with
    little visible difference at this resolution; 1 for a final render.

    smooth: gaussian sigma in-plane (halved along z, where voxels are much
    thicker). 0 disables all smoothing and thresholds the raw volume.

    level: intensity threshold. Default None picks the 60th percentile of the
    blurred volume -- per-volume, since get_volume_with_spacing's output is
    normalized per slice and no constant is comparable across series.

    min_component_voxels: connected components smaller than this are dropped
    as noise. Default None scales it to the volume (1/5000th of it).
    """
    vol = volume[:, ::downsample, ::downsample]
    spacing = (spacing[0], spacing[1] * downsample, spacing[2] * downsample)

    if smooth > 0:
        blurred = ndi.gaussian_filter(vol, sigma=(smooth * 0.5, smooth, smooth))
    else:
        blurred = vol

    if level is None:
        level = float(np.percentile(blurred, 60))

    mask = blurred > level
    if min_component_voxels is None:
        min_component_voxels = max(64, mask.size // 5000)
    mask = morphology.remove_small_objects(mask, min_size=min_component_voxels)
    mask = ndi.binary_fill_holes(mask)

    if not mask.any():
        raise ValueError(
            f"level={level:.3f} left an empty mask -- lower it, or pass level=None "
            "to pick one from this volume's own intensity distribution."
        )

    # Marching cubes on the blurred *mask* rather than the mask itself, so the
    # surface interpolates smoothly instead of coming out voxel-stepped.
    surface = ndi.gaussian_filter(mask.astype(np.float32), sigma=1.0)
    verts, faces, _, _ = measure.marching_cubes(
        surface, level=0.5, spacing=spacing, step_size=step_size
    )

    # Sub-micron vertex precision costs bytes in the JSON and buys nothing on
    # a 0.28 mm voxel grid.
    return np.round(verts, 2).astype(np.float32), faces.astype(np.int32)


# ---------------------------------------------------------------------------
# 4. Rendering
# ---------------------------------------------------------------------------

def _default_render_dir() -> Path:
    """<repo root>/data/renders -- data/ is gitignored, so renders stay local."""
    root = Path.cwd()
    while not (root / ".git").exists() and root != root.parent:
        root = root.parent
    return root / "data" / "renders"


def plot_3d_volume(volume, spacing, level=None, downsample=2, title="",
                   step_size=2, smooth=1.0, out_html=None, inline=False,
                   open_browser=True):
    """
    Extract an isosurface with marching cubes and render it with Plotly.

    By default this writes a self-contained .html next to the data (loading
    plotly.js from the CDN) and opens it in a browser, printing the path and
    file size. Nothing lands in the notebook, so the .ipynb stays small.

    Returns the Path written -- deliberately NOT the Figure, since a cell
    ending on this call would then render the returned Figure inline and
    embed the whole mesh in the .ipynb regardless, which is the exact thing
    this is here to avoid. Use build_surface if you want to assemble your own.

    out_html: where to write. Default None -> <repo>/data/renders/<title>.html.
    inline: True displays in the notebook (and returns the Figure) instead of
        writing a file. Embeds the whole mesh -- expect tens of MB per figure.
    open_browser: False to write the file without opening it (batch use).

    See build_surface for level / downsample / step_size / smooth.
    """
    verts, faces = build_surface(
        volume, spacing, level=level, downsample=downsample,
        step_size=step_size, smooth=smooth,
    )

    fig = go.Figure(data=[go.Mesh3d(
        x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=1, color="lightblue", flatshading=False,
    )])
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
    )

    if inline:
        fig.show()
        return fig

    if out_html is None:
        slug = "".join(c if c.isalnum() or c in "-_." else "_"
                       for c in (title or "volume")) or "volume"
        out_html = _default_render_dir() / f"{slug}.html"
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    # include_plotlyjs="cdn" keeps the ~3.5 MB plotly bundle out of every file.
    fig.write_html(str(out_html), include_plotlyjs="cdn")

    size_mb = out_html.stat().st_size / 1e6
    print(f"{len(verts):,} verts / {len(faces):,} faces -> "
          f"{out_html} ({size_mb:.1f} MB)")
    if open_browser:
        webbrowser.open(out_html.resolve().as_uri())

    return out_html


# ---------------------------------------------------------------------------
# 5. Patient-space geometry (needed to put several series in one frame)
# ---------------------------------------------------------------------------

def series_affine(datasets):
    """
    4x4 affine mapping voxel index (slice, row, col) -> patient coordinates
    in mm, for a spatially sorted series (i.e. load_series output).

    Per the DICOM standard, ImageOrientationPatient holds the direction
    cosines of the first row (increasing column index) then the first column
    (increasing row index), and PixelSpacing is [row spacing, column
    spacing]. The slice axis is taken from the actual first->last
    ImagePositionPatient step rather than SliceThickness, which is routinely
    wrong or missing -- the same reasoning as get_volume_with_spacing.
    """
    ds0 = datasets[0]
    iop = np.array(ds0.ImageOrientationPatient, dtype=float)
    col_dir, row_dir = iop[0:3], iop[3:6]
    row_spacing, col_spacing = [float(x) for x in ds0.PixelSpacing]
    origin = np.array(ds0.ImagePositionPatient, dtype=float)

    if len(datasets) > 1:
        last = np.array(datasets[-1].ImagePositionPatient, dtype=float)
        slice_vec = (last - origin) / (len(datasets) - 1)
    else:
        normal = np.cross(col_dir, row_dir)
        slice_vec = normal * float(getattr(ds0, "SliceThickness", 1.0))

    affine = np.eye(4)
    affine[:3, 0] = slice_vec
    affine[:3, 1] = row_dir * row_spacing
    affine[:3, 2] = col_dir * col_spacing
    affine[:3, 3] = origin
    return affine


def _world_corners(shape, affine):
    """The 8 corners of a volume, in patient coordinates."""
    n_s, n_r, n_c = shape
    idx = np.array([[s, r, c, 1.0]
                    for s in (0, n_s - 1)
                    for r in (0, n_r - 1)
                    for c in (0, n_c - 1)])
    return (affine @ idx.T).T[:, :3]


def resample_to_grid(volume, affine, grid_origin, grid_shape, target_spacing):
    """
    Sample `volume` onto an axis-aligned isotropic patient-space grid.

    Returns a float32 array of `grid_shape` holding NaN wherever the grid
    falls outside this series' field of view -- so that averaging several
    series together can ignore the parts each one simply doesn't cover,
    rather than pulling the mean down towards zero at the edges.
    """
    zz, yy, xx = np.meshgrid(*[np.arange(n) for n in grid_shape], indexing="ij")
    world = (grid_origin
             + np.stack([zz, yy, xx], axis=-1).reshape(-1, 3) * target_spacing)

    homogeneous = np.concatenate([world, np.ones((len(world), 1))], axis=1)
    idx = (np.linalg.inv(affine) @ homogeneous.T).T[:, :3]

    inside = np.all((idx >= 0) & (idx <= np.array(volume.shape) - 1), axis=1)

    sampled = ndi.map_coordinates(volume, idx.T, order=1, mode="constant", cval=0.0)
    sampled[~inside] = np.nan
    return sampled.reshape(grid_shape).astype(np.float32)


# ---------------------------------------------------------------------------
# 6. Whole-study rendering
# ---------------------------------------------------------------------------

def _series_label(datasets, folder):
    return str(getattr(datasets[0], "SeriesDescription", None) or Path(folder).name)


def plot_sequences_separately(series_folders, out_dir=None, open_browser=False,
                              **kwargs):
    """
    Render one independent 3D surface per series -- each in its own .html,
    each in its own voxel frame. Use this to eyeball the sequences one by
    one; use fuse_and_plot to see them in a shared patient-space frame.

    open_browser defaults to False here (unlike plot_3d_volume) so a study
    with five series doesn't open five tabs. Extra kwargs pass through to
    plot_3d_volume -- e.g. downsample, step_size, smooth, level.

    Returns the list of Paths written, one per series.
    """
    rendered = []
    for folder in series_folders:
        datasets = load_series(folder)
        if not datasets:
            print(f"skipping {folder} -- no .dcm files")
            continue

        label = _series_label(datasets, folder)
        volume, spacing = get_volume_with_spacing(datasets)
        print(f"\n{label}: {len(datasets)} slices, volume {volume.shape}")

        out_html = None if out_dir is None else Path(out_dir) / f"{label}.html"
        rendered.append(plot_3d_volume(volume, spacing, title=label,
                                       out_html=out_html,
                                       open_browser=open_browser, **kwargs))
    return rendered


def fuse_and_plot(series_folders, target_spacing=1.5, labels=None,
                  out_html=None, open_browser=True, **kwargs):
    """
    Resample every series onto one shared isotropic patient-space grid and
    render a single surface through the combined volume.

    The sagittal / axial / coronal stacks of one knee are all thick-slice
    (~3.5-4 mm) but finely sampled in plane (~0.3 mm). Each one is therefore
    blurry along a *different* axis, so resampling them into a common frame
    and averaging recovers detail no single series has -- the standard
    multi-plane fusion trick.

    target_spacing: isotropic grid pitch in mm. Coarser = faster and smaller;
    finer = more detail. Memory grows with the cube of 1/target_spacing.

    labels: names for the series, used in the title. Defaults to each
    series' SeriesDescription. A short list is padded, so a stale `labels`
    won't crash the call.

    Returns the Path written (see plot_3d_volume on why not the Figure).

    Note the fused intensities mix contrasts (a T1 and a T2 FS mean nothing
    averaged together in a physical sense) -- this is for surface geometry,
    not for quantitative reads. Extra kwargs pass through to build_surface.
    """
    loaded = []
    for folder in series_folders:
        datasets = load_series(folder)
        if not datasets:
            print(f"skipping {folder} -- no .dcm files")
            continue
        loaded.append((folder, datasets))

    if not loaded:
        raise ValueError("no series could be loaded from series_folders")

    if labels is None:
        labels = []
    labels = list(labels) + [_series_label(ds, f)
                             for f, ds in loaded[len(labels):]]

    volumes = []
    for (folder, datasets), label in zip(loaded, labels):
        volume = series_to_volume(datasets, normalize="none")
        affine = series_affine(datasets)
        volumes.append((label, volume, affine))
        print(f"{label}: {volume.shape}")

    # One grid covering the union of every series' field of view.
    corners = np.vstack([_world_corners(v.shape, a) for _, v, a in volumes])
    grid_origin = corners.min(axis=0)
    grid_shape = tuple(
        int(np.ceil(e / target_spacing)) + 1
        for e in (corners.max(axis=0) - grid_origin)
    )
    print(f"\nfusing onto {grid_shape} grid at {target_spacing} mm isotropic")

    resampled = np.stack([
        resample_to_grid(v, a, grid_origin, grid_shape, target_spacing)
        for _, v, a in volumes
    ])

    # nanmean so a voxel is the average of only the series that actually
    # cover it; voxels no series covers stay 0.
    # Voxels outside every series' FOV are all-NaN; nanmean warns on those,
    # and nan_to_num below turns them into the 0 we want.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fused = np.nanmean(resampled, axis=0)
    coverage = np.count_nonzero(~np.isnan(resampled), axis=0)
    fused = np.nan_to_num(fused, nan=0.0)
    print(f"coverage: {np.count_nonzero(coverage) / coverage.size:.0%} of the "
          f"grid seen by >=1 series, {np.count_nonzero(coverage == len(volumes)) / coverage.size:.0%} by all")

    title = "fused: " + " + ".join(labels[:len(volumes)])
    # The grid is already isotropic, so no further in-plane downsampling.
    kwargs.setdefault("downsample", 1)
    return plot_3d_volume(fused, (target_spacing,) * 3, title=title,
                          out_html=out_html, open_browser=open_browser, **kwargs)

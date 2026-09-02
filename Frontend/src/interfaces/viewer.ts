/**
 * Types the DICOM viewers consume.
 *
 * The viewers are pure UI: whoever renders them (a page) fetches by study UID,
 * decodes, and hands the pixels down. Nothing in `components/viewer` talks to
 * the API, so the same components serve a stored study, an upload, or a fixture.
 */

import type { Plane } from "./api";

/**
 * Anything a viewer can paint. `ImageData` is what `lib/dicom.ts` produces for
 * an uploaded series; the image/bitmap/canvas variants let a page feed slices
 * carved out of a server-rendered contact sheet instead.
 */
export type SliceImage = ImageData | ImageBitmap | HTMLImageElement | HTMLCanvasElement;

/** One slice of a stack, already decoded. */
export interface ViewerSlice {
  image: SliceImage;
  /** Stable React key; falls back to the position in the stack. */
  id?: string;
  /** Overlay caption, e.g. "Slice 12". */
  label?: string;
}

/**
 * One acquisition run — the unit both viewers show at a time. A study usually
 * contributes one stack per plane, which is what turns the picker into axis
 * tabs, but nothing stops a page passing several series of the same plane.
 */
export interface ViewerStack {
  /** Unique within the array; the series UID is the natural choice. */
  id: string;
  slices: ViewerSlice[];
  /** Acquisition plane, when the page knows it. Badges the stage. */
  plane?: Plane;
  /**
   * Tab text; falls back to `plane`. A study with two series of one plane
   * needs this to tell them apart, which `plane` alone cannot do.
   */
  label?: string;
  /** Sequence details, shown to the right of the tabs. */
  description?: string;
  /**
   * `(slice, row, column)` spacing in millimetres, exactly as
   * `GET /view/{uid}/3d_image_sequence` reports it. Omit it and the 3D viewer
   * falls back to a ratio that keeps the volume from rendering flat — see
   * `buildModelVolume`.
   */
  spacingMm?: [number, number, number];
  /**
   * The model's own input tensor — what `Dicom3DViewer` builds its volume from.
   *
   * The same pixels `slices` holds, in the form a volume needs them. `slices`
   * is the contact sheet cut back apart, which is right for leafing through one
   * plane at a time; this is the whole `(24, 224, 224)` block, which is what
   * Cornerstone can actually render as a volume. Both come out of
   * `sequence_to_tensor` on the server, so the two viewers cannot disagree
   * about what the model was shown.
   */
  volume?: ModelVolume;
  /**
   * Cornerstone image IDs for the study's own DICOM files, in scan order —
   * what `DicomMprViewer` builds its volume from.
   *
   * The counterpart to `volume`, and deliberately a different thing. `volume`
   * is what the model was shown; this is what the scanner wrote, with the
   * geometry still in the headers. That is what makes the reference view able
   * to reslice into all three planes, tie a point in one of them to the same
   * point in the others, and report a length in millimetres that is true.
   *
   * The page fetches the *list*; the loader behind these IDs streams the pixels
   * itself. Expect it to be populated only while the reference view is open —
   * a full series is a large download to make on the chance someone opens it.
   */
  imageIds?: string[];
}

/**
 * One series as the model receives it: `(slices, rows, columns)` float32,
 * values in [0, 1].
 *
 * Exactly the payload of `GET /studies/{uid}/series/{uid}/tensor`, minus the
 * trailing channel axis. The preprocessing that produces it — 24 slices sampled
 * across the series, resized to 224 square, one percentile window for the whole
 * run — throws the acquisition geometry away, so a page that wants the volume
 * drawn to scale has to supply `spacingMm` separately.
 */
export interface ModelVolume {
  shape: [number, number, number];
  data: Float32Array;
}

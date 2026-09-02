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
   * `GET /view/{uid}/3d_image_sequence` reports it.
   */
  spacingMm?: [number, number, number];
  /**
   * Cornerstone image IDs for the raw DICOM slices, in scan order — what
   * `Dicom3DViewer` builds its volume from.
   *
   * Separate from `slices` because the two viewers want different things from
   * the same series. `slices` is the model's view: 24 tiles, 224 square,
   * already windowed, no geometry. Cornerstone wants the files, so it can read
   * the real millimetres out of the headers and reconstruct a volume — which is
   * the whole reason it can reslice and volume-render where a stack of PNGs
   * could only be leafed through. The page still does the fetching (of the
   * *list*); the loader behind these IDs streams the pixels itself.
   */
  imageIds?: string[];
}

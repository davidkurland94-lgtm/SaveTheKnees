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
  /** Acquisition plane, when the page knows it. Labels the axis tab. */
  plane?: Plane;
  /** Series description, shown next to the tabs. */
  label?: string;
  /**
   * `(slice, row, column)` spacing in millimetres, exactly as
   * `GET /view/{uid}/3d_image_sequence` reports it. Drives how deep the 3D
   * stack is drawn; without it the depth falls back to a plausible constant.
   */
  spacingMm?: [number, number, number];
}

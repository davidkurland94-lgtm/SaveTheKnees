/** Turns the API's contact-sheet PNGs into per-slice images for the viewers. */

import type { ViewerSlice } from "@/interfaces";

/**
 * `GET /view/{uid}/2d_image_sequence` and `.../preview.png` both return the 24
 * slices the model sees, tiled into a grid. This cuts that grid back apart.
 *
 * `columns` must match what was asked of the server. It defaults to 1 — a
 * single column makes the tile size exactly the sheet width and the slice count
 * exactly `height / width`, so nothing has to be assumed about how many slices
 * came back, and the grid never ends on a row of blank filler tiles.
 */
export async function splitContactSheet(blob: Blob, columns = 1): Promise<ViewerSlice[]> {
  const sheet = await createImageBitmap(blob);
  try {
    const tile = Math.round(sheet.width / columns);
    if (tile <= 0) return [];
    const rows = Math.round(sheet.height / tile);

    const slices: ViewerSlice[] = [];
    for (let position = 0; position < rows * columns; position += 1) {
      const row = Math.floor(position / columns);
      const column = position % columns;
      slices.push({
        id: `slice-${position}`,
        label: `Slice ${position + 1}`,
        image: await createImageBitmap(sheet, column * tile, row * tile, tile, tile),
      });
    }
    return slices;
  } finally {
    // The full sheet is a native-memory resource; the crops are what we keep.
    sheet.close();
  }
}

/** Releases the bitmaps behind a set of slices. */
export function closeSlices(slices: ViewerSlice[]): void {
  for (const slice of slices) {
    if (slice.image instanceof ImageBitmap) slice.image.close();
  }
}

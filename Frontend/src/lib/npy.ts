/** Minimal reader for the `.npy` the tensor endpoint serves. */

export interface NpyArray {
  /** Dimensions in C order, trailing singleton axes dropped. */
  shape: number[];
  data: Float32Array;
}

/**
 * Parses a NumPy `.npy` buffer.
 *
 * `GET /studies/{uid}/series/{uid}/tensor` returns exactly one shape and dtype
 * — `(24, 224, 224, 1)` float32, the model's own input — so this handles that
 * and refuses anything else rather than pretending to be a general reader.
 */
export function parseNpy(buffer: ArrayBuffer): NpyArray {
  const view = new DataView(buffer);
  const magic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
  for (let i = 0; i < magic.length; i += 1) {
    if (view.getUint8(i) !== magic[i]) throw new Error("Not a .npy file");
  }

  const major = view.getUint8(6);
  const headerLength = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerStart = major === 1 ? 10 : 12;
  const header = new TextDecoder("latin1").decode(
    buffer.slice(headerStart, headerStart + headerLength),
  );

  if (!/'descr': *'[<|]f4'/.test(header)) {
    throw new Error(`Expected float32 .npy, got header: ${header.slice(0, 80)}`);
  }
  if (/'fortran_order': *True/.test(header)) throw new Error("Fortran-ordered .npy unsupported");

  const shapeMatch = header.match(/'shape': *\(([^)]*)\)/);
  if (!shapeMatch) throw new Error("No shape in .npy header");
  const shape = shapeMatch[1]
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map(Number);

  const data = new Float32Array(buffer, headerStart + headerLength);
  // The tensor ships a trailing channel axis of 1; it carries no information.
  while (shape.length > 3 && shape[shape.length - 1] === 1) shape.pop();
  return { shape, data };
}

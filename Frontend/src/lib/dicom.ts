/** Browser-side DICOM decoding, extracted from App.tsx. */

import dicomParser from "dicom-parser";

import type { ParsedScan, ParsedSeries, ScanMetadata } from "@/interfaces";

// DICOM tags, lower-case hex as dicom-parser expects them.
const TAG = {
  rows: "x00280010",
  columns: "x00280011",
  bitsAllocated: "x00280100",
  pixelRepresentation: "x00280103",
  windowCenter: "x00281050",
  windowWidth: "x00281051",
  pixelData: "x7fe00010",
  modality: "x00080060",
  seriesDescription: "x0008103e",
  instanceNumber: "x00200013",
  patientName: "x00100010",
  patientAge: "x00101010",
  patientSex: "x00100040",
  studyDate: "x00080020",
  accessionNumber: "x00080050",
  studyUid: "x0020000d",
  seriesUid: "x0020000e",
  bodyPart: "x00180015",
} as const;

const EMPTY_METADATA: ScanMetadata = {
  patientName: "Unknown Patient",
  patientAge: "",
  patientSex: "",
  studyDate: "Unknown",
  accessionNumber: "",
  studyUid: "",
  seriesUid: "",
  bodyPart: "",
};

/** DICOM dates arrive as `YYYYMMDD`. */
export function formatStudyDate(raw: string): string {
  if (raw.length === 8) return `${raw.slice(0, 4)}-${raw.slice(4, 6)}-${raw.slice(6, 8)}`;
  return raw || "Unknown";
}

/** Multi-valued numeric tags are backslash-delimited; take the first. */
function firstNumber(raw: string | undefined): number | undefined {
  if (!raw) return undefined;
  const value = Number.parseFloat(raw.split("\\")[0]);
  return Number.isFinite(value) ? value : undefined;
}

/**
 * Decodes one file into greyscale `ImageData`.
 *
 * Returns `null` for anything that is not a parseable single-frame DICOM with
 * pixel data, so a stray file in a dropped folder cannot break the batch.
 */
export async function parseDicomFile(file: File): Promise<ParsedScan | null> {
  return (await decodeFile(file))?.scan ?? null;
}

/** One decode pass, yielding both the pixels and the study metadata. */
async function decodeFile(
  file: File,
): Promise<{ scan: ParsedScan | null; metadata: ScanMetadata } | null> {
  let dataset: ReturnType<typeof dicomParser.parseDicom>;
  let buffer: ArrayBuffer;
  try {
    buffer = await file.arrayBuffer();
    dataset = dicomParser.parseDicom(new Uint8Array(buffer));
  } catch {
    return null;
  }

  const metadata = readMetadata(dataset);

  try {
    const pixelDataElement = dataset.elements[TAG.pixelData];
    if (!pixelDataElement) return { scan: null, metadata };

    const rows = dataset.uint16(TAG.rows) ?? 512;
    const cols = dataset.uint16(TAG.columns) ?? 512;
    const bitsAllocated = dataset.uint16(TAG.bitsAllocated) ?? 16;
    const isSigned = dataset.uint16(TAG.pixelRepresentation) === 1;

    // Slice the pixel data out of the original buffer. The typed-array views
    // below need the byte offset to be even, so copy when it is not.
    const start = pixelDataElement.dataOffset;
    const end = start + pixelDataElement.length;
    const pixelBuffer = buffer.slice(start, end);

    let pixels: ArrayLike<number>;
    let minValue = Infinity;
    let maxValue = -Infinity;

    if (bitsAllocated === 16) {
      const view = isSigned ? new Int16Array(pixelBuffer) : new Uint16Array(pixelBuffer);
      pixels = view;
      for (let i = 0; i < view.length; i += 1) {
        if (view[i] < minValue) minValue = view[i];
        if (view[i] > maxValue) maxValue = view[i];
      }
    } else {
      pixels = new Uint8Array(pixelBuffer);
      minValue = 0;
      maxValue = 255;
    }

    // Fall back to the full intensity range when no window is stored.
    const center = firstNumber(dataset.string(TAG.windowCenter)) ?? (minValue + maxValue) / 2;
    const width = firstNumber(dataset.string(TAG.windowWidth)) ?? (maxValue - minValue || 1);
    const low = center - width / 2;

    const imageData = new ImageData(cols, rows);
    for (let i = 0; i < rows * cols; i += 1) {
      const value = pixels[i] ?? 0;
      const grey = Math.max(0, Math.min(255, Math.round(((value - low) / width) * 255)));
      const offset = i * 4;
      imageData.data[offset] = grey;
      imageData.data[offset + 1] = grey;
      imageData.data[offset + 2] = grey;
      imageData.data[offset + 3] = 255;
    }

    const instanceRaw = dataset.string(TAG.instanceNumber);
    const instanceNumber = instanceRaw ? Number.parseInt(instanceRaw, 10) : Number.NaN;
    const seriesDescription = dataset.string(TAG.seriesDescription) ?? "";

    return {
      metadata,
      scan: {
        id: file.name,
        label: seriesDescription || file.name.replace(/\.dcm$/i, ""),
        modality: dataset.string(TAG.modality) ?? "",
        slice: instanceRaw ? `Slice ${instanceRaw}` : file.name,
        instanceNumber: Number.isFinite(instanceNumber) ? instanceNumber : null,
        imageData,
        rows,
        cols,
        seriesDescription,
      },
    };
  } catch {
    // Metadata survives even when the pixel data cannot be decoded.
    return { scan: null, metadata };
  }
}

/** Reads study-level metadata from a dataset-bearing file. */
function readMetadata(dataset: ReturnType<typeof dicomParser.parseDicom>): ScanMetadata {
  return {
    patientName:
      (dataset.string(TAG.patientName) ?? "").replace(/\^/g, " ").trim() || "Unknown Patient",
    patientAge: dataset.string(TAG.patientAge) ?? "",
    patientSex: dataset.string(TAG.patientSex) ?? "",
    studyDate: formatStudyDate(dataset.string(TAG.studyDate) ?? ""),
    accessionNumber: dataset.string(TAG.accessionNumber) ?? "",
    studyUid: dataset.string(TAG.studyUid) ?? "",
    seriesUid: dataset.string(TAG.seriesUid) ?? "",
    bodyPart: dataset.string(TAG.bodyPart) ?? "",
  };
}

/**
 * Parses a dropped batch: decodes every file, orders slices by instance number,
 * and lifts study metadata off the first file that yields any.
 */
export async function parseDicomSeries(files: File[]): Promise<ParsedSeries> {
  const results = await Promise.all(files.map(decodeFile));
  const scans = results
    .map((result) => result?.scan)
    .filter((scan): scan is ParsedScan => scan != null);

  scans.sort((a, b) => {
    if (a.instanceNumber !== null && b.instanceNumber !== null) {
      return a.instanceNumber - b.instanceNumber;
    }
    return a.id.localeCompare(b.id, undefined, { numeric: true });
  });

  // Take metadata from the first file that parsed at all — a folder drop may
  // lead with something that is not a DICOM.
  const metadata = results.find((result) => result !== null)?.metadata ?? EMPTY_METADATA;

  return { scans, metadata, failedCount: files.length - scans.length };
}

/** Keeps only the `.dcm` files out of a drop or folder pick. */
export function filterDicomFiles(files: File[]): File[] {
  return files.filter((file) => file.name.toLowerCase().endsWith(".dcm"));
}

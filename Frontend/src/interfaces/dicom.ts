/** Types for DICOM files parsed in the browser by `lib/dicom.ts`. */

/** One decoded DICOM slice, ready to paint onto a canvas. */
export interface ParsedScan {
  id: string;
  /** Series description, falling back to the file name. */
  label: string;
  /** Modality, e.g. "MR". */
  modality: string;
  /** Human-readable slice marker, e.g. "Slice 12". */
  slice: string;
  /** Instance number used for ordering; `null` when the tag is absent. */
  instanceNumber: number | null;
  imageData: ImageData;
  rows: number;
  cols: number;
  seriesDescription: string;
}

/** Study-level metadata lifted from the first slice of an uploaded series. */
export interface ScanMetadata {
  patientName: string;
  patientAge: string;
  patientSex: string;
  studyDate: string;
  accessionNumber: string;
  studyUid: string;
  seriesUid: string;
  bodyPart: string;
}

/** Result of parsing a batch of uploaded files. */
export interface ParsedSeries {
  scans: ParsedScan[];
  metadata: ScanMetadata;
  /** Files that could not be decoded — surfaced to the user rather than dropped. */
  failedCount: number;
}

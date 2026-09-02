/**
 * Response shapes of the Save the Knees FastAPI backend.
 *
 * Mirrors the OpenAPI document served at `<API_BASE_URL>/openapi.json`. Most of
 * the dataset routes are declared there with an empty response schema, so the
 * types below were derived from live responses.
 */

// ─── Labels ───────────────────────────────────────────────────────────────────

/** The twelve findings the models score, in the order the backend returns them. */
export const LABEL_NAMES = [
  "ACL",
  "MCL",
  "Medial Meniscus",
  "Lateral Meniscus",
  "Medial OA",
  "Lateral OA",
  "PF OA",
  "Effusion",
  "Synovitis",
  "Baker's",
  "Contusion",
  "Fracture",
] as const;

export type LabelName = (typeof LABEL_NAMES)[number];

/** A probability per finding. Keyed loosely: the backend owns the label set. */
export type LabelScores = Record<string, number>;

/** Ground-truth annotations are 0/1 flags rather than probabilities. */
export type GoldenLabels = Record<string, 0 | 1>;

export type Verdict = "YES" | "NO" | "UNK";

export interface LabelAssessment {
  probability: number;
  confidence: number;
  verdict: Verdict;
}

// ─── Service metadata ─────────────────────────────────────────────────────────

/** `GET /` */
export interface ServiceInfo {
  service: string;
  labels: string[];
  docs: string;
}

/** `GET /health` - reports whether the dataset is visible to the server. */
export interface HealthResponse {
  status: string;
  dataset: "ready" | "unavailable";
  n_studies?: number;
  detail?: string;
}

// ─── Studies ──────────────────────────────────────────────────────────────────

export type Plane = "Sagittal" | "Coronal" | "Axial";

/** `GET /studies/{study_uid}/series` entries, also inlined in the study detail. */
export interface Series {
  study_uid: string;
  series_uid: string;
  plane: Plane;
  axis: "X" | "Y" | "Z";
  fluid_sensitive: boolean;
  fat_suppression: boolean;
  n_slices: number;
  available: boolean;
}

/** One row of `GET /studies` (the paginated full-corpus listing). */
export interface StudyListEntry {
  study_uid: string;
  golden: boolean;
  has_report: boolean;
  /** "upload" for studies added through the app; absent on older responses. */
  source?: "upload" | "dataset";
}

/** `GET /studies` */
export interface StudiesListResponse {
  total: number;
  offset: number;
  limit: number;
  studies: StudyListEntry[];
}

/** One row of `GET /studies/golden`. */
export interface GoldenStudy {
  study_uid: string;
  labels: GoldenLabels;
}

/** `GET /studies/golden` — the 58 hand-labelled studies. */
export interface GoldenStudiesResponse {
  count: number;
  labels: string[];
  studies: GoldenStudy[];
}

/** One series of an uploaded study, as `POST /upload/study` reports it. */
export interface UploadedSeries extends Series {
  /** SeriesDescription off the scanner, when it wrote one. */
  description?: string;
}

/**
 * `POST /upload/study` — a stored study, built from a folder of DICOM.
 *
 * `predicted_labels` is a model's read, not a radiologist's: it is kept under
 * its own key, beside the name of the model that produced it, so it can never
 * be mistaken for the hand-labelled ground truth. `has_report` starts false by
 * design — writing it is the doctor's job.
 */
export interface UploadedStudy extends Omit<StudyDetail, "series"> {
  series: UploadedSeries[];
  source: "upload";
  created_at: string;
  predicted_labels: Record<string, number> | null;
  label_model: string | null;
  /** DICOM files stored, and files in the drop that were not DICOM. */
  n_files: number;
  skipped: number;
}

/** `GET /studies/{study_uid}` */
export interface StudyDetail {
  study_uid: string;
  n_series: number;
  planes: Partial<Record<Plane, number>>;
  is_golden: boolean;
  has_report: boolean;
  golden_labels: GoldenLabels | null;
  series: Series[];
}

/** `GET /studies/{study_uid}/series` */
export interface SeriesListResponse {
  study_uid: string;
  count: number;
  series: Series[];
}

/**
 * `GET /studies/{uid}/series/{uid}/instances` — the raw DICOM file names.
 *
 * Ordered along the scan axis by the server. A browser-side DICOM reader reads
 * geometry from the first, middle and last instance only and takes the rest of
 * the order on trust, so this array is what puts the volume the right way up.
 */
export interface SeriesInstancesResponse {
  study_uid: string;
  series_uid: string;
  count: number;
  instances: string[];
}

/** `GET /studies/{study_uid}/labels` — the pilkwang report-derived labels. */
export interface StudyLabelsResponse {
  study_uid: string;
  source: string;
  labels: Record<string, LabelAssessment>;
}

/** `GET /studies/{study_uid}/labels/{label}` */
export interface LabelResponse extends LabelAssessment {
  study_uid: string;
  source: string;
  label: string;
}

export type ReportLang = "original" | "en";

/** `GET /studies/{study_uid}/report` */
export interface ReportResponse {
  study_uid: string;
  language: string;
  report: string;
}

// ─── Inference ────────────────────────────────────────────────────────────────

export type ModelName = "sagittal" | "multiplane" | "fusion";

/** `GET /studies/{study_uid}/predict` — one of our models run over a stored study. */
export interface StudyPredictionResponse {
  study_uid: string;
  model: ModelName;
  predictions: LabelScores;
}

/** `POST /predict` — an uploaded DICOM series scored by the image model. */
export interface PredictionResponse {
  model_status: string;
  n_slices_received: number;
  predictions: LabelScores;
}

/**
 * `GET /predict/report/terms` — the medical dictionary the report model counts.
 *
 * Data, not code: the list is a file on the server and a deployment without it
 * answers with an empty array rather than an error.
 */
export interface ReportTermsResponse {
  count: number;
  terms: string[];
}

/** `POST /predict/report` — free-text report scored by the report model. */
export interface ReportPredictionResponse {
  model_status: string;
  predictions: LabelScores;
}

// ─── Benchmark report ─────────────────────────────────────────────────────────

/**
 * One row of `GET /report/table`: per-label AUC for every reader.
 *
 * The trailing "MEAN (defined)" row carries a null `positives`.
 */
export interface ReportTableRow {
  label: string;
  positives: number | null;
  llm_v4_blend: number;
  llm_full: number;
  llm_v2: number;
  pilkwang_v2: number;
  image_model: number;
  image_multiplane: number;
  report_model: number;
  report_bagged: number;
  fusion_model: number;
}

export interface ReportTableResponse {
  rows: ReportTableRow[];
}

/** One row of `GET /report/verdicts`: where a written report and the images disagree. */
export interface VerdictRow {
  StudyInstanceUID: string;
  report_says: string;
  images_say: string;
  "missed?": string;
  quality_score: number;
}

export interface VerdictsResponse {
  rows: VerdictRow[];
}

/** `POST /upload/{study_uid}/image_sequence` */
export interface UploadSequenceResponse {
  study_uid: string;
  series_uid: string;
  n_slices: number;
  stored_at: string;
  files: string[];
}

/** A user-written report stored by the backend (NOT a dataset report). */
export interface StoredReportRecord {
  study_uid: string;
  text: string;
  author: string;
  created_at: string;
  updated_at: string;
}

/** `GET /view/{study_uid}/information` */
export interface StudyInformation extends StudyDetail {
  report: ReportResponse | null;
  pilkwang_labels: Record<string, LabelAssessment> | null;
  my_report: StoredReportRecord | null;
}

/** `GET /view/{study_uid}/3d_image_sequence` - raw mesh for three.js/plotly. */
export interface Mesh3DResponse {
  study_uid: string;
  series_uid: string;
  plane: Plane;
  spacing_mm: number[];
  n_vertices: number;
  n_faces: number;
  vertices: number[][];
  faces: number[][];
}

/** `GET /models/{name}/summary` */
export interface ModelSummaryResponse {
  name: string;
  parameters: number;
  summary: string;
}

export type CompareWhich = "image" | "report" | "showdown";

/** `GET /report/compare/{which}` — shape varies by comparison, so kept loose. */
export interface CompareResponse {
  rows?: Array<Record<string, unknown>>;
  [key: string]: unknown;
}

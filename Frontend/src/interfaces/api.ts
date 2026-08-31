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

/** `GET /health` */
export interface HealthResponse {
  status: string;
  model: string;
  report_model: string;
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

export type CompareWhich = "image" | "report" | "showdown";

/** `GET /report/compare/{which}` — shape varies by comparison, so kept loose. */
export interface CompareResponse {
  rows?: Array<Record<string, unknown>>;
  [key: string]: unknown;
}

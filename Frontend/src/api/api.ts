/**
 * Every endpoint of the Save the Knees backend, one function per route.
 *
 * Source of truth: the OpenAPI document at `<API_BASE_URL>/openapi.json`
 * (Swagger UI at `<API_BASE_URL>/docs`). Nothing here is mocked.
 */

import type {
  CompareResponse,
  CompareWhich,
  GoldenStudiesResponse,
  HealthResponse,
  LabelResponse,
  ModelName,
  Plane,
  PredictionResponse,
  ReportLang,
  ReportPredictionResponse,
  ReportResponse,
  ReportTableResponse,
  SeriesListResponse,
  ServiceInfo,
  StudyDetail,
  StudyLabelsResponse,
  StudyPredictionResponse,
  VerdictsResponse,
} from "@/interfaces";
import { buildUrl, postJson, requestBlob, requestJson } from "./client";

const encode = encodeURIComponent;

// ─── Service ──────────────────────────────────────────────────────────────────

/** `GET /` — service name, the label set, and where the docs live. */
export function getServiceInfo(signal?: AbortSignal): Promise<ServiceInfo> {
  return requestJson<ServiceInfo>("/", { signal });
}

/** `GET /health` — readiness probe; reports whether both models are loaded. */
export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/health", { signal, timeoutMs: 15_000 });
}

// ─── Dataset ──────────────────────────────────────────────────────────────────

/** `GET /studies/golden` — the 58 hand-labelled studies, the only ground truth. */
export function getGoldenStudies(signal?: AbortSignal): Promise<GoldenStudiesResponse> {
  return requestJson<GoldenStudiesResponse>("/studies/golden", { signal });
}

/** `GET /studies/{study_uid}` — series inventory, planes, and golden labels. */
export function getStudy(studyUid: string, signal?: AbortSignal): Promise<StudyDetail> {
  return requestJson<StudyDetail>(`/studies/${encode(studyUid)}`, { signal });
}

/** `GET /studies/{study_uid}/series` — every sequence, optionally one plane. */
export function getSeriesList(
  studyUid: string,
  plane?: Plane,
  signal?: AbortSignal,
): Promise<SeriesListResponse> {
  return requestJson<SeriesListResponse>(`/studies/${encode(studyUid)}/series`, {
    query: { plane },
    signal,
  });
}

/** `GET /studies/{study_uid}/labels` — all twelve pilkwang labels. */
export function getStudyLabels(
  studyUid: string,
  signal?: AbortSignal,
): Promise<StudyLabelsResponse> {
  return requestJson<StudyLabelsResponse>(`/studies/${encode(studyUid)}/labels`, { signal });
}

/** `GET /studies/{study_uid}/labels/{label}` — accepts a slug or the exact name. */
export function getStudyLabel(
  studyUid: string,
  label: string,
  signal?: AbortSignal,
): Promise<LabelResponse> {
  return requestJson<LabelResponse>(`/studies/${encode(studyUid)}/labels/${encode(label)}`, {
    signal,
  });
}

/**
 * `GET /studies/{study_uid}/report` — the radiology report.
 *
 * The corpus is mixed-language; `lang: "en"` serves the machine translation,
 * `"original"` (the default) serves it exactly as the dataset ships it.
 */
export function getStudyReport(
  studyUid: string,
  lang: ReportLang = "original",
  signal?: AbortSignal,
): Promise<ReportResponse> {
  return requestJson<ReportResponse>(`/studies/${encode(studyUid)}/report`, {
    query: { lang },
    signal,
  });
}

/** `GET /studies/{study_uid}/predict` — run one of our models over a stored study. */
export function predictStudy(
  studyUid: string,
  model: ModelName = "fusion",
  signal?: AbortSignal,
): Promise<StudyPredictionResponse> {
  return requestJson<StudyPredictionResponse>(`/studies/${encode(studyUid)}/predict`, {
    query: { model },
    signal,
  });
}

/** `GET /studies/{uid}/series/{uid}/tensor` — model-ready (24,224,224,1) .npy. */
export function getSeriesTensor(
  studyUid: string,
  seriesUid: string,
  signal?: AbortSignal,
): Promise<Blob> {
  return requestBlob(`/studies/${encode(studyUid)}/series/${encode(seriesUid)}/tensor`, { signal });
}

/**
 * URL of `GET /studies/{uid}/series/{uid}/preview.png` — a contact sheet of the
 * 24 sampled slices. Returned as a URL rather than a blob so it can go straight
 * into an `<img src>` and use the browser cache.
 */
export function seriesPreviewUrl(studyUid: string, seriesUid: string, columns = 6): string {
  return buildUrl(`/studies/${encode(studyUid)}/series/${encode(seriesUid)}/preview.png`, {
    columns,
  });
}

// ─── Inference on uploads ─────────────────────────────────────────────────────

/**
 * `POST /predict` — score an uploaded DICOM series.
 *
 * Every `.dcm` of ONE series (one plane of one study) must be sent: intensity is
 * normalised with a single percentile window across the whole stack, so a lone
 * slice is not enough.
 */
export function predictSeries(files: File[], signal?: AbortSignal): Promise<PredictionResponse> {
  const form = new FormData();
  for (const file of files) form.append("files", file, file.name);
  return requestJson<PredictionResponse>("/predict", { method: "POST", body: form, signal });
}

/** `POST /predict/report` — English report text in, twelve probabilities out. */
export function predictReport(
  text: string,
  signal?: AbortSignal,
): Promise<ReportPredictionResponse> {
  return postJson<ReportPredictionResponse>("/predict/report", { text }, { signal });
}

// ─── Benchmark report ─────────────────────────────────────────────────────────

/** `GET /report/table` — every reader scored against the 58 gold studies. */
export function getReportTable(signal?: AbortSignal): Promise<ReportTableResponse> {
  return requestJson<ReportTableResponse>("/report/table", { signal });
}

/** `GET /report/compare/{which}` — image, report, or showdown comparison. */
export function getReportCompare(
  which: CompareWhich,
  signal?: AbortSignal,
): Promise<CompareResponse> {
  return requestJson<CompareResponse>(`/report/compare/${encode(which)}`, { signal });
}

/** `GET /report/verdicts` — worst reports first: what was written vs what the images say. */
export function getReportVerdicts(top = 20, signal?: AbortSignal): Promise<VerdictsResponse> {
  return requestJson<VerdictsResponse>("/report/verdicts", { query: { top }, signal });
}

/** `GET /models/{name}/summary` — Keras layer table of a trained model, as text. */
export function getModelSummary(name: string, signal?: AbortSignal): Promise<unknown> {
  return requestJson<unknown>(`/models/${encode(name)}/summary`, { signal });
}

/** Namespaced handle for callers that prefer `api.getStudy(...)`. */
export const api = {
  getServiceInfo,
  getHealth,
  getGoldenStudies,
  getStudy,
  getSeriesList,
  getStudyLabels,
  getStudyLabel,
  getStudyReport,
  predictStudy,
  getSeriesTensor,
  seriesPreviewUrl,
  predictSeries,
  predictReport,
  getReportTable,
  getReportCompare,
  getReportVerdicts,
  getModelSummary,
};

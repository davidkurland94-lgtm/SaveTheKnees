/** View-model types owned by the UI, not by the backend. */

import type { Verdict } from "./api";

/** Which screen the shell is showing. */
export type Route =
  | { name: "home" }
  | { name: "processing" }
  | { name: "upload-result" }
  | { name: "study"; studyUid: string }
  | { name: "benchmark" };

export type UploadMode = "single" | "sequence" | "folder";

/** Bucket used to colour a probability. Derived, never sent by the backend. */
export type Severity = "high" | "moderate" | "low";

/**
 * A single finding rendered in a prediction list — the shape every panel in the
 * app consumes, whatever endpoint produced the numbers.
 */
export interface Finding {
  label: string;
  probability: number;
  severity: Severity;
  /** Ground-truth flag when the study is one of the 58 golden ones. */
  truth?: 0 | 1;
  /** Report-derived verdict, when labels were loaded for the study. */
  verdict?: Verdict;
  confidence?: number;
}

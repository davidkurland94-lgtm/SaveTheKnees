/** View-model types owned by the UI, not by the backend. */

import type { Verdict } from "./api";

/**
 * Index of the step the upload is on.
 *
 * Only two, because only two are observable from here: the browser gathering
 * the folder, and one request in which the server stores the study and scores
 * it. A third step would be a timer pretending to be progress.
 */
export type ProcessingStep = 0 | 1;

export interface UploadState {
  /** UID of the study the last run created; null until one succeeds. */
  studyUid: string | null;
  /** Files accepted by the most recent run; 0 before the first one. */
  fileCount: number;
  step: ProcessingStep;
  /** Why the most recent run failed, if it did. */
  error: string | null;
  dismissError: () => void;
  start: (files: File[]) => void;
}

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

/**
 * A study's on-screen identity, derived from its UID by `lib/patients.ts`.
 *
 * Entirely synthetic. The corpus is de-identified and carries no demographics,
 * so these fields exist to make the showcase legible — never to be read as
 * something the dataset knows about a real person.
 */
export interface PatientIdentity {
  name: string;
  /** Two letters for the avatar. */
  initials: string;
  /** Medical record number, digits only; views add the `MRN` prefix. */
  mrn: string;
  age: number;
  sex: "F" | "M";
  /** Tailwind background/text pair for the avatar. */
  tone: string;
}

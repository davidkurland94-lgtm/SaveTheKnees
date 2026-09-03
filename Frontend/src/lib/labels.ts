/** Helpers for turning backend label maps into something the UI can render. */

import { LABEL_NAMES } from "@/interfaces";
import type { Finding, GoldenLabels, LabelScores, Severity, Verdict } from "@/interfaces";

/** Probability bands. Unchanged; what they are *called* is what changed. */
const HIGH_THRESHOLD = 0.7;
const MODERATE_THRESHOLD = 0.4;

export function severityOf(probability: number): Severity {
  if (probability >= HIGH_THRESHOLD) return "high";
  if (probability >= MODERATE_THRESHOLD) return "moderate";
  return "low";
}

/**
 * What each band means to the person reading it, which is not what it means to
 * the model.
 *
 * The panel used to print the probability and leave the reader to do this
 * translation in their head, every finding, every study. It is the same
 * translation every time and the model is better placed to make it, so it makes
 * it here instead. A number also implies a precision the model has not earned:
 * 0.62 and 0.68 are not a distinction anyone should act on.
 *
 * The middle band is the important one, and it is not "a middling chance of
 * injury". It is the model declining to answer — and that is precisely the case
 * that needs a human, because the other two it has already answered.
 */
export const SEVERITY_LABEL: Record<Severity, string> = {
  high: "Injured",
  moderate: "Needs review",
  low: "Healthy",
};

/**
 * Splits findings into the three bands, each keeping the order it came in.
 *
 * The panel reads them out of order — flagged first, whatever the model thought
 * of the rest — so the grouping has to happen before the rendering, not inside
 * a sort.
 */
export function groupBySeverity(findings: Finding[]): Record<Severity, Finding[]> {
  const groups: Record<Severity, Finding[]> = { high: [], moderate: [], low: [] };
  for (const finding of findings) groups[finding.severity].push(finding);
  return groups;
}

/** `GET /studies/{uid}/labels/{label}` accepts these slugs as well as exact names. */
export function labelSlug(label: string): string {
  return label
    .toLowerCase()
    .replace(/'/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

/**
 * Orders a score map by the canonical label order, then by probability for any
 * label the backend added that we do not know about yet.
 */
function orderedEntries(scores: LabelScores): Array<[string, number]> {
  const known = LABEL_NAMES.filter((name) => name in scores).map(
    (name) => [name as string, scores[name]] as [string, number],
  );
  const extra = Object.entries(scores)
    .filter(([name]) => !(LABEL_NAMES as readonly string[]).includes(name))
    .sort((a, b) => b[1] - a[1]);
  return [...known, ...extra];
}

export interface ToFindingsOptions {
  truth?: GoldenLabels | null;
  verdicts?: Record<string, { verdict: Verdict; confidence: number }> | null;
  /** Sort by probability instead of the canonical label order. */
  sortByProbability?: boolean;
}

/** Normalises any score map into the `Finding[]` every prediction panel renders. */
export function toFindings(scores: LabelScores, options: ToFindingsOptions = {}): Finding[] {
  const { truth, verdicts, sortByProbability = false } = options;

  const findings = orderedEntries(scores).map(([label, probability]) => ({
    label,
    probability,
    severity: severityOf(probability),
    truth: truth?.[label],
    verdict: verdicts?.[label]?.verdict,
    confidence: verdicts?.[label]?.confidence,
  }));

  return sortByProbability ? findings.sort((a, b) => b.probability - a.probability) : findings;
}

/** Names of the positive findings in a ground-truth map. */
export function positiveLabels(labels: GoldenLabels | null | undefined): string[] {
  if (!labels) return [];
  return Object.entries(labels)
    .filter(([, value]) => value === 1)
    .map(([name]) => name);
}

export const VERDICT_TONE: Record<Verdict, string> = {
  YES: "bg-red-50 text-red-600 border-red-100",
  NO: "bg-emerald-50 text-emerald-700 border-emerald-100",
  UNK: "bg-[#f5f3ff] text-[#6d5da8] border-[#e9e4f8]",
};

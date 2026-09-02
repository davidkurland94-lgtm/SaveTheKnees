/**
 * Reading a stored report the way the models read it.
 *
 * A doctor writes in whichever language they think in; the report model and the
 * fusion model's text branch were both trained on English, and the vectorizer
 * behind them was fitted on `data/meta/reports_en.csv`. The server settles that
 * when the report is saved, so the record carries both renderings.
 */

import type { StoredReportRecord } from "@/interfaces";

/**
 * What to send to `POST /predict/report` for a stored report.
 *
 * Falls back to the text as written, which covers two cases that look the same
 * from here: a record stored before the server translated on save, and a
 * deployment whose translator is unavailable — in both, the server left the
 * English empty rather than claiming a translation it did not make.
 */
export function englishReport(record: StoredReportRecord): string {
  return record.report_en?.trim() || record.text;
}

/** Whether the stored English is a translation rather than the text itself. */
export function wasTranslated(record: StoredReportRecord): boolean {
  return Boolean(record.report_en?.trim()) && record.report_en?.trim() !== record.text.trim();
}

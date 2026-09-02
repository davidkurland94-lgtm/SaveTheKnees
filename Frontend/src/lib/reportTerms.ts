/**
 * Finding the report model's own vocabulary inside a report.
 *
 * `GET /predict/report/terms` serves the dictionary the report branch counts as
 * features — the same list `term_features` builds its log1p counts from — so
 * marking those terms in the text shows which of them this particular report
 * gave the model to work with.
 *
 * What it does not show is how much each one moved the answer. The API returns
 * the vocabulary, not per-term weights, so a mark here means "the model counted
 * this", never "this is why the number came out that way".
 */

/** A run of report text, flagged if it is one of the model's terms. */
export interface TextSegment {
  text: string;
  term: boolean;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * One alternation over the whole dictionary rather than a pass per term: a
 * report is scanned once however long the list gets.
 *
 * Longest first, because the alternation is ordered — without it "meniscus"
 * would match inside "medial meniscus tear" and the longer term would never be
 * seen. The lookarounds mirror the `(?<!\w)…(?!\w)` the Python side uses, so a
 * term is matched on the same boundaries the model counted it on.
 */
function buildPattern(terms: string[]): RegExp | null {
  const cleaned = [...new Set(terms.map((term) => term.trim()).filter(Boolean))].sort(
    (a, b) => b.length - a.length,
  );
  if (cleaned.length === 0) return null;

  try {
    return new RegExp(`(?<!\\w)(?:${cleaned.map(escapeRegExp).join("|")})(?!\\w)`, "gi");
  } catch {
    // Lookbehind is the one piece here an older engine may not have. Losing the
    // marks is a fair trade for the report still rendering.
    return null;
  }
}

/**
 * Splits `text` into plain and term segments, in order, losing nothing —
 * concatenating the segments always returns the original text.
 */
export function markTerms(text: string, terms: string[]): TextSegment[] {
  const pattern = buildPattern(terms);
  if (!pattern) return [{ text, term: false }];

  const segments: TextSegment[] = [];
  let cursor = 0;

  for (const match of text.matchAll(pattern)) {
    const start = match.index;
    if (start > cursor) segments.push({ text: text.slice(cursor, start), term: false });
    segments.push({ text: match[0], term: true });
    cursor = start + match[0].length;
  }
  if (cursor < text.length) segments.push({ text: text.slice(cursor), term: false });

  return segments;
}

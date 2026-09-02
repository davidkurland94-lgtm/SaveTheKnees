/** Small formatting helpers shared across the UI. */

/** `0.8234` → `82%`. */
export function percent(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function pluralize(count: number, singular: string, plural = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

/** Joins truthy parts with a separator — for metadata lines with optional fields. */
export function joinParts(parts: Array<string | null | undefined | false>, sep = " · "): string {
  return parts.filter(Boolean).join(sep);
}

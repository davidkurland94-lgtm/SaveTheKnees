/** Joins conditional class names. Falsy entries are dropped. */
export function cn(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(" ");
}

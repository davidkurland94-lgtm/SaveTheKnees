/**
 * Decoded slices, kept across view switches and page navigations.
 *
 * The server rebuilds each contact sheet from raw DICOM on every request, which
 * takes seconds, so a study's slices are held at module scope rather than in a
 * component: leaving a study and coming back reuses them.
 */

import type { ViewerSlice } from "@/interfaces";

/** Studies held at once. Each is 24 tiles of 224², so roughly 5 MB apiece. */
const MAX_ENTRIES = 12;

/**
 * Promises rather than results, so two viewers asking for the same series at
 * the same moment share one request instead of racing for it.
 */
const entries = new Map<string, Promise<ViewerSlice[]>>();

/**
 * Returns the cached slices for `key`, running `load` only on a miss.
 *
 * `load` deliberately gets no abort signal: a request cancelled halfway leaves
 * nothing to cache, and the point here is that the next view switch is instant.
 * Callers that unmount mid-flight simply ignore the result.
 */
export function cachedSlices(
  key: string,
  load: () => Promise<ViewerSlice[]>,
): Promise<ViewerSlice[]> {
  const hit = entries.get(key);
  if (hit) {
    // Re-insert so the Map's insertion order doubles as a recency list.
    entries.delete(key);
    entries.set(key, hit);
    return hit;
  }

  const pending = load().catch((cause: unknown) => {
    // A failure must not be cached, or a retry could never succeed.
    entries.delete(key);
    throw cause;
  });
  entries.set(key, pending);

  // Evicted bitmaps are dropped, not closed: a stack that is still on screen
  // would go blank, and the GC reclaims them once nothing references them.
  while (entries.size > MAX_ENTRIES) {
    const oldest = entries.keys().next();
    if (oldest.done) break;
    entries.delete(oldest.value);
  }

  return pending;
}

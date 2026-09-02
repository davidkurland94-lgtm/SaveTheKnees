/**
 * Decoded slices, kept across view switches and page navigations.
 *
 * The server rebuilds each contact sheet from raw DICOM on every request, which
 * takes seconds, so a study's slices are held at module scope rather than in a
 * component: leaving a study and coming back reuses them.
 */

import type { ViewerSlice } from "@/interfaces";
import { createPromiseCache } from "./promiseCache";

/** Studies held at once. Each is 24 tiles of 224², so roughly 5 MB apiece. */
const MAX_ENTRIES = 12;

/**
 * Returns the cached slices for a key, running the loader only on a miss.
 *
 * The loader deliberately gets no abort signal: a request cancelled halfway
 * leaves nothing to cache, and the point here is that the next view switch is
 * instant. Callers that unmount mid-flight simply ignore the result.
 *
 * Evicted bitmaps are dropped, not closed: a stack that is still on screen
 * would go blank, and the GC reclaims them once nothing references them.
 */
export const cachedSlices = createPromiseCache<ViewerSlice[]>(MAX_ENTRIES);

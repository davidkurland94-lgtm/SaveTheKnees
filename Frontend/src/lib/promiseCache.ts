/**
 * A keyed cache of requests, held at module scope so it outlives the components
 * that read it.
 *
 * It stores promises rather than results, which buys two things at once: two
 * callers asking for the same key in the same tick share one request instead of
 * racing for it, and a caller that arrives while a request is still in flight
 * waits on that one rather than starting a second.
 */
export function createPromiseCache<T>(maxEntries: number) {
  const entries = new Map<string, Promise<T>>();

  return function cached(key: string, load: () => Promise<T>): Promise<T> {
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

    while (entries.size > maxEntries) {
      const oldest = entries.keys().next();
      if (oldest.done) break;
      entries.delete(oldest.value);
    }

    return pending;
  };
}

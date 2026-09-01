/** Data-fetching hooks used by the pages. */

import { useCallback, useEffect, useRef, useState } from "react";

import { describeError } from "@/api";

export interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  /** Re-runs the request; useful behind a "Retry" button. */
  reload: () => void;
}

/**
 * Runs `fn` on mount and whenever `deps` change, aborting the in-flight request
 * when the inputs change or the component unmounts — so a slow response can
 * never overwrite the state of a newer one.
 */
export function useAsync<T>(
  fn: (signal: AbortSignal) => Promise<T>,
  deps: React.DependencyList,
): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  // Keep the latest fn in a ref so callers can pass an inline arrow without
  // re-running the effect on every render.
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    const controller = new AbortController();
    let active = true;

    setLoading(true);
    setError(null);

    fnRef
      .current(controller.signal)
      .then((result) => {
        if (active) setData(result);
      })
      .catch((cause: unknown) => {
        if (!active || controller.signal.aborted) return;
        setError(describeError(cause));
        setData(null);
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
      controller.abort();
    };
    // `fn` is intentionally read through a ref, so only the caller's deps
    // (plus the reload nonce) drive re-fetching.
  }, [...deps, nonce]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);

  return { data, loading, error, reload };
}

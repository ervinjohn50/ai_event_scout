import { useCallback, useEffect, useState } from "react";

import { ApiError } from "./api";

/**
 * Loads data when a page opens. `data` and `error` are both null while loading. `load` must be
 * a function defined outside the component, because a new one on each render would reload.
 */
export function useApiData<T>(load: (signal: AbortSignal) => Promise<T>) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal).then(setData, (err: unknown) => {
      if ((err as Error).name === "AbortError") return; // the page closed before it finished
      setError(err instanceof ApiError ? err.message : "Something went wrong.");
    });
    return () => controller.abort();
  }, [load, attempt]);

  const retry = useCallback(() => {
    setData(null);
    setError(null);
    setAttempt((current) => current + 1);
  }, []);

  return { data, error, retry, setData };
}

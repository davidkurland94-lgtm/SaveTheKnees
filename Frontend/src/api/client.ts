/**
 * Thin fetch wrapper shared by every endpoint in `api.ts`.
 *
 * Set `VITE_API_BASE_URL` in `.env` (see `.env.example`). The trailing slash is
 * stripped so callers can always write paths with a leading slash.
 */

const RAW_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export const API_BASE_URL = RAW_BASE_URL.replace(/\/+$/, "");

/** Inference over a full series is slow; the default read timeout is generous. */
const DEFAULT_TIMEOUT_MS = 120_000;

export class ApiError extends Error {
  readonly status: number;
  readonly url: string;
  readonly body: string;

  constructor(status: number, url: string, body: string) {
    super(`${status} ${url}${body ? ` — ${body.slice(0, 300)}` : ""}`);
    this.name = "ApiError";
    this.status = status;
    this.url = url;
    this.body = body;
  }
}

type QueryValue = string | number | boolean | null | undefined;

export interface RequestOptions {
  method?: string;
  query?: Record<string, QueryValue>;
  body?: BodyInit;
  headers?: HeadersInit;
  signal?: AbortSignal;
  timeoutMs?: number;
}

/** Builds an absolute URL, dropping query params that are null/undefined. */
export function buildUrl(path: string, query?: Record<string, QueryValue>): string {
  const url = new URL(`${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`);
  for (const [key, value] of Object.entries(query ?? {})) {
    if (value !== null && value !== undefined) url.searchParams.set(key, String(value));
  }
  return url.toString();
}

async function request(path: string, options: RequestOptions = {}): Promise<Response> {
  const { method = "GET", query, body, headers, signal, timeoutMs = DEFAULT_TIMEOUT_MS } = options;
  const url = buildUrl(path, query);

  // Compose the caller's signal with our timeout so either can abort the request.
  const timeout = AbortSignal.timeout(timeoutMs);
  const composed = signal ? AbortSignal.any([signal, timeout]) : timeout;

  const response = await fetch(url, { method, body, headers, signal: composed });
  if (!response.ok) {
    throw new ApiError(response.status, url, await response.text().catch(() => ""));
  }
  return response;
}

/** GET/POST returning JSON. */
export async function requestJson<T>(path: string, options?: RequestOptions): Promise<T> {
  const response = await request(path, options);
  return response.json() as Promise<T>;
}

/** POST a JSON body and read a JSON response. */
export function postJson<T>(path: string, payload: unknown, options?: RequestOptions): Promise<T> {
  return requestJson<T>(path, {
    ...options,
    method: "POST",
    body: JSON.stringify(payload),
    headers: { "Content-Type": "application/json", ...(options?.headers ?? {}) },
  });
}

/** Fetch a binary endpoint (tensor downloads, PNG contact sheets). */
export async function requestBlob(path: string, options?: RequestOptions): Promise<Blob> {
  const response = await request(path, options);
  return response.blob();
}

/** Turns any thrown value into something worth putting in front of a user. */
export function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 404) return "Not found on the server.";
    return `Server returned ${error.status}.`;
  }
  if (error instanceof DOMException && error.name === "TimeoutError") {
    return "The request timed out. Inference over a full series can take a while.";
  }
  if (error instanceof DOMException && error.name === "AbortError") return "Request cancelled.";
  if (error instanceof TypeError) return `Could not reach the API at ${API_BASE_URL}.`;
  return error instanceof Error ? error.message : "Unexpected error.";
}

import { init as initCore } from "@cornerstonejs/core";
import { init as initDicomImageLoader } from "@cornerstonejs/dicom-image-loader";
import { init as initTools } from "@cornerstonejs/tools";

/**
 * Decoding is the expensive part of a full-resolution series, and it happens
 * off the main thread. Capped well under `hardwareConcurrency` because the
 * viewer shares the machine with whatever else the page is doing.
 */
const MAX_WORKERS = 4;

let started: Promise<void> | null = null;

/** Initialises Cornerstone once per page; safe to call from every viewer. */
export function initCornerstone(): Promise<void> {
  started ??= (async () => {
    initCore();
    initDicomImageLoader({
      maxWebWorkers: Math.max(1, Math.min(navigator.hardwareConcurrency ?? 2, MAX_WORKERS)),
    });
    initTools();
  })();
  return started;
}

export function wadouriImageId(url: string): string {
  return `wadouri:${url}`;
}

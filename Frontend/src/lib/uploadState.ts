import { useOutletContext } from "react-router";

import type { UploadState } from "@/interfaces";

export function useUploadState(): UploadState {
  return useOutletContext<UploadState>();
}

import { useCallback, useMemo, useState } from "react";
import { Outlet, useNavigate } from "react-router";

import { describeError, uploadStudy } from "@/api";
import type { ProcessingStep, UploadState } from "@/interfaces";
import { paths } from "@/lib";

/**
 * Owns the one long-running thing the app does: turning a dropped folder into a
 * stored study.
 *
 * The files are not decoded here any more. They used to be, because the old
 * flow scored a series and threw it away, so the browser was the only place its
 * pixels ever existed. Now the study is stored server-side and the viewer reads
 * it back over the API like any other, which makes a local decode duplicated
 * work on the way to the same picture.
 */
export default function App() {
  const navigate = useNavigate();
  const [studyUid, setStudyUid] = useState<string | null>(null);
  const [fileCount, setFileCount] = useState(0);
  const [step, setStep] = useState<ProcessingStep>(0);
  const [error, setError] = useState<string | null>(null);

  const dismissError = useCallback(() => setError(null), []);

  const start = useCallback(
    async (files: File[]) => {
      setStudyUid(null);
      setError(null);
      setFileCount(files.length);
      setStep(0);
      navigate(paths.processing);

      try {
        setStep(1);
        // One request: the server stores the study, splits it into its series
        // and pre-fills the labels from the image model.
        const study = await uploadStudy(files);
        setStudyUid(study.study_uid);
        // replace, so Back from the new study returns to the list rather than
        // to a progress screen for work that is finished.
        navigate(paths.study(study.study_uid), { replace: true });
      } catch (cause) {
        // Without this the app used to sit on the spinner forever.
        setError(describeError(cause));
        navigate(paths.home, { replace: true });
      }
    },
    [navigate],
  );

  const upload = useMemo<UploadState>(
    () => ({ studyUid, fileCount, step, error, dismissError, start }),
    [studyUid, fileCount, step, error, dismissError, start],
  );

  return <Outlet context={upload} />;
}

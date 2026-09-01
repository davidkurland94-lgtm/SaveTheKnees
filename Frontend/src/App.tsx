import { useCallback, useState } from "react";

import { describeError, predictSeries } from "@/api";
import type { ParsedSeries, PredictionResponse, Route } from "@/interfaces";
import { parseDicomSeries } from "@/lib";
import { ErrorState } from "@/components/ui";
import BenchmarkPage from "@/components/pages/BenchmarkPage";
import HomePage from "@/components/pages/HomePage";
import ProcessingPage, { type ProcessingStep } from "@/components/pages/ProcessingPage";
import StudyPage from "@/components/pages/StudyPage";
import UploadResultPage from "@/components/pages/UploadResultPage";

/** Result of a completed upload: the decoded slices plus the model's scores. */
interface UploadResult {
  series: ParsedSeries;
  prediction: PredictionResponse;
}

export default function App() {
  const [route, setRoute] = useState<Route>({ name: "home" });
  const [upload, setUpload] = useState<UploadResult | null>(null);
  const [fileCount, setFileCount] = useState(0);
  const [step, setStep] = useState<ProcessingStep>(0);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const goHome = useCallback(() => setRoute({ name: "home" }), []);

  const handleUpload = useCallback(async (files: File[]) => {
    setFileCount(files.length);
    setStep(0);
    setUploadError(null);
    setRoute({ name: "processing" });

    try {
      // Decode locally first so the viewer has pixels regardless of the API.
      const series = await parseDicomSeries(files);

      setStep(1);
      const prediction = await predictSeries(files);

      setStep(2);
      setUpload({ series, prediction });
      setRoute({ name: "upload-result" });
    } catch (cause) {
      // Without this the app used to sit on the spinner forever.
      setUploadError(describeError(cause));
      setRoute({ name: "home" });
    }
  }, []);

  const openStudy = useCallback((studyUid: string) => setRoute({ name: "study", studyUid }), []);

  if (route.name === "processing") {
    return <ProcessingPage fileCount={fileCount} step={step} />;
  }

  if (route.name === "upload-result" && upload) {
    return (
      <UploadResultPage
        series={upload.series}
        prediction={upload.prediction}
        onBack={goHome}
      />
    );
  }

  if (route.name === "study") {
    return <StudyPage studyUid={route.studyUid} onBack={goHome} />;
  }

  if (route.name === "benchmark") {
    return <BenchmarkPage onBack={goHome} onOpenStudy={openStudy} />;
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {uploadError && (
        <div className="mx-auto w-full max-w-7xl shrink-0 px-6 pt-6">
          <ErrorState message={uploadError} onRetry={() => setUploadError(null)} />
        </div>
      )}
      <div className="min-h-0 flex-1">
        <HomePage
          onUpload={handleUpload}
          onOpenStudy={openStudy}
          onOpenBenchmark={() => setRoute({ name: "benchmark" })}
        />
      </div>
    </div>
  );
}

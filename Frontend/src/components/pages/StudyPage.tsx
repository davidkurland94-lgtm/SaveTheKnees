import { useState } from "react";

import { getStudy, getStudyLabels, predictStudy } from "@/api";
import type { GoldenLabels, ModelName } from "@/interfaces";
import { cn, joinParts, pluralize, shortUid, toFindings, useAsync } from "@/lib";
import { Chip, ErrorState, GoldenBadge, Icon, Loading, NavBar } from "@/components/ui";
import ReportPanel from "@/components/studies/ReportPanel";
import SeriesList from "@/components/studies/SeriesList";
import FindingList from "@/components/viewer/FindingList";

const MODELS: Array<{ id: ModelName; label: string; hint: string }> = [
  { id: "fusion", label: "Fusion", hint: "Images and report combined" },
  { id: "multiplane", label: "Multiplane", hint: "3-seed sagittal + coronal + axial" },
  { id: "sagittal", label: "Sagittal", hint: "Single-plane 3D CNN, also behind POST /predict" },
];

type Tab = "model" | "labels" | "report" | "series";

const TABS: Array<{ id: Tab; label: string }> = [
  { id: "model", label: "Model" },
  { id: "labels", label: "Report labels" },
  { id: "report", label: "Report text" },
  { id: "series", label: "Series" },
];

interface StudyPageProps {
  studyUid: string;
  onBack: () => void;
}

export function StudyPage({ studyUid, onBack }: StudyPageProps) {
  const [tab, setTab] = useState<Tab>("model");
  const study = useAsync((signal) => getStudy(studyUid, signal), [studyUid]);

  return (
    <div className="flex min-h-full flex-col bg-background">
      <NavBar onHome={onBack}>
        {study.data?.is_golden && <GoldenBadge />}
        <button
          type="button"
          onClick={onBack}
          className="flex items-center gap-1.5 rounded-full border border-border px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:border-accent-soft hover:text-primary"
        >
          <Icon name="arrow-left" size={13} />
          Back to studies
        </button>
      </NavBar>

      <div className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-6 px-6 py-8">
        {study.loading ? (
          <Loading label="Loading study…" />
        ) : study.error ? (
          <ErrorState message={study.error} onRetry={study.reload} />
        ) : study.data ? (
          <>
            <header className="flex flex-col gap-2">
              <h1 className="text-2xl text-foreground">Study {shortUid(studyUid, 16)}</h1>
              <p className="break-all font-mono text-xs text-muted-foreground">{studyUid}</p>
              <div className="mt-1 flex flex-wrap gap-2">
                <Chip>{pluralize(study.data.n_series, "series", "series")}</Chip>
                {Object.entries(study.data.planes).map(([plane, count]) => (
                  <Chip key={plane}>
                    {plane} × {count}
                  </Chip>
                ))}
                {study.data.has_report && <Chip>has report</Chip>}
              </div>
            </header>

            <nav className="flex gap-1 rounded-xl bg-muted p-1">
              {TABS.map((entry) => {
                const disabled = entry.id === "report" && !study.data?.has_report;
                return (
                  <button
                    key={entry.id}
                    type="button"
                    disabled={disabled}
                    onClick={() => setTab(entry.id)}
                    className={cn(
                      "rounded-lg px-4 py-2 text-xs font-semibold transition-all",
                      disabled && "cursor-not-allowed opacity-40",
                      tab === entry.id
                        ? "bg-white text-primary shadow-sm"
                        : "text-muted-foreground hover:text-primary",
                    )}
                  >
                    {entry.label}
                  </button>
                );
              })}
            </nav>

            {tab === "model" && (
              <ModelTab studyUid={studyUid} truth={study.data.golden_labels} />
            )}
            {tab === "labels" && <LabelsTab studyUid={studyUid} truth={study.data.golden_labels} />}
            {tab === "report" && <ReportPanel studyUid={studyUid} />}
            {tab === "series" && <SeriesList series={study.data.series} />}
          </>
        ) : null}
      </div>
    </div>
  );
}

function ModelTab({ studyUid, truth }: { studyUid: string; truth: GoldenLabels | null }) {
  const [model, setModel] = useState<ModelName>("fusion");
  const prediction = useAsync(
    (signal) => predictStudy(studyUid, model, signal),
    [studyUid, model],
  );

  const active = MODELS.find((entry) => entry.id === model) ?? MODELS[0];

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex gap-1 rounded-xl border border-border p-1">
          {MODELS.map((entry) => (
            <button
              key={entry.id}
              type="button"
              title={entry.hint}
              onClick={() => setModel(entry.id)}
              className={cn(
                "rounded-lg px-3 py-1.5 text-xs font-semibold transition-all",
                model === entry.id
                  ? "bg-secondary text-secondary-foreground"
                  : "text-muted-foreground hover:text-primary",
              )}
            >
              {entry.label}
            </button>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">{active.hint}</p>
      </div>

      {prediction.loading ? (
        <Loading label={`Running the ${model} model…`} />
      ) : prediction.error ? (
        <ErrorState message={prediction.error} onRetry={prediction.reload} />
      ) : prediction.data ? (
        <div className="rounded-2xl border border-border p-5">
          <FindingList
            findings={toFindings(prediction.data.predictions, { truth })}
            note={joinParts([
              `GET /studies/{uid}/predict?model=${model}`,
              truth ? "green notches mark the hand-labelled positives" : null,
            ])}
          />
        </div>
      ) : null}
    </div>
  );
}

function LabelsTab({ studyUid, truth }: { studyUid: string; truth: GoldenLabels | null }) {
  const labels = useAsync((signal) => getStudyLabels(studyUid, signal), [studyUid]);

  if (labels.loading) return <Loading label="Loading report labels…" />;
  if (labels.error) return <ErrorState message={labels.error} onRetry={labels.reload} />;
  if (!labels.data) return null;

  const scores = Object.fromEntries(
    Object.entries(labels.data.labels).map(([name, entry]) => [name, entry.probability]),
  );

  return (
    <div className="flex flex-col gap-4">
      <p className="text-xs text-muted-foreground">
        Source: <span className="font-mono">{labels.data.source}</span> — labels derived from the
        written report, with a YES/NO/UNK verdict per finding.
      </p>
      <div className="rounded-2xl border border-border p-5">
        <FindingList
          findings={toFindings(scores, { truth, verdicts: labels.data.labels })}
          showPrimary={false}
        />
      </div>
    </div>
  );
}

export default StudyPage;

import { useState } from "react";

import type { ParsedSeries, PredictionResponse } from "@/interfaces";
import { joinParts, pluralize, toFindings } from "@/lib";
import { Chip, Icon, NavBar, Tag } from "@/components/ui";
import FindingList from "@/components/viewer/FindingList";
import ImageStage from "@/components/viewer/ImageStage";
import SliceSidebar from "@/components/viewer/SliceSidebar";

interface UploadResultPageProps {
  series: ParsedSeries;
  prediction: PredictionResponse;
  onBack: () => void;
}

/** Viewer for a locally uploaded series scored by `POST /predict`. */
export function UploadResultPage({ series, prediction, onBack }: UploadResultPageProps) {
  const { scans, metadata, failedCount } = series;

  const [selectedId, setSelectedId] = useState(scans[0]?.id ?? null);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);

  const selectedIndex = Math.max(
    0,
    scans.findIndex((scan) => scan.id === selectedId),
  );
  const selected = scans[selectedIndex];

  const step = (delta: number) => {
    const next = scans[Math.min(scans.length - 1, Math.max(0, selectedIndex + delta))];
    if (next) setSelectedId(next.id);
  };

  const findings = toFindings(prediction.predictions);

  return (
    <div className="flex min-h-full flex-col bg-background">
      <NavBar onHome={onBack}>
        <Meta label="Patient" value={metadata.patientName} />
        {metadata.studyDate !== "Unknown" && <Meta label="Study date" value={metadata.studyDate} />}
        {metadata.accessionNumber && <Meta label="Accession" value={metadata.accessionNumber} mono />}
        <button
          type="button"
          onClick={onBack}
          className="flex items-center gap-1.5 rounded-full border border-border px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:border-accent-soft hover:text-primary"
        >
          <Icon name="arrow-left" size={13} />
          Back
        </button>
      </NavBar>

      <div className="flex min-h-0 flex-1">
        <SliceSidebar scans={scans} selectedId={selected?.id ?? null} onSelect={setSelectedId} />

        <ImageStage
          scan={selected}
          brightness={brightness}
          contrast={contrast}
          onBrightness={setBrightness}
          onContrast={setContrast}
          onReset={() => {
            setBrightness(100);
            setContrast(100);
          }}
          onStep={step}
        />

        <aside className="flex w-80 shrink-0 flex-col border-l border-border bg-white">
          <div className="border-b border-border bg-card px-5 py-4">
            <h2 className="text-base text-foreground">
              {selected?.seriesDescription || metadata.bodyPart || "Uploaded series"}
            </h2>
            <p className="mt-0.5 text-xs text-muted-foreground">
              {joinParts([
                pluralize(scans.length, "slice"),
                selected && `${selected.cols} × ${selected.rows}px`,
              ])}
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              {selected?.modality && <Tag>{selected.modality}</Tag>}
              {metadata.patientAge && <Tag>Age {metadata.patientAge}</Tag>}
              {metadata.patientSex && <Tag>{metadata.patientSex}</Tag>}
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              <Chip>model: {prediction.model_status}</Chip>
              <Chip>{prediction.n_slices_received} slices scored</Chip>
            </div>
            {failedCount > 0 && (
              <p className="mt-2 text-xs text-amber-600">
                {failedCount} file{failedCount === 1 ? "" : "s"} could not be decoded locally, but
                every uploaded file was still sent to the model.
              </p>
            )}
          </div>

          <div className="flex-1 overflow-y-auto px-5 py-5">
            <FindingList
              findings={findings}
              note={`POST /predict — the image model scored ${prediction.n_slices_received} slices of this series. Advisory only.`}
            />
          </div>
        </aside>
      </div>
    </div>
  );
}

function Meta({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="text-right">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div
        className={
          mono
            ? "font-mono text-sm text-foreground"
            : "text-sm font-semibold text-foreground"
        }
      >
        {value}
      </div>
    </div>
  );
}

export default UploadResultPage;

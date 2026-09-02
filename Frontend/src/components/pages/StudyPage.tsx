import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router";

import {
  getSeriesInstances,
  getStudy,
  getStudyLabels,
  predictStudy,
  seriesInstanceUrl,
  view2dSheet,
} from "@/api";
import type { GoldenLabels, ModelName, Series, ViewerSlice, ViewerStack } from "@/interfaces";
import {
  closeSlices,
  cn,
  joinParts,
  patientOf,
  paths,
  pluralize,
  splitContactSheet,
  toFindings,
  useAsync,
  wadouriImageId,
} from "@/lib";
import { Avatar, Chip, ErrorState, GoldenBadge, Icon, Loading, NavBar } from "@/components/ui";
import ReportPanel from "@/components/studies/ReportPanel";
import SeriesList from "@/components/studies/SeriesList";
import Dicom2DViewer from "@/components/viewer/Dicom2DViewer";
import Dicom3DViewer from "@/components/viewer/Dicom3DViewer";
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

/** `/{StudyInstanceUID}` — the whole page is addressed by the UID in the path. */
export function StudyPage() {
  const { studyUid = "" } = useParams<{ studyUid: string }>();
  // The URL still carries the UID; the page itself never shows one. See
  // `lib/patients.ts` for why a study wears a name here.
  const patient = patientOf(studyUid);
  const [tab, setTab] = useState<Tab>("model");
  const study = useAsync((signal) => getStudy(studyUid, signal), [studyUid]);

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-background">
      <NavBar homeTo={paths.home}>
        {study.data?.is_golden && <GoldenBadge />}
        <Link
          to={paths.home}
          className="flex items-center gap-1.5 rounded-full border border-border px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:border-accent-soft hover:text-primary"
        >
          <Icon name="arrow-left" size={13} />
          Back to studies
        </Link>
      </NavBar>

      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        {/* The narrow rail: everything the study knows about itself. */}
        <aside className="flex w-full shrink-0 flex-col gap-6 overflow-y-auto border-b border-border px-5 py-6 lg:w-1/5 lg:min-w-80 lg:max-w-sm lg:border-b-0 lg:border-r">
          {study.loading ? (
            <Loading label="Loading study…" />
          ) : study.error ? (
            <ErrorState message={study.error} onRetry={study.reload} />
          ) : study.data ? (
            <>
              <header className="flex flex-col gap-2">
                <div className="flex items-center gap-3">
                  <Avatar patient={patient} size="md" />
                  <div className="min-w-0">
                    <h1 className="truncate text-xl text-foreground">{patient.name}</h1>
                    <p className="text-[11px] tabular-nums text-muted-foreground">
                      {patient.age}
                      {patient.sex} · MRN {patient.mrn}
                    </p>
                  </div>
                </div>
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

              <nav className="flex flex-wrap gap-1 rounded-xl bg-muted p-1">
                {TABS.map((entry) => {
                  const disabled = entry.id === "report" && !study.data?.has_report;
                  return (
                    <button
                      key={entry.id}
                      type="button"
                      disabled={disabled}
                      onClick={() => setTab(entry.id)}
                      className={cn(
                        "rounded-lg px-3 py-1.5 text-xs font-semibold transition-all",
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

              {tab === "model" && <ModelTab studyUid={studyUid} truth={study.data.golden_labels} />}
              {tab === "labels" && (
                <LabelsTab studyUid={studyUid} truth={study.data.golden_labels} />
              )}
              {tab === "report" && <ReportPanel studyUid={studyUid} />}
              {tab === "series" && <SeriesList series={study.data.series} />}
            </>
          ) : null}
        </aside>

        {/* The wide column: both viewers, full height. */}
        <section className="flex min-h-0 min-w-0 flex-1 flex-col gap-3 p-3 xl:flex-row">
          <SeriesViewers
            studyUid={studyUid}
            series={study.data?.series ?? []}
            loading={study.loading}
          />
        </section>
      </div>
    </div>
  );
}

/**
 * One column of the sheet per request: `columns: 1` makes the tile size exactly
 * the sheet width and the slice count exactly `height / width`, so nothing has
 * to be assumed about how many slices came back.
 */
const SHEET_COLUMNS = 1;

/** "Sagittal" on its own, but "Sagittal 1" / "Sagittal 2" when a plane repeats. */
function planeLabels(series: Series[]): string[] {
  const totals = new Map<string, number>();
  for (const entry of series) totals.set(entry.plane, (totals.get(entry.plane) ?? 0) + 1);

  const seen = new Map<string, number>();
  return series.map((entry) => {
    const nth = (seen.get(entry.plane) ?? 0) + 1;
    seen.set(entry.plane, nth);
    return (totals.get(entry.plane) ?? 0) > 1 ? `${entry.plane} ${nth}` : entry.plane;
  });
}

function describeSeries(entry: Series): string {
  return joinParts([
    pluralize(entry.n_slices, "slice"),
    entry.fluid_sensitive ? "fluid-sensitive" : null,
    entry.fat_suppression ? "fat-sat" : null,
  ]);
}

/**
 * Owns what the two viewers draw — they are UI only, so the fetching happens
 * here. Each viewer wants the series in a different form, so there are two
 * requests per series, both for the selected one only.
 *
 * `Dicom2DViewer` gets pixels: `/view/{uid}/2d_image_sequence` returns the 24
 * slices the model sees as one contact sheet, fetched here and cut back apart.
 * The server rebuilds each sheet from raw DICOM and that is slow, so sheets
 * already seen are kept and flipping between tabs is instant.
 *
 * `Dicom3DViewer` gets image IDs: `/studies/{uid}/series/{uid}/instances` lists
 * the raw DICOM files in scan order, and Cornerstone streams the bytes behind
 * them itself. Only the list is fetched here — the pixels never pass through
 * this component.
 */
function SeriesViewers({
  studyUid,
  series,
  loading,
}: {
  studyUid: string;
  series: Series[];
  loading: boolean;
}) {
  const available = series.filter((entry) => entry.available);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  // Falls back to the first series until one is picked, and again if the page
  // swaps to a study that does not have the previously selected series.
  const active =
    available.find((entry) => entry.series_uid === selectedId) ?? available[0] ?? null;
  const activeId = active?.series_uid ?? null;

  const cache = useRef(new Map<string, ViewerSlice[]>());

  const slices = useAsync(
    async (signal) => {
      if (!activeId) return [];
      const cached = cache.current.get(activeId);
      if (cached) return cached;

      const sheet = await view2dSheet(
        studyUid,
        { seriesUid: activeId, columns: SHEET_COLUMNS },
        signal,
      );
      const cut = await splitContactSheet(sheet, SHEET_COLUMNS);
      cache.current.set(activeId, cut);
      return cut;
    },
    [studyUid, activeId],
  );

  // Cheap next to the sheet — a list of file names — but still one request per
  // series, so it is fetched for the selected one only, like the sheet is.
  const imageIds = useAsync(
    async (signal) => {
      if (!activeId) return [];
      const { instances } = await getSeriesInstances(studyUid, activeId, signal);
      return instances.map((name) => wadouriImageId(seriesInstanceUrl(studyUid, activeId, name)));
    },
    [studyUid, activeId],
  );

  // ImageBitmaps hold native memory the GC is in no hurry to reclaim, and a
  // five-series study caches 120 of them.
  useEffect(() => {
    const held = cache.current;
    return () => {
      for (const entry of held.values()) closeSlices(entry);
      held.clear();
    };
  }, []);

  const labels = planeLabels(available);
  const stacks: ViewerStack[] = available.map((entry, position) => ({
    id: entry.series_uid,
    plane: entry.plane,
    label: labels[position],
    description: describeSeries(entry),
    // Only the selected series carries data; the rest are tabs waiting to be
    // clicked, which is what keeps the page to one request at a time per view.
    slices: entry.series_uid === activeId ? (slices.data ?? []) : [],
    imageIds: entry.series_uid === activeId ? (imageIds.data ?? []) : [],
  }));

  // The stacks are empty for several different reasons, and saying which one is
  // the difference between "wait" and "this study has nothing to show".
  const emptyReason = (busy: boolean, failure: string | null, waiting: string) =>
    loading
      ? "Loading study…"
      : busy
        ? waiting
        : (failure ??
          (available.length === 0
            ? "No series available for this study."
            : "Nothing to show for this series."));

  const shared = {
    stacks,
    stackId: activeId ?? undefined,
    onStackChange: setSelectedId,
    className: "min-h-0 min-w-0 flex-1",
  };

  return (
    <>
      <Dicom2DViewer
        {...shared}
        emptyLabel={emptyReason(
          slices.loading,
          slices.error,
          "Building the contact sheet on the server…",
        )}
      />
      <Dicom3DViewer
        {...shared}
        emptyLabel={emptyReason(imageIds.loading, imageIds.error, "Listing the DICOM slices…")}
      />
    </>
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

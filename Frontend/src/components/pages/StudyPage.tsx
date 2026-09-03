import { useCallback, useRef, useState } from "react";
import { Link, useParams } from "react-router";

import {
  getSeriesInstances,
  getSeriesTensor,
  getStudyInformation,
  predictStudy,
  seriesInstanceUrl,
} from "@/api";
import type {
  GoldenLabels,
  ModelName,
  ModelVolume,
  PatientIdentity,
  Plane,
  Series,
  StoredReportRecord,
  StudyInformation,
  StudyPredictionResponse,
  ViewerStack,
} from "@/interfaces";
import type { AsyncState } from "@/lib";
import {
  cn,
  createPromiseCache,
  groupBySeverity,
  joinParts,
  npyToVolume,
  patientOf,
  paths,
  pluralize,
  toFindings,
  useAsync,
  wadouriImageId,
} from "@/lib";
import { Avatar, Chip, ErrorState, GoldenBadge, Icon, Loading, NavBar } from "@/components/ui";
import ReportPanel from "@/components/studies/ReportPanel";
import SeriesList from "@/components/studies/SeriesList";
import Dicom3DViewer from "@/components/viewer/Dicom3DViewer";
import DicomMprViewer from "@/components/viewer/DicomMprViewer";
import FindingList, { AttentionFlag } from "@/components/viewer/FindingList";

/**
 * The two models this page offers, in the order the pipeline produces them.
 *
 * Images first, because images are all a study has when it arrives: a folder of
 * DICOM is stored, split into series and scored by the image model before
 * anyone has read it. Fusion is second and gated, because its other input is
 * the report — it cannot run until a doctor has written one, and listing it
 * here greyed out says that far more plainly than hiding it would.
 *
 * Sagittal is not offered. It reads one plane where multiplane reads three, so
 * on any study that has all three it is strictly the weaker answer, and the
 * choice between them is not one a doctor should have to make. The server still
 * falls back to it when a study is missing a plane, and says so through the
 * `model` it reports — see `served` below.
 */
const MODELS: Array<{ id: ModelName; label: string; hint: string; needsReport?: true }> = [
  { id: "multiplane", label: "Multiplane", hint: "Images only — sagittal + coronal + axial" },
  {
    id: "fusion",
    label: "Fusion",
    hint: "Images and the report, jointly",
    needsReport: true,
  },
];

/**
 * Model runs, kept for as long as the tab is open.
 *
 * `GET /studies/{uid}/predict` runs a network over the whole volume and takes
 * seconds, while its answer for one study and one model never changes. Without
 * this, flipping between Fusion and Multiplane to compare the two re-runs both
 * models every time — which is exactly what someone comparing them does.
 *
 * A cached response is a few hundred bytes, so the cap is about how many
 * studies a session visits rather than about memory.
 */
const cachedPrediction = createPromiseCache<StudyPredictionResponse>(60);

/**
 * Which series the rail is showing, or `null` for none.
 *
 * The banner already counts the study's series and breaks them down by plane,
 * so those counts are the control: clicking "Sagittal × 2" is a more direct way
 * of asking to see those two than a "Series" button that shows all of them and
 * leaves the reader to find the plane again.
 */
type SeriesFilter = "all" | Plane;

/**
 * `/{StudyInstanceUID}` — the whole page is addressed by the UID in the path.
 *
 * Reads `/view/{uid}/information` rather than `/studies/{uid}`: it is the same
 * study record plus the two reports, and the reports are what decide which
 * models this page may offer. A study uploaded a minute ago has neither, so it
 * opens on the image models with an empty report to write; a corpus study
 * arrives with the radiologist's already attached, so fusion is live from the
 * first paint.
 */
export function StudyPage() {
  const { studyUid = "" } = useParams<{ studyUid: string }>();
  // The URL still carries the UID; the page itself never shows one. See
  // `lib/patients.ts` for why a study wears a name here.
  const patient = patientOf(studyUid);
  const [series, setSeries] = useState<SeriesFilter | null>(null);
  const [showReport, setShowReport] = useState(true);
  const study = useAsync((signal) => getStudyInformation(studyUid, signal), [studyUid]);

  /**
   * The doctor's report, once this session has written one.
   *
   * Held here rather than in the panel because two different parts of the page
   * depend on it: the panel that writes it, and the model rail, where saving
   * the first report is what makes Fusion runnable. Carrying the UID alongside
   * means a move to another study falls back to that study's own record instead
   * of showing the last one's.
   */
  const [saved, setSaved] = useState<{ uid: string; record: StoredReportRecord } | null>(null);
  const stored = saved?.uid === studyUid ? saved.record : (study.data?.my_report ?? null);
  const onSaved = useCallback(
    (record: StoredReportRecord) => setSaved({ uid: studyUid, record }),
    [studyUid],
  );

  /**
   * Identifies the report the fusion model would read, or null for "there is
   * none". A stamp rather than a flag, so re-scoring after an edit is a cache
   * miss instead of a stale answer.
   */
  const reportStamp = stored
    ? `mine:${stored.updated_at}`
    : study.data?.report
      ? "dataset"
      : null;

  // Clicking the chip that opened a view closes it again, so every chip is its
  // own way back and the rail needs no separate dismiss.
  const toggleSeries = (filter: SeriesFilter) =>
    setSeries((current) => (current === filter ? null : filter));

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

      <StudyBanner
        patient={patient}
        study={study.data}
        series={series}
        onSeries={toggleSeries}
        showReport={showReport}
        onToggleReport={() => setShowReport((shown) => !shown)}
        hasReport={reportStamp !== null}
      />

      {/* Reversed, not reordered: on a small screen the rail stacks above the
          viewers, which is where a panel of findings belongs; `row-reverse`
          paints it down the right-hand side once there is room for a column. */}
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row-reverse">
        {/* The narrow rail, on the right: what the models make of the study. */}
        <aside className="flex w-full shrink-0 flex-col gap-6 overflow-y-auto border-b border-border px-5 py-6 lg:w-1/5 lg:min-w-80 lg:max-w-sm lg:overflow-hidden lg:border-b-0 lg:border-l">
          {study.loading ? (
            <Loading label="Loading study…" />
          ) : study.error ? (
            <ErrorState message={study.error} onRetry={study.reload} />
          ) : !study.data ? null : series ? (
            <SeriesPanel
              series={study.data.series}
              filter={series}
              onClose={() => setSeries(null)}
            />
          ) : (
            <ModelTab
              studyUid={studyUid}
              truth={study.data.golden_labels}
              reportStamp={reportStamp}
            />
          )}
        </aside>

        {/* The wide column: the viewer, with the written report beneath it. */}
        <section className="flex min-h-0 min-w-0 flex-1 flex-col gap-3 p-3">
          <SeriesViewers
            studyUid={studyUid}
            series={study.data?.series ?? []}
            loading={study.loading}
          />
          {/* Always offered, never conditional on a report existing: on an
              uploaded study this empty box IS the next step, and a panel that
              only appears once there is something in it can never be the place
              the first report gets written. Keyed by study so moving between
              two of them cannot carry a draft across. */}
          {study.data && showReport && (
            <ReportPanel
              key={studyUid}
              studyUid={studyUid}
              dataset={study.data.report}
              stored={stored}
              onSaved={onSaved}
            />
          )}
        </section>
      </div>
    </div>
  );
}

/**
 * The band under the nav bar: who the study belongs to, and what is in it.
 *
 * Full width and one line deep on purpose. It is the only thing on the page
 * that both columns need to agree about, and pinning it above them means
 * neither the viewers nor the findings rail has to spend its own space saying
 * whose knee this is.
 *
 * The identity comes from the UID in the URL, so it paints immediately; only
 * the study's own facts wait on the request.
 */
function StudyBanner({
  patient,
  study,
  series,
  onSeries,
  showReport,
  onToggleReport,
  hasReport,
}: {
  patient: PatientIdentity;
  study: StudyInformation | null;
  series: SeriesFilter | null;
  onSeries: (filter: SeriesFilter) => void;
  showReport: boolean;
  onToggleReport: () => void;
  /** Whether either report exists — what the chip reports and Fusion needs. */
  hasReport: boolean;
}) {
  return (
    <div className="flex shrink-0 flex-wrap items-center gap-x-4 gap-y-2 border-b border-border bg-card px-5 py-2">
      <div className="flex min-w-0 items-center gap-3">
        <Avatar patient={patient} />
        <h1 className="truncate text-base text-foreground">{patient.name}</h1>
        <p className="shrink-0 text-[11px] tabular-nums text-muted-foreground">
          {patient.age}
          {patient.sex} · MRN {patient.mrn}
        </p>
      </div>

      {study && (
        <div className="flex flex-wrap items-center gap-2 lg:ml-auto">
          <Chip
            onClick={() => onSeries("all")}
            active={series === "all"}
            title="Show every series in this study"
          >
            {pluralize(study.n_series, "series", "series")}
          </Chip>
          {Object.entries(study.planes).map(([plane, count]) => (
            <Chip
              key={plane}
              onClick={() => onSeries(plane as Plane)}
              active={series === plane}
              title={`Show only the ${plane.toLowerCase()} series`}
            >
              {plane} × {count}
            </Chip>
          ))}
          {/* Shown either way. "no report yet" is the single most useful thing
              this banner can say about a study that has just been uploaded, and
              it is the chip that opens the place to fix it. */}
          <Chip
            onClick={onToggleReport}
            active={showReport}
            title={showReport ? "Hide the report" : "Show the report"}
          >
            {hasReport ? "has report" : "no report yet"}
          </Chip>
        </div>
      )}
    </div>
  );
}

/** The rail's series view, opened by a chip in the banner rather than by a tab. */
function SeriesPanel({
  series,
  filter,
  onClose,
}: {
  series: Series[];
  filter: SeriesFilter;
  onClose: () => void;
}) {
  const shown = filter === "all" ? series : series.filter((entry) => entry.plane === filter);

  return (
    <div className="flex flex-col gap-4 lg:min-h-0 lg:flex-1">
      <div className="flex shrink-0 items-center justify-between gap-2">
        <h2 className="text-base text-foreground">
          {filter === "all" ? "All series" : `${filter} series`}
        </h2>
        <button
          type="button"
          onClick={onClose}
          className="flex items-center gap-1 rounded-full border border-border px-2.5 py-1 text-[11px] font-medium text-muted-foreground transition-colors hover:border-accent-soft hover:text-primary"
        >
          <Icon name="arrow-left" size={11} />
          Back
        </button>
      </div>
      <div className="lg:min-h-0 lg:flex-1 lg:overflow-y-auto">
        <SeriesList series={shown} />
      </div>
    </div>
  );
}

/**
 * What the right-hand panel is showing.
 *
 * Two answers to two different questions, which is why they are a switch rather
 * than two panels. "Reference" is the study's own DICOM, where the geometry
 * survives — the only place a length is in millimetres and a point marked in
 * one plane can be found in the others. "Volume" is the model's input rendered
 * whole, which is what the scores in the rail were actually computed from.
 *
 * Reference leads, and is the default. It is the study as the scanner wrote it,
 * which is what a doctor opens a study to look at; the model's view of it is
 * the second question, asked once the first has been answered.
 *
 * Neither has to be waited for, because neither is fetched on demand — see the
 * two requests in `SeriesViewers`, which both start as soon as the series is
 * known, whichever panel is on screen.
 */
const PANEL_VIEWS = [
  {
    id: "reference",
    label: "Reference",
    hint: "The study's own DICOM: three planes, measurable",
  },
  { id: "volume", label: "Volume", hint: "The model's input, rendered whole" },
] as const;

type PanelView = (typeof PANEL_VIEWS)[number]["id"];

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
 * A fetched value, but only while it still belongs to the series on screen.
 *
 * `useAsync` holds its last result through the next request, which is what
 * makes an already-seen tab flip back instantly — and is wrong the moment the
 * tabs disagree. Without this the previous series' pixels are painted under the
 * new tab's name until the new ones land, and the 3D viewer, which keys its
 * GPU-side volume by series UID, would cache them there for good.
 */
function forSeries<T>(
  state: AsyncState<{ seriesUid: string; value: T } | null>,
  seriesUid: string | null,
): T | null {
  return state.data && state.data.seriesUid === seriesUid ? state.data.value : null;
}

/**
 * Owns what the viewer draws — it is UI only, so the fetching happens here.
 *
 * One request per series, for the selected series and the chosen panel only.
 * `Dicom3DViewer` reads the model's own input from
 * `/studies/{uid}/series/{uid}/tensor`, a float32 `.npy`; `DicomMprViewer`
 * reads the study's DICOM file names from `.../instances` and lets Cornerstone
 * stream the pixels. Neither is fetched for a panel that is not on screen.
 *
 * The server rebuilds both from raw DICOM on every request and that is slow, so
 * each is kept once seen and flipping back to a series is instant.
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
  const [panel, setPanel] = useState<PanelView>("reference");
  // Falls back to the first series until one is picked, and again if the page
  // swaps to a study that does not have the previously selected series.
  const active =
    available.find((entry) => entry.series_uid === selectedId) ?? available[0] ?? null;
  const activeId = active?.series_uid ?? null;

  // About 4.6 MB of float32 per series, held for the tab switch: the server
  // rebuilds the tensor from raw DICOM on every request, and the volume the 3D
  // viewer keeps on the GPU is keyed by series too, so re-fetching one already
  // uploaded would be paid for and then thrown away.
  const volumes = useRef(new Map<string, ModelVolume>());

  // Deliberately ungated by `panel`: the tensor is fetched even while the
  // reference view is the one on screen, so that switching to the volume is a
  // render rather than a wait. It is the slow half of this page — the server
  // rebuilds it from raw DICOM, a couple of seconds before a byte is sent — and
  // there is nothing to gain by not having started.
  const tensors = useAsync(
    async (signal) => {
      if (!activeId) return null;
      const cached = volumes.current.get(activeId);
      if (cached) return { seriesUid: activeId, value: cached };

      const npy = await getSeriesTensor(studyUid, activeId, signal);
      const built = npyToVolume(await npy.arrayBuffer());
      volumes.current.set(activeId, built);
      return { seriesUid: activeId, value: built };
    },
    [studyUid, activeId],
  );

  // The list of file names, which is cheap; the pixels behind them are streamed
  // by Cornerstone and cached there. Gated on the panel only so that a reader
  // who has moved to the volume does not re-list on the way back and forth —
  // the reference view being the default, this fetches on load either way.
  const instances = useAsync(
    async (signal) => {
      if (!activeId || panel !== "reference") return null;
      const { instances: names } = await getSeriesInstances(studyUid, activeId, signal);
      return {
        seriesUid: activeId,
        value: names.map((name) => wadouriImageId(seriesInstanceUrl(studyUid, activeId, name))),
      };
    },
    [studyUid, activeId, panel],
  );

  const volume = forSeries(tensors, activeId) ?? undefined;
  const imageIds = forSeries(instances, activeId) ?? [];

  const labels = planeLabels(available);
  const stacks: ViewerStack[] = available.map((entry, position) => ({
    id: entry.series_uid,
    plane: entry.plane,
    label: labels[position],
    description: describeSeries(entry),
    // Only the selected series carries data; the rest are tabs waiting to be
    // clicked, which is what keeps the page to one request at a time per view.
    volume: entry.series_uid === activeId ? volume : undefined,
    imageIds: entry.series_uid === activeId ? imageIds : [],
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
  };

  const views = {
    options: PANEL_VIEWS,
    active: panel,
    onSelect: (id: string) => setPanel(id as PanelView),
  };

  return panel === "reference" ? (
    <DicomMprViewer
      {...shared}
      views={views}
      className="min-h-0 min-w-0 flex-1"
      emptyLabel={emptyReason(
        instances.loading,
        instances.error,
        "Listing the study's DICOM slices…",
      )}
    />
  ) : (
    <Dicom3DViewer
      {...shared}
      views={views}
      className="min-h-0 min-w-0 flex-1"
      emptyLabel={emptyReason(tensors.loading, tensors.error, "Downloading the model's tensor…")}
    />
  );
}

/**
 * The rail's model view: which model, and what it made of this study.
 *
 * `reportStamp` is the whole gate. Null means no report exists, so the fusion
 * model has only one of its two inputs and its tab is dead — and the panel opens
 * on the images, which are the only thing there is to go on. When a stamp
 * appears (the doctor saved one, or the study came out of the corpus with the
 * radiologist's attached) fusion becomes selectable, and a reader who has not
 * chosen otherwise is moved onto it, because that is the answer this project
 * exists to give.
 */
function ModelTab({
  studyUid,
  truth,
  reportStamp,
}: {
  studyUid: string;
  truth: GoldenLabels | null;
  reportStamp: string | null;
}) {
  const hasReport = reportStamp !== null;
  // Null until a reader picks, so availability decides for as long as nobody
  // has: no effect has to chase the report appearing, and an explicit choice is
  // never overridden by one arriving.
  const [picked, setPicked] = useState<ModelName | null>(null);
  const model: ModelName =
    picked && (picked !== "fusion" || hasReport) ? picked : hasReport ? "fusion" : "multiplane";

  // The report is an input to fusion, so an edited report is a different run.
  // The image models do not read it and keep their answer across a save.
  const key = model === "fusion" ? `${studyUid}|fusion|${reportStamp}` : `${studyUid}|${model}`;
  const prediction = useAsync(
    // No abort signal on purpose: a run cancelled halfway leaves nothing to
    // cache, and a switch away and back should be the case this makes instant.
    () => cachedPrediction(key, () => predictStudy(studyUid, model)),
    [key],
  );

  // Grouped once here rather than inside the list: the flagged band is lifted
  // out and pinned above the scroll area, so the two consumers need the same
  // split and must not disagree about it.
  const groups = prediction.data
    ? groupBySeverity(toFindings(prediction.data.predictions, { truth, sortByProbability: true }))
    : null;

  const active = MODELS.find((entry) => entry.id === model) ?? MODELS[0];
  // An uploaded study missing a plane is scored by whatever the server could
  // actually run, and it says which in the response. Believing the request
  // instead would label a single-plane answer "Multiplane".
  const served = prediction.data && prediction.data.model !== model ? prediction.data.model : null;

  return (
    <div className="flex flex-col gap-4 lg:min-h-0 lg:flex-1">
      {/* Fixed height while the findings below it scroll: switching models is
          the one control on this panel, and scrolling to reach it would be
          scrolling away from the numbers it changes. */}
      <div className="flex shrink-0 flex-wrap items-center gap-3">
        <div className="flex gap-1 rounded-xl border border-border p-1">
          {MODELS.map((entry) => {
            const locked = entry.needsReport && !hasReport;
            return (
              <button
                key={entry.id}
                type="button"
                disabled={locked}
                aria-pressed={model === entry.id}
                title={
                  locked
                    ? "Fusion reads the images and the report together. Write a report below to run it."
                    : entry.hint
                }
                onClick={() => setPicked(entry.id)}
                className={cn(
                  "flex items-center gap-1 rounded-lg px-3 py-1.5 text-xs font-semibold transition-all",
                  locked
                    ? "cursor-not-allowed text-subtle"
                    : model === entry.id
                      ? "bg-secondary text-secondary-foreground"
                      : "text-muted-foreground hover:text-primary",
                )}
              >
                {locked && <Icon name="lock" size={10} strokeWidth={2.5} />}
                {entry.label}
              </button>
            );
          })}
        </div>
        <p className="text-xs text-muted-foreground">{active.hint}</p>
      </div>

      {/* Above everything the panel has to say about this study, and outside the
          scroll area, because a flag a reader has to scroll to find is not a
          flag. */}
      {groups && <AttentionFlag findings={groups.moderate} />}

      {!hasReport && (
        <p className="shrink-0 rounded-xl border border-dashed border-border px-3 py-2 text-[11px] leading-relaxed text-muted-foreground">
          Images only so far. The report model has nothing to read until a report
          is written, and fusion needs both.
        </p>
      )}

      {served && (
        <p className="shrink-0 text-[11px] text-muted-foreground">
          Served by the {served} model — the server could not run {model} over this study.
        </p>
      )}

      <div className="lg:min-h-0 lg:flex-1 lg:overflow-y-auto">
        {prediction.loading ? (
          <Loading label={`Running the ${model} model…`} />
        ) : prediction.error ? (
          <ErrorState message={prediction.error} onRetry={prediction.reload} />
        ) : groups ? (
          <div className="rounded-2xl border border-border p-5">
            <FindingList
              groups={groups}
              note={truth ? "Hand-labelled ground truth is shown beside each finding." : undefined}
            />
          </div>
        ) : null}
      </div>
    </div>
  );
}

export default StudyPage;

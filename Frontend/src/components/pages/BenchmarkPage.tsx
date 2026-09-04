import { Link } from "react-router";

import { getReportTable, getReportVerdicts, getStudy } from "@/api";
import type { ReportTableRow, VerdictRow } from "@/interfaces";
import { cn, patientOf, paths, useAsync } from "@/lib";
import { ErrorState, Icon, Loading, NavBar } from "@/components/ui";

/** The deck, in presentation order. Served straight out of `public/`. */
const SLIDES = [
  { title: "The project", file: "/Slide01-Project.svg" },
  { title: "The data", file: "/Slide02-The Data.svg" },
  { title: "The process", file: "/Slide03 - The Process.svg" },
] as const;

/**
 * The sign-off, held back to close the page.
 *
 * It is the deck's fourth slide, but it reads as an ending, so it belongs
 * after the numbers rather than in front of them with the other three.
 */
const CLOSING_SLIDE = { title: "Thank you", file: "/Slide04 - ThankYou.svg" } as const;

/**
 * The five readers we trained — the same tuple the backend calls `OUR_MODELS`
 * in `models/evaluate_labels.py`. `GET /report/table` also scores the LLM
 * extractions and pilkwang; those are other people's readers, and this page
 * does not show them.
 */
const OUR_MODELS = [
  { key: "image_model", label: "Image", note: "sagittal CNN, the serving checkpoint" },
  { key: "image_multiplane", label: "Multiplane", note: "3-seed image ensemble" },
  { key: "report_model", label: "Report", note: "TF-IDF + medical terms" },
  { key: "report_bagged", label: "Report bagged", note: "5-fold × 2-seed bag" },
  { key: "fusion_model", label: "Fusion", note: "images + report, jointly trained" },
] as const satisfies ReadonlyArray<{ key: keyof ReportTableRow; label: string; note: string }>;

/**
 * How many verdict rows to pull in one go.
 *
 * `GET /report/verdicts` serves the corpus worst-first and caps `top` at 500,
 * so the widest window the API offers is also the only way to reach the good
 * end of the ranking: the best reports are its tail.
 */
const VERDICT_WINDOW = 500;

/** How many studies the images-vs-reports section shows. */
const TOP_N = 10;

/**
 * Availability probes to run at once, and how far down the ranking to look.
 *
 * A verdict row is scored from CSVs that cover the whole 4,407-study corpus,
 * but only part of that corpus has its DICOM on any given deployment — so a
 * row can rank top of the sheet and still open onto a study with nothing to
 * show. There is no bulk "has images" route, so the only way to know is to ask
 * per study; these two numbers keep that to a few batched rounds.
 */
const PROBE_BATCH = 12;
const PROBE_DEPTH = 180;

export function BenchmarkPage() {
  return (
    <div className="flex min-h-full flex-col bg-background">
      <NavBar homeTo={paths.home}>
        <Link
          to={paths.home}
          className="flex items-center gap-1.5 rounded-full border border-border px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:border-accent-soft hover:text-primary"
        >
          <Icon name="arrow-left" size={13} />
          Back to studies
        </Link>
      </NavBar>

      <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-12 px-6 py-8">
        <SlideDeck />
        <BestAgreement />
        <OurModels />
        <ClosingSlide />
      </div>
    </div>
  );
}

/** A section heading with its one-line explanation. */
function SectionHead({ title, blurb }: { title: string; blurb: string }) {
  return (
    <div>
      <h2 className="text-lg text-foreground">{title}</h2>
      <p className="mt-0.5 text-xs text-muted-foreground">{blurb}</p>
    </div>
  );
}

// ─── The deck ─────────────────────────────────────────────────────────────────

/**
 * The three presentation slides, stacked in order and scrolled through.
 *
 * Plain `<img>`, because they are vector: each slide carries its own
 * 1280x832 viewBox, so the browser sizes it from the width and nothing here
 * has to know the shape of a slide. They were PDFs first, and a PDF in a page
 * is never an image — it is a viewer, with its own scrollbar and its own idea
 * of how to fit the page.
 */
function SlideDeck() {
  return (
    <section className="flex flex-col gap-16 py-8">
      {SLIDES.map((slide) => (
        <img key={slide.file} src={encodeURI(slide.file)} alt={slide.title} className="w-full" />
      ))}
    </section>
  );
}

/** The closing slide, sized exactly like the deck's, but at the foot of the page. */
function ClosingSlide() {
  return (
    <section className="py-8">
      <img src={encodeURI(CLOSING_SLIDE.file)} alt={CLOSING_SLIDE.title} className="w-full" />
    </section>
  );
}

// ─── Our five readers ─────────────────────────────────────────────────────────

/** The AUC sheet with the other people's readers taken out. */
function OurModels() {
  const table = useAsync((signal) => getReportTable(signal), []);

  return (
    <section className="flex flex-col gap-3 pt-8">
      <SectionHead
        title="Our models"
        blurb="AUC on the 58 hand-labelled studies."
      />

      {table.loading ? (
        <Loading label="Loading the benchmark sheet…" />
      ) : table.error ? (
        <ErrorState message={table.error} onRetry={table.reload} />
      ) : (
        <OurModelsBody rows={table.data?.rows ?? []} />
      )}
    </section>
  );
}

function OurModelsBody({ rows }: { rows: ReportTableRow[] }) {
  // The sheet's trailing summary row is the one the API sends with no positive
  // count; the count itself is not a column here, only the tell.
  const mean = rows.find((row) => typeof row.positives !== "number");
  const labels = rows.filter((row) => typeof row.positives === "number");

  return (
    <div className="flex flex-col gap-4">
      {mean && <MeanAuc mean={mean} />}

      <div className="overflow-x-auto rounded-2xl border border-border">
        <table className="w-full min-w-[40rem] text-sm">
          <thead>
            <tr className="border-b border-border bg-card">
              <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Label
              </th>
              {OUR_MODELS.map((model) => (
                <th
                  key={model.key}
                  className="px-3 py-3 text-right text-xs font-semibold uppercase tracking-wide text-muted-foreground"
                >
                  {model.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {labels.map((row, index) => (
              <OurModelsRow
                key={row.label}
                row={row}
                last={index === labels.length - 1 && !mean}
              />
            ))}
            {mean && <OurModelsRow row={mean} last summary />}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function OurModelsRow({
  row,
  last,
  summary,
}: {
  row: ReportTableRow;
  last: boolean;
  summary?: boolean;
}) {
  // Best of ours on this row — the other readers no longer compete for it.
  const best = Math.max(...OUR_MODELS.map((model) => Number(row[model.key])));

  return (
    <tr
      className={cn(
        "transition-colors hover:bg-card",
        !last && "border-b border-border-soft",
        summary && "border-t-2 border-t-border bg-muted/60 font-semibold",
      )}
    >
      <td className="px-4 py-2.5 font-medium text-foreground">{row.label}</td>
      {OUR_MODELS.map((model) => {
        const value = Number(row[model.key]);
        return (
          <td
            key={model.key}
            className={cn(
              "px-3 py-2.5 text-right tabular-nums",
              value === best ? "font-semibold text-primary" : "text-foreground",
            )}
          >
            {value.toFixed(3)}
          </td>
        );
      })}
    </tr>
  );
}

/**
 * Mean AUC per model, ranked — the headline of the sheet below it.
 *
 * One measure, one hue: the bars carry magnitude, so there is nothing for a
 * second colour to mean. Each track is drawn full width with a hairline at the
 * half, because 0.5 is not "half as good" but a coin toss, and a reader who
 * cannot see where chance sits cannot read an AUC at all.
 */
function MeanAuc({ mean }: { mean: ReportTableRow }) {
  const ranked = [...OUR_MODELS].sort((a, b) => Number(mean[b.key]) - Number(mean[a.key]));

  return (
    <div className="rounded-2xl border border-border bg-card p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-sm text-foreground">Mean AUC across the twelve findings</h3>
        <span className="text-xs text-muted-foreground">1.0 = perfect · 0.5 = chance</span>
      </div>

      <ul className="mt-4 flex flex-col gap-3">
        {ranked.map((model) => {
          const value = Number(mean[model.key]);
          return (
            <li
              key={model.key}
              className="grid grid-cols-[minmax(6rem,10rem)_1fr_3rem] items-center gap-3"
            >
              <div>
                <p className="text-xs font-semibold text-foreground">{model.label}</p>
                <p className="text-[10px] leading-tight text-subtle">{model.note}</p>
              </div>
              <div
                className="relative h-4 rounded-r-sm bg-muted"
                title={`${model.label}: mean AUC ${value.toFixed(3)}`}
              >
                <div
                  className="h-full rounded-r-[4px] bg-primary"
                  style={{ width: `${value * 100}%` }}
                />
                {/* Chance, drawn over the fill so it stays visible inside it. */}
                <span className="absolute inset-y-[-4px] left-1/2 w-px bg-accent-soft" />
              </div>
              <span className="text-right text-xs font-semibold tabular-nums text-foreground">
                {value.toFixed(3)}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

// ─── Images vs the report ─────────────────────────────────────────────────────

/** "ACL, Effusion" -> ["ACL", "Effusion"]; the sheet's "(nothing)" -> []. */
function findings(cell: string): string[] {
  return cell === "(nothing)" ? [] : cell.split(", ").filter(Boolean);
}

/**
 * Neither witness saw anything.
 *
 * A perfect score, and a true one — but a clean knee nobody wrote up is no
 * evidence that the two readers agree, so it does not belong in a list whose
 * whole point is showing them agreeing.
 */
function silent(row: VerdictRow): boolean {
  return findings(row.report_says).length === 0 && findings(row.images_say).length === 0;
}

/** The findings both witnesses name — the agreement a good report is made of. */
function agreed(row: VerdictRow): string[] {
  const images = new Set(findings(row.images_say));
  return findings(row.report_says).filter((finding) => images.has(finding));
}

/** Whether this deployment holds any of the study's images. */
async function hasImages(uid: string, signal: AbortSignal): Promise<boolean> {
  try {
    const study = await getStudy(uid, signal);
    return study.series.some((entry) => entry.available);
  } catch (cause) {
    // An abort has to keep travelling, or the search below carries on firing
    // requests for a section nobody is looking at any more. Anything else is
    // a study we cannot describe, which is a study we cannot open.
    if (signal.aborted) throw cause;
    return false;
  }
}

/**
 * The best `want` rows whose study can actually be opened.
 *
 * Walks the ranking in order, probing a batch at a time and stopping as soon
 * as enough have answered yes — so a deployment carrying the whole corpus pays
 * for one round, and a partial one pays for a few.
 */
async function openable(
  ranked: VerdictRow[],
  want: number,
  signal: AbortSignal,
): Promise<VerdictRow[]> {
  const kept: VerdictRow[] = [];
  const pool = ranked.slice(0, PROBE_DEPTH);

  for (let start = 0; start < pool.length && kept.length < want; start += PROBE_BATCH) {
    const batch = pool.slice(start, start + PROBE_BATCH);
    const answers = await Promise.all(
      batch.map((row) => hasImages(row.StudyInstanceUID, signal)),
    );
    batch.forEach((row, index) => {
      if (answers[index]) kept.push(row);
    });
  }

  return kept.slice(0, want);
}

/**
 * The ten reports the images corroborate best, of the studies this deployment
 * can open.
 *
 * The score tops out at 1.00 (nothing the images read is missing from the
 * report) and plenty of studies sit there, so the tie is broken on how much
 * the two actually agree about: a report that names eight findings the images
 * also read is a better witness than one that names two. Studies where neither
 * witness saw anything score a perfect 1.00 on a technicality and are dropped.
 */
function BestAgreement() {
  const best = useAsync(async (signal) => {
    const { rows } = await getReportVerdicts(VERDICT_WINDOW, signal);
    const ranked = rows
      .filter((row) => !silent(row))
      .sort((a, b) => b.quality_score - a.quality_score || agreed(b).length - agreed(a).length);
    return openable(ranked, TOP_N, signal);
  }, []);

  const rows = best.data ?? [];

  return (
    <section className="flex flex-col gap-3">
      <SectionHead
        title="Top 10 — images vs reports"
        blurb="The best of the two witnesses: every finding the images read is already in the report. Every row opens."
      />

      {best.loading ? (
        <Loading label="Finding the ten best…" />
      ) : best.error ? (
        <ErrorState message={best.error} onRetry={best.reload} />
      ) : (
        <div className="overflow-x-auto rounded-2xl border border-border">
          <table className="w-full min-w-[56rem] text-sm">
            <thead>
              <tr className="border-b border-border bg-card">
                {["#", "Study", "Report says", "Images say", "Both agree", "Quality"].map(
                  (heading) => (
                    <th
                      key={heading}
                      className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground"
                    >
                      {heading}
                    </th>
                  ),
                )}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr
                  key={row.StudyInstanceUID}
                  className={cn(
                    "align-top transition-colors hover:bg-card",
                    index < rows.length - 1 && "border-b border-border-soft",
                  )}
                >
                  <td className="px-4 py-3 text-xs tabular-nums text-subtle">{index + 1}</td>
                  <td className="px-4 py-3">
                    <PatientCell uid={row.StudyInstanceUID} />
                  </td>
                  <td className="max-w-56 px-4 py-3 text-xs text-foreground">{row.report_says}</td>
                  <td className="max-w-56 px-4 py-3 text-xs text-muted-foreground">
                    {row.images_say}
                  </td>
                  <td className="px-4 py-3 text-xs tabular-nums text-emerald-600">
                    {agreed(row).length}
                  </td>
                  <td className="px-4 py-3 text-xs tabular-nums text-foreground">
                    {row.quality_score.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!best.loading && !best.error && rows.length < TOP_N && (
        <p className="text-xs text-muted-foreground">
          Showing {rows.length}. The rest of the ranking points at studies whose images are not on
          this deployment, so there would be nothing to open.
        </p>
      )}
    </section>
  );
}

/** A verdict row's study, named rather than numbered, linking to the study. */
function PatientCell({ uid }: { uid: string }) {
  const patient = patientOf(uid);
  return (
    <>
      <Link to={paths.study(uid)} className="text-xs font-medium text-primary hover:underline">
        {patient.name}
      </Link>
      <p className="text-[10px] tabular-nums text-subtle">MRN {patient.mrn}</p>
    </>
  );
}

export default BenchmarkPage;

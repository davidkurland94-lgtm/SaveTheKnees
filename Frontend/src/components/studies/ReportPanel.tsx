import { useCallback, useEffect, useMemo, useState } from "react";

import {
  describeError,
  getReportTerms,
  getStudyReport,
  predictReport,
  saveStudyReport,
} from "@/api";
import type { LabelScores, ReportLang, ReportResponse, StoredReportRecord } from "@/interfaces";
import {
  cn,
  createPromiseCache,
  englishReport,
  markTerms,
  pluralize,
  toFindings,
  useAsync,
  wasTranslated,
} from "@/lib";
import { Button, ErrorState, Icon, Loading, Spinner } from "@/components/ui";
import FindingList from "@/components/viewer/FindingList";

/**
 * The dictionary, fetched once for the whole session.
 *
 * It is the same list for every study and it does not change while the tab is
 * open, so one request serves every report the reader opens. A deployment that
 * predates the route answers 404; that is caught here and cached as "no
 * dictionary", which costs one failed request rather than one per study.
 */
const cachedTerms = createPromiseCache<string[]>(1);

const loadTerms = () =>
  cachedTerms("dictionary", () =>
    getReportTerms()
      .then((response) => response.terms)
      .catch(() => []),
  );

/**
 * Report model runs, keyed by the text itself rather than by the study.
 *
 * The panel scores automatically, so without this every switch between the two
 * reports — and every remount as the reader moves around the study — would send
 * the same paragraph to the same model again. Keying on the text is what makes
 * "saved, then reopened" free while a genuine edit still scores.
 */
const cachedScores = createPromiseCache<LabelScores>(30);

const scoreText = (text: string) =>
  cachedScores(text, () => predictReport(text).then((response) => response.predictions));

const LANGS: Array<{ id: ReportLang; label: string; hint: string }> = [
  { id: "original", label: "As written", hint: "Exactly as the dataset ships it" },
  { id: "en", label: "English", hint: "Machine translation" },
];

/**
 * Which report the panel is showing.
 *
 * "mine" is the doctor's, written here. "dataset" is the radiologist's, shipped
 * with the corpus and read-only — an uploaded study has no such thing, which is
 * why the tab for it only appears when there is one.
 */
type Source = "mine" | "dataset";

interface ReportPanelProps {
  studyUid: string;
  /** The dataset's radiologist report, or null for an uploaded study. */
  dataset: ReportResponse | null;
  /** The doctor's own report as the API last stored it; null until written. */
  stored: StoredReportRecord | null;
  /** Handed the record the save returned, so the page can unlock Fusion. */
  onSaved: (record: StoredReportRecord) => void;
  className?: string;
}

/**
 * The report, and what the report model makes of it.
 *
 * Laid out as a wide row to sit beneath the viewers: prose reads badly in a
 * narrow rail, and putting it under the images is what lets someone read a line
 * of the report and look straight up at the slice it describes. The panel keeps
 * its own height and scrolls inside it, so a long report never pushes the
 * viewers off the screen.
 *
 * THE ORDER THIS ENFORCES. A study arrives as images and nothing else, so the
 * image models can run immediately and there is nothing for the report model to
 * read. The doctor writes the report from those images; saving it translates it
 * server-side and scores it here, without being asked — a report that has to be
 * scored by pressing a button is a report whose score is usually missing, and
 * the fusion model upstairs needs it to exist, not to have been requested.
 */
export function ReportPanel({ studyUid, dataset, stored, onSaved, className }: ReportPanelProps) {
  // Opens on whichever report the study actually has, and on the doctor's when
  // it has both: this panel is somewhere to write, and only incidentally an
  // archive of what the corpus shipped.
  const [source, setSource] = useState<Source>(() => (!stored && dataset ? "dataset" : "mine"));
  const [lang, setLang] = useState<ReportLang>("original");
  const terms = useAsync(loadTerms, []);

  /**
   * The English behind whichever report is showing — what the model reads.
   *
   * Two different provenances, deliberately not merged. The doctor's was
   * translated by the server when it was saved and travels inside the record,
   * so reading it costs nothing. The dataset's comes from the translation cache
   * built once, offline, over the corpus; a study missing from it has no
   * English and therefore no score, which is a real answer rather than a
   * failure.
   */
  const english = useAsync(
    async (signal): Promise<{ text: string; language: string } | null> => {
      if (source === "mine") {
        if (!stored) return null;
        return { text: englishReport(stored), language: stored.language ?? "unknown" };
      }
      if (!dataset) return null;
      if (dataset.language === "en") return { text: dataset.report, language: "en" };
      const translated = await getStudyReport(studyUid, "en", signal).catch(() => null);
      return translated ? { text: translated.report, language: dataset.language } : null;
    },
    [studyUid, source, stored?.updated_at, dataset?.report],
  );

  // No abort signal: a run cancelled halfway leaves nothing in the cache, and
  // flipping between the two reports is exactly the case the cache is for.
  const scores = useAsync(async () => {
    const text = english.data?.text.trim();
    return text ? scoreText(text) : null;
  }, [english.data?.text]);

  return (
    <section
      className={cn(
        "flex shrink-0 flex-col overflow-hidden rounded-2xl border border-border bg-card",
        className,
      )}
    >
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-2.5">
        <div className="flex flex-wrap items-center gap-3">
          <h2 className="text-sm text-foreground">Report</h2>

          {dataset && (
            <Segmented
              options={[
                { id: "mine", label: "Mine", hint: "The report you write from these images" },
                {
                  id: "dataset",
                  label: "Radiologist",
                  hint: "The report the dataset ships with this study",
                },
              ]}
              value={source}
              onChange={(next) => setSource(next as Source)}
            />
          )}

          {source === "dataset" && (
            <Segmented
              options={LANGS}
              value={lang}
              onChange={(next) => setLang(next as ReportLang)}
            />
          )}

          {english.data && english.data.language !== "en" && (
            <span
              className="font-mono text-[10px] text-subtle"
              title="Detected language of the text as written. The model reads the English rendering."
            >
              {english.data.language} → en
            </span>
          )}
        </div>
      </header>

      <div className="flex min-h-0 flex-col gap-4 p-4 lg:flex-row">
        <div className="flex min-w-0 flex-1 flex-col gap-2">
          {source === "mine" ? (
            <MyReportEditor
              studyUid={studyUid}
              stored={stored}
              onSaved={onSaved}
              terms={terms.data ?? []}
            />
          ) : (
            <DatasetReport studyUid={studyUid} lang={lang} terms={terms.data ?? []} />
          )}
        </div>

        {/* Beside the text rather than under it: the point of scoring a report
            is to read the two against each other. */}
        <div className="lg:w-80 lg:shrink-0">
          <ScorePanel
            findings={scores.data}
            // `english` resolving to null means there is nothing to score, and
            // that is already known — without this the empty state flashes a
            // spinner for a request that is never sent.
            loading={english.loading || (Boolean(english.data) && scores.loading)}
            error={scores.error ?? english.error}
            onRetry={scores.reload}
            empty={
              source === "mine"
                ? "Nothing written yet. Save a report and it is translated, scored here by the report model, and — with the images — becomes the fusion model's second half."
                : "No English rendering of this report is cached on the server, so the report model has nothing to read."
            }
          />
        </div>
      </div>
    </section>
  );
}

/** The small pill groups in the header — one shape, three uses. */
function Segmented<T extends string>({
  options,
  value,
  onChange,
}: {
  options: Array<{ id: T; label: string; hint: string }>;
  value: T;
  onChange: (next: T) => void;
}) {
  return (
    <div className="flex gap-1 rounded-lg bg-muted p-0.5">
      {options.map((option) => (
        <button
          key={option.id}
          type="button"
          title={option.hint}
          aria-pressed={value === option.id}
          onClick={() => onChange(option.id)}
          className={cn(
            "rounded-md px-2.5 py-1 text-[11px] font-semibold transition-all",
            value === option.id
              ? "bg-white text-primary shadow-sm"
              : "text-muted-foreground hover:text-primary",
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

/**
 * The doctor's report: a text area, a save, and nothing else to press.
 *
 * The draft is local until saved and re-seeded from the record afterwards, so
 * `updated_at` moving is what ends an edit. That also means a save that fails
 * leaves the text in the box rather than reverting it — the one state in which
 * losing the draft would matter most.
 */
function MyReportEditor({
  studyUid,
  stored,
  onSaved,
  terms,
}: {
  studyUid: string;
  stored: StoredReportRecord | null;
  onSaved: (record: StoredReportRecord) => void;
  terms: string[];
}) {
  const [draft, setDraft] = useState(stored?.text ?? "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDraft(stored?.text ?? "");
  }, [stored?.updated_at]);

  const text = draft.trim();
  const dirty = text !== (stored?.text.trim() ?? "");

  const save = useCallback(async () => {
    const body = draft.trim();
    if (!body || saving) return;
    setSaving(true);
    setError(null);
    try {
      onSaved(await saveStudyReport(studyUid, body));
    } catch (cause) {
      setError(describeError(cause));
    } finally {
      setSaving(false);
    }
  }, [draft, saving, studyUid, onSaved]);

  return (
    <>
      <textarea
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        // The one control on the panel, reachable without leaving the text.
        onKeyDown={(event) => {
          if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
            event.preventDefault();
            void save();
          }
        }}
        spellCheck
        placeholder="Findings from these images…"
        aria-label="Your report for this study"
        className="min-h-32 w-full resize-y rounded-xl border border-border bg-background p-3 text-sm leading-relaxed text-foreground outline-none transition-colors placeholder:text-subtle focus:border-accent-soft"
      />

      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={save} disabled={saving || !text || !dirty} className="px-3 py-1.5 text-xs">
          {saving ? "Saving…" : stored ? "Save changes" : "Save report"}
        </Button>

        {saving ? (
          <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <Spinner className="h-3 w-3" />
            Storing, translating, scoring…
          </span>
        ) : dirty && text ? (
          <span className="text-[11px] text-muted-foreground">
            Unsaved — ⌘/Ctrl + Enter saves.
          </span>
        ) : stored ? (
          <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <Icon name="check" size={11} strokeWidth={3} className="text-emerald-600" />
            Saved {new Date(stored.updated_at).toLocaleString()}
          </span>
        ) : null}
      </div>

      {error && <ErrorState message={error} onRetry={save} />}

      {stored && wasTranslated(stored) && (
        <details className="rounded-xl border border-border-soft bg-background px-3 py-2">
          <summary className="cursor-pointer text-[11px] font-semibold text-muted-foreground">
            What the model read (translated to English)
          </summary>
          <MarkedText text={englishReport(stored)} terms={terms} className="mt-2 max-h-32" />
        </details>
      )}
    </>
  );
}

/** The corpus's own report — read-only, in the dataset's language or in English. */
function DatasetReport({
  studyUid,
  lang,
  terms,
}: {
  studyUid: string;
  lang: ReportLang;
  terms: string[];
}) {
  const report = useAsync((signal) => getStudyReport(studyUid, lang, signal), [studyUid, lang]);

  if (report.loading) return <Loading label="Loading report…" />;
  if (report.error) return <ErrorState message={report.error} onRetry={report.reload} />;
  if (!report.data) return null;
  return <MarkedText text={report.data.report} terms={terms} className="max-h-40" />;
}

/**
 * Report text with the report model's own vocabulary underlined.
 *
 * A mark means "the model counted this term", never "this is why the number
 * came out that way" — the API serves the dictionary, not per-term weights. The
 * dictionary is English, so text in another language simply has nothing to mark.
 */
function MarkedText({
  text,
  terms,
  className,
}: {
  text: string;
  terms: string[];
  className?: string;
}) {
  const segments = useMemo(() => markTerms(text, terms), [text, terms]);
  const marked = segments.filter((segment) => segment.term).length;

  return (
    <div className="flex flex-col gap-1.5">
      <article
        className={cn(
          "overflow-y-auto whitespace-pre-wrap text-sm leading-relaxed text-foreground",
          className,
        )}
      >
        {segments.map((segment, index) =>
          segment.term ? (
            <mark
              key={index}
              className="bg-transparent text-foreground underline decoration-accent decoration-2 underline-offset-2"
            >
              {segment.text}
            </mark>
          ) : (
            <span key={index}>{segment.text}</span>
          ),
        )}
      </article>
      {marked > 0 && (
        <p className="text-[10px] text-subtle">
          <span className="underline decoration-accent decoration-2 underline-offset-2">
            Underlined
          </span>
          : {pluralize(marked, "term")} the report model counts as a feature — what it read, not how
          much each one moved the answer.
        </p>
      )}
    </div>
  );
}

/** The right-hand column: the report model's twelve numbers, or why there are none. */
function ScorePanel({
  findings,
  loading,
  error,
  onRetry,
  empty,
}: {
  findings: LabelScores | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
  empty: string;
}) {
  if (error) return <ErrorState message={error} onRetry={onRetry} />;
  if (loading) {
    return (
      <div className="flex items-center justify-center gap-2 rounded-xl border border-border bg-background px-4 py-8 text-xs text-muted-foreground">
        <Spinner className="h-3.5 w-3.5" />
        Scoring the report…
      </div>
    );
  }
  if (!findings) {
    return (
      <p className="rounded-xl border border-dashed border-border px-4 py-6 text-xs leading-relaxed text-muted-foreground">
        {empty}
      </p>
    );
  }
  return (
    <div className="max-h-40 overflow-y-auto rounded-xl border border-border bg-background p-4">
      <FindingList
        findings={toFindings(findings, { sortByProbability: true })}
        note="The text beside this, scored by the report model — independent of the images."
      />
    </div>
  );
}

export default ReportPanel;

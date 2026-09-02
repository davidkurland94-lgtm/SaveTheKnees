import { useCallback, useMemo, useState } from "react";

import { describeError, getReportTerms, getStudyReport, predictReport } from "@/api";
import type { ReportLang } from "@/interfaces";
import { cn, createPromiseCache, markTerms, pluralize, toFindings, useAsync } from "@/lib";
import { Button, ErrorState, Loading } from "@/components/ui";
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

const LANGS: Array<{ id: ReportLang; label: string; hint: string }> = [
  { id: "original", label: "As written", hint: "Exactly as the dataset ships it" },
  { id: "en", label: "English", hint: "Machine translation" },
];

interface ReportPanelProps {
  studyUid: string;
  className?: string;
}

/**
 * The radiology report, plus an on-demand run of the report model over its text.
 *
 * Laid out as a wide row to sit beneath the viewers: prose reads badly in a
 * narrow rail, and putting it under the images is what lets someone read a line
 * of the report and look straight up at the slice it describes. The panel keeps
 * its own height and scrolls inside it, so a long report never pushes the
 * viewers off the screen.
 *
 * The corpus is mixed-language, so `POST /predict/report` is only offered for the
 * English rendering — that is what the model was trained on.
 */
export function ReportPanel({ studyUid, className }: ReportPanelProps) {
  const [lang, setLang] = useState<ReportLang>("original");
  const [scoring, setScoring] = useState(false);
  const [scores, setScores] = useState<Record<string, number> | null>(null);
  const [scoreError, setScoreError] = useState<string | null>(null);

  const report = useAsync(
    (signal) => getStudyReport(studyUid, lang, signal),
    [studyUid, lang],
  );
  const terms = useAsync(loadTerms, []);

  // The dictionary is English, so the "As written" rendering of a report the
  // dataset ships in another language simply has nothing to mark.
  const segments = useMemo(
    () => markTerms(report.data?.report ?? "", terms.data ?? []),
    [report.data, terms.data],
  );
  const marked = segments.filter((segment) => segment.term).length;

  const runReportModel = useCallback(async () => {
    const text = report.data?.report;
    if (!text) return;
    setScoring(true);
    setScoreError(null);
    try {
      const result = await predictReport(text);
      setScores(result.predictions);
    } catch (cause) {
      setScoreError(describeError(cause));
    } finally {
      setScoring(false);
    }
  }, [report.data]);

  // The frame stays put through every state, so the viewers above it do not
  // resize while the report loads.
  const frame = (children: React.ReactNode) => (
    <section
      className={cn(
        "flex shrink-0 flex-col overflow-hidden rounded-2xl border border-border bg-card",
        className,
      )}
    >
      {children}
    </section>
  );

  if (report.loading) return frame(<Loading label="Loading report…" />);
  if (report.error) {
    return frame(<ErrorState message={report.error} onRetry={report.reload} />);
  }
  if (!report.data) return null;

  return frame(
    <>
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-2.5">
        <div className="flex flex-wrap items-center gap-3">
          <h2 className="text-sm text-foreground">Report</h2>
          <div className="flex gap-1 rounded-lg bg-muted p-0.5">
            {LANGS.map((entry) => (
              <button
                key={entry.id}
                type="button"
                title={entry.hint}
                onClick={() => {
                  setLang(entry.id);
                  setScores(null);
                  setScoreError(null);
                }}
                className={cn(
                  "rounded-md px-2.5 py-1 text-[11px] font-semibold transition-all",
                  lang === entry.id
                    ? "bg-white text-primary shadow-sm"
                    : "text-muted-foreground hover:text-primary",
                )}
              >
                {entry.label}
              </button>
            ))}
          </div>
          <span className="font-mono text-[10px] text-subtle">{report.data.language}</span>
        </div>

        <Button
          onClick={runReportModel}
          disabled={scoring || lang !== "en"}
          title={
            lang === "en"
              ? "Run the report model over this text"
              : "The report model expects English — switch to it first."
          }
          className="px-3 py-1.5 text-xs"
        >
          {scoring ? "Scoring…" : "Score this text"}
        </Button>
      </header>

      <div className="flex min-h-0 flex-col gap-4 p-4 lg:flex-row">
        <div className="flex min-w-0 flex-1 flex-col gap-1.5">
          <article className="max-h-40 overflow-y-auto whitespace-pre-wrap text-sm leading-relaxed text-foreground">
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
              : {pluralize(marked, "term")} the report model counts as a feature — what it read,
              not how much each one moved the answer.
            </p>
          )}
        </div>

        {scoreError && (
          <div className="lg:w-80 lg:shrink-0">
            <ErrorState message={scoreError} onRetry={runReportModel} />
          </div>
        )}

        {/* Beside the text rather than under it: the point of scoring a report
            is to read the two against each other. */}
        {scores && (
          <div className="max-h-40 overflow-y-auto rounded-xl border border-border bg-background p-4 lg:w-80 lg:shrink-0">
            <FindingList
              findings={toFindings(scores, { sortByProbability: true })}
              note="The text beside this, scored by the report model — independent of the images."
            />
          </div>
        )}
      </div>
    </>,
  );
}

export default ReportPanel;

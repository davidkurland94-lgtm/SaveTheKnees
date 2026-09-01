import { useCallback, useState } from "react";

import { describeError, getStudyReport, predictReport } from "@/api";
import type { ReportLang } from "@/interfaces";
import { cn, toFindings, useAsync } from "@/lib";
import { Button, ErrorState, Loading } from "@/components/ui";
import FindingList from "@/components/viewer/FindingList";

const LANGS: Array<{ id: ReportLang; label: string; hint: string }> = [
  { id: "original", label: "As written", hint: "Exactly as the dataset ships it" },
  { id: "en", label: "English", hint: "Machine translation" },
];

interface ReportPanelProps {
  studyUid: string;
}

/**
 * The radiology report, plus an on-demand run of the report model over its text.
 *
 * The corpus is mixed-language, so `POST /predict/report` is only offered for the
 * English rendering — that is what the model was trained on.
 */
export function ReportPanel({ studyUid }: ReportPanelProps) {
  const [lang, setLang] = useState<ReportLang>("original");
  const [scoring, setScoring] = useState(false);
  const [scores, setScores] = useState<Record<string, number> | null>(null);
  const [scoreError, setScoreError] = useState<string | null>(null);

  const report = useAsync(
    (signal) => getStudyReport(studyUid, lang, signal),
    [studyUid, lang],
  );

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

  if (report.loading) return <Loading label="Loading report…" />;
  if (report.error) return <ErrorState message={report.error} onRetry={report.reload} />;
  if (!report.data) return null;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex gap-1 rounded-xl bg-muted p-1">
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
                "rounded-lg px-3 py-1.5 text-xs font-semibold transition-all",
                lang === entry.id
                  ? "bg-white text-primary shadow-sm"
                  : "text-muted-foreground hover:text-primary",
              )}
            >
              {entry.label}
            </button>
          ))}
        </div>

        <Button
          onClick={runReportModel}
          disabled={scoring || lang !== "en"}
          title={
            lang === "en"
              ? "POST /predict/report"
              : "The report model expects English — switch the tab first."
          }
          className="px-3 py-2 text-xs"
        >
          {scoring ? "Scoring…" : "Score this text"}
        </Button>
      </div>

      <article className="max-h-[28rem] overflow-y-auto whitespace-pre-wrap rounded-2xl border border-border bg-card px-5 py-4 text-sm leading-relaxed text-foreground">
        {report.data.report}
      </article>
      <p className="text-xs text-muted-foreground">
        Language: <span className="font-mono">{report.data.language}</span>
      </p>

      {scoreError && <ErrorState message={scoreError} onRetry={runReportModel} />}

      {scores && (
        <div className="rounded-2xl border border-border p-5">
          <h3 className="mb-4 text-base text-foreground">Report model output</h3>
          <FindingList
            findings={toFindings(scores)}
            note="POST /predict/report — the text above scored by the report model, independent of the images."
          />
        </div>
      )}
    </div>
  );
}

export default ReportPanel;

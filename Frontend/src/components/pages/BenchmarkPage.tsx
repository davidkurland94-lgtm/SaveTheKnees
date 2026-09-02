import { useState } from "react";
import { Link } from "react-router";

import { getReportTable, getReportVerdicts } from "@/api";
import type { ReportTableRow } from "@/interfaces";
import { cn, patientOf, paths, useAsync } from "@/lib";
import { ErrorState, Icon, Loading, NavBar } from "@/components/ui";

type Tab = "table" | "verdicts";

/** Column order for the AUC sheet, with readable headings. */
const READERS: Array<{ key: keyof ReportTableRow; label: string }> = [
  { key: "positives", label: "Pos" },
  { key: "llm_v4_blend", label: "LLM v4" },
  { key: "llm_full", label: "LLM full" },
  { key: "llm_v2", label: "LLM v2" },
  { key: "pilkwang_v2", label: "Pilkwang" },
  { key: "image_model", label: "Image" },
  { key: "image_multiplane", label: "Multiplane" },
  { key: "report_model", label: "Report" },
  { key: "report_bagged", label: "Report bag" },
  { key: "fusion_model", label: "Fusion" },
];

export function BenchmarkPage() {
  const [tab, setTab] = useState<Tab>("table");

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

      <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-6 px-6 py-8">
        <header>
          <h1 className="text-2xl text-foreground">Benchmark</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Every reader scored against the 58 hand-labelled studies.
          </p>
        </header>

        <nav className="flex w-fit gap-1 rounded-xl bg-muted p-1">
          {(
            [
              { id: "table", label: "AUC by label" },
              { id: "verdicts", label: "Report vs images" },
            ] as Array<{ id: Tab; label: string }>
          ).map((entry) => (
            <button
              key={entry.id}
              type="button"
              onClick={() => setTab(entry.id)}
              className={cn(
                "rounded-lg px-4 py-2 text-xs font-semibold transition-all",
                tab === entry.id
                  ? "bg-white text-primary shadow-sm"
                  : "text-muted-foreground hover:text-primary",
              )}
            >
              {entry.label}
            </button>
          ))}
        </nav>

        {tab === "table" ? <AucTable /> : <VerdictsTable />}
      </div>
    </div>
  );
}

function AucTable() {
  const table = useAsync((signal) => getReportTable(signal), []);

  if (table.loading) return <Loading label="Loading the benchmark sheet…" />;
  if (table.error) return <ErrorState message={table.error} onRetry={table.reload} />;

  const rows = table.data?.rows ?? [];

  return (
    <div className="overflow-x-auto rounded-2xl border border-border">
      <table className="w-full min-w-[52rem] text-sm">
        <thead>
          <tr className="border-b border-border bg-card">
            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Label
            </th>
            {READERS.map((reader) => (
              <th
                key={reader.key}
                className="px-3 py-3 text-right text-xs font-semibold uppercase tracking-wide text-muted-foreground"
              >
                {reader.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => {
            // Highlight the strongest reader on each row (the count column aside).
            const best = Math.max(...READERS.slice(1).map((r) => Number(row[r.key])));
            return (
              <tr
                key={row.label}
                className={cn(
                  "transition-colors hover:bg-card",
                  index < rows.length - 1 && "border-b border-border-soft",
                )}
              >
                <td className="px-4 py-2.5 font-medium text-foreground">{row.label}</td>
                {READERS.map((reader) => {
                  const value = row[reader.key];
                  const isCount = reader.key === "positives";
                  return (
                    <td
                      key={reader.key}
                      className={cn(
                        "px-3 py-2.5 text-right tabular-nums",
                        isCount
                          ? "text-muted-foreground"
                          : value === best
                            ? "font-semibold text-primary"
                            : "text-foreground",
                      )}
                    >
                      {/* The summary row carries a null count. */}
                      {typeof value !== "number" ? "—" : isCount ? value : value.toFixed(3)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function VerdictsTable() {
  const [top, setTop] = useState(20);
  const verdicts = useAsync((signal) => getReportVerdicts(top, signal), [top]);

  if (verdicts.loading) return <Loading label="Loading verdicts…" />;
  if (verdicts.error) return <ErrorState message={verdicts.error} onRetry={verdicts.reload} />;

  const rows = verdicts.data?.rows ?? [];

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <label htmlFor="verdict-top" className="text-xs text-muted-foreground">
          Worst
        </label>
        <select
          id="verdict-top"
          value={top}
          onChange={(event) => setTop(Number(event.target.value))}
          className="rounded-lg border border-border bg-card px-2 py-1 text-xs text-foreground"
        >
          {[10, 20, 50, 100].map((value) => (
            <option key={value} value={value}>
              {value}
            </option>
          ))}
        </select>
        <span className="text-xs text-muted-foreground">reports first</span>
      </div>

      <div className="overflow-x-auto rounded-2xl border border-border">
        <table className="w-full min-w-[56rem] text-sm">
          <thead>
            <tr className="border-b border-border bg-card">
              {["Study", "Report says", "Images say", "Missed", "Quality"].map((heading) => (
                <th
                  key={heading}
                  className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground"
                >
                  {heading}
                </th>
              ))}
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
                <td className="px-4 py-3">
                  <PatientCell uid={row.StudyInstanceUID} />
                </td>
                <td className="max-w-56 px-4 py-3 text-xs text-foreground">{row.report_says}</td>
                <td className="max-w-56 px-4 py-3 text-xs text-muted-foreground">
                  {row.images_say}
                </td>
                <td className="max-w-48 px-4 py-3 text-xs text-red-600">{row["missed?"]}</td>
                <td className="px-4 py-3 text-xs tabular-nums text-foreground">
                  {row.quality_score.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
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

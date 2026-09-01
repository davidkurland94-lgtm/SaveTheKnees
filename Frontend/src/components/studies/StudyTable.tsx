import { useMemo, useState } from "react";

import type { GoldenStudy } from "@/interfaces";
import { positiveLabels, shortUid } from "@/lib";
import { EmptyState, Icon, Tag } from "@/components/ui";

const COLUMNS = ["Study UID", "Positive findings", "Count", ""];

interface StudyTableProps {
  studies: GoldenStudy[];
  onOpen: (studyUid: string) => void;
}

/** The 58 hand-labelled studies, filtered by UID or by finding name. */
export function StudyTable({ studies, onOpen }: StudyTableProps) {
  const [search, setSearch] = useState("");

  const rows = useMemo(() => {
    const enriched = studies.map((study) => ({
      study,
      positives: positiveLabels(study.labels),
    }));

    const query = search.trim().toLowerCase();
    if (!query) return enriched;

    return enriched.filter(
      ({ study, positives }) =>
        study.study_uid.toLowerCase().includes(query) ||
        positives.some((label) => label.toLowerCase().includes(query)),
    );
  }, [studies, search]);

  return (
    <section className="flex flex-col gap-4">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="text-xl text-foreground">Golden dataset</h2>
          <p className="mt-0.5 text-xs text-muted-foreground">
            The {studies.length} hand-labelled studies — the only ground truth in the dataset.
          </p>
        </div>

        <div className="flex w-64 items-center gap-2 rounded-xl border border-border bg-card px-3 py-2">
          <Icon name="search" size={14} className="text-accent" />
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search UID or finding…"
            aria-label="Search studies"
            className="flex-1 bg-transparent text-xs text-foreground outline-none placeholder:text-subtle"
          />
        </div>
      </div>

      {rows.length === 0 ? (
        <EmptyState message="No studies match that search." />
      ) : (
        <div className="overflow-hidden rounded-2xl border border-border">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-card">
                {COLUMNS.map((heading) => (
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
              {rows.map(({ study, positives }, index) => (
                <tr
                  key={study.study_uid}
                  className={
                    index === rows.length - 1
                      ? "transition-colors hover:bg-card"
                      : "border-b border-border-soft transition-colors hover:bg-card"
                  }
                >
                  <td className="px-4 py-3">
                    <span
                      className="font-mono text-xs text-foreground"
                      title={study.study_uid}
                    >
                      {shortUid(study.study_uid, 18)}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    {positives.length === 0 ? (
                      <span className="text-xs text-muted-foreground">No positive findings</span>
                    ) : (
                      <div className="flex flex-wrap gap-1">
                        {positives.map((label) => (
                          <Tag key={label}>{label}</Tag>
                        ))}
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-3 tabular-nums text-muted-foreground">
                    {positives.length}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <button
                      type="button"
                      onClick={() => onOpen(study.study_uid)}
                      className="text-xs font-semibold text-primary hover:text-primary-hover hover:underline"
                    >
                      Open →
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export default StudyTable;

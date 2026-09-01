import { useMemo, useState } from "react";
import { Link } from "react-router";

import type { StudyListEntry } from "@/interfaces";
import { cn, paths, shortUid } from "@/lib";
import { EmptyState, Icon } from "@/components/ui";

const COLUMNS = ["Study UID", "", ""];

interface StudyTableProps {
  /** One page of `GET /studies`, already fetched by the page. */
  studies: StudyListEntry[];
  /** Studies in the whole corpus, for the page count. */
  total: number;
  /** Zero-based. */
  page: number;
  pageSize: number;
  /** True while the next page is in flight; the current rows stay up. */
  loading?: boolean;
  onPage: (page: number) => void;
  className?: string;
}

/**
 * The corpus, one page at a time.
 *
 * `GET /studies` is paginated server-side and the whole list is 4,407 rows, so
 * the page index is the source of truth and only twenty rows exist in the
 * browser at once. The table body scrolls inside its own box rather than
 * growing the page, which is what keeps the upload zone above it always
 * reachable.
 */
export function StudyTable({
  studies,
  total,
  page,
  pageSize,
  loading = false,
  onPage,
  className,
}: StudyTableProps) {
  const [search, setSearch] = useState("");

  // Deliberately scoped to the rows in hand: `GET /studies` takes limit and
  // offset but no query, so anything wider than this would need a server-side
  // search to exist first. The placeholder says so rather than implying a
  // corpus-wide hit.
  const rows = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) return studies;
    return studies.filter((entry) => entry.study_uid.toLowerCase().includes(query));
  }, [studies, search]);

  const pages = Math.max(1, Math.ceil(total / pageSize));
  const first = total === 0 ? 0 : page * pageSize + 1;
  const last = Math.min((page + 1) * pageSize, total);

  return (
    <section className={cn("flex min-h-0 flex-col gap-4", className)}>
      <div className="flex shrink-0 items-center justify-between gap-4">
        <div>
          <h2 className="text-xl text-foreground">Studies</h2>
          <p className="mt-0.5 text-xs text-muted-foreground">
            All {total.toLocaleString()} studies in the corpus. Ground-truth rows are the 58
            hand-labelled ones.
          </p>
        </div>

        <div className="flex w-64 shrink-0 items-center gap-2 rounded-xl border border-border bg-card px-3 py-2">
          <Icon name="search" size={14} className="text-accent" />
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Filter this page by UID…"
            aria-label="Filter the studies on this page"
            className="flex-1 bg-transparent text-xs text-foreground outline-none placeholder:text-subtle"
          />
        </div>
      </div>

      <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-border">
        <div className="min-h-0 flex-1 overflow-y-auto">
          {rows.length === 0 ? (
            <EmptyState
              message={loading ? "Loading studies…" : "No study on this page matches that filter."}
            />
          ) : (
            // border-separate, not the default collapse: a collapsed border on a
            // sticky header does not travel with it, so the heading row loses its
            // underline the moment the body scrolls.
            <table
              className={cn(
                "w-full border-separate border-spacing-0 text-sm transition-opacity",
                loading && "opacity-50",
              )}
            >
              <thead>
                <tr>
                  {COLUMNS.map((heading, index) => (
                    <th
                      key={index}
                      className="sticky top-0 z-10 border-b border-border bg-card px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground"
                    >
                      {heading}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((entry) => (
                  <tr key={entry.study_uid} className="transition-colors hover:bg-card">
                    <td className="border-b border-border-soft px-4 py-3">
                      <span className="font-mono text-xs text-foreground" title={entry.study_uid}>
                        {shortUid(entry.study_uid, 24)}
                      </span>
                    </td>
                    <td className="border-b border-border-soft px-4 py-3">
                      <div className="flex flex-wrap gap-1">
                        {entry.golden && (
                          <span className="inline-flex items-center rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-[10px] font-medium text-amber-700">
                            Ground truth
                          </span>
                        )}
                        {entry.has_report && (
                          <span className="inline-flex items-center rounded-full bg-secondary px-2 py-0.5 text-[10px] font-medium text-secondary-foreground">
                            Report
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="border-b border-border-soft px-4 py-3 text-right">
                      {/* A link, not a button: a study is a URL, so this one
                          opens in a new tab on cmd-click like any other. */}
                      <Link
                        to={paths.study(entry.study_uid)}
                        className="text-xs font-semibold text-primary hover:text-primary-hover hover:underline"
                      >
                        Open →
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <div className="flex shrink-0 items-center justify-between gap-4 border-t border-border bg-card px-4 py-2.5">
          <span className="text-xs tabular-nums text-muted-foreground">
            {first.toLocaleString()}–{last.toLocaleString()} of {total.toLocaleString()}
          </span>
          <div className="flex items-center gap-1">
            <PageButton label="Previous" disabled={page === 0} onClick={() => onPage(page - 1)}>
              <Icon name="arrow-left" size={13} />
            </PageButton>
            <span className="px-2 text-xs tabular-nums text-muted-foreground">
              Page {(page + 1).toLocaleString()} of {pages.toLocaleString()}
            </span>
            <PageButton
              label="Next"
              disabled={page + 1 >= pages}
              onClick={() => onPage(page + 1)}
            >
              <Icon name="arrow-left" size={13} className="rotate-180" />
            </PageButton>
          </div>
        </div>
      </div>
    </section>
  );
}

function PageButton({
  label,
  disabled,
  onClick,
  children,
}: {
  label: string;
  disabled: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      disabled={disabled}
      onClick={onClick}
      className="rounded-lg border border-border p-1.5 text-muted-foreground transition-colors hover:border-accent-soft hover:text-primary disabled:pointer-events-none disabled:opacity-30"
    >
      {children}
    </button>
  );
}

export default StudyTable;

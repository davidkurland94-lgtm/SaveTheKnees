import { useState } from "react";
import { Link } from "react-router";

import { getHealth, getStudies } from "@/api";
import { paths, useAsync, useUploadState } from "@/lib";
import { ErrorState, Icon, Loading, NavBar } from "@/components/ui";
import StudyTable from "@/components/studies/StudyTable";
import UploadStudyButton from "@/components/upload/UploadStudyButton";

const PAGE_SIZE = 20;

export function HomePage() {
  const { start, error, dismissError } = useUploadState();
  const [page, setPage] = useState(0);
  const studies = useAsync(
    (signal) => getStudies(PAGE_SIZE, page * PAGE_SIZE, signal),
    [page],
  );
  const health = useAsync((signal) => getHealth(signal), []);

  return (
    <div className="flex h-full flex-col overflow-hidden bg-background">
      <NavBar>
        {/* In the bar, not in the table: adding a study is the one thing that
            must stay reachable when the list itself fails to load. */}
        <UploadStudyButton onFiles={start} />
        <Link
          to={paths.benchmark}
          className="flex items-center gap-1.5 rounded-full border border-border px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:border-accent-soft hover:text-primary"
        >
          <Icon name="chart" size={13} />
          Benchmark
        </Link>
      </NavBar>

      <div className="mx-auto flex w-full min-h-0 max-w-7xl flex-1 flex-col gap-8 px-6 py-6">
        {/* A failed upload lands back here, so this is where it reports. */}
        {error && (
          <div className="shrink-0">
            <ErrorState message={error} onRetry={dismissError} />
          </div>
        )}

        {studies.error ? (
          <ErrorState message={studies.error} onRetry={studies.reload} />
        ) : studies.data ? (
          <StudyTable
            studies={studies.data.studies}
            total={studies.data.total}
            page={page}
            pageSize={PAGE_SIZE}
            loading={studies.loading}
            onPage={setPage}
            className="flex-1"
          />
        ) : (
          <Loading label="Loading studies…" />
        )}
      </div>

      <HealthPill
        loading={health.loading}
        status={health.error ? "unreachable" : health.data?.status}
      />
    </div>
  );
}

/**
 * Whether `GET /health` answers, parked in the corner of the viewport.
 *
 * It sits below the nav bar in the stack (z-1 against its z-10) because the two
 * never meet — one is pinned to the top of the window and this to the bottom —
 * and anything the app raises later should be able to cover it.
 */
function HealthPill({ loading, status }: { loading: boolean; status?: string }) {
  const ok = status === "ok";
  return (
    <div className="fixed bottom-4 right-4 z-[1] flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5 shadow-md">
      <span
        className={
          loading
            ? "h-2 w-2 animate-pulse rounded-full bg-accent-soft"
            : ok
              ? "h-2 w-2 rounded-full bg-emerald-500"
              : "h-2 w-2 rounded-full bg-red-500"
        }
      />
      <span className="text-xs font-semibold text-foreground">
        {loading ? "Checking API…" : ok ? "API online" : "API unreachable"}
      </span>
    </div>
  );
}

export default HomePage;

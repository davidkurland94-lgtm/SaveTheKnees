import { useState } from "react";

import { getHealth, getStudies } from "@/api";
import { useAsync } from "@/lib";
import { ErrorState, Icon, Loading, NavBar } from "@/components/ui";
import StudyTable from "@/components/studies/StudyTable";
import UploadZone from "@/components/upload/UploadZone";

const PAGE_SIZE = 20;

interface HomePageProps {
  onUpload: (files: File[]) => void;
  onOpenStudy: (studyUid: string) => void;
  onOpenBenchmark: () => void;
}

export function HomePage({ onUpload, onOpenStudy, onOpenBenchmark }: HomePageProps) {
  const [page, setPage] = useState(0);
  const studies = useAsync(
    (signal) => getStudies(PAGE_SIZE, page * PAGE_SIZE, signal),
    [page],
  );
  const health = useAsync((signal) => getHealth(signal), []);

  return (
    <div className="flex h-full flex-col overflow-hidden bg-background">
      <NavBar>
        <HealthPill
          loading={health.loading}
          status={health.error ? "unreachable" : health.data?.status}
        />
        <button
          type="button"
          onClick={onOpenBenchmark}
          className="flex items-center gap-1.5 rounded-full border border-border px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:border-accent-soft hover:text-primary"
        >
          <Icon name="chart" size={13} />
          Benchmark
        </button>
      </NavBar>

      <div className="mx-auto flex w-full min-h-0 max-w-7xl flex-1 flex-col gap-8 px-6 py-6">
        <div className="shrink-0">
          <UploadZone onFiles={onUpload} />
        </div>

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
            onOpen={onOpenStudy}
            className="flex-1"
          />
        ) : (
          <Loading label="Loading studies…" />
        )}
      </div>
    </div>
  );
}

function HealthPill({ loading, status }: { loading: boolean; status?: string }) {
  const ok = status === "ok";
  return (
    <div className="flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5">
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

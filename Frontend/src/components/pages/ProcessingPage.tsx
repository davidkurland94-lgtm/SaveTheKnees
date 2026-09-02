import { Navigate } from "react-router";

import { cn, paths, pluralize, useUploadState } from "@/lib";
import { Icon, NavBar } from "@/components/ui";

const STEPS = [
  "Reading the folder",
  "Storing the study and pre-filling its labels",
];

/**
 * Progress while a dropped folder becomes a stored study. The step is driven by
 * the actual upload pipeline rather than a timer, so it cannot claim work that
 * has not happened — which is why there are two steps and not four: the server
 * splits the folder into series and runs the model inside one request, and this
 * screen cannot see where inside it the work has got to.
 */
export function ProcessingPage() {
  const { fileCount, step, studyUid, error } = useUploadState();

  // Addressable but not resumable: the files being scored only exist in the tab
  // that dropped them. `fileCount` is the one thing a run sets before it
  // navigates here, so it is what says whether this tab has a run at all.
  if (fileCount === 0 || error) return <Navigate to={paths.home} replace />;
  // The run this screen was following already finished — a reopened /processing
  // should not sit on a spinner for work that is done.
  if (studyUid) return <Navigate to={paths.study(studyUid)} replace />;

  return (
    <div className="flex min-h-full flex-col bg-background">
      <NavBar />
      <div className="flex flex-1 items-center justify-center px-6">
        <div className="flex w-full max-w-md flex-col items-center gap-8 text-center">
          <div className="relative h-20 w-20">
            <svg className="h-full w-full animate-spin" viewBox="0 0 80 80" aria-hidden="true">
              <circle cx="40" cy="40" r="34" fill="none" stroke="#ede9fe" strokeWidth="6" />
              <circle
                cx="40"
                cy="40"
                r="34"
                fill="none"
                stroke="#7c3aed"
                strokeWidth="6"
                strokeLinecap="round"
                strokeDasharray="50 164"
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center text-primary">
              <Icon name="images" size={24} strokeWidth={1.5} />
            </div>
          </div>

          <div>
            <h2 className="mb-2 text-2xl text-foreground">Adding study</h2>
            <p className="text-sm text-muted-foreground">{pluralize(fileCount, "file")}</p>
          </div>

          <ol className="flex w-full flex-col gap-2">
            {STEPS.map((text, index) => {
              const done = index < step;
              const active = index === step;
              return (
                <li
                  key={text}
                  className={cn(
                    "flex items-center gap-3 rounded-xl border px-4 py-3 transition-all",
                    active
                      ? "border-accent-soft bg-muted"
                      : done
                        ? "border-border bg-white"
                        : "border-border-soft bg-white opacity-40",
                  )}
                >
                  <span
                    className={cn(
                      "flex h-5 w-5 shrink-0 items-center justify-center rounded-full",
                      done ? "bg-primary text-white" : active ? "bg-secondary" : "bg-border-soft",
                    )}
                  >
                    {done ? (
                      <Icon name="check" size={10} strokeWidth={3} />
                    ) : (
                      <span
                        className={cn(
                          "h-2 w-2 rounded-full",
                          active ? "animate-pulse bg-primary" : "bg-accent-soft",
                        )}
                      />
                    )}
                  </span>
                  <span
                    className={cn(
                      "text-sm",
                      active
                        ? "font-semibold text-primary"
                        : done
                          ? "text-foreground"
                          : "text-subtle",
                    )}
                  >
                    {text}
                  </span>
                </li>
              );
            })}
          </ol>
        </div>
      </div>
    </div>
  );
}

export default ProcessingPage;

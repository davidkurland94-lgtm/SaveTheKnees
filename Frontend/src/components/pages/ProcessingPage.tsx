import { cn, pluralize } from "@/lib";
import { Icon, NavBar } from "@/components/ui";

/** Index of the step currently running. */
export type ProcessingStep = 0 | 1 | 2;

const STEPS = [
  "Decoding DICOM slices in the browser",
  "Uploading the series to /predict",
  "Running the image model",
];

interface ProcessingPageProps {
  fileCount: number;
  step: ProcessingStep;
}

/**
 * Progress while an uploaded series is scored. The step is driven by the actual
 * upload pipeline rather than a timer, so it cannot claim work that has not
 * happened.
 */
export function ProcessingPage({ fileCount, step }: ProcessingPageProps) {
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
            <h2 className="mb-2 text-2xl text-foreground">Analysing series</h2>
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

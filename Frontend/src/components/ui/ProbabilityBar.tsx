import { percent, SEVERITY_COLOR, severityOf } from "@/lib";

interface ProbabilityBarProps {
  value: number;
  /** Shows a ground-truth notch so model output can be read against the label. */
  truth?: 0 | 1;
}

export function ProbabilityBar({ value, truth }: ProbabilityBarProps) {
  const clamped = Math.max(0, Math.min(1, value));
  const color = SEVERITY_COLOR[severityOf(clamped)];

  return (
    <div className="flex items-center gap-3">
      <div
        className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-muted"
        role="meter"
        aria-valuenow={Math.round(clamped * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: percent(clamped), backgroundColor: color }}
        />
        {truth === 1 && (
          <span
            className="absolute inset-y-0 right-0 w-0.5 bg-emerald-500"
            title="Positive in the ground-truth labels"
          />
        )}
      </div>
      <span
        className="w-9 text-right text-xs font-semibold tabular-nums"
        style={{ color }}
      >
        {percent(clamped)}
      </span>
    </div>
  );
}

export default ProbabilityBar;

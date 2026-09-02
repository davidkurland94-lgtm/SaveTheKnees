import type { Finding } from "@/interfaces";
import { ProbabilityBar, SeverityBadge, VerdictBadge } from "@/components/ui";

interface FindingListProps {
  findings: Finding[];
  /** Rendered above the list — where the numbers came from. */
  note?: string;
}

/** The shared renderer for any set of twelve label probabilities. */
export function FindingList({ findings, note }: FindingListProps) {
  return (
    <div className="flex flex-col gap-5">
      {note && (
        <div className="flex items-start gap-2">
          <div className="min-h-8 w-1 shrink-0 rounded-full bg-gradient-to-b from-primary to-accent" />
          <p className="text-xs leading-relaxed text-muted-foreground">{note}</p>
        </div>
      )}

      <div className="flex flex-col gap-4">
        {findings.map((finding) => (
          <div key={finding.label} className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-medium text-foreground">{finding.label}</span>
              <div className="flex items-center gap-1.5">
                {finding.verdict && <VerdictBadge verdict={finding.verdict} />}
                <SeverityBadge severity={finding.severity} />
              </div>
            </div>
            <ProbabilityBar value={finding.probability} truth={finding.truth} />
            {finding.truth !== undefined && (
              <p className="text-[10px] text-muted-foreground">
                Ground truth: {finding.truth === 1 ? "positive" : "negative"}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default FindingList;

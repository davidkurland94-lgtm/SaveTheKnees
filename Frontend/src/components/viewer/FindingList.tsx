import type { Finding, Severity } from "@/interfaces";
import { pluralize } from "@/lib";
import { Icon, SeverityBadge, Tag } from "@/components/ui";

/**
 * The flagged findings, and nothing else.
 *
 * Meant to be rendered outside the scrolling list and above it, because it is
 * the one thing on this panel a reader must not be able to miss. The model has
 * already made its call on everything in the list below; these are the ones it
 * could not, so these are the ones that need a person.
 *
 * It renders when there is nothing to flag too. "No flags" is a real result and
 * a useful one — silence would leave a reader wondering whether the panel had
 * finished thinking.
 */
export function AttentionFlag({ findings }: { findings: Finding[] }) {
  if (findings.length === 0) {
    return (
      <div className="flex shrink-0 items-center gap-2 rounded-xl border border-emerald-100 bg-emerald-50 px-3 py-2">
        <Icon name="check" size={14} strokeWidth={2.5} className="shrink-0 text-emerald-600" />
        <p className="text-xs font-medium text-emerald-800">
          Nothing flagged — the model called every finding either way.
        </p>
      </div>
    );
  }

  return (
    <div className="flex shrink-0 flex-col gap-2 rounded-xl border-2 border-amber-300 bg-amber-50 px-3 py-2.5">
      <div className="flex items-center gap-2">
        <Icon name="alert" size={16} strokeWidth={2.5} className="shrink-0 text-amber-600" />
        <p className="text-sm font-semibold text-amber-900">
          {pluralize(findings.length, "finding")} need
          {findings.length === 1 ? "s" : ""} your review
        </p>
      </div>
      <ul className="flex flex-wrap gap-1.5">
        {findings.map((finding) => (
          <li
            key={finding.label}
            className="rounded-full border border-amber-300 bg-white px-2 py-0.5 text-xs font-medium text-amber-900"
          >
            {finding.label}
            {finding.truth !== undefined && (
              <span className="ml-1 font-normal text-amber-700/70">
                · truth {finding.truth === 1 ? "positive" : "negative"}
              </span>
            )}
          </li>
        ))}
      </ul>
      <p className="text-[11px] leading-relaxed text-amber-800/80">
        The model could not call these one way or the other. Everything below it was sure about.
      </p>
    </div>
  );
}

/** One band of findings, under a heading that carries the verdict for all of them. */
function FindingGroup({
  severity,
  findings,
  empty,
}: {
  severity: Severity;
  findings: Finding[];
  /** Shown in place of the rows, so an empty band still says something. */
  empty: string;
}) {
  return (
    <section className="flex flex-col gap-2">
      {/* The verdict is the heading. Carried once for the whole band rather than
          repeated on every row, which is what lets twelve findings read as two
          short lists instead of twelve things to compare. */}
      <h3 className="flex items-center gap-2">
        <SeverityBadge severity={severity} />
        <span className="text-xs tabular-nums text-subtle">{findings.length}</span>
      </h3>

      {findings.length === 0 ? (
        <p className="pl-4 text-xs text-subtle">{empty}</p>
      ) : (
        <ul className="flex flex-col">
          {findings.map((finding) => (
            <li
              key={finding.label}
              className="flex items-center justify-between gap-2 border-b border-border/60 py-1.5 last:border-b-0"
            >
              <span className="text-sm text-foreground">{finding.label}</span>
              {finding.truth !== undefined && (
                <Tag>truth {finding.truth === 1 ? "positive" : "negative"}</Tag>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

interface FindingListProps {
  /** Everything the model was sure about; the flagged band belongs to `AttentionFlag`. */
  groups: Record<Severity, Finding[]>;
  /** Rendered above the list — where the numbers came from. */
  note?: string;
}

/**
 * The findings the model did call, grouped by which way it called them.
 *
 * No bars and no percentages. The verdict is carried once by the heading rather
 * than repeated on every row, which is what makes twelve findings readable as
 * two short lists instead of twelve numbers to compare.
 */
export function FindingList({ groups, note }: FindingListProps) {
  return (
    <div className="flex flex-col gap-5">
      {note && (
        <div className="flex items-start gap-2">
          <div className="min-h-8 w-1 shrink-0 rounded-full bg-gradient-to-b from-primary to-accent" />
          <p className="text-xs leading-relaxed text-muted-foreground">{note}</p>
        </div>
      )}

      <FindingGroup severity="high" findings={groups.high} empty="No injury called." />
      <FindingGroup severity="low" findings={groups.low} empty="Nothing came back clear." />
    </div>
  );
}

export default FindingList;

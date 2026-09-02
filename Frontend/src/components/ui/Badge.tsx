import { cn, SEVERITY_LABEL, VERDICT_TONE } from "@/lib";
import type { Severity, Verdict } from "@/interfaces";

const SEVERITY_TONE: Record<Severity, string> = {
  high: "bg-red-50 text-red-600 border-red-100",
  moderate: "bg-amber-50 text-amber-600 border-amber-100",
  low: "bg-muted text-muted-foreground border-border",
};

const BASE = "inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium";

export function SeverityBadge({ severity }: { severity: Severity }) {
  return <span className={cn(BASE, SEVERITY_TONE[severity])}>{SEVERITY_LABEL[severity]}</span>;
}

export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  return <span className={cn(BASE, "font-mono", VERDICT_TONE[verdict])}>{verdict}</span>;
}

/** Marks a study as part of the 58-study hand-labelled set. */
export function GoldenBadge() {
  return (
    <span className={cn(BASE, "border-amber-200 bg-amber-50 text-amber-700")}>Ground truth</span>
  );
}

export function Tag({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center rounded-full bg-secondary px-2 py-0.5 text-[10px] font-medium text-secondary-foreground">
      {children}
    </span>
  );
}

const CHIP = "inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium";

interface ChipProps {
  children: React.ReactNode;
  /** Given, the chip becomes a button; omitted, it stays a plain label. */
  onClick?: () => void;
  /** Only meaningful on a clickable chip: whether what it opens is open. */
  active?: boolean;
  title?: string;
}

/**
 * A small rounded fact. Clickable chips are real buttons and carry
 * `aria-pressed`, so the state a chip shows with colour is also announced.
 */
export function Chip({ children, onClick, active = false, title }: ChipProps) {
  const className = cn(
    CHIP,
    active
      ? "border-accent bg-secondary text-secondary-foreground"
      : "border-border bg-white text-muted-foreground",
    onClick && "cursor-pointer transition-colors hover:border-accent-soft hover:text-primary",
  );

  if (!onClick) {
    return (
      <span className={className} title={title}>
        {children}
      </span>
    );
  }

  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      title={title}
      className={className}
    >
      {children}
    </button>
  );
}

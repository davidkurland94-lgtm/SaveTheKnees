/** The bits of viewer furniture `Dicom3DViewer` and `DicomMprViewer` share. */

import type { ViewerStack } from "@/interfaces";
import { cn } from "@/lib";
import { Icon, type IconName } from "@/components/ui";

interface StackTabsProps {
  stacks: ViewerStack[];
  activeId: string;
  onSelect: (id: string) => void;
  /** Right-aligned read-out, e.g. the sequence details. */
  detail?: string;
}

/**
 * The axis picker. A study contributing one stack per plane reads as Sagittal /
 * Coronal / Axial tabs; where a plane repeats, the page disambiguates through
 * each stack's `label`.
 */
export function StackTabs({ stacks, activeId, onSelect, detail }: StackTabsProps) {
  return (
    <div className="flex flex-wrap items-center gap-1 border-b border-white/5 bg-viewer-panel px-3 py-2">
      {stacks.map((stack, position) => {
        const active = stack.id === activeId;
        return (
          <button
            key={stack.id}
            type="button"
            title={stack.label}
            aria-pressed={active}
            onClick={() => onSelect(stack.id)}
            className={cn(
              "rounded-lg px-3 py-1 text-[11px] font-semibold transition-colors",
              active
                ? "bg-accent/20 text-accent"
                : "text-white/40 hover:bg-white/5 hover:text-white/70",
            )}
          >
            {stack.label ?? stack.plane ?? `Series ${position + 1}`}
          </button>
        );
      })}
      {detail && (
        <span className="ml-auto truncate pl-2 font-mono text-[11px] text-white/30">{detail}</span>
      )}
    </div>
  );
}

/** One choice in a `SegmentedTabs` row. */
export interface SegmentedOption<T extends string> {
  id: T;
  label: string;
  /** Tooltip; the place to say what the choice actually does. */
  hint?: string;
}

interface SegmentedTabsProps<T extends string> {
  options: ReadonlyArray<SegmentedOption<T>>;
  active: T;
  onSelect: (id: T) => void;
}

/**
 * A row of small exclusive buttons — render modes, view switches, the tool
 * palette. Styled to match `StackTabs`, which is the same idea one level up.
 */
export function SegmentedTabs<T extends string>({
  options,
  active,
  onSelect,
}: SegmentedTabsProps<T>) {
  return (
    <>
      {options.map((option) => (
        <button
          key={option.id}
          type="button"
          title={option.hint}
          aria-pressed={active === option.id}
          onClick={() => onSelect(option.id)}
          className={cn(
            "rounded-lg px-3 py-1 text-[11px] font-semibold transition-colors",
            active === option.id
              ? "bg-accent/20 text-accent"
              : "text-white/40 hover:bg-white/5 hover:text-white/70",
          )}
        >
          {option.label}
        </button>
      ))}
    </>
  );
}

/**
 * The panel-level view switch, owned by the page and drawn by whichever viewer
 * is currently in the slot — so the control sits with the thing it changes
 * rather than floating above both.
 */
export interface ViewSwitch {
  options: ReadonlyArray<SegmentedOption<string>>;
  active: string;
  onSelect: (id: string) => void;
}

/** Separates the page's view switch from a viewer's own controls. */
export function ChromeDivider() {
  return <span aria-hidden className="mx-1 h-4 w-px shrink-0 bg-white/10" />;
}

interface ViewerButtonProps {
  icon?: IconName;
  label: string;
  onClick: () => void;
  disabled?: boolean;
}

/** A small ghost button for the control bars. */
export function ViewerButton({ icon, label, onClick, disabled = false }: ViewerButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="flex shrink-0 items-center gap-1.5 rounded-lg px-2 py-1 text-[11px] text-white/40 transition-colors hover:bg-white/5 hover:text-accent disabled:pointer-events-none disabled:opacity-30"
    >
      {icon && <Icon name={icon} size={13} />}
      {label}
    </button>
  );
}

/** Fills the stage when there is nothing to draw. */
export function ViewerPlaceholder({ label }: { label: string }) {
  return (
    <div className="flex flex-1 items-center justify-center p-8 text-center text-xs text-white/40">
      {label}
    </div>
  );
}

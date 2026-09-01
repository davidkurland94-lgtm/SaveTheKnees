/** The bits of viewer furniture both `Dicom2DViewer` and `Dicom3DViewer` use. */

import { useId } from "react";

import type { ViewerStack } from "@/interfaces";
import { cn } from "@/lib";
import { Icon, type IconName } from "@/components/ui";

interface StackTabsProps {
  stacks: ViewerStack[];
  activeId: string;
  onSelect: (id: string) => void;
  /** Right-aligned read-out, e.g. the series description. */
  detail?: string;
}

/**
 * The axis picker. A study contributes one stack per plane, so these read as
 * Sagittal / Coronal / Axial tabs; a page passing several series of one plane
 * gets the series description as the label instead.
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
            {stack.plane ?? stack.label ?? `Series ${position + 1}`}
          </button>
        );
      })}
      {detail && (
        <span className="ml-auto truncate pl-2 font-mono text-[11px] text-white/30">{detail}</span>
      )}
    </div>
  );
}

interface ViewerSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  icon?: IconName;
  /** Right-hand read-out; defaults to the raw value. */
  format?: (value: number) => string;
  onChange: (value: number) => void;
  disabled?: boolean;
  className?: string;
}

/** A labelled range input in the viewer's dark palette. */
export function ViewerSlider({
  label,
  value,
  min,
  max,
  step = 1,
  icon,
  format,
  onChange,
  disabled = false,
  className,
}: ViewerSliderProps) {
  const id = useId();
  return (
    <div
      className={cn("flex min-w-0 flex-1 items-center gap-2", disabled && "opacity-40", className)}
    >
      {icon && <Icon name={icon} size={13} className="shrink-0 text-accent" />}
      <label htmlFor={id} className="shrink-0 text-[11px] text-white/40">
        {label}
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-1 min-w-0 flex-1 accent-accent"
      />
      <span className="w-10 shrink-0 text-right font-mono text-[11px] text-white/30">
        {format ? format(value) : value}
      </span>
    </div>
  );
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

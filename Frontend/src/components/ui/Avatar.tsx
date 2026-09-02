import type { PatientIdentity } from "@/interfaces";
import { cn } from "@/lib";

const SIZES = {
  sm: "h-8 w-8 text-[11px]",
  md: "h-11 w-11 text-sm",
} as const;

interface AvatarProps {
  patient: PatientIdentity;
  /** `sm` for table rows, `md` for a page header. */
  size?: keyof typeof SIZES;
  className?: string;
}

/**
 * The initials disc that stands in for a patient photo.
 *
 * Its colour comes from the identity rather than from a prop, so one study
 * keeps the same disc everywhere it appears — which is what lets the eye find a
 * row it has seen before without reading the name.
 */
export function Avatar({ patient, size = "sm", className }: AvatarProps) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "inline-flex shrink-0 items-center justify-center rounded-full font-semibold uppercase",
        SIZES[size],
        patient.tone,
        className,
      )}
    >
      {patient.initials}
    </span>
  );
}

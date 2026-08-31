import { useCallback, useEffect, useRef, useState } from "react";

import type { UploadMode } from "@/interfaces";
import { cn, filterDicomFiles } from "@/lib";
import { Chip, Icon } from "@/components/ui";

const MODES: Array<{ id: UploadMode; tab: string; prompt: string; icon: "upload" | "images" | "folder" }> = [
  { id: "single", tab: "Single image", prompt: "Drop a single .dcm file", icon: "upload" },
  { id: "sequence", tab: "Image sequence", prompt: "Drop every .dcm of one series", icon: "images" },
  { id: "folder", tab: "Folder", prompt: "Drop a folder of .dcm files", icon: "folder" },
];

interface UploadZoneProps {
  onFiles: (files: File[]) => void;
  disabled?: boolean;
}

/**
 * Drop target for a DICOM series.
 *
 * `POST /predict` scores a whole series at once — intensity is normalised across
 * the stack — so "Image sequence" is the default and single-file uploads are
 * flagged as a partial input.
 */
export function UploadZone({ onFiles, disabled = false }: UploadZoneProps) {
  const [mode, setMode] = useState<UploadMode>("sequence");
  const [isDragging, setIsDragging] = useState(false);
  const [rejected, setRejected] = useState(0);

  const inputRef = useRef<HTMLInputElement>(null);

  // `webkitdirectory` is not a standard React prop, so it is set imperatively.
  useEffect(() => {
    const input = inputRef.current;
    if (!input) return;
    if (mode === "folder") input.setAttribute("webkitdirectory", "");
    else input.removeAttribute("webkitdirectory");
  }, [mode]);

  const handleFiles = useCallback(
    (files: File[]) => {
      const dicom = filterDicomFiles(files);
      setRejected(files.length - dicom.length);
      if (dicom.length) onFiles(dicom);
    },
    [onFiles],
  );

  const active = MODES.find((entry) => entry.id === mode) ?? MODES[1];

  return (
    <section>
      <h2 className="mb-4 text-xl text-foreground">Score a series</h2>

      <div className="mb-4 flex w-fit gap-1 rounded-xl bg-muted p-1">
        {MODES.map((entry) => (
          <button
            key={entry.id}
            type="button"
            onClick={() => setMode(entry.id)}
            className={cn(
              "rounded-lg px-4 py-1.5 text-xs font-semibold transition-all",
              mode === entry.id
                ? "bg-white text-primary shadow-sm"
                : "text-muted-foreground hover:text-primary",
            )}
          >
            {entry.tab}
          </button>
        ))}
      </div>

      {/* One input, reconfigured per mode — `key` forces a fresh element so
          picking the same folder twice still fires onChange. */}
      <input
        key={mode}
        ref={inputRef}
        type="file"
        accept=".dcm,application/dicom"
        multiple={mode !== "single"}
        className="hidden"
        onChange={(event) => handleFiles(Array.from(event.target.files ?? []))}
      />

      <div
        role="button"
        tabIndex={0}
        aria-disabled={disabled}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          if (!disabled) handleFiles(Array.from(event.dataTransfer.files));
        }}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            if (!disabled) inputRef.current?.click();
          }
        }}
        className={cn(
          "flex select-none items-center gap-6 rounded-2xl border-2 border-dashed px-10 py-10 transition-all duration-200",
          disabled
            ? "cursor-not-allowed border-border bg-card opacity-60"
            : "cursor-pointer",
          !disabled && isDragging
            ? "scale-[1.005] border-primary bg-muted shadow-lg shadow-purple-100"
            : !disabled && "border-accent-soft bg-card hover:border-accent hover:bg-muted",
        )}
      >
        <div
          className={cn(
            "flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl transition-colors",
            isDragging ? "bg-secondary text-primary" : "border border-border bg-white text-accent",
          )}
        >
          <Icon name={active.icon} size={26} strokeWidth={1.5} />
        </div>

        <div>
          <p
            className={cn(
              "text-base font-semibold transition-colors",
              isDragging ? "text-primary" : "text-foreground",
            )}
          >
            {isDragging ? "Release to load" : active.prompt}
          </p>
          <p className="mt-0.5 text-sm text-muted-foreground">
            or <span className="text-primary underline underline-offset-2">browse your computer</span>
            {" · "}
            <span className="font-mono text-xs">DICOM (.dcm) only</span>
          </p>
          {mode === "single" && (
            <p className="mt-1 text-xs text-amber-600">
              The model needs a whole series — a single slice will score poorly.
            </p>
          )}
          {rejected > 0 && (
            <p className="mt-1 text-xs text-muted-foreground">
              Ignored {rejected} non-DICOM file{rejected === 1 ? "" : "s"}.
            </p>
          )}
        </div>

        <div className="ml-auto flex flex-wrap gap-2">
          {["One plane", "Uncompressed", "16-bit"].map((tag) => (
            <Chip key={tag}>{tag}</Chip>
          ))}
        </div>
      </div>
    </section>
  );
}

export default UploadZone;

import { useEffect, useRef } from "react";

import { filterDicomFiles } from "@/lib";
import { Icon } from "@/components/ui";

interface UploadStudyButtonProps {
  /** Called with the DICOM files of one study; non-DICOM files are dropped. */
  onFiles: (files: File[]) => void;
  disabled?: boolean;
}
export function UploadStudyButton({ onFiles, disabled = false }: UploadStudyButtonProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  // `webkitdirectory` is not a standard React prop, so it is set imperatively.
  useEffect(() => {
    inputRef.current?.setAttribute("webkitdirectory", "");
  }, []);

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".dcm,application/dicom"
        className="hidden"
        onChange={(event) => {
          const picked = filterDicomFiles(Array.from(event.target.files ?? []));
          // Reset first: picking the same folder twice must still fire onChange.
          event.target.value = "";
          if (picked.length) onFiles(picked);
        }}
      />
      <button
        type="button"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        className="flex shrink-0 items-center gap-2 rounded-xl bg-primary px-4 py-2 text-xs font-semibold text-primary-foreground transition-colors hover:bg-primary-hover disabled:pointer-events-none disabled:opacity-50"
      >
        <Icon name="upload" size={14} />
        Upload DICOM
      </button>
    </>
  );
}

export default UploadStudyButton;

import { useEffect, useRef } from "react";

interface DicomCanvasProps {
  imageData: ImageData;
  /** CSS filter percentages; 100 leaves the decoded pixels untouched. */
  brightness?: number;
  contrast?: number;
  className?: string;
}

/** Paints decoded DICOM pixels; window/level is applied as a cheap CSS filter. */
export function DicomCanvas({
  imageData,
  brightness = 100,
  contrast = 100,
  className,
}: DicomCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;

    canvas.width = imageData.width;
    canvas.height = imageData.height;
    context.putImageData(imageData, 0, 0);
  }, [imageData]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ filter: `brightness(${brightness}%) contrast(${contrast}%)` }}
    />
  );
}

export default DicomCanvas;

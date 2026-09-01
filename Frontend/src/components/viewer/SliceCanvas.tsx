import { useEffect, useRef } from "react";
import type { CSSProperties } from "react";

import type { SliceImage } from "@/interfaces";

interface SliceCanvasProps {
  image: SliceImage;
  /** CSS filter percentages; 100 leaves the decoded pixels untouched. */
  brightness?: number;
  contrast?: number;
  /**
   * Longest edge of the backing bitmap. The 3D stack paints dozens of slices at
   * once, so it caps this well under the native 512² to keep GPU memory sane.
   */
  maxSize?: number;
  className?: string;
  style?: CSSProperties;
}

/** Pixel dimensions of anything the viewers accept as a slice. */
export function sliceSize(image: SliceImage): { width: number; height: number } {
  if (typeof HTMLImageElement !== "undefined" && image instanceof HTMLImageElement) {
    return {
      width: image.naturalWidth || image.width,
      height: image.naturalHeight || image.height,
    };
  }
  return { width: image.width, height: image.height };
}

/**
 * Paints one slice, whatever form the page decoded it into.
 *
 * `DicomCanvas` stays the single-purpose `ImageData` painter the upload flow
 * uses; this one also takes bitmaps and `<img>` elements, so a page can feed
 * the 2D/3D viewers slices cut out of a server-rendered contact sheet.
 */
export function SliceCanvas({
  image,
  brightness = 100,
  contrast = 100,
  maxSize,
  className,
  style,
}: SliceCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;

    const paint = () => {
      const { width, height } = sliceSize(image);
      if (width === 0 || height === 0) return;

      const scale = maxSize ? Math.min(1, maxSize / Math.max(width, height)) : 1;
      canvas.width = Math.max(1, Math.round(width * scale));
      canvas.height = Math.max(1, Math.round(height * scale));

      if (image instanceof ImageData) {
        if (scale === 1) {
          context.putImageData(image, 0, 0);
          return;
        }
        // putImageData ignores transforms, so downscaling needs a bounce buffer.
        const buffer = document.createElement("canvas");
        buffer.width = width;
        buffer.height = height;
        buffer.getContext("2d")?.putImageData(image, 0, 0);
        context.drawImage(buffer, 0, 0, canvas.width, canvas.height);
        return;
      }

      context.drawImage(image, 0, 0, canvas.width, canvas.height);
    };

    // A page may pass an <img> that has not decoded yet; repaint when it does
    // rather than leaving a blank canvas behind.
    if (image instanceof HTMLImageElement && !image.complete) {
      image.addEventListener("load", paint);
      return () => image.removeEventListener("load", paint);
    }

    paint();
  }, [image, maxSize]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ ...style, filter: `brightness(${brightness}%) contrast(${contrast}%)` }}
    />
  );
}

export default SliceCanvas;

import { useState } from "react";

import { seriesPreviewUrl } from "@/api";
import type { Series } from "@/interfaces";
import { cn, joinParts } from "@/lib";
import { Chip } from "@/components/ui";

interface SeriesListProps {
  series: Series[];
}

/** Every sequence of a study, each with its 24-slice PNG contact sheet. */
export function SeriesList({ series }: SeriesListProps) {
  if (series.length === 0) {
    return <p className="text-sm text-muted-foreground">This study has no series on the server.</p>;
  }

  return (
    <div className="grid gap-4 sm:grid-cols-2">
      {series.map((entry) => (
        <SeriesCard key={entry.series_uid} series={entry} />
      ))}
    </div>
  );
}

function SeriesCard({ series }: { series: Series }) {
  const [state, setState] = useState<"loading" | "ready" | "failed">("loading");

  const tags = joinParts([
    series.plane,
    `axis ${series.axis}`,
    `${series.n_slices} slices`,
    series.fluid_sensitive && "fluid-sensitive",
    series.fat_suppression && "fat-suppressed",
  ]);

  return (
    <figure className="overflow-hidden rounded-2xl border border-border bg-card">
      <div className="relative flex min-h-40 items-center justify-center bg-black">
        {series.available ? (
          <>
            {state === "loading" && (
              <span className="absolute text-xs text-white/40">Rendering contact sheet…</span>
            )}
            {state === "failed" ? (
              <span className="p-6 text-xs text-white/40">Preview unavailable.</span>
            ) : (
              <img
                src={seriesPreviewUrl(series.study_uid, series.series_uid)}
                alt={`Contact sheet of 24 sampled slices — ${series.plane} series`}
                loading="lazy"
                onLoad={() => setState("ready")}
                onError={() => setState("failed")}
                className={cn(
                  "w-full transition-opacity duration-300",
                  state === "ready" ? "opacity-100" : "opacity-0",
                )}
              />
            )}
          </>
        ) : (
          <span className="p-6 text-xs text-white/40">Files not present on the server.</span>
        )}
      </div>
      <figcaption className="flex flex-col gap-2 px-4 py-3">
        <div className="flex items-center justify-between gap-2">
          <span className="text-sm font-semibold text-foreground">{series.plane}</span>
          <Chip>{series.n_slices} slices</Chip>
        </div>
        <p className="text-[11px] text-muted-foreground">{tags}</p>
        <p className="truncate font-mono text-[10px] text-subtle" title={series.series_uid}>
          {series.series_uid}
        </p>
      </figcaption>
    </figure>
  );
}

export default SeriesList;

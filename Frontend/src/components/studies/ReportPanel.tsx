import { useCallback, useEffect, useMemo, useState } from "react";

import { describeError, getReportTerms, getStudyReport, saveStudyReport } from "@/api";
import type { ReportLang, ReportResponse, StoredReportRecord } from "@/interfaces";
import { cn, createPromiseCache, englishReport, markTerms, pluralize, useAsync } from "@/lib";
import { ErrorState, Icon, Spinner } from "@/components/ui";

/**
 * The dictionary, fetched once for the whole session.
 *
 * It is the same list for every study and it does not change while the tab is
 * open, so one request serves every report the reader opens. A deployment that
 * predates the route answers 404; that is caught here and cached as "no
 * dictionary", which costs one failed request rather than one per study.
 */
const cachedTerms = createPromiseCache<string[]>(1);

const loadTerms = () =>
  cachedTerms("dictionary", () =>
    getReportTerms()
      .then((response) => response.terms)
      .catch(() => []),
  );

/**
 * The two ways to read the same report.
 *
 * "As written" is the editable one and the only one that is ever saved.
 * "English" is derived and read-only — it is the text the models were handed,
 * with the report model's own vocabulary underlined.
 */
const LANGS: Array<{ id: ReportLang; label: string; hint: string }> = [
  { id: "original", label: "As written", hint: "The report itself — edit and save here" },
  { id: "en", label: "English", hint: "What the models read" },
];

interface ReportPanelProps {
  studyUid: string;
  /** The report this study shipped with, or null for an uploaded one. */
  dataset: ReportResponse | null;
  /** The report as this app last stored it; null until it has been saved here. */
  stored: StoredReportRecord | null;
  /** Handed the record the save returned, so the page can unlock Fusion. */
  onSaved: (record: StoredReportRecord) => void;
  className?: string;
}

/**
 * The study's report: read it, edit it, save it.
 *
 * There is deliberately no "mine versus the radiologist's" split here. Every
 * report in this project was written by a doctor reading these images; the only
 * difference between one that shipped with the corpus and one typed in this app
 * is which doctor, and when. So the panel opens whatever text the study already
 * has, lets it be edited, and saves it. An uploaded study opens empty, because
 * nobody has read it yet.
 *
 * NO SCORES LIVE HERE. Saving translates the report server-side and hands it to
 * the fusion model as its second input, so what the report did to the answer
 * shows up in the rail — in the numbers that change when Fusion re-runs. Twelve
 * more probabilities beside the text would be a second, weaker reading of the
 * same study competing with the one the page is actually built to give.
 *
 * Laid out as a wide row to sit beneath the viewers: prose reads badly in a
 * narrow rail, and putting it under the images is what lets someone read a line
 * of the report and look straight up at the slice it describes.
 */
export function ReportPanel({ studyUid, dataset, stored, onSaved, className }: ReportPanelProps) {
  const [lang, setLang] = useState<ReportLang>("original");
  const terms = useAsync(loadTerms, []);

  // The draft lives here rather than in the editor because the English view
  // needs to know whether it is behind it.
  const original = stored?.text ?? dataset?.report ?? "";
  const [draft, setDraft] = useState(original);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Re-seeds when a save lands, and only then: while the doctor is typing the
  // record does not move, so the draft is never pulled out from under them.
  useEffect(() => {
    setDraft(stored?.text ?? dataset?.report ?? "");
  }, [stored?.updated_at]);

  /**
   * The English behind the report — the text the models actually read.
   *
   * Two provenances for the same one report. Once it has been saved here the
   * server has already translated it and the English travels inside the record,
   * so reading it costs nothing. Before that, a corpus report's English comes
   * from the translation cache built once, offline, over the whole dataset; a
   * study missing from that cache simply has no English rendering to show.
   */
  const english = useAsync(
    async (signal): Promise<{ text: string; language: string } | null> => {
      if (stored) return { text: englishReport(stored), language: stored.language ?? "unknown" };
      if (!dataset) return null;
      if (dataset.language === "en") return { text: dataset.report, language: "en" };
      const translated = await getStudyReport(studyUid, "en", signal).catch(() => null);
      return translated ? { text: translated.report, language: dataset.language } : null;
    },
    [studyUid, stored?.updated_at, dataset?.report],
  );

  const text = draft.trim();
  // Saving is for changes and nothing else. A report opened and left alone —
  // whether it came from the corpus or from a previous save — has nothing to
  // write, so the control is dead until the text actually differs.
  const edited = text !== original.trim();
  const savable = Boolean(text) && edited;

  const save = useCallback(async () => {
    const body = draft.trim();
    if (!body || saving) return;
    setSaving(true);
    setError(null);
    try {
      onSaved(await saveStudyReport(studyUid, body, { exists: Boolean(stored) }));
    } catch (cause) {
      setError(describeError(cause));
    } finally {
      setSaving(false);
    }
  }, [draft, saving, studyUid, stored, onSaved]);

  return (
    <section
      className={cn(
        "flex shrink-0 flex-col overflow-hidden rounded-2xl border border-border bg-card",
        className,
      )}
    >
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-2.5">
        <div className="flex flex-wrap items-center gap-3">
          <h2 className="text-sm text-foreground">Report</h2>

          <div className="flex gap-1 rounded-lg bg-muted p-0.5">
            {LANGS.map((entry) => (
              <button
                key={entry.id}
                type="button"
                title={entry.hint}
                aria-pressed={lang === entry.id}
                onClick={() => setLang(entry.id)}
                className={cn(
                  "rounded-md px-2.5 py-1 text-[11px] font-semibold transition-all",
                  lang === entry.id
                    ? "bg-white text-primary shadow-sm"
                    : "text-muted-foreground hover:text-primary",
                )}
              >
                {entry.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Where the text came from. Not a mode to switch between — just the
              provenance of the one report being edited. */}
          <span className="text-[11px] text-muted-foreground">
            {stored
              ? `Saved by ${stored.author}`
              : dataset
                ? "As the study was reported"
                : "Not written yet"}
          </span>

          {english.data && english.data.language !== "en" && (
            <span
              className="font-mono text-[10px] text-subtle"
              title="Detected language of the text as written. The models read the English rendering."
            >
              {english.data.language} → en
            </span>
          )}
        </div>
      </header>

      <div className="flex min-h-0 flex-col gap-2 p-4">
        {lang === "original" ? (
          /* Relative so the save control can sit in the corner of the box it
             saves, rather than on a row of its own below it. */
          <div className="relative">
            <textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              // The one control on the panel, reachable without leaving the text.
              onKeyDown={(event) => {
                if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                  event.preventDefault();
                  void save();
                }
              }}
              spellCheck
              placeholder="Findings from these images…"
              aria-label="Report for this study"
              // Fixed height, not resizable: the corner belongs to the save
              // control, and a box that grows would push the viewers off the
              // screen. Long reports scroll inside it.
              className="h-36 w-full resize-none overflow-y-auto rounded-xl border border-border bg-background p-3 pb-9 text-sm leading-relaxed text-foreground outline-none transition-colors placeholder:text-subtle focus:border-accent-soft"
            />
            <SaveButton
              saving={saving}
              savable={savable}
              edited={edited}
              stored={stored}
              onSave={save}
            />
          </div>
        ) : english.loading ? (
          <div className="flex h-36 items-center justify-center gap-2 rounded-xl border border-border bg-background text-xs text-muted-foreground">
            <Spinner className="h-3.5 w-3.5" />
            Loading the English rendering…
          </div>
        ) : english.data ? (
          <div className="flex flex-col gap-1.5 rounded-xl border border-border bg-background p-3">
            <MarkedText text={english.data.text} terms={terms.data ?? []} className="h-28" />
            {edited && (
              <p className="text-[10px] text-amber-600">
                This is the saved text. Unsaved edits are translated when you save.
              </p>
            )}
          </div>
        ) : (
          <p className="flex h-36 items-center justify-center rounded-xl border border-dashed border-border px-4 text-center text-xs text-muted-foreground">
            No English rendering yet — saving the report produces one.
          </p>
        )}

        {error && <ErrorState message={error} onRetry={save} />}
      </div>
    </section>
  );
}

/**
 * Save, in the corner of the box it saves.
 *
 * The icon carries the state that used to need a sentence beside it: a spinner
 * while the round trip runs, a tick once what is on screen is what is stored,
 * and the save glyph only when there is genuinely something to write. The
 * detail people actually want — when it was last saved — is in the tooltip,
 * which is where a timestamp belongs when it is reassurance rather than
 * information.
 */
function SaveButton({
  saving,
  savable,
  edited,
  stored,
  onSave,
}: {
  saving: boolean;
  savable: boolean;
  edited: boolean;
  stored: StoredReportRecord | null;
  onSave: () => void;
}) {
  const saved = Boolean(stored) && !edited;
  const title = saving
    ? "Saving…"
    : savable
      ? `${stored ? "Save changes" : "Save report"} — ⌘/Ctrl + Enter`
      : saved && stored
        ? `No changes — last saved ${new Date(stored.updated_at).toLocaleString()}`
        : "No changes to save";

  return (
    <button
      type="button"
      onClick={onSave}
      disabled={saving || !savable}
      title={title}
      aria-label={title}
      className={cn(
        "absolute bottom-2 right-2 flex h-7 w-7 items-center justify-center rounded-lg border transition-colors",
        saving
          ? "border-border text-muted-foreground"
          : savable
            ? "border-accent bg-primary text-primary-foreground hover:bg-primary-hover"
            : saved
              ? "border-transparent text-emerald-600"
              : "border-transparent text-subtle",
      )}
    >
      {saving ? (
        <Spinner className="h-3.5 w-3.5" />
      ) : saved ? (
        <Icon name="check" size={13} strokeWidth={3} />
      ) : (
        <Icon name="save" size={13} />
      )}
    </button>
  );
}

/**
 * Report text with the report model's own vocabulary underlined.
 *
 * A mark means "the model counted this term", never "this is why the number
 * came out that way" — the API serves the dictionary, not per-term weights.
 */
function MarkedText({
  text,
  terms,
  className,
}: {
  text: string;
  terms: string[];
  className?: string;
}) {
  const segments = useMemo(() => markTerms(text, terms), [text, terms]);
  const marked = segments.filter((segment) => segment.term).length;

  return (
    <div className="flex flex-col gap-1.5">
      <article
        className={cn(
          "overflow-y-auto whitespace-pre-wrap text-sm leading-relaxed text-foreground",
          className,
        )}
      >
        {segments.map((segment, index) =>
          segment.term ? (
            <mark
              key={index}
              className="bg-transparent text-foreground underline decoration-accent decoration-2 underline-offset-2"
            >
              {segment.text}
            </mark>
          ) : (
            <span key={index}>{segment.text}</span>
          ),
        )}
      </article>
      {marked > 0 && (
        <p className="text-[10px] text-subtle">
          <span className="underline decoration-accent decoration-2 underline-offset-2">
            Underlined
          </span>
          : {pluralize(marked, "term")} the report model counts as a feature — what it read, not how
          much each one moved the answer.
        </p>
      )}
    </div>
  );
}

export default ReportPanel;

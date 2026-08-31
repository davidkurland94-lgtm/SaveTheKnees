from pathlib import Path

import pandas as pd
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSLATIONS_CSV = REPO_ROOT / "data" / "meta" / "reports_en.csv"

# Google's per-request ceiling is ~5000 chars; the longest report is 4,743.
# The guard exists so one pathological future report degrades instead of dying.
MAX_CHARS = 4900


def detect_language(text):
    """ISO code of the report's language, or "unknown". (Kevin's original.)"""
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def translate_report(text):
    """One report, any language -> (English text, detected source language).

    English input passes through untouched -- about half the corpus. On any
    translator failure the ORIGINAL text is returned; build_translation_cache
    detects that case and leaves the study uncached for retry.
    """
    text = str(text)
    language = detect_language(text)
    if language in ("en", "unknown"):
        return text, language
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text[:MAX_CHARS])
        return (translated or text), language
    except Exception:
        return text, language


def build_translation_cache(train_csv=None, out_csv=TRANSLATIONS_CSV,
                            only_uids=None, verbose=True):
    """Translate every report ONCE -> reports_en.csv. Resumable and verified.

    only_uids limits the pass (e.g. just the 58 gold studies).
    """
    train_csv = train_csv or (REPO_ROOT / "data" / "train.csv")
    df = pd.read_csv(train_csv)[["StudyInstanceUID", "Report"]]
    if only_uids is not None:
        df = df[df.StudyInstanceUID.isin(set(only_uids))]

    out_csv = Path(out_csv)
    done = {}
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = dict(zip(prev.StudyInstanceUID, zip(prev.report_en, prev.language)))

    def flush(records):
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"StudyInstanceUID": u, "language": l, "report_en": e}
             for u, (e, l) in records.items()]
        ).to_csv(out_csv, index=False)

    records = dict(done)
    n_failed = 0
    todo = [r for r in df.itertuples(index=False) if r.StudyInstanceUID not in done]
    for n, row in enumerate(todo, start=1):
        en, lang = translate_report(row.Report)

        # A "translation" still in its source language did not translate.
        if lang not in ("en", "unknown") and detect_language(en) == lang:
            n_failed += 1
            if verbose:
                print(f"NOT cached (still {lang}): {row.StudyInstanceUID}", flush=True)
            continue

        records[row.StudyInstanceUID] = (en, lang)
        if n % 50 == 0:
            flush(records)
            if verbose:
                print(f"translated {n}/{len(todo)} (cached {len(records)})", flush=True)

    flush(records)
    if verbose:
        print(f"{len(records)} translations -> {out_csv} "
              f"({n_failed} failed, will retry on next run)", flush=True)
    return out_csv

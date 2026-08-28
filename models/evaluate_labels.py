import argparse
import sys
from pathlib import Path

# Allow `python models/evaluate_labels.py` as well as `python -m models.evaluate_labels`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from functions import paths
from functions.labels import LABELS, gold_labels

# The report-derived tables the repo already carries.
# report-reader: an LLM's twelve scores per study, produced from the report text
# with no access to the images.
KNOWN_READERS = {
    "llm_v4_blend": paths.DATA / "meta/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
    "llm_full": paths.DATA / "meta/rsna-knee-llm-report-labels/llm_labels_full.csv",
    "llm_v2": paths.DATA / "meta/rsna-knee-llm-report-labels/llm_labels_v2.csv",
    "pilkwang_v2": paths.DATA / "meta/rsna-knee-llm-labels-pilkwang/report_labels_v2.csv",
}


def score_reader(table, gold):
    """One reader's table vs the gold labels -> one row per finding.

    Joins on StudyInstanceUID, so a table missing some gold studies is scored
    on the studies it covers (the `n` column says how many). A finding with a
    single class in the covered gold rows has no defined AUC -> NaN.
    """
    merged = gold.merge(table, on="StudyInstanceUID", suffixes=("_gold", "_pred"))
    rows = []
    for label in LABELS:
        y = merged[f"{label}_gold"].to_numpy(dtype=float)
        p = merged[f"{label}_pred"].to_numpy(dtype=float)
        keep = ~np.isnan(p)
        y, p = y[keep], p[keep]
        auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
        rows.append({"label": label, "n": len(y), "positives": int(y.sum()), "auc": auc})
    return pd.DataFrame(rows).set_index("label")


def compare(reader_tables, gold=None):
    """{name: DataFrame} -> one AUC column per reader, side by side."""
    gold = gold_labels(paths.TRAIN_CSV) if gold is None else gold
    out, positives = None, None
    for name, table in reader_tables.items():
        scored = score_reader(table, gold)
        col = scored[["auc"]].rename(columns={"auc": name})
        out = col if out is None else out.join(col)
        if positives is None:
            positives = scored["positives"]
    out.insert(0, "positives", positives)
    mean = out.mean(numeric_only=True)
    mean["positives"] = np.nan
    out.loc["MEAN (defined)"] = mean
    return out


IMAGE_GOLD = paths.DATA / "meta" / "image_model_gold.csv"
REPORT_GOLD = paths.DATA / "meta" / "report_model_gold.csv"


def build_image_model_gold(force=False):
    """Run the trained IMAGE model over the 58 gold studies -> predictions CSV.

    Refuses to write anything while the checkpoint is untrained -- random
    numbers in the referee's table would be worse than an empty seat.
    """
    if IMAGE_GOLD.exists() and not force:
        return IMAGE_GOLD
    from functions.predict import predict_study

    gold = gold_labels(paths.TRAIN_CSV)
    rows = []
    for uid in gold.StudyInstanceUID:
        pred = predict_study(uid)
        if pred is None:
            continue
        if pred.pop("model_status") != "trained":
            print("image model is still untrained -- train it, then rerun this")
            return None
        rows.append({"StudyInstanceUID": uid, **pred})
    pd.DataFrame(rows).to_csv(IMAGE_GOLD, index=False)
    print(f"image model gold predictions -> {IMAGE_GOLD}")
    return IMAGE_GOLD


def image_model_vs_llm():
    """David's image model against every LLM report reader, on the 58 gold."""
    if not IMAGE_GOLD.exists() and build_image_model_gold() is None:
        return None
    gold = gold_labels(paths.TRAIN_CSV)
    readers = {name: pd.read_csv(path)
               for name, path in KNOWN_READERS.items() if path.exists()}
    readers["image_model"] = pd.read_csv(IMAGE_GOLD)
    return compare(readers, gold).round(3)


def david_vs_best_reader():
    """The final showdown: the image model against whichever report-side
    reader (Kevin's model or an LLM table) scores best on gold.

    Where the image model wins a finding, the pixels saw something the report
    pipeline missed -- the "doctor missed it" signal the project exists for.
    """
    if not IMAGE_GOLD.exists() and build_image_model_gold() is None:
        return None
    gold = gold_labels(paths.TRAIN_CSV)

    candidates = {name: pd.read_csv(path)
                  for name, path in KNOWN_READERS.items() if path.exists()}
    if REPORT_GOLD.exists():
        candidates["report_model"] = pd.read_csv(REPORT_GOLD)
    scored = compare(candidates, gold)
    best = scored.drop(index="MEAN (defined)").drop(columns="positives")                  .mean().idxmax()

    table = compare({"image_model": pd.read_csv(IMAGE_GOLD),
                     best: candidates[best]}, gold).round(3)
    table["images_win"] = table["image_model"] > table[best]
    print(f"best report-side reader: {best}")
    return table


def report_model_vs_llm():
    """The models_pipe.ipynb one-liner: the trained report model, Kevin's rule
    baseline, and every LLM reader, side by side on the 58 gold studies."""
    gold = gold_labels(paths.TRAIN_CSV)
    readers = {name: pd.read_csv(path)
               for name, path in KNOWN_READERS.items() if path.exists()}
    for name, fname in [("report_model", "report_model_gold.csv"),
                        ("kevin_rules", "kevin_rules_gold.csv")]:
        path = paths.DATA / "meta" / fname
        if path.exists():
            readers[name] = pd.read_csv(path)
    return compare(readers, gold).round(3)


OUR_MODELS = ("image_model", "report_model", "kevin_rules")


def styled(table):
    """The notebook view: OUR model's number turns BLUE (and bold) wherever it
    beats every LLM reader on that finding -- the wins jump out of the sheet.
    Returns a pandas Styler; use the plain DataFrame for anything programmatic.
    """
    llm_cols = [c for c in table.columns if c.startswith(("llm_", "pilkwang"))]
    ours = [c for c in table.columns if c in OUR_MODELS]

    def paint(row):
        if not llm_cols or row.name == "MEAN (defined)":
            bar = None
        else:
            bar = row[llm_cols].max()
        out = []
        for col in table.columns:
            win = (bar is not None and col in ours
                   and pd.notna(row[col]) and row[col] > bar)
            out.append("color: #1565C0; font-weight: bold" if win else "")
        return out

    return table.style.apply(paint, axis=1).format(precision=3, na_rep="")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Score label tables against the 58 gold studies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("tables", nargs="*", type=Path,
                        help="extra reader CSVs (StudyInstanceUID + the 12 labels), "
                             "e.g. a trained model's gold predictions")
    parser.add_argument("--only", action="store_true",
                        help="score only the given tables, skip the built-in readers")
    parser.add_argument("--out", type=Path, help="also write the table to CSV")
    args = parser.parse_args(argv)

    gold = gold_labels(paths.TRAIN_CSV)

    # Sanity anchor: the gold table scored against itself must be perfect.
    # If this ever prints anything but 1.0, the harness is broken, not a reader.
    self_score = score_reader(gold, gold)["auc"]
    assert (self_score.dropna() == 1.0).all(), "harness self-check failed"

    readers = {} if args.only else {
        name: pd.read_csv(path) for name, path in KNOWN_READERS.items() if path.exists()}
    for path in args.tables:
        readers[path.stem] = pd.read_csv(path)
    if not readers:
        raise SystemExit("no reader tables found - pass CSV paths or add the meta datasets")

    table = compare(readers, gold)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(f"\nAUC vs the {len(gold)} gold studies (self-check passed: gold vs gold = 1.0)\n")
        print(table.to_string())

    if args.out:
        table.to_csv(args.out)
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()

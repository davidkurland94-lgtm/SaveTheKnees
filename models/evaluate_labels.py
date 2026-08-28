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

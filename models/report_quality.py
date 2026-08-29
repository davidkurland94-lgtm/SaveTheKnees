"""Which reports are bad, and what did each one miss? The project's endgame.

THE IDEA
    Two independent witnesses look at every knee: the IMAGES (the 3-seed
    multi-plane ensemble -- it has never read a report) and the REPORT (the
    LLM extraction of what the doctor wrote). Where the images say a finding
    is clearly present and the report says nothing about it, the doctor may
    have missed it. Count a report's validated misses and you get a quality
    score; rank them and you get the review worklist.

WHY ANYONE SHOULD BELIEVE A FLAG
    Every flag type is CALIBRATED AND VALIDATED on the 58 gold studies, where
    the human truth is known:

    1. Thresholds: per finding, the image score cut is chosen on gold for HIGH
       SPECIFICITY (>= 90%) -- a flag should be rare and confident, not chatty.
    2. Validation: on gold, every image-vs-report disagreement has a verdict
       (gold says who was right). The per-finding PRECISION of "missed" flags
       -- how often the images win the argument -- ships with every flag, with
       its sample size, because n=58 keeps these estimates honest but noisy.
    3. Only findings whose flags are validated at or above --min-precision
       (default 0.5) fire on the corpus at all. A finding whose image judge
       cannot beat a coin toss against the report stays silent.

USAGE
    python -m models.report_quality                # validate + score the corpus
    python -m models.report_quality --top 20       # show the 20 worst reports

OUTPUT
    data/meta/report_quality.csv : one row per study --
        n_flags, flagged findings (comma-joined), quality_score in [0, 1]
        (1 = images corroborate the report everywhere we are allowed to judge)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from functions import paths
from functions.labels import LABELS, gold_labels

IMAGE_ALL = paths.DATA / "meta" / "image_ens_all.csv"
IMAGE_GOLD = paths.DATA / "meta" / "image_multiplane_gold.csv"   # same ensemble, gold rows
# The claim side is the TRAINED report model, not the raw LLM extraction: the
# extraction's absolute scale is uncalibrated per finding (it "claims" Effusion
# at 0.9 even on gold-negative knees, so a flat cut can never see a denial).
# The model's sigmoids are comparable across findings, and the denial threshold
# is still learned PER FINDING on gold.
REPORT_CLAIMS = paths.DATA / "meta" / "report_model_all.csv"
REPORT_CLAIMS_GOLD = paths.DATA / "meta" / "report_model_gold.csv"
OUT = paths.DATA / "meta" / "report_quality.csv"


def calibrate_thresholds(image_gold, gold, specificity=0.90):
    """Per finding: the smallest image score that keeps specificity >= target
    on gold. High specificity because a flag is an accusation."""
    thresholds = {}
    for label in LABELS:
        y = gold[label].to_numpy(float)
        p = image_gold[label].to_numpy(float)
        negatives = np.sort(p[y == 0])
        # the score exceeded by at most (1 - specificity) of true negatives
        k = int(np.ceil(specificity * len(negatives))) - 1
        thresholds[label] = float(negatives[min(k, len(negatives) - 1)]) + 1e-9
    return thresholds


def calibrate_denial(report_gold, gold):
    """Per finding: the claim score below which "the report denies it" --
    Youden-optimal split of the claim scores against gold."""
    from sklearn.metrics import roc_curve
    cuts = {}
    for label in LABELS:
        fpr, tpr, thr = roc_curve(gold[label], report_gold[label])
        cuts[label] = float(thr[(tpr - fpr).argmax()])
    return cuts


def validate_flags(image_gold, report_gold, gold, thresholds, denial):
    """On gold: when images flag a finding the report denies, who is right?

    Returns per finding: precision of the flag (gold agreed with the images),
    and the number of such disagreements (the honesty column).
    """
    rows = []
    for label in LABELS:
        truth = gold[label].to_numpy(float)
        img = image_gold[label].to_numpy(float) >= thresholds[label]
        rep = report_gold[label].to_numpy(float) < denial[label]
        flags = img & rep
        n = int(flags.sum())
        correct = int((truth[flags] == 1).sum()) if n else 0
        rows.append({"label": label, "gold_flags": n,
                     "flag_precision": correct / n if n else np.nan})
    return pd.DataFrame(rows).set_index("label")


def score_corpus(thresholds, denial, validation, min_precision=0.5):
    """Every study: which validated findings do the images say the report missed."""
    image = pd.read_csv(IMAGE_ALL).set_index("StudyInstanceUID")
    claims = pd.read_csv(REPORT_CLAIMS).set_index("StudyInstanceUID")
    both = image.index.intersection(claims.index)
    image, claims = image.loc[both], claims.loc[both]

    usable = [l for l in LABELS
              if validation.loc[l, "gold_flags"] > 0
              and validation.loc[l, "flag_precision"] >= min_precision]

    records = []
    for uid in both:
        flags = [l for l in usable
                 if image.loc[uid, l] >= thresholds[l]
                 and claims.loc[uid, l] < denial[l]]
        # weight each flag by how often that flag type is right on gold
        penalty = sum(validation.loc[l, "flag_precision"] for l in flags)
        records.append({
            "StudyInstanceUID": uid,
            "n_flags": len(flags),
            "flagged_findings": ", ".join(flags),
            "quality_score": round(max(0.0, 1.0 - penalty / max(len(usable), 1)), 3),
        })
    table = pd.DataFrame(records).sort_values("quality_score").reset_index(drop=True)
    table.to_csv(OUT, index=False)
    return table, usable


def run(top=10, min_precision=0.5, specificity=0.90):
    gold = gold_labels(paths.TRAIN_CSV).set_index("StudyInstanceUID")
    image_gold = pd.read_csv(IMAGE_GOLD).set_index("StudyInstanceUID").reindex(gold.index)
    report_gold = pd.read_csv(REPORT_CLAIMS_GOLD).set_index("StudyInstanceUID").reindex(gold.index)

    thresholds = calibrate_thresholds(image_gold, gold, specificity)
    denial = calibrate_denial(report_gold, gold)
    validation = validate_flags(image_gold, report_gold, gold, thresholds, denial)
    print("flag validation on the 58 gold studies "
          "(precision = when images accused the report, gold sided with the images):")
    print(validation.round(3).to_string(), "\n")

    table, usable = score_corpus(thresholds, denial, validation, min_precision)
    print(f"corpus scored: {len(table)} reports | flagging enabled for: {usable}")
    n_flagged = int((table.n_flags > 0).sum())
    print(f"reports with >=1 validated flag: {n_flagged} "
          f"({n_flagged / len(table):.1%}) | >=2: {int((table.n_flags >= 2).sum())}\n")
    print(f"the {top} most suspicious reports:")
    print(table.head(top).to_string(index=False))
    print(f"\nfull table -> {OUT}")
    return table


def verdict_sheet(top=20):
    """The notebook view: one row per study, side by side --
    what the REPORT claimed vs what the IMAGES think it should be.

    "report_says"  : findings the report claims (claim score above its own
                     gold-calibrated cut)
    "images_say"   : findings the image ensemble reads in the volumes (score
                     above ITS gold-calibrated balanced cut)
    "missed?"      : the validated flags -- images say yes, report says no,
                     and that flag type earned >= 50% precision on gold
    Rows are the corpus ranked worst-first by quality_score.
    """
    from sklearn.metrics import roc_curve

    gold = gold_labels(paths.TRAIN_CSV).set_index("StudyInstanceUID")
    image_gold = pd.read_csv(IMAGE_GOLD).set_index("StudyInstanceUID").reindex(gold.index)
    report_gold = pd.read_csv(REPORT_CLAIMS_GOLD).set_index("StudyInstanceUID").reindex(gold.index)

    # balanced (Youden) display cuts for BOTH witnesses -- these answer "what
    # does each one think", unlike the deliberately strict flag thresholds
    def youden(scores_gold):
        cuts = {}
        for label in LABELS:
            fpr, tpr, thr = roc_curve(gold[label], scores_gold[label])
            cuts[label] = float(thr[(tpr - fpr).argmax()])
        return cuts
    img_cut, rep_cut = youden(image_gold), youden(report_gold)

    ranked = pd.read_csv(OUT).head(top)
    image = pd.read_csv(IMAGE_ALL).set_index("StudyInstanceUID")
    claims = pd.read_csv(REPORT_CLAIMS).set_index("StudyInstanceUID")

    rows = []
    for r in ranked.itertuples(index=False):
        uid = r.StudyInstanceUID
        rows.append({
            "StudyInstanceUID": uid,
            "report_says": ", ".join(l for l in LABELS if claims.loc[uid, l] >= rep_cut[l]) or "(nothing)",
            "images_say": ", ".join(l for l in LABELS if image.loc[uid, l] >= img_cut[l]) or "(nothing)",
            "missed?": r.flagged_findings if isinstance(r.flagged_findings, str) else "",
            "quality_score": r.quality_score,
        })
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description="Rank reports by image-vs-report disagreement")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--min-precision", type=float, default=0.5,
                   help="a finding's flags fire only if gold validates them at this rate")
    p.add_argument("--specificity", type=float, default=0.90,
                   help="per-finding image threshold calibration target")
    a = p.parse_args(argv)
    run(top=a.top, min_precision=a.min_precision, specificity=a.specificity)


if __name__ == "__main__":
    main()

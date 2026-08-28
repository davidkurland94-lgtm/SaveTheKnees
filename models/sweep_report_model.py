"""Hyperparameter sweep for the report model. Hours of CPU, best model wins.

USAGE
    python -m models.sweep_report_model              # full grid (~2h CPU)
    python -m models.sweep_report_model --quick      # 6 configs, smoke
"""
import argparse
import itertools
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from functions import paths
from models import report_model
from models.evaluate_labels import score_reader
from functions.labels import gold_labels

GRID = {
    "max_features": [10000, 20000, 40000],
    "ngram_max": [1, 2, 3],
    "hidden": [(256, 64), (512, 128), (128,)],
    "dropout": [0.3, 0.5],
    "seed": [0, 1],
}


def sweep(quick=False):
    gold = gold_labels(paths.TRAIN_CSV)
    keys = list(GRID)
    configs = [dict(zip(keys, values)) for values in itertools.product(*GRID.values())]
    if quick:
        configs = configs[:6]
    print(f"{len(configs)} configurations", flush=True)

    work = Path(tempfile.mkdtemp(prefix="stk_sweep_"))

    # The incumbent sets the bar: a sweep whose best is worse than the model
    # already serving must not overwrite it. (Learned the hard way.)
    incumbent_csv = paths.DATA / "meta" / "report_model_gold.csv"
    incumbent_auc = (float(score_reader(pd.read_csv(incumbent_csv), gold)["auc"].mean())
                     if incumbent_csv.exists() else -1.0)
    best = {"auc": incumbent_auc, "config": "incumbent", "run": None}
    print(f"incumbent gold auc: {incumbent_auc:.4f}", flush=True)
    results = []
    for i, config in enumerate(configs, start=1):
        run = work / f"run{i}"
        run.mkdir()
        report_model.train(
            epochs=40, patience=5, verbose=False,
            model_path=run / "model.keras",
            vectorizer_path=run / "vec.joblib",
            gold_csv=run / "gold.csv",
            **config,
        )
        auc = float(score_reader(pd.read_csv(run / "gold.csv"), gold)["auc"].mean())
        results.append({**config, "hidden": str(config["hidden"]), "gold_auc": auc})
        marker = ""
        if auc > best["auc"]:
            best = {"auc": auc, "config": config, "run": run}
            marker = "  <-- best so far"
        print(f"[{i:>3}/{len(configs)}] auc={auc:.4f}  {config}{marker}", flush=True)

    # Promote only a genuine improvement over the incumbent, once.
    if best["run"] is not None:
        shutil.copy2(best["run"] / "model.keras", report_model.MODEL_PATH)
        shutil.copy2(best["run"] / "vec.joblib", report_model.VECTORIZER_PATH)
        shutil.copy2(best["run"] / "gold.csv", paths.DATA / "meta" / "report_model_gold.csv")
        print(f"promoted to {report_model.MODEL_PATH.name}")
    else:
        print("no configuration beat the incumbent; canonical artifacts untouched")

    table = pd.DataFrame(results).sort_values("gold_auc", ascending=False)
    out = paths.DATA / "meta" / "report_model_sweep.csv"
    table.to_csv(out, index=False)
    print(f"\nWINNER auc={best['auc']:.4f}  {best['config']}")
    print(f"full results -> {out}")
    shutil.rmtree(work, ignore_errors=True)


def main(argv=None):
    p = argparse.ArgumentParser(description="Sweep the report model")
    p.add_argument("--quick", action="store_true", help="6 configs only")
    sweep(quick=p.parse_args(argv).quick)


if __name__ == "__main__":
    main()

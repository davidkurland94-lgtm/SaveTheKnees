"""One training session = one model name. Nothing else to know.

    python -m models.train fusion                # canonical recipe, seed 0
    python -m models.train multiplane --seed 2   # a new seed for the ensemble
    python -m models.train report --epochs 60    # any trainer flag passes through
    python -m models.train --list                # the available models

What it does, in order:
  1. looks up in the REGISTRY what the model eats (tensor_cache axes,
     metadata, warm-start artifacts);
  2. downloads from GCS ONLY the blobs absent from disk
     (functions/blobs.py) -- a machine that has everything downloads nothing;
  3. runs the canonical trainer under the usual discipline (train on the
     derived labels, validate on the 58 gold, monitor val_auc).

Outputs are named after the run (knee_<name>_s<seed>.keras + the gold CSV for
the referee): NEVER the production checkpoints. Promotion goes through
models/evaluate_labels.py, as always.

ADDING A MODEL = one entry here. Declare its blobs (axes, meta, artifacts)
and its trainer command; if it eats cache blobs, it runs -- on any machine
of the project.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from functions import blobs, paths

# Metadata EVERY trainer reads (label index + the gold referee).
BASE_META = ["train.csv", "train_series.csv",
             "meta/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
             "meta/derived_labels.csv"]
TEXT_META = ["meta/reports_en.csv", "meta/medical_terms.txt"]

REGISTRY = {
    "sagittal": {
        "axes": ("X",), "meta": BASE_META, "artifacts": [],
        "argv": lambda s: ["--augment", "--epochs", "100", "--patience", "15",
                           "--seed", str(s),
                           "--checkpoint", f"models/knee_sagittal_s{s}.keras",
                           "--scores-out", f"data/meta/sagittal_s{s}_gold.csv"],
    },
    "multiplane": {
        "axes": ("X", "Y", "Z"), "meta": BASE_META, "artifacts": [],
        "argv": lambda s: ["--multi-plane", "--augment", "--epochs", "100",
                           "--patience", "15", "--seed", str(s),
                           "--checkpoint", f"models/knee_multiplane_s{s}.keras",
                           "--scores-out", f"data/meta/multiplane_s{s}_gold.csv"],
    },
    "fusion": {
        "axes": ("X", "Y", "Z"), "meta": BASE_META + TEXT_META,
        "artifacts": ["knee_multiplane.keras", "report_vectorizer.joblib"],
        "argv": lambda s: ["--augment", "--mixup",
                           "--warm-start", "models/knee_multiplane.keras",
                           "--epochs", "100", "--patience", "15", "--seed", str(s),
                           "--checkpoint", f"models/knee_fusion_s{s}.keras",
                           "--gold-out", f"data/meta/fusion_s{s}_gold.csv"],
    },
    "report": {   # text only: no tensors, runs on CPU in minutes
        "axes": (), "meta": BASE_META + TEXT_META, "artifacts": [],
        "argv": lambda s: ["--epochs", "40", "--patience", "5", "--seed", str(s)],
    },
}


def _uids_for(spec):
    """The studies an image session will read: derived labels + the 58 gold."""
    import pandas as pd
    from functions.labels import gold_labels
    gold = gold_labels(paths.TRAIN_CSV)
    derived = pd.read_csv(paths.DERIVED_LABELS)
    return list(dict.fromkeys(list(derived.StudyInstanceUID) +
                              list(gold.StudyInstanceUID)))


def run(name, seed=0, extra=()):
    spec = REGISTRY[name]
    blobs.ensure_meta(spec["meta"])
    if spec["artifacts"]:
        blobs.ensure_artifacts(spec["artifacts"])
    if spec["axes"]:
        blobs.ensure_cache(_uids_for(spec), axes=spec["axes"])

    # Canonical flags first, yours after: argparse keeps the last occurrence,
    # so `--epochs 3` on the command line overrides the recipe's 100.
    argv = spec["argv"](seed) + list(extra)
    print(f"train[{name}] seed={seed} -> {' '.join(argv)}")
    if name == "fusion":
        from models import train_fusion
        train_fusion.run(train_fusion.parse_args(argv))
    elif name == "report":
        from models import report_model
        a = argparse.ArgumentParser()
        a.add_argument("--epochs", type=int, default=40)
        a.add_argument("--patience", type=int, default=5)
        a.add_argument("--seed", type=int, default=seed)
        a.add_argument("--max-features", type=int, default=20000)
        r = a.parse_args(argv)
        report_model.train(
            epochs=r.epochs, patience=r.patience, seed=r.seed,
            max_features=r.max_features,
            model_path=paths.REPO_ROOT / "models" / f"report_model_s{r.seed}.keras",
            vectorizer_path=paths.REPO_ROOT / "models" / f"report_vectorizer_s{r.seed}.joblib",
            gold_csv=paths.DATA / "meta" / f"report_s{r.seed}_gold.csv")
    else:
        from models import train_model
        train_model.run(train_model.parse_args(argv))


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Run a training session by model name",
        epilog="any extra flag is passed through to the underlying trainer")
    p.add_argument("name", nargs="?", choices=sorted(REGISTRY))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--list", action="store_true", help="show the registry")
    args, extra = p.parse_known_args(argv)

    if args.list or not args.name:
        for n, s in REGISTRY.items():
            inputs = "+".join(s["axes"]) if s["axes"] else "text"
            print(f"  {n:<11} blobs: {inputs:<7} meta: {len(s['meta'])} files"
                  + (f"  warm-start: {s['artifacts'][0]}" if s["artifacts"] else ""))
        return
    run(args.name, seed=args.seed, extra=extra)


if __name__ == "__main__":
    main()

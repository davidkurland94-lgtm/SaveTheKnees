"""Train the knee-findings model. This is the entry point.

PIPELINE POSITION
    Step 5 of 5, and the only file here that touches data or the command line.
    Everything above it is definitions; this is the orchestrator.

        labels.py  ->  tensor_cache.py  ->  datasets.py  ->  architectures.py
                                                          -> TRAIN_MODEL.PY

USAGE
    python -m models.train_model                       # all defaults
    python -m models.train_model --epochs 30 --as-channels
    python -m models.train_model --train-images data/train_series \\
                                 --derived-labels data/meta/derived_labels.csv

    Run it from the repo root. `--help` lists every knob.

WHAT IT DOES, IN ORDER
    1. gold_labels()      - the 58 human-read studies -> validation set
    2. derived_labels()   - the ~4350 Claude-derived studies, minus the gold UIDs
                            -> training set
    3. build_cache()      - decode both sets to .npy once (~15 min the first
                            time, near-instant after)
    4. make_dataset()     - stream the cache into two tf.data pipelines
    5. build/compile      - the input shape is read off the data, never hardcoded
    6. train_model()      - fit, with the callbacks below
    7. evaluate_per_label - per-finding AUC on gold, printed and optionally saved

THE CALLBACKS
    They run at the end of each epoch and are what stop this from being a
    guessing game:

        EarlyStopping     - stop when the gold AUC has not improved for
                            `--patience` epochs, and put back the best weights
                            seen, not whatever the last epoch happened to produce
        ModelCheckpoint   - save the best model to a file as it goes
        ReduceLROnPlateau - when progress stalls, halve the learning rate and
                            take smaller steps

    All three watch `val_auc` - the AUC on the gold set. `mode="max"` says higher
    is better; the default guesses from the metric's name and guesses wrong for a
    custom one, so it is always worth stating.

WHY AUC
    Roughly: pick a knee that has the finding and one that does not. AUC is the
    chance the model gives the first a higher score than the second. 0.5 is
    coin-flip, 1.0 is perfect. It is used here instead of accuracy because the
    findings are rare - a model that answers "no" to everything scores high on
    accuracy and is useless.
"""
import argparse
import logging
import sys
from pathlib import Path

# Allow `python models/train_model.py` as well as `python -m models.train_model`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks as keras_callbacks

from functions import paths
from functions.datasets import make_dataset, make_multiplane_dataset
from functions.labels import LABELS, derived_labels, gold_labels
from functions.tensor_cache import build_cache, cached_subset
from models.architectures import build_model, build_model_multiplane, compile_model

log = logging.getLogger("train")


def training_callbacks(checkpoint_path="best_model.keras", patience=5):
    """Stop when gold AUC stops improving, and keep the best weights.

    Without these the epoch count is a guess, and the model you end up with is
    whatever the last epoch produced rather than the best one seen.
    """
    return [
        keras_callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=patience, restore_best_weights=True),
        keras_callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_auc", mode="max", save_best_only=True),
        keras_callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.5, patience=max(1, patience // 2)),
    ]


def train_model(model, train_ds, val_ds=None, epochs=10, callbacks=None):
    """Fit an already-compiled model. Returns the Keras History.

    Callbacks are only attached when there is a validation set for them to watch.
    """
    if val_ds is None:
        return model.fit(train_ds, epochs=epochs)

    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=training_callbacks() if callbacks is None else callbacks,
    )


def evaluate_per_label(model, ds, labels):
    """AUC per finding on the gold set, not one number for all twelve.

    A single averaged AUC hides everything: gold prevalence runs from 35 positives
    (Effusion) to 9 (MCL), so one strong common label carries the average while the
    rare ones sit at chance. With 58 studies these numbers are noisy either way -
    `positives` is printed alongside so a score can be read with its sample size.

    A label with only one class present has no defined AUC and comes back NaN.
    """
    probs = model.predict(ds, verbose=0)
    truth = np.asarray(labels, dtype="float32")

    rows = []
    for i, name in enumerate(LABELS):
        y, score = truth[:, i], probs[:, i]
        if len(np.unique(y)) < 2:
            auc = np.nan
        else:
            metric = tf.keras.metrics.AUC()
            metric.update_state(y, score)
            auc = float(metric.result())
        rows.append({"label": name, "positives": int(y.sum()), "auc": auc})

    return pd.DataFrame(rows).set_index("label")


def check_inputs(args):
    """Fail before the expensive work, naming the path that is missing."""
    required = {
        "--train-csv": args.train_csv,
        "--series-csv": args.series_csv,
        "--train-images": args.train_images,
        "--gold-images": args.gold_images,
        "--derived-labels": args.derived_labels,
    }
    missing = [f"{flag} -> {path}" for flag, path in required.items() if not path.exists()]
    if missing:
        raise SystemExit(
            "cannot start, these inputs do not exist:\n  " + "\n  ".join(missing))


def run(args):
    """The whole training run. Takes plain values, so a notebook or a sweep
    script can call it directly without going through the command line."""
    tf.keras.utils.set_random_seed(args.seed)
    check_inputs(args)

    # 1-2. Labels. The gold UIDs are excluded from training, always.
    series_df = pd.read_csv(args.series_csv)
    gold = gold_labels(args.train_csv)
    train_index = derived_labels(args.derived_labels, exclude=gold.StudyInstanceUID)
    log.info("%d gold studies (validation), %d derived studies (train)",
             len(gold), len(train_index))
    log.info("gold prevalence:\n%s", gold[LABELS].sum().astype(int).to_string())

    # 3. Decode once. Re-running is cheap: anything already cached is skipped.
    #    cached_subset is what keeps the label rows aligned with the cache.
    axes = ("X", "Y", "Z") if args.multi_plane else (args.axis,)
    gold_ok, train_ok = None, None
    for axis in axes:
        g = set(build_cache(gold, series_df, args.gold_images, args.cache, axis=axis))
        t = set(build_cache(train_index, series_df, args.train_images, args.cache, axis=axis))
        gold_ok = g if gold_ok is None else gold_ok & g
        train_ok = t if train_ok is None else train_ok & t
    gold = cached_subset(gold, sorted(gold_ok))
    train_index = cached_subset(train_index, sorted(train_ok))
    log.info("%d train and %d gold studies cached on %s", len(train_index), len(gold), axes)

    # 4. Stream.
    if args.multi_plane:
        train_ds = make_multiplane_dataset(train_index.StudyInstanceUID,
                                           train_index[LABELS], args.cache,
                                           batch_size=args.batch_size, shuffle=True,
                                           augment=args.augment)
        gold_ds = make_multiplane_dataset(gold.StudyInstanceUID, gold[LABELS],
                                          args.cache, batch_size=args.batch_size)
    else:
        train_ds = make_dataset(train_index.StudyInstanceUID, train_index[LABELS],
                                args.cache, batch_size=args.batch_size, shuffle=True,
                                axis=args.axis, as_channels=args.as_channels,
                                augment=args.augment)
        gold_ds = make_dataset(gold.StudyInstanceUID, gold[LABELS], args.cache,
                               batch_size=args.batch_size, axis=args.axis,
                               as_channels=args.as_channels)

    # 5. The per-sample shape comes off the data, never hardcoded (the
    #    multi-plane model's three inputs share one per-plane shape).
    if args.multi_plane:
        one_sample = next(iter(train_ds))[0][0].shape[1:]
        log.info("one sample %s x 3 planes", one_sample)
        model = compile_model(build_model_multiplane(one_sample))
    else:
        one_sample = next(iter(train_ds))[0].shape[1:]
        log.info("one sample %s", one_sample)
        model = compile_model(build_model(one_sample))
    model.summary(print_fn=log.info)

    # 6. Fit.
    train_model(model, train_ds, gold_ds, epochs=args.epochs,
                callbacks=training_callbacks(args.checkpoint, args.patience))

    # 7. Score per finding, with the positive count beside each number.
    scores = evaluate_per_label(model, gold_ds, gold[LABELS])
    log.info("gold AUC per finding:\n%s", scores.to_string())
    if args.scores_out:
        scores.to_csv(args.scores_out)
        log.info("scores written to %s", args.scores_out)

    return model, scores


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train the knee-findings model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    data = p.add_argument_group("data")
    data.add_argument("--train-csv", type=Path, default=paths.TRAIN_CSV,
                      help="train.csv; the gold labels are the fully-filled rows")
    data.add_argument("--series-csv", type=Path, default=paths.SERIES_CSV,
                      help="train_series.csv, used to pick a series per study")
    data.add_argument("--train-images", type=Path, default=paths.TRAIN_IMAGES,
                      help="<study>/<series>/*.dcm for the training studies")
    data.add_argument("--gold-images", type=Path, default=paths.GOLD_IMAGES,
                      help="<study>/<series>/*.dcm for the 58 gold studies")
    data.add_argument("--derived-labels", type=Path, default=paths.DERIVED_LABELS,
                      help="the Claude-derived label table")
    data.add_argument("--cache", type=Path, default=paths.CACHE,
                      help="where the decoded .npy tensors live")

    model = p.add_argument_group("model")
    model.add_argument("--as-channels", action="store_true",
                       help="2.5D layout (H, W, K) -> Conv2D; default is 3D (K, H, W, 1)")
    model.add_argument("--axis", default="X", choices=["X", "Y", "Z"],
                       help="X sagittal, Y coronal, Z axial; each axis has its own cache")
    model.add_argument("--multi-plane", action="store_true",
                       help="one model reading sagittal+coronal+axial per study "
                            "(needs all three axis caches; overrides --axis)")
    model.add_argument("--augment", action="store_true",
                       help="crop-shift + intensity jitter on the training split only")

    train = p.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--patience", type=int, default=5,
                       help="epochs without val_auc improvement before stopping")
    train.add_argument("--seed", type=int, default=0)

    out = p.add_argument_group("output")
    out.add_argument("--checkpoint", type=Path, default=Path("best_model.keras"),
                     help="best-so-far weights, written every time val_auc improves")
    out.add_argument("--scores-out", type=Path, default=None,
                     help="optional CSV for the per-label AUC table")

    return p.parse_args(argv)


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args(argv))


if __name__ == "__main__":
    main()

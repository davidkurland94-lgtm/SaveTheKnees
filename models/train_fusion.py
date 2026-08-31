"""Train the capstone model: three image planes + the report -> 12 findings.

WHY THIS MODEL
    The two single-modality models are strongly complementary on gold --
    images win Effusion (0.913), text wins ACL (0.923) -- and even a plain
    50/50 average of their outputs scores 0.877 against 0.683/0.855 alone.
    This model trains the combination end to end, so the head learns
    per-finding trust instead of a fixed vote.

SAME DISCIPLINE AS EVERYTHING ELSE
    train on the derived soft labels (gold excluded), validate on the 58 gold,
    monitor val_auc, promote nothing without the referee's approval.

INPUTS PER STUDY
    - sagittal + coronal + axial volumes from the tensor cache (all 3 axes
      must be cached: run the YZ cache build first)
    - the report's TF-IDF vector from the ALREADY-FITTED report vectorizer
      (models/report_vectorizer.joblib) -- reusing it keeps the text features
      identical to the standalone report model's and avoids a second fit.

USAGE
    python -m models.train_fusion --epochs 100 --patience 15 --augment
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks as keras_callbacks

from functions import paths
from functions.datasets import make_fusion_dataset
from functions.labels import LABELS, derived_labels, gold_labels
from functions.tensor_cache import build_cache, cached_subset

CHECKPOINT = paths.REPO_ROOT / "models" / "knee_fusion.keras"


def texts_for(uids):
    """uid -> English report text from the translation cache (complete: 4,407)."""
    t = pd.read_csv(paths.DATA / "meta" / "reports_en.csv")
    lookup = dict(zip(t.StudyInstanceUID, t.report_en))
    return [str(lookup.get(u, "")) for u in uids]


def run(args):
    tf.keras.utils.set_random_seed(args.seed)
    from models.architectures import build_model_fusion, compile_model

    series_df = pd.read_csv(paths.SERIES_CSV)
    gold = gold_labels(paths.TRAIN_CSV)
    train_index = derived_labels(args.derived_labels, exclude=gold.StudyInstanceUID)

    # All three axis caches, intersected, labels realigned.
    gold_ok, train_ok = None, None
    for axis in ("X", "Y", "Z"):
        g = set(build_cache(gold, series_df, paths.TRAIN_IMAGES, args.cache, axis=axis))
        t = set(build_cache(train_index, series_df, paths.TRAIN_IMAGES, args.cache, axis=axis))
        gold_ok = g if gold_ok is None else gold_ok & g
        train_ok = t if train_ok is None else train_ok & t
    gold = cached_subset(gold, sorted(gold_ok))
    train_index = cached_subset(train_index, sorted(train_ok))
    print(f"{len(train_index)} train / {len(gold)} gold studies with all 3 planes cached")

    # Text features from the report model's fitted vectorizer.
    vectorizer = joblib.load(paths.REPO_ROOT / "models" / "report_vectorizer.joblib")
    x_text_train = vectorizer.transform(texts_for(train_index.StudyInstanceUID)) \
                             .toarray().astype("float32")
    x_text_gold = vectorizer.transform(texts_for(gold.StudyInstanceUID)) \
                            .toarray().astype("float32")

    train_ds = make_fusion_dataset(train_index.StudyInstanceUID, train_index[LABELS],
                                   x_text_train, args.cache, batch_size=args.batch_size,
                                   shuffle=True, augment=args.augment)
    if args.mixup:
        from functions.datasets import add_mixup
        train_ds = add_mixup(train_ds, n_inputs=4)
    gold_ds = make_fusion_dataset(gold.StudyInstanceUID, gold[LABELS],
                                  x_text_gold, args.cache, batch_size=args.batch_size)

    model = compile_model(build_model_fusion((24, 224, 224, 1), x_text_train.shape[1]))
    print(f"fusion model: {model.count_params():,} params, {len(model.inputs)} inputs")

    if args.warm_start:
        # Transplant the trained multi-plane encoders: both models build their
        # image branches from the same _encode_3d blocks, so the Conv3D layers
        # correspond one-to-one in build order. The text branch and head stay
        # fresh -- they are what fusion exists to learn.
        import keras
        donor = keras.saving.load_model(args.warm_start, compile=False)
        src = [l for l in donor.layers if "Conv3D" in type(l).__name__]
        dst = [l for l in model.layers if "Conv3D" in type(l).__name__]
        assert len(src) == len(dst) == 9, f"encoder mismatch: {len(src)} vs {len(dst)}"
        for a, b in zip(src, dst):
            b.set_weights(a.get_weights())
        print(f"warm-started 9 Conv3D layers from {args.warm_start}")

    model.fit(train_ds, validation_data=gold_ds, epochs=args.epochs, verbose=2,
              callbacks=[
                  keras_callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                                patience=args.patience,
                                                restore_best_weights=True),
                  keras_callbacks.ModelCheckpoint(args.checkpoint, monitor="val_auc",
                                                  mode="max", save_best_only=True),
                  keras_callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max",
                                                    factor=0.5,
                                                    patience=max(1, args.patience // 2)),
              ])

    # Gold predictions in the shared readers format, for the referee.
    probs = model.predict(gold_ds, verbose=0)
    out = pd.DataFrame(probs, columns=LABELS)
    out.insert(0, "StudyInstanceUID", gold.StudyInstanceUID.values)
    out_csv = args.gold_out or (paths.DATA / "meta" / "fusion_model_gold.csv")
    out.to_csv(out_csv, index=False)
    print(f"gold predictions -> {out_csv}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the image+report fusion model")
    p.add_argument("--derived-labels", type=Path, default=paths.DERIVED_LABELS)
    p.add_argument("--cache", type=Path, default=paths.CACHE)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--mixup", action="store_true",
                   help="batch-level mixup on images+text+soft labels (train only)")
    p.add_argument("--warm-start", type=Path, default=None,
                   help="multi-plane checkpoint whose Conv3D encoders seed this model")
    p.add_argument("--gold-out", type=Path, default=None,
                   help="where to write gold predictions (default fusion_model_gold.csv; "
                        "give each seed its own file so ensembles can average them)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

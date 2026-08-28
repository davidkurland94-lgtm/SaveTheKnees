"""
Train the image model and write models/knee_resnet50.keras.

This is what compose.yaml's `train` service runs, and what turns the API's
model_status from "untrained" to "trained" -- predict.py loads that exact path,
so a finished run here is picked up by the next container start with no code
change.

Start frozen. build_model(trainable_base=False) leaves ~25k trainable
parameters against 23.6M frozen ones; 4,407 studies is not enough to fine-tune
a full ResNet50 without memorising it. Unfreeze only once the frozen baseline
has a validation AUC worth improving on.

Usage:
    python -m functions.train --smoke                 # 16 studies, 1 epoch
    python -m functions.train --epochs 10 --fold 0
    python -m functions.train --source dicom          # slower, matches serving
    python -m functions.train --unfreeze --lr 1e-5    # after a frozen baseline

On this machine TensorFlow cannot see the RTX 5080 (Windows builds are CPU-only
since 2.10), so a real run belongs in WSL2 or the compose `train` service.
"""

import argparse
import os
from pathlib import Path

import keras

from functions.dataset import make_datasets, studies_for_fold
from functions.model import LABELS, build_model, compile_model

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", REPO_ROOT / "models" / "knee_resnet50.keras"))


def train(fold=0, epochs=10, batch_size=4, lr=1e-3, axis="X", source="mirror",
          unfreeze=False, limit=None, model_path=MODEL_PATH):
    """Fit, then save. Returns the Keras History."""
    train_ds, val_ds = make_datasets(val_fold=fold, batch_size=batch_size,
                                     axis=axis, limit=limit, source=source)

    model = compile_model(build_model(trainable_base=unfreeze), learning_rate=lr)
    trainable = sum(int(v.numpy().size) for v in model.trainable_weights)
    print(f"trainable parameters: {trainable:,}  (base {'UNFROZEN' if unfreeze else 'frozen'})")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        # save_best_only, so a late overfitting epoch cannot overwrite a good
        # checkpoint. The API reads whatever file is here.
        keras.callbacks.ModelCheckpoint(model_path, monitor="val_auc", mode="max",
                                        save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5,
                                          patience=2, verbose=1),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        callbacks=callbacks, verbose=1)

    # EarlyStopping restored the best weights; ModelCheckpoint may hold an
    # earlier epoch. Write once more so the file matches the model in memory.
    model.save(model_path)
    print(f"saved -> {model_path}")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train the knee MRI image model.")
    parser.add_argument("--fold", type=int, default=0, help="fold to hold out (0-4)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--axis", default="X", choices=["X", "Y", "Z"])
    parser.add_argument("--source", default="mirror", choices=["mirror", "dicom"],
                        help="mirror is ~20x faster; dicom matches what the API serves")
    parser.add_argument("--unfreeze", action="store_true",
                        help="fine-tune ResNet50 too (needs a GPU and a low --lr)")
    parser.add_argument("--limit", type=int, help="cap studies per split, for testing")
    parser.add_argument("--smoke", action="store_true",
                        help="16 studies, 1 epoch -- proves the pipeline runs")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    if args.smoke:
        args.limit, args.epochs = 16, 1

    print(f"fold {args.fold} held out | "
          f"{len(studies_for_fold(args.fold, exclude=True))} train / "
          f"{len(studies_for_fold(args.fold))} val studies | "
          f"source={args.source} | {len(LABELS)} labels")

    train(fold=args.fold, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, axis=args.axis, source=args.source,
          unfreeze=args.unfreeze, limit=args.limit, model_path=args.model_path)


if __name__ == "__main__":
    main()

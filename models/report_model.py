"""The trainable report reader: English report text in, 12 findings out.

WHY THIS EXISTS (and what it replaced)
    The first report reader was a rule engine: a hand-written 500-entry
    dictionary and 600 lines of patterns (see notebooks/kevin/NLP_Knees.ipynb,
    where it still lives). This model learns the vocabulary FROM the corpus
    instead: TF-IDF turns each translated report into weighted word/bigram
    counts, a small dense head maps those to the twelve findings, and nothing
    about knees is written into it by hand. On the 58 gold studies the rules
    scored mean AUC 0.690 (snapshot: data/meta/kevin_rules_gold.csv); this
    model scores 0.820.

SAME DISCIPLINE AS THE IMAGE MODEL
    train on   the ~4.3k derived soft labels, gold UIDs excluded, always
    validate   the 58 human-labelled gold studies (hard 0/1)
    loss       binary_crossentropy on soft targets; SoftAUC metrics
    monitor    val_auc, EarlyStopping + checkpoint, exactly like train_model.py

    Text comes from data/meta/reports_en.csv (report_translation's cache).
    While that cache is still filling, reports whose ORIGINAL text is already
    English are used directly, so the model can train before every language
    is translated; retrain when the cache completes to add the rest.

USAGE
    python -m models.report_model                 # train with defaults
    python -m models.report_model --epochs 40

    from models.report_model import predict_report
    predict_report("1.2.826.0.1.3680043.8.498...")     # a StudyInstanceUID
    predict_report_text("ACL rupture. Joint effusion.")  # raw text variant

ARTIFACTS
    models/report_model.keras        the trained head
    models/report_vectorizer.joblib  the fitted TF-IDF vocabulary
    Both are needed to predict; load_report_predictor() loads the pair.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from functions import paths
from functions.labels import LABELS, derived_labels, gold_labels
from functions.report_translation import TRANSLATIONS_CSV, detect_language

MODEL_PATH = paths.REPO_ROOT / "models" / "report_model.keras"
VECTORIZER_PATH = paths.REPO_ROOT / "models" / "report_vectorizer.joblib"


def english_texts(uids, verbose=True):
    """uid -> English text, for every study we can honestly read today.

    Priority: the translation cache. Fallback: the raw report when it is
    already English (about half the corpus). Studies that are neither are
    returned in `missing` -- they join the corpus when the cache catches up.
    """
    reports = pd.read_csv(paths.TRAIN_CSV).set_index("StudyInstanceUID").Report
    cache = {}
    if Path(TRANSLATIONS_CSV).exists():
        t = pd.read_csv(TRANSLATIONS_CSV)
        cache = dict(zip(t.StudyInstanceUID, t.report_en))

    texts, missing = {}, []
    for uid in uids:
        if uid in cache and isinstance(cache[uid], str):
            texts[uid] = cache[uid]
        else:
            raw = str(reports.get(uid, ""))
            if raw and detect_language(raw) == "en":
                texts[uid] = raw
            else:
                missing.append(uid)
    if verbose:
        print(f"texts: {len(texts)} usable ({len(cache)} from cache), "
              f"{len(missing)} awaiting translation")
    return texts, missing


def build_head(n_features, n_labels=len(LABELS)):
    """TF-IDF vector -> 12 sigmoids. Small on purpose: with ~4k training
    reports, capacity is the enemy."""
    import keras
    from keras import layers
    from models.architectures import SoftAUC, SoftBinaryAccuracy

    model = keras.Sequential([
        keras.Input(shape=(n_features,), name="tfidf"),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(n_labels, activation="sigmoid", name="findings"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=[SoftAUC(name="auc", multi_label=True),
                           SoftBinaryAccuracy(name="accuracy")])
    return model


def train(epochs=30, max_features=20000, patience=5, seed=0,
          model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH, verbose=True):
    import keras
    from sklearn.feature_extraction.text import TfidfVectorizer

    keras.utils.set_random_seed(seed)

    gold = gold_labels(paths.TRAIN_CSV)
    derived = derived_labels(paths.DERIVED_LABELS, exclude=gold.StudyInstanceUID)

    train_texts, _ = english_texts(derived.StudyInstanceUID, verbose)
    gold_texts, gold_missing = english_texts(gold.StudyInstanceUID, verbose)
    if gold_missing:
        raise SystemExit(f"{len(gold_missing)} gold reports lack English text -- "
                         "run functions.report_translation.build_translation_cache first")

    derived = derived[derived.StudyInstanceUID.isin(train_texts)].reset_index(drop=True)
    x_text = [train_texts[u] for u in derived.StudyInstanceUID]
    y = derived[LABELS].to_numpy(dtype="float32")

    # Fit the vocabulary on TRAINING text only -- letting gold shape the idf
    # weights would leak the validation set into the features.
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                                 sublinear_tf=True, min_df=3)
    x = vectorizer.fit_transform(x_text).toarray().astype("float32")
    x_gold = vectorizer.transform(
        [gold_texts[u] for u in gold.StudyInstanceUID]).toarray().astype("float32")
    y_gold = gold[LABELS].to_numpy(dtype="float32")
    if verbose:
        print(f"training on {x.shape[0]} reports x {x.shape[1]} features; "
              f"validating on {x_gold.shape[0]} gold")

    model = build_head(x.shape[1])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.fit(
        x, y, validation_data=(x_gold, y_gold),
        epochs=epochs, batch_size=32, verbose=2 if verbose else 0,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                          patience=patience, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(model_path, monitor="val_auc",
                                            mode="max", save_best_only=True),
        ])
    model.save(model_path)
    joblib.dump(vectorizer, vectorizer_path)

    # Gold predictions in the shared readers format, straight into the referee.
    probs = model.predict(x_gold, verbose=0)
    scores = pd.DataFrame(probs, columns=LABELS)
    scores.insert(0, "StudyInstanceUID", gold.StudyInstanceUID.values)
    out = paths.DATA / "meta" / "report_model_gold.csv"
    scores.to_csv(out, index=False)
    if verbose:
        print(f"saved {model_path.name} + {vectorizer_path.name}; gold predictions -> {out}")
    return model, vectorizer


# ---------------------------------------------------------------------------
# Predict side -- the notebook one-liner
# ---------------------------------------------------------------------------

_predictor = None


def load_report_predictor(model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH):
    import keras
    model = keras.saving.load_model(model_path, compile=False)
    return model, joblib.load(vectorizer_path)


def predict_report_text(text):
    """Raw English report text -> {label: probability}. Loads artifacts once."""
    global _predictor
    if _predictor is None:
        _predictor = load_report_predictor()
    model, vectorizer = _predictor
    x = vectorizer.transform([str(text)]).toarray().astype("float32")
    probs = model.predict(x, verbose=0)[0]
    return {label: float(p) for label, p in zip(LABELS, probs)}


def predict_report(study_uid):
    """StudyInstanceUID -> {label: probability}, from that study's report.

    Same interface convention as functions.predict.predict_study: every
    predict_* in this project is keyed by the study UID. The English text
    comes from the translation cache; a study not yet cached is translated
    live (~1s) so the call works for the whole corpus at any time.
    """
    cache = pd.read_csv(TRANSLATIONS_CSV).set_index("StudyInstanceUID")         if Path(TRANSLATIONS_CSV).exists() else None
    if cache is not None and study_uid in cache.index:
        text = str(cache.loc[study_uid, "report_en"])
    else:
        from functions.report_translation import translate_report
        raw = pd.read_csv(paths.TRAIN_CSV).set_index("StudyInstanceUID").Report
        if study_uid not in raw.index:
            return None
        text, _ = translate_report(raw[study_uid])
    return predict_report_text(text)


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the report->findings model")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max-features", type=int, default=20000)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    train(epochs=a.epochs, max_features=a.max_features,
          patience=a.patience, seed=a.seed)


if __name__ == "__main__":
    main()

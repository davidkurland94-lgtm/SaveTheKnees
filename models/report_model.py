import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from functions import paths
from functions.labels import LABELS, derived_labels, gold_labels

# NOTE: functions.report_translation (deep_translator, langdetect) is imported
# LAZILY inside the functions that need it. The serving container carries
# neither package -- predict_report_text and load_report_predictor must import
# clean with only keras + sklearn on board.
TRANSLATIONS_CSV = paths.DATA / "meta" / "reports_en.csv"

MODEL_PATH = paths.REPO_ROOT / "models" / "report_model.keras"
MEDICAL_TERMS = paths.DATA / "meta" / "medical_terms.txt"


def medical_terms(terms_file=MEDICAL_TERMS):
    """Kevin's dictionary, one term per line, in file order -- which is also
    feature order, so callers may read it but must not reorder it. Empty list
    when the file is absent, exactly as term_features treats that case."""
    if not Path(terms_file).exists():
        return []
    return [t for t in Path(terms_file).read_text(encoding="utf-8").splitlines() if t.strip()]


def term_features(texts, terms_file=MEDICAL_TERMS):
    """log1p count of each medical term per report -- Kevin's dictionary as
    FEATURES the model weighs, not rules that decide. Measured on gold:
    the same terms score 0.765 as features vs 0.690 as the old rule engine,
    and concatenated to TF-IDF they lift the bagged model 0.854 -> 0.858.
    The term list is DATA (one term per line), editable without touching code;
    when the file is absent the feature block is simply empty."""
    import re
    terms = medical_terms(terms_file)
    if not terms:
        return np.zeros((len(texts), 0), dtype="float32")
    patterns = [re.compile(r"(?<!\w)" + re.escape(t) + r"(?!\w)", re.IGNORECASE) for t in terms]
    counts = np.array([[len(p.findall(str(t)))for p in patterns] for t in texts], dtype="float32")
    return np.log1p(counts)
VECTORIZER_PATH = paths.REPO_ROOT / "models" / "report_vectorizer.joblib"


def english_texts(uids, verbose=True):
    """uid -> English text, for every study we can honestly read today.

    Priority: the translation cache. Fallback: the raw report when it is
    already English (about half the corpus). Studies that are neither are
    returned in `missing` -- they join the corpus when the cache catches up.
    """
    from functions.report_translation import detect_language

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


def build_head(n_features, n_labels=len(LABELS), hidden=(256, 64), dropout=0.4):
    """TF-IDF vector -> 12 sigmoids. Small on purpose: with ~4k training
    reports, capacity is the enemy."""
    import keras
    from keras import layers
    from models.architectures import SoftAUC, SoftBinaryAccuracy

    stack = [keras.Input(shape=(n_features,), name="tfidf")]
    for i, width in enumerate(hidden):
        stack.append(layers.Dense(width, activation="relu"))
        stack.append(layers.Dropout(dropout if i == 0 else dropout / 2))
    stack.append(layers.Dense(n_labels, activation="sigmoid", name="findings"))
    model = keras.Sequential(stack)
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=[SoftAUC(name="auc", multi_label=True),
                           SoftBinaryAccuracy(name="accuracy")])
    return model


def train(epochs=30, max_features=20000, patience=5, seed=0, ngram_max=2,
          hidden=(256, 64), dropout=0.4, gold_csv=None, use_terms=False,
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
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_max),
                                 sublinear_tf=True, min_df=3)
    gold_text_list = [gold_texts[u] for u in gold.StudyInstanceUID]
    blocks = [vectorizer.fit_transform(x_text).toarray().astype("float32")]
    gold_blocks = [vectorizer.transform(gold_text_list).toarray().astype("float32")]
    if use_terms:
        blocks.append(term_features(x_text))
        gold_blocks.append(term_features(gold_text_list))
    x, x_gold = np.hstack(blocks), np.hstack(gold_blocks)
    y_gold = gold[LABELS].to_numpy(dtype="float32")
    if verbose:
        print(f"training on {x.shape[0]} reports x {x.shape[1]} features; "
              f"validating on {x_gold.shape[0]} gold")

    model = build_head(x.shape[1], hidden=hidden, dropout=dropout)
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
    out = Path(gold_csv) if gold_csv else paths.DATA / "meta" / "report_model_gold.csv"
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

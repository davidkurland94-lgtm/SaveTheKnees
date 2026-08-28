"""The twelve findings, and the two tables that supply them.

PIPELINE POSITION
    Step 1 of 5. Runs first, before any image is touched. Produces the label
    tables that steps 2-4 carry alongside the tensors.

        labels.py  ->  tensor_cache.py  ->  datasets.py  ->  architectures.py
                                                          -> train_model.py

WHEN TO USE IT
    Any time you need to know what a study's findings are: training, evaluation,
    or just counting prevalence. It reads CSVs and knows nothing about
    TensorFlow, so it imports fast and is cheap to test.

TWO LABEL SOURCES, TWO DIFFERENT JOBS

    | source  | studies | where the labels come from                          | job      |
    | ------- | ------- | --------------------------------------------------- | -------- |
    | derived | ~4350   | the Claude API reads each free-text `Report`,       | train    |
    |         |         | translates it, and scores the twelve findings       |          |
    | gold    | 58      | a human read them; the rows of `data/train.csv`     | validate |
    |         |         | that are not NaN                                    |          |

    Training on derived and validating on gold is deliberately this way round.
    The derived labels will have their own quirks - the extractor may
    systematically miss a finding, or over-call another. If validation used
    derived labels too, the model could reproduce those quirks perfectly and
    score wonderfully, and you would never find out. Scoring against gold asks
    the question you actually care about: does the model agree with a
    radiologist?

    For that to mean anything the 58 gold studies must be held out of training
    completely, even where a derived label also exists for them. A model tested
    on something it trained on tells you what it memorised, not what it learned.
    `derived_labels(exclude=...)` is what enforces the separation - pass it the
    gold UIDs, always.

LABELS is the single source of truth for the twelve findings. It fixes the CSV
column names, the model's output width, and the row order of the evaluation
table. Do not redefine it anywhere else.
"""
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def gold_labels(train_csv):
    """The 58 human-read studies: the rows with all twelve labels filled in.

    This is the validation set, and the only labels here a radiologist wrote.

    Args:
        train_csv: path to `data/train.csv`.

    Returns:
        DataFrame of StudyInstanceUID + the twelve label columns.
    """
    train = pd.read_csv(train_csv)
    gold = train[train[LABELS].notna().all(axis=1)]
    return gold[["StudyInstanceUID"] + LABELS].reset_index(drop=True)


def derived_labels(path, exclude=()):
    """The Claude-derived labels: the training set.

    `exclude` drops study UIDs that must not be trained on. Pass the gold UIDs, so
    a study present in both tables cannot leak into training and quietly inflate
    the validation score.

    Values may be soft (floats in [0, 1]) rather than 0/1. They are NOT rounded:
    binary_crossentropy accepts soft targets, and a 0.6 from the extractor carries
    real information that thresholding to 1 would throw away.

    Args:
        path: path to the derived label table (StudyInstanceUID + the 12 columns).
        exclude: iterable of StudyInstanceUIDs to drop - the gold UIDs.

    Returns:
        DataFrame of StudyInstanceUID + the twelve label columns.

    Raises:
        ValueError: if the table is missing any of the twelve label columns.
    """
    table = pd.read_csv(path)

    missing = [c for c in LABELS if c not in table.columns]
    if missing:
        raise ValueError(f"derived label table is missing columns: {missing}")

    keep = table[~table.StudyInstanceUID.isin(set(exclude))]
    keep = keep[keep[LABELS].notna().all(axis=1)]
    return keep[["StudyInstanceUID"] + LABELS].reset_index(drop=True)

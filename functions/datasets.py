"""Stream cached studies into TensorFlow, one batch at a time.

PIPELINE POSITION
    Step 3 of 5. Runs after the cache is built, and feeds the model built by
    `models/architectures.py`.

        labels.py  ->  tensor_cache.py  ->  DATASETS.PY  ->  architectures.py
                                                          -> train_model.py

WHEN TO USE IT
    Every training or evaluation run. Call it once per split - one dataset for
    training (shuffle=True), one for validation (shuffle=False).

THE SHAPE CONTRACT
    `sequence_to_tensor` returns ONE sample. There is no batch axis in it.

    | call                                        | returns             | reading                                                            | pairs with |
    | ------------------------------------------- | ------------------- | ------------------------------------------------------------------ | ---------- |
    | sequence_to_tensor(series_dir)              | (24, 224, 224, 1)   | (K, H, W, C) - the 24 slices are a DEPTH axis, 1 channel (mono)     | Conv3D     |
    | sequence_to_tensor(series_dir, as_channels) | (224, 224, 24)      | (H, W, K) - the 24 slices are CHANNELS                              | Conv2D     |

    The batch axis is added by stacking samples, so a batch of B series is
    (B, 24, 224, 224, 1) or (B, 224, 224, 24). A batch of one is x[None, ...],
    i.e. (1, 24, 224, 224, 1) - that is where the leading 1 comes from, and it
    belongs to the data, never to `layers.Input(shape=...)`.

    `layers.Input(shape=...)` takes the PER-SAMPLE shape; Keras adds the batch
    axis itself. So `Input(shape=(1, 24, 224, 224, 1))` declares a rank-6 tensor
    (None, 1, 24, 224, 224, 1), while Conv3D needs rank 5 (B, d, h, w, c). That
    was the bug. The shape is never hardcoded downstream - it is read off the
    data with `x.shape[1:]`, which drops the batch axis and leaves exactly the
    per-sample shape.

    Labels are separate: the twelve columns of the label table, one row per
    study, so (B, 12) against the model's (B, 12) output.

WHY STREAMING
    The whole training set as float32 is ~21 GB, which will not fit in memory. So
    the data is streamed: the model asks for the next batch, and only those few
    studies are read off disk. Memory holds one batch at a time instead of the
    corpus. Read the chain at the bottom of `make_dataset` in order:

        from_tensor_slices  - start from the study IDs and their labels
        shuffle             - random order, different every epoch, so the model
                              never sees the same sequence twice
        map(read)           - load the image for each ID
        batch(8)            - group into batches, because the model trains on batches
        prefetch            - prepare the next batch while the model is busy

    Note the order: shuffling happens BEFORE loading, on a list of ID strings.
    That is nearly free. Shuffling after loading would need a buffer big enough
    to hold the tensors, which is the 21 GB this design exists to avoid.
"""
import numpy as np
import tensorflow as tf

from functions.labels import LABELS
from functions.sequence_to_tensor import IMG_SIZE, K
from functions.tensor_cache import load_cached


def make_dataset(uids, labels, cache_dir, batch_size=8, shuffle=False,
                 axis="X", as_channels=False):
    """(study UIDs, label array) -> a batched tf.data.Dataset reading the cache.

    uids and labels must be row-aligned: labels[i] belongs to uids[i]. That is
    what `tensor_cache.cached_subset` guarantees - use it.

    Args:
        uids: sequence of StudyInstanceUIDs that are present in the cache.
        labels: (N, 12) array or DataFrame of label values, row-aligned to uids.
        cache_dir: the directory `build_cache` wrote to.
        shuffle: True for the training split, False for validation.
        as_channels: True for the 2.5D layout, False for 3D.

    Returns:
        tf.data.Dataset yielding (x, y) batches.

    Raises:
        ValueError: if uids and labels have different lengths.
    """
    uids = np.asarray(uids)
    labels = np.asarray(labels, dtype="float32")
    if len(uids) != len(labels):
        raise ValueError(f"{len(uids)} studies but {len(labels)} label rows")

    sample_shape = (IMG_SIZE, IMG_SIZE, K) if as_channels else (K, IMG_SIZE, IMG_SIZE, 1)

    # This runs once per study, when the dataset is asked for the next batch.
    def read(uid, y):
        # np.load is ordinary Python, and a tf.data pipeline is not Python - it is
        # a graph. numpy_function is the door between the two: it lets a plain
        # Python function run inside the pipeline.
        #
        # What comes back through that door has no shape attached, so we say what
        # it is. Without set_shape, Keras cannot work out the model input shape
        # and the error appears much later, somewhere less obvious.
        x = tf.numpy_function(
            lambda u: load_cached(u.decode(), cache_dir, axis, as_channels),
            [uid], tf.float32)
        x.set_shape(sample_shape)
        y.set_shape((len(LABELS),))
        return x, y

    ds = tf.data.Dataset.from_tensor_slices((uids, labels))
    if shuffle:
        ds = ds.shuffle(len(uids), reshuffle_each_iteration=True)   # UIDs, not tensors

    return (ds.map(read, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch_size)
              .prefetch(tf.data.AUTOTUNE))

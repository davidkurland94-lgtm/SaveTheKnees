"""What the network looks like, and how it learns.

PIPELINE POSITION
    Step 4 of 5. Takes the per-sample shape read off a dataset from
    `functions/datasets.py`, and returns a model for `train_model.py` to fit.

        labels.py  ->  tensor_cache.py  ->  datasets.py  ->  ARCHITECTURES.PY
                                                          -> train_model.py

WHEN TO USE IT
    Whenever a model object is needed - training, evaluation, or loading weights
    for inference. Pure Keras: no file I/O, no paths, no data. You can import it
    and call `build_model((24, 224, 224, 1)).summary()` with nothing on disk,
    which makes it the cheapest part of the pipeline to inspect and to test.

CALL ORDER
    build_model(input_shape)  ->  compile_model(model)  ->  fit

    Kept as separate calls on purpose. A single function that builds, compiles
    and fits cannot train more epochs, resume later, or train a model loaded
    from a file - calling it twice throws the first model away and starts over.

CHOOSING THE ARCHITECTURE
    The layout decides the convolution, and `build_model` picks by reading the
    rank of one sample. You never name the architecture directly; you choose the
    data layout (`as_channels` in `make_dataset`) and this follows.

        rank 4 (K, H, W, 1) -> a 3D CNN. The slices are a depth axis the kernels
            move through, so the model can see across slices.
        rank 3 (H, W, K)    -> a 2.5D CNN. The slices are channels, so every
            kernel already sees all 24 at once but the model has no notion of
            slice order. Far cheaper to train, which is the trade.

TWO THINGS THAT WERE IN THE FIRST DRAFT AND ARE DELIBERATELY GONE
    TimeDistributed is dropped. It exists to apply a layer at every step of a
    TIME axis, and there is no time axis here. Wrapped around the pooling it made
    Keras read the 24 slices as time and hand a rank-4 slice to a 3D pool; wrapped
    around the last Dense it produced (B, T, 12) against labels of (B, 12).

    The duplicated Dense(20) is chained, not parallel. Both lines read
    `shared_feature`, so the second overwrote the first and one layer was
    silently dropped. See `classification_head`.
"""
import tensorflow as tf
from tensorflow.keras import layers, models

from functions.labels import LABELS

# The derived labels are SOFT -- floats like 0.08 and 0.94, never 0 or 1 (see
# labels.derived_labels, which deliberately does not round them). The loss is
# fine with that; the stock Keras metrics are not, and they fail in the worst
# possible way -- silently, with numbers that look meaningful:
#
#     stock metric on soft targets     auc=0.000  accuracy=0.000  precision=1.000
#     same predictions, binarized      auc=1.000  accuracy=0.998  precision=0.998
#
# AUC and BinaryAccuracy compare y_true against thresholds it never hits, so
# they report zero. Precision/Recall cast y_true to bool, and every soft value
# is nonzero, so every target counts as positive: precision pins to 1.000 and
# looks perfect. None of it raises.
#
# The gold validation labels ARE 0/1, so val_auc -- which EarlyStopping,
# ModelCheckpoint and ReduceLROnPlateau all watch -- was never broken. Only the
# train-side numbers lied. These wrappers binarize y_true inside the metric, so
# the progress bar tells the truth while the loss keeps the soft targets, where
# the gradient information lives. On 0/1 targets binarizing is the identity, so
# validation numbers are unchanged.
LABEL_THRESHOLD = 0.5


def _hard(y_true, y_pred):
    return tf.cast(y_true > LABEL_THRESHOLD, y_pred.dtype)


@tf.keras.saving.register_keras_serializable(package="stk")
class SoftAUC(tf.keras.metrics.AUC):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(_hard(y_true, y_pred), y_pred, sample_weight)


@tf.keras.saving.register_keras_serializable(package="stk")
class SoftBinaryAccuracy(tf.keras.metrics.BinaryAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(_hard(y_true, y_pred), y_pred, sample_weight)


@tf.keras.saving.register_keras_serializable(package="stk")
class SoftRecall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(_hard(y_true, y_pred), y_pred, sample_weight)


@tf.keras.saving.register_keras_serializable(package="stk")
class SoftPrecision(tf.keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(_hard(y_true, y_pred), y_pred, sample_weight)


def build_model_3d(input_shape, n_labels):
    """rank 4 (K, H, W, 1) -> a 3D CNN.

    Conv3D kernels move through depth as well as height and width, so a filter
    sees the same spot on three neighbouring slices at once and the model can
    learn how a finding develops across the stack.
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1. (1, 2, 2) pools the two image axes but keeps all 24 slices: there
    # are only 24 of them against 224 pixels, so depth is the axis we can least
    # afford to spend first.        (24, 224, 224, 1) -> (24, 112, 112, 32)
    net = layers.Conv3D(32, 3, padding="same", activation="relu")(inputs)
    net = layers.MaxPooling3D(pool_size=(1, 2, 2), padding="same")(net)

    # Block 2. Now all three axes.  (24, 112, 112, 32) -> (12, 56, 56, 64)
    net = layers.Conv3D(64, 3, padding="same", activation="relu")(net)
    net = layers.MaxPooling3D(pool_size=(2, 2, 2), padding="same")(net)

    # Block 3. All three again.     (12, 56, 56, 64) -> (6, 28, 28, 64)
    net = layers.Conv3D(64, 3, padding="same", activation="relu")(net)
    net = layers.MaxPooling3D(pool_size=(2, 2, 2), padding="same")(net)

    # Global pooling, not Flatten: it collapses every spatial axis to one number
    # per filter, so the head does not depend on the image size.
    #                               (6, 28, 28, 64) -> (64,)
    net = layers.GlobalAveragePooling3D()(net)

    return models.Model(inputs=inputs, outputs=classification_head(net, n_labels))


def build_model_25d(input_shape, n_labels):
    """rank 3 (H, W, K) -> a 2.5D CNN.

    The 24 slices are channels, so every Conv2D kernel already reads all 24 at
    once - but channels are unordered, so unlike the 3D model this one cannot
    tell slice 5 from slice 20. It is far cheaper to train, which is the trade.
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1.                      (224, 224, 24) -> (112, 112, 32)
    net = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    net = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(net)

    # Block 2.                      (112, 112, 32) -> (56, 56, 64)
    net = layers.Conv2D(64, 3, padding="same", activation="relu")(net)
    net = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(net)

    # Block 3.                      (56, 56, 64) -> (28, 28, 64)
    net = layers.Conv2D(64, 3, padding="same", activation="relu")(net)
    net = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(net)

    # Same reason as in the 3D model.
    #                               (28, 28, 64) -> (64,)
    net = layers.GlobalAveragePooling2D()(net)

    return models.Model(inputs=inputs, outputs=classification_head(net, n_labels))

def classification_head(features, n_labels):
    """The tail both models share: (64,) of pooled features -> one score per label.

    The two Dense(20) layers are chained, not both read off `features` - that was
    the bug in the first draft, where the second silently replaced the first.

    sigmoid, not softmax: softmax would make the 12 scores sum to 1, i.e. "pick
    exactly one finding". A knee can carry several at once, so each label gets an
    independent 0-1 score.
    """
    net = layers.Dense(20, activation="relu")(features)
    net = layers.Dense(20, activation="relu")(net)
    return layers.Dense(n_labels, activation="sigmoid")(net)


def build_model(input_shape, n_labels=len(LABELS)):
    """One sample's shape -> a compile-ready model.

    input_shape is the shape of ONE series, with no batch axis:
        (24, 224, 224, 1) -> 3D CNN     (sequence_to_tensor default)
        (224, 224, 24)    -> 2.5D CNN   (as_channels=True)
    """
    if len(input_shape) == 4:
        return build_model_3d(input_shape, n_labels)
    if len(input_shape) == 3:
        return build_model_25d(input_shape, n_labels)
    raise ValueError(
        f"one sample is (K, H, W, 1) or (H, W, K), got {input_shape}. "
        "If this has a leading 1, you passed a batch of one instead of a sample."
    )


def compile_model(model):
    """Attach the optimizer, loss, and metrics. Separate from build_model so the
    architecture and the training setup can be changed independently, and so a
    model can be rebuilt without re-deciding how it is trained.

    binary_crossentropy, not categorical: it scores each of the 12 sigmoid outputs
    on its own, which is what "several findings at once" needs. It also takes soft
    targets (floats in [0, 1]) unchanged, which is what the derived labels are.

    The metrics are the Soft* wrappers above, not the stock classes, because the
    training targets are soft -- see the comment block at LABEL_THRESHOLD. The
    names ("auc" etc.) are load-bearing: train_model's callbacks monitor
    "val_auc" by that exact string.
    """
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            SoftBinaryAccuracy(name="accuracy"),
            SoftAUC(name="auc"),
            SoftRecall(name="recall"),
            SoftPrecision(name="precision"),
        ],
    )
    return model

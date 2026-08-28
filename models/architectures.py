import tensorflow as tf
from tensorflow.keras import layers, models

from functions.labels import LABELS

LABEL_THRESHOLD = 0.5

def _hard(y_true, y_pred):
    return tf.cast(y_true > LABEL_THRESHOLD, y_pred.dtype)


@tf.keras.utils.register_keras_serializable(package="stk")
class SoftAUC(tf.keras.metrics.AUC):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(_hard(y_true, y_pred), y_pred, sample_weight)


@tf.keras.utils.register_keras_serializable(package="stk")
class SoftBinaryAccuracy(tf.keras.metrics.BinaryAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(_hard(y_true, y_pred), y_pred, sample_weight)


@tf.keras.utils.register_keras_serializable(package="stk")
class SoftRecall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(_hard(y_true, y_pred), y_pred, sample_weight)


@tf.keras.utils.register_keras_serializable(package="stk")
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


def _encode_3d(inputs):
    """The 3-block Conv3D stack shared by the single- and multi-plane models:
    (K, H, W, 1) -> (64,) pooled features. Fresh layers per call, so each plane
    in the multi-plane model learns its own filters -- sagittal anatomy and
    axial anatomy do not look alike."""
    net = layers.Conv3D(32, 3, padding="same", activation="relu")(inputs)
    net = layers.MaxPooling3D(pool_size=(1, 2, 2), padding="same")(net)
    net = layers.Conv3D(64, 3, padding="same", activation="relu")(net)
    net = layers.MaxPooling3D(pool_size=(2, 2, 2), padding="same")(net)
    net = layers.Conv3D(64, 3, padding="same", activation="relu")(net)
    net = layers.MaxPooling3D(pool_size=(2, 2, 2), padding="same")(net)
    return layers.GlobalAveragePooling3D()(net)


def build_model_multiplane(input_shape, n_labels=len(LABELS)):
    """Three volumes of ONE study -- sagittal, coronal, axial -- one prediction.

    Every study in the dataset has all three planes (verified: 4,407/4,407),
    so each training step reads the same knee from three directions and the
    head learns from the fused 192 features. This is the fusion upgrade of the
    per-plane-ensemble idea: instead of averaging three opinions after the
    fact, the planes inform each other during training.

    input_shape is ONE plane's per-sample shape, (K, H, W, 1); all three share it.
    """
    plane_inputs = [layers.Input(shape=input_shape, name=f"plane_{name}")
                    for name in ("sagittal", "coronal", "axial")]
    fused = layers.Concatenate()([_encode_3d(inp) for inp in plane_inputs])
    return models.Model(inputs=plane_inputs,
                        outputs=classification_head(fused, n_labels))


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

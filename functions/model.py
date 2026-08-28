"""
The model that consumes what sequence_to_tensor.py produces.

ResNet50 is a 2D CNN and wants (224, 224, 3). A study is (24, 224, 224, 1) --
a stack of slices, not one image. So the backbone is wrapped in
TimeDistributed: every slice is encoded independently, then the 24 embeddings
are pooled into one study-level vector.

    (B, 24, 224, 224, 1)  ->  (B, 24, 2048)  ->  (B, 2048)  ->  (B, 12)

Two details are easy to get wrong and fail SILENTLY:

  1. resnet50.preprocess_input is mode="caffe" and expects [0, 255].
     sequence_to_tensor gives [0, 1]. Feeding [0, 1] straight in subtracts the
     ImageNet mean (~110) from numbers below 1, so every feature is garbage and
     nothing raises. Rescaling(255) below is not cosmetic.

  2. Mean-pooling over slices is only valid because pick_slice_indices repeats
     REAL slices instead of black-padding short series. A padded loader would
     drag every embedding toward the padding.

Preprocessing is done with Rescaling + Normalization rather than a Lambda, so
the model survives keras.saving.save_model / load_model with no custom_objects
and no safe_mode=False.
"""

import keras
from keras import layers
from keras.applications import ResNet50

from functions.sequence_to_tensor import K, IMG_SIZE

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

# What preprocess_input(mode="caffe") subtracts. It also swaps RGB->BGR, which
# is a no-op here because the three channels are copies of one grayscale slice.
CAFFE_MEAN = [103.939, 116.779, 123.68]


def build_model(k=K, img_size=IMG_SIZE, labels=LABELS,
                trainable_base=False, dropout=0.3, weights="imagenet"):
    """(k, img_size, img_size, 1) -> one probability per label.

    trainable_base=False freezes ResNet50, leaving ~25k trainable params. That
    is the setting to start from: it trains in seconds on precomputed
    embeddings and will not overfit 4,407 studies the way a full 23.6M-param
    fine-tune does.
    """
    base = ResNet50(weights=weights, include_top=False, pooling="avg",
                    input_shape=(img_size, img_size, 3))
    base.trainable = trainable_base

    inputs = keras.Input(shape=(k, img_size, img_size, 1), name="sequence")
    x = layers.Concatenate(axis=-1, name="gray_to_rgb")([inputs, inputs, inputs])
    x = layers.Rescaling(255.0, name="to_uint8_range")(x)
    x = layers.Normalization(axis=-1, mean=CAFFE_MEAN, variance=[1.0, 1.0, 1.0],
                             name="imagenet_centre")(x)
    x = layers.TimeDistributed(base, name="slice_encoder")(x)
    x = layers.GlobalAveragePooling1D(name="pool_slices")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(len(labels), activation="sigmoid", name="findings")(x)

    return keras.Model(inputs, outputs, name="knee_resnet50")


def compile_model(model, learning_rate=1e-3):
    """Multi-label, so binary crossentropy per finding -- NOT categorical.
    A knee can have an ACL tear and an effusion at the same time.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc", multi_label=True),
                 keras.metrics.BinaryAccuracy(name="acc")],
    )
    return model


if __name__ == "__main__":
    m = compile_model(build_model())
    m.summary()

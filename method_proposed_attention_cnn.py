"""
Paper-based implementation (TensorFlow/Keras)

Paper:
Sumon et al., "Automatic Fracture Detection Convolutional Neural Network
with Multiple Attention Blocks Using Multi-Region X-Ray Data",
Life 2025, 15, 1135.
DOI: 10.3390/life15071135

This implementation follows the METHOD text of the paper, not only Figure 3.

Paper-specified points used here
-------------------------------
- Input: 128 x 128 x 3
- Two initial convolution layers with 32 filters
- Squeeze-and-Excitation (SE) block
- Subsequent convolution layers with 64 filters
- CBAM: channel attention followed by spatial attention
- Residual skip paths resized with 1x1 convolution + max pooling
- Residual fusion: x = ReLU(x + skip_adjusted)
- Conv block with 128 filters
- Additional spatial-attention block
- Skip from CBAM-stage output to spatial-attention-stage output
- Final convolution with 256 filters
- Flatten -> Dense -> Dropout -> 1-unit sigmoid output
- Adam optimizer, learning rate 0.001
- Batch size 32
- Binary classification

Ambiguities in the article
--------------------------
1) The paper does not unambiguously state the number of units in the final Dense layer.
   DENSE_UNITS=1024 is therefore an implementation choice.
2) The paper's CBAM equation describes a 3x3 spatial-attention convolution,
   whereas Sec. 3.6 describes the standalone spatial-attention module using 7x7.
   This code uses 3x3 inside CBAM and 7x7 for the later standalone spatial block.
3) Exact dropout rate is not completely specified in the method text used here.
   DROP_RATE=0.5 is a common/reported setting used in the implementation.
"""

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
)

# ============================================================
# Configuration
# ============================================================
SEED = 42
IMG_SIZE = (128, 128)
INPUT_SHAPE = (128, 128, 3)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
DROP_RATE = 0.5
DENSE_UNITS = 1024
SE_REDUCTION = 16
CBAM_REDUCTION = 16

tf.keras.utils.set_random_seed(SEED)


# ============================================================
# Squeeze-and-Excitation block
# Paper Eq. (1): GAP -> FC/ReLU -> FC/Sigmoid -> channel scaling
# ============================================================
def se_block(x, reduction=16, name="se"):
    channels = int(x.shape[-1])
    reduced = max(channels // reduction, 1)

    z = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    z = layers.Dense(
        reduced,
        activation="relu",
        name=f"{name}_fc1",
    )(z)
    z = layers.Dense(
        channels,
        activation="sigmoid",
        name=f"{name}_fc2",
    )(z)
    z = layers.Reshape((1, 1, channels), name=f"{name}_reshape")(z)

    return layers.Multiply(name=f"{name}_scale")([x, z])


# ============================================================
# CBAM channel attention
# Paper Eq. (2):
# sigmoid(MLP(AvgPool(F)) + MLP(MaxPool(F)))
# ============================================================
def cbam_channel_attention(x, reduction=16, name="cbam_channel"):
    channels = int(x.shape[-1])
    reduced = max(channels // reduction, 1)

    shared_fc1 = layers.Dense(
        reduced,
        activation="relu",
        name=f"{name}_fc1",
    )
    shared_fc2 = layers.Dense(
        channels,
        activation=None,
        name=f"{name}_fc2",
    )

    avg = layers.GlobalAveragePooling2D(name=f"{name}_avg_pool")(x)
    mx = layers.GlobalMaxPooling2D(name=f"{name}_max_pool")(x)

    avg = shared_fc2(shared_fc1(avg))
    mx = shared_fc2(shared_fc1(mx))

    a = layers.Add(name=f"{name}_add")([avg, mx])
    a = layers.Activation("sigmoid", name=f"{name}_sigmoid")(a)
    a = layers.Reshape((1, 1, channels), name=f"{name}_reshape")(a)

    return layers.Multiply(name=f"{name}_scale")([x, a])


# ============================================================
# Spatial attention
# Average and max pooling over channel axis, concatenate, Conv, sigmoid.
# ============================================================
def spatial_attention(x, kernel_size=7, name="spatial_attention"):
    avg_map = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
        name=f"{name}_avg",
    )(x)

    max_map = layers.Lambda(
        lambda t: tf.reduce_max(t, axis=-1, keepdims=True),
        name=f"{name}_max",
    )(x)

    z = layers.Concatenate(axis=-1, name=f"{name}_concat")(
        [avg_map, max_map]
    )

    a = layers.Conv2D(
        filters=1,
        kernel_size=kernel_size,
        padding="same",
        activation="sigmoid",
        name=f"{name}_conv",
    )(z)

    return layers.Multiply(name=f"{name}_scale")([x, a])


# ============================================================
# Complete CBAM
# Paper Sec. 3.5: channel attention followed by spatial attention.
# The equation in Sec. 3.3 denotes f^(3x3), so kernel_size=3 here.
# ============================================================
def cbam_block(x, reduction=16, name="cbam"):
    x = cbam_channel_attention(
        x,
        reduction=reduction,
        name=f"{name}_channel",
    )
    x = spatial_attention(
        x,
        kernel_size=3,
        name=f"{name}_spatial",
    )
    return x


# ============================================================
# Residual projection
# Paper Eq. (5):
# Skip_adj = MaxPool(Conv2D_1x1(skip))
# ============================================================
def projected_skip(x, filters, pool_size, name):
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        activation=None,
        name=f"{name}_conv1x1",
    )(x)

    x = layers.MaxPooling2D(
        pool_size=pool_size,
        name=f"{name}_pool",
    )(x)

    return x


# ============================================================
# Proposed network
# ============================================================
def build_proposed_cnn(
    input_shape=INPUT_SHAPE,
    dense_units=DENSE_UNITS,
    dropout_rate=DROP_RATE,
):
    inp = layers.Input(shape=input_shape, name="input_xray")

    # --------------------------------------------------------
    # Block 1: two 32-filter convolutional layers + SE
    # --------------------------------------------------------
    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
        name="conv32_1",
    )(inp)

    x = layers.Dropout(
        dropout_rate,
        name="dropout_32",
    )(x)

    x = layers.MaxPooling2D(
        (2, 2),
        name="pool32_1",
    )(x)

    x = layers.Conv2D(
        32,
        (2, 2),
        padding="same",
        activation="relu",
        name="conv32_2",
    )(x)

    x = layers.MaxPooling2D(
        (2, 2),
        name="pool32_2",
    )(x)

    se_out = se_block(
        x,
        reduction=SE_REDUCTION,
        name="se32",
    )

    # --------------------------------------------------------
    # Block 2: 64-filter convolutional stage
    # --------------------------------------------------------
    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
        name="conv64_1",
    )(se_out)

    x = layers.Dropout(
        dropout_rate,
        name="dropout_64",
    )(x)

    x = layers.MaxPooling2D(
        (2, 2),
        name="pool64_1",
    )(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
        name="conv64_2",
    )(x)

    x = layers.MaxPooling2D(
        (2, 2),
        name="pool64_2",
    )(x)

    # CBAM = channel attention + spatial attention
    cbam_out = cbam_block(
        x,
        reduction=CBAM_REDUCTION,
        name="cbam64",
    )

    # --------------------------------------------------------
    # Residual connection from SE output to CBAM output.
    # Paper: 1x1 Conv + MaxPool for dimension matching,
    # then x = ReLU(x + Skip_adj)
    #
    # SE output:     32x32x32
    # projected:      8x8x64
    # CBAM output:    8x8x64
    # --------------------------------------------------------
    skip1 = projected_skip(
        se_out,
        filters=64,
        pool_size=(4, 4),
        name="skip_se_to_cbam",
    )

    x = layers.Add(name="residual_add_64")([cbam_out, skip1])
    x = layers.ReLU(name="residual_relu_64")(x)

    cbam_residual_out = x

    # --------------------------------------------------------
    # 128-filter block + standalone spatial attention
    # --------------------------------------------------------
    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu",
        name="conv128",
    )(x)

    x = layers.MaxPooling2D(
        (2, 2),
        name="pool128",
    )(x)

    # Sec. 3.6 explicitly describes a 7x7 conv in this SAB.
    spatial_out = spatial_attention(
        x,
        kernel_size=7,
        name="standalone_spatial128",
    )

    # --------------------------------------------------------
    # Additional skip from CBAM-stage output to spatial block
    #
    # CBAM residual: 8x8x64
    # projected:     4x4x128
    # spatial out:   4x4x128
    # --------------------------------------------------------
    skip2 = projected_skip(
        cbam_residual_out,
        filters=128,
        pool_size=(2, 2),
        name="skip_cbam_to_spatial",
    )

    x = layers.Add(name="residual_add_128")([spatial_out, skip2])
    x = layers.ReLU(name="residual_relu_128")(x)

    # --------------------------------------------------------
    # Final 256-filter convolutional stage
    # --------------------------------------------------------
    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu",
        name="conv256",
    )(x)

    x = layers.MaxPooling2D(
        (2, 2),
        name="pool256",
    )(x)

    # Paper explicitly says global flattening, not GAP here.
    x = layers.Flatten(name="flatten")(x)

    x = layers.Dense(
        dense_units,
        activation="relu",
        name="dense",
    )(x)

    x = layers.Dropout(
        dropout_rate,
        name="dropout_final",
    )(x)

    out = layers.Dense(
        1,
        activation="sigmoid",
        name="fracture_output",
    )(x)

    return Model(inp, out, name="Proposed_Attention_CNN")


# ============================================================
# Build and compile
# ============================================================
model = build_proposed_cnn()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

model.summary()
print("\nTotal parameters:", model.count_params())


# ============================================================
# Dataset loader
#
# Expected:
# dataset/
#   train/
#       fractured/
#       non_fractured/
#   val/
#       fractured/
#       non_fractured/
#   test/
#       fractured/
#       non_fractured/
# ============================================================
def make_dataset(directory, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )


# ============================================================
# Training helper
# ============================================================
def train_model(
    train_dir,
    val_dir,
    epochs=EPOCHS,
):
    train_ds = make_dataset(train_dir, shuffle=True)
    val_ds = make_dataset(val_dir, shuffle=False)

    # Normalize images to [0, 1].
    normalizer = layers.Rescaling(1.0 / 255.0)

    train_ds = train_ds.map(
        lambda x, y: (normalizer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(
        lambda x, y: (normalizer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "best_proposed_attention_cnn.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    return history


# ============================================================
# Test helper
# ============================================================
def evaluate_model(test_dir):
    test_ds = make_dataset(test_dir, shuffle=False)
    class_names = test_ds.class_names

    normalizer = layers.Rescaling(1.0 / 255.0)
    test_ds_norm = test_ds.map(
        lambda x, y: (normalizer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    results = model.evaluate(
        test_ds_norm,
        return_dict=True,
        verbose=1,
    )

    y_true = np.concatenate(
        [y.numpy().ravel() for _, y in test_ds]
    ).astype(int)

    y_prob = model.predict(
        test_ds_norm,
        verbose=1,
    ).ravel()

    y_pred = (y_prob >= 0.5).astype(int)

    print("\nKeras metrics")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    print("\nClassification report")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
        )
    )

    print("Confusion matrix")
    print(confusion_matrix(y_true, y_pred))

    paper_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "cohen_kappa": float(
            cohen_kappa_score(y_true, y_pred)
        ),
    }

    print("\nPaper-style metrics")
    for k, v in paper_metrics.items():
        print(f"{k}: {v:.4f}")

    return paper_metrics


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    # Just building the model is enough to inspect it.
    #
    # To train:
    #
    # history = train_model(
    #     "dataset/train",
    #     "dataset/val",
    #     epochs=50,
    # )
    #
    # To test:
    #
    # metrics = evaluate_model("dataset/test")
    #
    pass

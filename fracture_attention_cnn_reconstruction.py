"""

Paper:
"Automatic Fracture Detection Convolutional Neural Network with Multiple
Attention Blocks Using Multi-Region X-Ray Data"
Life 2025, 15, 1135
DOI: 10.3390/life15071135

Expected dataset directory:
dataset/
    train/
        fractured/
        non_fractured/
    val/
        fractured/
        non_fractured/
    test/
        fractured/
        non_fractured/

- Input: 128 x 128 x 3
- Binary fracture classification
- Convolution filters progressing through 32, 64, 128, 256
- Squeeze/SE attention + CBAM attention
- ReLU hidden activations
- Sigmoid output
- Adam, lr=0.001
- Batch size=32
- Dropout=0.5 after dense layer
"""

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
)

# ============================================================
# 1. Configuration
# ============================================================
SEED = 42
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 50
DROPOUT = 0.5

DATA_DIR = "dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

tf.keras.utils.set_random_seed(SEED)

# ============================================================
# 2. Dataset
# ============================================================
def load_directory_dataset(directory, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )

train_ds = load_directory_dataset(TRAIN_DIR, shuffle=True)
val_ds = load_directory_dataset(VAL_DIR, shuffle=False)
test_ds = load_directory_dataset(TEST_DIR, shuffle=False)

class_names = train_ds.class_names
print("Class names:", class_names)

AUTOTUNE = tf.data.AUTOTUNE

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.05, 0.05),
    ],
    name="data_augmentation",
)

def prepare(ds, training=False):
    if training:
        ds = ds.shuffle(1024, seed=SEED)
    return ds.prefetch(AUTOTUNE)

train_ds = prepare(train_ds, training=True)
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)

# ============================================================
# 3. Squeeze-and-Excitation (SE / "Squeeze") block
# ============================================================
def squeeze_excitation_block(x, reduction=16, name="se"):
    channels = int(x.shape[-1])

    se = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    se = layers.Reshape((1, 1, channels), name=f"{name}_reshape")(se)

    reduced_channels = max(channels // reduction, 1)

    se = layers.Dense(
        reduced_channels,
        activation="relu",
        use_bias=False,
        name=f"{name}_fc1",
    )(se)
    se = layers.Dense(
        channels,
        activation="sigmoid",
        use_bias=False,
        name=f"{name}_fc2",
    )(se)

    return layers.Multiply(name=f"{name}_scale")([x, se])

# ============================================================
# 4. CBAM: Channel Attention
# ============================================================
def channel_attention(x, reduction=16, name="ca"):
    channels = int(x.shape[-1])
    reduced_channels = max(channels // reduction, 1)

    shared_dense_1 = layers.Dense(
        reduced_channels,
        activation="relu",
        use_bias=True,
        name=f"{name}_shared_fc1",
    )
    shared_dense_2 = layers.Dense(
        channels,
        use_bias=True,
        name=f"{name}_shared_fc2",
    )

    avg_pool = layers.GlobalAveragePooling2D(name=f"{name}_avg_pool")(x)
    max_pool = layers.GlobalMaxPooling2D(name=f"{name}_max_pool")(x)

    avg_pool = layers.Reshape((1, 1, channels))(avg_pool)
    max_pool = layers.Reshape((1, 1, channels))(max_pool)

    avg_attn = shared_dense_2(shared_dense_1(avg_pool))
    max_attn = shared_dense_2(shared_dense_1(max_pool))

    attn = layers.Add(name=f"{name}_add")([avg_attn, max_attn])
    attn = layers.Activation("sigmoid", name=f"{name}_sigmoid")(attn)

    return layers.Multiply(name=f"{name}_scale")([x, attn])

# ============================================================
# 5. CBAM: Spatial Attention

# concatenation, 7x7 convolution, sigmoid.
# ============================================================
def spatial_attention(x, name="sa"):
    avg_map = layers.Lambda(
        lambda z: tf.reduce_mean(z, axis=-1, keepdims=True),
        name=f"{name}_avg",
    )(x)

    max_map = layers.Lambda(
        lambda z: tf.reduce_max(z, axis=-1, keepdims=True),
        name=f"{name}_max",
    )(x)

    concat = layers.Concatenate(axis=-1, name=f"{name}_concat")(
        [avg_map, max_map]
    )

    attn = layers.Conv2D(
        1,
        kernel_size=7,
        strides=1,
        padding="same",
        activation="sigmoid",
        use_bias=False,
        name=f"{name}_conv7",
    )(concat)

    return layers.Multiply(name=f"{name}_scale")([x, attn])

def cbam_block(x, reduction=16, name="cbam"):
    x = channel_attention(x, reduction=reduction, name=f"{name}_channel")
    x = spatial_attention(x, name=f"{name}_spatial")
    return x

# ============================================================
# 6. Residual convolutional attention block
# Figure 3 residual/skip connections.
# ============================================================
def residual_attention_block(x, filters, use_se=True, use_cbam=True, name="block"):
    shortcut = x

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        name=f"{name}_conv1",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        name=f"{name}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)

    if use_se:
        x = squeeze_excitation_block(x, reduction=16, name=f"{name}_se")

    if use_cbam:
        x = cbam_block(x, reduction=16, name=f"{name}_cbam")

    if int(shortcut.shape[-1]) != filters:
        shortcut = layers.Conv2D(
            filters,
            1,
            padding="same",
            use_bias=False,
            name=f"{name}_shortcut_conv",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_shortcut_bn")(shortcut)

    x = layers.Add(name=f"{name}_residual_add")([x, shortcut])
    x = layers.ReLU(name=f"{name}_out_relu")(x)

    return x

# ============================================================
# 7. Proposed CNN reconstruction
# ============================================================
def build_proposed_model():
    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="xray")

    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(x)

    # Stem: 32 filters
    x = layers.Conv2D(
        32, 3, padding="same", use_bias=False, name="stem_conv"
    )(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    # Progressive filters reported in the paper: 32, 64, 128, 256
    x = residual_attention_block(
        x, 32, use_se=True, use_cbam=True, name="stage1"
    )
    x = layers.MaxPooling2D(2, name="pool1")(x)

    x = residual_attention_block(
        x, 64, use_se=True, use_cbam=True, name="stage2"
    )
    x = layers.MaxPooling2D(2, name="pool2")(x)

    x = residual_attention_block(
        x, 128, use_se=True, use_cbam=True, name="stage3"
    )
    x = layers.MaxPooling2D(2, name="pool3")(x)

    x = residual_attention_block(
        x, 256, use_se=True, use_cbam=True, name="stage4"
    )

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # A compact classification head. Paper specifies 0.5 dropout after FC layers.
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.Dropout(DROPOUT, name="dropout")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="fracture_probability")(x)

    return models.Model(inputs, outputs, name="Fracture_Attention_CNN")

model = build_proposed_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

model.summary()

# ============================================================
# 8. Training
# ============================================================
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_fracture_attention_cnn.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
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
    tf.keras.callbacks.CSVLogger("training_history.csv"),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ============================================================
# 9. Learning curves
# ============================================================
def plot_training_history(history):
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_curve.png", dpi=300)
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=300)
    plt.show()

plot_training_history(history)

# ============================================================
# 10. Test-set evaluation
# ============================================================
model = tf.keras.models.load_model(
    "best_fracture_attention_cnn.keras",
    custom_objects={},
    safe_mode=False,
)

test_values = model.evaluate(test_ds, verbose=1, return_dict=True)
print("\nKeras test metrics:")
for key, value in test_values.items():
    print(f"{key}: {value:.5f}")

y_true = np.concatenate([y.numpy().ravel() for _, y in test_ds]).astype(int)
y_prob = model.predict(test_ds, verbose=1).ravel()
y_pred = (y_prob >= 0.5).astype(int)

metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred, zero_division=0),
    "recall": recall_score(y_true, y_pred, zero_division=0),
    "f1_score": f1_score(y_true, y_pred, zero_division=0),
    "cohen_kappa": cohen_kappa_score(y_true, y_pred),
}

print("\nPaper-style metrics:")
for key, value in metrics.items():
    print(f"{key}: {value:.5f}")

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion matrix:")
print(cm)

with open("test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# ============================================================
# 11. Confusion matrix plot
# ============================================================
plt.figure(figsize=(5, 5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks([0, 1], class_names, rotation=30)
plt.yticks([0, 1], class_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# ============================================================
# 12. Single-image inference
# ============================================================
def predict_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.expand_dims(arr, axis=0)

    probability = float(model.predict(arr, verbose=0)[0, 0])
    predicted_index = int(probability >= 0.5)

    return {
        "predicted_class": class_names[predicted_index],
        "fracture_probability": probability,
    }

# Example:
# print(predict_image("example_xray.jpg"))

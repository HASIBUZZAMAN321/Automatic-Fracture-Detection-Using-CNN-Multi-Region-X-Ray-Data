from pathlib import Path
import tensorflow as tf
from model import build_model

IMG_SIZE=(128,128)
BATCH_SIZE=32
EPOCHS=100
LR=1e-3
SEED=42

def load_split(path, shuffle):
    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )
    return ds

def main(data_root="dataset"):
    train_ds = load_split(f"{data_root}/train", True)
    val_ds   = load_split(f"{data_root}/val", False)

    aug = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1/255.0),
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(20/360.0),
        tf.keras.layers.RandomZoom(0.10),
    ])

    norm = tf.keras.layers.Rescaling(1/255.0)

    train_ds = train_ds.map(lambda x,y:(aug(x, training=True),y),
                            num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x,y:(norm(x),y),
                        num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    model = build_model(
        input_shape=(128,128,3),
        dense_units=1024,       # NOT specified in paper; tune if exact parameter matching is required
        conv_dropout=0.3,       # Results section reports 0.3
        fc_dropout=0.5,         # Discussion reports 0.5 after FC
        cbam_spatial_kernel=7,  # set 3 to follow Eq. (3) literally
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    model.summary()
    print("Parameter count:", model.count_params())

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "best_model.keras", monitor="val_accuracy",
            save_best_only=True, mode="max"
        ),
        tf.keras.callbacks.CSVLogger("training_log.csv"),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )
    return history

if __name__ == "__main__":
    main()

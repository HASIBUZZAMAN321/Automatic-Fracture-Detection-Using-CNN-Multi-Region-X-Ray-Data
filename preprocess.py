import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = (128,128)

def denoise_numpy(img):
    """
    Implements the paper-described Gaussian + median filtering.
    img is uint8 RGB.
    """
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.medianBlur(img, 3)
    return img

def preprocess_image(path, training=False):
    b = tf.io.read_file(path)
    img = tf.image.decode_image(b, channels=3, expand_animations=False)
    img.set_shape([None,None,3])
    img = tf.image.resize(img, IMG_SIZE)

    # OpenCV denoising through numpy_function.
    img = tf.numpy_function(
        lambda z: denoise_numpy(np.clip(z,0,255).astype(np.uint8)),
        [img], tf.uint8
    )
    img.set_shape([128,128,3])
    img = tf.cast(img, tf.float32) / 255.0

    if training:
        # Paper explicitly states horizontal + vertical flips,
        # rotation within +/-20 degrees, and scaling.
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)

    return img

def augmentation_layer():
    # Rotation factor: 20/360 ≈ 0.0556 turns.
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(20/360.0, fill_mode="nearest"),
        tf.keras.layers.RandomZoom(height_factor=(-0.10,0.10),
                                   width_factor=(-0.10,0.10)),
    ], name="paper_augmentation")

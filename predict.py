import tensorflow as tf
import numpy as np

IMG_SIZE=(128,128)

model = tf.keras.models.load_model("best_model.keras", safe_mode=False)

def predict(path):
    img = tf.keras.utils.load_img(path,target_size=IMG_SIZE)
    x = tf.keras.utils.img_to_array(img) / 255.0
    p = float(model.predict(x[None,...],verbose=0)[0,0])
    return {
        "probability_fracture": p,
        "prediction": "Fracture" if p >= 0.5 else "Non-Fracture"
    }

# Example:
# print(predict("sample.jpg"))

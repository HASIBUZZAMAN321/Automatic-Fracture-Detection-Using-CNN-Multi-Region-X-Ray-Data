import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, cohen_kappa_score
)

IMG_SIZE=(128,128)
BATCH_SIZE=32

def main(test_dir="dataset/test", model_path="best_model.keras"):
    raw = tf.keras.utils.image_dataset_from_directory(
        test_dir, labels="inferred", label_mode="binary",
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
    )
    class_names = raw.class_names
    norm = tf.keras.layers.Rescaling(1/255.0)
    ds = raw.map(lambda x,y:(norm(x),y)).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.load_model(model_path, safe_mode=False)
    y_true = np.concatenate([y.numpy().ravel() for _,y in raw]).astype(int)
    y_prob = model.predict(ds).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true,y_pred)
    metrics = {
        "accuracy": accuracy_score(y_true,y_pred),
        "precision": precision_score(y_true,y_pred,zero_division=0),
        "recall": recall_score(y_true,y_pred,zero_division=0),
        "f1": f1_score(y_true,y_pred,zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true,y_pred),
    }
    print(metrics)
    print(classification_report(y_true,y_pred,target_names=class_names,digits=4))
    print("Confusion matrix:\n", cm)

    with open("metrics.json","w") as f:
        json.dump(metrics,f,indent=2)

    plt.figure(figsize=(5,5))
    plt.imshow(cm)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=25)
    plt.yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,str(cm[i,j]),ha="center",va="center")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png",dpi=300)
    plt.show()

if __name__ == "__main__":
    main()

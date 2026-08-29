import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_log.csv")

plt.figure(figsize=(7,5))
plt.plot(df["epoch"],df["accuracy"],label="Training")
plt.plot(df["epoch"],df["val_accuracy"],label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_curve.png",dpi=300)
plt.show()

plt.figure(figsize=(7,5))
plt.plot(df["epoch"],df["loss"],label="Training")
plt.plot(df["epoch"],df["val_loss"],label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png",dpi=300)
plt.show()

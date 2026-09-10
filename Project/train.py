import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score


SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "Data")
PREP_DIR     = os.path.join(DATA_DIR, "Preprocessed")
PRE_IMG_DIR  = os.path.join(PREP_DIR, "Images")
PRE_MSK_DIR  = os.path.join(PREP_DIR, "Masks")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "Models")

IMG_FILE = os.path.join(PRE_IMG_DIR,  "images_preprocessed.npy")
MSK_FILE = os.path.join(PRE_MSK_DIR,  "masks_preprocessed.npy")


from model import unet_model


def main():
    start_time = time.time()

    print(f"Loading data from:\n  {IMG_FILE}\n  {MSK_FILE}")
    X = np.load(IMG_FILE)  
    Y = np.load(MSK_FILE)   

    Y = (Y > 0.5).astype(np.uint8)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    print(f"Train/val split: {len(X_train)} / {len(X_val)} samples")

    model = unet_model(input_size=X.shape[1:])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=50,
        batch_size=16,
        verbose=1
    )

    preds     = model.predict(X_val)
    preds_bin = (preds > 0.5).astype(np.uint8)

    y_true = Y_val.flatten()
    y_pred = preds_bin.flatten()

    dice      = f1_score(y_true, y_pred)
    iou       = jaccard_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)

    df_metrics = pd.DataFrame({
        "Metric":    ["Dice", "IoU", "Precision", "Recall"],
        "Score":     [dice, iou, precision, recall]
    })

    print("\nSegmentation Metrics on Validation Set:\n")
    print(df_metrics.to_string(index=False))

    plt.figure(figsize=(8,4))
    plt.plot(history.history["loss"],   label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    idx = np.random.randint(0, X_val.shape[0])
    fig, axes = plt.subplots(1,3, figsize=(12,4))
    for ax, img, title in zip(axes,
                              [X_val[idx,:,:,0], Y_val[idx,:,:,0], preds_bin[idx,:,:,0]],
                              ["Input Image","Ground Truth","Prediction"]):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "unet_model.keras")
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
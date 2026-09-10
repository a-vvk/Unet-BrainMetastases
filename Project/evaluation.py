import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
import pandas as pd


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SCRIPT_DIR)

    img_pre_path = os.path.join(
        BASE_DIR, "Data", "Preprocessed", "Images", "images_preprocessed.npy"
    )
    msk_pre_path = os.path.join(
        BASE_DIR, "Data", "Preprocessed", "Masks", "masks_preprocessed.npy"
    )

    X = np.load(img_pre_path)  
    Y = np.load(msk_pre_path)  
    Y = (Y > 0.5).astype(np.uint8)  

    _, X_val, _, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    model_path = os.path.join(BASE_DIR, "Models", "unet_model.keras")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.MeanIoU(num_classes=2, name="mean_iou"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    preds = model.predict(X_val)
    preds_bin = (preds > 0.5).astype(np.uint8)

    y_true = Y_val.flatten()
    y_pred = preds_bin.flatten()

    dice      = f1_score(y_true, y_pred)
    iou       = jaccard_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)

    df_metrics = pd.DataFrame({
        "Metric":   ["Dice", "IoU", "Precision", "Recall"],
        "Score":    [dice, iou, precision, recall]
    })

    print("\nSegmentation Metrics on Validation Set:\n")
    print(df_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
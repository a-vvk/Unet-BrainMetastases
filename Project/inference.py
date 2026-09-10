import os
import argparse
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="Run U-Net inference on preprocessed data")
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to your saved Keras model (.keras or .h5)."
    )
    p.add_argument(
        "--images",
        type=str,
        default=None,
        help="Path to images_preprocessed.npy"
    )
    p.add_argument(
        "--masks",
        type=str,
        default=None,
        help="(Optional) Path to masks_preprocessed.npy for computing metrics"
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarization threshold"
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save predictions and plots"
    )
    return p.parse_args()

def autodiscover(args):
    SCRIPT = os.path.dirname(os.path.abspath(__file__))
    BASE   = os.path.dirname(SCRIPT)

    if args.model is None:
        args.model = os.path.join(BASE, "Models", "unet_model.keras")
    if args.images is None:
        args.images = os.path.join(BASE, "Data", "Preprocessed", "Images", "images_preprocessed.npy")
    if args.masks is None:
        maybe = os.path.join(BASE, "Data", "Preprocessed", "Masks", "masks_preprocessed.npy")
        if os.path.isfile(maybe):
            args.masks = maybe
    if args.output_dir is None:
        args.output_dir = os.path.join(BASE, "Results", "inference")
    os.makedirs(args.output_dir, exist_ok=True)
    return args

def load_npy(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Could not find `{path}`")
    return np.load(path)

def binarize(arr, thresh):
    return (arr > thresh).astype(np.uint8)

def compute_metrics(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return {
        "Dice"     : f1_score(y_true, y_pred, zero_division=0),
        "IoU"      : jaccard_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall"   : recall_score(y_true, y_pred, zero_division=0),
    }

def plot_examples(X, Y_true, Y_pred, outdir, n=3):
    os.makedirs(outdir, exist_ok=True)
    N = X.shape[0]
    idxs = np.random.choice(N, size=min(n, N), replace=False)
    for i, idx in enumerate(idxs):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(X[idx,:,:,0], cmap="gray")
        axes[0].set_title("Input")
        axes[1].imshow(Y_true[idx,:,:,0], cmap="gray") if Y_true is not None else None
        axes[1].set_title("Ground Truth")
        axes[2].imshow(Y_pred[idx,:,:,0], cmap="gray")
        axes[2].set_title("Predicted")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"example_{i}.png"))
        plt.close(fig)

def main():
    args = parse_args()
    args = autodiscover(args)

    print(f"Loading model from {args.model}…")
    model = tf.keras.models.load_model(args.model, compile=False)

    print(f"Loading images from {args.images}…")
    X = load_npy(args.images)            

    Y_true = None
    if args.masks:
        print(f"Loading masks from {args.masks}…")
        Y_true = load_npy(args.masks)    
        Y_true = (Y_true > args.threshold).astype(np.uint8)

    print(f"Running inference on {X.shape[0]} slices (batch size={args.batch_size})…")
    t0 = time.time()
    preds = model.predict(X, batch_size=args.batch_size, verbose=1)
    preds_bin = binarize(preds, args.threshold)
    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s → {dt / X.shape[0]:.3f}s per slice")

    out_npy = os.path.join(args.output_dir, "preds_binary.npy")
    print(f"Saving binary preds to {out_npy}")
    np.save(out_npy, preds_bin)

    if Y_true is not None:
        print("\nComputing metrics against ground truth…")
        mets = compute_metrics(Y_true, preds_bin)
        for m,v in mets.items():
            print(f"  {m:10s}: {v:.4f}")

        import pandas as pd
        df = pd.DataFrame.from_dict(mets, orient="index", columns=["Score"])
        df.index.name = "Metric"
        df.to_csv(os.path.join(args.output_dir, "inference_metrics.csv"))

    if Y_true is not None:
        print("Plotting a few examples…")
        plot_examples(X, Y_true, preds_bin, outdir=os.path.join(args.output_dir, "examples"))

    print("\nAll done. Outputs in:", args.output_dir)

if __name__ == "__main__":
    main()
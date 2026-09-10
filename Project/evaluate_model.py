import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import unet_model
X_val = np.load("Data/Preprocessed/Images/images_preprocessed.npy")
Y_val = np.load("Data/Preprocessed/Masks/masks_preprocessed.npy")

model = tf.keras.models.load_model("Models/unet_model.h5")

from evaluation import hd95, lesionwise_dsc, lesionwise_nsd, lesionwise_sensitivity, lesionwise_precision

sample_idx = 0  
sample_input = np.expand_dims(X_val[sample_idx], axis=0)
pred_mask = model.predict(sample_input)[0, :, :, 0]

gt_mask = Y_val[sample_idx, :, :, 0]

hd95_val = hd95(gt_mask, pred_mask)
dsc_val = lesionwise_dsc(gt_mask, pred_mask)
nsd_val = lesionwise_nsd(gt_mask, pred_mask)
sensitivity_val = lesionwise_sensitivity(gt_mask, pred_mask)
precision_val = lesionwise_precision(gt_mask, pred_mask)

print("Sample Metrics:")
print("HD95:", hd95_val)
print("Lesionwise DSC:", dsc_val)
print("Lesionwise NSD:", nsd_val)
print("Lesionwise Sensitivity:", sensitivity_val)
print("Lesionwise Precision:", precision_val)

iou_list, dice_list = [], [] 
metric_list = []
for i in range(len(X_val)):
    sample_input = np.expand_dims(X_val[i], axis=0)
    pred_mask = model.predict(sample_input)[0, :, :, 0]
    gt_mask = Y_val[i, :, :, 0]
    metrics = {
        "HD95": hd95(gt_mask, pred_mask),
        "DSC": lesionwise_dsc(gt_mask, pred_mask),
        "NSD": lesionwise_nsd(gt_mask, pred_mask),
        "Sensitivity": lesionwise_sensitivity(gt_mask, pred_mask),
        "Precision": lesionwise_precision(gt_mask, pred_mask)
    }
    metric_list.append(metrics)

avg_metrics = {key: np.mean([m[key] for m in metric_list]) for key in metric_list[0]}
print("Average Validation Metrics:", avg_metrics)

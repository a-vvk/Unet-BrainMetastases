# Brain Metastases Segmentation (BraTS-MET)

A lightweight 2D U-Net for automated segmentation of brain metastases in multi-modal MRI scans, built for Uni. Designed as a low-compute, accessible alternative to heavier 3D segmentation architectures like nnU-Net, for use in resource-constrained clinical settings.

# Overview

Brain metastases are typically assessed manually using the RANO-BM guidelines — measuring a single tumour diameter per scan. This is fast but ignores tumour shape and volume entirely, and doesn't scale well when a patient has multiple metastases that need to be tracked over time. 

This project automates that process with a slice-wise 2D U-Net that segments tumours directly from MRI volumes, producing a full binary segmentation mask rather than a single measurement.

# Dataset

Trained and evaluated on subsets of the BraTS-MET (Brain Tumour Segmentation — Metastases) dataset, using four MRI modalities per patient:
* T1c (T1-weighted, contrast-enhanced)
* T1n (T1-weighted, native)
* T2-FLAIR
* T2w (T2-weighted)

Each patient volume is a consistent 240×240×155 voxels, with a corresponding binary segmentation mask.

# Pipeline

Project \
├── data_preprocessing.py  &emsp; &emsp; &emsp; # NIfTI loading, normalisation, slicing, modality fusion \
├── model.py    &emsp; &emsp; &emsp;   &emsp; &emsp; &emsp; &emsp; &emsp;          # Lightweight 2D U-Net architecture \
├── train.py   &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &ensp; &emsp;             # Training loop (Adam, binary cross-entropy) \
├── evaluation.py   &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &ensp;         # Dice, IoU, precision, recall \
├── evaluate_model.py &emsp; &emsp; &emsp;  &emsp; &nbsp;   # Extended lesion-wise evaluation (HD95, NSD, sensitivity) \
├── inference.py    &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &ensp;      # CLI inference on new patient scans, with metric export \
└── generators.py    &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;      # Data generators for training/validation

Raw .nii.gz volumes are converted to .npy, sliced along the axial plane, normalised, and fused across the four modalities before being fed into the model.

# Model

A compact 2D U-Net (3 down-sampling / 3 up-sampling blocks, 16→128 filter progression, batch normalisation throughout), trained on 128×128 slices. Deliberately kept small enough to train on consumer-grade GPUs.

# Results

Averaged across validation subsets of 5–20 patients:
* Metric	Score
* Dice	0.95 – 0.96
* IoU	0.90 – 0.92
* Precision	0.93 – 0.95
* Recall	0.95 – 0.97

Benchmarked against published nnU-Net and CNN segmentation results from literature (Isensee et al., 2021; Song et al., 2022).



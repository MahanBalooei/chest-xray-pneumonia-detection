# Chest X-Ray Pneumonia Detection — Transfer Learning Pipeline

> Binary pneumonia classifier trained on chest X-rays using ResNet50, EfficientNetB0, and DenseNet-121 XRV. Includes class-imbalance handling, three decision-threshold strategies, ensemble, TTA, and GradCAM. **Best model: DenseNet-121 XRV (max-MCC strategy).**

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [EDA & Preprocessing](#eda--preprocessing)
  - [Models & Fine-Tuning Strategy](#models--fine-tuning-strategy)
  - [Threshold Selection](#threshold-selection)
  - [Supplementary Analyses](#supplementary-analyses)
- [Results](#results)
- [Limitations](#limitations)
- [Getting Started](#getting-started)
- [Requirements](#requirements)

---

## Overview

Pneumonia is a leading cause of mortality worldwide. Chest X-rays are the standard diagnostic tool, but interpretation requires trained radiologists and is subject to inter-reader variability. This project builds and compares automated classifiers that could serve as a screening aid—flagging suspicious images for priority review.

Three pretrained CNN backbones are compared under identical experimental conditions:

| Model | Pretraining | Input |
|---|---|---|
| ResNet50 | ImageNet | RGB 224×224 |
| EfficientNetB0 | ImageNet | RGB 224×224 |
| DenseNet-121 XRV | Chest X-ray datasets (CheXpert, MIMIC-CXR, NIH ChestX-ray14) | Grayscale 224×224 |

All models are trained on the same stratified data split, evaluated on the same untouched test set, and assessed under three decision-threshold strategies to make the FP/FN trade-off explicit.

---

## Dataset

**Chest X-Ray Images (Pneumonia)** — Kermany et al. (2018), available on [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

| Split | NORMAL | PNEUMONIA | Total | Imbalance ratio |
|---|---|---|---|---|
| Train | ~1,341 | ~3,875 | ~5,216 | ~2.9× |
| Val (official) | 8 | 8 | 16 | 1.0× |
| Test | 234 | 390 | 624 | 1.7× |

> **Note:** The official 16-image validation folder is too small for reliable model selection. A stratified 80/20 split is carved from the training folder instead. The official `val/` directory is never used for model decisions.

---

## Project Structure

```
chest-xray-pneumonia-detection/
├── chest_xray_pneumonia_detection.ipynb   # Full pipeline notebook
├── LICENSE
└── README.md
```

Model checkpoints (`resnet50_best.pth`, `efficientnet_b0_best.pth`, `densenet121_xrv_best.pth`) are saved locally during training and are not tracked in the repository.

---

## Methodology

### EDA & Preprocessing

Before training, exploratory analysis was performed to directly inform pipeline decisions:

- **Class imbalance (~74.3% PNEUMONIA in training)** → `WeightedRandomSampler` during training; per-class precision/recall/F1/MCC reported; three threshold strategies evaluated
- **Tiny official validation set (16 images)** → stratified 80/20 split from training folder; test set held out until final evaluation
- **Variable image sizes and aspect ratios** → resize to 256 then center-crop to 224×224
- **Near-identical class-mean intensities** → rules out simple intensity thresholds; confirms need for learned features
- **Subtle visual differences between classes** → transfer learning preferred over training from scratch; GradCAM as qualitative sanity check

Two preprocessing pipelines are used:

- **ImageNet pipeline** (ResNet50, EfficientNetB0): grayscale → 3-channel replication → ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`); light augmentation during training (horizontal flip, ±10° rotation, brightness jitter)
- **XRV pipeline** (DenseNet-121 XRV): grayscale → `XRayCenterCrop` + `XRayResizer` → XRV intensity normalization (`[-1024, 1024]`); same augmentation

### Models & Fine-Tuning Strategy

All three models use a **two-phase fine-tuning** approach:

| Phase | Layers trained | LR |
|---|---|---|
| Phase 1 | Classification head only | 1e-3 |
| Phase 2 (ResNet/EffNet) | Entire backbone | 1e-4 |
| Phase 2 (DenseNet XRV) | Head + `denseblock4` + `norm5` only | 1e-5 |

DenseNet-121 XRV uses **partial fine-tuning** in Phase 2 — earlier blocks stay frozen to preserve domain-specific radiological representations from XRV pretraining.

Training details:
- **Loss:** `CrossEntropyLoss` with label smoothing ε=0.1 (ResNet/EffNet); `BCEWithLogitsLoss` (DenseNet)
- **Optimizer:** AdamW with weight_decay=1e-4
- **Scheduler:** CosineAnnealingLR per phase
- **Early stopping:** validation MCC (ResNet/EffNet), validation AUROC (DenseNet); patience=5

### Threshold Selection

The default 0.5 threshold is not used. Three strategies are evaluated **on the validation set only**, then applied once to the test set:

| Strategy | Criterion | Use case |
|---|---|---|
| A | Maximize PNEUMONIA F1-score | Balanced positive-class performance |
| B | PNEUMONIA recall ≥ 0.97, then max MCC | Screening — minimize missed PNEUMONIA |
| C | Maximize MCC directly | Most balanced overall performance |

### Supplementary Analyses

- **False negative / false positive inspection:** most-confident misclassified images are visualized to surface failure modes
- **Ensemble:** equal-weight probability averaging of ResNet50 + DenseNet-121 XRV; threshold re-selected on averaged validation probabilities
- **Test-Time Augmentation (TTA):** 5-view averaging for ResNet50 (flip, rotation, brightness, combined); compared against standard inference at Strategy A threshold
- **GradCAM:** gradient-weighted class activation maps on `layer4[-1]` (ResNet50), `features[-1]` (EfficientNetB0), and `denseblock4` (DenseNet-121 XRV) as a spatial sanity check

---

## Results

### Key metrics on the test set (Strategy C — Max MCC)

| Model | Accuracy | AUROC | MCC | Recall-PNEUMONIA | Recall-NORMAL | FN | FP |
|---|---|---|---|---|---|---|---|
| ResNet50 | — | — | — | — | — | — | — |
| EfficientNetB0 | — | — | — | — | — | — | — |
| **DenseNet-121 XRV** | **best** | **best** | **best** | high | **best** | low | **lowest** |
| Ensemble (R50+DN) | — | highest | lower | — | — | — | higher |

> Run the notebook to populate exact numbers — values depend on your training run and random seed.

### Conclusions

- **DenseNet-121 XRV + Strategy C** is the recommended model: highest MCC, best NORMAL recall, lowest false-positive count, while maintaining high PNEUMONIA recall. Domain-specific XRV pretraining appears to provide a more suitable feature initialization for this task than ImageNet pretraining.
- **Strategy B** is the preferred operating point when minimizing missed PNEUMONIA-labeled cases is the priority; it guarantees PNEUMONIA recall ≥ 0.97 on the validation set at the cost of more false positives.
- **Ensemble** achieves the highest AUROC and AUPRC, confirming that ResNet50 and DenseNet capture complementary information, but does not outperform DenseNet-121 XRV in thresholded classification due to uncalibrated equal-weight averaging.
- **TTA** improved AUROC/AUPRC for ResNet50 but did not substantially improve thresholded metrics, since the threshold was not re-tuned for TTA-transformed probabilities.

---

## Limitations

- **Single data source:** results come from one pediatric hospital dataset. Generalization to different scanners, patient populations, or acquisition protocols is unknown.
- **Train/test class-proportion mismatch:** PNEUMONIA makes up ~74.3% of the training/validation pool but only 62.5% of the test set. Thresholds may be slightly biased toward over-predicting PNEUMONIA at test time.
- **No external validation:** the model was not evaluated on an independent dataset from a different institution.
- **No radiologist review:** failure-case images and GradCAM heatmaps were not reviewed by a trained radiologist. Visual interpretations are hypotheses, not clinical findings.
- **Threshold choice is context-dependent:** the optimal FN/FP trade-off depends on downstream care pathways and patient risk — factors a single validation metric cannot capture.
- **Pneumonia subtype not modeled:** bacterial and viral pneumonia have different imaging presentations and treatments. The dataset does not distinguish between them.
- **Single random seed (42):** performance was not evaluated over multiple seeds or cross-validation splits.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/MahanBalooei/chest-xray-pneumonia-detection.git
cd chest-xray-pneumonia-detection
```

### 2. Install dependencies

```bash
pip install torch torchvision torchxrayvision scikit-learn matplotlib seaborn pandas numpy Pillow kagglehub opencv-python
```

### 3. Run the notebook

```bash
jupyter notebook chest_xray_pneumonia_detection.ipynb
```

The notebook will automatically download the dataset from Kaggle using `kagglehub`. Make sure your Kaggle API credentials are configured (`~/.kaggle/kaggle.json`).

---

## Requirements

| Package | Purpose |
|---|---|
| `torch` / `torchvision` | Model training and transforms |
| `torchxrayvision` | DenseNet-121 pretrained on chest X-rays |
| `scikit-learn` | Metrics (MCC, AUROC, AUPRC, PR curves) |
| `matplotlib` / `seaborn` | Visualizations |
| `pandas` / `numpy` | Data handling |
| `Pillow` | Image loading |
| `kagglehub` | Automatic dataset download |
| `opencv-python` | GradCAM heatmap resizing |

---

## Citation

If you use the dataset, please cite:

> Kermany, D. S., Goldbaum, M., Cai, W., et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. Cell, 172(5), 1122–1131.

TorchXRayVision:

> Cohen, J. P., Viviano, J. D., Bertrand, P., et al. (2022). *TorchXRayVision: A library of chest X-ray datasets and models*. Medical Imaging with Deep Learning.

---
## Contact

For questions about the technical implementation, contact:

- **Mahan Balooei** — mahan.balooei@studio.unibo.it
---
## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

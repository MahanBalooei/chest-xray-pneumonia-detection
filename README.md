# Pneumonia Detection on Chest X-Rays with Transfer Learning

A course project in which I compare three pretrained CNNs for telling pneumonia from normal chest X-rays, pick decision thresholds on a validation split, and check what happens when those thresholds meet the official test set. The whole pipeline is a single notebook that runs on Google Colab.

The short version: all three models look excellent on the validation split and noticeably worse on the test set, and the gap is the most useful thing this project shows.

## Results on the test set (624 images: 390 pneumonia, 234 normal)

Thresholds were chosen on the validation split to maximise MCC (strategy C below) and then applied once to the test set.

| Model | Pretraining | Threshold | Accuracy | AUROC | AUPRC | MCC | Recall, pneumonia | Recall, normal | FN | FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | ImageNet | 0.511 | 86.4% | 0.943 | 0.934 | 0.722 | 0.997 | 0.641 | 1 | 84 |
| EfficientNetB0 | ImageNet | 0.263 | 83.0% | 0.927 | 0.935 | 0.654 | 0.997 | 0.551 | 1 | 105 |
| DenseNet-121 (TorchXRayVision) | Chest X-rays | 0.367 | **87.5%** | 0.954 | **0.967** | **0.739** | 0.985 | **0.692** | 6 | **72** |
| Ensemble, ResNet50 + DenseNet-121 | | 0.303 | 84.5% | **0.964** | 0.964 | 0.683 | 0.997 | 0.590 | 1 | 96 |

FN: pneumonia images predicted normal. FP: normal images predicted pneumonia. The figures come from one training run with seed 42.

### What the numbers say

- **Every model misses almost no pneumonia but flags many normal images.** Pneumonia recall is 0.98 to 1.00 everywhere, while normal recall sits between 0.55 and 0.69. In practice the models lean towards "pneumonia" whenever they are unsure.
- **DenseNet-121 pretrained on chest X-rays is the best single model.** It has the highest MCC (0.739), the highest AUPRC and the fewest false positives (72), at the cost of 6 missed pneumonia cases instead of 1.
- **The ensemble ranks images best but classifies them worse.** Averaging ResNet50 and DenseNet-121 gives the highest AUROC (0.964), but at its chosen threshold its MCC (0.683) is below both of its members'. Its threshold, picked on the validation split, sits too low for the test set.
- **Test-time augmentation helps ranking, not decisions.** For ResNet50, averaging five augmented views raised AUROC from 0.943 to 0.959 and AUPRC from 0.934 to 0.952, but MCC stayed essentially flat (0.712 vs 0.709 at the strategy A threshold), because the threshold was not re-tuned for the averaged probabilities.

### The validation/test gap

| Model | MCC on validation split | MCC on test set |
|---|---:|---:|
| ResNet50 | 0.975 | 0.722 |
| EfficientNetB0 | 0.975 | 0.654 |
| DenseNet-121 (TorchXRayVision) | 0.901 | 0.739 |
| Ensemble | 0.962 | 0.683 |

The validation split comes from the official training folder, and the test set is a separate folder. Two things probably explain most of the drop:

1. **The split is by image, not by patient.** The training folder contains several images from the same patient, so the same person can appear in both my training and validation subsets. That makes the validation scores optimistic, and the thresholds tuned on them fit the training patients rather than new ones.
2. **The test set has a different class balance and is known to differ from the training folder.** Pneumonia is 74% of the training and validation images but 62.5% of the test images.

It is telling that the DenseNet pretrained on chest X-rays scores lowest on validation but best on test. The two ImageNet models appear to have fitted the training distribution more tightly, and that did not carry over.

## Data

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), Kermany et al., Cell 2018. Paediatric chest X-rays from a single hospital.

| Folder | Normal | Pneumonia | Total |
|---|---:|---:|---:|
| train | 1,341 | 3,875 | 5,216 |
| val (official) | 8 | 8 | 16 |
| test | 234 | 390 | 624 |

The official validation folder has only 16 images, so I did not use it. Instead I split the training folder 80/20, stratified by class: 4,173 images for training and 1,043 for validation. The test folder was used only for the final evaluation.

## Method

**Models.** ResNet50 and EfficientNetB0 with ImageNet weights, and DenseNet-121 from TorchXRayVision, which was pretrained on several large chest X-ray collections (NIH, CheXpert, MIMIC-CXR and others).

**Preprocessing.** For the ImageNet models: grayscale images copied to three channels, resized to 256 and centre-cropped to 224, ImageNet normalisation. For DenseNet-121: TorchXRayVision's own crop, resize and intensity scaling to [-1024, 1024], single channel. Training augmentation for all: horizontal flip, rotation up to 10° and brightness jitter.

**Training.** Two phases: first the classification head alone (lr 1e-3), then fine-tuning. ResNet50 and EfficientNetB0 were fine-tuned end to end (lr 1e-4). For DenseNet-121 only the last dense block and final normalisation layer were unfrozen (lr 1e-5), to keep the chest X-ray features it already has. AdamW with weight decay 1e-4 and cosine learning-rate decay in each phase. A weighted sampler balanced the classes during training.

The models were not trained in exactly the same way. The ImageNet models used cross-entropy with label smoothing (0.1) and kept the epoch with the best validation MCC. DenseNet-121 used binary cross-entropy and kept the epoch with the best validation AUROC. So the comparison is fair in data and evaluation, but not a controlled comparison of pretraining alone.

**Thresholds.** Instead of the default 0.5, three thresholds were chosen on the validation split for each model:

- A: maximise F1 for pneumonia
- B: keep pneumonia recall at or above 0.97, then maximise MCC
- C: maximise MCC

For most models B and C landed on the same threshold. All three are reported in the notebook; the table above uses C. For DenseNet-121, strategy B gives 5 false negatives and 77 false positives (MCC 0.727).

**Other analyses.** The most confident false negatives and false positives are shown for each model. Grad-CAM maps (ResNet50 `layer4`, EfficientNetB0 last feature block, DenseNet-121 `denseblock4`) were used as a rough check that the models look at the lungs. Nobody with radiology training reviewed these maps, so they are a sanity check and nothing more.

## Limitations

- Image-level rather than patient-level split, as discussed above. A patient-level split would give more honest validation scores and probably better thresholds.
- One random seed and one split, so no confidence intervals. Differences of a few points in MCC between models should not be over-read.
- Single hospital, paediatric patients only, no external test set.
- The labels do not separate bacterial from viral pneumonia.
- This is a study of methods on a public dataset, not a tool fit for clinical use.

## Running it

The notebook is written for Google Colab with a GPU. It installs TorchXRayVision itself and downloads the dataset with `kagglehub`, so you need Kaggle credentials (`~/.kaggle/kaggle.json`) if you run it outside Colab.

```bash
git clone https://github.com/MahanBalooei/chest-xray-pneumonia-detection.git
cd chest-xray-pneumonia-detection
pip install torch torchvision torchxrayvision scikit-learn matplotlib seaborn pandas numpy Pillow kagglehub opencv-python
jupyter notebook chest_xray_pneumonia_detection.ipynb
```

The committed notebook includes the outputs of the run reported above. Model checkpoints are saved during training but are not included in the repository.

## References

- Kermany D.S. et al. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell* 172(5), 1122–1131.
- Cohen J.P. et al. (2022). TorchXRayVision: a library of chest X-ray datasets and models. *Medical Imaging with Deep Learning*.

## Contact

Mahan Balooei, mahan.balooei@studio.unibo.it

## License

MIT. See `LICENSE`.

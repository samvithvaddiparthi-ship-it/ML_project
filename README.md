# Pneumonia Diagnosis, Detection and Localization on Chest X-rays

**UE24CS352A Machine Learning · Mini Project** · PES University
**Team:** Samvith Vaddiparthi (PES1UG24AM468) · Sanjit C (PES1UG24AM448)
**Problem statement:** *Pneumonia Diagnosis Detection and Localization* (Stanford CS229 list, #34)

Given a chest X-ray, we (1) **diagnose** whether it shows pneumonia and (2) **localize** the lung
opacity with bounding boxes. We re-implement the reference project
([Huang, Cortes & Monam, *Weakly Supervised Pneumonia Localization*, CS229 2018](https://cs229.stanford.edu/proj2018/report/56.pdf))
and extend it with transfer learning, a supervised detector, validation-tuned thresholds and the official
RSNA competition metric.

| Step | Reference report | This project |
|---|---|---|
| Baselines | LR, SVM, RF on flattened pixels | same (64×64 pixels, PCA for LR/SVM) |
| Classifier | 10-layer CNN from scratch, 128 px | same CNN **+** ImageNet-pretrained **DenseNet-121** (CheXNet backbone), 384 px, augmentation |
| Weak localization | CAM + DFS clustering, fixed cut-off | CAM + connected components, **cut-off tuned on validation** |
| Supervised localization | R-CNN baseline | **Faster R-CNN ResNet-50-FPN** (COCO-pretrained) **+ classifier gating** |
| Metrics | accuracy, mean IoU | accuracy, precision, recall, specificity, F1, ROC-AUC, mean IoU, **RSNA mAP@[0.4:0.75]** |
| Data setting | Normal vs Pneumonia only | that **and** the full, harder 3-class dataset (pneumonia vs everything) |

<!-- RESULTS:START -->
## Results

Results are written to `outputs/results.json` by the pipeline; run `python scripts/update_readme_results.py`
after training to fill this section with the numbers from your run.
<!-- RESULTS:END -->

---

## Repository layout

```
├── src/pneumo/
│   ├── config.py        all hyper-parameters (full / --fast / --smoke presets)
│   ├── data.py          RSNA labels -> one row per image, patient split, DICOM cache, datasets, augmentation
│   ├── baselines.py     Logistic Regression / SVM / Random Forest on pixels
│   ├── models.py        PaperCNN (10 conv), DenseNet-121 + GAP/FC head (CAM-ready), Faster R-CNN
│   ├── train.py         training loops (AMP, cosine LR, best-val-AUC checkpointing), detector loop
│   ├── cam.py           Class Activation Map -> heat-map -> connected blobs -> boxes
│   ├── metrics.py       classification metrics, union-mask IoU, official RSNA mAP
│   ├── pipeline.py      the experiment stages (prepare, baselines, cnn, detector, localize, demo)
│   ├── viz.py           every figure in the write-up / slides
│   └── infer.py         single-image inference used by the demo and CLI
├── scripts/
│   ├── run_all.py               run all (or selected) stages
│   ├── make_synthetic_data.py   fake RSNA-format data for the smoke test
│   ├── build_kaggle_notebook.py regenerates notebooks/kaggle_pipeline.ipynb from the source
│   └── update_readme_results.py writes the results tables into this README
├── notebooks/kaggle_pipeline.ipynb   one-click Kaggle run (GPU) - recommended way to train
├── app.py               Gradio live demo
├── predict.py           command-line prediction
├── tests/test_metrics.py
├── outputs/             results.json + figures (weights and predictions are git-ignored)
└── report/              write-up source
```

## Setup

```bash
git clone <this repo> && cd pneumonia-detection-localization
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                              # 6 unit tests for the metrics / CAM code
```

## Data

[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) (Kaggle):
26,684 frontal chest X-rays (1024×1024 DICOM) with three labels - *Normal*, *No Lung Opacity / Not Normal*,
*Lung Opacity* (pneumonia) - and 9,555 bounding boxes on the pneumonia images. Join the competition, then:

```bash
kaggle competitions download -c rsna-pneumonia-detection-challenge -p data/
unzip -q data/rsna-pneumonia-detection-challenge.zip -d data/rsna
```

The images are **not** in this repository (size and data-use terms).

## How to run

### Option A - Kaggle notebook (recommended, free GPU, no download)
1. Upload `notebooks/kaggle_pipeline.ipynb` to Kaggle (*Create → New Notebook → File → Import Notebook*).
2. *Add Input* → **RSNA Pneumonia Detection Challenge**; *Settings* → **GPU T4 x1**, **Internet on**.
3. *Save Version → Save & Run All*. The full configuration takes about 2-3 h (`MODE = "--fast"`: ~45 min).
4. From the *Output* tab download `results_light.zip` (results + figures) and `model_weights.zip`.
   Unzip both into `outputs/` of this repo.

### Option B - any machine with a CUDA GPU
```bash
python scripts/run_all.py --data-dir data/rsna --cache-dir cache --out-dir outputs        # full
python scripts/run_all.py --data-dir data/rsna --cache-dir cache --out-dir outputs --fast # quicker
python scripts/run_all.py ... --stages localize,figures                                   # re-run parts
```
Stages: `prepare, baselines, cnn, detector, localize, figures, demo`.

### Option C - 3-minute CPU smoke test (no data needed)
```bash
python scripts/run_all.py --smoke      # generates synthetic RSNA-format DICOMs and runs every stage
```

### Write-up
```bash
python report/make_writeup.py      # builds report/writeup.pdf from outputs/results.json (needs pdflatex)
python scripts/update_readme_results.py   # fills the Results section above
```

### Live demo
```bash
python app.py                 # opens http://127.0.0.1:7860  (needs outputs/models/*.pt)
python app.py --share         # public link, e.g. from Kaggle / Colab
python predict.py some_xray.dcm another.png       # CLI; saves visualisations to predictions/
```
The demo shows p(pneumonia), the CAM heat-map with weakly-supervised boxes (blue), Faster R-CNN boxes (green)
and, for the bundled test examples, the ground truth (red).

## Method

1. **Data preparation.** The label CSV has one row per box; we collapse it to one row per image (target, class,
   list of boxes) and split *by patient*, stratified by the 3-way class, 70/15/15 (seed 42). Every DICOM is decoded
   once (MONOCHROME1 inverted, min-max scaled to uint8), resized to 512×512 and stored in a memory-mapped array.
2. **Settings.** *paper*: Normal vs Lung Opacity, as in the reference report. *full*: Lung Opacity vs everything
   else, including the 'not normal' lungs that also look cloudy - the realistic task.
3. **Baselines.** Images flattened at 64×64 (4,096 features). LR and RBF-SVM on 150 PCA components of the
   standardised pixels; Random Forest (300 trees) on raw pixels.
4. **Paper-style CNN.** 10 conv layers (3×3, zero padding, BatchNorm, ReLU, three 2×2 max-pools) → GAP → one FC.
   128×128 input, Adam(W) lr 1e-4, 20 epochs.
5. **DenseNet-121.** ImageNet-pretrained, GAP + Linear(1024→1) head, 384×384 input, random affine
   (±10°, ±10% scale, ±6% shift), brightness/contrast jitter and horizontal flips; AdamW 3e-4, cosine schedule,
   10 epochs, mixed precision. Class imbalance handled with `pos_weight` in the BCE loss. The epoch with the
   best validation ROC-AUC is kept, and the decision threshold is the one maximising validation F1.
6. **Weakly-supervised localization (CAM).** M(x,y) = Σ_k w_k f_k(x,y) from the last conv map and FC weights;
   ReLU → bilinear up-sampling → divide by the map maximum → threshold → connected components (the same clusters
   as the report's DFS) → one box per blob (blobs < 1% of the image dropped, ≤ 3 boxes). The threshold is chosen
   on the validation set. No box labels are used for training.
7. **Supervised localization.** Faster R-CNN ResNet-50-FPN (COCO-pretrained), 512×512, every epoch uses all
   pneumonia images plus an equal random sample of negatives; SGD lr 0.01, warm-up + cosine, 6 epochs.
   **Gating:** boxes are kept only if DenseNet's p(pneumonia) exceeds a validation-tuned gate, which removes
   false positives on healthy lungs.
8. **Evaluation (test split only).** Classification: accuracy, precision, recall (sensitivity), specificity,
   F1, ROC-AUC. Localization: (a) the report's **mean IoU**: IoU between the union of predicted and ground-truth
   boxes on every pneumonia image, 0 if nothing is predicted; *oracle* = localizing every pneumonia image
   regardless of the classifier (the report's 0.379 analysis); (b) the **official RSNA metric**: mean over
   images of the average of TP/(TP+FP+FN) at IoU thresholds 0.40-0.75.

### Deviations from the reference report (and why)
- No U-Net lung-segmentation channel: RSNA has no lung masks and the report found it helps little and hurts on
  severe cases; we use pretrained features instead.
- BatchNorm and max-pooling in the paper CNN (the report doesn't specify down-sampling; without it CAM would be
  computed at full resolution with a huge memory cost).
- Stage-2 labels (26,684 images) instead of stage-1, and a 70/15/15 split so thresholds can be tuned on a
  separate validation set without touching the test set.

## Team contributions

| Member | Work |
|---|---|
| Samvith Vaddiparthi (PES1UG24AM468) | data pipeline (label table, patient-level split, DICOM caching), DenseNet-121 transfer learning, CAM localization and threshold tuning, metrics (union IoU, RSNA mAP) and tests, Gradio demo |
| Sanjit C (PES1UG24AM448) | classical baselines, paper-CNN replication, Faster R-CNN detector and classifier gating, figures and result analysis, write-up and slides |

## References
1. S.-C. Huang, E. Cortes, M. Monam. *Weakly Supervised Pneumonia Localization.* Stanford CS229, 2018.
2. B. Zhou et al. *Learning Deep Features for Discriminative Localization (CAM).* CVPR 2016.
3. P. Rajpurkar et al. *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays.* arXiv:1711.05225, 2017.
4. G. Huang et al. *Densely Connected Convolutional Networks.* CVPR 2017.
5. S. Ren et al. *Faster R-CNN.* NeurIPS 2015; T.-Y. Lin et al. *Feature Pyramid Networks.* CVPR 2017.
6. RSNA Pneumonia Detection Challenge, Kaggle, 2018.

> Educational project. Not a medical device and not for clinical use.

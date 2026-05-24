# UNet Satellite Image Segmentation

Multi-class semantic segmentation of satellite imagery using a UNet architecture implemented from scratch in TensorFlow/Keras. Trained on aerial imagery of Dubai to classify each pixel into one of 6 land-cover categories.

---

## Problem

Satellite imagery produces high-resolution aerial photographs where every pixel belongs to a real-world surface category — road, building, vegetation, water, and so on. Manually labelling these at scale is infeasible. The goal of this project is to train a model that can automatically produce dense pixel-level land-cover maps from raw satellite images.

This is a **semantic segmentation** problem: unlike classification (one label per image) or detection (bounding boxes), segmentation requires a class prediction for every single pixel in a 256×256 patch.

---

## Dataset

**Semantic Segmentation of Aerial Imagery** — [Kaggle](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)

- 72 satellite images of Dubai across 8 map tiles
- Each image paired with an RGB mask where colour encodes land-cover class
- 6 classes: Building, Land, Road, Vegetation, Water, Unlabeled

| Class | Colour | Label ID |
|---|---|---|
| Water | `#E2A929` | 0 |
| Land | `#8429F6` | 1 |
| Road | `#6EC1E4` | 2 |
| Building | `#3C1098` | 3 |
| Vegetation | `#FEDD3A` | 4 |
| Unlabeled | `#9B9B9B` | 5 |

---

## Approach

### 1. Image Patching
Raw satellite tiles are large (typically 800×600+). Rather than resizing and losing spatial detail, each image is cropped to the nearest multiple of 256 and divided into non-overlapping **256×256 patches** using `patchify`. This produces 448 image-mask patch pairs for training.

### 2. Preprocessing
- Satellite images: MinMax normalised to [0, 1] per channel
- RGB masks: converted to integer label maps (one integer 0–5 per pixel) via colour matching, then one-hot encoded for categorical training

### 3. Architecture — UNet
UNet uses a symmetric encoder-decoder structure with **skip connections** that concatenate encoder feature maps directly into the decoder. This preserves fine spatial detail that would otherwise be lost through downsampling — critical for pixel-accurate segmentation.

```
Input (256×256×3)
    ↓
[Conv(16) → Conv(16)] → MaxPool         skip → concat
    ↓
[Conv(32) → Conv(32)] → MaxPool         skip → concat
    ↓
[Conv(64) → Conv(64)] → MaxPool         skip → concat
    ↓
[Conv(128) → Conv(128)] → MaxPool       skip → concat
    ↓
[Conv(256) → Conv(256)]   ← Bottleneck
    ↓
TransposeConv(128) + skip → [Conv(128) → Conv(128)]
    ↓
TransposeConv(64)  + skip → [Conv(64)  → Conv(64)]
    ↓
TransposeConv(32)  + skip → [Conv(32)  → Conv(32)]
    ↓
TransposeConv(16)  + skip → [Conv(16)  → Conv(16)]
    ↓
Conv(1×1) → Softmax over 6 classes
```

Total parameters: **1,941,334**  
Dropout(0.2) applied after each conv block.

### 4. Loss Function
Standard categorical cross-entropy performs poorly on imbalanced segmentation datasets. This project uses a **combined Focal + Dice loss**:

- **Focal Loss** — down-weights easy, well-classified pixels so the model focuses on hard regions (e.g. thin roads, small buildings)
- **Dice Loss** — directly optimises the overlap between prediction and ground truth, complementing the pixel-wise focal objective

```python
total_loss = CategoricalFocalLoss() + DiceLoss(class_weights=[1/6]*6)
```

### 5. Metric
**Jaccard Coefficient (IoU)** — measures the overlap between predicted and ground truth masks. Standard metric for semantic segmentation.

```
IoU = |A ∩ B| / |A ∪ B|
```

### 6. Experiment Tracking
Training metrics logged with **Weights & Biases (WandB)** — loss curves, IoU per epoch, and model config tracked across runs.

---

## Results

| Epoch | Train Loss | Val Loss | Train IoU | Val IoU |
|---|---|---|---|---|
| 1 | 0.824 | 0.761 | 0.182 | 0.210 |
| 2 | 0.710 | 0.689 | 0.254 | 0.288 |
| 3 | 0.621 | 0.630 | 0.319 | 0.334 |
| 4 | 0.559 | 0.593 | 0.371 | 0.370 |
| 5 | 0.510 | 0.572 | 0.409 | 0.399 |

Val IoU of **0.40 at 5 epochs** — model is learning consistently. 30–50 epochs with a learning rate schedule would push this to 0.65+.

---

## Activation Map Analysis

Intermediate encoder activations were visualised using **keract** to inspect what the model learns at different depths:
- Shallow layers: edge and texture detection
- Mid layers: shape and boundary features
- Deep layers: semantic region groupings (vegetation clusters, building footprints)

---

## Limitations & Next Steps

- **Underfit at 5 epochs** — longer training with cosine LR annealing would significantly improve IoU
- **No data augmentation** — horizontal/vertical flips and random rotations would improve generalisation
- **Class imbalance** — road and unlabeled pixels are rare; per-class focal weights tuned to class frequency would help
- **Potential extensions:** EfficientNet/ResNet encoder (transfer learning), per-class IoU breakdown, TFLite export for edge deployment

---

## Stack

- Python, TensorFlow 2.13, Keras
- patchify, segmentation-models, keract
- scikit-learn, OpenCV, Pillow
- Weights & Biases (WandB)
- Google Colab (T4 GPU)

---

## Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)
2. Upload to Google Drive
3. Open `satellite_segmentation_clean.ipynb` in Google Colab
4. Update `DATASET_ROOT` in Section 3 to your Drive path
5. Run all cells top to bottom

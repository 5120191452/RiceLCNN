# Real-Time Rice Seed Segmentation, Classification, and Phenotyping
**YOLOv11-LA + RiceLCNN + Integrated Tracking & Morphometry**

> An end-to-end pipeline for high-throughput rice seed analysis: real-time instance detection/segmentation, single-grain classification, and sub-pixel phenotypic measurement—optimized for both workstation GPUs and Jetson Orin edge devices.

---

## 1) Overview

- **Goal.** Real-time, accurate, and scalable analysis of rice seed images for breeding, quality control, and field-to-factory digital agriculture.
- **Pipeline.** `YOLOv11-LA (detection/segmentation) → DeepSORT (ID tracking) → RiceLCNN (classification) → Sub-pixel morphometry (length/width/aspect/roundness)`
- **Highlights.**
  - **Lightweight & fast:** YOLOv11-LA reduces parameters and FLOPs while improving mAP; RiceLCNN attains the best accuracy among compared classifiers **with ~0.53M params**.
  - **Robust phenotyping:** Sub-pixel contour fitting + dynamic scale calibration (mm/px) for key traits (length, width, aspect ratio, roundness).
  - **Edge-deployable:** Stable on **Jetson Orin 8GB**, supporting online grading, anomaly removal, and adaptive control loops.

---

## 2) Dataset Description

- **Source.** Rice Research Institute, Jilin Academy of Agricultural Sciences  
- **Variety.** *Tongjing 612*  
- **Treatments.** Nine field management combinations to induce phenotypic variation (straw incorporation, enzyme application, organic & chemical fertilizers).

| Plot | Straw | Enzymes | Organic Fert. | Chemical Fert. |
|:---:|:-----:|:-------:|:-------------:|:--------------:|
| 1 | × | × | × | × |
| 2 | ✓ | ✓ | × | × |
| 3 | ✓ | × | × | × |
| 4 | ✓ | ✓ | ✓ | ✓ |
| 5 | ✓ | ✓ | × | ✓ |
| 6 | ✓ | ✓ | ✓ | × |
| 7 | ✓ | × | ✓ | × |
| 8 | ✓ | × | × | ✓ |
| 9 | ✓ | × | ✓ | ✓ |

- **Imaging.** Nikon D7100, uniform black background, standardized lighting; **6000×4000 (300 dpi)**; **200 grains per image**.
- **Detection dataset.** 25,000 seed instances (YOLO format), split **8:1:1** (train/val/test).  
- **Classification dataset.** Auto-cropped from detections, cleaned to **16,731** seed images (224×224), split **8:1:1**.
- **Scale calibration.** Black light-absorbing discs placed in-scene provide dynamic mm/px conversion for physical measurements.

> The repo can also integrate public datasets (RiceNet, Japanese Rice, RiceSeedSize) to enhance cross-domain robustness.

---

## 3) Methods

### 3.1 YOLOv11-LA (Lightweight-Attention) — Detection/Segmentation
**Key optimizations (over YOLOv11n):**
- **DWConv** substitution to reduce FLOPs.
- **C3k2 slimming + channel compression** (cap 512) for compactness.
- **CBAM attention** before the head to emphasize small/adhesive grains.
- **Post-detect refinement:** Otsu-based fine segmentation + minimum bounding rectangle (better for tilted elongated grains).

**Performance snapshot (rice seed detection):**
- **mAP@0.5 = 99.50%**, **mAP@0.5:0.95 = 93.06%**, **GFLOPs ≈ 3.1**, **Params < 0.96M**.  
- ~**51.6%** FLOP reduction vs baseline with improved/maintained accuracy.

### 3.2 DeepSORT — Tracking
- Frame-to-frame association with Kalman filtering + cascade matching + IoU gating; reduces ID switches in dense scenes and supports precise counting.

### 3.3 RiceLCNN — Classification
- **Backbone:** six lightweight conv stages (`1×1` bottlenecking + `3×3` local modeling), BN + LeakyReLU.
- **Attention:** mid-level **SE** to recalibrate channels.
- **Head:** global feature (224-d) → FC to `C` classes.
- **Size:** ~**0.53M** parameters with state-of-the-art accuracy/efficiency trade-off.

**Classification results (private dataset):**
- **RiceLCNN Acc = 89.78%**, best among MobileNetV2/V3, Xception, ResNet50, EfficientNetV2, ShuffleNetV2.
- **Vision Transformer (ViT-B/16)** added as a transformer baseline: **Acc = 65.89%**; training speed notably slower (≈240s/epoch), indicating data-scale and inductive-bias limitations in this fine-grained, moderate-scale setting.

**Generalization (public dataset):**
- Overall **Acc = 96.32%** with strong macro-averaged metrics.

### 3.4 Sub-pixel Morphometry
- Otsu → Gaussian smoothing → Sobel → sub-pixel edge fitting.
- Dynamic scale α from calibration disk → **mm-level** traits:
  - `length_mm`, `width_mm`, `aspect_ratio`, `roundness`.
- Empirical measurement error **≤ 0.1 mm** (workstation & edge).

---

## 4) Repository Structure (suggested)
├─ datasets/
│ ├─ detection/ # YOLO txt labels + images
│ └─ classification/ # Cropped 224×224 single-grain images (train/val/test)
├─ yolov11_la/ # Detection model code & configs
├─ ricelcnn/ # Classification model code
├─ tracking/ # DeepSORT utilities
├─ phenotyping/ # Sub-pixel measurement & mm/px scale
├─ deployment/ # Jetson Orin scripts/configs
├─ examples/ # Minimal training/eval/infer scripts
└─ README.md

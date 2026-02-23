# Global Deforestation Driver Classification from Multi-Temporal Satellite Imagery

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.6+-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/PyTorch--Lightning-0.9-blueviolet.svg)](https://www.pytorchlightning.ai/)
[![CI](https://img.shields.io/badge/CI-CircleCI%20%7C%20Travis-brightgreen.svg)](#cicd)

**Stanford AI for Climate Change (AICC) &mdash; Stanford AI Lab**

An end-to-end deep learning system for classifying the dominant drivers of global forest loss from Landsat satellite imagery. The pipeline spans raw satellite data acquisition through multi-modal inference, achieving **80% classification accuracy** across five deforestation driver categories on a global dataset derived from Curtis et al. (2018, *Science*) and the Hansen Global Forest Change product.

---

## Table of Contents

- [Research Context](#research-context)
- [Architecture Overview](#architecture-overview)
- [Data Pipeline](#data-pipeline)
- [Model Zoo](#model-zoo)
- [Training & Evaluation](#training--evaluation)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [CI/CD](#cicd)

---

## Research Context

Tropical deforestation accounts for ~8% of global CO2 emissions, yet the **drivers** behind forest loss&mdash;commodity agriculture, shifting cultivation, forestry, wildfire, urbanization&mdash;vary drastically by geography and require fundamentally different policy interventions. Curtis et al. (2018) produced the first global, spatially explicit map of deforestation drivers, but their methodology relies on manual interpretation and static feature engineering.

This project reformulates the problem as a **multi-temporal image classification task**, leveraging deep learning to automatically classify deforestation drivers from sequences of Landsat satellite imagery. The system ingests raw geospatial data, constructs cloud-free composites, and trains CNN, LSTM, and multimodal fusion architectures to learn spatiotemporal patterns of forest loss across diverse biomes and geographies.

### Classification Taxonomy

| Class ID | Driver | Description |
|----------|--------|-------------|
| 0 | Commodity-Driven | Permanent conversion for agriculture, mining, energy |
| 1 | Shifting Agriculture | Small-scale, rotational cultivation with regrowth |
| 2 | Forestry | Large-scale timber harvesting with planned regrowth |
| 3 | Wildfire | Natural or anthropogenic fire-driven loss |
| 4 | Urbanization | Expansion of settlements and infrastructure |

---

## Architecture Overview

```
                          ┌─────────────────────────────────────┐
                          │       Descartes Labs API            │
                          │   (Landsat 5 / 7 / 8, GFC)         │
                          └──────────────┬──────────────────────┘
                                         │
                          ┌──────────────▼──────────────────────┐
                          │     Satellite Data Acquisition      │
                          │  Cloud masking · NDVI filtering     │
                          │  Multi-sensor fallback chain        │
                          │  Annual & median compositing        │
                          └──────────────┬──────────────────────┘
                                         │
              ┌──────────────────────────▼──────────────────────────┐
              │               Preprocessing & Augmentation          │
              │  Region mapping · Aux feature imputation (Z-score)  │
              │  Stratified sampling · Aggressive augmentation      │
              │  (flip, affine, elastic, synthetic clouds, S&P)     │
              └────────┬──────────────┬─────────────┬───────────────┘
                       │              │             │
            ┌──────────▼───┐  ┌───────▼──────┐  ┌──▼────────────────┐
            │  CNN Branch  │  │ LRCN Branch  │  │  FusionNet Branch │
            │  (14 archs)  │  │  CNN + LSTM  │  │  CNN + Geo + Aux  │
            └──────────────┘  └──────────────┘  └───────────────────┘
                       │              │             │
              ┌────────▼──────────────▼─────────────▼───────────────┐
              │            Evaluation & Inference                    │
              │  Per-class P/R/F1 · Loss-area-weighted metrics      │
              │  Per-region breakdown · Confusion matrices           │
              └─────────────────────────────────────────────────────┘
```

---

## Data Pipeline

### Satellite Imagery Acquisition

The data acquisition module (`data/download_images.py`) implements a multi-sensor, cloud-aware download pipeline via the Descartes Labs geospatial platform:

- **Spatial tiling**: Each deforestation event is centered in a 10 km &times; 10 km tile at 15 m/pixel resolution (Landsat), yielding 666 &times; 666 pixel scenes
- **Multi-sensor fallback chain**: Landsat 8 Tier 1 (post-2012) &rarr; Landsat 7 Pre-Collection &rarr; Landsat 5, ensuring temporal coverage from 2000&ndash;2018
- **Spectral bands**: RGB (red, green, blue) + infrared (NIR, SWIR1, SWIR2) + cloud/cirrus QA bands
- **Cloud filtering**: Scene-level cloud fraction thresholding (< 0.5 for search, < 0.005 for acceptance), brightness masking, cirrus band filtering, and NDVI-based quality control (mean NDVI > 48,000 for Landsat 7)
- **Compositing**: Masked median compositing of the top-5 lowest-cloud scenes per location, plus per-year annual median composites for temporal modeling
- **Parallelization**: Multithreaded download via `ThreadPoolExecutor` for large-scale batch acquisition

### Forest Loss Polygon Extraction

The polygon module (`data/download_polygons.py`) extracts deforestation event boundaries from Global Forest Change loss-year rasters:

- Contour detection via `cv2.findContours` on the loss-year mask
- Filtered by `treecover2000 > 30%` canopy density threshold
- Polygon areas computed per loss year and serialized as Shapely geometries
- Loss area serves as a sample weight for evaluation and as a feature for fusion models

### Preprocessing & Feature Engineering

Post-download preprocessing (`data/intermediate_module.py`) constructs the training-ready dataset:

- **Label mapping**: Curtis et al. GoodeR grid IDs mapped to 5-class taxonomy (classes 6/7 filtered)
- **Geographic encoding**: GoodeR IDs mapped to 7 continental regions (NA, LA, EU, AF, AS, SEA, OC) via boundary lookup
- **Auxiliary feature imputation**: 82 geospatial features (fire radiative power, population density, tree cover, land cover fractions, forest gain/loss metrics) with anomalous values (< &minus;10<sup>30</sup>) replaced by zero, following Curtis et al. methodology
- **Standardization**: Z-score normalization of auxiliary features using training set statistics, persisted for val/test consistency

### Data Augmentation

Seven augmentation presets are implemented via `imgaug`, with the **aggressive** preset used in production:

| Preset | Operations |
|--------|-----------|
| `flip` | Horizontal + vertical flips (p=0.5 each) |
| `affine` | 2-of: flips, scale (0.95&ndash;1.05), translate (&pm;3%), 90/180/270&deg; rotation |
| `cloud` | Flips + rotation, then 50% synthetic clouds/fog/snowflakes |
| `sap` | 2-of: salt-and-pepper noise, flips, scale, translate, fine rotation |
| `aggressive` | 2-of: salt-and-pepper (up to 0.1), flips, scale, translate (&pm;5%), rotation, shear (&pm;5&deg;), elastic deformation |

Image preprocessing: resize to 300&times;300 &rarr; random crop to 224&times;224 (train) / center crop (val/test) &rarr; augmentation &rarr; ImageNet normalization.

### Stratified Sampling & Class Balancing

- 85/15 stratified train/validation split preserving class and geographic distribution
- Optional inverse-frequency class weighting for cross-entropy loss to mitigate geographic and class imbalance
- Year cutoff filtering (default: 2012+) to focus on the Landsat 8 era where image quality is most consistent

---

## Model Zoo

### CNN Backbones (14 architectures)

All backbones are initialized with ImageNet-pretrained weights and support multi-temporal input via channel expansion (3 &times; 4 = 12 input channels for 4-frame sequences, with pretrained weights replicated across expanded channels):

| Family | Variants | Params |
|--------|----------|--------|
| DenseNet | DenseNet-121, DenseNet-161, DenseNet-201 | 7M&ndash;18M |
| ResNet | ResNet-18, ResNet-34, ResNet-101, ResNet-152 | 11M&ndash;58M |
| Inception | Inception-v3, Inception-v4 | 22M&ndash;41M |
| SE-Net | SENet-154, SE-ResNeXt-101 (32&times;4d) | 113M&ndash;145M |
| NASNet | NASNet-A-Large, MNASNet (Mobile) | 4M&ndash;84M |
| ResNeXt | ResNeXt-101 (64&times;4d) | 83M |

### LRCN &mdash; Long-term Recurrent Convolutional Network

A spatiotemporal architecture for modeling deforestation dynamics across multi-year satellite imagery:

```
Input: (B, C, S, H, W)     # S = 4 time steps (annual composites)
  │
  ├─► CNN Backbone          # Shared-weight feature extraction per frame
  │     └─► f_t ∈ ℝ^d      # Per-frame feature vector (d = backbone-dependent)
  │
  ├─► LSTM                  # Temporal sequence modeling
  │     hidden_dim = 128
  │     num_layers = 1
  │     └─► h_S ∈ ℝ^128    # Final hidden state
  │
  └─► Linear(128, 5)        # Classification head
        └─► logits ∈ ℝ^5
```

- Captures temporal evolution of forest loss (e.g., gradual commodity clearing vs. sudden wildfire)
- Supports variable-length sequences with zero-padding
- Compatible with all 14 CNN backbones (e.g., `Sequential2DClassifier-DenseNet121`)

### FusionNet &mdash; Multimodal Late Fusion

Fuses visual, geographic, and tabular features for comprehensive deforestation driver classification:

```
CNN Backbone ──► Linear(d, 128) ──┐
                                   │
Positional Encoding (lat/lon) ────┤    ┌──► FC(dim, 128) ──► ReLU ──► Dropout(0.2)
  sin/cos, N=5, λ=10 → 40-dim     ├────┤
                                   │    └──► FC(128, 128) ──► ReLU ──► Dropout(0.2)
One-hot Region (7-dim) ──────────┤                │
                                   │          FC(128, 5) ──► logits
Polygon Loss Area (1-dim) ───────┤
                                   │
Auxiliary Features (82-dim) ──────┘
  Fire stats · Population · Tree cover · Land cover · Forest gain/loss
```

- **Positional encoding**: Sinusoidal encoding of (lat, lon) with 5 frequency bands and wavelength 10, producing 40-dimensional geographic embeddings
- **Auxiliary features (82 total)**: Fire radiative power (21 features at 1 km and 10 km), forest gain/loss metrics (28), land cover fractions (4), population density and change (21), tree cover (7)
- **Fusion architecture**: 2-layer MLP (128 &rarr; 128 &rarr; 5) with ReLU activations and dropout regularization (p=0.2)

### RegionModel &mdash; Per-Continent Ensemble

An ensemble of independently trained, region-specific models that routes each sample to its corresponding geographic expert at inference time. Supports 7 continental regions with any backbone architecture.

### SeCo &mdash; Self-Supervised Pretrained Classifier

Leverages Seasonal Contrast (SeCo) self-supervised pretraining on large-scale unlabeled satellite imagery (100K&ndash;1M samples) to initialize ResNet-18/50 backbones, with a linear classification head fine-tuned on the deforestation task.

### Traditional ML Baselines

- **Random Forest** and **Logistic Regression** with one-vs-rest strategy and GridSearchCV
- Post-processing via nearest-neighbor fallback (haversine distance) for low-confidence predictions (max probability < 0.5)
- Loss-area-weighted evaluation for fair comparison with deep learning methods

---

## Training & Evaluation

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch Lightning 0.9 |
| Optimizer | Adam (lr=3&times;10<sup>-5</sup>, weight_decay=10<sup>-2</sup>) |
| LR Schedule | ReduceLROnPlateau (patience=4, monitor=val_acc) |
| Early Stopping | patience=25, monitor=avg_val_acc |
| Gradient Clipping | max_norm=0.5 |
| Batch Size | 10 (train) / 1 (val/test) |
| Max Epochs | 100 |
| Loss Function | CrossEntropyLoss (optional class weighting) |
| Parallelism | DataParallel (multi-GPU) |
| Compute | SLURM-managed HPC cluster (16 GB/GPU) |

### Evaluation Protocol

- **Per-class metrics**: Precision, recall, F1-score, and support via `sklearn.metrics.classification_report`
- **Loss-area-weighted evaluation**: Samples weighted by polygon area to prioritize large-scale deforestation events
- **Per-region breakdown**: Independent evaluation across 7 continental regions to assess geographic generalization
- **Confusion matrices**: Normalized by true class, visualized as seaborn heatmaps
- **TensorBoard logging**: Training/validation curves, per-class accuracy, sample predictions with confidence scores

### Usage

```bash
# Training
python main.py train \
    --model="DenseNet121" \
    --lr=3e-5 \
    --weight_decay=1e-2 \
    --batch_size=10 \
    --max_epochs=100 \
    --augmentation="aggressive" \
    --pretrained=True

# Training with LRCN (temporal modeling)
python main.py train \
    --model="Sequential2DClassifier-DenseNet121" \
    --hidden_dim=128 \
    --num_lstm_layers=1 \
    --augmentation="aggressive"

# Training with FusionNet (multimodal)
python main.py train \
    --model="DenseNet121" \
    --late_fusion=True \
    --load_aux=True \
    --late_fusion_regions="onehot" \
    --late_fusion_polygon_loss=True

# Evaluation
python main.py test \
    --ckpt_path="path/to/best_checkpoint.ckpt"
```

---

## Hyperparameter Optimization

Three complementary tuning strategies were employed:

1. **Grid search** via Launchpad + SLURM: Exhaustive search over learning rate (3&times;10<sup>-6</sup>&ndash;3&times;10<sup>-2</sup>) and weight decay (10<sup>-3</sup>&ndash;10<sup>-1</sup>) across model variants
2. **Bayesian optimization** via NNI (Neural Network Intelligence): Tree-structured Parzen Estimator (TPE) tuner with 50 trials, 12 concurrent, and 2-hour budget
3. **Architecture search**: Systematic evaluation across 14 CNN backbones, LRCN temporal depths, and FusionNet feature ablations

---

## Results

- **80% classification accuracy** on 5-class global deforestation driver prediction
- Evaluated on multi-temporal Landsat imagery spanning 2000&ndash;2018 across all major forested biomes
- Loss-area-weighted metrics ensure results reflect real-world deforestation impact
- Per-region evaluation confirms generalization across geographically diverse landscapes (tropical, boreal, temperate)

---

## Project Structure

```
aicc-spr20-deforest-global/
├── main.py                      # CLI entry point (train / test)
├── config.yaml                  # Launchpad HP tuning (grid search)
├── config_nni.yaml              # NNI HP tuning (TPE Bayesian)
├── fnet_config.yaml             # FusionNet HP tuning
├── requirements.txt             # Python dependencies
│
├── data/                        # Data acquisition & preprocessing
│   ├── download_images.py       # Satellite image downloader (Landsat 5/7/8)
│   ├── download_images_v3.py    # Parallel downloader (ThreadPoolExecutor)
│   ├── download_polygons.py     # Forest loss polygon extraction (GFC)
│   ├── intermediate_module.py   # Region mapping, aux imputation, Z-score norm
│   ├── base_dataset.py          # Abstract PyTorch Dataset (multi-temporal loading)
│   ├── classification_dataset.py # __getitem__ with augmentation & fusion features
│   ├── hansen.py                # HansenDriversDataset (CSV filtering, splits)
│   ├── data_util.py             # Band manipulation, cloud masking, compositing
│   └── train_val_split.py       # Stratified 85/15 split
│
├── models/                      # Model architectures
│   ├── pretrained.py            # 14 CNN backbones (DenseNet, ResNet, SE-Net, NASNet)
│   ├── models_3d.py             # LRCN (CNN + LSTM temporal classifier)
│   ├── fusion.py                # FusionNet (late fusion: CNN + geo + aux)
│   ├── region.py                # RegionModel (per-continent ensemble)
│   ├── seco.py                  # SeCo self-supervised pretrained classifier
│   └── baseline.py              # Random Forest, Logistic Regression baselines
│
├── lightning/                   # Training framework
│   ├── model.py                 # LightningModule (train/val/test loops, data loaders)
│   ├── util.py                  # Checkpointing, early stopping callbacks
│   └── logger.py                # TensorBoard logging, confusion matrices
│
├── eval/                        # Evaluation
│   ├── loss.py                  # Loss functions (CE, BCE + class weighting)
│   └── metrics.py               # Accuracy metric
│
├── util/                        # Utilities
│   └── constants.py             # All constants (605 lines: paths, bands, labels, aux features)
│
├── testing_main.py              # Integration tests (train + test pipeline)
├── testing_model.py             # Unit tests (all model architectures)
├── testing_loss.py              # Unit tests (loss functions)
│
└── .circleci/                   # CI pipeline (Python 3.7, PyTorch 1.6)
    └── config.yml               # Model instantiation, LRCN, DenseNet121 tests
```

---

## Setup & Usage

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (16 GB+ VRAM recommended)
- Descartes Labs API credentials (for satellite data download)

### Installation

```bash
git clone https://github.com/<org>/aicc-spr20-deforest-global.git
cd aicc-spr20-deforest-global
pip install -r requirements.txt
```

### Dependencies

Core: `torch`, `pytorch-lightning`, `numpy`, `scikit-learn`, `imgaug`, `pretrainedmodels`
Geospatial: `descarteslabs`, `shapely`, `pyproj`, `geopy`, `pycountry_convert`
Visualization: `tensorboardX`, `seaborn`, `matplotlib`

---

## CI/CD

Dual CI pipeline ensuring model correctness and pipeline integrity:

- **CircleCI**: Python 3.7, PyTorch 1.6 CPU &mdash; tests model instantiation, LRCN forward/backward, DenseNet121 train/test, deterministic mode, ImageNet normalization
- **Travis CI**: Python 3.6 & 3.7 &mdash; runs full test suite (`testing_main.py`, `testing_model.py`, `testing_loss.py`)

---

## References

- Curtis, P.G., Slay, C.M., Harris, N.L., Tyukavina, A. & Hansen, M.C. *Classifying drivers of global forest loss.* Science 361, 1108&ndash;1111 (2018).
- Hansen, M.C. et al. *High-resolution global maps of 21st-century forest cover change.* Science 342, 850&ndash;853 (2013).
- Manas, O. et al. *Seasonal Contrast: Unsupervised Pre-Training from Uncurated Remote Sensing Data.* ICCV 2021.

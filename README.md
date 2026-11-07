<div align="center">

# Sentinel-1 Chennai LULC Mapping

### Monthly Land Use and Land Cover Prediction Using Sentinel-1 SAR and U-Net V2

**Study Area:** Chennai, Tamil Nadu, India  
**Study Period:** January–December 2021  
**Input Data:** Sentinel-1 VH and VV Polarization  
**Deep Learning Model:** U-Net V2

</div>

---

> **Research Note:** This project generates experimental LULC predictions, not field-verified land-cover maps. Comparison with ESA WorldCover 2021 achieved 46.60% overall accuracy and 16.93% mean IoU. Performance was particularly weak for agriculture, wetlands and water.

## 1. Project Overview

This project explores the use of Sentinel-1 Synthetic Aperture Radar (SAR) imagery and deep learning for monthly Land Use and Land Cover (LULC) prediction over Chennai, Tamil Nadu, during 2021.

The workflow uses VH and VV polarization bands from Sentinel-1. A U-Net V2 semantic segmentation model is trained using labelled samples from the S1_AI4LCC dataset and applied to monthly Sentinel-1 composites for Chennai.

The project generates twelve monthly LULC prediction maps, class distribution statistics, a monthly trend summary and a combined twelve-month visualization.

## 2. Project Objectives

1. Collect Sentinel-1 VH and VV observations for Chennai.
2. Prepare monthly satellite composites for 2021.
3. Prepare labelled training and validation patches using S1_AI4LCC.
4. Train a U-Net V2 model for five-class LULC segmentation.
5. Generate twelve monthly LULC prediction maps.
6. Calculate the percentage of the study area covered by each class.
7. Compare predictions with ESA WorldCover 2021.

## 3. Study Area

The study area is located in Chennai, Tamil Nadu, India.

| Boundary | Coordinate |
|---|---|
| West | 80.15° E |
| East | 80.30° E |
| South | 12.90° N |
| North | 13.05° N |

- **Coordinate Reference System:** EPSG:32644
- **Spatial Resolution:** 10 metres
- **Raster Dimensions:** 1,664 × 1,633 pixels

## 4. Datasets

### 4.1 S1_AI4LCC Training Dataset

The S1_AI4LCC dataset provides Sentinel-1 SAR input data and labelled land-cover samples.

- Dataset: https://huggingface.co/datasets/wtr001/S1_AI4LCC
- Input bands: VH and VV
- Output classes: Five land-cover classes

### 4.2 Sentinel-1 Chennai Imagery

Sentinel-1 GRD imagery was obtained through Microsoft Planetary Computer.

- Data source: https://planetarycomputer.microsoft.com/
- Collection: `sentinel-1-grd`
- Study period: 2021
- Polarizations: VH and VV
- Processing: Monthly median composites

### 4.3 ESA WorldCover 2021

ESA WorldCover 2021 was used as an independent reference product for validation.

- Reference data: https://worldcover2021.esa.int/

## 5. Land-Cover Classes

| Class ID | Class Name |
|---|---|
| 1 | Urban areas |
| 2 | Agricultural areas |
| 3 | Forests and semi-natural areas |
| 4 | Wetlands |
| 5 | Water surfaces |

## 6. Project Workflow

1. Sentinel-1 data collection.
2. VH/VV preprocessing and monthly compositing.
3. AI4LCC training-data preparation.
4. U-Net V2 model development.
5. Model training and validation.
6. Chennai monthly prediction.
7. LULC map generation.
8. Class distribution calculation.
9. ESA WorldCover comparison.

## 7. U-Net V2 Model Architecture

The project uses an encoder-decoder U-Net architecture for semantic segmentation.

| Component | Configuration |
|---|---|
| Input channels | 2 (VH and VV) |
| Encoder channels | 32, 64, 128 |
| Bridge channels | 256 |
| Decoder | Transposed convolution |
| Skip connections | Encoder to decoder |
| Output channels | 5 |
| Activation | ReLU |
| Normalization | Batch Normalization |

The saved model checkpoint contains the model state dictionary and training metadata.

**Checkpoint details:**

- Training epoch: 13
- Validation loss: 0.88901167

## 8. Repository Structure

```text
Sentinel1-Chennai-LULC-2021/
│
├── README.md
├── deep_learning.ipynb
└── prototype_prediction.py

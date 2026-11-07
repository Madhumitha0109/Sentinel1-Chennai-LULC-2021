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

The project generates twelve monthly LULC prediction maps, class distribution statistics, month-to-month change analysis, first-to-last-month change statistics and a combined twelve-month visualization.

## 2. Project Objectives

1. Collect Sentinel-1 VH and VV observations.
2. Prepare monthly satellite composites.
3. Prepare labelled training and validation patches using S1_AI4LCC.
4. Train a U-Net V2 model for five-class LULC segmentation.
5. Generate twelve monthly LULC prediction maps.
6. Calculate the percentage of the study area covered by each class.
7. Analyse month-to-month changes in predicted class distribution.
8. Compare predictions with ESA WorldCover 2021.

## 3. Study Area

The primary study area is Chennai, Tamil Nadu, India.

| Boundary | Coordinate |
|---|---|
| West | 80.15° E |
| East | 80.30° E |
| South | 12.90° N |
| North | 13.05° N |

- **Coordinate Reference System:** EPSG:32644
- **Spatial Resolution:** 10 metres
- **Raster Dimensions:** 1,664 × 1,633 pixels

The prediction prototype also allows users to enter a different central coordinate and define the starting year and month.

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
9. Monthly change analysis.
10. ESA WorldCover comparison.

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
├── prototype_prediction.py
├── requirements.txt
│
└── outputs/
    └── Monthly_Predictions/
        ├── LULC_2021_01.tif
        ├── LULC_2021_02.tif
        ├── LULC_2021_03.tif
        ├── LULC_2021_04.tif
        ├── LULC_2021_05.tif
        ├── LULC_2021_06.tif
        ├── LULC_2021_07.tif
        ├── LULC_2021_08.tif
        ├── LULC_2021_09.tif
        ├── LULC_2021_10.tif
        ├── LULC_2021_11.tif
        ├── LULC_2021_12.tif
        ├── Monthly_LULC_Maps.png
        ├── Monthly_Class_Distribution.csv
        ├── Monthly_Change_Percentage_Points.csv
        └── Total_First_to_Last_Month_Change.csv
```

## 9. Model Training

The model was trained using the S1_AI4LCC dataset.

**Training configuration:**

- Training patches: 173
- Validation patches: 42
- Patch size: 256 × 256 pixels
- Input channels: VH and VV
- Output classes: 5
- Best checkpoint: Epoch 13

Training notebook: `deep_learning.ipynb`

## 10. Prediction Prototype

The `prototype_prediction.py` script provides an interactive workflow for generating monthly LULC predictions.

### User inputs

The script requests:

1. Latitude
2. Longitude
3. Starting year
4. Starting month

The selected coordinate defines the centre of the prediction area. The script creates a bounding box around that coordinate and selects an appropriate UTM projection.

### Processing steps

1. Connects to Microsoft Planetary Computer.
2. Searches for Sentinel-1 GRD observations.
3. Creates monthly median VH and VV composites.
4. Downloads the trained U-Net V2 checkpoint.
5. Performs monthly image segmentation using overlapping patches.
6. Generates monthly LULC GeoTIFF maps.
7. Calculates monthly class percentages.
8. Calculates month-to-month class distribution changes.
9. Calculates first-to-last-month changes.
10. Generates a combined twelve-month visualization.

### Running the prototype

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the prediction script:

```bash
python prototype_prediction.py
```

Enter the requested coordinates, starting year and starting month when prompted.

The model checkpoint is hosted on Hugging Face:

https://huggingface.co/Madhumitha01/Sentinel1-Chennai-LULC-UNetV2

## 11. Prediction Outputs

All generated prediction outputs are stored in:

`outputs/Monthly_Predictions/`

### 11.1 Monthly LULC GeoTIFF Maps

The prototype generates twelve GeoTIFF files:

- `LULC_2021_01.tif`
- `LULC_2021_02.tif`
- `LULC_2021_03.tif`
- `LULC_2021_04.tif`
- `LULC_2021_05.tif`
- `LULC_2021_06.tif`
- `LULC_2021_07.tif`
- `LULC_2021_08.tif`
- `LULC_2021_09.tif`
- `LULC_2021_10.tif`
- `LULC_2021_11.tif`
- `LULC_2021_12.tif`

Each GeoTIFF contains the predicted land-cover class labels for the corresponding month.

### 11.2 Monthly Visualization

`Monthly_LULC_Maps.png` contains a combined visualization of the twelve monthly prediction maps arranged in a 3 × 4 layout.

### 11.3 Monthly Class Distribution

`Monthly_Class_Distribution.csv` contains the percentage distribution of the five predicted land-cover classes for each month.

### 11.4 Month-to-Month Change Analysis

`Monthly_Change_Percentage_Points.csv` records the changes in predicted class percentages between consecutive months.

### 11.5 First-to-Last-Month Change

`Total_First_to_Last_Month_Change.csv` summarizes the difference in predicted class percentages between the first and last prediction months.

These CSV files describe changes in predicted area proportions. They should not be interpreted as verified physical land-cover conversions.

## 12. Model Validation

The January prediction was compared with ESA WorldCover 2021 after spatial alignment and class grouping.

**Validation results:**

| Metric | Result |
|---|---:|
| Valid comparison pixels | 2,683,855 |
| Overall Accuracy | 46.60% |
| Macro F1-score | 23.52% |
| Mean IoU | 16.93% |

**Class-wise F1-score:**

| Class | F1-score |
|---|---:|
| Urban | 0.6984 |
| Agriculture | 0.0099 |
| Forest and semi-natural | 0.4669 |
| Wetlands | 0.0000 |
| Water | 0.0006 |

The validation indicates better agreement for urban and forest/semi-natural classes, while agriculture, wetlands and water have very weak agreement.

## 13. Limitations

1. The predictions are experimental and have not been validated using field-survey ground truth.
2. Agreement with ESA WorldCover does not establish true classification accuracy.
3. The model shows weak performance for agriculture, wetlands and water.
4. Differences between the training dataset and Chennai Sentinel-1 imagery may affect model generalization.
5. Monthly class distribution changes may reflect variations in SAR observations and classification uncertainty, not only actual land-cover change.
6. The monthly predictions should therefore be interpreted with caution.

## 14. Technologies Used

- Python
- PyTorch
- U-Net V2
- Sentinel-1 SAR
- Microsoft Planetary Computer
- Rasterio
- NumPy
- Pandas
- Matplotlib
- Hugging Face
- ESA WorldCover

## 15. Project Resources

**GitHub Repository:**  
https://github.com/Madhumitha0109/Sentinel1-Chennai-LULC-2021

**Training Dataset:**  
https://huggingface.co/datasets/wtr001/S1_AI4LCC

**Trained Model:**  
https://huggingface.co/Madhumitha01/Sentinel1-Chennai-LULC-UNetV2

**Monthly Prediction Outputs:**  
https://github.com/Madhumitha0109/Sentinel1-Chennai-LULC-2021/tree/main/outputs/Monthly_Predictions

---

<div align="center">

**Sentinel-1-Based Monthly Land Use and Land Cover Mapping**

Chennai, Tamil Nadu, India | 2021

</div>

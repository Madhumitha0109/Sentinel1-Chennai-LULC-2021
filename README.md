# Sentinel-1-Based Monthly Land Use and Land Cover Mapping

---

## 1. Project Overview

This project investigates semantic segmentation of Land Use and Land Cover (LULC) using Sentinel-1 Synthetic Aperture Radar (SAR) imagery and a U-Net deep-learning model.

The original workflow generates monthly Sentinel-1 VH/VV composites for Chennai, predicts five LULC classes, calculates class proportions and compares the January 2021 prediction with an aligned ESA WorldCover 2021 reference raster.

The working notebook also includes an additional geographical experiment using labelled S1_AI4LCC tiles. This experiment investigates geographical generalization, SAR-specific class confusion and the reliability of land-cover predictions across different regions.

> **Note:** The notebook contains the original Chennai workflow and separate experimental training sections. Not every cell should be executed sequentially without checking the data paths, model checkpoints and experiment configuration.

## 2. Project Objectives

1. Prepare monthly Sentinel-1 VH and VV composites for Chennai.
2. Develop a U-Net-based semantic segmentation model.
3. Classify five land-cover categories.
4. Generate monthly LULC prediction maps.
5. Calculate monthly class distributions and percentage-point changes.
6. Compare predictions against ESA WorldCover 2021.
7. Investigate additional labelled geographical tiles from S1_AI4LCC.
8. Evaluate geographical generalization and class-specific classification errors.
9. Discuss seasonal SAR effects and uncertainty in calculated land-cover changes.

## 3. Data Sources

### 3.1 Sentinel-1 Data

| Parameter | Description |
|---|---|
| Data source | Google Earth Engine |
| Collection | `COPERNICUS/S1_GRD` |
| Study area | Chennai, Tamil Nadu, India |
| Period | 01/01/2021 – 12/31/2021 |
| Sensor | Sentinel-1 SAR |
| Polarizations | VH and VV |
| Acquisition mode | Interferometric Wide (IW) |
| Composite method | Monthly median |
| Coordinate reference system | EPSG:32644 |
| Spatial resolution | 10 metres |

### 3.2 ESA WorldCover 2021

ESA WorldCover 2021 is used as an independent reference product for comparison with the January 2021 Chennai prediction.

**Reference dataset:**  
https://worldcover2021.esa.int/

The following approximate class mapping is used:

| Model Class | LULC Category | ESA WorldCover Code |
|---|---|---|
| 1 | Urban | 50 — Built-up |
| 2 | Agriculture | 40 — Cropland |
| 3 | Forest / Semi-natural | 10, 20, 30 |
| 4 | Wetlands | 90 — Herbaceous wetland |
| 5 | Water | 80 — Permanent water bodies |

Unmapped ESA categories are excluded from the comparison. This mapping is approximate and does not represent field-verified ground truth.

### 3.3 S1_AI4LCC Additional Training Dataset

The project also explores additional labelled Sentinel-1 SAR data from the S1_AI4LCC dataset.

**Dataset:**  
https://huggingface.co/datasets/wtr001/S1_AI4LCC

**Repository identifier:**
`wtr001/S1_AI4LCC`

The notebook explores multiple labelled GeoTIFF tiles. The additional-tile experiment uses the following geographical tile identifiers:

| Dataset Role | Tile Identifiers |
|---|---|
| Training tiles | `30TYR`, `31TCL`, `30TYP` |
| Geographical hold-out tile | `31TDL` |

The tile identifiers represent geographical grid references, not city names.

The selected GeoTIFF files contain three bands:

- Band 1: VH
- Band 2: VV
- Band 3: Land-cover label

The additional-tile experiment keeps the hold-out tile separate from training to evaluate performance on a geographically distinct dataset tile.

## 4. Land-Use and Land-Cover Classes

The model uses five target categories:

| Class ID | Category |
|---|---|
| 1 | Urban |
| 2 | Agriculture |
| 3 | Forest / Semi-natural |
| 4 | Wetlands |
| 5 | Water |

In the additional-tile workflow, class `0` represents unlabelled or ignored pixels. Classes `1–5` are the target categories.

## 5. Model Architecture and Training

### 5.1 Original Chennai U-Net V2 Model

The original Chennai model uses two input channels, VH and VV, and five output classes.

| Component | Configuration |
|---|---|
| Architecture | U-Net V2 |
| Input channels | 2 |
| Input bands | VH and VV |
| Encoder channels | 32, 64, 128 |
| Bridge channels | 256 |
| Decoder | Transposed-convolution upsampling |
| Skip connections | Encoder-to-decoder |
| Output classes | 5 |
| Recorded checkpoint epoch | 13 |
| Recorded validation loss | 0.88901167 |

The original checkpoint is hosted on Hugging Face:

https://huggingface.co/Madhumitha01/Sentinel1-Chennai-LULC-UNetV2

### 5.2 Additional Geographical Training Experiment

A separate U-Net experiment uses additional S1_AI4LCC training tiles.

| Parameter | Configuration |
|---|---|
| Training tiles | 30TYR, 31TCL, 30TYP |
| Hold-out tile | 31TDL |
| Training patches | 500 |
| Hold-out patches | 150 |
| Patch size | 128 × 128 |
| Input bands | VH and VV |
| Output channels | 6 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 8 |
| Epochs | 15 |
| Data augmentation | Horizontal and vertical flips |
| Loss function | Weighted cross-entropy |
| Ignored label | 0 |
| Checkpoint selection | Highest validation mIoU |

The additional experiment uses a separate normalization strategy based on median and robust scale statistics calculated from the training patches.

The original Chennai checkpoint and the additional-tile model are separate experiments and should not be treated as interchangeable.

## 6. Experimental Results

Results from different experiments are reported separately because they use different datasets, models and evaluation procedures.

### 6.1 Original Chennai Evaluation

The January 2021 Chennai prediction was compared with the aligned ESA WorldCover 2021 reference.

| Evaluation Metric | Result |
|---|---:|
| Evaluated pixels | 2,683,855 |
| Overall Accuracy | 46.60% |
| Macro F1-score | 23.52% |
| Mean IoU | 16.93% |

This is a cross-product comparison using an approximate class mapping rather than a field-based accuracy assessment.

### 6.2 Additional-Tile Geographical Hold-out Evaluation

The additional-tile experiment reports the following validation results:

| Metric | Result |
|---|---:|
| Best validation mIoU | 24.23% |
| Best epoch | 8 |
| Overall Accuracy | 68.22% |

#### Class-wise Performance

| Class | Precision | Recall | F1-score | IoU |
|---|---:|---:|---:|---:|
| Urban | 19.67% | 12.88% | 15.56% | 8.44% |
| Agriculture | 61.86% | 75.65% | 68.07% | 51.59% |
| Forest | 80.37% | 67.14% | 73.16% | 57.68% |
| Wetlands | 0.00% | 0.00% | 0.00% | 0.00% |
| Water | 5.35% | 8.84% | 6.66% | 3.45% |

The results show that performance varies considerably between land-cover classes. Agriculture and forest achieve stronger scores, while urban, wetlands and water remain difficult to classify.

Overall accuracy alone is therefore insufficient to describe the model's performance.

### 6.3 Separate Chennai Evaluation Cell

A later notebook evaluation records:

| Metric | Recorded Result |
|---|---:|
| Evaluated pixels | 2,702,120 |
| Overall Accuracy | 59.28% |
| Macro F1-score | 39.02% |
| Mean IoU | 28.94% |

**Important:** The notebook's prediction cell loads the original `best_sentinel1_unet_v2.pth` checkpoint, while nearby notes describe a France-to-Chennai experiment. This checkpoint-path inconsistency must be resolved before these metrics are presented as verified geographical transfer results.

## 7. Monthly LULC Prediction

The original Chennai workflow generates monthly LULC maps for 2021 using Sentinel-1 VH and VV observations.

The workflow includes:

1. Sentinel-1 data collection.
2. Monthly median composite generation.
3. VH/VV preprocessing.
4. U-Net V2 inference.
5. Monthly LULC map generation.
6. Monthly class-distribution calculation.
7. Percentage-point change analysis.
8. ESA WorldCover comparison.

The generated outputs include monthly GeoTIFF predictions, CSV summaries and map visualizations.

The class percentages represent predicted class distributions. They do not independently establish confirmed pixel-level land-cover transitions.

## 8. Critical Discussion and Limitations

### 8.1 Performance on Geographically Different Regions

A model trained using one geographical domain may not perform equally well in another region.

Differences in vegetation, soil, settlement structure, surface roughness, moisture and radar acquisition conditions can affect SAR backscatter values.

The additional S1_AI4LCC geographical tiles provide a separate tile-level hold-out experiment. However, this experiment does not establish reliable generalization across multiple countries or cities.

### 8.2 SAR-Specific Confusion Between Classes

Sentinel-1 backscatter responds to surface roughness, vegetation structure, soil moisture, geometric orientation and built structures.

Different land-cover classes can therefore produce similar VH and VV signatures.

The recorded confusion matrices indicate weak classification performance for some categories, particularly wetlands and water.

Class-wise precision, recall, F1-score and IoU are important for understanding these errors.

### 8.3 Seasonal Effects

Monthly SAR composites may contain different numbers of observations and different acquisition conditions.

Seasonal rainfall, soil moisture, crop cycles, vegetation growth and changing water levels can alter radar backscatter even when the underlying land-cover category remains unchanged.

Consequently, monthly changes in predicted class percentages may reflect seasonal or acquisition effects rather than actual land-use conversion.

### 8.4 Classification Errors and Calculated Change Percentages

Monthly class percentages are calculated from predicted labels.

Classification errors can:

- Overestimate the percentage of one land-cover category.
- Underestimate another category.
- Create artificial month-to-month fluctuations.
- Produce apparent land-cover changes that have not occurred.

Therefore, percentage-point changes should be treated as model-derived indicators rather than confirmed land-cover transitions.

### 8.5 Class Imbalance

Wetlands and water have very weak validation results in the additional-tile experiment.

A model may achieve reasonable overall accuracy while performing poorly on minority classes.

Balanced training samples, improved class representation and additional reference labels are required to improve these categories.

### 8.6 Reference Data Limitations

ESA WorldCover is an independent reference product, but its thematic categories do not perfectly match the five classes used in this project.

The comparison should therefore be interpreted as approximate cross-product agreement rather than definitive ground-truth accuracy.

## 9. Technologies and Libraries

The project uses:

- Python
- PyTorch
- U-Net V2
- Sentinel-1 SAR
- Google Earth Engine
- ESA WorldCover
- S1_AI4LCC
- NumPy
- Pandas
- Rasterio
- Matplotlib
- Scikit-learn
- GeoPandas
- Hugging Face Hub
- Google Colab

## 10. Running the Notebook

The notebook was developed using Google Colab.

### Execution Steps

1. Open the notebook in Google Colab.
2. Select a GPU runtime when available.
3. Run the environment setup cells.
4. Mount Google Drive.
5. Verify dataset and checkpoint file paths.
6. Authenticate Google Earth Engine if running the Chennai acquisition workflow.
7. Check the GeoTIFF band order, CRS and label values.
8. Execute the intended training or prediction section.
9. Generate evaluation metrics and prediction outputs.
10. Save the results and required figures.

> **Reproducibility note:** The notebook contains historical cells and multiple experimental workflows. Check the checkpoint path, normalization statistics and data split before running each experiment. Re-run the intended experiment from a clean runtime before treating saved notebook outputs as final results.

## 11. Known Limitations and Reproducibility Notes

- The original Chennai and additional-tile experiments use separate model configurations.
- The original Chennai checkpoint must not be confused with the additional-tile checkpoint.
- Input units and normalization need to be consistent between training and inference.
- The additional-tile evaluation covers a limited number of geographical tiles.
- Wetlands and water classification remain weak.
- Seasonal SAR variation can affect monthly predictions.
- The ESA WorldCover class mapping is approximate.
- The notebook contains historical cells and Google Drive-specific paths.
- Some package installation cells may require compatible versions in a fresh Colab runtime.
- The separate Chennai evaluation requires checkpoint provenance verification.
- The monthly class-percentage changes are not confirmed pixel-level land-cover transitions.

The reported results should be interpreted in the context of these limitations.

## 12. Project Resources

| Resource | Link |
|---|---|
| GitHub Repository | https://github.com/Madhumitha0109/Sentinel1-Chennai-LULC-2021 |
| S1_AI4LCC Dataset | https://huggingface.co/datasets/wtr001/S1_AI4LCC |
| Sentinel-1 GRD | https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD |
| ESA WorldCover 2021 | https://worldcover2021.esa.int/ |
| Original Model Repository | https://huggingface.co/Madhumitha01/Sentinel1-Chennai-LULC-UNetV2 |

## 13. Conclusion

This project implements a Sentinel-1 VH/VV-based LULC segmentation workflow for monthly Chennai mapping using a U-Net architecture.

The additional S1_AI4LCC experiment expands the geographical training data and introduces a separate tile-level hold-out evaluation.

The results demonstrate that classification performance varies significantly between land-cover classes. SAR-specific confusion, seasonal backscatter variation, class imbalance and geographical differences remain important challenges.

The monthly maps and percentage-change outputs should therefore be treated as experimental results rather than definitive land-cover measurements.

Future improvements should focus on geographically diverse training data, consistent preprocessing, balanced class representation, verified checkpoint provenance and independent reference validation.

---

**Developed as part of the Deep Learning Project**  
**SRH University Heidelberg**

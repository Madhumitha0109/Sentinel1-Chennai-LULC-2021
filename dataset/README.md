## Dataset Access and Sources

This project uses external geospatial datasets and satellite data services for model training, monthly land-cover prediction and reference-based evaluation.

The original satellite imagery and benchmark datasets are not stored directly in this GitHub repository because of their size and external access and licensing arrangements.

---

### 1. Training Dataset: S1_AI4LCC

The S1_AI4LCC dataset provides labelled Sentinel-1 SAR imagery and corresponding land-cover reference labels for semantic segmentation model development.

**Dataset links:**

- **Hugging Face Dataset:** https://huggingface.co/datasets/wtr001/S1_AI4LCC
- **Original AI4LCC Benchmark:** https://doi.theia.data-terra.org/ai4lcc/?lang=en
- **Dataset Repository:** `wtr001/S1_AI4LCC`

**Dataset information:**

| Parameter | Description |
|---|---|
| Dataset | S1_AI4LCC |
| Data type | Sentinel-1 SAR |
| Input bands | VH and VV |
| Reference data | Land-cover labels |
| File format | GeoTIFF and Zarr |
| Target classes | 5 |
| Original spatial data | France geographical regions |
| Primary use | Supervised semantic segmentation |

**Use in this project:**

The S1_AI4LCC dataset is used to prepare labelled training and validation samples for the five-class U-Net segmentation model.

The original experiment used a selected labelled source tile. Additional geographically distinct French Sentinel-1 tiles were also inspected and incorporated into the notebook's expanded training-data preparation workflow.

**Additional geographical training data:**

The expanded patch-extraction cell is configured to process the following additional GeoTIFF files:

| Dataset role | GeoTIFF filename |
|---|---|
| Original training source | `sentinel1_NA_30TXQ_clip1_reduced2020_labelclass1.tif` |
| Additional training tile 1 | `sentinel1_NA_30TXQ_clip0_reduced2020_labelclass1.tif` |
| Additional training tile 2 | `sentinel1_NA_30TXR_clip3_reduced2020_labelclass1.tif` |

The notebook uses the following data preparation conditions for these additional tiles:

- Patch size: 256 × 256 pixels.
- Minimum valid radar coverage: 80%.
- Minimum labelled pixel coverage: 25%.
- Radar inputs: VH and VV.
- Source label IDs: 1–5.
- Unlabelled pixels: 0 (ignored during model training).
- Normalization: Existing training-set mean and standard deviation.
- Patch extraction: Non-overlapping patches.

The additional patches are designed to be appended to the existing training arrays. The existing validation arrays are retained separately.

> **Geographical note:** The notebook's dataset audit identifies 12 available granules in France. These are geographical SAR tiles, not 12 different countries or cities.

**Geographical hold-out consideration:**

The notebook identifies `31TDL` as a potential geographical hold-out tile. It must remain excluded from model training if it is used for independent geographical evaluation.

A completed independent evaluation result should only be reported after the hold-out tile has been processed and evaluated without using it during training or model selection.

**Dataset size and licence:**

- Approximate dataset size: 23.8 GB.
- Licence shown on the Hugging Face dataset page: CC-BY-4.0.
- Original dataset files are accessed from their respective public sources.
- Appropriate attribution and the original dataset terms must be followed when using or redistributing the data.

---

### 2. Chennai Deployment Imagery: Sentinel-1 GRD

Sentinel-1 Ground Range Detected (GRD) imagery is used to generate monthly VH and VV composites for the Chennai study area.

**Satellite data sources:**

- **Google Earth Engine Dataset:** https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD
- **Microsoft Planetary Computer:** https://planetarycomputer.microsoft.com/
- **Planetary Computer STAC Collection:** `sentinel-1-grd`

**Study-area information:**

| Parameter | Description |
|---|---|
| Study area | Chennai, Tamil Nadu, India |
| West longitude | 80.15° E |
| South latitude | 12.90° N |
| East longitude | 80.30° E |
| North latitude | 13.05° N |
| Study period | January–December 2021 |
| Input bands | VH and VV |
| Coordinate reference system | EPSG:32644 |
| Spatial resolution | 10 metres |
| Composite method | Monthly median |

**Use in this project:**

Sentinel-1 GRD imagery is used to generate monthly SAR composites for the Chennai deployment area.

The notebook's Google Earth Engine workflow uses the `COPERNICUS/S1_GRD` collection, filters observations by the Chennai study area and 2021 date range, selects the IW acquisition mode and obtains VV and VH polarization data.

The standalone prediction prototype also uses the Microsoft Planetary Computer `sentinel-1-grd` STAC collection to retrieve monthly observations.

The resulting monthly VH/VV composites are supplied to the trained U-Net model for LULC prediction.

---

### 3. Reference Dataset: ESA WorldCover 2021

ESA WorldCover 2021 provides an independent land-cover reference product used to compare the project's Chennai prediction results.

**Dataset link:**

https://worldcover2021.esa.int/

**Dataset information:**

| Parameter | Description |
|---|---|
| Dataset | ESA WorldCover 2021 |
| Provider | European Space Agency |
| Reference year | 2021 |
| Spatial resolution | 10 metres |
| Data type | Global land-cover map |
| Project use | Reference-based validation |

**Use in this project:**

The January 2021 Chennai prediction is compared against an aligned ESA WorldCover 2021 reference raster.

The ESA categories are grouped into the project's five target categories using an approximate class correspondence.

| Project Class | Land-cover category | ESA WorldCover class |
|---|---|---|
| 1 | Urban | Built-up |
| 2 | Agriculture | Cropland |
| 3 | Forest / Semi-natural | Tree cover, shrubland and grassland |
| 4 | Wetlands | Herbaceous wetland |
| 5 | Water | Permanent water bodies |

Unmapped and excluded ESA categories are not included in the reported comparison.

> **Reference-data limitation:** ESA WorldCover is an independent land-cover product, not field-verified ground truth. Differences in class definitions and thematic mapping can affect the calculated evaluation metrics.

---

### 4. Data Processing and Usage Summary

The following table summarizes how each source contributes to the project.

| Data source | Main purpose | Project component |
|---|---|---|
| S1_AI4LCC | Labelled VH/VV SAR data and land-cover labels | U-Net model training |
| Additional French S1_AI4LCC tiles | Geographical diversity in training samples | Expanded training experiment |
| Sentinel-1 GRD | Monthly Chennai SAR observations | Monthly prediction |
| ESA WorldCover 2021 | Independent reference land-cover product | Chennai comparison |

**Data handling principles:**

1. Source datasets remain hosted by their original providers.
2. Sentinel-1 VH and VV bands are used as the model input channels.
3. Label values are checked and mapped to the five target classes.
4. Unlabelled pixels are excluded from supervised loss calculations.
5. The original validation data are kept separate from the additional training patches.
6. A geographical hold-out tile must not be used for training or model selection.
7. Reference-product comparisons are reported separately from model-training validation.

---

### 5. Dataset Attribution and References

The following external resources are acknowledged as data and service providers for this project:

1. **AI4LCC / S1_AI4LCC Dataset**  
   https://huggingface.co/datasets/wtr001/S1_AI4LCC

2. **Original AI4LCC Benchmark**  
   https://doi.theia.data-terra.org/ai4lcc/?lang=en

3. **European Space Agency — Sentinel-1 GRD**  
   https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD

4. **Microsoft Planetary Computer**  
   https://planetarycomputer.microsoft.com/

5. **European Space Agency — ESA WorldCover 2021**  
   https://worldcover2021.esa.int/

The project acknowledges the original dataset providers and satellite-data services. Their respective licences, citation requirements and data-use conditions apply.

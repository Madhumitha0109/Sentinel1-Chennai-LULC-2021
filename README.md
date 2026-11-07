Sentinel-1-Based Monthly Land Use and Land Cover Mapping
Deep Learning Project — SRH University Heidelberg
Student: Madhumitha Ramakrishnan
Student ID: 100004761
Primary study area: Chennai, Tamil Nadu, India
Study period: 2021
1. Project overview
This project investigates semantic segmentation of land use and land cover (LULC) using Sentinel-1 Synthetic Aperture Radar (SAR) imagery and a U-Net-style deep-learning model. The original workflow creates monthly Sentinel-1 VH/VV composites for Chennai, predicts five LULC classes, summarizes class proportions, and compares a January 2021 prediction with an aligned ESA WorldCover 2021 reference raster.

The working notebook also includes an exploratory geographical-expansion experiment using additional labelled S1_AI4LCC tiles. That experiment was added to study how a model behaves across different geographical data and to examine class-specific errors. It is reported separately from the original Chennai workflow because the two experiments use different data preparation, normalization and model configurations.
Notebook status: This is a cleaned working notebook containing the original project, diagnostics and experimental branches. It is not a single, fully consolidated script that should be run from the first cell to the last without review. Some cells are historical alternatives, duplicate diagnostics or rely on specific Google Drive paths. Run the relevant section in order and verify the paths and checkpoint used by each experiment.

2. Main objectives
1. Prepare monthly Sentinel-1 VH and VV composites for Chennai during 2021.
2. Use a U-Net segmentation model to classify five LULC categories.
3. Generate monthly prediction maps and class-distribution summaries.
4. Compare the January 2021 prediction with ESA WorldCover 2021.
5. Inspect SAR-specific class confusion and the effect of class imbalance.
6. Explore additional geographically separate S1_AI4LCC tiles and evaluate a separate experimental U-Net model.
7. Discuss transferability, seasonal backscatter variation and uncertainty in percentage-based change estimates.
3. Data sources
3.1 Sentinel-1 data for Chennai
The Chennai workflow uses the Google Earth Engine collection COPERNICUS/S1_GRD.
Item	Configuration
Study area	Chennai, Tamil Nadu, India
Approximate bounding box	80.15–80.30° E, 12.90–13.05° N
Period	01/01/2021–12/31/2021
Sensor	Sentinel-1 SAR, GRD
Polarizations	VH and VV
Composite	Monthly median
Recorded output grid	EPSG:32644, 10 m pixel size
Recorded raster dimensions	1664 × 1633 pixels


The notebook filters the collection for Interferometric Wide (IW) mode and scenes containing both VV and VH polarizations. Monthly composites are produced from the available observations for each month. Scene availability can vary between months.
3.2 Chennai reference data
ESA WorldCover 2021 is used as a reference product for an external comparison. The reference raster is aligned to the Chennai Sentinel-1 grid before pixel-wise evaluation.

The notebook uses this approximate mapping from ESA WorldCover classes to the project's five-class scheme:
Model class	LULC category	ESA WorldCover source codes
1	Urban	50 — Built-up
2	Agriculture	40 — Cropland
3	Forest / semi-natural	10 — Tree cover; 20 — Shrubland; 30 — Grassland
4	Wetlands	90 — Herbaceous wetland
5	Water	80 — Permanent water bodies


Unmapped ESA categories are excluded from this five-class comparison. The mapping is approximate: ESA categories do not correspond perfectly to the project's classes, and ESA WorldCover is a reference land-cover product rather than field-verified ground truth.
3.3 Additional labelled SAR data
The notebook accesses the public S1_AI4LCC dataset hosted on Hugging Face:
- Dataset: https://huggingface.co/datasets/wtr001/S1_AI4LCC
- Repository identifier: wtr001/S1_AI4LCC
- GeoTIFF folder used in the notebook: sentinel1_AI4LCC_NA_clips_reduced2020_labelclass1
The repository listing recorded in the notebook contains 148 files, including 48 GeoTIFF files across 12 tile identifiers. The notebook downloads and explores selected tiles for the additional experiment. The selected tile identifiers are grid/tile codes, not city names.

The additional-tile training experiment uses:
Role	Tile identifiers	Files used in the recorded patch split
Training	30TYR, 31TCL, 30TYP	9
Geographical hold-out validation	31TDL	3


The patch-sampling cell excludes clip3, resulting in three selected files per tile. The notebook's working notes refer to these as France-region tiles. Their location should be described using the GeoTIFF CRS and transformed geographic bounds, rather than inferring a country or city from the filename prefix.

The inspected GeoTIFFs contain three bands in the order VH, VV and label. The additional-tile workflow uses the first two bands as model inputs and the third band as the label. The recorded inspected tiles have a 10 m resolution; CRS and bounds should be checked for each file before combining tiles.
4. LULC classes
The project uses five output land-cover classes:
1. Urban
2. Agriculture
3. Forest / semi-natural
4. Wetlands
5. Water
In the S1_AI4LCC raster workflow, label value 0 is treated as unlabelled/ignored and values 1–`5` represent the five target classes. The original Chennai model converts source class IDs 1–5 to model indices 0–4 and uses an ignore value for pixels without a valid label. The later additional-tile experiment retains source IDs 1–5 and ignores label 0 during the loss/evaluation calculations.
5. Model and training approaches
5.1 Original Chennai U-Net V2 checkpoint
The original model uses two input channels (VH and VV) and five output classes. The recorded architecture has an encoder–decoder structure with convolution blocks, batch normalization, ReLU activations and skip connections.
Component	Configuration
Input channels	2 — VH and VV
Encoder feature channels	32, 64, 128
Bridge feature channels	256
Decoder	Transposed-convolution upsampling
Skip connections	Encoder-to-decoder connections
Output classes	5
Recorded checkpoint epoch	13
Recorded validation loss	0.88901167
Recorded input-band order	VH, VV
Stored normalization means	68.72029, 126.65641
Stored normalization standard deviations	19.140371, 35.750645


The notebook inspects this previously saved checkpoint from Google Drive. The checkpoint metadata is not evidence that every historical training cell in the working notebook is currently reproducible.
5.2 Additional-tile U-Net experiment
A separate U-Net experiment is trained using patches sampled from the three additional training tiles and evaluated on patches from the separate 31TDL tile.
Item	Recorded setting
Training tiles	30TYR, 31TCL, 30TYP
Hold-out tile	31TDL
Training patches	500
Hold-out patches	150
Patch dimensions	128 × 128
Input bands	VH and VV
Model output channels	6 (class 0 is reserved/ignored; classes 1–5 are target classes)
Optimizer	AdamW
Learning rate	0.001
Weight decay	0.0001
Batch size	8
Epochs	15
Augmentation	Random horizontal and vertical flips
Normalization	Median and robust scale estimated from training patches; values clipped to −5…5
Loss	Weighted cross-entropy, ignoring class 0
Checkpoint selection	Highest validation mIoU


The patch sampler draws random windows from the source files. Training and hold-out data are separated by tile identifier, which is preferable to randomly splitting neighbouring patches from the same tile. The recorded run is an experiment and should be repeated from a fresh runtime before treating its saved metrics as final submission results.
6. Recorded results
The notebook contains results from more than one experiment. The metrics below must not be combined because they refer to different models, datasets and evaluation setups.
6.1 Original Chennai prediction compared with ESA WorldCover
The notebook records a January 2021 comparison against an aligned ESA WorldCover reference:
Metric	Recorded result
Evaluated pixels	2,683,855
Overall accuracy	46.60%
Macro F1-score	23.52%
Mean IoU	16.93%


This is a cross-product agreement assessment using an approximate class mapping, not a field-verified accuracy assessment.
6.2 Additional-tile experiment: hold-out tile validation
The recorded additional-tile U-Net run reports:
Metric	Recorded result
Best validation mIoU	24.23%
Best epoch	8
Overall accuracy	68.22%


Recorded class-wise validation metrics:
Class	Precision	Recall	F1-score	IoU
Urban	19.67%	12.88%	15.56%	8.44%
Agriculture	61.86%	75.65%	68.07%	51.59%
Forest	80.37%	67.14%	73.16%	57.68%
Wetlands	0.00%	0.00%	0.00%	0.00%
Water	5.35%	8.84%	6.66%	3.45%


The result shows substantial variation by class. The model performed better for agriculture and forest than for urban, wetlands and water. In the recorded confusion matrix, wetland predictions were absent, which is consistent with the very low wetland score. Overall accuracy therefore does not adequately describe performance on every class.
6.3 Separate January Chennai evaluation cell
A later notebook cell records a separate January Chennai evaluation with 2,702,120 evaluated pixels, 59.28% overall accuracy, 39.02% macro F1 and 28.94% mean IoU. Do not label this as a verified France-to-Chennai transfer result without first correcting the experiment: the prediction cell loads best_sentinel1_unet_v2.pth, the original Chennai checkpoint, while a nearby results cell describes the experiment as France-to-Chennai. The checkpoint path and experiment description are inconsistent in the current notebook.
7. Monthly prediction and change summaries
The original Chennai workflow produces twelve monthly LULC GeoTIFFs for 2021 and a monthly class-distribution summary. The associated project outputs recorded in Google Drive include:
- Twelve monthly Chennai prediction rasters (Chennai_LULC_2021_01.tif through Chennai_LULC_2021_12.tif).
- A monthly class-distribution CSV.
- An aligned ESA WorldCover reference raster.
- Validation class metrics and a confusion matrix.
- A twelve-month map visualization.
Monthly class proportions and their differences are calculated from model predictions. They describe changes in predicted class composition, not confirmed pixel-level land-cover transitions.
8. Critical discussion and limitations
8.1 Geographic transferability
A model trained on one geographic domain may encounter different surface materials, settlement patterns, vegetation, climate and radar acquisition conditions elsewhere. Adding separate S1_AI4LCC tiles introduces more geographical variation, and the 31TDL tile provides a tile-level hold-out evaluation for the additional experiment. However, the available experiment does not represent a broad multi-country evaluation. Tile identifiers must not be presented as city names or as proof of distinct countries.
8.2 SAR-specific class confusion
Sentinel-1 backscatter responds to surface roughness, moisture, vegetation structure, geometry and built structures. Different land-cover classes can therefore produce similar VH/VV signatures. The recorded confusion matrices show notable class confusion, including weak discrimination for wetlands and water in the additional-tile experiment. These errors should be discussed using per-class precision, recall, F1 and IoU rather than overall accuracy alone.
8.3 Radar scale and normalization differences
The notebook's diagnostic cells identify a major input-domain issue: the inspected additional S1_AI4LCC tile contains positive-valued VH/VV backscatter data, whereas the Chennai monthly raster values are stored in dB and are predominantly negative. The original checkpoint normalization statistics cannot automatically be transferred between these differently represented inputs. The additional-tile experiment uses a separate robust normalization calculated from its training patches. Preprocessing, units, band order, invalid-pixel handling and normalization must be made consistent before interpreting cross-region predictions.
8.4 Seasonal effects
Monthly median composites can contain different numbers of observations and different acquisition conditions. Rainfall, soil moisture, crop cycles, vegetation phenology and changing water levels can alter SAR backscatter even when the underlying land-use category has not changed. A month-to-month change in a predicted class percentage may therefore reflect seasonal or acquisition effects rather than permanent land-cover conversion.
8.5 Classification errors and change percentages
Class-percentage summaries are calculated from predicted labels. A systematic confusion can inflate one class and suppress another; if the confusion changes between months, it can create apparent fluctuations that are not real land-cover changes. The monthly percentage-point outputs should be interpreted as model-derived indicators and not as confirmed land-use transitions without independent reference data.
8.6 Class imbalance and reference limitations
The additional-tile validation results show very weak wetland and water performance. Rare classes have few labelled pixels, and a model can obtain a reasonable overall accuracy while failing to identify a minority class. ESA WorldCover is also a thematic reference product with a class mapping that is not identical to the project's categories. More local reference samples and independent validation are needed for stronger claims.
9. Running the notebook
The notebook was developed for Google Colab and uses Google Drive for project files and checkpoints.
1. Open the notebook in Google Colab.
2. Select a GPU runtime when available.
3. Run the environment check and mount Google Drive.
4. Confirm that the expected project folders and checkpoint files exist.
5. Authenticate Google Earth Engine and configure an authorized Cloud project if running the Sentinel-1 acquisition cells.
6. Review the dataset, CRS, band order, label IDs and normalization before preparing patches.
7. Run only the intended experiment section and ensure the checkpoint path matches that experiment.
8. Export the evaluation metrics, confusion matrix and prediction products required for the report.
Main packages used
The notebook imports or installs the following packages and services in different sections:
- Python and PyTorch
- NumPy and Pandas
- Rasterio
- Matplotlib
- scikit-learn
- GeoPandas
- tqdm
- Hugging Face Hub and Datasets
- Zarr, numcodecs, xarray and Dask for dataset inspection
- Google Earth Engine Python API (earthengine-api)
- Google Colab / Google Drive integration
The exact package versions should be recorded from the final execution environment. The notebook contains a Hugging Face Hub version downgrade cell that produced dependency-conflict warnings; avoid blindly running that legacy install cell. Install compatible versions in a fresh Colab runtime as required.
10. Reproducibility and known notebook cautions
- Several cells use absolute Google Drive paths such as /content/drive/MyDrive/Sentinel1_LULC_Project and /content/drive/MyDrive/Chennai_LULC_2021. Update them if the folder structure differs.
- The notebook includes multiple versions of dataset preparation and training. Do not mix arrays, normalization files or checkpoints from different experiments.
- The original Chennai checkpoint (best_sentinel1_unet_v2.pth) and the additional-tile checkpoint (France_Improved_UNet.pth) are separate model artifacts.
- Some historical Git clone cells fail in the saved execution record; the Hugging Face dataset access cell succeeds.
- The notebook contains a transfer-evaluation description whose text and checkpoint path do not match. Resolve this before reporting it as France-to-Chennai transfer performance.
- Confirm the exact geographical bounds of every tile with its raster metadata before naming countries or cities.
- Re-run the intended experiment from a clean runtime before using saved cell outputs as final results; notebook outputs may be stale relative to edited code.
- Do not claim that all performance metrics exceed 70%. The recorded additional-tile validation mIoU is 24.23%, and wetland performance is 0%.
11. Project resources
Resource	Link
S1_AI4LCC dataset	https://huggingface.co/datasets/wtr001/S1_AI4LCC
Google Earth Engine Sentinel-1 GRD collection	https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD
ESA WorldCover 2021	https://worldcover2021.esa.int/
Project GitHub repository	https://github.com/Madhumitha0109/Sentinel1-Chennai-LULC-2021
Original trained model repository	https://huggingface.co/Madhumitha01/Sentinel1-Chennai-LULC-UNetV2


12. Conclusion
The project demonstrates a Sentinel-1 VH/VV to LULC segmentation workflow for monthly Chennai mapping and includes an additional geographically held-out S1_AI4LCC experiment. The added tile experiment broadens the spatial data used in model development, but the recorded class-wise results remain uneven and do not establish reliable performance for every land-cover class. The monthly maps and percentage changes should therefore be treated as experimental outputs. Stronger conclusions require consistent preprocessing, verified checkpoint provenance, additional balanced labels, multi-region evaluation and independent reference data.

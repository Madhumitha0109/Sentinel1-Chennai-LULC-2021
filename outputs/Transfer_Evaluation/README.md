# Chennai transfer evaluation

This folder contains the metrics for the separate January 2021 Chennai transfer experiment using the France-trained S1_AI4LCC U-Net V2 checkpoint.

- Prediction raster: Chennai_January_Transfer_Prediction.tif (not included here)
- Evaluation reference: aligned ESA WorldCover 2021
- Evaluated pixels: 2,702,120
- ESA mapping: 50→Urban; 40→Agriculture; 10,20,30,60→Forest/semi-natural; 90→Wetlands; 80→Water.
- ESA class 60 (bare/sparse vegetation) is grouped with forest/semi-natural in this experiment. This mapping differs from the earlier comparison in the main README.
- Prediction raster class values: model indices 0–4; nodata 255.

The metrics represent cross-product agreement, not field-verified accuracy. The prediction GeoTIFF and visualization should be added from the project Google Drive output directory if they are to be distributed with this repository.

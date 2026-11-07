# Dataset Access and Sources

This project uses the following external geospatial datasets and data services. The original satellite and benchmark data are not copied into this GitHub repository because of their size and external licensing/access arrangements.

## 1. Training dataset: S1_AI4LCC

- **Hugging Face dataset:** https://huggingface.co/datasets/wtr001/S1_AI4LCC
- **Original AI4LCC benchmark source:** https://doi.theia.data-terra.org/ai4lcc/?lang=en
- **Use in this project:** Source Sentinel-1 VH/VV imagery and corresponding land-cover labels used to prepare the five-class U-Net training and validation patches.
- **License shown on the Hugging Face dataset page:** CC-BY-4.0. Check the dataset card and original AI4LCC terms before redistribution or reuse.
- **Important:** The Hugging Face dataset is approximately 23.8 GB. Download/access it from its source page; it is not stored in this repository.

## 2. Chennai deployment imagery: Sentinel-1 GRD

- **Microsoft Planetary Computer:** https://planetarycomputer.microsoft.com/
- **STAC collection:** `sentinel-1-grd`
- **Use in this project:** Sentinel-1 GRD VH and VV observations queried by the prototype to create monthly median composites for the Chennai 2021 demonstration.

## 3. Reference data: ESA WorldCover 2021

- **ESA WorldCover 2021:** https://worldcover2021.esa.int/
- **Use in this project:** Independent reference product used for the January 2021 Chennai comparison. The ESA-to-model class mapping is approximate and is documented in the project README and evaluation results.

## Reproducibility notes

1. Read the dataset card and original benchmark terms before downloading.
2. Keep the VH, VV and label layers aligned when preparing training patches.
3. Follow the preprocessing and normalization steps documented in the training notebook and main README.
4. Do not treat the S1_AI4LCC spatial hold-out validation metrics and the Chennai ESA WorldCover comparison metrics as the same evaluation.

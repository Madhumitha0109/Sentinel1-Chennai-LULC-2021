"""
Prototype: Monthly Chennai LULC prediction using Sentinel-1 VH/VV and U-Net V2.

This script reproduces the project's monthly inference workflow:
1. Queries 2021 Sentinel-1 GRD scenes from Microsoft Planetary Computer.
2. Builds monthly VH/VV median composites over the Chennai study area.
3. Downloads the trained U-Net V2 checkpoint from the public Hugging Face model repository.
4. Computes annual Chennai channel statistics and uses the same fixed statistics for all months.
5. Runs overlapping 256x256 patch inference with stride 128 and Hann-window logit blending.
6. Saves twelve georeferenced GeoTIFF prediction maps and a monthly class-distribution CSV.

Important:
- These are experimental model predictions, not ground-truth LULC maps.
- The model was trained on AI4LCC data from a different geographical domain.
- The normalization strategy follows the project's final Colab inference workflow.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import rasterio
from affine import Affine
from rasterio.crs import CRS
import planetary_computer
import pystac_client
import odc.stac
from huggingface_hub import hf_hub_download


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

STUDY_BBOX = [80.15, 12.90, 80.30, 13.05]  # west, south, east, north
STUDY_CRS = "EPSG:32644"
OUTPUT_DIR = Path("outputs/Monthly_Predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REPO = "Madhumitha01/Sentinel1-Chennai-LULC-UNetV2"
MODEL_FILENAME = "best_sentinel1_unet_v2.pth"

PATCH_SIZE = 256
STRIDE = 128
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {
    1: "Urban areas",
    2: "Agricultural areas",
    3: "Forests and semi-natural areas",
    4: "Wetlands",
    5: "Water surfaces",
}


# ---------------------------------------------------------------------
# U-Net V2 architecture (same architecture used by the project)
# ---------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, in_channels=2, num_classes=5):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.bridge = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bridge = self.bridge(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(bridge), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


# ---------------------------------------------------------------------
# Retrieve Sentinel-1 scenes and create monthly median composites
# ---------------------------------------------------------------------

def load_monthly_composites():
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )
    search = catalog.search(
        collections=["sentinel-1-grd"],
        bbox=STUDY_BBOX,
        datetime="2021-01-01/2022-01-01",
    )
    items = [
        item for item in search.items()
        if item.datetime is not None and item.datetime.year == 2021
    ]

    if not items:
        raise RuntimeError("No Sentinel-1 scenes were returned for 2021.")

    print(f"Sentinel-1 scenes found: {len(items)}")
    monthly = {}

    for month in range(1, 13):
        month_items = [
            item for item in items
            if item.datetime.month == month
        ]
        if not month_items:
            raise RuntimeError(f"No Sentinel-1 scenes found for month {month:02d}.")

        for item in month_items:
            planetary_computer.sign_inplace(item)

        ds = odc.stac.load(
            month_items,
            bands=["vh", "vv"],
            bbox=STUDY_BBOX,
            crs=STUDY_CRS,
            resolution=10,
            groupby="solar_day",
        )
        monthly[month] = ds.median(dim="time", skipna=True)
        print(
            f"Month {month:02d}: {len(month_items)} scenes; "
            f"raster shape {monthly[month].vh.shape}"
        )

    return monthly


# ---------------------------------------------------------------------
# Load the public trained checkpoint
# ---------------------------------------------------------------------

def load_model():
    checkpoint_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        repo_type="model",
    )

    # The checkpoint is the project's own trusted file and includes metadata.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model = UNet(in_channels=2, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model checkpoint:", checkpoint_path)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Checkpoint validation loss:", checkpoint.get("val_loss"))
    return model, checkpoint


# ---------------------------------------------------------------------
# Overlapping-patch prediction with Hann-window weighted logit blending
# ---------------------------------------------------------------------

def tile_starts(length, patch_size=PATCH_SIZE, stride=STRIDE):
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def predict_month(image, model):
    _, height, width = image.shape

    pad_h = max(0, PATCH_SIZE - height)
    pad_w = max(0, PATCH_SIZE - width)
    padded = np.pad(
        image,
        ((0, 0), (0, pad_h), (0, pad_w)),
        mode="reflect",
    )
    padded_h, padded_w = padded.shape[1:]

    rows = tile_starts(padded_h)
    cols = tile_starts(padded_w)

    window = np.hanning(PATCH_SIZE).astype(np.float32)
    window = np.maximum(window, 0.05)
    weights = np.outer(window, window)

    logit_sum = np.zeros(
        (NUM_CLASSES, padded_h, padded_w), dtype=np.float32
    )
    weight_sum = np.zeros((padded_h, padded_w), dtype=np.float32)

    with torch.no_grad():
        for row in rows:
            for col in cols:
                patch = padded[
                    :, row:row + PATCH_SIZE, col:col + PATCH_SIZE
                ].copy()
                tensor = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)
                logits = model(tensor).squeeze(0).cpu().numpy()

                logit_sum[
                    :, row:row + PATCH_SIZE, col:col + PATCH_SIZE
                ] += logits * weights[None, :, :]
                weight_sum[
                    row:row + PATCH_SIZE, col:col + PATCH_SIZE
                ] += weights

    average_logits = logit_sum / np.maximum(
        weight_sum[None, :, :], 1e-8
    )
    prediction = np.argmax(average_logits, axis=0).astype(np.uint8)
    return prediction[:height, :width]


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main():
    print("Running on:", DEVICE)
    monthly = load_monthly_composites()
    model, checkpoint = load_model()

    # Build annual median image and calculate the fixed Chennai statistics.
    vh_stack = np.stack([monthly[m]["vh"].values for m in range(1, 13)])
    vv_stack = np.stack([monthly[m]["vv"].values for m in range(1, 13)])
    annual_vh = np.nanmedian(vh_stack, axis=0)
    annual_vv = np.nanmedian(vv_stack, axis=0)
    annual_image = np.stack([annual_vh, annual_vv], axis=0).astype(np.float32)

    chennai_means = np.nanmean(annual_image, axis=(1, 2), keepdims=True)
    chennai_stds = np.nanstd(annual_image, axis=(1, 2), keepdims=True)

    print("Fixed annual Chennai means:", chennai_means.reshape(-1))
    print("Fixed annual Chennai standard deviations:", chennai_stds.reshape(-1))

    # Derive GeoTIFF grid metadata from the monthly x/y coordinates.
    reference = monthly[1]
    x_coords = reference.x.values
    y_coords = reference.y.values

    pixel_width = float(np.median(np.diff(x_coords)))
    pixel_height = float(np.median(np.diff(y_coords)))
    transform = Affine(
        pixel_width, 0, float(x_coords[0]) - pixel_width / 2,
        0, pixel_height, float(y_coords[0]) - pixel_height / 2,
    )
    height, width = len(y_coords), len(x_coords)

    summary_rows = []

    for month in range(1, 13):
        print(f"\nPredicting month {month:02d}/12")
        vh = monthly[month]["vh"].values.astype(np.float32)
        vv = monthly[month]["vv"].values.astype(np.float32)
        image = np.stack([vh, vv], axis=0)

        normalized = (image - chennai_means) / (chennai_stds + 1e-6)
        normalized = np.nan_to_num(
            normalized, nan=0.0, posinf=0.0, neginf=0.0
        )

        # Model outputs are 0-4; exported project class IDs are 1-5.
        prediction = predict_month(normalized, model)
        prediction_labels = (prediction + 1).astype(np.uint8)

        row = {"Month": f"{month:02d}"}
        for model_index in range(NUM_CLASSES):
            class_id = model_index + 1
            percentage = float(np.mean(prediction == model_index) * 100)
            row[CLASS_NAMES[class_id]] = percentage
            print(f"{CLASS_NAMES[class_id]}: {percentage:.2f}%")
        summary_rows.append(row)

        output_path = OUTPUT_DIR / f"Chennai_LULC_2021_{month:02d}.tif"
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "uint8",
            "crs": CRS.from_epsg(32644),
            "transform": transform,
            "nodata": 0,
            "compress": "lzw",
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(prediction_labels, 1)
            dst.set_band_description(1, "Predicted LULC class (1-5)")
            dst.update_tags(
                month=month,
                year=2021,
                method="Experimental Sentinel-1 U-Net V2",
                input_bands="VH,VV",
                patch_size=PATCH_SIZE,
                stride=STRIDE,
            )

        print("Saved:", output_path)

    summary_path = OUTPUT_DIR / "Chennai_2021_Monthly_Class_Distribution.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print("\nAll twelve monthly prediction files have been written.")
    print("Output directory:", OUTPUT_DIR.resolve())
    print("Monthly summary:", summary_path.resolve())
    print(
        "\nCaution: These outputs are experimental predictions. "
        "The project validation against ESA WorldCover showed limited "
        "cross-product agreement (46.60% overall accuracy; 16.93% mean IoU)."
    )


if __name__ == "__main__":
    main()

"""
Sentinel-1 Monthly LULC Mapping Prototype
Sentinel-1 VH/VV + U-Net V2

Features:
1. Latitude and longitude input.
2. Start month and year input.
3. Twelve consecutive monthly LULC maps.
4. Monthly class-area percentages.
5. Month-to-month changes in percentage points.
6. Total change between the first and last months.
7. Twelve-month visualization.
8. GeoTIFF, CSV and PNG outputs.
"""

from pathlib import Path
from datetime import datetime
import calendar
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import rasterio
import matplotlib.pyplot as plt

from affine import Affine
from rasterio.crs import CRS
import planetary_computer
import pystac_client
import odc.stac
from huggingface_hub import hf_hub_download


# ============================================================
# 1. USER INPUTS
# ============================================================

LATITUDE = float(input("Enter latitude: "))
LONGITUDE = float(input("Enter longitude: "))
START_YEAR = int(input("Enter start year (e.g. 2021): "))
START_MONTH = int(input("Enter start month (1-12): "))

# Half-width of the study area around the selected coordinates.
# 0.075 degrees reproduces approximately the original Chennai area.
AREA_HALF_SIZE = 0.075

if not -90 <= LATITUDE <= 90:
    raise ValueError("Latitude must be between -90 and 90.")

if not -180 <= LONGITUDE <= 180:
    raise ValueError("Longitude must be between -180 and 180.")

if not 1 <= START_MONTH <= 12:
    raise ValueError("Start month must be between 1 and 12.")


# ============================================================
# 2. CONFIGURATION
# ============================================================

STUDY_BBOX = [
    LONGITUDE - AREA_HALF_SIZE,
    LATITUDE - AREA_HALF_SIZE,
    LONGITUDE + AREA_HALF_SIZE,
    LATITUDE + AREA_HALF_SIZE
]

STUDY_CRS = "EPSG:32644" if 72 <= LONGITUDE < 78 else None

# Automatically select UTM zone for the selected location.
utm_zone = int((LONGITUDE + 180) // 6) + 1
hemisphere = "326" if LATITUDE >= 0 else "327"
STUDY_CRS = f"EPSG:{hemisphere}{utm_zone:02d}"

OUTPUT_DIR = Path("outputs/Monthly_Predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REPO = "Madhumitha01/Sentinel1-Chennai-LULC-UNetV2"
MODEL_FILENAME = "best_sentinel1_unet_v2.pth"

PATCH_SIZE = 256
STRIDE = 128
NUM_CLASSES = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = {
    1: "Urban areas",
    2: "Agricultural areas",
    3: "Forests and semi-natural areas",
    4: "Wetlands",
    5: "Water surfaces"
}

CLASS_COLORS = [
    "#F03B45",
    "#F5D04C",
    "#279E91",
    "#4682B4",
    "#253C9C"
]


# ============================================================
# 3. DATE GENERATION
# ============================================================

def generate_months(year, month):
    months = []

    for i in range(12):
        month_index = month - 1 + i
        current_year = year + month_index // 12
        current_month = month_index % 12 + 1

        start_date = datetime(
            current_year, current_month, 1
        )

        last_day = calendar.monthrange(
            current_year, current_month
        )[1]

        end_date = datetime(
            current_year, current_month, last_day
        )

        months.append((start_date, end_date))

    return months


# ============================================================
# 4. U-NET V2 ARCHITECTURE
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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

        self.up3 = nn.ConvTranspose2d(
            256, 128, 2, stride=2
        )
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(
            128, 64, 2, stride=2
        )
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(
            64, 32, 2, stride=2
        )
        self.dec1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        bridge = self.bridge(self.pool(e3))

        d3 = self.dec3(
            torch.cat([self.up3(bridge), e3], dim=1)
        )

        d2 = self.dec2(
            torch.cat([self.up2(d3), e2], dim=1)
        )

        d1 = self.dec1(
            torch.cat([self.up1(d2), e1], dim=1)
        )

        return self.out(d1)


# ============================================================
# 5. LOAD SENTINEL-1 MONTHLY COMPOSITES
# ============================================================

def load_monthly_composites(month_ranges):

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )

    overall_start = month_ranges[0][0]
    overall_end = month_ranges[-1][1]

    search = catalog.search(
        collections=["sentinel-1-grd"],
        bbox=STUDY_BBOX,
        datetime=(
            f"{overall_start.isoformat()}/"
            f"{overall_end.isoformat()}"
        )
    )

    items = list(search.items())

    if not items:
        raise RuntimeError(
            "No Sentinel-1 scenes found for the selected location and dates."
        )

    print("Total Sentinel-1 scenes:", len(items))

    monthly = {}

    for index, (start_date, end_date) in enumerate(
        month_ranges, start=1
    ):

        month_items = [
            item for item in items
            if item.datetime is not None
            and start_date <= item.datetime.replace(
                tzinfo=None
            ) <= end_date
        ]

        if not month_items:
            raise RuntimeError(
                f"No Sentinel-1 scenes for "
                f"{start_date.strftime('%B %Y')}."
            )

        for item in month_items:
            planetary_computer.sign_inplace(item)

        dataset = odc.stac.load(
            month_items,
            bands=["vh", "vv"],
            bbox=STUDY_BBOX,
            crs=STUDY_CRS,
            resolution=10,
            groupby="solar_day"
        )

        monthly[index] = dataset.median(
            dim="time",
            skipna=True
        )

        print(
            f"{start_date.strftime('%B %Y')}: "
            f"{len(month_items)} scenes loaded."
        )

    return monthly


# ============================================================
# 6. LOAD TRAINED MODEL
# ============================================================

def load_model():

    checkpoint_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        repo_type="model"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model = UNet(
        in_channels=2,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Model loaded successfully.")
    print("Training epoch:", checkpoint.get("epoch"))
    print("Validation loss:", checkpoint.get("val_loss"))

    return model


# ============================================================
# 7. OVERLAPPING PATCH INFERENCE
# ============================================================

def tile_starts(length):

    if length <= PATCH_SIZE:
        return [0]

    starts = list(
        range(0, length - PATCH_SIZE + 1, STRIDE)
    )

    if starts[-1] != length - PATCH_SIZE:
        starts.append(length - PATCH_SIZE)

    return starts


def predict_month(image, model):

    _, height, width = image.shape

    pad_h = max(0, PATCH_SIZE - height)
    pad_w = max(0, PATCH_SIZE - width)

    padded = np.pad(
        image,
        ((0, 0), (0, pad_h), (0, pad_w)),
        mode="reflect"
    )

    padded_h, padded_w = padded.shape[1:]

    rows = tile_starts(padded_h)
    cols = tile_starts(padded_w)

    window = np.hanning(PATCH_SIZE).astype(np.float32)
    window = np.maximum(window, 0.05)

    weights = np.outer(window, window)

    logit_sum = np.zeros(
        (NUM_CLASSES, padded_h, padded_w),
        dtype=np.float32
    )

    weight_sum = np.zeros(
        (padded_h, padded_w),
        dtype=np.float32
    )

    with torch.no_grad():

        for row in rows:
            for col in cols:

                patch = padded[
                    :,
                    row:row + PATCH_SIZE,
                    col:col + PATCH_SIZE
                ].copy()

                tensor = torch.from_numpy(
                    patch
                ).unsqueeze(0).to(DEVICE)

                logits = model(
                    tensor
                ).squeeze(0).cpu().numpy()

                logit_sum[
                    :,
                    row:row + PATCH_SIZE,
                    col:col + PATCH_SIZE
                ] += logits * weights[None, :, :]

                weight_sum[
                    row:row + PATCH_SIZE,
                    col:col + PATCH_SIZE
                ] += weights

    average_logits = logit_sum / np.maximum(
        weight_sum[None, :, :],
        1e-8
    )

    prediction = np.argmax(
        average_logits,
        axis=0
    ).astype(np.uint8)

    return prediction[:height, :width]


# ============================================================
# 8. MONTHLY PERCENTAGES AND CHANGE CALCULATIONS
# ============================================================

def calculate_statistics(prediction):

    total_pixels = prediction.size

    return {
        CLASS_NAMES[class_id]:
        float(np.sum(prediction == class_id) / total_pixels * 100)
        for class_id in range(1, NUM_CLASSES + 1)
    }


# ============================================================
# 9. VISUALIZE TWELVE MONTHLY MAPS
# ============================================================

def create_visualization(predictions, month_ranges):

    fig, axes = plt.subplots(
        3, 4,
        figsize=(18, 13)
    )

    for index, ax in enumerate(axes.flat):

        prediction = predictions[index]

        start_date = month_ranges[index][0]

        ax.imshow(
            prediction,
            cmap=plt.matplotlib.colors.ListedColormap(
                CLASS_COLORS
            ),
            vmin=1,
            vmax=5,
            interpolation="nearest"
        )

        ax.set_title(
            start_date.strftime("%B %Y"),
            fontsize=12,
            fontweight="bold"
        )

        ax.axis("off")

    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=CLASS_COLORS[i],
            label=CLASS_NAMES[i + 1]
        )
        for i in range(NUM_CLASSES)
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        fontsize=9
    )

    plt.suptitle(
        "Monthly Sentinel-1 LULC Predictions",
        fontsize=17,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    output_path = OUTPUT_DIR / "Monthly_LULC_Maps.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("Visualization saved:", output_path)


# ============================================================
# 10. MAIN WORKFLOW
# ============================================================

def main():

    print("\nDevice:", DEVICE)
    print("Study area:", STUDY_BBOX)
    print("CRS:", STUDY_CRS)

    month_ranges = generate_months(
        START_YEAR,
        START_MONTH
    )

    monthly = load_monthly_composites(
        month_ranges
    )

    model = load_model()

    # Annual median composite for fixed normalization.
    vh_stack = np.stack([
        monthly[m]["vh"].values
        for m in range(1, 13)
    ])

    vv_stack = np.stack([
        monthly[m]["vv"].values
        for m in range(1, 13)
    ])

    annual_vh = np.nanmedian(
        vh_stack,
        axis=0
    )

    annual_vv = np.nanmedian(
        vv_stack,
        axis=0
    )

    annual_image = np.stack(
        [annual_vh, annual_vv],
        axis=0
    ).astype(np.float32)

    channel_means = np.nanmean(
        annual_image,
        axis=(1, 2),
        keepdims=True
    )

    channel_stds = np.nanstd(
        annual_image,
        axis=(1, 2),
        keepdims=True
    )

    print("Normalization means:", channel_means.ravel())
    print("Normalization standard deviations:", channel_stds.ravel())

    # GeoTIFF grid metadata.
    reference = monthly[1]

    x_coords = reference.x.values
    y_coords = reference.y.values

    pixel_width = float(np.median(np.diff(x_coords)))
    pixel_height = float(np.median(np.diff(y_coords)))

    transform = Affine(
        pixel_width,
        0,
        float(x_coords[0]) - pixel_width / 2,
        0,
        pixel_height,
        float(y_coords[0]) - pixel_height / 2
    )

    height = len(y_coords)
    width = len(x_coords)

    summary_rows = []
    change_rows = []
    predictions = []

    # Generate all twelve monthly maps.
    for index in range(1, 13):

        start_date = month_ranges[index - 1][0]

        print(
            f"\nProcessing {start_date.strftime('%B %Y')}..."
        )

        vh = monthly[index]["vh"].values.astype(np.float32)
        vv = monthly[index]["vv"].values.astype(np.float32)

        image = np.stack(
            [vh, vv],
            axis=0
        )

        normalized = (
            image - channel_means
        ) / (channel_stds + 1e-6)

        normalized = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        prediction = predict_month(
            normalized,
            model
        )

        prediction_labels = (
            prediction + 1
        ).astype(np.uint8)

        predictions.append(prediction_labels)

        percentages = calculate_statistics(
            prediction_labels
        )

        row = {
            "Month": start_date.strftime("%B %Y"),
            **percentages
        }

        summary_rows.append(row)

        for class_name, percentage in percentages.items():
            print(f"{class_name}: {percentage:.2f}%")

        output_path = OUTPUT_DIR / (
            f"LULC_{start_date.strftime('%Y_%m')}.tif"
        )

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "uint8",
            "crs": CRS.from_string(STUDY_CRS),
            "transform": transform,
            "nodata": 0,
            "compress": "lzw"
        }

        with rasterio.open(
            output_path,
            "w",
            **profile
        ) as dst:

            dst.write(
                prediction_labels,
                1
            )

            dst.set_band_description(
                1,
                "Predicted LULC class (1-5)"
            )

            dst.update_tags(
                year=start_date.year,
                month=start_date.month,
                input_bands="VH,VV",
                method="Experimental Sentinel-1 U-Net V2"
            )

        print("Saved:", output_path)

    # --------------------------------------------------------
    # Monthly percentage-point changes
    # --------------------------------------------------------

    class_columns = list(CLASS_NAMES.values())

    for index in range(12):

        current = summary_rows[index]

        if index == 0:
            previous = None
            change = {
                name: np.nan
                for name in class_columns
            }
        else:
            previous = summary_rows[index - 1]

            change = {
                name: (
                    current[name] - previous[name]
                )
                for name in class_columns
            }

        change_rows.append({
            "Month": current["Month"],
            **{
                name: round(value, 4)
                if not np.isnan(value)
                else np.nan
                for name, value in change.items()
            }
        })

    # --------------------------------------------------------
    # Total change: first month to last month
    # --------------------------------------------------------

    first_month = summary_rows[0]
    last_month = summary_rows[-1]

    total_change = {
        name: (
            last_month[name] - first_month[name]
        )
        for name in class_columns
    }

    total_change_row = {
        "Comparison": (
            f"{first_month['Month']} to {last_month['Month']}"
        ),
        **{
            name: round(value, 4)
            for name, value in total_change.items()
        }
    }

    # --------------------------------------------------------
    # Save CSV files
    # --------------------------------------------------------

    distribution_path = (
        OUTPUT_DIR / "Monthly_Class_Distribution.csv"
    )

    change_path = (
        OUTPUT_DIR / "Monthly_Change_Percentage_Points.csv"
    )

    total_change_path = (
        OUTPUT_DIR / "Total_First_to_Last_Month_Change.csv"
    )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        distribution_path,
        index=False
    )

    pd.DataFrame(
        change_rows
    ).to_csv(
        change_path,
        index=False
    )

    pd.DataFrame(
        [total_change_row]
    ).to_csv(
        total_change_path,
        index=False
    )

    print("\nMONTH-TO-MONTH CHANGE (percentage points)")
    print(pd.DataFrame(change_rows).to_string(index=False))

    print("\nTOTAL CHANGE (first month to last month)")
    print(pd.DataFrame([total_change_row]).to_string(index=False))

    # --------------------------------------------------------
    # Save twelve-month visualization
    # --------------------------------------------------------

    create_visualization(
        predictions,
        month_ranges
    )

    print("\n======================================")
    print("PROTOTYPE EXECUTION COMPLETED")
    print("======================================")
    print("Output folder:", OUTPUT_DIR.resolve())
    print("Monthly maps: 12 GeoTIFF files")
    print("Monthly class distribution:", distribution_path)
    print("Monthly percentage-point changes:", change_path)
    print("First-to-last-month change:", total_change_path)
    print("Twelve-month visualization: Monthly_LULC_Maps.png")

    print(
        "\nScientific note: Predictions are experimental. "
        "The reported ESA WorldCover comparison was "
        "46.60% overall accuracy and 16.93% mean IoU."
    )


if __name__ == "__main__":
    main()

import os
import calendar
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

folder = r"outputs\Monthly_Predictions"

colors = [
    "#d73027",
    "#fee08b",
    "#1a9850",
    "#74add1",
    "#2c7bb6"
]

names = [
    "Urban",
    "Agriculture",
    "Forest / Semi-natural",
    "Wetlands",
    "Water"
]

cmap = ListedColormap(colors)
norm = BoundaryNorm(
    [0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 5
)

fig, axes = plt.subplots(3, 4, figsize=(18, 13))

for month in range(1, 13):
    path = os.path.join(
        folder, f"LULC_2021_{month:02d}.tif"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with rasterio.open(path) as src:
        data = src.read(1)
        nodata = src.nodata

    if nodata is not None:
        data = np.ma.masked_equal(data, nodata)

    ax = axes[(month-1)//4, (month-1)%4]
    ax.imshow(data, cmap=cmap, norm=norm)
    ax.set_title(calendar.month_name[month] + " 2021")
    ax.axis("off")

legend = [
    Patch(facecolor=colors[i], label=names[i])
    for i in range(5)
]

fig.legend(
    handles=legend,
    loc="lower center",
    ncol=5,
    fontsize=11
)

fig.suptitle(
    "Monthly LULC Prediction Maps – Chennai 2021",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0.07, 1, 0.94])

output = os.path.join(folder, "Monthly_LULC_Maps_Report.png")
plt.savefig(output, dpi=300, bbox_inches="tight")
plt.close()

print("SUCCESS! Image saved at:", output)
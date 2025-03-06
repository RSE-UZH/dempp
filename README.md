# dempp

A Python package for DEM (Digital Elevation Model) processing.

## Features

-   Difference of DEM (DoD) computation
-   Raster statistics calculation and plotting
-   Convenient DEM loading and saving
-   Utilities for DEM manipulation and analysis

## Installation

### Install from source

Create a new Conda/Mamba environment with Python=3.11 (or your preferred Python version):

```bash
mamba create -n dempp python=3.11
mamba activate dempp
```

Clone the repository and install the package:

```bash
git clone https://github.com/your-username/dempp.git # Replace with your repository URL
cd dempp
pip install -e .
```

### Dependencies

`dempp` relies on the following packages:

-   `xdem`
-   `geoutils`
-   `numpy`
-   `matplotlib`
-   `tqdm`

These dependencies should be automatically installed when you install `dempp` using `pip`.

## Basic Usage

```python
import dempp
import xdem
from pathlib import Path

# Setup logging (optional)
logger = dempp.setup_logger(level="info")

# Example DEM paths (replace with your actual paths)
dem_path = Path("path/to/dem.tif")
reference_path = Path("path/to/reference_dem.tif")

# Load DEMs
dem = dempp.io.load_dem(dem_path)
reference = dempp.io.load_dem(reference_path)

# Compute the Difference of DEMs (DoD)
ddem = dempp.dod.compute_dod(dem, reference)

# Save the DoD
output_path = Path("output/ddem.tif")
dempp.io.save_dem(ddem, output_path)

# Compute and plot statistics
stats = dempp.dod.compute_dod_stats(dem, reference, make_plot=True)
```

## Modules

### `dempp.dod`

Contains functions for computing and analyzing Difference of DEMs (DoDs).

-   `compute_dod(dem: xdem.DEM, reference: xdem.DEM) -> xdem.DEM`: Computes the DoD between two DEMs.
-   `compute_dod_stats(dem: xdem.DEM, reference: xdem.DEM, make_plot: bool = True) -> RasterStatistics`: Computes statistics on the DoD and optionally generates a plot.

### `dempp.statistics`

Defines classes and functions for calculating raster statistics.

-   `RasterStatistics`: A class to store raster statistics.
-   `compute_raster_statistics(raster: xdem.DEM) -> RasterStatistics`: Computes statistics for a given raster.

### `dempp.elevation_bands`

Contains functions for working with elevation bands.

### `dempp.filter`

Contains functions for filtering DEMs.
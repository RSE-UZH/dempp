import os
import tempfile

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import rasterio.mask
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from dempp.elevation_bands import (
    extract_dem_window,
    extract_elevation_bands,
    round_to_decimal,
    vector_to_mask,
)


@pytest.fixture
def sample_dem():
    """Create a sample DEM with known elevation bands"""
    # Create sample DEM data: 10x10 array with elevation bands
    dem_data = np.array(
        [
            [-9999, -9999, 2150, 2160, 2170, 2185, 2200, 2210, -9999, -9999],
            [-9999, 2120, 2135, 2150, 2165, 2180, 2195, 2205, 2215, -9999],
            [2100, 2115, 2130, 2145, 2160, 2175, 2190, 2200, 2210, 2220],
            [2090, 2105, 2120, 2135, 2150, 2165, 2175, 2185, 2190, 2195],
            [2080, 2095, 2110, 2125, 2140, 2150, 2160, 2165, 2170, 2175],
            [2070, 2095, 2100, 2175, 2130, 2140, 2145, 2150, 2155, 2160],  # X
            [2060, 2075, 2090, 2105, 2115, 2125, 2130, 2135, 2140, 2145],
            [2050, 1995, 2000, 2080, 2090, 2100, 2110, 2115, 2120, 2125],  # X
            [2040, 1990, 2000, 2060, 2070, 2080, 2090, 2095, 2100, 2105],  # X
            [-9999, 2030, 2035, 2040, 2050, 2060, 2070, 2075, -9999, -9999],
        ],
        dtype=np.float32,
    )

    # Add random variability (±2m)
    np.random.seed(42)  # for reproducibility
    noise = np.random.uniform(-2, 2, dem_data.shape)
    mask = dem_data != -9999
    dem_data[mask] += noise[mask]

    # Define transform (10m resolution)
    transform = from_origin(0, 100, 10, 10)

    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        with rasterio.open(
            tmp.name,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=1,
            dtype=np.float32,
            crs="EPSG:32632",
            transform=transform,
            nodata=-9999,
        ) as dst:
            dst.write(dem_data, 1)

    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def sample_glacier():
    """Create a sample glacier polygon"""
    # Create polygon covering the central part of the DEM
    polygon = Polygon([(20, 20), (20, 80), (80, 80), (80, 20), (20, 20)])
    return gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:32632")


def test_extract_dem_window(sample_dem, sample_glacier):
    """Test DEM window extraction"""
    dem_window = extract_dem_window(sample_dem, sample_glacier)

    assert dem_window is not None
    assert dem_window.data.shape == (8, 8)  # Expected window size
    assert dem_window.transform[0] == 10  # Check x resolution
    assert dem_window.transform[4] == -10  # Check y resolution
    assert dem_window.transform[2] == 10  # Check x translation (1 pixel padding)
    assert dem_window.transform[5] == 90  # Check y translation (1 pixel padding)
    assert dem_window.no_data == -9999  # Check nodata value
    assert dem_window.window.col_off == 1  # Check column offset (1 pixel padding)
    assert dem_window.window.row_off == 1  # Check row offset (1 pixel padding)

    # save window to temporary file
    with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
        with rasterio.open(
            tmp.name,
            "w",
            driver="GTiff",
            height=dem_window.data.shape[0],
            width=dem_window.data.shape[1],
            count=1,
            dtype=dem_window.data.dtype,
            crs=dem_window.crs,
            transform=dem_window.transform,
            nodata=dem_window.no_data,
        ) as dst:
            dst.write(dem_window.data, 1)

        # Check if window is valid
        with rasterio.open(tmp.name) as src:
            assert src.width == 8
            assert src.height == 8
            assert src.crs == "EPSG:32632"
            assert src.transform[0] == 10
            assert src.transform[4] == -10
            assert src.transform[2] == 10
            assert src.transform[5] == 90
            assert src.nodata == -9999


def test_no_overlap_case(sample_dem):
    """Test case where glacier doesn't overlap with DEM"""
    outside_polygon = Polygon(
        [(1000, 1000), (1000, 1100), (1100, 1100), (1100, 1000), (1000, 1000)]
    )
    outside_glacier = gpd.GeoDataFrame(geometry=[outside_polygon], crs="EPSG:32632")

    dem_window = extract_dem_window(sample_dem, outside_glacier)
    assert dem_window is None


def test_vector_to_mask(sample_dem, sample_glacier):
    """Test glacier mask creation"""
    with rasterio.open(sample_dem) as src:
        mask = vector_to_mask(sample_glacier, (10, 10), src.transform, src.crs)

        # mask_rio = rasterio.mask.raster_geometry_mask(
        #     src, [sample_glacier.geometry.values[0]], invert=True, crop=True
        # )
        # with rasterio.open("mask_rio.tif", "w", **src.profile) as dst:
        #     dst.write(mask_rio[0].astype(np.uint8), 1)

    assert mask.shape == (10, 10)
    assert mask.dtype == bool
    assert np.sum(mask) > 0  # Should have some True values

    # save mask to file
    with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
        with rasterio.open(
            tmp.name,
            "w",
            driver="GTiff",
            height=mask.shape[0],
            width=mask.shape[1],
            dtype=np.uint8,
            count=1,
            crs=src.crs,
            transform=src.transform,
        ) as dst:
            dst.write(mask, 1)

        # Check if mask is valid
        with rasterio.open(tmp.name) as src:
            assert src.width == 10
            assert src.height == 10
            assert src.crs == "EPSG:32632"
            assert src.transform[0] == 10
            assert src.transform[4] == -10
            assert src.transform[2] == 0
            assert src.transform[5] == 100


def test_extract_elevation_bands_basic():
    """Test basic elevation band extraction"""
    # Create simple elevation data
    dem = np.array(
        [[100, 150, 200], [120, 160, 210], [140, 170, 220]], dtype=np.float32
    )
    mask = np.zeros_like(dem, dtype=bool)
    masked_dem = np.ma.masked_array(dem, mask=mask)

    bands = extract_elevation_bands(masked_dem, band_width=50, smooth_dem=False)

    assert bands is not None
    assert len(bands) >= 2  # Should have at least 2 bands
    assert all(hasattr(band, "label") for band in bands)
    assert all(hasattr(band, "band_lower") for band in bands)
    assert all(hasattr(band, "band_upper") for band in bands)
    assert all(hasattr(band, "data") for band in bands)


def test_round_to_decimal():
    """Test decimal rounding function"""
    assert round_to_decimal(123.456, decimal=0) == 123.0
    assert round_to_decimal(123.456, decimal=1) == 120.0
    assert round_to_decimal(123.456, decimal=2) == 100.0


def test_round_to_decimal_with_functions():
    """Test decimal rounding with different rounding functions"""
    assert round_to_decimal(123.6, decimal=0, func=np.floor) == 123.0
    assert round_to_decimal(123.4, decimal=0, func=np.ceil) == 124.0


if __name__ == "__main__":
    pytest.main([__file__])

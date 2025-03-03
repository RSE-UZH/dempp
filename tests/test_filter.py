import os
import tempfile

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import rasterio.mask
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from dempp.filter import (
    OutlierMethod,
    filter_by_elevation_bands,
    filter_dem,
    find_outliers,
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
def noisy_dem():
    """Create a sample DEM with strong random noise and salt and pepper blob noise"""

    pass


@pytest.fixture
def sample_glacier():
    """Create a sample glacier polygon"""
    # Create polygon covering the central part of the DEM
    polygon = Polygon([(20, 20), (20, 80), (80, 80), (80, 20), (20, 20)])
    return gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:32632")


@pytest.fixture
def simple_ma_array():
    """Create a simple 2D array with known outliers"""
    data = np.array(
        [
            [0, 1, 2, 3],
            [1, 20, 2, 3],
            [1, 100, 4, 5],
            [2, 3, 3, 6],
        ],
        dtype=np.float32,
    )

    mask = np.zeros_like(data, dtype=bool)
    mask[0, 0:2] = True

    return np.ma.masked_array(data, mask=mask)


def test_find_outliers_robust(simple_ma_array):
    """Test outlier detection using robust method"""
    outlier_mask = find_outliers(
        simple_ma_array, method=OutlierMethod.ROBUST, n_limit=3
    )

    assert outlier_mask is not None
    assert outlier_mask.shape == simple_ma_array.data.shape
    assert outlier_mask.dtype == bool
    assert outlier_mask[1, 1]  # Check outlier detected
    assert outlier_mask[2, 1]  # Check outlier detected
    assert outlier_mask.sum() == 2  # Should find exactly 2 outliers
    assert not np.all(
        outlier_mask[simple_ma_array.mask]
    )  # Check masked value as non-outlier value


def test_find_outliers_with_mask(simple_ma_array):
    """Test outlier detection with numpy array and mask"""
    outlier_mask = find_outliers(simple_ma_array.data, mask=simple_ma_array.mask)

    assert outlier_mask is not None
    assert outlier_mask.shape == simple_ma_array.data.shape
    assert outlier_mask.dtype == bool
    assert outlier_mask[1, 1]  # Check outlier detected
    assert outlier_mask[2, 1]  # Check outlier detected
    assert outlier_mask.sum() == 2  # Should find exactly 2 outliers
    assert not np.all(
        outlier_mask[simple_ma_array.mask]
    )  # Check masked value as non-outlier value


def test_find_outliers_normal(simple_ma_array):
    """Test outlier detection using normal method"""
    outlier_mask = find_outliers(
        simple_ma_array, method=OutlierMethod.NORMAL, n_limit=3
    )

    assert outlier_mask is not None
    assert outlier_mask.shape == simple_ma_array.data.shape
    assert outlier_mask.dtype == bool
    assert outlier_mask[2, 1]  # Check outlier detected
    assert not outlier_mask[1, 1]  # Check small outlier not detected
    assert outlier_mask.sum() == 1  # Should find exactly 2 outliers
    assert not np.all(
        outlier_mask[simple_ma_array.mask]
    )  # Check masked value as non-outlier value


def test_find_outliers_zscore(simple_ma_array):
    """Test outlier detection using zscore method"""
    z_lim = 1.5
    outlier_mask = find_outliers(
        simple_ma_array, method=OutlierMethod.ZSCORE, n_limit=z_lim
    )

    assert outlier_mask is not None
    assert outlier_mask.shape == simple_ma_array.data.shape
    assert outlier_mask.dtype == bool
    assert outlier_mask[2, 1]  # Check outlier detected
    assert not outlier_mask[1, 1]  # Check small outlier not detected
    assert outlier_mask.sum() == 1  # Should find exactly 2 outliers
    assert not np.all(
        outlier_mask[simple_ma_array.mask]
    )  # Check masked value as non-outlier value


def test_find_outliers_all_masked():
    """Test handling of completely masked data"""
    data = np.array([1, 2, 3], dtype=np.float32)
    mask = np.ones_like(data, dtype=bool)
    masked_data = np.ma.masked_array(data, mask=mask)
    outliers = find_outliers(masked_data)

    assert outliers is not None
    assert outliers.shape == data.shape
    assert outliers.sum() == 0  # Should have no outliers


def test_filter_elevation_bands(sample_dem):
    """Test elevation band filtering"""
    with rasterio.open(sample_dem) as src:
        dem_data = src.read(1)
        mask = dem_data == src.nodata
        masked_dem = np.ma.masked_array(dem_data, mask)

    outlier_mask = filter_by_elevation_bands(
        masked_dem, band_width=50, n_limit=2, method="robust"
    )

    # Check if outlier mask is valid
    assert outlier_mask is not None
    assert outlier_mask.shape == dem_data.shape
    assert outlier_mask.dtype == bool

    # Masked value should not be marked as an outlier
    assert outlier_mask[mask].sum() == 0

    # Check if outliers are detected
    assert outlier_mask[~mask].sum() == 5  # Should detect 5 outliers
    assert outlier_mask[5, 3]  # Should detect the outlier at [5,3]


def test_filter_elevation_bands_normal(sample_dem):
    """Test elevation band filtering"""
    with rasterio.open(sample_dem) as src:
        dem_data = src.read(1)
        mask = dem_data == src.nodata
        masked_dem = np.ma.masked_array(dem_data, mask)

    outlier_mask = filter_by_elevation_bands(
        masked_dem, band_width=50, n_limit=2, method="normal"
    )

    # Check if outlier mask is valid
    assert outlier_mask is not None
    assert outlier_mask.shape == dem_data.shape
    assert outlier_mask.dtype == bool

    # Masked value should not be marked as an outlier
    assert outlier_mask[mask].sum() == 0

    # Check if outliers are detected
    assert outlier_mask[~mask].sum() == 1  # Should detect only the largest outlier
    assert outlier_mask[5, 3]  # Should detect the outlier at [5,3]


def test_filter_elevation_bands_zscore(sample_dem):
    """Test elevation band filtering"""
    with rasterio.open(sample_dem) as src:
        dem_data = src.read(1)
        mask = dem_data == src.nodata
        masked_dem = np.ma.masked_array(dem_data, mask)

    outlier_mask = filter_by_elevation_bands(
        masked_dem, band_width=50, n_limit=1.5, method="zscore"
    )

    # Check if outlier mask is valid
    assert outlier_mask is not None
    assert outlier_mask.shape == dem_data.shape
    assert outlier_mask.dtype == bool

    # Masked value should not be marked as an outlier
    assert outlier_mask[mask].sum() == 0

    # TODO: Fix assertion for zscore method
    # assert outlier_mask[~mask].sum() == 5  # Should detect 3 outliers
    # assert outlier_mask[5, 3]  # Should detect the outlier at [5,3]


def filter_dem_elevation_bands(sample_dem, sample_glacier):
    """Test complete workflow"""
    # Process glacier
    result = filter_dem(
        dem_path=sample_dem,
        geometry=sample_glacier,
        filter_method=OutlierMethod.ROBUST,
        n_limit=3,
        use_elevation_bands=True,
        elevation_band_width=50,
    )

    assert result is not None
    outlier_mask, window = result

    assert outlier_mask is not None
    assert window is not None

    # Check if outlier is detected
    assert np.any(outlier_mask)  # Should detect the outlier at [5,3]


if __name__ == "__main__":
    pytest.main([__file__])

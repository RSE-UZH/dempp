import tempfile
from pathlib import Path

import geoutils as gu
import numpy as np
import pyproj
import pytest
import rasterio as rio
import xdem

from dempp.dod import (
    apply_mask,
    compute_dod,
    process_dod,
)
from dempp.statistics import RasterStatistics


@pytest.fixture
def sample_dems():
    """Fixture to provide sample DEMs for testing."""
    # Load example DEMs from xdem
    dem1 = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem_coreg"))
    dem2 = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
    return dem1, dem2


@pytest.fixture
def sample_dem_paths(sample_dems):
    """Fixture to provide paths to sample DEMs saved as temporary files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        dem1_path = temp_dir / "dem1.tif"
        dem2_path = temp_dir / "dem2.tif"

        # Save the DEMs to temporary files
        sample_dems[0].save(dem1_path)
        sample_dems[1].save(dem2_path)

        yield dem1_path, dem2_path


@pytest.fixture
def sample_mask(sample_dems):
    """Fixture to provide a sample mask."""
    dem = sample_dems[0]

    # Create a simple mask (center region of the DEM)
    mask_array = np.zeros(dem.shape, dtype=bool)
    center_y, center_x = mask_array.shape[0] // 2, mask_array.shape[1] // 2
    size = min(mask_array.shape[0], mask_array.shape[1]) // 4

    mask_array[center_y - size : center_y + size, center_x - size : center_x + size] = (
        True
    )

    # Create a mask with the same georeference as the DEM
    mask = gu.Mask.from_array(mask_array, transform=dem.transform, crs=dem.crs)
    return mask


@pytest.fixture
def sample_mask_path(sample_mask):
    """Fixture to provide a path to sample mask saved as a temporary file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        mask_path = temp_dir / "mask.tif"

        # Save the mask to a temporary file
        sample_mask.save(mask_path)

        yield mask_path


@pytest.fixture
def temp_output_dir():
    """Fixture to provide a temporary directory for output files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def dummy_dem():
    """Create a sample geoutils.Raster for testing."""
    np.random.seed(42)
    arr = np.random.randint(0, 255, size=(5, 5), dtype="uint8")
    raster_mask = np.random.randint(0, 2, size=(5, 5), dtype="bool")
    ma = np.ma.masked_array(data=arr, mask=raster_mask)
    return xdem.DEM.from_array(
        data=ma,
        transform=rio.transform.from_bounds(0, 0, 1, 1, 5, 5),
        crs=pyproj.CRS.from_epsg(4326),
        nodata=255,
    )


@pytest.fixture
def dummy_inlier_mask():
    """Create a sample geoutils.Mask for testing."""
    mask_data = np.array(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    return gu.Mask.from_array(
        data=mask_data,
        transform=rio.transform.from_bounds(0, 0, 1, 1, 5, 5),
        crs=pyproj.CRS.from_epsg(4326),
    )


def test_apply_mask(dummy_dem, dummy_inlier_mask):
    """Test apply_mask function."""
    # Create a masked array
    masked_array = np.ma.masked_array(
        data=dummy_dem.data, mask=~dummy_inlier_mask.data.astype(bool)
    )
    # Apply the mask
    result = apply_mask(dummy_dem, dummy_inlier_mask)

    # Check that the result is a masked array
    assert isinstance(result, xdem.DEM)
    assert isinstance(result.data, np.ma.MaskedArray)
    assert result.shape == dummy_dem.shape
    assert np.array_equal(result.data.mask, masked_array.mask)


class TestComputeDoD:
    """Tests for the compute_dod function."""

    def test_same_crs(self, sample_dems):
        """Test compute_dod with DEMs having the same CRS."""
        dem1, dem2 = sample_dems

        # Compute DoD
        dod = compute_dod(dem1, dem2)

        # Check that the result is an xdem.DEM object with the expected properties
        assert isinstance(dod, xdem.DEM)
        assert dod.shape == dem2.shape
        assert dod.crs == dem2.crs
        assert dod.transform == dem2.transform

    def test_different_resolution(self, sample_dems):
        """Test compute_dod with DEMs having different resolutions."""
        dem1, dem2 = sample_dems

        # Create a version of dem2 with different resolution
        dem2_lowres = dem2.reproject(res=dem2.res[0] * 2)

        # Test warp_on="reference"
        dod1 = compute_dod(dem1, dem2_lowres)
        assert dod1.shape == dem2_lowres.shape
        assert dod1.res == dem2_lowres.res

        # Test warp_on="dem"
        dod2 = compute_dod(dem1, dem2_lowres, warp_on="dem")
        assert dod2.shape == dem1.shape
        assert dod2.res == dem1.res

    def test_invalid_warp(self, sample_dems):
        """Test compute_dod with invalid warp_on value."""
        dem1, dem2 = sample_dems

        # Reproject dem2 with half the resolution
        dem2_reprojected = dem2.reproject(res=dem2.res[0] * 2)

        # Should raise ValueError for invalid warp_on value
        with pytest.raises(ValueError):
            compute_dod(dem1, dem2_reprojected, warp_on="invalid")


class TestProcessDoD:
    """Tests for the process_dod function."""

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Test nonexistent file paths
        with pytest.raises(FileNotFoundError):
            process_dod("nonexistent.tif", "nonexistent2.tif")

        # Test invalid input types
        with pytest.raises((ValueError, TypeError)):
            process_dod(None, None)

    def test_file_inputs(self):
        """Test using xdem example file paths directly as inputs."""
        dem_path = xdem.examples.get_path("longyearbyen_tba_dem_coreg")
        ref_path = xdem.examples.get_path("longyearbyen_ref_dem")

        # Process DoD using direct file paths
        dod, stats = process_dod(dem=dem_path, reference=ref_path)

        # Check results
        assert isinstance(dod, xdem.DEM)
        assert isinstance(stats, RasterStatistics)

    def test_basic(self, sample_dem_paths, temp_output_dir):
        """Test process_dod with basic inputs."""
        dem_path, ref_path = sample_dem_paths

        # Run process_dod
        dod, stats = process_dod(
            dem=dem_path,
            reference=ref_path,
            output_dir=temp_output_dir,
            output_prefix="test_basic",
        )

        # Check results
        assert isinstance(dod, xdem.DEM)
        assert isinstance(stats, RasterStatistics)

    def test_with_mask(self, sample_dem_paths, sample_mask_path, temp_output_dir):
        """Test process_dod with mask."""
        dem_path, ref_path = sample_dem_paths

        # Run process_dod with mask - updated parameter name from 'mask' to 'inlier_mask'
        dod, stats = process_dod(
            dem=dem_path,
            reference=ref_path,
            inlier_mask=sample_mask_path,  # Updated parameter name
            output_dir=temp_output_dir,
            output_prefix="test_mask",
        )

        # Check results
        assert isinstance(dod, xdem.DEM)
        assert isinstance(stats, RasterStatistics)

    def test_with_plot(self, sample_dem_paths, temp_output_dir):
        """Test process_dod with plot generation."""
        dem_path, ref_path = sample_dem_paths

        # Run process_dod with plot
        dod, stats = process_dod(
            dem=dem_path,
            reference=ref_path,
            output_dir=temp_output_dir,
            output_prefix="test_plot",
            make_plot=True,
            xlim=(-10, 10),
        )

        # Check that computation was successful
        assert isinstance(dod, xdem.DEM)
        assert isinstance(stats, RasterStatistics)

    def test_warp_options(self, sample_dem_paths, temp_output_dir):
        """Test process_dod with different warping options."""
        dem_path, ref_path = sample_dem_paths

        # Test with warp_on="dem" and resampling="nearest"
        dod, stats = process_dod(
            dem=dem_path,
            reference=ref_path,
            output_dir=temp_output_dir,
            output_prefix="warp_test",
            warp_on="dem",
            resampling="nearest",
        )

        # Check results
        assert isinstance(dod, xdem.DEM)
        assert isinstance(stats, RasterStatistics)

    def test_no_output(self, sample_dem_paths):
        """Test process_dod without output directory."""
        dem_path, ref_path = sample_dem_paths

        # Run process_dod without output
        dod, stats = process_dod(dem=dem_path, reference=ref_path)

        # Check that computation was successful but no outputs were saved
        assert isinstance(dod, xdem.DEM)
        assert isinstance(stats, RasterStatistics)


if __name__ == "__main__":
    pytest.main([__file__])

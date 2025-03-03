import os
import tempfile
from pathlib import Path

import geoutils as gu
import numpy as np
import pytest
import xdem

from dempp.dod import (
    apply_mask,
    compute_dod,
    process_dod,
    save_outputs,
)
from dempp.raster_statistics import RasterStatistics


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


# Add test for invalid inputs to process_dod
def test_process_dod_invalid_inputs():
    """Test error handling for invalid inputs."""
    # Test nonexistent file paths
    with pytest.raises(FileNotFoundError):
        process_dod("nonexistent.tif", "nonexistent2.tif")

    # Test invalid input types
    with pytest.raises((ValueError, TypeError)):
        process_dod(None, None)


# Add test for file path inputs
def test_process_dod_file_inputs():
    """Test using xdem example file paths directly as inputs."""
    dem_path = xdem.examples.get_path("longyearbyen_tba_dem_coreg")
    ref_path = xdem.examples.get_path("longyearbyen_ref_dem")

    # Process DoD using direct file paths
    dod, stats, outputs = process_dod(dem=dem_path, reference=ref_path)

    # Check results
    assert isinstance(dod, xdem.DEM)
    assert isinstance(stats, RasterStatistics)
    assert outputs is None  # No output directory provided


# Add test for custom plotting configuration
def test_process_dod_custom_plot_config(sample_dem_paths, temp_output_dir):
    """Test custom plotting configuration."""
    dem_path, ref_path = sample_dem_paths

    # Custom plot configuration
    plt_cfg = {
        "figsize": (12, 12),
        "bins": 100,
        "title": "Custom DoD Plot",
    }

    # Run process_dod with custom plot config
    dod, stats, outputs = process_dod(
        dem=dem_path,
        reference=ref_path,
        output_dir=temp_output_dir,
        output_prefix="custom_plot",
        make_plot=True,
        plt_cfg=plt_cfg,
    )

    # Check that plot was created
    assert "plot" in outputs
    assert outputs["plot"].exists()

    # Verify it's a valid image file
    assert os.path.getsize(outputs["plot"]) > 0


# Add test for different warp_on and resampling parameters
def test_process_dod_warp_options(sample_dem_paths, temp_output_dir):
    """Test process_dod with different warping options."""
    dem_path, ref_path = sample_dem_paths

    # Test with warp_on="dem" and resampling="nearest"
    dod, stats, outputs = process_dod(
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
    assert isinstance(outputs, dict)
    assert "dod" in outputs
    assert outputs["dod"].exists()


# Add test for compute_dod with explicit mask
def test_compute_dod_with_explicit_mask(sample_dems, sample_mask):
    """Test computing DoD with an explicit mask."""
    dem1, dem2 = sample_dems

    # Compute DoD
    dod = compute_dod(dem1, dem2)

    # Apply mask and check statistics
    masked_data = apply_mask(dod, sample_mask)
    stats = RasterStatistics.from_array(masked_data)

    # Basic checks on masked statistics
    assert isinstance(stats, RasterStatistics)
    assert hasattr(stats, "mean")
    assert hasattr(stats, "std")
    assert hasattr(stats, "nmad")


def test_compute_dod_same_crs(sample_dems):
    """Test compute_dod with DEMs having the same CRS."""
    dem1, dem2 = sample_dems

    # Compute DoD
    dod = compute_dod(dem1, dem2)

    # Check that the result is an xdem.DEM object with the expected properties
    assert isinstance(dod, xdem.DEM)
    assert dod.shape == dem2.shape
    assert dod.crs == dem2.crs
    assert dod.transform == dem2.transform


def test_compute_dod_different_resolution(sample_dems):
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


def test_compute_dod_invalid_warp(sample_dems):
    """Test compute_dod with invalid warp_on value."""
    dem1, dem2 = sample_dems

    # Should raise ValueError for invalid warp_on value
    with pytest.raises(ValueError):
        compute_dod(dem1, dem2, warp_on="invalid")


def test_apply_mask(sample_dems, sample_mask):
    """Test apply_mask function."""
    dem1, dem2 = sample_dems
    dod = compute_dod(dem1, dem2)

    # Apply mask
    masked_data = apply_mask(dod, sample_mask)

    # Check result
    assert isinstance(masked_data, np.ma.MaskedArray)
    assert np.sum(~masked_data.mask) < dod.data.size  # Some data should be masked


def test_save_outputs(sample_dems, temp_output_dir):
    """Test save_outputs function."""
    dem1, dem2 = sample_dems
    dod = compute_dod(dem1, dem2)

    # Compute simple statistics
    stats = RasterStatistics.from_array(dod.data)

    # Save outputs
    outputs = save_outputs(dod, stats, temp_output_dir, "test")

    # Check that files were created
    assert (temp_output_dir / "test_dod.tif").exists()
    assert (temp_output_dir / "test_stats.json").exists()

    # Check that returned paths are correct
    assert outputs["dod"] == temp_output_dir / "test_dod.tif"
    assert outputs["stats"] == temp_output_dir / "test_stats.json"


def test_process_dod_basic(sample_dem_paths, temp_output_dir):
    """Test process_dod with basic inputs."""
    dem_path, ref_path = sample_dem_paths

    # Run process_dod
    dod, stats, outputs = process_dod(
        dem=dem_path,
        reference=ref_path,
        output_dir=temp_output_dir,
        output_prefix="test_basic",
    )

    # Check results
    assert isinstance(dod, xdem.DEM)
    assert isinstance(stats, RasterStatistics)
    assert isinstance(outputs, dict)
    assert "dod" in outputs
    assert "stats" in outputs


def test_process_dod_with_mask(sample_dem_paths, sample_mask_path, temp_output_dir):
    """Test process_dod with mask."""
    dem_path, ref_path = sample_dem_paths

    # Run process_dod with mask
    dod, stats, outputs = process_dod(
        dem=dem_path,
        reference=ref_path,
        mask=sample_mask_path,
        output_dir=temp_output_dir,
        output_prefix="test_mask",
    )

    # Check results
    assert isinstance(dod, xdem.DEM)
    assert isinstance(stats, RasterStatistics)
    assert isinstance(outputs, dict)


def test_process_dod_with_plot(sample_dem_paths, temp_output_dir):
    """Test process_dod with plot generation."""
    dem_path, ref_path = sample_dem_paths

    # Run process_dod with plot
    dod, stats, outputs = process_dod(
        dem=dem_path,
        reference=ref_path,
        output_dir=temp_output_dir,
        output_prefix="test_plot",
        make_plot=True,
        xlim=(-10, 10),
    )

    # Check if plot was created
    assert "plot" in outputs
    assert outputs["plot"].exists()

    # Verify it's a valid image file
    assert os.path.getsize(outputs["plot"]) > 0


def test_process_dod_no_output(sample_dem_paths):
    """Test process_dod without output directory."""
    dem_path, ref_path = sample_dem_paths

    # Run process_dod without output
    dod, stats, outputs = process_dod(dem=dem_path, reference=ref_path)

    # Check that computation was successful but no outputs were saved
    assert isinstance(dod, xdem.DEM)
    assert isinstance(stats, RasterStatistics)
    assert outputs is None

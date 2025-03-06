import geoutils as gu
import numpy as np
import pyproj
import pytest
import rasterio as rio
from matplotlib import pyplot as plt

from dempp.statistics import (
    RasterStatistics,
    compute_area_in_mask,
    compute_raster_statistics,
    plot_raster_statistics,
)


@pytest.fixture
def sample_stats() -> RasterStatistics:
    return RasterStatistics(
        mean=1.5,
        median=1.0,
        std=0.5,
        min=-1.0,
        max=4.0,
        percentile25=0.5,
        percentile75=2.5,
        nmad=0.4,
        valid_percentage=95.5,
    )


@pytest.fixture
def sample_raster():
    """Create a sample geoutils.Raster for testing."""
    np.random.seed(42)
    arr = np.random.randint(0, 255, size=(5, 5), dtype="uint8")
    raster_mask = np.random.randint(0, 2, size=(5, 5), dtype="bool")
    ma = np.ma.masked_array(data=arr, mask=raster_mask)

    # Create a raster from array
    return gu.Raster.from_array(
        data=ma,
        transform=rio.transform.from_bounds(0, 0, 1, 1, 5, 5),
        crs=pyproj.CRS.from_epsg(4326),
        nodata=255,
    )


@pytest.fixture
def sample_mask():
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


class TestRasterStatistics:
    def test_create_stats(self, sample_stats):
        assert isinstance(sample_stats, RasterStatistics)
        assert sample_stats.mean == 1.5
        assert sample_stats.valid_percentage == 95.5

    def test_to_dict(self, sample_stats):
        stats_dict = sample_stats.to_dict()
        assert isinstance(stats_dict, dict)
        assert stats_dict["mean"] == 1.5
        assert len(stats_dict) == 9  # Now includes percentile25 and percentile75

    def test_from_dict(self):
        data = {
            "mean": 1.0,
            "median": 0.9,
            "std": 0.5,
            "min": -1.0,
            "max": 3.0,
            "percentile25": 0.5,
            "percentile75": 1.5,
            "nmad": 0.4,
            "valid_percentage": 90.0,
        }
        stats = RasterStatistics.from_dict(data)
        assert isinstance(stats, RasterStatistics)
        assert stats.mean == 1.0

    def test_save_load(self, sample_stats, tmp_path):
        save_path = tmp_path / "stats.json"
        sample_stats.save(save_path)
        assert save_path.exists()

        loaded = RasterStatistics.from_file(save_path)
        assert loaded.mean == sample_stats.mean
        assert loaded.valid_percentage == sample_stats.valid_percentage


class TestComputeRasterStatistics:
    def test_compute_basic(self, sample_raster):
        stats = compute_raster_statistics(sample_raster)
        assert isinstance(stats, RasterStatistics)
        assert isinstance(stats.mean, float)
        assert stats.valid_percentage == -1  # No mask provided

    def test_compute_with_mask(self, sample_raster, sample_mask):
        stats = compute_raster_statistics(sample_raster, inlier_mask=sample_mask)
        assert isinstance(stats, RasterStatistics)
        assert 0 <= stats.valid_percentage <= 100

    def test_compute_with_output(self, sample_raster, tmp_path):
        output_file = tmp_path / "stats.json"
        stats = compute_raster_statistics(sample_raster, output_file=output_file)
        assert output_file.exists()
        assert isinstance(stats, RasterStatistics)

    def test_invalid_input(self):
        # Test that an invalid raster type raises TypeError
        with pytest.raises(TypeError, match="Input raster must be"):
            compute_raster_statistics(np.ones((10, 10)))


class TestComputeAreaInMask:
    def test_compute_area_with_sample_data(self, sample_raster, sample_mask):
        """Test compute_area_in_mask using the sample data from the module."""
        pixels, area, percentage = compute_area_in_mask(sample_raster, sample_mask)

        # There are 6 True values in the sample mask
        assert sample_mask.data.sum() == 6

        # Number of valid pixels should be less than or equal to the mask area
        # The exact number depends on the random mask in sample_raster
        assert pixels <= 6

        # Check area calculation
        expected_area = pixels * sample_raster.res[0] * sample_raster.res[1]
        assert area == expected_area

        # Check percentage calculation
        expected_percentage = pixels / 6  # 6 is the total mask pixels
        assert percentage == expected_percentage

    def test_invalid_inputs(self, sample_raster, sample_mask):
        """Test error handling for invalid inputs."""
        # Test with invalid raster type
        with pytest.raises(TypeError, match="Input raster must be"):
            compute_area_in_mask(np.ones((5, 5)), sample_mask)

        # Test with invalid mask type
        with pytest.raises(TypeError, match="Input mask must be"):
            compute_area_in_mask(sample_raster, np.ones((5, 5), dtype=bool))


class TestPlotRasterStatistics:
    def test_basic_plot(self, sample_raster):
        fig, (ax1, ax2) = plot_raster_statistics(sample_raster.data)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_plot_with_stats(self, sample_raster, sample_stats):
        fig, (ax1, ax2) = plot_raster_statistics(sample_raster.data, stats=sample_stats)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_save(self, sample_raster, tmp_path):
        output_file = tmp_path / "plot.png"
        fig, _ = plot_raster_statistics(sample_raster.data, output_file=output_file)
        assert output_file.exists()
        plt.close(fig)

    def test_plot_configuration(self, sample_raster):
        # Test that configuration parameters are properly passed
        fig_cfg = {"figsize": (12, 8)}
        hist_cfg = {"bins": 30, "color": "red"}

        fig, _ = plot_raster_statistics(
            sample_raster.data, fig_cfg=fig_cfg, hist_cfg=hist_cfg
        )

        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 8
        plt.close(fig)

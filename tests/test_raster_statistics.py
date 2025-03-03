import numpy as np
import pytest
from matplotlib import pyplot as plt

from dempp.raster_statistics import (
    RasterStatistics,
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
        nmad=0.4,
        valid_percentage=95.5,
    )


@pytest.fixture
def sample_raster():
    data = np.random.normal(0, 1, (100, 100))
    mask = np.random.choice([True, False], (100, 100), p=[0.1, 0.9])
    return np.ma.masked_array(data, mask=mask)


class TestRasterStatistics:
    def test_create_stats(self, sample_stats):
        assert isinstance(sample_stats, RasterStatistics)
        assert sample_stats.mean == 1.5
        assert sample_stats.valid_percentage == 95.5

    def test_to_dict(self, sample_stats):
        stats_dict = sample_stats.to_dict()
        assert isinstance(stats_dict, dict)
        assert stats_dict["mean"] == 1.5
        assert len(stats_dict) == 7

    def test_from_dict(self):
        data = {
            "mean": 1.0,
            "median": 0.9,
            "std": 0.5,
            "min": -1.0,
            "max": 3.0,
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
        assert 0 <= stats.valid_percentage <= 100
        assert isinstance(stats.mean, float)

    def test_compute_with_mask(self, sample_raster):
        mask = np.ones_like(sample_raster.data, dtype=bool)
        mask[25:75, 25:75] = False
        stats = compute_raster_statistics(sample_raster, mask=mask)
        assert isinstance(stats, RasterStatistics)

    def test_compute_with_output(self, sample_raster, tmp_path):
        output_file = tmp_path / "stats.json"
        stats = compute_raster_statistics(sample_raster, output_file=output_file)
        assert output_file.exists()
        assert isinstance(stats, RasterStatistics)


class TestPlotRasterStatistics:
    def test_basic_plot(self, sample_raster):
        fig, (ax1, ax2) = plot_raster_statistics(sample_raster)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_plot_with_stats(self, sample_raster, sample_stats):
        fig, (ax1, ax2) = plot_raster_statistics(sample_raster, stats=sample_stats)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_save(self, sample_raster, tmp_path):
        output_file = tmp_path / "plot.png"
        fig, _ = plot_raster_statistics(sample_raster, output_file=output_file)
        assert output_file.exists()
        plt.close(fig)

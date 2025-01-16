import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import dask.array as da
import geopandas as gpd
import geoutils as gu
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xdem

logger = logging.getLogger("pyasp")


@dataclass()
class RasterStatistics:
    """Container for raster statistics with strict typing."""

    mean: float
    std: float
    min: float
    max: float
    percentile25: float
    median: float
    percentile75: float
    nmad: float
    valid_percentage: float

    def to_dict(self) -> dict[str, float]:
        """Convert statistics to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RasterStatistics":
        """Create RasterStatistics from dictionary."""
        return cls(**data)

    @classmethod
    def from_file(cls, input_file: Path) -> "RasterStatistics":
        """Load statistics from a JSON file."""
        return load_stats_from_file(input_file)

    def __str__(self) -> str:
        """Return string representation of the statistics."""
        return "\n".join(
            [f"{key}: {value:.2f}" for key, value in self.to_dict().items()]
        )

    def __repr__(self) -> str:
        """Return string representation of the statistics."""
        return self.__str__()

    def save(self, output_file: Path) -> None:
        """Save statistics to a JSON file."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_stats_to_file(self, output_file)


def compute_raster_statistics(
    raster: xdem.DEM | np.ma.MaskedArray,
    mask: gu.Mask | np.ma.MaskedArray | None = None,
    output_file: Path | None = None,
) -> dict[str, float]:
    """Compute statistics for a raster.
    Args:
        raster (xdem.DEM | np.ma.MaskedArray): DEM raster or masked array.
        mask (gu.Mask | None, optional): Inlier mask where statistics are to be computed.
        output_file (Path | None, optional): Path to save statistics JSON file.

    Returns:
        RasterStatistics: Container with computed statistics.
    """
    if isinstance(raster, xdem.DEM):
        raster = raster.data
    elif not isinstance(raster, np.ma.MaskedArray):
        raise TypeError("Input raster must be a masked array or xdem.DEM object.")

    if isinstance(mask, gu.Mask):
        mask = mask.data
    elif not isinstance(mask, np.ma.MaskedArray):
        logger.warning("Invalid mask provided. Using all pixels.")
        mask = None
    if mask is None:
        mask = np.ones_like(raster, dtype=bool)

    # Convert masked array to Dask array
    raster_dask = da.from_array(raster[mask].compressed(), chunks="auto")

    # Compute statistics in parallel
    mean = da.mean(raster_dask)
    std = da.std(raster_dask)
    min_val = da.min(raster_dask)
    max_val = da.max(raster_dask)
    percentile25 = da.percentile(raster_dask, 25)
    median = da.percentile(raster_dask, 50)
    percentile75 = da.percentile(raster_dask, 75)

    # Compute the number of empty cells and percentage of valid cells
    empty_cells = raster.mask.sum()
    total_cells = raster.size
    valid_cells_percentage = (1 - empty_cells / total_cells) * 100

    # Compute the statistics
    stats = RasterStatistics(
        mean=mean.compute(),
        std=std.compute(),
        min=min_val.compute(),
        max=max_val.compute(),
        percentile25=percentile25.compute()[0],
        median=median.compute()[0],
        percentile75=percentile75.compute()[0],
        nmad=xdem.spatialstats.nmad(raster_dask),
        valid_percentage=round(valid_cells_percentage, 2),
    )

    if output_file is not None:
        stats.save(output_file)

    return stats


def plot_raster_statistics(
    raster: np.ma.MaskedArray,
    output_file: Path | str | None = None,
    stats: RasterStatistics = None,
    xlim: tuple[float, float] | None = None,
    fig_cfg: dict | None = None,
    ax_cfg: dict | None = None,
    hist_cfg: dict | None = None,
    box_cfg: dict | None = None,
    annotate_cfg: dict | None = None,
    save_cfg: dict | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot histogram with KDE and boxplot of a r.

    Args:
        raster (np.ma.MaskedArray): Masked array of.
        output_file (Path | str | None, optional): Path to save the output figure. Defaults to None.
        stats (RasterStatistics | None, optional): Statistics object to annotate on the plot. Defaults to None.
        xlim (tuple[float, float] | None, optional): Tuple of (min, max) to set x-axis limits. Defaults to None.
        fig_cfg (dict | None, optional): Configuration for plt.subplots() call. Defaults to None.
        ax_cfg (dict | None, optional): Configuration for axes (xticks, grid, etc.). Defaults to None.
        hist_cfg (dict | None, optional): Configuration for sns.histplot(). Defaults to None.
        box_cfg (dict | None, optional): Configuration for sns.boxplot(). Defaults to None.
        annotate_cfg (dict | None, optional): Configuration for statistics annotation. Defaults to None.
        save_cfg (dict | None, optional): Configuration for fig.savefig(). Defaults to None.

    Returns:
        tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]: Tuple containing:
            - The generated figure.
            - Tuple of axes containing the histogram and boxplot.
    """
    # Set default parameters for the figure and plots
    fig_params = {
        "figsize": (10, 10),
        "sharex": True,
    }
    ax_params = {
        "xticks": None,
        "grid": False,
    }
    hist_params = {
        "bins": 50,
        "kde": True,
        "stat": "density",
        "edgecolor": "black",
    }
    box_params = {
        "orient": "h",
        "width": 0.5,
        "color": "skyblue",
    }
    annotate_params = {
        "fontsize": 10,
    }
    save_params = {
        "dpi": 300,
    }

    # Extract specific kwargs for different components and update defaults
    if fig_cfg is not None:
        fig_params.update(fig_cfg)
    if ax_cfg is not None:
        ax_params.update(ax_cfg)
    if hist_cfg is not None:
        hist_params.update(hist_cfg)
    if box_cfg is not None:
        box_params.update(box_cfg)
    if annotate_cfg is not None:
        annotate_params.update(annotate_cfg)
    if save_cfg is not None:
        save_params.update(save_cfg)

    # Create figure with Histogram and Boxplot
    fig, (ax1, ax2) = plt.subplots(2, 1, **fig_params)

    # Histogram with KDE on the first axis
    sns.histplot(raster.compressed(), ax=ax1, **hist_params)
    ax1.set_title("Histogram of Differences with KDE")
    ax1.set_ylabel("Density")

    # Boxplot on the second axis
    sns.boxplot(x=raster.compressed(), ax=ax2, **box_params)
    ax2.set_title("Boxplot of Differences")
    ax2.set_xlabel("Height Difference [m]")

    # Set x-axis limits if provided
    if xlim is not None:
        ax1.set_xlim(xlim)
        ax2.set_xlim(xlim)

    # Set custom xticks if provided
    if xticks := ax_params.pop("xticks", None):
        ax1.set_xticks(xticks)
        ax2.set_xticks(xticks)

    if grid := ax_params.pop("grid", False):
        ax1.grid(grid)
        ax2.grid(grid)

    # Add statistics to the plot
    if stats is not None:
        stats_text = "\n".join(
            [f"{key}: {value:.2f}" for key, value in stats.to_dict().items()]
        )
        plt.figtext(
            0.8,
            0.8,
            stats_text,
            bbox=dict(facecolor="white", alpha=0.5),
            **annotate_params,
        )

    # Adjust layout and save the figure
    fig.tight_layout()

    if output_file is not None:
        fig.savefig(output_file, **save_params)

    return fig, (ax1, ax2)


def save_stats_to_file(
    stats: RasterStatistics,
    output_file: Path,
    float_precision: int = 3,
) -> None:
    """Save statistics to a JSON file.

    Args:
        stats (RasterStatistics): Statistics object to save
        output_file (Path): Path to save the JSON file
        float_precision (int, optional): Number of decimal places for float values. Defaults to 3.
    """

    def _format_value(value: Any, float_precision: int) -> Any:
        """Format single value with proper type conversion."""
        if isinstance(value, (np.integer | np.floating)):
            value = value.item()
        if isinstance(value, float):
            return str(round(value, float_precision))
        return value

    def _format_dict_recursive(data: dict, float_precision: int) -> dict:
        """Recursively format dictionary values."""
        formatted = {}
        for key, value in data.items():
            if isinstance(value, dict):
                formatted[key] = _format_dict_recursive(value, float_precision)
            else:
                formatted[key] = _format_value(value, float_precision)
        return formatted

    output_file = Path(output_file)
    if output_file.suffix != ".json":
        output_file = output_file.with_suffix(".json")
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)

    # Convert to dict and format floats
    stats_dict = stats.to_dict()
    formatted_stats = _format_dict_recursive(stats_dict, float_precision)

    with open(output_file, "w") as f:
        json.dump(formatted_stats, f, indent=4)


def load_stats_from_file(input_file: Path) -> RasterStatistics:
    """Load statistics from a JSON file.

    Args:
        input_file (Path): Path to the JSON file containing statistics.

    Returns:
        RasterStatistics: Statistics object loaded from file.

    Raises:
        JSONDecodeError: If file contains invalid JSON
        KeyError: If JSON is missing required statistics fields
    """
    with open(input_file) as f:
        loaded_stats = json.load(f)

    loaded_stats = {key: float(value) for key, value in loaded_stats.items()}

    return RasterStatistics.from_dict(loaded_stats)


def compute_valid_pixel_in_polygon(
    raster: Path | str | gu.Raster,
    polygon: Path | str | gu.Vector,
    satellite_footprint: Path | str | gu.Vector = None,
) -> tuple[int, float, float]:
    """Count number of valid pixels in raster within given polygon.

    Args:
        raster (Path | str | gu.Raster): Path to raster file or geoutils.Raster object.
        polygon (Path | str | gu.Vector): Path to vector file or geoutils.Vector object.
        satellite_footprint (Path | str | gu.Vector, optional): Path to satellite footprint file or geoutils.Vector object. Defaults to None.

    Returns:
        tuple[int, float, float]: Tuple containing:
            - valid_pixels: Number of valid pixels in raster within polygon.
            - valid_area_m2: Area of valid pixels in square meters.
            - valid_area_perc: Percentage of valid pixels in the polygon.
    """
    # Load data if paths provided
    if isinstance(raster, Path | str):
        if not Path(raster).exists():
            raise FileNotFoundError(f"Raster file not found: {raster}")
        raster = gu.Raster(raster)
    if isinstance(polygon, Path | str):
        if not Path(polygon).exists():
            raise FileNotFoundError(f"Polygon file not found: {polygon}")
        polygon = gu.Vector(polygon)
    if satellite_footprint is not None:
        if isinstance(satellite_footprint, (Path | str)):
            if not Path(satellite_footprint).exists():
                raise FileNotFoundError(
                    f"Satellite footprint file not found: {satellite_footprint}"
                )
            satellite_footprint = gu.Vector(satellite_footprint)
        else:
            logger.warning("Invalid satellite footprint provided. Using raster extent.")
            satellite_footprint = None

    # Reproject polygon to raster CRS
    polygon = polygon.reproject(crs=raster.crs)

    # Intersect polygon with satellite footprint if provided, else crop to raster extent
    if satellite_footprint is not None:

        def extract_glaciers_in_footprint(glacier_gdf, footprint_gdf):
            # Ensure same CRS
            if glacier_gdf.crs != footprint_gdf.crs:
                footprint_gdf = footprint_gdf.to_crs(glacier_gdf.crs)

            # Get footprint polygon (assuming single polygon in footprint_gdf)
            footprint = footprint_gdf.geometry.iloc[0]

            # Find intersecting glaciers
            intersecting_glaciers = glacier_gdf[glacier_gdf.intersects(footprint)]

            # Get actual intersection geometry
            intersected_glaciers = gpd.overlay(
                intersecting_glaciers, footprint_gdf, how="intersection"
            )

            return intersected_glaciers

        satellite_footprint = satellite_footprint.reproject(crs=raster.crs)
        polygon_ds = extract_glaciers_in_footprint(polygon.ds, satellite_footprint.ds)
        polygon = gu.Vector(polygon_ds)
    else:
        logger.info("No satellite footprint provided. Using the raster extent.")
        polygon = polygon.crop(raster, clip=True)

    # Create binary mask from polygon
    mask = polygon.create_mask(raster)

    # Apply mask to data
    masked_data = raster[mask]

    # Count valid pixels (non-nodata values)
    valid_pixels = (~np.isnan(masked_data)).sum().astype(int)

    # Convert in metric unit
    try:
        valid_area_m2 = valid_pixels * raster.res[0] * raster.res[1]

        # Compute the percentage of valid pixels in the polygon
        valid_area_perc = valid_area_m2 / polygon.area.sum()

    except AttributeError:
        logger.warning("Could not compute valid area")
        valid_area_m2 = -1
        valid_area_perc = -1

    return valid_pixels, valid_area_m2, valid_area_perc

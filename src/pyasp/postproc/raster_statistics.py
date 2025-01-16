import json
import logging
from pathlib import Path
from typing import Any

import dask.array as da
import geopandas as gpd
import geoutils as gu
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xdem
from dask import visualize

logger = logging.getLogger("pyasp")


def save_stats_to_file(
    diff_stats: dict[str, Any],
    output_file: Path,
    float_precision: int = 4,
) -> None:
    """Save statistics to a JSON file.

    Args:
        diff_stats (dict[str, Any]): Dictionary of statistics to save.
        output_file (Path): Path where to save the JSON file.
        float_precision (int, optional): Number of decimal places for float values. Defaults to 4.
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

    formatted_stats = _format_dict_recursive(diff_stats, float_precision)

    with open(output_file, "w") as f:
        json.dump(formatted_stats, f, indent=4)

    logger.info(f"Statistics written to {output_file}")


def load_stats_from_file(input_file: Path) -> dict[str, Any]:
    """Load statistics from a JSON file.

    Args:
        input_file (Path): Path to the JSON file containing statistics.

    Returns:
        dict[str, Any]: Dictionary of loaded statistics.
    """
    with open(input_file) as f:
        loaded_stats = json.load(f)
    return loaded_stats


def compute_raster_statistics(
    raster: np.ma.MaskedArray,
    mask: gu.Mask | None = None,
    output_file: Path | None = None,
    graph_output_file: Path | None = None,
) -> dict[str, float]:
    """Compute statistics for a DEM difference raster.

    Args:
        raster (np.ma.MaskedArray): The difference raster as a masked array.
        mask (gu.Mask | None, optional): Inlier mask where statistics are to be computed. Defaults to None.
        output_file (Path | None, optional): Path to save statistics JSON file. Defaults to None.
        graph_output_file (Path | None, optional): Path to save computation graph visualization. Defaults to None.

    Returns:
        dict[str, float]: Dictionary containing computed statistics:
            - mean: Mean difference
            - median: Median difference
            - std: Standard deviation
            - min: Minimum value
            - max: Maximum value
            - nmad: Normalized median absolute deviation
            - valid_percentage: Percentage of valid cells
    """
    if mask is None:
        mask = np.ones_like(raster, dtype=bool)

    # Convert masked array to Dask array
    raster_dask = da.from_array(raster[mask].compressed(), chunks="auto")

    # Compute statistics in parallel
    mean = da.mean(raster_dask)
    median = da.percentile(raster_dask, 50)
    std = da.std(raster_dask)
    min_val = da.min(raster_dask)
    max_val = da.max(raster_dask)
    nmad = da.map_blocks(xdem.spatialstats.nmad, raster_dask)

    if graph_output_file is not None:
        visualize(mean, median, std, min_val, max_val, nmad, filename=graph_output_file)

    # Compute the number of empty cells and percentage of valid cells
    empty_cells = raster.data.mask.sum()
    total_cells = raster.data.size
    valid_cells_percentage = (1 - empty_cells / total_cells) * 100

    # Compute the statistics
    raster_stats = {
        "mean": mean.compute(),
        "median": median.compute()[0],
        "std": std.compute(),
        "min": min_val.compute(),
        "max": max_val.compute(),
        "nmad": nmad.compute(),
        "valid_percentage": round(valid_cells_percentage, 2),
    }

    # Save the statistics to a file
    if output_file is not None:
        save_stats_to_file(raster_stats, output_file)

    return raster_stats


def plot_raster_statistics(
    raster: np.ma.MaskedArray,
    stats: dict[str, float],
    output_file: Path | str | None = None,
    xlim: tuple[float, float] | None = None,
    fig_cfg: dict | None = None,
    ax_cfg: dict | None = None,
    hist_cfg: dict | None = None,
    box_cfg: dict | None = None,
    annotate_cfg: dict | None = None,
    save_cfg: dict | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot histogram with KDE and boxplot of DEM differences.

    Args:
        raster (np.ma.MaskedArray): Masked array of DEM differences.
        stats (dict[str, float]): Dictionary of computed statistics.
        output_file (Path | str | None, optional): Path to save the output figure. Defaults to None.
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
    stats_text = "\n".join([f"{key}: {value:.2f}" for key, value in stats.items()])
    plt.figtext(
        0.8,
        0.85,
        stats_text,
        bbox=dict(facecolor="white", alpha=0.5),
        **annotate_params,
    )

    # Adjust layout and save the figure
    fig.tight_layout()

    if output_file is not None:
        fig.savefig(output_file, **save_params)

    return fig, (ax1, ax2)


def compute_dod_stats(
    dem: Path | xdem.DEM,
    reference: Path | xdem.DEM,
    mask: gu.Mask | Path | None = None,
    output_dir: Path = None,
    figure_path: Path | None = None,
    resampling: str = "bilinear",
    skip_plot: bool = False,
    xlim: tuple[float, float] | None = None,
    plt_cfg: dict | None = None,
) -> tuple[dict[str, float], Path, Path]:
    """Compute difference between two DEMs, calculate statistics and generate plots.

    Args:
        dem (Path | xdem.DEM): Path to the DEM file or xdem.DEM object to process.
        reference (Path | xdem.DEM): Path to the reference DEM file or xdem.DEM object.
        output_dir (Path, optional): Directory where to save output statistics and plots. Defaults to None.
        mask (gu.Mask | Path | None, optional): Optional mask for stable areas, can be a Path or gu.Mask. Defaults to None.
        resampling (str, optional): Resampling method for reprojection. Defaults to "bilinear".
        xlim (tuple[float, float] | None, optional): X-axis limits for the plots as (min, max). Defaults to None.
        plt_cfg (dict | None, optional): Configuration dictionary for plots. Defaults to None.

    Returns:
        tuple[dict[str, float], Path, Path]: Tuple containing:
            - Dictionary of computed statistics.
            - Path to the saved statistics file.
            - Path to the saved plot file.

    Raises:
        ValueError: If input DEMs are not valid or compatible.
        FileNotFoundError: If input paths don't exist.
    """
    logger.info("Processing DEM pair")

    # Load DEMs if paths are provided
    if isinstance(dem, Path):
        if not dem.exists():
            raise FileNotFoundError(f"DEM file not found: {dem}")
        dem = xdem.DEM(dem)
    if isinstance(reference, Path):
        if not reference.exists():
            raise FileNotFoundError(f"Reference DEM file not found: {reference}")
        reference = xdem.DEM(reference)

    # Validate inputs
    if not isinstance(dem, xdem.DEM) or not isinstance(reference, xdem.DEM):
        raise ValueError("Both dem and reference must be Path or xdem.DEM objects")

    # Compute the difference
    compare = dem.reproject(reference, resampling=resampling)
    diff = reference - compare
    logger.info("Computed difference between DEMs")

    # Apply mask if provided
    if mask is not None:
        if isinstance(mask, Path):
            if not mask.exists():
                raise FileNotFoundError(f"Mask file not found: {mask}")
            mask = gu.Mask(mask)
            mask.set_area_or_point("Point")
        elif not isinstance(mask, gu.Mask):
            raise ValueError("Mask must be a Path or geoutils.Mask object")
        mask_warped = mask.reproject(reference)
        diff_masked = diff[mask_warped]
        logger.info("Applied mask to difference")
    else:
        diff_masked = diff
        mask_warped = None

    # Compute statistics
    diff_stats = compute_raster_statistics(raster=diff, mask=mask_warped)

    # Make plots
    if not skip_plot:
        # Define output path
        if figure_path is not None:
            figure_path = Path(figure_path)
        else:
            if output_dir is None:
                output_dir = Path.cwd()
            output_dir = Path(output_dir)
            try:
                output_stem = Path(dem.filename).stem
            except (AttributeError, TypeError):
                logger.warning("Unable to get DEM filename. Using default name")
                output_stem = "stats"
            figure_path = output_dir / f"{output_stem}_diff_plot.png"
        figure_path.parent.mkdir(parents=True, exist_ok=True)

        if plt_cfg is None:
            plt_cfg = {}
        plot_raster_statistics(
            diff_masked,
            diff_stats,
            output_file=figure_path,
            xlim=xlim,
            **plt_cfg,
        )
        logger.debug(f"Saved plot: {figure_path}")

    return diff_stats


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

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import dask.array as da
import geoutils as gu
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xdem
from dask import visualize
from tqdm import tqdm

logger = logging.getLogger("pyasp")


def load_dem(
    dem_path: Path,
    area_or_point: str | None = None,
    vrcs: str | None = None,
) -> xdem.DEM:
    """Load DEM from disk and set its properties.

    Args:
        dem_path: Path to the DEM file
        area_or_point: Set area_or_point property of DEM ('area' or 'point')
        vrcs: Vertical reference coordinate system to set

    Returns:
        Loaded DEM object

    Raises:
        ValueError: If area_or_point is not 'area' or 'point'
        FileNotFoundError: If DEM file doesn't exist
    """
    dem = xdem.DEM(dem_path)
    if area_or_point:
        if area_or_point not in ["area", "point"]:
            raise ValueError(
                "Invalid value for area_or_point. Must be 'area' or 'point'."
            )
        dem.set_area_or_point(area_or_point)
    if vrcs:
        dem.set_vcrs(vrcs)

    return dem


def load_dems(
    paths: list[Path],
    parallel: bool = True,
    num_threads: int = None,
    exclude_duplicates: bool = False,
) -> list[xdem.DEM]:
    """Load DEMs from disk, either in parallel or sequentially.

    Args:
        paths: List of paths to DEM files
        parallel: If True, load DEMs in parallel using threading
        num_threads: Number of threads to use for parallel loading
        exclude_duplicates: If True, only load unique paths

    Returns:
        List of loaded DEMs in the same order as input paths
    """
    paths = [Path(path) for path in paths]

    if exclude_duplicates:
        paths = list(dict.fromkeys(paths))  # Preserve order while removing duplicates

    if not parallel:
        return [load_dem(path) for path in tqdm(paths, desc="Loading DEMs")]

    if num_threads is None:
        num_threads = min(len(paths), 8)

    results = [None] * len(paths)

    # Store all indices for each path to handle duplicates
    path_to_indices = {}
    for idx, path in enumerate(paths):
        if path in path_to_indices:
            path_to_indices[path].append(idx)
        else:
            path_to_indices[path] = [idx]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        unique_paths = paths if exclude_duplicates else set(paths)
        futures = {executor.submit(load_dem, path): path for path in unique_paths}

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Loading DEMs"
        ):
            path = futures[future]
            indices = path_to_indices[path]
            try:
                dem = future.result()
                logging.debug(f"Loaded DEM: {path.name}, {dem}")
                # Assign the same DEM to all duplicate indices
                for idx in indices:
                    results[idx] = dem
            except Exception as e:
                logging.error(f"Error loading DEM from {path}: {e}")
                for idx in indices:
                    results[idx] = None

    return results


def save_dem(
    dem: xdem.DEM,
    save_path: Path,
    dtype: str | np.dtype = "float32",
    compress: str = "LZW",
    **kwargs: Any,
) -> None:
    """Save DEM to disk.

    Args:
        dem: DEM object to save
        save_path: Path where to save the DEM
        dtype: Data type for the saved DEM
        compress: Compression method to use
        **kwargs: Additional arguments passed to xdem.DEM.save()
    """
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)

    if dem.data.dtype == "float64":
        dem = dem.astype(dtype)

    dem.save(save_path, dtype=dtype, compress=compress, **kwargs)


def save_stats_to_file(
    diff_stats: dict[str, Any],
    output_file: Path,
    float_precision: int = 4,
) -> None:
    """Save statistics to a JSON file.

    Args:
        diff_stats: Dictionary of statistics to save
        output_file: Path where to save the JSON file
        float_precision: Number of decimal places for float values
    """
    output_file = Path(output_file)
    if output_file.suffix != ".json":
        output_file = output_file.with_suffix(".json")
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)

    formatted_stats = {
        k: str(round(v, float_precision)) if isinstance(v, float | np.float32) else v
        for k, v in diff_stats.items()
    }

    with open(output_file, "w") as f:
        json.dump(formatted_stats, f, indent=4)

    logger.info(f"Statistics written to {output_file}")


def load_stats_from_file(input_file: Path) -> dict[str, Any]:
    """Load statistics from a JSON file.

    Args:
        input_file: Path to the JSON file containing statistics

    Returns:
        Dictionary of loaded statistics
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
        raster: The difference raster as a masked array
        mask: Inlier mask where statistics are to be computed
        output_file: Path to save statistics JSON file
        graph_output_file: Path to save computation graph visualization

    Returns:
        Dictionary containing computed statistics:
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
    output_file: Path,
    xlim: tuple[float, float] | None = None,
    fig_cfg: dict = {},
    ax_cfg: dict = {},
    hist_cfg: dict = {},
    box_cfg: dict = {},
    annotate_cfg: dict = {},
    save_cfg: dict = {},
) -> None:
    """Plot histogram with KDE and boxplot of DEM differences.

    Args:
        raster: Masked array of DEM differences
        stats: Dictionary of computed statistics
        output_file: Path to save the output figure
        xlim: Tuple of (min, max) to set x-axis limits
        fig_cfg: Configuration for plt.subplots() call
        ax_cfg: Configuration for axes (xticks, grid, etc.)
        hist_cfg: Configuration for sns.histplot()
        box_cfg: Configuration for sns.boxplot()
        annotate_cfg: Configuration for statistics annotation
        save_cfg: Configuration for fig.savefig()
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
    fig_params.update(fig_cfg)
    ax_params.update(ax_cfg)
    hist_params.update(hist_cfg)
    box_params.update(box_cfg)
    annotate_params.update(annotate_cfg)
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
    fig.savefig(output_file, **save_params)


def compute_dem_difference_stats(
    dem: Path | xdem.DEM,
    reference: Path | xdem.DEM,
    output_dir: Path,
    mask: gu.Mask | None = None,
    resampling: str = "bilinear",
    xlim: tuple[float, float] | None = None,
    plt_cfg: dict | None = None,
) -> tuple[dict[str, float], Path, Path]:
    """Compute difference between two DEMs, calculate statistics and generate plots.

    Args:
        dem: Path to the DEM file or xdem.DEM object to process
        reference: Path to the reference DEM file or xdem.DEM object
        output_dir: Directory where to save output statistics and plots
        mask: Optional mask for stable areas
        resampling: Resampling method for reprojection
        xlim: X-axis limits for the plots as (min, max)
        plt_cfg: Configuration dictionary for plots

    Returns:
        Tuple containing:
            - Dictionary of computed statistics
            - Path to the saved statistics file
            - Path to the saved plot file

    Raises:
        ValueError: If input DEMs are not valid or compatible
        FileNotFoundError: If input paths don't exist
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

    # Create output file paths
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = dem.filename.stem if hasattr(dem, "filename") else "diff"
    stats_file = output_dir / f"{output_stem}_diff_stats.json"
    plot_file = output_dir / f"{output_stem}_diff_plot.png"
    logger.debug(f"Output files: {stats_file}, {plot_file}")

    # Compute the difference
    compare = dem.reproject(reference, resampling=resampling)
    diff = reference - compare
    logger.info("Computed difference between DEMs")

    # Apply mask if provided
    if mask is not None:
        if not isinstance(mask, gu.Mask):
            raise ValueError("Mask must be a geoutils.Mask object")
        mask_warped = mask.reproject(reference)
        diff_masked = diff[mask_warped]
        logger.info("Applied mask to difference")
    else:
        diff_masked = diff
        mask_warped = None

    # Compute statistics
    diff_stats = compute_raster_statistics(
        raster=diff, mask=mask_warped, output_file=stats_file
    )
    logger.debug(f"Computed statistics: {diff_stats}")

    # Make plots
    plt_cfg = plt_cfg or {}
    plot_raster_statistics(
        diff_masked,
        diff_stats,
        plot_file,
        xlim=xlim,
        **plt_cfg,
    )
    logger.info(f"Saved plot: {plot_file}")

    return diff_stats, stats_file, plot_file

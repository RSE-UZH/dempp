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
    area_or_point: str = None,
    vrcs: str = None,
) -> xdem.DEM:
    """Load DEM from disk"""
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
    dtype: str | Any = "float32",
    compress="LZW",
    **kwargs,
) -> None:
    """Save DoD to disk"""
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)

    if dem.data.dtype == "float64":
        dem = dem.astype(dtype)

    dem.save(save_path, dtype=dtype, compress=compress, **kwargs)


def save_stats_to_file(diff_stats, output_file, float_precision=4):
    """Save statistics to a JSON file."""

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


def load_stats_from_file(input_file: Path) -> dict:
    """Load statistics from a JSON file."""
    with open(input_file) as f:
        loaded_stats = json.load(f)
    return loaded_stats


def compute_raster_statistics(
    raster: np.ma.MaskedArray,
    mask: gu.Mask = None,
    output_file: Path = None,
    graph_output_file: Path = None,
) -> dict:
    """
    Compute statistics for a DEM difference raster.

    Parameters:
        raster (np.ma.MaskedArray): The difference raster as a masked array.
        mask (gu.Mask, optional): Inlier mask where statistics are to be computed. Defaults to None.
        output_file (Path, optional): Path to save statistics. Defaults to None.

    Returns:
        dict: A dictionary of computed statistics.
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


def plot_raster_statistics(raster, stats, output_file):
    """
    Plot histogram with KDE and boxplot of DEM differences and save the figure.

    Parameters:
        raster (np.ma.MaskedArray): Masked array of DEM differences.
        stats (dict): Dictionary of computed statistics.
        output_file (Path): Path to save the output figure.
    """
    # Create figure with Histogram and Boxplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Histogram with KDE on the first axis
    sns.histplot(
        raster.compressed(),
        bins=50,
        kde=True,
        stat="density",
        ax=ax1,
        edgecolor="black",
    )
    ax1.set_title("Histogram of Differences with KDE")
    ax1.set_ylabel("Density")

    # Boxplot on the second axis
    sns.boxplot(x=raster.compressed(), ax=ax2, orient="h")
    ax2.set_title("Boxplot of Differences")
    ax2.set_xlabel("Height Difference [m]")

    # Add statistics to the plot
    stats_text = "\n".join([f"{key}: {value:.2f}" for key, value in stats.items()])
    plt.figtext(
        0.8, 0.85, stats_text, fontsize=10, bbox=dict(facecolor="white", alpha=0.5)
    )

    # Adjust layout and save the figure
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)

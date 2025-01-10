import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import xdem
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

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import geoutils as gu
import numpy as np
import xdem
from tqdm import tqdm

logger = logging.getLogger("pyasp")


def load_dem(
    dem_path: Path,
    area_or_point: str = "Area",
    vrcs: str = None,
) -> xdem.DEM:
    """Load DEM from disk"""
    dem = xdem.DEM(dem_path)
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
    formatted_stats = {
        k: str(round(v, float_precision)) if isinstance(v, (float, np.float32)) else v
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


def compute_dod(
    dem: xdem.DEM,
    reference: xdem.DEM,
    warp_on: str = "reference",
    resampling="bilinear",
) -> xdem.DEM:
    """
    Compute the difference between two DEMs, reprojecting if necessary.

    Parameters:
        dem (xdem.DEM): The DEM to compare.
        reference (xdem.DEM): The reference DEM.
        warp_on (str, optional): The DEM to reproject. Must be 'reference' or 'dem'. Defaults to 'reference'.
        resampling (str, optional): Resampling method for reprojection. Defaults to 'bilinear'.

    Returns:
        xdem.DEM: The difference DEM (reference - dem).

    Raises:
        ValueError: If DEMs have mismatched dimensions after reprojection or incompatible CRS.
    """
    # Check if the DEMs have the same CRS
    if dem.crs != reference.crs or dem.data.shape != reference.data.shape:
        logger.info("Reprojecting DEM to match the reference DEM...")
        if warp_on == "reference":
            dem = dem.reproject(reference, resampling=resampling)
        elif warp_on == "dem":
            dem = reference.reproject(dem, resampling=resampling)
        else:
            raise ValueError("Invalid `warp_on` value. Must be 'reference' or 'dem'.")

    # Compute the difference
    diff = reference - dem

    return diff


def compute_dod_statistics(
    diff: np.ma.MaskedArray,
    stable_mask: gu.Mask = None,
    output_file: Path = None,
) -> dict:
    """
    Compute statistics for a DEM difference raster.

    Parameters:
        diff (np.ma.MaskedArray): The difference raster as a masked array.
        stable_mask (gu.Mask, optional): Mask for stable (non-glacierized) areas. Defaults to None.
        output_file (Path, optional): Path to save statistics. Defaults to None.

    Returns:
        dict: A dictionary of computed statistics.
    """
    if stable_mask is None:
        stable_mask = np.ones_like(diff, dtype=bool)

    # Compute the number of empty cells and percentage of valid cells
    empty_cells = diff.data.mask.sum()
    total_cells = diff.data.size
    valid_cells_percentage = (1 - empty_cells / total_cells) * 100

    # Compute statistics of the differences
    diff_stats = {
        "mean": np.ma.mean(diff[stable_mask]),
        "median": np.ma.median(diff[stable_mask]),
        "std": np.ma.std(diff[stable_mask]),
        "min": np.ma.min(diff[stable_mask]),
        "max": np.ma.max(diff[stable_mask]),
        "nmad": xdem.spatialstats.nmad(diff[stable_mask]),
        "valid_percentage": round(valid_cells_percentage, 2),
    }

    # Save the statistics to a file
    if output_file is not None:
        if output_file.suffix != ".json":
            output_file = output_file.with_suffix(".json")
        if not output_file.parent.exists():
            output_file.parent.mkdir(parents=True)
        save_stats_to_file(diff_stats, output_file)

    return diff_stats


def process_dem(
    dem: Path | xdem.DEM,
    reference_dem: Path | xdem.DEM,
    glacier_outlines: Path | gu.Vector | None = None,
    diff_dem_path: Path | None = None,
    diff_stats_path: Path | None = None,
    eval_folder: Path | None = None,
    verbose: bool = False,
    use_parent_as_name: bool = False,
) -> dict | None:
    """Process DEM and compute differences with reference DEM.

    Args:
        dem: Input DEM file path or xdem.DEM object
        reference_dem: Reference DEM file path or xdem.DEM object
        glacier_outlines: Optional glacier outlines for masking
        diff_dem_path: Explicit path for difference DEM output
        diff_stats_path: Explicit path for statistics output
        eval_folder: Folder for outputs (used if specific paths not provided)
        verbose: Enable detailed logging
        use_parent_as_name: Use parent folder name instead of file stem

    Returns:
        Dictionary with computed statistics or None if error occurs

    Raises:
        ValueError: If no output location is specified
    """
    # Get DEM name for default filenames
    if isinstance(dem, Path):
        dem_name = dem.parent.name if use_parent_as_name else dem.stem
    elif isinstance(dem, xdem.DEM):
        path = Path(dem.name)
        dem_name = path.parent.name if use_parent_as_name else path.stem
    else:
        dem_name = "input_dem"

    # Validate output locations
    if not any([diff_dem_path, diff_stats_path, eval_folder]):
        raise ValueError(
            "Must specify either output paths (diff_dem_path, diff_stats_path) "
            "or eval_folder for outputs"
        )

    # Set output paths with priority for explicit paths and create directories
    eval_folder = Path(eval_folder) if eval_folder else None
    diff_dem_path = Path(diff_dem_path) if diff_dem_path else None
    diff_stats_path = Path(diff_stats_path) if diff_stats_path else None
    if diff_dem_path is None and eval_folder:
        diff_dem_path = eval_folder / f"{dem_name}_dod.tif"
    elif diff_dem_path is None:
        diff_dem_path = dem.parent / f"{dem_name}_dod.tif"
    if diff_stats_path is None and eval_folder:
        diff_stats_path = eval_folder / f"{dem_name}_dod_stats.json"
    elif diff_stats_path is None:
        diff_stats_path = diff_dem_path.parent / f"{dem_name}_dod_stats.json"
    for path in [diff_dem_path, diff_stats_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    # Load DEMs if paths are provided, otherwise assume they are already xdem.DEM objects
    if isinstance(dem, Path) and isinstance(reference_dem, Path):
        logger.info(f"Loading DEMs: {dem_name} and reference DEM...")
        dem, reference_dem = load_dems([dem, reference_dem])
    elif isinstance(dem, Path):
        logger.info(f"Loading DEM: {dem_name}...")
        dem = xdem.DEM(dem)
    elif isinstance(reference_dem, Path):
        logger.info("Loading reference DEM...")
        reference_dem = xdem.DEM(reference_dem)
    if not isinstance(dem, xdem.DEM) or not isinstance(reference_dem, xdem.DEM):
        raise ValueError("Invalid DEM or reference DEM provided.")

    # Load and process glacier outlines if provided
    if glacier_outlines is not None:
        logger.info("Loading glacier outlines...")
        if isinstance(glacier_outlines, Path):
            glacier_outlines = gu.Vector(glacier_outlines)
            glacier_outlines = glacier_outlines.crop(reference_dem, clip=True)
        stable_mask = ~glacier_outlines.create_mask(raster=reference_dem)
    else:
        stable_mask = None  # No masking if glacier_outlines is None

    # Compute differences between DEMs and relevant statistics
    logger.info("Computing Dem of Difference...")
    dod = compute_dod(dem=dem, reference=reference_dem, resampling="bilinear")

    # Save DoD and compute statistics in separate threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        save_future = executor.submit(
            save_dem, dod, diff_dem_path, dtype="float32", compress="lzw"
        )
        stats_future = executor.submit(
            compute_dod_statistics,
            diff=dod,
            stable_mask=stable_mask,
            output_file=diff_stats_path,
        )
        save_future.result()  # This will raise any exceptions that occurred
        diff_stats = stats_future.result()

    logger.info(f"Processed DEM: {dem_name} successfully.")

    return dod, diff_stats


if __name__ == "__main__":
    pass

    # Folder containing all DEM folders
    # dem_folder = Path("outputs")
    # glacier_outlines_path = Path("data/11_rgi60_ceu_wgs84_utm32n.shp")
    # eval_folder = Path("dem_differences")

    # reference_dem_path = Path(
    #     "DEM_002-006_S5_053-256-0_2005-01-04_32632_EGM2008_10m-adj.tif"
    # )
    # dem_fname_pattern = "*_EGM2008*.tif"

    # parallel = False

    # # Create output folder if it doesn't exist
    # eval_folder.mkdir(parents=True, exist_ok=True)

    # # Get file names of all other DEMs
    # dem_files = sorted(dem_folder.rglob(dem_fname_pattern))

    # # Load Refence DEM
    # logger.info("Loading reference DEM...")
    # reference_dem = xdem.DEM(reference_dem_path)
    # reference_dem.set_vcrs("EGM08")
    # print(reference_dem.info())

    # # Load glacier outlines and clip to reference DEM extent
    # logger.info("Loading glacier outlines...")
    # glacier_outlines = gu.Vector(glacier_outlines_path)
    # glacier_outlines = glacier_outlines.crop(reference_dem, clip=True)

    # logger.info("Processing DEMs...")

    # # List to store statistics
    # stats = []
    # if parallel:
    #     # Using Joblib's Parallel to parallelize DEM processing
    #     results = Parallel(n_jobs=-1, backend="loky")(
    #         delayed(process_dem)(
    #             dem_file,
    #             reference_dem,
    #             eval_folder,
    #             glacier_outlines,
    #             use_parent_as_name=True,
    #         )
    #         for dem_file in tqdm(dem_files, desc="Processing DEMs")
    #     )

    # else:
    #     # Using a for loop to process DEMs sequentially
    #     results = []
    #     for dem_file in tqdm(dem_files, desc="Processing DEMs"):
    #         res = process_dem(
    #             dem_file,
    #             reference_dem,
    #             eval_folder,
    #             glacier_outlines,
    #             use_parent_as_name=True,
    #         )
    #         results.append(res)

    # # Filter out any None results (from failed processing)
    # stats = [res for res in results if res is not None]

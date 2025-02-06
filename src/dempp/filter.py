import logging
from enum import Enum
from pathlib import Path
from time import perf_counter

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import xdem
from joblib import Parallel, delayed
from rasterio.windows import Window
from scipy.stats import zscore
from shapely.geometry import Polygon
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dempp.elevation_bands import (
    ElevationBand,
    extract_dem_window,
    extract_elevation_bands,
    vector_to_mask,
)
from dempp.math import compute_nmad

logger = logging.getLogger("dempp")


class OutlierMethod(Enum):
    """Methods for outlier detection"""

    ZSCORE = "zscore"  # z-score
    NORMAL = "normal"  # mean/std
    ROBUST = "robust"  # median/nmad


def find_outliers(
    values: np.ma.MaskedArray | np.ndarray,
    method: OutlierMethod = OutlierMethod.ROBUST,
    n_limit: float = 3,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Find outliers in array using specified method.

    Args:
        values (np.ma.MaskedArray | np.ndarray): Input array to check for outliers.
        method (OutlierMethod, optional): Outlier detection method. Defaults to OutlierMethod.ROBUST.
        n_limit (float, optional): Number of std/mad for outlier threshold. Defaults to 3.
        mask (Optional[np.ndarray], optional): Optional boolean mask where True indicates invalid/masked values. Only used if values is not a MaskedArray. Masked pixels will not be included in outlier detection. Defaults to None.

    Returns:
        np.ndarray: Boolean array marking outliers (True = outlier). Masked pixels in the input array will not be considered as outliers and will be False in output.
    """
    # Handle input masking
    if isinstance(values, np.ma.MaskedArray):
        data = values.data
        mask = values.mask
    else:
        data = values
        if mask is None:
            mask = np.isnan(data)

    # Get valid values (not masked)
    valid_values = data[~mask]

    if len(valid_values) == 0:
        return np.zeros_like(data, dtype=bool)  # No outliers if all masked

    # Initialize outlier mask
    outliers = np.zeros_like(data, dtype=bool)  # Start with no outliers

    if method == OutlierMethod.ZSCORE:
        band_z_scores = zscore(valid_values, nan_policy="omit")
        outliers[~mask] = np.abs(band_z_scores) > n_limit

    elif method == OutlierMethod.NORMAL:
        mean = np.mean(valid_values)
        std = np.std(valid_values)
        outliers[~mask] = np.abs(data[~mask] - mean) > n_limit * std

    elif method == OutlierMethod.ROBUST:
        median = np.median(valid_values)
        nmad = compute_nmad(valid_values)
        outliers[~mask] = np.abs(data[~mask] - median) > n_limit * nmad

    else:
        raise ValueError(f"Unknown outlier method: {method}")

    return outliers


def filter_by_elevation_bands(
    dem: np.ndarray,
    band_width: float,
    method: OutlierMethod | list[OutlierMethod] = OutlierMethod.ROBUST,
    n_limit: float = 3,
    smooth_elband_dem: bool = True,
    elevation_bands: list[ElevationBand] = None,
    n_jobs: int = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Filter DEM by elevation bands using numpy operations.

    Args:
        dem (np.ndarray): The DEM data.
        band_width (float): The width of each elevation band.
        method (OutlierMethod | List[OutlierMethod], optional): Outlier detection method. If a list is provided, the filters will be applied sequentially and the final mask will be the union of all masks. Defaults to OutlierMethod.ROBUST.
        n_limit (float, optional): Number of std/mad for outlier threshold. Defaults to 3.
        smooth_elband_dem (bool, optional): Whether to smooth the DEM for the extraction of the elevation bands. Defaults to True.
        elevation_bands (list[ElevationBand], optional): List of elevation bands. If None, the bands will be extracted from the DEM. Defaults to None.

        n_jobs (int, optional): Number of parallel jobs (-1 for all cores, 1 to process sequentially). Defaults to None.

    Returns:
        tuple(np.ndarray, pd.DataFrame): The outlier mask and the elevation bands statistics.
    """

    def _process_single_band(band, method, n_limit):
        """Process a single elevation band with multiple outlier detection methods."""
        # Find outliers in band
        outlier_mask = np.zeros_like(band.masked_data.data, dtype=bool)
        for m in method:
            band_outliers = find_outliers(band.masked_data, method=m, n_limit=n_limit)
            outlier_mask |= band_outliers

        return outlier_mask

    if not isinstance(method, list):
        method = [method]
    method = [m if isinstance(m, OutlierMethod) else OutlierMethod(m) for m in method]

    # Get base mask
    if np.ma.is_masked(dem):
        base_mask = dem.mask
        dem_data = dem.data
    else:
        base_mask = np.isnan(dem)
        dem_data = dem

    # Extract elevation bands
    if elevation_bands is None:
        logger.debug("Extracting elevation bands...")
        elevation_bands = extract_elevation_bands(
            dem=dem,
            band_width=band_width,
            mask=base_mask,
            smooth_dem=smooth_elband_dem,
        )
    else:
        logger.debug("Using provided elevation bands.")
        if not all(isinstance(b, ElevationBand) for b in elevation_bands):
            raise ValueError("All elevation bands must be of type ElevationBand.")

    # Decide if to use parallel processing or not depending on the number of bands and the number of filtering methods. If this number is high, use parallel processing, otherwise process the bands sequentially.
    if len(elevation_bands) * len(method) > 20:
        n_jobs = n_jobs if n_jobs is not None else -1

    # Process all the bands
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_process_single_band)(band=band, method=method, n_limit=n_limit)
        for band in elevation_bands
    )

    # Combine the outlier masks derived from each band
    outlier_mask = np.zeros_like(dem_data, dtype=bool)
    for band_mask in results:
        outlier_mask |= band_mask

    # Make sure to exclude masked values from outlier mask
    outlier_mask[base_mask] = False

    # Update each band's mask with the outliers
    for band in elevation_bands:
        band.masked_data.mask |= outlier_mask

    # Create DataFrame with the elevation band statistics for further analysis
    def _get_band_stats(band: ElevationBand):
        """Get statistics for a single band."""
        band.compute_statistics()
        return {
            "center": band.center,
            "lower": band.lower,
            "upper": band.upper,
            "mean": band.stats["mean"],
            "std": band.stats["std"],
            "median": band.stats["median"],
            "nmad": band.stats["nmad"],
            "pixel": band.stats["pixel"],
        }

    df = pd.DataFrame([{**_get_band_stats(band)} for band in elevation_bands])

    return outlier_mask, df


def get_outlier_mask(
    dem_path: Path | str,
    boundary: gpd.GeoDataFrame | Polygon | None = None,
    filter_method: OutlierMethod | list[OutlierMethod] = OutlierMethod.ROBUST,
    n_limit: float = 3,
    use_elevation_bands: bool = True,
    elevation_band_width: float = 100,
    geometry_id: str = None,  # Optional for logging
    **kwargs,
) -> tuple[np.ndarray, Window, pd.DataFrame]:
    """Process a single glacier and return its outlier mask.

    Args:
        dem_path (Path | str): Path to the DEM file.
        geometry (gpd.GeoDataFrame | Polygon): Vector geometry with the boundaries that will be used to filter the DEM within it. Defaults to None.
        filter_method (OutlierMethod | List[OutlierMethod], optional): Outlier detection method. If a list is provided, the filters will be applied sequentially and the final mask will be the union of all masks. Defaults to OutlierMethod.ROBUST.
        n_limit (float, optional): Number of std/mad for outlier threshold. Defaults to 3.
        use_elevation_bands (bool, optional): Whether to use elevation bands for filtering. Defaults to True.
        elevation_band_width (float): Width of elevation bands.
        geometry_id (str, optional): Optional geometry_id for logging. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the filter_by_elevation_bands() function.

    Returns:
        Tuple[np.ndarray, Window, pd.Dataframe]: The outlier mask, the DEM window and the elevation bands statistics (if use_elevation_bands=True, else None).
    """

    if geometry_id is None:
        geometry_id = ""

    logger.debug(f"Processing glacier {geometry_id}")

    if boundary is not None:
        # Extract DEM window around the given geometry
        dem_window = extract_dem_window(dem_path, boundary)

        if dem_window is None:
            logger.debug(f"Skipping glacier {geometry_id}: no valid DEM window")
            return None, None

        window = dem_window.window
        logger.debug(f"Extracted DEM window: {window}")

        # Create glacier mask for window
        mask = vector_to_mask(
            boundary,
            (window.height, window.width),
            dem_window.transform,
            crs=dem_window.crs,
        )
        logger.debug("Created glacier mask for window")

        # Combine glacier mask with nodata mask
        combined_mask = ~mask | dem_window.mask

        # Apply combined mask
        masked_dem = np.ma.masked_array(dem_window.data, mask=combined_mask)

    else:
        # Load full DEM with rasterio
        with rio.open(dem_path) as src:
            masked_dem = src.read(1, masked=True)
        window = None

    # Check if the dem is completely masked
    if masked_dem.mask.all():
        logger.debug(
            f"Glacier {geometry_id} is completely masked, returning empty mask"
        )
        empty_mask = np.zeros_like(masked_dem.data, dtype=bool)
        return empty_mask, window

    logger.debug(
        f"Processing glacier {geometry_id} with {masked_dem.count()} valid pixels"
    )

    # Filter DEM based on elevation bands
    if use_elevation_bands:
        outlier_mask, bands_stats = filter_by_elevation_bands(
            dem=masked_dem,
            band_width=elevation_band_width,
            method=filter_method,
            n_limit=n_limit,
            **kwargs,
        )
    else:
        # Filter the entire DEM
        outlier_mask = find_outliers(
            values=masked_dem,
            method=filter_method,
            n_limit=n_limit,
        )
        bands_stats = None
    logger.debug(f"Finished processing glacier {geometry_id}")

    return outlier_mask, window, bands_stats


def apply_mask_to_dem(
    dem_path: Path | str,
    mask: np.ndarray,
    output_path: Path | str = None,
) -> None:
    """Apply a mask to a DEM and save the masked DEM."""

    dem = xdem.DEM(dem_path)
    dem.load()
    dem.set_mask(mask)

    # Save masked DEM
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dem.save(output_path)
        logger.info(f"Saved masked DEM to {output_path}")

    return dem


def filter_dem(
    dem_path: Path | str,
    output_path: Path | str = None,
    boundary: gpd.GeoDataFrame | Path | str | None = None,
    filter_method: OutlierMethod | list[OutlierMethod] = OutlierMethod.ROBUST,
    n_limit: float = 3,
    use_elevation_bands: bool = True,
    elevation_band_width: float = 100,
    **kwargs,
) -> tuple[xdem.DEM, np.ndarray]:
    """
    filter_dem Filter a DEM, optionally within a given boundary (e.g. glacier) using outlier detection.

    Args:
        dem_path (Path | str): Path to DEM file.
        output_path (Path | str): Path to save the output DEM file. Defaults to None.
        boundary (gpd.GeoDataFrame | Path | str | None): Boundary geometry to filter the DEM within. Defaults to None.
        filter_method (OutlierMethod | list[OutlierMethod], optional): Outlier detection method. If a list is provided, the filters will be applied sequentially and the final mask will be the union of all masks. Defaults to OutlierMethod.ROBUST.
        n_limit (float, optional): Number of std/mad for outlier threshold. Defaults to 3.
        use_elevation_bands (bool, optional): Whether to use elevation bands for filtering. Defaults to True.
        elevation_band_width (float, optional): Width of elevation bands. Defaults to 100.
        **kwargs: Additional keyword arguments to be passed to the filter_by_elevation_bands() function.


    Returns:
        tuple[xdem.DEM, np.ndarray]: The filtered DEM and the final outlier mask.
    """

    with rio.open(dem_path) as src:
        dem_crs = src.crs
        dem_shape = src.shape

    if boundary is not None:
        if isinstance(boundary, (Path | str)):
            boundary = gpd.read_file(boundary).to_crs(dem_crs)
        boundary = boundary.to_crs(dem_crs)

    # Get outlier mask
    outlier_mask, window, stats = get_outlier_mask(
        dem_path=dem_path,
        boundary=boundary,
        filter_method=filter_method,
        n_limit=n_limit,
        use_elevation_bands=use_elevation_bands,
        elevation_band_width=elevation_band_width,
        **kwargs,
    )

    # Create final mask
    final_mask = np.zeros(dem_shape, dtype=bool)
    if window is not None:
        final_mask[
            window.row_off : window.row_off + window.height,
            window.col_off : window.col_off + window.width,
        ] = outlier_mask
    else:
        final_mask = outlier_mask
    logger.info(
        f"Masked {final_mask.sum()} outliers in total ({final_mask.sum() / final_mask.size * 100:.2f}%)."
    )

    # Apply mask to DEM and save
    logger.info("Applying final mask to DEM and saving output...")
    filtered_dem = apply_mask_to_dem(
        dem_path=dem_path, mask=final_mask, output_path=output_path
    )

    return filtered_dem, final_mask, stats


def filter_dem_by_glacier(
    dem_path: Path | str,
    glacier_outline_path: Path | str,
    output_path: Path | str = None,
    filter_method: OutlierMethod | list[OutlierMethod] = OutlierMethod.ROBUST,
    n_limit: float = 3,
    use_elevation_bands: bool = True,
    elevation_band_width: float = 100,
    n_jobs: int = -1,
    **kwargs,
) -> tuple[xdem.DEM, np.ndarray]:
    """Process multiple glaciers and combine their outlier masks.

    Args:
        dem_path (Path | str): Path to DEM file.
        glacier_outline_path (Path | str): Path to glacier outlines (GeoJSON).
        output_path (Path | str): Path to save the output DEM file. Defaults to None.
        filter_method (OutlierMethod | List[OutlierMethod], optional): Outlier detection method. If a list is provided, the filters will be applied sequentially and the final mask will be the union of all masks. Defaults to OutlierMethod.ROBUST.
        n_limit (float, optional): Number of std/mad for outlier threshold. Defaults to 3.
        use_elevation_bands (bool, optional): Whether to use elevation bands for filtering. Defaults to True.
        elevation_band_width (float, optional): Width of elevation bands. Defaults to 50.
        n_jobs (int, optional): Number of parallel jobs (-1 for all cores, 1 to process sequentially). Defaults to -1.
        **kwargs: Additional keyword arguments to be passed to the filter_by_elevation_bands() function.

    Returns:
        Tuple[xdem.DEM, np.ndarray]: The filtered DEM and the final outlier mask.
    """
    # Get DEM metadata
    with rio.open(dem_path) as src:
        dem_crs = src.crs
        dem_shape = src.shape

    # Load glacier outlines
    glacier_outlines = gpd.read_file(glacier_outline_path).to_crs(dem_crs)

    # Process glaciers in parallel
    logger.info("Filtering DEM based on elevation bands for each glacier...")
    with logging_redirect_tqdm([logger]):
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(get_outlier_mask)(
                dem_path=dem_path,
                boundary=row.geometry,
                elevation_band_width=elevation_band_width,
                filter_method=filter_method,
                n_limit=n_limit,
                use_elevation_bands=use_elevation_bands,
                geometry_id=row.get("rgi_id", None),
                **kwargs,
            )
            for _, row in tqdm(glacier_outlines.iterrows())
        )
    logger.info("Finished processing all glaciers. Combining results...")

    # Combine results into final mask
    final_mask = np.zeros(dem_shape, dtype=bool)
    for outlier_mask, window in results:
        if outlier_mask is not None and window is not None:
            final_mask[
                window.row_off : window.row_off + window.height,
                window.col_off : window.col_off + window.width,
            ] |= outlier_mask
    logger.info("Combined all glacier masks into final mask.")
    logger.info(
        f"Masked {final_mask.sum()} outliers in total ({final_mask.sum() / final_mask.size * 100:.2f}%)."
    )

    # Apply mask to DEM and save
    logger.info("Applying final mask to DEM and saving output...")
    filtered_dem = apply_mask_to_dem(
        dem_path=dem_path, mask=final_mask, output_path=output_path
    )

    return filtered_dem, final_mask


if __name__ == "__main__":
    N_JOBS = -1

    dem_dir = Path("outputs/proc/009_003-009_S5_054-256-0_2003-11-15/opals")
    dem_name = "stereo-DEM_transLSM_robMovingPlanes_10m_filled_adaptive.tif"
    rgi_path = Path("el_bands/rgi_clip.geojson")
    output_dir = Path("el_bands")

    elevation_band_width = 100
    filter_method = [OutlierMethod.ROBUST]
    n_limit = 3

    dem_path = dem_dir / dem_name
    output_path = output_dir / f"dem_filtered_{elevation_band_width}_n{n_limit}.tif"
    t0 = perf_counter()
    dem_filtered, mask = filter_dem_by_glacier(
        dem_path=dem_path,
        glacier_outline_path=rgi_path,
        output_path=output_path,
        filter_method=filter_method,
        n_limit=n_limit,
        use_elevation_bands=True,
        elevation_band_width=elevation_band_width,
        n_jobs=N_JOBS,
    )
    logger.info(f"Filering by glacier took {perf_counter() - t0:.3f} seconds.")

    output_path = output_dir / "dem_filtered_full.tif"
    t0 = perf_counter()
    dem_filtered, mask = filter_dem(
        dem_path=dem_path,
        output_path=output_path,
        filter_method=filter_method,
        n_limit=n_limit,
        use_elevation_bands=True,
        elevation_band_width=elevation_band_width,
    )
    logger.info(f"Filering full dem took {perf_counter() - t0:.3f} seconds.")

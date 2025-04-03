import logging
from enum import Enum
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio as rio
import xdem
from joblib import Parallel, delayed
from rasterio import features
from rasterio.windows import Window, from_bounds
from scipy.stats import zscore
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from xdem.spatialstats import nmad

logger = logging.getLogger("dempp")


class OutlierMethod(Enum):
    """Methods for outlier detection"""

    ZSCORE = "zscore"  # z-score
    NORMAL = "normal"  # mean/std
    ROBUST = "robust"  # median/nmad


def find_outliers(
    values: np.ma.MaskedArray | np.ndarray,
    method: OutlierMethod = OutlierMethod.ROBUST,
    outlier_threshold: float = 3,
    mask: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    """Find outliers in array using specified method.

    Args:
        values (np.ma.MaskedArray | np.ndarray): Input array to check for outliers.
        method (OutlierMethod, optional): Outlier detection method. Defaults to OutlierMethod.ROBUST.
        outlier_threshold (float, optional): Number of std/mad for outlier threshold. Defaults to 3.
        mask (Optional[np.ndarray], optional): Optional boolean mask where True indicates invalid/masked values.
            Only used if values is not a MaskedArray. Masked pixels will not be included in outlier detection.
            Defaults to None.
        **kwargs: Additional parameters passed to the specific filter method.

    Returns:
        np.ndarray: Boolean array marking outliers (True = outlier). Masked pixels in the input array
            will not be considered as outliers and will be False in output.
    """
    # Handle input masking
    if isinstance(values, np.ma.MaskedArray):
        data = values.data
        mask = values.mask
    else:
        data = values
        if mask is None:
            mask = np.isnan(data)

    # Check if all values are masked
    if mask.all():
        return np.zeros_like(data, dtype=bool)  # No outliers if all masked

    # Use method-specific implementation
    if method == OutlierMethod.ZSCORE:
        outliers = _filter_zscore(data, outlier_threshold=outlier_threshold, mask=mask)
    elif method == OutlierMethod.NORMAL:
        outliers = _filter_normal(data, outlier_threshold=outlier_threshold, mask=mask)
    elif method == OutlierMethod.ROBUST:
        outliers = _filter_robust(data, outlier_threshold=outlier_threshold, mask=mask)
    else:
        raise ValueError(f"Unknown outlier method: {method}")

    return outliers


def _filter_zscore(
    values: np.ndarray,
    outlier_threshold: float = 3,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Find outliers using z-score method.

    This function computes z-scores for the array and identifies values that deviate
    more than a specified threshold from the mean in terms of standard deviations.

    Args:
        values (np.ndarray): Input 2D array to check for outliers.
        outlier_threshold (float, optional): Number of standard deviations for outlier threshold.
            Values with |z-score| > outlier_threshold are considered outliers. Defaults to 3.
        mask (np.ndarray | None, optional): Boolean mask where True indicates invalid/masked values.
            If provided, outlier detection is applied only on non-masked values.
            If None, outlier detection is applied to the entire array. Defaults to None.
        **kwargs: Additional parameters (unused for this method).

    Returns:
        np.ndarray: Boolean array marking outliers (True = outlier).
            Masked pixels (if mask provided) will be False in output.
    """
    # Initialize outlier mask with zeros (no outliers by default)
    outliers = np.zeros_like(values, dtype=bool)

    # Check if we have a mask
    if mask is not None:
        # Get valid (non-masked) values
        valid_values = values[~mask]

        # Calculate z-scores only for valid values
        z_scores = zscore(valid_values, nan_policy="omit")

        # Apply threshold to valid values only
        outliers[~mask] = np.abs(z_scores) > outlier_threshold
    else:
        # Apply to entire array
        z_scores = zscore(values, nan_policy="omit")
        outliers = np.abs(z_scores) > outlier_threshold

    return outliers


def _filter_normal(
    values: np.ndarray,
    outlier_threshold: float = 3,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Find outliers using normal distribution method (mean/std).

    This function identifies outliers based on how far values deviate from the mean
    in terms of standard deviations.

    Args:
        values (np.ndarray): Input 2D array to check for outliers.
        outlier_threshold (float, optional): Number of standard deviations for outlier threshold.
            Values with |value-mean| > outlier_threshold*std are considered outliers. Defaults to 3.
        mask (np.ndarray | None, optional): Boolean mask where True indicates invalid/masked values.
            If provided, outlier detection is applied only on non-masked values.
            If None, outlier detection is applied to the entire array. Defaults to None.
        **kwargs: Additional parameters (unused for this method).

    Returns:
        np.ndarray: Boolean array marking outliers (True = outlier).
            Masked pixels (if mask provided) will be False in output.
    """
    # Initialize outlier mask with zeros (no outliers by default)
    outliers = np.zeros_like(values, dtype=bool)

    # Calculate statistics based on masking
    if mask is not None:
        # Only use non-masked values for statistics
        valid_values = values[~mask]
        mean = np.mean(valid_values)
        std = np.std(valid_values)

        # Apply threshold to valid values only
        outliers[~mask] = np.abs(values[~mask] - mean) > outlier_threshold * std
    else:
        # Use entire array
        mean = np.mean(values)
        std = np.std(values)
        outliers = np.abs(values - mean) > outlier_threshold * std

    return outliers


def _filter_robust(
    values: np.ndarray,
    outlier_threshold: float = 3,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Find outliers using robust method (median/nmad).

    This function identifies outliers based on how far values deviate from the median
    in terms of normalized median absolute deviations (NMAD), providing more
    robustness against existing outliers than mean/std methods.

    Args:
        values (np.ndarray): Input 2D array to check for outliers.
        outlier_threshold (float, optional): Number of NMADs for outlier threshold.
            Values with |value-median| > outlier_threshold*nmad are considered outliers. Defaults to 3.
        mask (np.ndarray | None, optional): Boolean mask where True indicates invalid/masked values.
            If provided, outlier detection is applied only on non-masked values.
            If None, outlier detection is applied to the entire array. Defaults to None.
        **kwargs: Additional parameters (unused for this method).

    Returns:
        np.ndarray: Boolean array marking outliers (True = outlier).
            Masked pixels (if mask provided) will be False in output.
    """
    # Initialize outlier mask with zeros (no outliers by default)
    outliers = np.zeros_like(values, dtype=bool)

    # Calculate statistics based on masking
    if mask is not None:
        # Only use non-masked values for statistics
        valid_values = values[~mask]
        median = np.median(valid_values)
        mad_value = nmad(valid_values)

        # Apply threshold to valid values only
        outliers[~mask] = np.abs(values[~mask] - median) > outlier_threshold * mad_value
    else:
        # Use entire array
        median = np.median(values)
        mad_value = nmad(values)
        outliers = np.abs(values - median) > outlier_threshold * mad_value

    return outliers


def get_outlier_mask(
    dem_path: Path | str,
    boundary: gpd.GeoDataFrame | dict | None = None,
    filter_method: OutlierMethod | list[OutlierMethod] = OutlierMethod.ROBUST,
    outlier_threshold: float = 3,
) -> tuple[np.ndarray, Window]:
    """Process a single glacier and return its outlier mask.

    Args:
        dem_path (Path | str): Path to the DEM file.
        boundary (gpd.GeoDataFrame | dict): Vector geometry with the boundaries that will be used to filter the DEM within it. It can be a GeoDataFrame or a dictionary compatible with the __geo_interface__ format (as it comes from geodataframe.iterfeatures()). Defaults to None.
        filter_method (OutlierMethod | List[OutlierMethod], optional): Outlier detection method. If a list is provided, the filters will be applied sequentially and the final mask will be the union of all masks. Defaults to OutlierMethod.ROBUST.
        outlier_threshold (float, optional): Number of std/mad for outlier threshold. Defaults to 3.

    Returns:
        Tuple[np.ndarray, Window]: The outlier mask and the DEM window.
    """

    if isinstance(filter_method, list):
        raise NotImplementedError(
            "Multiple filter methods are not implemented yet. Please use a single method."
        )

    with rio.open(dem_path) as src:
        if boundary is not None:
            # Convert dict to GeoDataFrame if needed
            if isinstance(boundary, dict):
                boundary = gpd.GeoDataFrame.from_features([boundary])

                # we manually set the CRS to the DEM CRS (it was reprojected beforehand)
                boundary.set_crs(src.crs, inplace=True)

            # Get bounds of the geometry
            minx, miny, maxx, maxy = boundary.total_bounds

            # Get pixel coordinates from bounds
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            window = window.round_lengths().round_offsets()

            # Make sure window is within raster bounds
            try:
                window = window.intersection(Window(0, 0, src.width, src.height))
            except Exception as e:
                logger.debug(f"Window intersection failed: {e}")
                return None, None

            # Check if window is valid
            if window.width <= 0 or window.height <= 0:
                logger.debug("Window outside raster bounds")
                return None, None

            # Read the DEM data within the window
            dem_data = src.read(1, window=window)
            transform = src.window_transform(window)

            # Create mask for the geometry
            geometry_mask = features.rasterize(
                shapes=boundary.geometry,
                out_shape=(window.height, window.width),
                transform=transform,
                fill=0,
                default_value=1,
                dtype="uint8",
            ).astype(bool)

            # Create mask for nodata values
            nodata_mask = (
                dem_data == src.nodata
                if src.nodata is not None
                else np.zeros_like(dem_data, dtype=bool)
            )

            # Combine masks (True = masked)
            combined_mask = ~geometry_mask | nodata_mask

            # Create masked array
            masked_dem = np.ma.masked_array(dem_data, mask=combined_mask)

        else:
            # Load full DEM
            masked_dem = src.read(1, masked=True)
            window = None

    # Check if the DEM is completely masked
    if masked_dem.mask.all():
        logger.debug(
            "DEM is completely masked within the given geometry, returning empty mask"
        )
        empty_mask = np.zeros_like(masked_dem.data, dtype=bool)
        return empty_mask, window

    # Filter the masked DEM using the specified method
    outlier_mask = find_outliers(
        values=masked_dem,
        method=filter_method,
        outlier_threshold=outlier_threshold,
    )

    return outlier_mask, window


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


def filter_dem_by_geometry(
    dem_path: Path | str,
    geometry_path: Path | str,
    filter_method: OutlierMethod | list[OutlierMethod] = OutlierMethod.ROBUST,
    outlier_threshold: float = 3,
    n_jobs: int = -1,
) -> tuple[xdem.DEM, np.ndarray]:
    """Process multiple glaciers and combine their outlier masks.

    Args:
        dem_path (Path | str): Path to DEM file.
        glacier_outline_path (Path | str): Path to glacier outlines (GeoJSON).
        output_path (Path | str): Path to save the output DEM file. Defaults to None.
        filter_method (OutlierMethod | List[OutlierMethod], optional): Outlier detection method. If a list is provided, the filters will be applied sequentially and the final mask will be the union of all masks. Defaults to OutlierMethod.ROBUST.
        outlier_threshold (float, optional): Number of std/mad for outlier threshold. Defaults to 3.
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
    geometries = gpd.read_file(geometry_path).to_crs(dem_crs)

    # Process glaciers in parallel
    logger.info("Filtering DEM based on elevation bands for each glacier...")
    with logging_redirect_tqdm([logger]):
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(get_outlier_mask)(
                dem_path=dem_path,
                boundary=feature,
                filter_method=filter_method,
                outlier_threshold=outlier_threshold,
            )
            for feature in tqdm(geometries.iterfeatures(na="drop", show_bbox=True))
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
    logger.info("Applying final mask to DEM...")
    filtered_dem = apply_mask_to_dem(dem_path=dem_path, mask=final_mask)

    return filtered_dem, final_mask


def filter_dem(
    dem_path: Path | str,
    boundary_path: Path | str | None = None,
    filter_method: OutlierMethod | list[OutlierMethod] = OutlierMethod.ROBUST,
    outlier_threshold: float = 3,
) -> tuple[xdem.DEM, np.ndarray]:
    """
    filter_dem Filter a DEM, optionally within a given boundary (e.g. glacier) using outlier detection.

    Args:
        dem_path (Path | str): Path to DEM file.
        output_path (Path | str): Path to save the output DEM file. Defaults to None.
        boundary (gpd.GeoDataFrame | Path | str | None): Boundary geometry to filter the DEM within. Defaults to None.
        filter_method (OutlierMethod | list[OutlierMethod], optional): Outlier detection method. If a list is provided, the filters will be applied sequentially and the final mask will be the union of all masks. Defaults to OutlierMethod.ROBUST.
        outlier_threshold (float, optional): Number of std/mad for outlier threshold. Defaults to 3.

    Returns:
        tuple[xdem.DEM, np.ndarray]: The filtered DEM and the final outlier mask.
    """

    with rio.open(dem_path) as src:
        dem_crs = src.crs
        dem_shape = src.shape

    if boundary_path is not None:
        boundary = gpd.read_file(boundary_path).to_crs(dem_crs)
        boundary = boundary.to_crs(dem_crs)
    else:
        boundary = None

    # Get outlier mask
    outlier_mask, window = get_outlier_mask(
        dem_path=dem_path,
        boundary=boundary,
        filter_method=filter_method,
        outlier_threshold=outlier_threshold,
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
    logger.info("Applying final mask to DEM...")
    filtered_dem = apply_mask_to_dem(dem_path=dem_path, mask=final_mask)

    return filtered_dem, final_mask

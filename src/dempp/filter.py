from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
import xdem
from joblib import Parallel, delayed
from rasterio import features
from rasterio.errors import WindowError
from rasterio.windows import Window, from_bounds
from scipy.stats import zscore
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from xdem._typing import NDArrayf
from xdem.filters import gaussian_filter_cv
from xdem.spatialstats import nmad

logger = logging.getLogger("dempp")


class OutlierMethod(Enum):
    """Methods for outlier detection"""

    NMAD = "nmad"  # median/nmad
    ZSCORE = "zscore"  # z-score
    NORMAL = "normal"  # mean/std
    DISTANCE = "distance"  # Distance filter


def distance_filter(
    array: NDArrayf,
    radius: float,
    outlier_threshold: float,
) -> NDArrayf:
    """
    Filter out pixels whose value is distant from the average of neighboring pixels.

    This function identifies pixels whose value differs from their neighborhood average
    by more than a specified threshold and sets them to NaN.

    Args:
        array: Input array to be filtered
        radius: Radius (in pixels) within which to calculate the average value
        outlier_threshold: Minimum absolute difference between pixel value and local
                          average for a pixel to be considered an outlier

    Returns:
        NDArrayf: Boolean array identifying outlier pixels (True = outlier)

    Todo:
        Add options for different averaging methods (Gaussian, median, etc.)
    """
    logger.debug(
        f"Running distance filter with radius={radius}, threshold={outlier_threshold}"
    )
    outliers = _distance_filter(array, radius, outlier_threshold)
    out_array = np.copy(array)
    out_array[outliers] = np.nan

    return outliers


def _distance_filter(
    array: NDArrayf,
    radius: float,
    outlier_threshold: float,
) -> NDArrayf:
    """
    Identify pixels whose value differs from their neighborhood average by more than a threshold.

    This internal function performs the actual distance-based outlier detection without
    modifying the input array.

    Args:
        array: Input array to check for outliers
        radius: Radius (in pixels) within which to calculate the average value (sigma for Gaussian filter)
        outlier_threshold: Minimum absolute difference between pixel value and local
                          average for a pixel to be considered an outlier

    Returns:
        NDArrayf: Boolean array identifying outlier pixels (True = outlier)

    Todo:
        Add options for different averaging methods (Gaussian, median, etc.)
    """
    # Calculate the average value within the radius
    smooth = gaussian_filter_cv(array, sigma=radius)

    # Filter outliers
    outliers = (np.abs(array - smooth)) > outlier_threshold

    logger.debug(f"Distance filter identified {np.sum(outliers)} outliers")
    return outliers


def _zscore_filter(
    values: np.ndarray,
    outlier_threshold: float = 3,
    **kwargs,
) -> np.ndarray:
    """
    Find outliers using z-score method.

    This function identifies values that deviate more than a specified threshold
    from the mean in terms of standard deviations.

    Args:
        values: Input array to check for outliers
        outlier_threshold: Maximum absolute z-score allowed; values with |z-score| > threshold
                          are considered outliers (default: 3)
        **kwargs: Additional parameters passed to scipy.stats.zscore
                 'nan_policy' is set to 'omit' by default to handle NaN values

    Returns:
        np.ndarray: Boolean array marking outliers (True = outlier)
    """
    outliers = np.zeros_like(values, dtype=bool)
    nan_policy = kwargs.pop("nan_policy", "omit")
    z_scores = zscore(values, nan_policy=nan_policy, **kwargs)
    outliers = np.abs(z_scores) > outlier_threshold

    logger.debug(
        f"Z-score filter identified {np.sum(outliers)} outliers using threshold {outlier_threshold}"
    )
    return outliers


def _normal_filter(
    values: np.ndarray,
    n_limit: float = 3,
) -> np.ndarray:
    """
    Find outliers using normal distribution method (mean/standard deviation).

    This function identifies values that deviate more than a specified number of
    standard deviations from the mean.

    Args:
        values: Input array to check for outliers
        n_limit: Number of standard deviations for outlier threshold; values with
                |value-mean| > n_limit*std are considered outliers (default: 3)

    Returns:
        np.ndarray: Boolean array marking outliers (True = outlier)
    """
    outliers = np.zeros_like(values, dtype=bool)
    mean = np.mean(values)
    std = np.std(values)
    outliers = np.abs(values - mean) > n_limit * std

    logger.debug(
        f"Normal filter identified {np.sum(outliers)} outliers using {n_limit} std limit"
    )
    logger.debug(f"Mean: {mean:.2f}, Std: {std:.2f}")
    return outliers


def _nmad_filter(
    values: np.ndarray,
    outlier_threshold: float = 3,
) -> np.ndarray:
    """
    Find outliers using robust method (median/normalized median absolute deviation).

    This function identifies values that deviate more than a specified number of NMADs
    from the median, providing greater robustness against existing outliers compared
    to mean/std methods.

    Args:
        values: Input array to check for outliers
        outlier_threshold: Number of NMADs for outlier threshold; values with
                          |value-median| > outlier_threshold*nmad are considered outliers (default: 3)

    Returns:
        np.ndarray: Boolean array marking outliers (True = outlier)
    """
    outliers = np.zeros_like(values, dtype=bool)
    median = np.median(values)
    mad_value = nmad(values)
    outliers = np.abs(values - median) > outlier_threshold * mad_value

    logger.debug(
        f"NMAD filter identified {np.sum(outliers)} outliers using {outlier_threshold} NMAD threshold"
    )
    logger.debug(f"Median: {median:.2f}, NMAD: {mad_value:.2f}")
    return outliers


def find_outliers(
    dem_path: Path | str,
    boundary: Optional[Union[gpd.GeoDataFrame, Dict[str, Any]]] = None,
    method: Union[OutlierMethod, str] = OutlierMethod.NMAD,
    **kwargs,
) -> Tuple[Optional[np.ndarray], Optional[Window]]:
    """
    Process a DEM within a specified boundary and identify outliers.

    Args:
        dem_path: Path to the DEM file
        boundary: Vector geometry defining the area of interest. Can be a GeoDataFrame
                 or a dictionary compatible with __geo_interface__ format (default: None)
        method: Outlier detection method to use (default: OutlierMethod.NMAD)
        **kwargs: Additional keyword arguments passed to the filter method

    Returns:
        Tuple containing:
            - np.ndarray: Boolean array identifying outlier pixels (True = outlier)
            - Window: The DEM window that was processed

        Returns (None, None) if no valid data is found within the boundary

    Raises:
        ValueError: If an unknown outlier method is specified
    """
    if isinstance(method, str):
        try:
            method = OutlierMethod[method.upper()]
        except KeyError:
            logger.error(f"Unknown outlier method: {method}")
            raise ValueError(f"Unknown outlier method: {method}")

    logger.debug(f"Processing DEM: {dem_path} with method: {method.name}")
    with rasterio.open(dem_path) as src:
        if boundary is not None:
            # Convert dict to GeoDataFrame if needed
            if isinstance(boundary, dict):
                boundary = gpd.GeoDataFrame.from_features([boundary])
                boundary.set_crs(src.crs, inplace=True)

            # Get bounds of the geometry
            minx, miny, maxx, maxy = boundary.total_bounds

            # Get pixel coordinates from bounds
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            window = window.round_lengths().round_offsets()

            # Make sure window is within raster bounds
            try:
                window = window.intersection(Window(0, 0, src.width, src.height))
            except WindowError as e:
                logger.debug(f"Window intersection failed: {e}")
                return None, None

            # Check if window is valid
            if window.width <= 0 or window.height <= 0:
                logger.debug("Window outside raster bounds")
                return None, None

            # Read the DEM data within the window
            dem_data = src.read(1, window=window)
            transform = src.window_transform(window)
            logger.debug(
                f"Read DEM data with shape {dem_data.shape} from window {window}"
            )

            # Create mask for the geometry (True = inside geometry)
            geometry_mask = features.rasterize(
                shapes=boundary.geometry,
                out_shape=(window.height, window.width),
                transform=transform,
                fill=0,
                default_value=1,
                dtype="uint8",
            ).astype(bool)

            # Create mask for nodata values (True = is nodata)
            if src.nodata is not None:
                nodata_mask = dem_data == src.nodata
            else:
                nodata_mask = np.isnan(dem_data)

            # Get valid data mask (True = valid data inside geometry)
            valid_mask = geometry_mask & ~nodata_mask
            valid_count = np.sum(valid_mask)
            logger.debug(
                f"Valid pixels in boundary: {valid_count} ({valid_count / valid_mask.size * 100:.2f}%)"
            )

            # If no valid data in the window, return empty mask
            if not np.any(valid_mask):
                logger.info("No valid data found within the specified boundary")
                return np.zeros_like(dem_data, dtype=bool), window

            # Create a copy of data for filtering
            data_for_filtering = dem_data.copy()

            # Replace nodata/outside geometry with NaN for filter processing
            data_for_filtering[~valid_mask] = np.nan

        else:
            # Load full DEM
            logger.debug("No boundary provided, processing entire DEM")
            window = None
            dem_data = src.read(1)
            logger.debug(f"Loaded full DEM with shape {dem_data.shape}")

            # Create mask for nodata values (True = is nodata)
            if src.nodata is not None:
                nodata_mask = dem_data == src.nodata
            else:
                nodata_mask = np.isnan(dem_data)

            # Create a copy of data for filtering
            data_for_filtering = dem_data.copy()

            # Replace nodata with NaN for filter processing
            data_for_filtering[nodata_mask] = np.nan

            # Valid mask is just the inverse of nodata mask
            valid_mask = ~nodata_mask
            valid_count = np.sum(valid_mask)
            logger.debug(
                f"Valid pixels in DEM: {valid_count} ({valid_count / valid_mask.size * 100:.2f}%)"
            )

    # If there are no valid values to process, return empty mask
    if not np.any(valid_mask):
        logger.info("No valid data to process in DEM")
        return np.zeros_like(dem_data, dtype=bool), window

    # Apply the appropriate filter to detect outliers
    logger.info(f"Applying {method.name} filter to detect outliers")
    if method == OutlierMethod.NMAD:
        outliers = _nmad_filter(data_for_filtering, **kwargs)
    elif method == OutlierMethod.DISTANCE:
        outliers = _distance_filter(data_for_filtering, **kwargs)
    elif method == OutlierMethod.ZSCORE:
        outliers = _zscore_filter(data_for_filtering, **kwargs)
    elif method == OutlierMethod.NORMAL:
        outliers = _normal_filter(data_for_filtering, **kwargs)
    else:
        logger.error(f"Unknown outlier method: {method}")
        raise ValueError(f"Unknown outlier method: {method}")

    # Ensure only valid data can be outliers (nodata and outside geometry pixels are not outliers)
    outliers = outliers & valid_mask
    logger.info(
        f"Identified {np.sum(outliers)} outliers ({np.sum(outliers) / np.sum(valid_mask) * 100:.2f}% of valid data)"
    )

    return outliers, window


def filter_dem_by_geometry(
    dem_path: Union[Path, str],
    geometry_path: Union[Path, str],
    method: OutlierMethod = OutlierMethod.NMAD,
    n_jobs: int = -1,
    **kwargs,
) -> Tuple[xdem.DEM, np.ndarray]:
    """
    Process a DEM using multiple geometries (e.g., glacier polygons) and identify outliers.

    This function processes each geometry in the provided file separately and combines
    the outlier masks into a single result.

    Args:
        dem_path: Path to DEM file
        geometry_path: Path to vector file containing geometries (e.g., glacier outlines)
        method: Outlier detection method to use (default: OutlierMethod.NMAD)
        n_jobs: Number of parallel jobs (-1 for all cores, 1 for sequential processing)
        **kwargs: Additional keyword arguments passed to find_outliers()

    Returns:
        Tuple containing:
            - xdem.DEM: The filtered DEM with outliers masked as NaN
            - np.ndarray: Boolean array identifying outlier pixels (True = outlier)
    """
    # Load geometry vector (e.g., glacier outlines)
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        dem_shape = src.shape
        geometries = gpd.read_file(geometry_path).to_crs(dem_crs)

    feature_count = len(geometries)
    logger.info(f"Loaded {feature_count} features from {geometry_path}")
    logger.info(
        f"Filtering DEM based on features using {method.name} method with {n_jobs} parallel jobs"
    )

    # Process different geometries in parallel
    with logging_redirect_tqdm([logger]):
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(find_outliers)(
                dem_path=dem_path,
                boundary=feature,
                method=method,
                **kwargs,
            )
            for feature in tqdm(
                geometries.iterfeatures(na="drop", show_bbox=True),
                desc="Processing features",
                total=feature_count,
            )
        )
    logger.info(f"Finished processing all {feature_count} features")

    # Combine results into final mask
    final_mask = np.zeros(dem_shape, dtype=bool)
    valid_result_count = 0
    for outlier_mask, window in results:
        if outlier_mask is not None and window is not None:
            valid_result_count += 1
            final_mask[
                window.row_off : window.row_off + window.height,
                window.col_off : window.col_off + window.width,
            ] |= outlier_mask

    logger.info(f"Combined results from {valid_result_count} features with valid data")
    logger.info(
        f"Masked {final_mask.sum()} outliers in total ({final_mask.sum() / final_mask.size * 100:.4f}% of all pixels)"
    )

    # Apply mask to DEM and save
    logger.info("Applying final mask to DEM...")
    filtered_dem = apply_mask_to_dem(dem_path=dem_path, mask=final_mask)

    return filtered_dem, final_mask


def filter_dem(
    dem_path: Union[Path, str],
    boundary_path: Optional[Union[Path, str]] = None,
    method: Union[OutlierMethod, List[OutlierMethod]] = OutlierMethod.NMAD,
    **kwargs,
) -> Tuple[xdem.DEM, np.ndarray]:
    """
    Filter a DEM to identify and remove outliers, optionally within a specified boundary.

    This function applies outlier detection to an entire DEM or within a specific boundary.

    Args:
        dem_path: Path to DEM file
        boundary_path: Optional path to vector file defining the boundary (default: None)
        method: Outlier detection method or list of methods to apply (default: OutlierMethod.NMAD)
        **kwargs: Additional keyword arguments passed to find_outliers()

    Returns:
        Tuple containing:
            - xdem.DEM: The filtered DEM with outliers masked as NaN
            - np.ndarray: Boolean array identifying outlier pixels (True = outlier)
    """
    logger.info(f"Filtering DEM {dem_path} using {method} method")

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        dem_shape = src.shape

    if boundary_path is not None:
        logger.info(f"Using boundary from {boundary_path}")
        boundary = gpd.read_file(boundary_path).to_crs(dem_crs)
    else:
        logger.info("No boundary provided, processing entire DEM")
        boundary = None

    # Get outlier mask
    outlier_mask, window = find_outliers(
        dem_path=dem_path, boundary=boundary, method=method, **kwargs
    )

    # Create final mask
    final_mask = np.zeros(dem_shape, dtype=bool)
    if window is not None:
        logger.debug(f"Applying outlier mask to window with shape {outlier_mask.shape}")
        final_mask[
            window.row_off : window.row_off + window.height,
            window.col_off : window.col_off + window.width,
        ] = outlier_mask
    else:
        logger.debug("Applying outlier mask to entire DEM")
        final_mask = outlier_mask

    logger.info(
        f"Masked {final_mask.sum()} outliers in total ({final_mask.sum() / final_mask.size * 100:.4f}% of all pixels)"
    )

    # Apply mask to DEM and save
    logger.info("Applying final mask to DEM...")
    filtered_dem = apply_mask_to_dem(dem_path=dem_path, mask=final_mask)

    return filtered_dem, final_mask


def apply_mask_to_dem(
    dem_path: Union[Path, str],
    mask: np.ndarray,
    output_path: Optional[Union[Path, str]] = None,
) -> xdem.DEM | None:
    """
    Apply a mask to a DEM, setting masked pixels to NaN, and optionally save the result.

    Args:
        dem_path: Path to DEM file
        mask: Boolean array where True indicates pixels to be masked
        output_path: Optional path to save the masked DEM (default: None)

    Returns:
        xdem.DEM: The masked DEM object if output_path is None, otherwise None
    """
    logger.info(f"Applying mask to DEM {dem_path}")
    logger.debug(
        f"Mask has {np.sum(mask)} True values out of {mask.size} pixels ({np.sum(mask) / mask.size * 100:.4f}%)"
    )

    dem = xdem.DEM(dem_path)
    dem.load()
    dem.set_mask(mask)

    # Save masked DEM if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dem.save(output_path)
        logger.info(f"Saved masked DEM to {output_path}")
        return None
    else:
        return dem


if __name__ == "__main__":
    # # Add some random noise to the DEM
    # import numpy as np
    # import scipy.ndimage

    # # Noise parameters
    # noise_scale = 5  # base magnitude of noise (meters)
    # correlation_length = 10  # correlation length in pixels
    # noise_variability = 0.0001  # how much the noise magnitude varies spatially (0-1)
    # salt_papper_noise_scale = 20
    # salt_papper_noise_prob = 0.005

    # # Load the DEM
    # dem_path = "dem_cut.tif"
    # np.random.seed(42)
    # dem = xdem.DEM(dem_path)
    # dem.load()

    # # Load a reference DEM to compute the difference
    # ref_dem = "data/ref_data/Hofsjokull_20131013_zmae_ps.tif"
    # ref_dem = xdem.DEM(ref_dem)
    # ref_dem.reproject(dem, inplace=True)

    # # Create spatially correlated random field for noise magnitude
    # # This determines how strong the noise will be at each location
    # magnitude_field = np.random.random(dem.shape)
    # magnitude_field = scipy.ndimage.gaussian_filter(
    #     magnitude_field, sigma=correlation_length
    # )
    # magnitude_field = (magnitude_field - np.min(magnitude_field)) / (
    #     np.max(magnitude_field) - np.min(magnitude_field)
    # )
    # magnitude_field = noise_variability * magnitude_field + (
    #     1 - noise_variability
    # )  # Scale between (1-noise_variability) and 1

    # # Create spatially correlated random fields for noise pattern
    # noise_pattern = np.random.normal(0, 1, dem.shape)
    # noise_pattern = scipy.ndimage.gaussian_filter(
    #     noise_pattern, sigma=correlation_length / 2
    # )

    # # Normalize the noise pattern to have standard deviation = 1
    # noise_pattern = noise_pattern / np.std(noise_pattern)

    # # Apply scaling to get the final noise
    # noise = noise_scale * magnitude_field * noise_pattern

    # # Add some salt and pepper noise on top
    # sp_mask = np.random.choice(
    #     [0, 1], size=dem.shape, p=[1 - salt_papper_noise_prob, salt_papper_noise_prob]
    # ).astype(bool)
    # salt_papper_noise = np.random.normal(0, salt_papper_noise_scale, size=dem.shape)
    # noise[sp_mask] += salt_papper_noise[sp_mask]

    # # Check the actual noise scale
    # actual_mean_abs = np.mean(np.abs(noise))
    # actual_std = np.std(noise)
    # actual_max = np.max(np.abs(noise))

    # print("Noise statistics before application:")
    # print(f"  Mean absolute value: {actual_mean_abs:.2f} m")
    # print(f"  Standard deviation: {actual_std:.2f} m")
    # print(f"  Max absolute value: {actual_max:.2f} m")

    # # Apply the noise to the DEM
    # dem.data += noise
    # dem.save("dem_noisy.tif")

    # # Compute the difference between the DEM and a reference DEM
    # dh = dem - ref_dem
    # dh.save("dem_dh_noisy.tif")

    # Load the elevation bands
    elevation_bands_path = Path("data/Hofsjokull_elevbands_100m.geojson")

    # Filter the DEM by using a geometry vector (eg., elevation bands)
    filtered, mask = filter_dem(
        dem_path="dem_dh_noisy.tif",
        boundary_path=elevation_bands_path,
        method=OutlierMethod.DISTANCE,
        radius=10,
        outlier_threshold=5,
    )
    filtered.save("dem_dh_filtered_dist.tif")
    apply_mask_to_dem(
        dem_path="dem_dh_noisy.tif",
        mask=mask,
        output_path="dem_filtered_dist.tif",
    )

    # Filter the DEM by using a geometry vector (eg., elevation bands)
    dh_filtered, mask = filter_dem_by_geometry(
        dem_path="dem_dh_noisy.tif",
        geometry_path=elevation_bands_path,
        method=OutlierMethod.NMAD,
        # radius=50,
        outlier_threshold=2,
        n_jobs=-1,
    )
    dh_filtered.save("dem_dh_filtered_eb.tif")
    apply_mask_to_dem(
        dem_path="dem_dh_noisy.tif",
        mask=mask,
        output_path="dem_filtered_eb.tif",
    )

    # Filter the entire DEM using the filter method
    # dh_filtered, mask = filter_dem(
    #     dem_path=dem_dh_path,
    #     boundary_path=geoemtry_path,
    #     filter_method=filter_method,
    #     outlier_threshold=outlier_threshold,
    # )

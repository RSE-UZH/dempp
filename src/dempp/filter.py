from __future__ import annotations

import logging
import warnings
from enum import Enum
from pathlib import Path

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
from xdem.spatialstats import nmad

try:
    import cv2

    _has_cv2 = True
except ImportError:
    _has_cv2 = False
import scipy
from xdem._typing import NDArrayf

logger = logging.getLogger("dempp")


class OutlierMethod(Enum):
    """Methods for outlier detection"""

    NMAD = "nmad"  # median/nmad
    ZSCORE = "zscore"  # z-score
    NORMAL = "normal"  # mean/std
    GAUSS_SCIPY = "gauss_scipy"  # Gaussian filter using scipy
    GAUSS_CV = "gauss_cv"  # Gaussian filter using OpenCV
    DISTANCE = "distance"  # Distance filter


def gaussian_filter_scipy(array: NDArrayf, sigma: float) -> NDArrayf:  # type: ignore
    """
    Apply a Gaussian filter to a raster that may contain NaNs, using scipy's implementation.
    gaussian_filter_cv is recommended as it is usually faster, but this depends on the value of sigma.

    N.B: kernel_size is set automatically based on sigma.

    :param array: the input array to be filtered.
    :param sigma: the sigma of the Gaussian kernel

    :returns: the filtered array (same shape as input)
    """
    # Check that array dimension is 2 or 3
    if np.ndim(array) not in [2, 3]:
        raise ValueError(
            f"Invalid array shape given: {array.shape}. Expected 2D or 3D array."
        )

    # In case array does not contain NaNs, use scipy's gaussian filter directly
    if np.count_nonzero(np.isnan(array)) == 0:
        return scipy.ndimage.gaussian_filter(array, sigma=sigma)

    # If array contain NaNs, need a more sophisticated approach
    # Inspired by https://stackoverflow.com/a/36307291
    else:
        # Run filter on a copy with NaNs set to 0
        array_no_nan = array.copy()
        array_no_nan[np.isnan(array)] = 0
        gauss_no_nan = scipy.ndimage.gaussian_filter(array_no_nan, sigma=sigma)
        del array_no_nan

        # Mask of NaN values
        nan_mask = 0 * array.copy() + 1
        nan_mask[np.isnan(array)] = 0
        gauss_mask = scipy.ndimage.gaussian_filter(nan_mask, sigma=sigma)
        del nan_mask

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered")
            gauss = gauss_no_nan / gauss_mask

        return gauss


def gaussian_filter_cv(array: NDArrayf, sigma: float) -> NDArrayf:  # type: ignore
    """
    Apply a Gaussian filter to a raster that may contain NaNs, using OpenCV's implementation.
    Arguments are for now hard-coded to be identical to scipy.

    N.B: kernel_size is set automatically based on sigma

    :param array: the input array to be filtered.
    :param sigma: the sigma of the Gaussian kernel

    :returns: the filtered array (same shape as input)
    """
    if not _has_cv2:
        raise ValueError("Optional dependency needed. Install 'opencv'.")

    # Check that array dimension is 2, or can be squeezed to 2D
    orig_shape = array.shape
    if len(orig_shape) == 2:
        pass
    elif len(orig_shape) == 3:
        if orig_shape[0] == 1:
            array = array.squeeze()
        else:
            raise NotImplementedError("Case of array of dimension 3 not implemented.")
    else:
        raise ValueError(
            f"Invalid array shape given: {orig_shape}. Expected 2D or 3D array."
        )

    # In case array does not contain NaNs, use OpenCV's gaussian filter directly
    # With kernel size (0, 0), i.e. set to default, and borderType=BORDER_REFLECT, the output is equivalent to scipy
    if np.count_nonzero(np.isnan(array)) == 0:
        gauss = cv2.GaussianBlur(
            array, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT
        )

    # If array contain NaNs, need a more sophisticated approach
    # Inspired by https://stackoverflow.com/a/36307291
    else:
        # Run filter on a copy with NaNs set to 0
        array_no_nan = array.copy()
        array_no_nan[np.isnan(array)] = 0
        gauss_no_nan = cv2.GaussianBlur(
            array_no_nan, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT
        )
        del array_no_nan

        # Mask of NaN values
        nan_mask = 0 * array.copy() + 1
        nan_mask[np.isnan(array)] = 0
        gauss_mask = cv2.GaussianBlur(
            nan_mask, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT
        )
        del nan_mask

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered")
            gauss = gauss_no_nan / gauss_mask

    return gauss.reshape(orig_shape)


def distance_filter(
    array: NDArrayf,  # type: ignore
    radius: float,
    outlier_threshold: float,
) -> NDArrayf:  # type: ignore
    """
    Filter out pixels whose value is distant more than a set threshold from the average value of all neighbor \
pixels within a given radius.
    Filtered pixels are set to NaN.

    TO DO: Add an option on how the "average" value should be calculated, i.e. using a Gaussian, median etc filter.

    :param array: the input array to be filtered.
    :param radius: the radius in which the average value is calculated (for Gaussian filter, this is sigma).
    :param outlier_threshold: the minimum difference abs(array - mean) for a pixel to be considered an outlier.

    :returns: the filtered array (same shape as input)
    """
    outliers = _distance_filter(array, radius, outlier_threshold)
    out_array = np.copy(array)
    out_array[outliers] = np.nan

    return outliers


def _distance_filter(
    array: NDArrayf,  # type: ignore
    radius: float,
    outlier_threshold: float,
) -> NDArrayf:  # type: ignore
    """
    Filter out pixels whose value is distant more than a set threshold from the average value of all neighbor \
pixels within a given radius.
    Filtered pixels are set to NaN.

    TO DO: Add an option on how the "average" value should be calculated, i.e. using a Gaussian, median etc filter.

    :param array: the input array to be filtered.
    :param radius: the radius in which the average value is calculated (for Gaussian filter, this is sigma).
    :param outlier_threshold: the minimum difference abs(array - mean) for a pixel to be considered an outlier.

    :returns: the filtered array (same shape as input)
    """
    # Calculate the average value within the radius
    smooth = gaussian_filter_cv(array, sigma=radius)

    # Filter outliers
    outliers = (np.abs(array - smooth)) > outlier_threshold

    return outliers


def _zscore_filter(
    values: np.ndarray,
    outlier_threshold: float = 3,
    **kwargs,
) -> np.ndarray:
    """Find outliers using z-score method.

    This function computes z-scores for the array and identifies values that deviate
    more than a specified threshold from the mean in terms of standard deviations.

    Args:
        values (np.ndarray): Input 2D array to check for outliers.
        outlier_threshold (float, optional): maximum value of the Z distribution that is accepted. Values with |z-score| > outlier_threshold are considered outliers. Defaults to 3.
        **kwargs: Additional parameters passed to zscore function.
            See scipy.stats.zscore for more details.
            Note: 'nan_policy' is set to 'omit' by default to handle NaN values.
            If you want to include NaN values in the calculation, set 'nan_policy' to 'raise'.


    Returns:
        np.ndarray: Boolean array marking outliers (True = outlier).
            Masked pixels (if mask provided) will be False in output.
    """
    outliers = np.zeros_like(values, dtype=bool)
    nan_policy = kwargs.pop("nan_policy", "omit")
    z_scores = zscore(values, nan_policy=nan_policy, **kwargs)
    outliers = np.abs(z_scores) > outlier_threshold

    return outliers


def _normal_filter(
    values: np.ndarray,
    n_limit: float = 3,
) -> np.ndarray:
    """Find outliers using normal distribution method (mean/std).

    This function identifies outliers based on how far values deviate from the mean
    in terms of standard deviations.

    Args:
        values (np.ndarray): Input 2D array to check for outliers.
        n_limit (float, optional): Number of standard deviations for outlier threshold. Values with |value-mean| > n_limit*std are considered outliers. Defaults to 3.

    Returns:
        np.ndarray: Boolean array marking outliers (True = outlier).
            Masked pixels (if mask provided) will be False in output.
    """
    outliers = np.zeros_like(values, dtype=bool)
    mean = np.mean(values)
    std = np.std(values)
    outliers = np.abs(values - mean) > n_limit * std

    return outliers


def _nmad_filter(
    values: np.ndarray,
    outlier_threshold: float = 3,
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
    outliers = np.zeros_like(values, dtype=bool)
    median = np.median(values)
    mad_value = nmad(values)
    outliers = np.abs(values - median) > outlier_threshold * mad_value

    return outliers


def find_outliers(
    dem_path: Path | str,
    boundary: gpd.GeoDataFrame | dict | None = None,
    method: OutlierMethod | str = OutlierMethod.NMAD,
    **kwargs,
) -> tuple[np.ndarray, Window]:
    """Process a single glacier and return its outlier mask.

    Args:
        dem_path (Path | str): Path to the DEM file.
        boundary (gpd.GeoDataFrame | dict): Vector geometry with the boundaries that will be used to filter the DEM within it. It can be a GeoDataFrame or a dictionary compatible with the __geo_interface__ format. Defaults to None.
        method (OutlierMethod, optional): Outlier detection method. Defaults to OutlierMethod.NMAD.
        **kwargs: Additional keyword arguments passed to the filter method.

    Returns:
        Tuple[np.ndarray, Window]: The outlier mask and the DEM window.
    """
    if isinstance(method, str):
        method = OutlierMethod[method.upper()]

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

            # If no valid data in the window, return empty mask
            if not np.any(valid_mask):
                logger.debug("No valid data in geometry")
                return np.zeros_like(dem_data, dtype=bool), window

            # Create a copy of data for filtering
            data_for_filtering = dem_data.copy()

            # Replace nodata/outside geometry with NaN for filter processing
            data_for_filtering[~valid_mask] = np.nan

        else:
            # Load full DEM
            window = None
            dem_data = src.read(1)

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

    # If there are no valid values to process, return empty mask
    if not np.any(valid_mask):
        logger.debug("No valid data to process")
        return np.zeros_like(dem_data, dtype=bool), window

    # Apply the appropriate filter to detect outliers
    # Each filter function should handle NaNs and return a boolean mask
    if method == OutlierMethod.NMAD:
        outliers = _nmad_filter(data_for_filtering, **kwargs)
    elif method == OutlierMethod.DISTANCE:
        outliers = _distance_filter(data_for_filtering, **kwargs)
    elif method == OutlierMethod.ZSCORE:
        outliers = _zscore_filter(data_for_filtering, **kwargs)
    elif method == OutlierMethod.NORMAL:
        outliers = _normal_filter(data_for_filtering, **kwargs)
    elif method == OutlierMethod.GAUSS_SCIPY:
        raise NotImplementedError(
            "Gaussian filter using scipy is not implemented yet. Use GAUSS_CV instead."
        )
    elif method == OutlierMethod.GAUSS_CV:
        raise NotImplementedError(
            "Gaussian filter using OpenCV is not implemented yet. Use GAUSS_SCIPY instead."
        )
    else:
        raise ValueError(f"Unknown outlier method: {method}")

    # Ensure only valid data can be outliers (nodata and outside geometry pixels are not outliers)
    outliers = outliers & valid_mask

    return outliers, window


def filter_dem_by_geometry(
    dem_path: Path | str,
    geometry_path: Path | str,
    method: OutlierMethod = OutlierMethod.NMAD,
    n_jobs: int = -1,
    **kwargs,
) -> tuple[xdem.DEM, np.ndarray]:
    """Process multiple glaciers and combine their outlier masks.

    Args:
        dem_path (Path | str): Path to DEM file.
        geometry_path (Path | str): Path to glacier outlines (GeoJSON).
        output_path (Path | str): Path to save the output DEM file. Defaults to None.
        method (OutlierMethod optional): Outlier detection method. Defaults to OutlierMethod.NMAD.
        n_jobs (int, optional): Number of parallel jobs (-1 for all cores, 1 to process sequentially). Defaults to -1.
        **kwargs: Additional keyword arguments to be passed to the find_outliers() function.

    Returns:
        Tuple[xdem.DEM, np.ndarray]: The filtered DEM and the final outlier mask.
    """
    # Load geometry vector (e.g., glacier outlines)
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        dem_shape = src.shape
        geometries = gpd.read_file(geometry_path).to_crs(dem_crs)

    # Process different geometries in parallel
    logger.info("Filtering DEM based on elevation bands for each glacier...")
    with logging_redirect_tqdm([logger]):
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(find_outliers)(
                dem_path=dem_path,
                boundary=feature,
                method=method,
                **kwargs,
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
    method: OutlierMethod | list[OutlierMethod] = OutlierMethod.NMAD,
    **kwargs,
) -> tuple[xdem.DEM, np.ndarray]:
    """
    filter_dem Filter a DEM, optionally within a given boundary (e.g. glacier) using outlier detection.

    Args:
        dem_path (Path | str): Path to DEM file.
        output_path (Path | str): Path to save the output DEM file. Defaults to None.
        boundary (gpd.GeoDataFrame | Path | str | None): Boundary geometry to filter the DEM within. Defaults to None.
        method (OutlierMethod | List[OutlierMethod], optional): Outlier detection method. Defaults to OutlierMethod.NMAD.

    Returns:
        tuple[xdem.DEM, np.ndarray]: The filtered DEM and the final outlier mask.
    """

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        dem_shape = src.shape

    if boundary_path is not None:
        boundary = gpd.read_file(boundary_path).to_crs(dem_crs)
        boundary = boundary.to_crs(dem_crs)
    else:
        boundary = None

    # Get outlier mask
    outlier_mask, window = find_outliers(
        dem_path=dem_path, boundary=boundary, method=method, **kwargs
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

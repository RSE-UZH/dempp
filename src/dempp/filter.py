import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio as rio
import xdem
from joblib import Parallel, delayed
from rasterio import features, warp
from rasterio.crs import CRS
from rasterio.windows import Window
from scipy.stats import zscore
from shapely.geometry import Polygon
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from xdem.spatialstats import nmad

logger = logging.getLogger("dempp")


class OutlierMethod(Enum):
    """Methods for outlier detection"""

    ZSCORE = "zscore"  # z-score
    NORMAL = "normal"  # mean/std
    ROBUST = "robust"  # median/nmad


@dataclass(kw_only=True)
class DEMWindow:
    """Class to store DEM window and its metadata"""

    data: np.ndarray
    window: Window
    bounds: tuple[float, float, float, float]
    transform: rio.Affine
    no_data: float = None
    mask: np.ndarray = None
    crs: CRS | str | int = None

    def to_file(self, path: Path):
        """Save DEM window to file"""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with rio.open(
            path,
            "w",
            driver="GTiff",
            height=self.window.height,
            width=self.window.width,
            count=1,
            dtype=self.data.dtype,
            crs=self.crs,
            transform=self.transform,
            nodata=self.no_data,
        ) as dst:
            dst.write(self.data, 1)


def extract_dem_window(
    dem_path: str, geom: gpd.GeoDataFrame | Polygon, padding: int = 1
) -> DEMWindow:
    """Extract a window from DEM based on geometry bounds with padding.

    Args:
        dem_path (str): Path to the DEM file.
        geom (gpd.GeoDataFrame | Polygon): Geometry to extract window for. if GeoDataFrame, uses first geometry.
        padding (int, optional): Padding to add around the geometry bounds. Defaults to 1.

    Returns:
        DEMWindow: The extracted DEM window and its metadata.
    """
    with rio.open(dem_path, mode="r") as src:
        # Get raster bounds
        raster_bounds = src.bounds

        # Get geometry bounds based on input type
        if isinstance(geom, gpd.GeoDataFrame):
            geom_bounds = geom.bounds.values[0]
            geom_id = f"ID: {geom.index[0]}"
        else:  # Polygon
            geom_bounds = geom.bounds
            geom_id = "Polygon"
        minx, miny, maxx, maxy = geom_bounds

        # Check intersection
        if not (
            minx < raster_bounds.right
            and maxx > raster_bounds.left
            and miny < raster_bounds.top
            and maxy > raster_bounds.bottom
        ):
            logger.debug(f"Geometry ({geom_id}) does not overlap with raster bounds")
            return None

        try:
            # Convert bounds to pixel coordinates
            row_start, col_start = src.index(minx, maxy)
            row_stop, col_stop = src.index(maxx, miny)
        except IndexError:
            logger.debug(f"Geometry ({geom_id}) coordinates outside raster bounds")
            return None

        # Add padding
        row_start = max(0, row_start - padding)
        col_start = max(0, col_start - padding)
        row_stop = min(src.height, row_stop + padding)
        col_stop = min(src.width, col_stop + padding)

        if row_stop <= row_start or col_stop <= col_start:
            logger.debug(f"Invalid window dimensions for geometry ({geom_id})")
            return None

        # Rest of the function remains the same
        window = Window(
            col_start, row_start, col_stop - col_start, row_stop - row_start
        )
        transform = src.window_transform(window)
        data = src.read(1, window=window)
        mask = (
            data == src.nodata
            if src.nodata is not None
            else np.zeros_like(data, dtype=bool)
        )
        bounds = rio.windows.bounds(window, src.transform)
        if src.crs is None:
            crs = None
        else:
            crs = src.crs if isinstance(src.crs, CRS) else CRS.from_string(src.crs)

    return DEMWindow(
        data=data,
        window=window,
        bounds=bounds,
        transform=transform,
        no_data=src.nodata,
        mask=mask,
        crs=crs,
    )


def vector_to_mask(
    geometry: gpd.GeoDataFrame | Polygon,
    window_shape: tuple[int, int],
    transform: rio.Affine,
    crs: CRS | None = None,
    buffer: int | float = 0,
    bounds: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Creates a rasterized boolean mask for vector geometries.

    Converts vector geometry to a raster mask using the provided spatial
    reference system and transform. Optionally applies a buffer to the geometries
    and crops to specified bounds.

    Args:
        geometry: Vector data as either GeoDataFrame or Shapely Polygon.
        window_shape: Output raster dimensions as (height, width).
        transform: Affine transform defining the raster's spatial reference.
        crs: Coordinate reference system for output raster. If None, uses geometry's CRS.
            Required when input is a Polygon or when GeoDataFrame has no CRS.
        buffer: Distance to buffer geometries. Zero means no buffer. Defaults to 0.
        bounds: Spatial bounds as (left, bottom, right, top). If None, uses geometry bounds.

    Returns:
        Boolean mask where True indicates the geometry.

    Raises:
        TypeError: If buffer is not a number.
        ValueError: If geometry is empty or invalid, or if CRS is None when required.
    """
    # Convert Polygon to GeoDataFrame if needed
    if not isinstance(geometry, gpd.GeoDataFrame):
        if crs is None:
            raise ValueError("CRS must be provided when input is a Polygon")
        geometry = gpd.GeoDataFrame(geometry=[geometry], crs=crs)

    # Make copy to avoid modifying input
    gdf = geometry.copy()

    # Handle CRS - crucial when working with features from iterfeatures()
    if crs is None:
        if gdf.crs is None:
            raise ValueError("CRS must be provided when GeoDataFrame has no CRS")
        target_crs = gdf.crs
    else:
        # If GeoDataFrame has no CRS but crs parameter is provided, set it
        target_crs = crs
        if gdf.crs is None:
            logger.debug("Setting CRS on GeoDataFrame that lacks one")
            gdf.set_crs(target_crs, inplace=True)

    # Crop to bounds if provided
    if bounds is not None:
        left, bottom, right, top = bounds
        try:
            x1, y1, x2, y2 = warp.transform_bounds(
                target_crs, gdf.crs, left, bottom, right, top
            )
            gdf = gdf.cx[x1:x2, y1:y2]
        except Exception as e:
            logger.warning(f"Error cropping to bounds: {e}")
            # Continue with uncropped geometries

    # Reproject to target CRS if needed
    if gdf.crs != target_crs:
        try:
            gdf = gdf.to_crs(target_crs)
        except ValueError as e:
            logger.warning(f"CRS transformation error: {e}. Setting CRS explicitly.")
            # Handle case where GeoDataFrame has invalid/incompatible CRS
            gdf.set_crs(target_crs, inplace=True)

    # Apply buffer if requested
    if buffer != 0:
        if not isinstance(buffer, (int, float)):
            raise TypeError(f"Buffer must be number, got {type(buffer)}")
        gdf.geometry = [geom.buffer(buffer) for geom in gdf.geometry]

    # Validate shapes are not empty
    if gdf.empty or gdf.geometry.is_empty.all():
        logger.warning("No valid geometries found after processing")
        return np.zeros(window_shape, dtype=bool)

    # Handle possible MultiPolygons that might cause issues with rasterization
    if any(geom.geom_type.startswith("Multi") for geom in gdf.geometry):
        # Explode MultiPolygons into individual Polygons
        gdf = gdf.explode(index_parts=False)

    # Rasterize geometries
    try:
        mask = features.rasterize(
            shapes=gdf.geometry,
            out_shape=window_shape,
            transform=transform,
            fill=0,
            default_value=1,
            dtype="uint8",
        ).astype(bool)
    except Exception as e:
        logger.error(f"Rasterization failed: {e}")
        # Return empty mask in case of failure
        return np.zeros(window_shape, dtype=bool)

    return mask


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
        geometry (gpd.GeoDataFrame | dict): Vector geometry with the boundaries that will be used to filter the DEM within it. It can be a GeoDataFrame or a dictionary compatible with the __geo_interface__ format (as it comes from geodataframe.iterfeatures()). Defaults to None.
        filter_method (OutlierMethod | List[OutlierMethod], optional): Outlier detection method. If a list is provided, the filters will be applied sequentially and the final mask will be the union of all masks. Defaults to OutlierMethod.ROBUST.
        outlier_threshold (float, optional): Number of std/mad for outlier threshold. Defaults to 3.
        use_elevation_bands (bool, optional): Whether to use elevation bands for filtering. Defaults to True.
        elevation_band_width (float): Width of elevation bands.
        geometry_id (str, optional): Optional geometry_id for logging. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the filter_by_elevation_bands() function.

    Returns:
        Tuple[np.ndarray, Window]: The outlier mask, the DEM window and the elevation bands statistics (if use_elevation_bands=True, else None).
    """

    if isinstance(filter_method, list):
        raise NotImplementedError(
            "Multiple filter methods are not implemented yet. Please use a single method."
        )

    if boundary is not None:
        if not isinstance(boundary, (gpd.GeoDataFrame, dict)):
            raise ValueError(
                "Boundary must be a GeoDataFrame or a dictionary compatible with the __geo_interface__ format."
            )
        if isinstance(boundary, dict):
            boundary = gpd.GeoDataFrame.from_features([boundary])

        # Extract DEM window around the given geometry
        dem_window = extract_dem_window(dem_path, boundary)
        if dem_window is None:
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
            "DEM is completely masked within the fiven geometry, returning empty mask"
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

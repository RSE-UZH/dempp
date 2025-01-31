import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio as rio
import rasterio.mask
from joblib import Parallel, delayed
from rasterio import features, warp
from rasterio.crs import CRS
from rasterio.windows import Window
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import gaussian_filter, median_filter
from scipy.stats import zscore

logger = logging.getLogger("pyasp")


class OutlierMethod(Enum):
    """Methods for outlier detection"""

    ZSCORE = "zscore"  # z-score
    NORMAL = "normal"  # mean/std
    ROBUST = "robust"  # median/nmad


@dataclass
class DEMWindow:
    """Class to store DEM window and its metadata"""

    data: np.ndarray
    window: Window
    bounds: tuple[float, float, float, float]
    transform: rasterio.Affine
    no_data: float = None
    mask: np.ndarray = None
    crs: CRS | str | int = None

    def to_file(self, path: Path):
        """Save DEM window to file"""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
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


@dataclass
class ElevationBand:
    label: str
    band_lower: float
    band_upper: float
    data: np.ma.MaskedArray


def round_to_decimal(
    x: float, decimal: int = 0, func: Callable = np.round, **kwargs
) -> float:
    """Round a number to a specified number of decimal places.

    Args:
        x (float): The number to round.
        decimal (int, optional): The number of decimal places to round to. Defaults to 0.
        func (Callable, optional): The rounding function to use. Defaults to np.round.
        **kwargs: Additional arguments to pass to the rounding function.

    Returns:
        float: The rounded number.
    """
    multiplier = 10.0**decimal
    return func(x / multiplier, **kwargs) * multiplier


def extract_dem_window(
    dem_path: str, geom: gpd.GeoDataFrame, padding: int = 1
) -> DEMWindow:
    """Extract a window from DEM based on geometry bounds with padding.

    Args:
        dem_path (str): Path to the DEM file.
        geom (gpd.GeoDataFrame): Geometry to extract the window for.
        padding (int, optional): Padding to add around the geometry bounds. Defaults to 1.

    Returns:
        DEMWindow: The extracted DEM window and its metadata.
    """
    with rasterio.open(dem_path, mode="r") as src:
        # Get raster bounds
        raster_bounds = src.bounds

        # Get geometry bounds
        geom_bounds = geom.bounds.values[0]
        minx, miny, maxx, maxy = geom_bounds

        # Check intersection
        if not (
            minx < raster_bounds.right
            and maxx > raster_bounds.left
            and miny < raster_bounds.top
            and maxy > raster_bounds.bottom
        ):
            logger.debug(
                f"Geometry (ID: {geom.index[0]}) does not overlap with raster bounds"
            )
            return None

        try:
            # Convert bounds to pixel coordinates
            row_start, col_start = src.index(minx, maxy)
            row_stop, col_stop = src.index(maxx, miny)
        except IndexError:
            logger.debug(
                f"Geometry (ID: {geom.index[0]}) coordinates outside raster bounds"
            )
            return None

        # Add padding
        row_start = max(0, row_start - padding)
        col_start = max(0, col_start - padding)
        row_stop = min(src.height, row_stop + padding)
        col_stop = min(src.width, col_stop + padding)

        if row_stop <= row_start or col_stop <= col_start:
            logger.debug(
                f"Invalid window dimensions for geometry (ID: {geom.index[0]})"
            )
            return None

        # Create valid window
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
        bounds = rasterio.windows.bounds(window, src.transform)
        if src.crs is None:
            crs = None
        else:
            crs = src.crs if isinstance(src.crs, CRS) else CRS.from_string(src.crs)

    return DEMWindow(data, window, bounds, transform, src.nodata, mask, crs)


def vector_to_mask(
    geometry: gpd.GeoDataFrame | gpd.GeoSeries,
    window_shape: tuple[int, int],
    transform: rio.Affine,
    crs: CRS | None = None,
    buffer: int | float = 0,
    bounds: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Creates a rasterized boolean mask for vector geometries.

    Converts vector geometry to a raster mask using the provided spatial
    reference system and transform. Optionally applies a buffer to the geometries and crops to specified bounds.

    Args:
        geometry (gpd.GeoDataFrame | gpd.GeoSeries): Vector data.
        window_shape (tuple[int, int]): Output raster dimensions as (height, width).
        transform (rio.Affine): Affine transform defining the raster's spatial reference.
        crs (CRS | None, optional): Coordinate reference system for output raster. If None, uses geometry's CRS.
        buffer (int | float, optional): Distance to buffer geometries. Zero means no buffer. Defaults to 0.
        bounds (tuple[float, float, float, float] | None, optional): Spatial bounds as (left, bottom, right, top). If None, uses geometry bounds.

    Returns:
        np.ndarray: Boolean mask where True indicates the geometry.

    Raises:
        TypeError: If buffer is not a number.
        ValueError: If geometry is empty or invalid.
    """
    # Convert GeoSeries to GeoDataFrame if needed
    if isinstance(geometry, gpd.GeoSeries):
        geometry = gpd.GeoDataFrame(geometry=geometry)

    # Make copy to avoid modifying input
    gdf = geometry.copy()

    # Set CRS if not provided
    if crs is None:
        crs = gdf.crs

    # Crop to bounds if provided
    if bounds is not None:
        left, bottom, right, top = bounds
        x1, y1, x2, y2 = warp.transform_bounds(crs, gdf.crs, left, bottom, right, top)
        gdf = gdf.cx[x1:x2, y1:y2]

    # Reproject to target CRS
    gdf = gdf.to_crs(crs)

    # Apply buffer if requested
    if buffer != 0:
        if not isinstance(buffer, (int | float)):
            raise TypeError(f"Buffer must be number, got {type(buffer)}")
        gdf.geometry = [geom.buffer(buffer) for geom in gdf.geometry]

    # Validate shapes are not empty
    if gdf.empty:
        logger.warning("No valid geometries found after processing")
        return np.zeros(window_shape, dtype=bool)

    # Rasterize geometries
    mask = features.rasterize(
        shapes=gdf.geometry,
        out_shape=window_shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype="uint8",
    ).astype(bool)

    return mask


def extract_elevation_bands(
    dem: np.ma.MaskedArray | np.ndarray,
    band_width: float,
    mask: np.ndarray = None,
    smooth_dem: bool = True,
) -> list[ElevationBand]:
    """Extract elevation bands from a DEM.

    Args:
        dem (np.ma.MaskedArray | np.ndarray): The DEM data.
        band_width (float): The width of each elevation band.
        mask (np.ndarray, optional): Mask for the DEM data. Defaults to None.
        smooth_dem (bool, optional): Whether to smooth the DEM before extracting bands. Defaults to True.

    Returns:
        List[ElevationBand]: List of elevation bands.
    """
    if isinstance(dem, np.ma.MaskedArray):
        dem_data = dem.data
        mask = dem.mask
    elif isinstance(dem, np.ndarray):
        dem_data = dem
        if mask is None:
            logger.warning("No mask provided, assuming np.nan as nodata")
            mask = np.isnan(dem_data)
    else:
        raise ValueError("Invalid dem input data type")

    if smooth_dem:
        # Create smoothed array for extracting elevation bands
        dem_smooth = dem_data.copy()
        dem_smooth[mask] = np.nan

        # Apply median filter to remove outliers
        filter_size = min(5, round(min(dem_smooth.shape) / 4))
        dem_smooth = median_filter(dem_smooth, size=filter_size)

        # Smoot dem with a gaussian filter (temporary fill nans with nearest neighbor and exclude them after smoothing)
        nan_mask = np.isnan(dem_smooth)
        interpolator = NearestNDInterpolator(
            np.argwhere(~nan_mask), dem_smooth[~nan_mask]
        )
        dem_smooth[nan_mask] = interpolator(np.argwhere(nan_mask))
        dem_smooth = gaussian_filter(dem_smooth, sigma=filter_size)
        dem_smooth[mask] = np.nan

        # Write to file both the original dem and the smoothed dem for debugging
        # with tempfile.NamedTemporaryFile(suffix="demdata.tif", dir=".") as tmp:
        #     with rasterio.open(
        #         tmp.name,
        #         "w",
        #         driver="GTiff",
        #         height=dem_data.shape[0],
        #         width=dem_data.shape[1],
        #         count=1,
        #         dtype=dem_data.dtype,
        #         crs=CRS.from_epsg(32632),
        #         transform=rio.Affine(10, 0, 0, 0, -10, 0),
        #         nodata=-9999,
        #     ) as dst:
        #         dst.write(dem_data, 1)

        # with tempfile.NamedTemporaryFile(suffix="demdata_smoothed.tif", dir=".") as tmp:
        #     with rasterio.open(
        #         tmp.name,
        #         "w",
        #         driver="GTiff",
        #         height=dem_smooth.shape[0],
        #         width=dem_smooth.shape[1],
        #         count=1,
        #         dtype=dem_smooth.dtype,
        #         crs=CRS.from_epsg(32632),
        #         transform=rio.Affine(10, 0, 0, 0, -10, 0),
        #         nodata=np.nan,
        #     ) as dst:
        #         dst.write(dem_smooth, 1)
    else:
        dem_smooth = dem_data

    # Get robust min/max values for elevation bands and round to precision
    round_decimal = int(np.log10(band_width))
    min_elev = np.percentile(dem_data[~mask], 1)
    max_elev = np.percentile(dem_data[~mask], 99)
    min_elev_bands = round_to_decimal(min_elev, round_decimal, np.floor)
    max_elev_bands = round_to_decimal(max_elev, round_decimal, np.ceil)

    # Create elevation bands from robust statistics
    bands = np.arange(min_elev_bands, max_elev_bands + band_width, band_width)
    logger.debug(f"Found {len(bands) - 1} elevation bands")
    logger.debug(f"Min band elevation: {min_elev_bands:.0f} m")
    logger.debug(f"Max band elevation: {max_elev_bands:.0f} m")

    # Extract elevation bands
    elevation_bands = []
    for i in range(len(bands) - 1):
        band_lower = bands[i]
        band_upper = bands[i + 1]
        label = f"{band_lower:.0f}-{band_upper:.0f}"

        # Create band mask (True where pixels should be masked)
        band_mask = ~((dem_smooth >= band_lower) & (dem_smooth < band_upper)) | mask

        # Create masked array for band
        band_data = np.ma.masked_array(dem_data, mask=band_mask)

        # Add band to list
        elevation_bands.append(
            ElevationBand(
                label=label,
                band_lower=band_lower,
                band_upper=band_upper,
                data=band_data,
            )
        )

    return elevation_bands


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
        nmad = 1.4826 * np.median(np.abs(valid_values - median))
        outliers[~mask] = np.abs(data[~mask] - median) > n_limit * nmad

    else:
        raise ValueError(f"Unknown outlier method: {method}")

    return outliers


def filter_by_elevation_bands(
    dem: np.ndarray,
    band_width: float,
    method: OutlierMethod | list[OutlierMethod] = OutlierMethod.ROBUST,
    n_limit: float = 3,
    n_jobs: int = None,
) -> np.ndarray:
    """Filter DEM by elevation bands using numpy operations.

    Args:
        dem (np.ndarray): The DEM data.
        band_width (float): The width of each elevation band.
        method (OutlierMethod | List[OutlierMethod], optional): Outlier detection method. If a list is provided, the filters will be applied sequentially and the final mask will be the union of all masks. Defaults to OutlierMethod.ROBUST.
        n_limit (float, optional): Number of std/mad for outlier threshold. Defaults to 3.
        n_jobs (int, optional): Number of parallel jobs (-1 for all cores, 1 to process sequentially). Defaults to None.

    Returns:
        np.ndarray: Boolean array marking outliers.
    """

    def _process_single_band(band, method, n_limit):
        """Process a single elevation band with multiple outlier detection methods."""
        outlier_mask = np.zeros_like(band.data.data, dtype=bool)
        for m in method:
            band_outliers = find_outliers(band.data, method=m, n_limit=n_limit)
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
    elevation_bands = extract_elevation_bands(dem, band_width, base_mask)

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

    return outlier_mask


def filter_glacier(
    dem_path: Path | str,
    geometry: gpd.GeoDataFrame | gpd.GeoSeries,
    elevation_band_width: float,
    filter_method: OutlierMethod | list[OutlierMethod] = OutlierMethod.ROBUST,
    n_limit: float = 3,
    use_elevation_bands: bool = True,
    rgi_id: Path | str = None,  # Optional for logging
) -> tuple[np.ndarray, Window]:
    """Process a single glacier and return its outlier mask.

    Args:
        dem_path (Path | str): Path to the DEM file.
        geometry (gpd.GeoDataFrame | gpd.GeoSeries): Geometry of the glacier.
        elevation_band_width (float): Width of elevation bands.
        filter_method (OutlierMethod | List[OutlierMethod], optional): Outlier detection method. If a list is provided, the filters will be applied sequentially and the final mask will be the union of all masks. Defaults to OutlierMethod.ROBUST.
        n_limit (float, optional): Number of std/mad for outlier threshold. Defaults to 3.
        use_elevation_bands (bool, optional): Whether to use elevation bands for filtering. Defaults to True.
        rgi_id (Path | str, optional): Optional RGI ID for logging. Defaults to None.

    Returns:
        Tuple[np.ndarray, Window]: The outlier mask and the DEM window.
    """

    if rgi_id is None:
        rgi_id = geometry.index[0]

    logger.debug(f"Processing glacier {rgi_id}")

    # Extract DEM window for glacier
    dem_window = extract_dem_window(dem_path, geometry)

    if dem_window is None:
        logger.debug(f"Skipping glacier {rgi_id}: no valid DEM window")
        return None, None
    logger.debug(f"Extracted DEM window: {dem_window.window}")

    # Create glacier mask for window
    glacier_mask = vector_to_mask(
        geometry,
        (dem_window.window.height, dem_window.window.width),
        dem_window.transform,
    )
    logger.debug("Created glacier mask for window")

    # Combine glacier mask with nodata mask
    combined_mask = ~glacier_mask | dem_window.mask

    # Apply combined mask
    masked_dem = np.ma.masked_array(dem_window.data, mask=combined_mask)

    # Filter elevations
    if masked_dem.mask.all():
        logger.debug(f"Glacier {rgi_id} is completely masked, returning empty mask")
        empty_mask = np.zeros_like(masked_dem.data, dtype=bool)
        return empty_mask, dem_window.window

    logger.debug(f"Processing glacier {rgi_id} with {masked_dem.count()} valid pixels")

    if use_elevation_bands:
        outlier_mask = filter_by_elevation_bands(
            dem=masked_dem,
            band_width=elevation_band_width,
            method=filter_method,
            n_limit=n_limit,
        )
    else:
        outlier_mask = find_outliers(
            values=masked_dem,
            method=filter_method,
            n_limit=n_limit,
        )
    logger.debug(f"Finished processing glacier {rgi_id}")

    return outlier_mask, dem_window.window

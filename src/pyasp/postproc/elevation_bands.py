import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import features, warp
from rasterio.crs import CRS
from rasterio.windows import Window
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import gaussian_filter, median_filter
from shapely.geometry import Polygon

logger = logging.getLogger("pyasp")


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
            Required when input is a Polygon.
        buffer: Distance to buffer geometries. Zero means no buffer. Defaults to 0.
        bounds: Spatial bounds as (left, bottom, right, top). If None, uses geometry bounds.

    Returns:
        Boolean mask where True indicates the geometry.

    Raises:
        TypeError: If buffer is not a number.
        ValueError: If geometry is empty or invalid, or if CRS is None for Polygon input.
    """
    # Convert Polygon to GeoDataFrame if needed
    if not isinstance(geometry, gpd.GeoDataFrame):
        if crs is None:
            raise ValueError("CRS must be provided when input is a Polygon")
        geometry = gpd.GeoDataFrame(geometry=[geometry], crs=crs)

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
        array = dem_data.copy()
        array[mask] = np.nan

        # Apply median filter to remove outliers
        filter_size = min(5, round(min(array.shape) / 4))
        array = median_filter(array, size=filter_size)

        # Smoot dem with a gaussian filter (temporary fill nans with nearest neighbor and exclude them after smoothing)
        nan_mask = np.isnan(array)
        interpolator = NearestNDInterpolator(np.argwhere(~nan_mask), array[~nan_mask])
        array[nan_mask] = interpolator(np.argwhere(nan_mask))
        array = gaussian_filter(array, sigma=filter_size)
        array[mask] = np.nan

        # For debug: write to file both the original dem and the smoothed dem
        # with tempfile.NamedTemporaryFile(suffix="demdata.tif", dir=".") as tmp:
        #     with rio.open(
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
        #     with rio.open(
        #         tmp.name,
        #         "w",
        #         driver="GTiff",
        #         height=array.shape[0],
        #         width=array.shape[1],
        #         count=1,
        #         dtype=array.dtype,
        #         crs=CRS.from_epsg(32632),
        #         transform=rio.Affine(10, 0, 0, 0, -10, 0),
        #         nodata=np.nan,
        #     ) as dst:
        #         dst.write(array, 1)
    else:
        array = dem_data

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
        band_mask = ~((array >= band_lower) & (array < band_upper)) | mask

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

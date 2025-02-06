import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import features, warp
from rasterio.crs import CRS
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt, gaussian_filter, median_filter
from shapely.geometry import Polygon

from dempp.math import round_to_decimal

logger = logging.getLogger("dempp")


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


@dataclass(kw_only=True)
class BandLimits:
    lower: float = None
    upper: float = None
    center: float = None
    width: float = None
    label: str = None

    def __post_init__(self):
        if self.lower is not None and self.upper is not None:
            self.center = (self.lower + self.upper) / 2
            self.width = self.upper - self.lower
        elif self.center is not None and self.width is not None:
            self.lower = self.center - self.width / 2
            self.upper = self.center + self.width / 2
        else:
            raise ValueError(
                "Either band_lower/band_upper or center/width must be provided"
            )
        if self.label is None:
            self.label = f"{self.band_lower:.0f}-{self.band_upper:.0f}"


class ElevationBands:
    def __init__(self):
        self.bands = []
        self.crs = None
        self.affine = None
        self.path = None

    def __getitem__(self, key):
        return self.bands[key]

    def __len__(self):
        return len(self.bands)

    def __iter__(self):
        return iter(self.bands)

    def __repr__(self):
        return f"ElevationBands({len(self.bands)} bands)"

    def __str__(self):
        return f"ElevationBands({len(self.bands)} bands)"

    def create_from_geotiff(
        self,
        geotiff_path: str,
        band_width: float,
        polygon: gpd.GeoDataFrame | Polygon = None,
        smooth_dem: bool = True,
    ):
        geotiff_path = Path(geotiff_path)
        if not geotiff_path.exists():
            raise FileNotFoundError(f"File not found: {geotiff_path}")
        self.path = geotiff_path

        with rio.open(self.path) as src:
            self.crs = src.crs
            self.affine = src.transform
            dem_data = src.read(1, masked=True)
            self.bands = extract_elevation_bands(
                dem_data, band_width, smooth_dem=smooth_dem
            )

    def to_file(
        self,
        categorical_path: Path,
        stats_output_path: Path = None,
        nodata_cat: int = -1,
        nodata_stats: float = -9999.0,
    ):
        """Export elevation bands as a categorical GeoTIFF and optionally as an additional multiband raster with band center, mean, and std.

        Args:
            categorical_path (Path): Output path for the categorical raster.
            stats_output_path (Path, optional): Output path for the multiband stats raster. Defaults to None.
            nodata_cat (int, optional): NoData value for categorical raster. Defaults to -1.
            nodata_stats (float, optional): NoData value for stats raster. Defaults to -9999.0.
        """
        if not self.bands:
            raise ValueError("No elevation bands available for export")

        # Read Dem shape with rasterio
        with rio.open(self.path) as src:
            dem_shape = src.shape
            dem_data = src.read(1)

            # Initialize output arrays
            cat_array = np.full(dem_shape, nodata_cat, dtype=np.int32)
            if stats_output_path:
                center_array = np.full(dem_shape, nodata_stats, dtype=np.float32)
                mean_array = np.full(dem_shape, nodata_stats, dtype=np.float32)
                std_array = np.full(dem_shape, nodata_stats, dtype=np.float32)
            # For each band, set pixels that fall into the band
            for idx, band in enumerate(self.bands):
                # Condition: pixels with value within [lower, upper) and not masked
                condition = np.logical_and(
                    dem_data >= band.lower, dem_data < band.upper
                ) & (~band.mask)
                cat_array[condition] = idx
                if stats_output_path:
                    center_array[condition] = band.center
                    mean_array[condition] = band.stats.get("mean", np.nan)
                    std_array[condition] = band.stats.get("std", np.nan)

            new_meta = {
                "driver": "GTiff",
                "height": dem_shape[0],
                "width": dem_shape[1],
                "count": 1,
                "dtype": cat_array.dtype,
                "crs": self.crs,
                "transform": self.affine,
                "nodata": nodata_cat,
            }
            categorical_path = Path(categorical_path)
            categorical_path.parent.mkdir(parents=True, exist_ok=True)
            with rio.open(categorical_path, "w", **new_meta) as dst:
                dst.write(cat_array, 1)

            # Write additional stats if output path provided
            if stats_output_path:
                stats_output_path = Path(stats_output_path)
                stats_output_path.parent.mkdir(parents=True, exist_ok=True)
                stats_meta = {
                    "driver": "GTiff",
                    "height": dem_shape[0],
                    "width": dem_shape[1],
                    "count": 3,
                    "dtype": center_array.dtype,
                    "crs": self.crs,
                    "transform": self.affine,
                    "nodata": nodata_stats,
                }
                with rio.open(stats_output_path, "w", **stats_meta) as dst:
                    dst.write(center_array, 1)
                    dst.write(mean_array, 2)
                    dst.write(std_array, 3)


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
    dem: Path | str,
    band_width: float,
    mask: np.ndarray = None,
    smooth_dem: bool = True,
) -> list[BandLimits]:
    """Extract elevation bands from a DEM.

    Args:
        dem (np.ma.MaskedArray | np.ndarray): The DEM data.
        band_width (float): The width of each elevation band.
        mask (np.ndarray, optional): Mask for the DEM data. Defaults to None.
        smooth_dem (bool, optional): Whether to smooth the DEM before extracting bands. Defaults to True.

    Returns:
        List[ElevationBand]: List of elevation bands.
    """

    with rio.open(dem) as src:
        array = src.read(1, masked=True)

        # Get robust min/max values for elevation bands and round to precision
        logger.info("Extracting elevation bands...")
        round_decimal = int(np.log10(band_width))
        min_elev = np.percentile(array[~mask], 1)
        max_elev = np.percentile(array[~mask], 99)
        min_elev_bands = round_to_decimal(min_elev, round_decimal, np.floor)
        max_elev_bands = round_to_decimal(max_elev, round_decimal, np.ceil)

        if smooth_dem:
            logger.info("Smoothing DEM before extracting elevation bands")
            # Create smoothed array for extracting elevation bands
            array[mask] = np.nan

            # Apply median filter to remove outliers
            filter_size = min(5, round(min(array.shape) / 4))
            array = median_filter(array, size=filter_size)

            # Smoot dem with a gaussian filter (temporary fill nans with nearest neighbor and exclude them after smoothing)
            nan_mask = np.isnan(array)
            if np.any(nan_mask):
                # Compute indices of the nearest non-NaN point for each NaN element
                _, (i_idx, j_idx) = distance_transform_edt(
                    nan_mask, return_distances=True, return_indices=True
                )
                array[nan_mask] = array[i_idx[nan_mask], j_idx[nan_mask]]
            array = gaussian_filter(array, sigma=filter_size)
            array[mask] = np.nan
            logger.info("DEM smoothing completed")

        # Compute band limits
        bands = np.arange(min_elev_bands, max_elev_bands + band_width, band_width)
        band_limits = [
            BandLimits(lower=bands[i], upper=bands[i + 1])
            for i in range(len(bands) - 1)
        ]

        # logger.debug(f"Found {len(bands) - 1} elevation bands")
        # logger.debug(f"Min band elevation: {min_elev_bands:.0f} m")
        # logger.debug(f"Max band elevation: {max_elev_bands:.0f} m")
        # logger.info(f"Extracted {len(elevation_bands)} elevation bands.")

    return band_limits


if __name__ == "__main__":
    data_path = Path("/home/fioli/rse_aletsch/SPOT5_aletsch/pyasp_new/el_bands_sandbox")
    dem_path = data_path / "dem_aletsch.tif"

    eb = ElevationBands()
    eb.create_from_geotiff(dem_path, 100)
    eb.to_file(data_path / "eb_test.tif")

    print("Done")

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.crs import CRS
from rasterio.mask import raster_geometry_mask
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt, gaussian_filter, median_filter
from shapely.geometry import Polygon
from tqdm import tqdm

from dempp.math import round_to_decimal
from dempp.utils.paths import check_path

logger = logging.getLogger("dempp")


@dataclass(kw_only=True)
class Band:
    lower: float = field(default=None)
    upper: float = field(default=None)
    width: float = field(default=None)
    center: float = field(default=None)
    mask: np.ndarray = field(default=None, repr=False)

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

    def __str__(self):
        return f"Band: {self.lower:.0f}-{self.upper:.0f} m"


class ElevationBands:
    def __init__(
        self,
    ):
        # Elevations bands extracted from the DEM
        self.band_width = None
        self.bands = None

        # DEM data
        self.dem_path = None
        self.polygon = None
        self.crs = None
        self.transform = None
        self.window = None

        # Temporary storage for DEM data
        # @TODO: Remove this to save memory (fix saving methods first)
        self.dem_array = None
        self.dem_mask = None

    def extract_bands(
        self,
        dem_path: Path,
        band_width: float,
        polygon: Polygon = None,
        crop: bool = True,
        pad: bool = True,
        smooth_before_extraction: bool = False,
    ):
        """Extract elevation bands from the reference DEM."""

        dem_path = check_path(dem_path, "DEM")
        self.dem_path = dem_path
        self.polygon = polygon
        self.band_width = band_width

        # Check if bands have already been extracted
        if self.bands:
            raise ValueError("Elevation bands already extracted")

        # Load DEM
        logger.info("Loading reference DEM...")
        dem, transform, window, crs = load_dem(
            path=dem_path,
            polygon=polygon,
            crop=crop,
            pad=pad,
        )
        self.crs = crs
        self.transform = transform
        self.window = window
        self.bands = extract_elevation_bands(
            dem.data,
            band_width,
            dem.mask,
            smooth_before_extraction,
        )

        # Temporary storage for DEM data
        # @TODO: Remove this to save memory
        self.dem_array = dem.data
        self.dem_mask = dem.mask

    def get_band_mask(self, band_idx: int) -> np.ndarray:
        """Get mask for a specific band."""
        if not self.bands:
            raise ValueError("Extract bands first")
        return self.bands[band_idx]["mask"]

    def extract_values_from_new_dem(
        self,
        dem_path: Path,
        polygon: Polygon = None,
        crop: bool = True,
        pad: bool = True,
    ) -> list[tuple[np.ndarray, tuple]]:
        """Extract values from a new DEM using stored elevation bands."""
        if not self.bands:
            raise ValueError("Extract bands from reference DEM first")

        # Load new DEM
        new_dem, new_mask, new_transform, new_window, new_crs = load_dem(
            dem_path,
            polygon=polygon,
            crop=crop,
            pad=pad,
        )

        # Reproject and align band masks if needed
        results = []
        for band_info in self.bands:
            # Convert band mask coordinates to geographic coordinates
            rows, cols = np.where(band_info["mask"])
            src_coords = rio.transform.xy(
                band_info["transform"], rows, cols, offset="center"
            )

            # Convert geographic coordinates to new DEM pixel coordinates
            dst_rows, dst_cols = rio.transform.rowcol(
                new_transform, *src_coords, offset="center"
            )

            # Create mask for new DEM
            new_mask = np.zeros_like(new_dem, dtype=bool)
            valid_idx = (
                (dst_rows >= 0)
                & (dst_rows < new_dem.shape[0])
                & (dst_cols >= 0)
                & (dst_cols < new_dem.shape[1])
            )
            new_mask[dst_rows[valid_idx], dst_cols[valid_idx]] = True

            # Extract values
            band_vals = new_dem[new_mask]
            results.append((band_vals, band_info["bounds"]))

        return results

    def write_band_masks(self, output_path: Path) -> None:
        """Write elevation band masks as a single categorical raster."""
        if not self.bands:
            raise ValueError("Extract bands first")

        logger.info("Writing band masks to file...")

        # Create categorical raster where each value represents a band
        categorical = np.zeros_like(self.dem_array, dtype=np.int16)
        for i, band in enumerate(self.bands, start=1):
            categorical[band.mask] = i

        # Write to file
        with rio.open(
            output_path,
            "w",
            driver="GTiff",
            height=categorical.shape[0],
            width=categorical.shape[1],
            count=1,
            dtype=categorical.dtype,
            crs=self.crs,
            transform=self.transform,
            nodata=0,
        ) as dst:
            dst.write(categorical, 1)

            # Add band metadata
            dst.update_tags(band_width=self.band_width, n_bands=len(self.bands))
            # Add band boundaries as metadata
            for i, band in enumerate(self.bands, start=1):
                dst.update_tags(**{f"band_{i:02}_bounds": f"{band.lower},{band.upper}"})

        logger.info(f"Band masks written to {output_path}")

    def write_band_values(
        self,
        output_dir: Path,
        dem_array: np.ndarray = None,
        transform: rio.Affine = None,
        crs: CRS = None,
    ) -> None:
        """Write elevation values for each band to separate files."""
        if not self.bands:
            raise ValueError("Extract bands first")

        logger.info("Writing band values to separate files...")

        # Use current DEM if no new one provided
        if dem_array is None:
            dem_array = self.dem_array
            transform = self.transform
            crs = self.crs

        output_dir.mkdir(parents=True, exist_ok=True)

        for i, band in tqdm(enumerate(self.bands), desc="Writing band values"):
            # Create masked array for band
            band_dem = np.full_like(dem_array, np.nan)
            if transform == self.transform:
                # Direct mask application if same transform
                band_dem[band.mask] = dem_array[band.mask]
            else:
                # Reproject mask if different transform
                rows, cols = np.where(band.mask)
                src_coords = rio.transform.xy(
                    self.transform, rows, cols, offset="center"
                )
                dst_rows, dst_cols = rio.transform.rowcol(
                    transform, *src_coords, offset="center"
                )
                valid_idx = (
                    (dst_rows >= 0)
                    & (dst_rows < dem_array.shape[0])
                    & (dst_cols >= 0)
                    & (dst_cols < dem_array.shape[1])
                )
                band_dem[dst_rows[valid_idx], dst_cols[valid_idx]] = dem_array[
                    dst_rows[valid_idx], dst_cols[valid_idx]
                ]

            # Write band values
            output_path = (
                output_dir / f"band_{i}_{band.lower:.0f}-{band.upper:.0f}m.tif"
            )
            with rio.open(
                output_path,
                "w",
                driver="GTiff",
                height=band_dem.shape[0],
                width=band_dem.shape[1],
                count=1,
                dtype="float32",
                crs=crs,
                transform=transform,
                nodata=np.nan,
            ) as dst:
                dst.write(band_dem.astype("float32"), 1)
                dst.update_tags(band_bounds=f"{band.lower},{band.upper}")

        logger.info(f"Band values written to {output_dir}")


def load_dem(
    path: Path,
    polygon: Polygon = None,
    crop: bool = True,
    pad: bool = True,
) -> tuple[np.ma.MaskedArray, rio.Affine, rio.windows.Window, CRS]:
    path = check_path(path)

    with rio.open(path) as src:
        crs = src.crs
        if polygon is not None:
            # Load the mask within the polygon
            mask, out_transform, window = raster_geometry_mask(
                src,
                [polygon],
                crop=crop,
                pad=pad,
            )
            transform = out_transform

            # Load the data within the window
            dem_array = src.read(1, window=window)

            # Mask the data
            dem = np.ma.masked_array(dem_array, mask=mask)

        else:
            # Load the entire DEM
            dem = src.read(1, masked=True)
            transform = src.transform
            window = Window(0, 0, src.width, src.height)

    return dem, transform, window, crs


def smooth_dem(
    dem: np.ndarray, mask: np.ndarray, filter_size: int = 5, inplace: bool = False
) -> np.ndarray:
    array = dem.copy() if not inplace else dem

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


def extract_elevation_bands(
    dem: np.ndarray | np.ma.MaskedArray,
    band_width: float,
    mask: np.ndarray,
    smooth_before_extraction: bool = False,
) -> list[Band]:
    """Extract elevation bands from a DEM.

    Args:
        dem (np.ma.MaskedArray | np.ndarray): The DEM data.
        band_width (float): The width of each elevation band.
        mask (np.ndarray, optional): Mask for the DEM data. Defaults to None.
        smooth_before_extraction (bool, optional): Whether to smooth the DEM before extracting bands. Defaults to True.

    Returns:
        List[Band]: List of elevation bands.
    """

    # Get robust min/max values for elevation bands and round to precision
    logger.info("Extracting elevation bands...")
    round_decimal = int(np.log10(band_width))
    min_elev = np.percentile(dem[~mask], 1)
    max_elev = np.percentile(dem[~mask], 99)
    min_elev_bands = round_to_decimal(min_elev, round_decimal, np.floor)
    max_elev_bands = round_to_decimal(max_elev, round_decimal, np.ceil)

    # Smooth DEM before extracting bands
    dem_data = smooth_dem(dem, mask) if smooth_before_extraction else dem

    # Compute band limits
    band_edges = np.arange(min_elev_bands, max_elev_bands + band_width, band_width)
    bands = []
    for i in range(len(band_edges) - 1):
        lower, upper = band_edges[i], band_edges[i + 1]
        band_mask = (dem_data >= lower) & (dem_data < upper) & (~mask)
        bands.append(
            Band(
                lower=lower,
                upper=upper,
                mask=band_mask,
            )
        )
    logger.info(f"Extracted {len(bands)} elevation bands.")
    logger.debug(f"Min band elevation: {min_elev_bands:.0f} m")
    logger.debug(f"Max band elevation: {max_elev_bands:.0f} m")

    return bands


if __name__ == "__main__":
    data_path = Path("/home/fioli/rse_aletsch/SPOT5_aletsch/pyasp_new/el_bands_sandbox")
    # dem_path = data_path / "dem_aletsch.tif"
    dem_path = data_path / "dems/004_005-006_S5_054-256-0_2003-07-08.tif"
    # polygon_path = data_path / "aletsch_polygon_dummy.geojson"

    # poly = gpd.read_file(polygon_path)
    # polygon = poly.geometry[0]

    eb = ElevationBands()
    eb.extract_bands(
        dem_path=dem_path,
        # polygon=polygon,
        band_width=100,
    )

    eb.write_band_masks(data_path / "band_masks.tif")
    # eb.write_band_values(data_path / "band_values")

    # shift dem to simulate glacier retreat
    # eb.dem_array -= 100
    # eb.write_band_values(data_path / "band_values_shifted")

    print("Done")

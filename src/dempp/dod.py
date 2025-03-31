"""This module has been deprecated and will be removed in future versions as the raster statistics computation have been already implemented in geoutils as a method of the Raster class.
The functions in this module are kept for backward compatibility and will be removed in future versions.
"""

import logging
from pathlib import Path
from typing import Any

import geoutils as gu
import numpy as np
import xdem

from dempp.io import load_dem
from dempp.statistics import (
    RasterStatistics,
    compute_raster_statistics,
    plot_raster_statistics,
)

logger = logging.getLogger("dempp")


def compute_dod(
    dem: xdem.DEM,
    reference: xdem.DEM,
    warp_on: str = "reference",
    resampling: str = "bilinear",
) -> xdem.DEM:
    """Compute the difference between two DEMs (reference - dem).

    Args:
        dem: The DEM to compare
        reference: The reference DEM
        warp_on: Which DEM to reproject ('reference' or 'dem')
        resampling: Resampling method for reprojection

    Returns:
        xdem.DEM: The difference DEM (reference - dem)

    Raises:
        ValueError: If warp_on value is invalid
    """
    # Check if reprojection is needed
    needs_reprojection = (
        dem.crs != reference.crs or dem.transform != reference.transform
    )

    if not needs_reprojection:
        logger.debug("DEMs have matching CRS and dimensions")
        return reference - dem

    # Reprojection is needed
    logger.info("Reprojecting DEMs (different CRS or dimensions)")
    if warp_on == "reference":
        logger.debug("Reprojecting DEM to match reference")
        aligned_dem = dem.reproject(reference, resampling=resampling)
        return reference - aligned_dem
    elif warp_on == "dem":
        logger.debug("Reprojecting reference to match DEM")
        aligned_ref = reference.reproject(dem, resampling=resampling)
        return aligned_ref - dem
    else:
        raise ValueError("Invalid warp_on value. Must be 'reference' or 'dem'")


def apply_mask(
    dem: xdem.DEM,
    inlier_mask: gu.Mask,
) -> xdem.DEM:
    """Apply an inlier mask to a DEM. Returns the DEM with the same shape as the original DEM.

    Args:
        dem: The difference DEM to mask
        mask: A geoutils Mask object containing the inlier mask

    Returns:
        xdem.DEM: The masked DEM

    Raises:
        ValueError: If input DEM is not an xdem.DEM object or mask is not a geoutils.Mask object
    """
    if not isinstance(dem, xdem.DEM):
        raise ValueError("DEM must be an xdem.DEM object")

    if not isinstance(inlier_mask, gu.Mask):
        raise ValueError("Mask must be a Path or geoutils.Mask object")

    if inlier_mask.transform != dem.transform:
        inlier_mask = inlier_mask.reproject(dem)

    dem.set_mask(~inlier_mask)

    return dem


def differenciate_dems(
    dem: Path | str,
    reference: Path | str,
    inlier_mask: Path | str | None = None,
    output_dir: Path | str | None = None,
    output_prefix: str | None = None,
    make_plot: bool = False,
    warp_on: str = "reference",
    resampling: str = "bilinear",
    xlim: tuple[float, float] | None = None,
    plt_cfg: dict[str, Any] | None = None,
) -> tuple[xdem.DEM, RasterStatistics]:
    """Process two DEMs to create a difference DEM with statistics.

    This high-level function performs the complete DoD workflow by calling several
    lower-level functions in sequence:
    1. Load DEMs from paths and Mask object if provided
    2. Calculate the difference between DEMs - compute_dod()
    3. Compute statistics on the difference DEM - compute_raster_statistics()
    4. Save results to files if output_dir is provided - save_outputs()
    5. Generate visualization if make_plot is True - plot_raster_statistics()

    Each of these functions can also be used independently for more granular control.

    Args:
        dem: Path to the DEM to compare
        reference: Path to the reference DEM
        mask: Optional path to mask for stable terrain
        output_dir: Directory to save outputs (if None, nothing is saved)
        output_prefix: Prefix for output filenames
        make_plot: Whether to generate and save a plot
        warp_on: Which DEM to reproject ('reference' or 'dem')
        resampling: Resampling method for reprojection
        xlim: Optional x-axis limits for the plot
        plt_cfg: Optional plotting configuration to be passed to the plot_raster_statistics() function

    Returns:
        tuple[xdem.DEM, RasterStatistics]: The difference DEM and statistics object
    """
    # Load DEMs from paths
    dem = load_dem(dem, area_or_point="area")
    reference = load_dem(reference, area_or_point="area")

    # Load mask if provided
    mask_raster = gu.Raster(inlier_mask) if inlier_mask else None
    mask = mask_raster == 1
    if mask is not None and mask.transform != reference.transform:
        mask.reproject(
            reference, inplace=True, force_source_nodata=255, resampling="nearest"
        )
        logger.info("Reprojected mask to match reference DEM")

    # Compute DoD
    dod = compute_dod(dem, reference, warp_on=warp_on, resampling=resampling)
    logger.info("Computed difference between DEMs")

    # Compute statistics
    stats = compute_raster_statistics(raster=dod, inlier_mask=mask)
    logger.info("Computed statistics for difference DEM")

    # Save outputs if requested
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_prefix is None:
            output_prefix = f"{Path(dem.name).stem}__{Path(reference.name).stem}"

        # Save difference DEM and statistics
        stats.save(output_dir / f"{output_prefix}_stats.json")
        dod.save(output_dir / f"{output_prefix}_dod.tif")
        logger.info(f"Saved difference DEM and statistics to: {output_dir}")

        # Generate plot if requested
        if make_plot:
            # Apply mask to the data for plotting
            plot_data = (
                apply_mask(dod, mask).data.compressed()
                if mask is not None
                else dod.data.compressed()
            )
            # Remove extreme quantiles
            q1 = np.percentile(plot_data, 0.01)
            q2 = np.percentile(plot_data, 99.99)
            plot_data = plot_data[(plot_data > q1) & (plot_data < q2)]
            logger.info("Removed extreme quantiles (<0.01% and >99.99%) for plotting")

            plot_path = output_dir / f"{output_prefix}_plot.png"
            plt_cfg = plt_cfg or {}
            plot_raster_statistics(
                raster=plot_data,
                output_file=plot_path,
                stats=stats,
                xlim=xlim,
                **plt_cfg,
            )
            logger.info(f"Saved plot to: {plot_path}")

    return dod, stats

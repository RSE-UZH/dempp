import logging
from pathlib import Path

import geoutils as gu
import xdem

from dempp.raster_statistics import (
    RasterStatistics,
    compute_raster_statistics,
    plot_raster_statistics,
)
from dempp.utils.paths import check_path

logger = logging.getLogger("dempp")


def compute_dod(
    dem: xdem.DEM,
    reference: xdem.DEM,
    warp_on: str = "reference",
    resampling="bilinear",
    verbose: bool = False,
) -> xdem.DEM:
    """
    Compute the difference between two DEMs, reprojecting if necessary.

    Parameters:
        dem (xdem.DEM): The DEM to compare.
        reference (xdem.DEM): The reference DEM.
        warp_on (str, optional): The DEM to reproject. Must be 'reference' or 'dem'. Defaults to 'reference'.
        resampling (str, optional): Resampling method for reprojection. Defaults to 'bilinear'.

    Returns:
        xdem.DEM: The difference DEM (reference - dem).

    Raises:
        ValueError: If DEMs have mismatched dimensions after reprojection or incompatible CRS.
    """
    # Check if the DEMs have the same CRS
    if dem.crs != reference.crs or dem.transform != reference.transform:
        logger.info("DEM and reference DEM have different CRS or dimensions.")
        if warp_on == "reference":
            logger.info("\tReprojecting DEM to match the reference DEM...")
            A = dem.reproject(reference, resampling=resampling)
            B = reference
        elif warp_on == "dem":
            logger.info("\tReprojecting reference DEM to match the DEM...")
            A = dem
            B = reference.reproject(dem, resampling=resampling)
        else:
            raise ValueError("Invalid `warp_on` value. Must be 'reference' or 'dem'.")

    else:
        logger.debug(
            "DEM and reference DEM have the same CRS and dimensions. No reprojection needed. Computing difference..."
        )
        A = dem
        B = reference

    # Compute the difference
    ddem = A - B
    logger.debug("Computed difference DEM")

    if verbose:
        logger.debug("Difference DEM info:")
        ddem.info(stats=True)

    return ddem


def compute_dod_stats(
    dem: xdem.DEM | Path | str,
    reference: xdem.DEM | Path | str,
    mask: gu.Mask | Path | None = None,
    output_dir: Path | None = None,
    output_prefix: str | None = None,
    make_plot: bool = False,
    xlim: tuple[float, float] | None = None,
    plt_cfg: dict | None = None,
    warp_on: str = "reference",
    resampling: str = "bilinear",
) -> tuple[xdem.DEM, RasterStatistics]:
    """Compute difference between two DEMs, calculate statistics and generate plots.

    Args:
        dem (Path | xdem.DEM): Path to the DEM file or xdem.DEM object to process.
        reference (Path | xdem.DEM): Path to the reference DEM file or xdem.DEM object.
        mask (gu.Mask | Path | None, optional): Optional mask for stable areas, can be a Path or gu.Mask. Defaults to None.
        output_dir (Path, optional): Directory where to save output statistics and plots. Defaults to None.
        figure_path (Path, optional): Path to save the difference plot. Defaults to None.
        warp_on (str, optional): DEM to reproject for comparison. Defaults to "reference".
        resampling (str, optional): Resampling method for reprojection. Defaults to "bilinear".
        xlim (tuple[float, float] | None, optional): X-axis limits for the plots as (min, max). Defaults to None.
        plt_cfg (dict | None, optional): Configuration dictionary for plots. Defaults to None.

    Returns:
        tuple[xdem.DEM, RasterStatistics]: The difference DEM and its statistics.

    Raises:
        ValueError: If input DEMs are not valid or compatible.
        FileNotFoundError: If input paths don't exist.
    """
    # Load DEMs if paths are provided
    if isinstance(dem, (Path | str)):
        dem = check_path(dem, "DEM")
        dem = xdem.DEM(dem)
    if isinstance(reference, (Path | str)):
        reference = check_path(reference, "Reference DEM")
        reference = xdem.DEM(reference)

    # Validate inputs
    if not isinstance(dem, xdem.DEM) or not isinstance(reference, xdem.DEM):
        raise ValueError("Both dem and reference must be Path or xdem.DEM objects")

    # Compute the difference
    diff = compute_dod(dem, reference, warp_on=warp_on, resampling=resampling)
    if diff is None:
        raise ValueError("Error computing difference between DEMs")
    logger.info("Computed difference between DEMs")

    # Create the output directory and output stem
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_prefix is None:
            try:
                output_prefix = Path(dem.filename).stem
            except (AttributeError, TypeError):
                logger.warning("Unable to get DEM filename. Using default name")
                output_prefix = "stats"

    # Save difference DEM if path is provided
    if output_dir is not None:
        ddem_path = output_dir / f"{output_prefix}_ddem.tif"
        diff.save(ddem_path)
        logger.info(f"Saved difference DEM to: {ddem_path}")

    # Apply mask if provided
    if mask is not None:
        if isinstance(mask, Path):
            if not mask.exists():
                raise FileNotFoundError(f"Mask file not found: {mask}")
            mask = gu.Mask(mask)
            mask.set_area_or_point("Point")
        elif not isinstance(mask, gu.Mask):
            raise ValueError("Mask must be a Path or geoutils.Mask object")
        mask_warped = mask.reproject(reference)
        diff_masked = diff[mask_warped]
        logger.info("Applied mask to difference")
    else:
        diff_masked = diff.data
        mask_warped = None

    # Compute statistics
    diff_stats = compute_raster_statistics(raster=diff, mask=mask_warped)
    logger.info("Computed statistics for difference DEM")

    # Create outputs if requested
    if output_dir is not None:
        # Save statistics to file
        try:
            stats_path = output_dir / f"{output_prefix}_stats.json"
            diff_stats.save(stats_path)
            logger.info(f"Saved statistics to: {stats_path}")
        except Exception as e:
            logger.error(f"Error saving statistics: {e}")

        # Make difference histogram plot
        if make_plot:
            logger.info("Generating difference plot...")
            figure_path = output_dir / f"{output_prefix}_plot.png"
            if plt_cfg is None:
                plt_cfg = {}
            try:
                plot_raster_statistics(
                    raster=diff_masked,
                    output_file=figure_path,
                    stats=diff_stats,
                    xlim=xlim,
                    **plt_cfg,
                )
                logger.info(f"Figure saved to: {figure_path}")
            except Exception as e:
                logger.error(f"Error saving figure: {e}")

    return diff, diff_stats


if __name__ == "__main__":
    dem_2009 = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
    dem_1990 = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem_coreg"))

    # Test compute_dod() function
    # - same CRS
    ddem = compute_dod(dem_1990, dem_2009)
    ddem.info()

    # - different resolutions
    dem_2009_lowres = dem_2009.reproject(res=dem_2009.res[0] * 2)
    ddem1 = compute_dod(dem_1990, dem_2009_lowres)
    ddem1.info(stats=True)

    ddem2 = compute_dod(dem_1990, dem_2009_lowres, warp_on="dem")
    ddem2.info(stats=True)

    # Test compute_dod_stats() function
    # - Pass DEM objects
    stats = compute_dod_stats(dem_1990, dem_2009, make_plot=True)

    # - Pass DEM paths
    stats = compute_dod_stats(
        xdem.examples.get_path("longyearbyen_tba_dem_coreg"),
        xdem.examples.get_path("longyearbyen_ref_dem"),
    )

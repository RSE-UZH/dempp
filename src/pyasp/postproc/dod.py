import logging
from pathlib import Path

import geoutils as gu
import xdem

from pyasp.postproc.raster_statistics import (
    RasterStatistics,
    compute_raster_statistics,
    plot_raster_statistics,
)

logger = logging.getLogger("pyasp")


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
        logger.info(
            "DEM and reference DEM have different CRS or dimensions. Reprojecting..."
        )
        if warp_on == "reference":
            logger.info("Reprojecting DEM to match the reference DEM...")
            A = dem.reproject(reference, resampling=resampling)
            B = reference
        elif warp_on == "dem":
            logger.info("Reprojecting reference DEM to match the DEM...")
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
    make_plot: bool = False,
    output_dir: Path | None = None,
    figure_path: Path | None = None,
    xlim: tuple[float, float] | None = None,
    plt_cfg: dict | None = None,
    warp_on: str = "reference",
    resampling: str = "bilinear",
) -> RasterStatistics:
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
        RasterStatistics: Object with computed statistics.

    Raises:
        ValueError: If input DEMs are not valid or compatible.
        FileNotFoundError: If input paths don't exist.
    """
    logger.info("Processing DEM pair")

    # Load DEMs if paths are provided
    if isinstance(dem, (Path | str)):
        dem = Path(dem)
        if not dem.exists():
            raise FileNotFoundError(f"DEM file not found: {dem}")
        dem = xdem.DEM(dem)
    if isinstance(reference, (Path | str)):
        reference = Path(reference)
        if not reference.exists():
            raise FileNotFoundError(f"Reference DEM file not found: {reference}")
        reference = xdem.DEM(reference)

    # Validate inputs
    if not isinstance(dem, xdem.DEM) or not isinstance(reference, xdem.DEM):
        raise ValueError("Both dem and reference must be Path or xdem.DEM objects")

    # Compute the difference
    diff = compute_dod(dem, reference, warp_on=warp_on, resampling=resampling)
    logger.info("Computed difference between DEMs")

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
        diff_masked = diff
        mask_warped = None

    # Compute statistics
    diff_stats = compute_raster_statistics(raster=diff, mask=mask_warped)

    # Make plot
    if make_plot:
        if figure_path is not None:
            figure_path = Path(figure_path)
            output_dir = figure_path.parent
        else:
            if output_dir is None:
                output_dir = Path.cwd()
            output_dir = Path(output_dir)
            try:
                output_stem = Path(dem.filename).stem
            except (AttributeError, TypeError):
                logger.warning("Unable to get DEM filename. Using default name")
                output_stem = "stats"
            figure_path = output_dir / f"{output_stem}_diff_plot.png"

        output_dir.mkdir(parents=True, exist_ok=True)

        if plt_cfg is None:
            plt_cfg = {}
        plot_raster_statistics(
            diff_masked.data,
            output_file=figure_path,
            stats=diff_stats,
            xlim=xlim,
            **plt_cfg,
        )
        logger.info(f"Figure saved to: {figure_path}")

        # Save statistics to file
        stats_path = figure_path.with_suffix(".json")
        diff_stats.save(stats_path)

    return diff_stats


# OLD FUNCTION
# def process_dem(
#     dem: Path | xdem.DEM,
#     reference_dem: Path | xdem.DEM,
#     glacier_outlines: Path | gu.Vector | None = None,
#     diff_dem_path: Path | None = None,
#     diff_stats_path: Path | None = None,
#     eval_folder: Path | None = None,
#     verbose: bool = False,
#     use_parent_as_name: bool = False,
# ) -> dict | None:
#     """Process DEM and compute differences with reference DEM.

#     Args:
#         dem: Input DEM file path or xdem.DEM object
#         reference_dem: Reference DEM file path or xdem.DEM object
#         glacier_outlines: Optional glacier outlines for masking
#         diff_dem_path: Explicit path for difference DEM output
#         diff_stats_path: Explicit path for statistics output
#         eval_folder: Folder for outputs (used if specific paths not provided)
#         verbose: Enable detailed logging
#         use_parent_as_name: Use parent folder name instead of file stem

#     Returns:
#         Dictionary with computed statistics or None if error occurs

#     Raises:
#         ValueError: If no output location is specified
#     """
#     # Get DEM name for default filenames
#     if isinstance(dem, Path):
#         dem_name = dem.parent.name if use_parent_as_name else dem.stem
#     elif isinstance(dem, xdem.DEM):
#         path = Path(dem.name)
#         dem_name = path.parent.name if use_parent_as_name else path.stem
#     else:
#         dem_name = "input_dem"

#     # Validate output locations
#     if not any([diff_dem_path, diff_stats_path, eval_folder]):
#         raise ValueError(
#             "Must specify either output paths (diff_dem_path, diff_stats_path) "
#             "or eval_folder for outputs"
#         )

#     # Set output paths with priority for explicit paths and create directories
#     eval_folder = Path(eval_folder) if eval_folder else None
#     diff_dem_path = Path(diff_dem_path) if diff_dem_path else None
#     diff_stats_path = Path(diff_stats_path) if diff_stats_path else None
#     if diff_dem_path is None and eval_folder:
#         diff_dem_path = eval_folder / f"{dem_name}_dod.tif"
#     elif diff_dem_path is None:
#         diff_dem_path = dem.parent / f"{dem_name}_dod.tif"
#     if diff_stats_path is None and eval_folder:
#         diff_stats_path = eval_folder / f"{dem_name}_dod_stats.json"
#     elif diff_stats_path is None:
#         diff_stats_path = diff_dem_path.parent / f"{dem_name}_dod_stats.json"
#     for path in [diff_dem_path, diff_stats_path]:
#         path.parent.mkdir(parents=True, exist_ok=True)

#     # Load DEMs if paths are provided, otherwise assume they are already xdem.DEM objects
#     if isinstance(dem, Path) and isinstance(reference_dem, Path):
#         logger.info(f"Loading DEMs: {dem_name} and reference DEM...")
#         dem, reference_dem = load_dems([dem, reference_dem])
#     elif isinstance(dem, Path):
#         logger.info(f"Loading DEM: {dem_name}...")
#         dem = xdem.DEM(dem)
#     elif isinstance(reference_dem, Path):
#         logger.info("Loading reference DEM...")
#         reference_dem = xdem.DEM(reference_dem)
#     if not isinstance(dem, xdem.DEM) or not isinstance(reference_dem, xdem.DEM):
#         raise ValueError("Invalid DEM or reference DEM provided.")

#     # Load and process glacier outlines if provided
#     if glacier_outlines is not None:
#         logger.info("Loading glacier outlines...")
#         if isinstance(glacier_outlines, Path):
#             glacier_outlines = gu.Vector(glacier_outlines)
#             glacier_outlines = glacier_outlines.crop(reference_dem, clip=True)
#         stable_mask = ~glacier_outlines.create_mask(raster=reference_dem)
#     else:
#         stable_mask = None  # No masking if glacier_outlines is None

#     # Compute differences between DEMs and relevant statistics
#     logger.info("Computing Dem of Difference...")
#     dod = compute_dod(dem=dem, reference=reference_dem, resampling="bilinear")

#     # Save DoD and compute statistics in separate threads
#     with ThreadPoolExecutor(max_workers=2) as executor:
#         save_future = executor.submit(
#             save_dem, dod, diff_dem_path, dtype="float32", compress="lzw"
#         )
#         stats_future = executor.submit(
#             compute_dod_statistics,
#             diff=dod,
#             stable_mask=stable_mask,
#             output_file=diff_stats_path,
#         )
#         save_future.result()  # This will raise any exceptions that occurred
#         diff_stats = stats_future.result()

#     logger.info(f"Processed DEM: {dem_name} successfully.")

#     return dod, diff_stats

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

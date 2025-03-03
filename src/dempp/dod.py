import logging
from pathlib import Path
from typing import Any

import geoutils as gu
import numpy as np
import xdem

from dempp.io import load_dem
from dempp.raster_statistics import (
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
    mask: gu.Mask,
) -> np.ma.MaskedArray:
    """Apply a mask to a difference DEM.

    Args:
        dem: The difference DEM to mask
        mask: A geoutils Mask object

    Returns:
        Masked array with the difference data

    Raises:
        ValueError: If input DEM is not an xdem.DEM object or mask is not a geoutils.Mask object
    """
    if not isinstance(dem, xdem.DEM):
        raise ValueError("DEM must be an xdem.DEM object")

    if not isinstance(mask, gu.Mask):
        raise ValueError("Mask must be a Path or geoutils.Mask object")

    mask_warped = mask.reproject(dem)
    return dem[mask_warped]


def save_outputs(
    dod: xdem.DEM,
    stats: RasterStatistics,
    output_dir: Path,
    output_prefix: str,
) -> dict[str, Path]:
    """Save DoD outputs to files.

    Args:
        dod: The difference DEM
        stats: Statistics for the difference DEM
        output_dir: Directory to save outputs
        output_prefix: Prefix for output filenames

    Returns:
        Dictionary of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    # Save difference DEM
    dod_path = output_dir / f"{output_prefix}_dod.tif"
    dod.save(dod_path)
    outputs["dod"] = dod_path
    logger.info(f"Saved difference DEM to: {dod_path}")

    # Save statistics
    stats_path = output_dir / f"{output_prefix}_stats.json"
    stats.save(stats_path)
    outputs["stats"] = stats_path
    logger.info(f"Saved statistics to: {stats_path}")

    return outputs


def process_dod(
    dem: Path | str,
    reference: Path | str,
    mask: Path | str | None = None,
    output_dir: Path | str | None = None,
    output_prefix: str | None = None,
    make_plot: bool = False,
    warp_on: str = "reference",
    resampling: str = "bilinear",
    xlim: tuple[float, float] | None = None,
    plt_cfg: dict[str, Any] | None = None,
) -> tuple[xdem.DEM, RasterStatistics, dict[str, Path] | None]:
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
        plt_cfg: Optional plotting configuration

    Returns:
        Tuple of (difference DEM, statistics, output paths dict or None)
    """
    # Load DEMs from paths
    dem_obj = load_dem(dem)
    reference_obj = load_dem(reference)

    # Load mask if provided
    mask_obj = gu.Mask(mask).reproject(reference_obj) if mask is not None else None

    # Convert paths to Path objects for consistency
    if output_dir is not None:
        output_dir = Path(output_dir)

    # Determine output prefix if not provided
    if output_dir is not None and output_prefix is None:
        try:
            output_prefix = Path(dem).stem
        except (AttributeError, TypeError):
            logger.warning("Unable to get DEM filename. Using default name")
            output_prefix = "dod"

    # Compute DoD
    dod = compute_dod(dem_obj, reference_obj, warp_on=warp_on, resampling=resampling)
    logger.info("Computed difference between DEMs")

    # Compute statistics
    stats = compute_raster_statistics(raster=dod, mask=mask_obj)
    logger.info("Computed statistics for difference DEM")

    # Save outputs if requested
    outputs = None
    if output_dir is not None:
        outputs = save_outputs(
            dod=dod,
            stats=stats,
            output_dir=output_dir,
            output_prefix=output_prefix,
        )
        logger.info(f"Saved outputs to {output_dir}")

        # Generate plot if requested
        if make_plot:
            plot_path = output_dir / f"{output_prefix}_plot.png"
            plt_cfg = plt_cfg or {}

            # Apply mask if provided
            plot_data = apply_mask(dod, mask_obj) if mask_obj is not None else dod.data
            plot_raster_statistics(
                raster=plot_data,
                output_file=plot_path,
                stats=stats,
                xlim=xlim,
                **plt_cfg,
            )
            outputs["plot"] = plot_path
            logger.info(f"Saved plot to: {plot_path}")

    return dod, stats, outputs


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    import geoutils as gu
    import matplotlib.pyplot as plt
    import numpy as np

    print("Testing DoD functions with sample DEMs...")

    # Load example DEMs from xdem
    dem_2009 = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
    dem_1990 = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem_coreg"))

    print("\n1. Basic DoD computation (same CRS):")
    dod = compute_dod(dem_1990, dem_2009)
    print(f"DoD shape: {dod.shape}, CRS: {dod.crs}")
    print(f"Min: {dod.data.min():.2f}, Max: {dod.data.max():.2f}")

    print("\n2. Testing different resolutions:")
    # Create a low-res version of the 2009 DEM
    dem_2009_lowres = dem_2009.reproject(res=dem_2009.res[0] * 2)
    print(f"Original res: {dem_2009.res}, Low-res: {dem_2009_lowres.res}")

    # Test with warp_on="reference" (default)
    print("- Warping DEM to match reference:")
    dod1 = compute_dod(dem_1990, dem_2009_lowres)
    print(f"  DoD shape: {dod1.shape}, Resolution: {dod1.res}")

    # Test with warp_on="dem"
    print("- Warping reference to match DEM:")
    dod2 = compute_dod(dem_1990, dem_2009_lowres, warp_on="dem")
    print(f"  DoD shape: {dod2.shape}, Resolution: {dod2.res}")

    print("\n3. Testing mask application:")
    # Create a simple mask (e.g., central area of the DEM)
    mask_array = np.zeros(dem_1990.shape, dtype=bool)
    center_y, center_x = mask_array.shape[0] // 2, mask_array.shape[1] // 2
    mask_size = min(mask_array.shape[0], mask_array.shape[1]) // 4
    mask_array[
        center_y - mask_size : center_y + mask_size,
        center_x - mask_size : center_x + mask_size,
    ] = True

    # Create a mask with the same georeference as the DEM
    mask = gu.Mask.from_array(
        mask_array, transform=dem_1990.transform, crs=dem_1990.crs
    )

    # Apply mask to DoD
    masked_dod = apply_mask(dod, mask)
    print(f"Masked DoD: {masked_dod.count()} valid pixels out of {mask_array.size}")

    print("\n4. Testing complete workflow with process_dod():")
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save the sample DEMs and mask to temporary files
        dem_path = temp_path / "dem_1990.tif"
        ref_path = temp_path / "dem_2009.tif"
        mask_path = temp_path / "mask.tif"

        dem_1990.save(dem_path)
        dem_2009.save(ref_path)
        mask.save(mask_path)

        # Process the DoDs using file paths
        result_dod, result_stats, outputs = process_dod(
            dem=dem_path,
            reference=ref_path,
            mask=mask_path,
            output_dir=temp_path,
            output_prefix="test",
            make_plot=True,
            xlim=(-10, 10),  # Set reasonable x-axis limits for the plot
        )

        print(f"Generated outputs in {temp_path}:")
        for output_type, path in outputs.items():
            print(f"- {output_type}: {path.name}")

        print("\nStatistics:")
        for key, value in result_stats.to_dict().items():
            print(f"- {key}: {value:.3f}")

        # Display the plot (only when running interactively)
        if outputs and "plot" in outputs:
            plt.figure(figsize=(10, 8))
            img = plt.imread(outputs["plot"])
            plt.imshow(img)
            plt.axis("off")
            plt.title("Generated DoD Plot")
            plt.tight_layout()
            plt.show()

    print("\nAll tests completed successfully!")
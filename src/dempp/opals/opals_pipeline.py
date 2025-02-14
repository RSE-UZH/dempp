import argparse  # Add argparse import
import json
import logging
from pathlib import Path

from opals import LSM, Algebra, FillGaps, Grid, Import, Info, Types  # noqa
from opals.tools import processor  # noqa

logger = logging.getLogger()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

# Default configurations
LSM_CFG = {}
GRID_CFG = {
    "gridSize": 10,
    "interpolation": Types.GridInterpolator.robMovingPlanes,
    "neighbours": 8,
    "searchRadius": 50,
    "selMode": Types.SelectionMode.quadrant,
    "weightFunc": "0.04",  # a-priori sz=5m -> w=0.04
    "feature": [
        Types.GridFeature.sigmaz,
    ],
}
max_fill_len_px = 30
FILL_CFG = {
    "method": Types.FillMethod.adaptive,
    "boundaryRatio": 0.7,
    "maxArea": (max_fill_len_px * GRID_CFG["gridSize"]) ** 2,
}


class OpalsPipeline:
    def __init__(
        self,
        opals_proc_dir: Path | None = None,
        fill_holes: bool = True,
        lsm_cfg: dict | None = None,
        grid_cfg: dict | None = None,
        fill_cfg: dict | None = None,
    ):
        self.opals_proc_dir = opals_proc_dir
        self.fill_holes = fill_holes
        self.lsm_cfg = lsm_cfg if lsm_cfg else {}
        self.grid_cfg = grid_cfg if grid_cfg else {}
        self.fill_cfg = fill_cfg if fill_cfg else {}
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        if exc_type is not None:
            self.logger.error(f"Error occurred: {exc_val}")
            return False
        return True

    def run(
        self,
        dem_path: Path,
        ref_path: Path,
        stable_mask_path: Path,
        pcd_path: Path | None = None,
    ) -> Path:
        return main(
            dem_path=dem_path,
            ref_path=ref_path,
            stable_mask_path=stable_mask_path,
            pcd_path=pcd_path,
            opals_proc_dir=self.opals_proc_dir,
            fill_holes=self.fill_holes,
            lsm_cfg=self.lsm_cfg,
            grid_cfg=self.grid_cfg,
            fill_cfg=self.fill_cfg,
        )

    def cleanup(self):
        if self.opals_proc_dir and self.opals_proc_dir.exists():
            clean_lsm_tmp_files(self.opals_proc_dir)


def main(
    dem_path: Path,
    ref_path: Path,
    stable_mask_path: Path,
    pcd_path: Path = None,
    opals_proc_dir: Path = None,
    fill_holes: bool = True,
    lsm_cfg: dict = None,
    grid_cfg: dict = None,
    fill_cfg: dict = None,
) -> Path:
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    if not ref_path.exists():
        raise FileNotFoundError(f"Reference DEM not found: {ref_path}")

    if not stable_mask_path.exists():
        raise FileNotFoundError(f"Stable mask not found: {stable_mask_path}")

    if pcd_path is None:
        logger.warning("No point cloud provided. Using the DEM.")
        pcd_path = dem_path

    # Update default configurations
    if lsm_cfg is None:
        lsm_cfg = {}
    if grid_cfg is None:
        grid_cfg = {}
    if fill_cfg is None:
        fill_cfg = {}
    lsm_cfg = {**LSM_CFG, **lsm_cfg}
    grid_cfg = {**GRID_CFG, **grid_cfg}
    fill_cfg = {**FILL_CFG, **fill_cfg}

    # Create the OPALS processing directory
    opals_proc_dir.mkdir(parents=True, exist_ok=True)
    # Snap Reference and DEM to the mask for coregistration
    ref_snap = ref_path.parent / f"{ref_path.stem}_snapOpals.tif"
    if not ref_snap.exists():
        logger.info(f"Snapped reference DEM {ref_snap} not found, snapping...")
        mask_dem(ref_path, stable_mask_path, ref_snap).run()
        logger.info("Snapping done")
    else:
        logger.info(f"Snapped reference DEM {ref_snap} already exists")

    dem_snap = opals_proc_dir / f"{dem_path.stem}_snapOpals.tif"
    if not dem_snap.exists():
        logger.info(f"Snapped dem {dem_snap} not found, snapping...")
        mask_dem(dem_path, stable_mask_path, dem_snap).run()
        logger.info("Snapping done")
    else:
        logger.info(f"Snapped dem {dem_snap} already exists")

    # Run LSM coregistration
    logger.info(f"Running LSM coregistration for {dem_path.stem}...")
    lsm = LSM.LSM(
        inFile=[str(ref_snap), str(dem_snap)],
        gridMask=[str(stable_mask_path), str(stable_mask_path)],
        lsmTrafo=Types.LSMTrafoType.full,
        outParamFile=str(opals_proc_dir / f"{dem_path.stem}_lsm_param.xml"),
        **lsm_cfg,
    )
    lsm.run()
    logger.info(f"LSM coregistration done for {dem_path.stem}")

    # Import point cloud to ODM applying the estimated LSM transformation
    pcd_odm_lsm_path = opals_proc_dir / f"{dem_path.stem}_transLSM.odm"
    if pcd_odm_lsm_path.exists():
        logger.warning(
            f"ODM file already exists: {pcd_odm_lsm_path}. Overwriting it..."
        )

    logger.info(f"Importing {pcd_path} to ODM...")
    importer = Import.Import(
        inFile=str(pcd_path),
        outFile=str(pcd_odm_lsm_path),
        trafo=lsm.outTrafPars[0],
    )
    importer.run()
    logger.info(f"Import done: {pcd_odm_lsm_path}")

    # Grid the ODM point cloud to DEM
    opals_dem_name = f"{pcd_odm_lsm_path.stem}_{grid_cfg['interpolation'].name}_{grid_cfg['gridSize']}m.tif"
    opals_dem_path = opals_proc_dir / opals_dem_name
    gridder = Grid.Grid(
        inFile=str(pcd_odm_lsm_path),
        outFile=str(opals_dem_path),
        **grid_cfg,
    )
    gridder.commons.nbThreads = 12
    gridder.run()

    # Filling gaps
    if fill_holes:
        out_name = opals_dem_path.stem + f"_filled_{fill_cfg['method'].name}.tif"
        output_path = opals_proc_dir / out_name
        filler = FillGaps.FillGaps(
            inFile=str(opals_dem_path), outFile=str(output_path), **fill_cfg
        )
        filler.run()
    else:
        output_path = opals_dem_path

    # Clean up
    clean_lsm_tmp_files()
    importer.reset()
    gridder.reset()
    if fill_holes:
        filler.reset()

    return output_path


def mask_dem(
    dem_path: Path,
    stable_mask_path: Path,
    output_path: Path,
) -> Algebra.Algebra:
    return Algebra.Algebra(
        inFile=[str(stable_mask_path), str(dem_path)],
        outFile=str(output_path),
        limit="dataset:0",
        formula="r[1]",
    )


def clean_lsm_tmp_files(dir: Path = Path(".")):
    lsm_tmp_files = ["fix_dx.tif", "fix_dy.tif", "fixSctn.tif", "movSctn.tif"]
    for file in lsm_tmp_files:
        (dir / file).unlink(missing_ok=True)


def filter_and_transform_pc(pcd_path, out_path, z_min=600, z_max=4000):
    try:
        import laspy
    except ImportError:
        raise ImportError(
            "laspy is required to read LAS/LAZ files. Please, install it first."
        ) from None

    # Read LAS/LAZ file
    las = laspy.read(pcd_path)

    # Filter points by Z coordinate
    mask = (las.z >= z_min) & (las.z <= z_max)
    filtered_las = las[mask]

    # Save filtered point cloud temporarily
    filtered_las.write(out_path)


def parse_cli():
    parser = argparse.ArgumentParser(description="Process DEM data using OPALS.")
    parser.add_argument(
        "asp_proc_dir",
        type=Path,
        nargs="?",
        help="Path to the ASP processed data directory.",
    )
    parser.add_argument(
        "ref_path",
        type=Path,
        nargs="?",
        help="Path to the reference DEM file.",
    )
    parser.add_argument(
        "stable_mask_path",
        type=Path,
        nargs="?",
        help="Path to the stable mask file.",
    )
    parser.add_argument(
        "opals_proc_dir",
        type=Path,
        nargs="?",
        default="opals",
        help="Path to the OPALS processed data directory.",
    )
    parser.add_argument(
        "--fill_holes",
        action="store_true",
        default=True,
        help="Flag to fill holes in the DEM.",
    )
    parser.add_argument(
        "--lsm_cfg", type=str, default="{}", help="LSM configuration as a JSON string."
    )
    parser.add_argument(
        "--grid_cfg",
        type=str,
        default=None,
        help="Grid configuration as a JSON string.",
    )
    parser.add_argument(
        "--fill_cfg",
        type=str,
        default=None,
        help="Fill configuration as a JSON string.",
    )

    args = parser.parse_args()

    args.lsm_cfg = json.loads(args.lsm_cfg if args.lsm_cfg else "{}")
    args.grid_cfg = json.loads(args.grid_cfg if args.grid_cfg else "{}")

    return args


if __name__ == "__main__":
    # Parse CLI arguments
    args = parse_cli()

    # Run opals processing
    output_path = main(
        dem_path=args.asp_proc_dir / "stereo-DEM.tif",
        ref_path=args.ref_path,
        stable_mask_path=args.stable_mask_path,
        pcd_path=args.asp_proc_dir / "stereo-PC.las",
        opals_proc_dir=args.opals_proc_dir,
        fill_holes=args.fill_holes,
        lsm_cfg=args.lsm_cfg,
        grid_cfg=args.grid_cfg,
        fill_cfg=args.fill_cfg,
    )

    print("Done")

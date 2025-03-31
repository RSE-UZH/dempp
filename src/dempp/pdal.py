import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed

logger = logging.getLogger("dempp")

# Check if PDAL is available, if not available, log a warning (but not raise an error).
try:
    subprocess.run(["pdal", "--version"], check=True, capture_output=True)
except FileNotFoundError:
    logger.warning(
        "PDAL is not installed or not found in the system PATH. Some features may not work."
    )


class PDALPipelineBuilder:
    """
    A builder class for creating and executing PDAL pipelines with customizable filters and outputs.`
    """

    def __init__(self, input_cloud: str | Path):
        """
        Initialize the PDAL pipeline builder with an input point cloud.

        Args:
            input_cloud: Path to the input point cloud file (.las or .laz)
        """
        self.input_cloud = Path(input_cloud)
        self.pipeline: list[dict[str, Any]] = [str(self.input_cloud)]

    def add_splitter(self, length: int = 10000) -> "PDALPipelineBuilder":
        """
        Add a splitter filter to the pipeline for processing large point clouds in chunks.

        Args:
            length: The length of each chunk in points

        Returns:
            self: The pipeline builder instance for method chaining
        """
        self.pipeline.append({"type": "filters.splitter", "length": length})
        return self

    def add_colorization(
        self,
        raster_path: str | Path,
        dimension_name: str = "NCC",
        band: int = 1,
        scale: float = 1.0,
    ) -> "PDALPipelineBuilder":
        """
        Add colorization from a raster to the point cloud.

        Args:
            raster_path: Path to the raster file for colorization
            dimension_name: Name of the dimension to store raster values
            band: Band number in the raster to use for colorization
            scale: Scale factor to apply to raster values

        Returns:
            self: The pipeline builder instance for method chaining
        """
        self.pipeline.append(
            {
                "type": "filters.colorization",
                "raster": str(Path(raster_path)),
                "dimensions": f"{dimension_name}:{band}:{scale}",
            }
        )
        return self

    def add_value_filter(
        self,
        dimension: str,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> "PDALPipelineBuilder":
        """
        Add a range filter for a specific dimension.

        Args:
            dimension: The dimension to filter (e.g., "NCC", "Z")
            min_value: Minimum value to keep (None for no minimum)
            max_value: Maximum value to keep (None for no maximum)

        Returns:
            self: The pipeline builder instance for method chaining
        """
        range_str = f"{dimension}[{min_value if min_value is not None else ''}:{max_value if max_value is not None else ''}]"
        self.pipeline.append({"type": "filters.range", "limits": range_str})
        return self

    def add_expression_filter(self, expression: str) -> "PDALPipelineBuilder":
        """
        Add a complex expression filter.

        Args:
            expression: Expression string for filtering points

        Returns:
            self: The pipeline builder instance for method chaining
        """
        self.pipeline.append({"type": "filters.expression", "expression": expression})
        return self

    def add_merge(self) -> "PDALPipelineBuilder":
        """
        Add a merge filter to combine split point clouds.

        Returns:
            self: The pipeline builder instance for method chaining
        """
        self.pipeline.append({"type": "filters.merge"})
        return self

    def add_statistical_outlier_filter(
        self,
        mean_k: int = 10,
        multiplier: float = 2.0,
        where_clause: str | None = None,
    ) -> "PDALPipelineBuilder":
        """
        Add a statistical outlier filter to remove noise.

        Args:
            mean_k: Number of neighbors to analyze
            multiplier: Standard deviation multiplier threshold
            where_clause: Optional where clause to apply filter selectively

        Returns:
            self: The pipeline builder instance for method chaining
        """
        filter_config = {
            "type": "filters.outlier",
            "method": "statistical",
            "mean_k": mean_k,
            "multiplier": multiplier,
        }

        if where_clause:
            filter_config["where"] = where_clause

        self.pipeline.append(filter_config)
        return self

    def add_radius_outlier_filter(
        self, radius: float = 30.0, min_k: int = 8, where_clause: str | None = None
    ) -> "PDALPipelineBuilder":
        """
        Add a radius-based outlier filter.

        Args:
            radius: Search radius
            min_k: Minimum number of neighbors required
            where_clause: Optional where clause to apply filter selectively

        Returns:
            self: The pipeline builder instance for method chaining
        """
        filter_config = {
            "type": "filters.outlier",
            "method": "radius",
            "radius": radius,
            "min_k": min_k,
        }

        if where_clause:
            filter_config["where"] = where_clause

        self.pipeline.append(filter_config)
        return self

    def add_dem_writer(
        self,
        output_path: str | Path,
        resolution: float = 10.0,
        gdaldriver: str = "GTiff",
        output_type: str = "mean",
    ) -> "PDALPipelineBuilder":
        """
        Add a GDAL writer to create a DEM from the point cloud.

        Args:
            output_path: Path for the output DEM file
            resolution: Cell size for the output raster in coordinate system units
            gdaldriver: GDAL driver to use (e.g., "GTiff")
            output_type: Output type ("all", "min", "max", "mean", etc.)

        Returns:
            self: The pipeline builder instance for method chaining
        """
        self.pipeline.append(
            {
                "type": "writers.gdal",
                "filename": str(Path(output_path)),
                "gdaldriver": gdaldriver,
                "output_type": output_type,
                "resolution": str(resolution),
            }
        )
        return self

    def add_las_writer(
        self,
        output_path: str | Path,
        compression: bool = True,
        dataformat_id: int = 3,
        minor_version: int = 2,
        extra_dims: str = "all",
    ) -> "PDALPipelineBuilder":
        """
        Add a LAS/LAZ writer to save the processed point cloud.

        Args:
            output_path: Path for the output point cloud file
            compression: Whether to compress the output (LAZ if True)
            dataformat_id: LAS format ID (3 supports RGB values)
            minor_version: LAS minor version
            extra_dims: Extra dimensions to include (usually "all")

        Returns:
            self: The pipeline builder instance for method chaining
        """
        self.pipeline.append(
            {
                "type": "writers.las",
                "filename": str(Path(output_path)),
                "compression": str(compression).lower(),
                "dataformat_id": str(dataformat_id),
                "minor_version": str(minor_version),
                "extra_dims": extra_dims,
            }
        )
        return self

    def save_pipeline(self, output_path: str | Path) -> Path:
        """
        Save the pipeline configuration to a JSON file.

        Args:
            output_path: Path where the pipeline JSON should be saved

        Returns:
            Path: The path to the saved pipeline file
        """
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(self.pipeline, f, indent=4)
        return output_path

    def execute(
        self,
        save_pipeline: bool = True,
        pipeline_path: str | Path | None = None,
        verbose: int = 1,
    ) -> tuple[bool, str]:
        """
        Execute the PDAL pipeline.

        Args:
            save_pipeline: Whether to save the pipeline JSON before execution
            pipeline_path: Path where to save the pipeline JSON (uses a temp file if None)
            verbose: Verbosity level (0-3) for PDAL output

        Returns:
            Tuple[bool, str]: Success status and output/error message
        """
        if save_pipeline:
            if pipeline_path is None:
                pipeline_path = self.input_cloud.with_suffix(".pipeline.json")
            else:
                pipeline_path = Path(pipeline_path)

            self.save_pipeline(pipeline_path)
        else:
            # Create a temporary pipeline file
            pipeline_path = Path("temp_pipeline.json")
            self.save_pipeline(pipeline_path)

        try:
            result = subprocess.run(
                ["pdal", "pipeline", str(pipeline_path), "-v", str(verbose)],
                check=True,
                capture_output=True,
                text=True,
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, f"Error executing pipeline: {e}\n{e.stderr}"
        finally:
            if not save_pipeline and pipeline_path.exists():
                pipeline_path.unlink()


def create_filter_pipeline(
    input_cloud: str | Path,
    output_cloud: str | Path,
    ncc_raster: str | Path | None = None,
    output_dem: str | Path | None = None,
    dem_resolution: float = 10.0,
    ncc_min: float = 0.3,
    z_min: float = 0,
    z_max: float = 2500,
    split_size: int = 10000,
) -> PDALPipelineBuilder:
    """
    Create a standard pipeline with common processing steps.

    Args:
        input_cloud: Path to input point cloud (.las/.laz)
        output_cloud: Path for processed output point cloud
        ncc_raster: Optional path to NCC raster for colorization
        output_dem: Optional path for output DEM
        dem_resolution: Resolution for output DEM in units of the coordinate system
        ncc_min: Minimum NCC value to keep
        z_min: Minimum elevation to keep
        z_max: Maximum elevation to keep
        split_size: Size of chunks for the splitter filter

    Returns:
        PDALPipelineBuilder: Configured pipeline builder
    """
    builder = PDALPipelineBuilder(input_cloud)

    # Add splitter for large point clouds
    builder.add_splitter(length=split_size)

    # Add colorization if NCC raster is provided
    if ncc_raster:
        builder.add_colorization(ncc_raster, dimension_name="NCC")
        # Add expression filter for NCC and Z range
        builder.pipeline.append(
            {
                "type": "filters.expression",
                "expression": f"NCC >= {ncc_min} && (Z >= {z_min} && Z <= {z_max})",
            }
        )
    else:
        # Just filter by Z if no NCC raster
        builder.add_value_filter("Z", min_value=z_min, max_value=z_max)

    # Merge the split parts
    builder.add_merge()

    # Add outlier filters
    builder.add_statistical_outlier_filter(mean_k=6, multiplier=2.5)
    builder.add_statistical_outlier_filter(
        mean_k=10, multiplier=1.8, where_clause="Classification != 7"
    )
    builder.add_radius_outlier_filter(
        radius=30, min_k=6, where_clause="Classification != 7"
    )

    # Keep only points that are not classified as noise
    builder.add_expression_filter("Classification != 7")

    # Add DEM writer if requested
    if output_dem:
        builder.add_dem_writer(output_dem, resolution=dem_resolution)

    # Add LAS/LAZ writer
    builder.add_las_writer(output_cloud)

    return builder


def _process_point_cloud(
    input_cloud: Path,
    output_dir: Path,
    dem_resolution: float = 10.0,
    ncc_min: float = 0.4,
    z_min: float = 0,
    z_max: float = 2500,
    save_dem: bool = True,
    verbose: int = 1,
) -> tuple[bool, str, Path]:
    """
    Process a single point cloud with PDAL pipeline.

    Args:
        input_cloud: Path to input point cloud file
        output_dir: Directory to save outputs
        dem_resolution: Resolution for output DEM in coordinate system units
        ncc_min: Minimum NCC value to keep
        z_min: Minimum elevation to keep
        z_max: Maximum elevation to keep
        save_dem: Whether to generate a DEM from the filtered point cloud
        verbose: Verbosity level for PDAL (0-3)

    Returns:
        Tuple[bool, str, Path]: Success status, message, and output path
    """
    # Define output paths
    output_cloud = output_dir / f"{input_cloud.stem}_filtered.laz"

    # Get matching NCC raster if it exists
    ncc_file = input_cloud.parent / f"{input_cloud.stem}_stereo-ncc.tif"
    if not ncc_file.exists():
        logger.warning(
            f"NCC file not found for {input_cloud}. Processing without NCC filtering."
        )
        ncc_file = None

    # Set up DEM output path if requested
    dem_file = output_dir / f"{input_cloud.stem}_filtered_DEM.tif" if save_dem else None

    # Create pipeline
    pipeline = create_filter_pipeline(
        input_cloud=input_cloud,
        output_cloud=output_cloud,
        ncc_raster=ncc_file,
        output_dem=dem_file,
        dem_resolution=dem_resolution,
        ncc_min=ncc_min,
        z_min=z_min,
        z_max=z_max,
    )

    # Execute pipeline
    success, message = pipeline.execute(
        save_pipeline=True,
        pipeline_path=output_dir / f"{input_cloud.stem}_pipeline.json",
        verbose=verbose,
    )

    if success:
        logger.info(f"Successfully processed {input_cloud.name}")
        return True, f"Success: {input_cloud.name}", output_cloud
    else:
        logger.error(f"Failed to process {input_cloud.name}: {message}")
        return False, f"Failed: {input_cloud.name} - {message}", input_cloud


def batch_process_point_clouds(
    pcd_dir: str | Path,
    output_dir: str | Path | None = None,
    pattern: str = "**/*.las",
    n_jobs: int = -1,
    dem_resolution: float = 10.0,
    ncc_min: float = 0.4,
    z_min: float = 0,
    z_max: float = 2500,
    save_dem: bool = True,
    verbose: int = 1,
) -> list[tuple[bool, str, Path]]:
    """
    Batch process multiple point clouds using parallel processing.

    Args:
        pcd_dir: Directory containing point clouds
        output_dir: Directory to save outputs (defaults to pcd_dir if None)
        pattern: Glob pattern to match point cloud files
        n_jobs: Number of parallel jobs (-1 for all cores)
        dem_resolution: Resolution for output DEMs
        ncc_min: Minimum NCC value to keep
        z_min: Minimum elevation to keep
        z_max: Maximum elevation to keep
        save_dem: Whether to generate DEMs from filtered point clouds
        verbose: Verbosity level for PDAL (0-3)

    Returns:
        List[Tuple[bool, str, Path]]: List of processing results
    """
    pcd_dir = Path(pcd_dir)

    # Use input directory as output if not specified
    if output_dir is None:
        output_dir = pcd_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    # Find all point cloud files
    input_files = sorted(list(pcd_dir.glob(pattern)))
    n_files = len(input_files)

    if n_files == 0:
        logger.warning(
            f"No point cloud files found in {pcd_dir} with pattern '{pattern}'"
        )
        return []

    logger.info(f"Found {n_files} point cloud files to process")

    # Process in parallel
    start_time = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_point_cloud)(
            input_file,
            output_dir,
            dem_resolution,
            ncc_min,
            z_min,
            z_max,
            save_dem,
            verbose,
        )
        for input_file in input_files
    )
    end_time = time.time()

    # Summarize results
    success_count = sum(1 for r in results if r[0])
    logger.info(f"Processing completed: {success_count}/{n_files} successful")
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")

    return results


if __name__ == "__main__":
    # Example usage
    input_file = Path("outputs/pcd/048_006-009_S5_713-217-0_2003-05-04_pcd_3413.las")
    ncc_file = Path(
        "outputs/pcd/048_006-009_S5_713-217-0_2003-05-04_pcd_3413_stereo-ncc.tif"
    )
    output_file = Path(
        "outputs/pcd/048_006-009_S5_713-217-0_2003-05-04_pcd_3413_filtered.laz"
    )
    dem_file = Path(
        "outputs/pcd/048_006-009_S5_713-217-0_2003-05-04_pcd_3413_filtered_DEM.tif"
    )

    # Create a pipeline with standard configuration
    pipeline = create_filter_pipeline(
        input_cloud=input_file,
        output_cloud=output_file,
        ncc_raster=ncc_file,
        output_dem=dem_file,
        dem_resolution=10,  # 10m resolution
        ncc_min=0.4,  # Higher NCC threshold
        z_min=0,
        z_max=2500,
    )

    # Or build a custom pipeline
    # pipeline = (PDALPipelineBuilder(input_file)
    #     .add_splitter(20000)
    #     .add_colorization(ncc_file)
    #     .add_value_filter("NCC", min_value=0.3)
    #     .add_value_filter("Z", min_value=0, max_value=2500)
    #     .add_merge()
    #     .add_statistical_outlier_filter()
    #     .add_las_writer(output_file)
    # )

    # Execute the pipeline
    success, message = pipeline.execute(save_pipeline=True, verbose=3)

    if success:
        print("Pipeline executed successfully!")
        print(f"Output point cloud: {output_file}")
        if dem_file:
            print(f"Output DEM: {dem_file}")
    else:
        print(f"Pipeline failed: {message}")

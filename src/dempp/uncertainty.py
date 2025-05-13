import logging
from itertools import combinations
from pathlib import Path

import cloudpickle
import geopandas as gpd
import geoutils as gu
import numpy as np
import pandas as pd
import xdem
from geoutils import Vector
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from shapely.geometry import shape
from tqdm import tqdm

# Initialize logger with propagate=False to prevent double logging
logger = logging.getLogger("dempp")
logger.propagate = False

# Only add a handler if the logger doesn't have any
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)


class UncertaintyData:
    """Class to store data for uncertainty analysis."""

    def __init__(self):
        """Initialize an empty data container."""
        # Data containers
        self.ref_dem = None
        self.dem = None
        self.dh = None
        self.stable_mask = None
        self.glacier_outlines = None
        self.glacier_mask = None

        # Paths
        self.ref_dem_path = None
        self.dem_path = None
        self.stable_mask_path = None
        self.glacier_outlines_path = None
        self.subsample_res = None
        self.no_data_value = None

        # Heteroscedasticity analysis
        self.predictors = {}
        self.df_binned = None
        self.zscores = None
        self.dh_err_fun = None

        # Spatial correlation analysis
        self.sigma_dh = None
        self.variogram_data = None
        self.variogram_function = None
        self.variogram_params = None

    def to_pickle(self, path):
        """Save the current state to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "wb") as f:
                cloudpickle.dump(self, f)
            return True
        except Exception as e:
            logger.error(f"Failed to save to {path}: {e}")
            return False

    @classmethod
    def from_pickle(cls, path):
        """Load the data state from a pickle file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist.")

        try:
            with open(path, "rb") as f:
                return cloudpickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load from {path}: {e}")
            raise


def analyze_dem_uncertainty(
    ref_dem_path,
    dem_path,
    stable_mask_path,
    glacier_outlines_path=None,
    compute_terrain_features=True,
    attibute_list=None,
    additional_predictors=None,
    subsample_res=None,
    output_dir=None,
    area_vector=None,
    column_name="rgi_id",
    area_name=None,
    min_area_fraction=0.05,
    neff_args=None,
    save_intermediate=False,
) -> tuple[UncertaintyData, pd.DataFrame]:
    """
    Analyze DEM uncertainty and return results.

    Args:
        ref_dem_path: Path to reference DEM
        dem_path: Path to DEM to analyze
        stable_mask_path: Path to stable terrain mask
        glacier_outlines_path: Path to glacier outlines
        compute_terrain_features: Whether to compute terrain features
        attibute_list: List of terrain attributes to extract
        additional_predictors: Additional predictors for uncertainty
        subsample_res: Resolution to subsample to
        output_dir: Directory to save outputs
        area_vector: Vector with areas of interest
        column_name: Column name for area identification
        area_name: Name(s) for area(s) to analyze
        min_area_fraction: Minimum area percentage
        neff_args: Arguments for effective samples calculation
        save_intermediate: Whether to save intermediate results

    Returns:
        tuple: (data, area_dh_uncertainty)
            - data: UncertaintyData container with analysis results
            - area_dh_uncertainty: DataFrame with uncertainty for the elevation difference of each area given in area_vector
    """
    # Load data
    data = load_uncertainty_data(
        ref_dem_path=ref_dem_path,
        dem_path=dem_path,
        stable_mask_path=stable_mask_path,
        glacier_outlines_path=glacier_outlines_path,
        subsample_res=subsample_res,
        additional_predictors=additional_predictors,
    )

    # Compute elevation difference
    compute_elevation_difference(data, get_statistics=True)

    # Extract terrain features
    if compute_terrain_features:
        if attibute_list is None:
            attibute_list = ["slope", "maximum_curvature"]
        extract_terrain_features(data, attribute=attibute_list, compute_abs_maxc=True)

    # Analyze heteroscedasticity
    analyze_heteroscedasticity(data)
    if output_dir is not None:
        plot_dir = Path(output_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_error_vs_predictors_1D(data, path=output_dir / "1D binning.png")
        plot_error_vs_predictors_nD(data, path=output_dir / "nD binning.png")
        plot_error_map(data, path=output_dir / "predicted_error_map.png")
        data.sigma_dh.save(output_dir / "predicted_error.tif")

    if save_intermediate and output_dir is not None:
        data.to_pickle(output_dir / "analyzer_state_intermediate.pkl")

    # Analyze spatial correlation
    analyze_spatial_correlation(data)
    if output_dir is not None:
        plot_variogram(data, path=output_dir / "variogram.png")
        data.variogram_params.to_csv(
            path_or_buf=output_dir / "variogram_params.csv", index=False, header=True
        )
    if save_intermediate and output_dir is not None:
        data.to_pickle(output_dir / "analyzer_state_intermediate.pkl")

    # Compute uncertainty for specific area
    if area_vector is not None:
        area_dh_uncertainty = compute_uncertainty_for_area(
            data,
            area_vector=area_vector,
            column_name=column_name,
            area_name=area_name,
            min_area_fraction=min_area_fraction,
            neff_args=neff_args,
            n_jobs=1,
        )
        if output_dir is not None and area_dh_uncertainty is not None:
            area_dh_uncertainty.to_csv(
                path_or_buf=output_dir / "areas_uncertainty.csv",
                index=True,
                header=True,
            )
    else:
        area_dh_uncertainty = None

    # Save the data state to a pickle file
    if output_dir is not None:
        data.to_pickle(output_dir / "uncertainty_data_state.pkl")
        if save_intermediate:
            (output_dir / "analyzer_state_intermediate.pkl").unlink(missing_ok=True)

    return data, area_dh_uncertainty


def load_uncertainty_data(
    ref_dem_path,
    dem_path,
    stable_mask_path=None,
    glacier_outlines_path=None,
    subsample_res=None,
    no_data_value=-9999,
    additional_predictors=None,
):
    """
    Load data for uncertainty analysis.

    Args:
        ref_dem_path: Path to reference DEM
        dem_path: Path to DEM to analyze
        stable_mask_path: Path to stable terrain mask
        glacier_outlines_path: Path to glacier outlines
        subsample_res: Resolution to subsample to
        no_data_value: No data value for DEMs
        additional_predictors: Dict mapping names to file paths

    Returns:
        UncertaintyData: Data container with loaded data
    """

    logger.info("Loading data...")
    data = UncertaintyData()

    # Store paths
    data.ref_dem_path = Path(ref_dem_path)
    data.dem_path = Path(dem_path)
    data.stable_mask_path = Path(stable_mask_path) if stable_mask_path else None
    data.glacier_outlines_path = (
        Path(glacier_outlines_path) if glacier_outlines_path else None
    )
    data.subsample_res = subsample_res
    data.no_data_value = no_data_value

    # Load reference DEM
    data.ref_dem = xdem.DEM(data.ref_dem_path)
    data.ref_dem.set_area_or_point("Area")
    data.ref_dem.set_nodata(data.no_data_value)

    # Load DEM to analyze
    data.dem = xdem.DEM(data.dem_path)
    data.dem.set_area_or_point("Area")
    data.dem.set_nodata(data.no_data_value)

    # Subsample if resolution specified
    if data.subsample_res:
        data.dem.reproject(res=data.subsample_res, resampling="bilinear", inplace=True)

    # Reproject the reference DEM to the DEM
    data.ref_dem.reproject(data.dem, resampling="bilinear", inplace=True)

    # Load glacier outlines if provided
    if data.glacier_outlines_path:
        data.glacier_outlines = gu.Vector(data.glacier_outlines_path).crop(data.dem)
        data.glacier_mask = data.glacier_outlines.create_mask(data.dem)

    # Load stable area mask if provided
    if data.stable_mask_path:
        stable_mask_raster = gu.Raster(data.stable_mask_path)
        stable_mask_raster.set_nodata(255)
        stable_mask_raster.reproject(data.dem, resampling="nearest", inplace=True)
        data.stable_mask = stable_mask_raster == 1

    # Load other data if provided (e.g., correlation value map)
    if additional_predictors:
        for key, value in additional_predictors.items():
            data.predictors[key] = gu.Raster(value).reproject(
                data.dem, resampling="bilinear"
            )

    logger.info("Data loaded successfully.")
    return data


def compute_elevation_difference(data, get_statistics=True):
    """
    Compute elevation difference between reference and analyzed DEMs.

    Args:
        data: UncertaintyData container
        get_statistics: Whether to get statistics

    Returns:
        tuple: (elevation difference DEM, stats dict)
    """
    logger.info("Computing elevation difference...")
    data.dh = data.ref_dem - data.dem

    if get_statistics:
        stats = data.dh.get_stats(inlier_mask=data.stable_mask)
        logger.info("Elevation difference statistics:")
        for key, value in stats.items():
            logger.info(f"\t{key}: {value:.2f}")
    else:
        stats = None

    logger.info("Elevation difference computed.")
    return data.dh, stats


def extract_terrain_features(data, attribute=None, compute_abs_maxc=True):
    """
    Extract terrain features as predictors for uncertainty.

    Args:
        data: UncertaintyData container
        attribute: List of terrain attributes to extract
        compute_abs_maxc: Whether to compute absolute max curvature

    Returns:
        dict: Dictionary of extracted features
    """

    logger.info(f"Extracting terrain features: {attribute}...")
    if attribute is None:
        attribute = ["slope", "maximum_curvature"]

    # Compute slope and maximum curvature
    outs = xdem.terrain.get_terrain_attribute(data.ref_dem, attribute=attribute)

    # Store the terrain features in the predictors dictionary
    for key, feature_data in zip(attribute, outs, strict=False):
        # If requested, compute absolute value of the maximum curvature
        if key == "maximum_curvature" and compute_abs_maxc:
            logger.debug("Computing absolute values of the maximum curvature")
            feature_data = np.abs(feature_data)
        data.predictors[key] = feature_data

    logger.info(f"Terrain features extracted successfully: {attribute}")
    return data.predictors


def prepare_stable_terrain_data(data, nmad_factor=5):
    """
    Extract data from stable terrain and filter outliers.

    Args:
        data: UncertaintyData container
        nmad_factor: Factor to multiply NMAD for outlier filtering

    Returns:
        tuple: (dh_arr, predictors_arrays)
    """
    import xdem

    logger.info("Preparing stable terrain data...")
    if data.stable_mask is None:
        raise ValueError("Stable mask not loaded. Call load_data first.")
    if data.dh is None:
        raise ValueError(
            "Elevation difference not computed. Call compute_elevation_difference first."
        )
    if not data.predictors:
        raise ValueError(
            "Predictors not computed. Call extract_terrain_features first."
        )

    # Apply mask to the elevation difference
    dh_arr = data.dh[data.stable_mask].filled(np.nan)

    # Filter outliers in the elevation difference using NMAD
    arr_lim = nmad_factor * xdem.spatialstats.nmad(dh_arr)
    logger.debug(f"Filtering outliers with limit: {arr_lim}")
    outlier_mask = np.abs(dh_arr) > arr_lim
    dh_arr[outlier_mask] = np.nan

    # Extract arrays for stable terrain for each predictor
    predictors_arrays = {}
    for key, pred_data in data.predictors.items():
        # Apply masks to the data (set masked values to NaN)
        arr = pred_data[data.stable_mask].filled(np.nan)
        arr[outlier_mask] = np.nan
        predictors_arrays[key] = arr

    logger.info("Stable terrain data prepared.")
    return dh_arr, predictors_arrays


def analyze_heteroscedasticity(
    data,
    nmad_factor=5,
    list_var_bins=None,
    list_ranges=None,
    statistics=None,
    nd_binning_statistic="nmad",
    nd_binning_min_count=100,
):
    """
    Analyze error heteroscedasticity and create error function.

    Args:
        data: UncertaintyData container
        nmad_factor: Factor for outlier removal
        list_var_bins: List of bin limits for each predictor
        list_ranges: List of ranges for each predictor
        statistics: List of statistics to compute
        nd_binning_statistic: Statistic for ND binning
        nd_binning_min_count: Minimum count for ND binning

    Returns:
        tuple: (df_binned, zscores, dh_err_fun, sigma_dh)
    """
    logger.info("Analyzing heteroscedasticity...")

    # Get stable terrain data
    dh_arr, stable_data = prepare_stable_terrain_data(data, nmad_factor=nmad_factor)

    # Prepare data for n-d binning
    var_names = list(stable_data.keys())
    predictor_list = [stable_data[key] for key in var_names]

    # Compute the bin limits for each predictor
    if list_var_bins is None:
        list_var_bins = [
            np.linspace(
                np.floor(np.nanpercentile(d, 0.5)),
                np.ceil(np.nanpercentile(d, 99.5)),
                10,
            )
            for d in predictor_list
        ]

    # Set the statistics to compute
    if statistics is None:
        statistics = ["count", np.nanmedian, xdem.spatialstats.nmad]

    # Compute n-d binning with all the predictors
    logger.info("Computing n-d binning of the predictors...")
    df = xdem.spatialstats.nd_binning(
        values=dh_arr,
        list_var=predictor_list,
        list_var_names=var_names,
        statistics=statistics,
        list_var_bins=list_var_bins,
        list_ranges=list_ranges,
    )

    # Fit error model
    logger.info("Fitting error model on the predictors...")
    unscaled_dh_err_fun = xdem.spatialstats.interp_nd_binning(
        df,
        list_var_names=var_names,
        statistic=nd_binning_statistic,
        min_count=nd_binning_min_count,
    )

    # Compute the mean predicted elevation error on the stable terrain
    dh_err_stable = unscaled_dh_err_fun(tuple([stable_data[key] for key in var_names]))
    logger.info(
        f"The spread of elevation difference is {xdem.spatialstats.nmad(dh_arr):.2f} compared to a mean predicted elevation error of {np.nanmean(dh_err_stable):.2f}."
    )

    # Two-step standardization
    logger.info("Standardizing the predicted errors...")
    zscores, dh_err_fun = xdem.spatialstats.two_step_standardization(
        dvalues=dh_arr,
        list_var=predictor_list,
        unscaled_error_fun=unscaled_dh_err_fun,
    )

    # Compute the error function for the whole DEM
    logger.info("Computing the error function for the whole DEM...")
    sigma_dh = data.dh.copy(
        new_array=dh_err_fun(tuple([data.predictors[key].data for key in var_names]))
    )
    sigma_dh.set_mask(data.dem.data.mask)

    # Save the results
    data.df_binned = df
    data.zscores = zscores
    data.dh_err_fun = dh_err_fun
    data.sigma_dh = sigma_dh

    logger.info("Heteroscedasticity analysis completed.")
    return df, zscores, dh_err_fun, sigma_dh


def analyze_spatial_correlation(
    data,
    standardize=True,
    n_samples=1000,
    subsample_method="cdist_equidistant",
    n_variograms=3,
    estimator="dowd",
    random_state=None,
    list_models=None,
    sample_kwargs=None,
    fit_kwargs=None,
):
    """
    Analyze spatial correlation of standardized errors.

    Args:
        data: UncertaintyData container
        standardize: Whether to standardize elevation differences
        n_samples: Number of samples for variogram
        subsample_method: Method for subsampling
        n_variograms: Number of variogram realizations
        estimator: Estimator for variogram
        random_state: Random seed
        list_models: List of models for variogram fitting
        sample_kwargs: Additional sampling arguments
        fit_kwargs: Additional fitting arguments

    Returns:
        tuple: (variogram_function, variogram_params)
    """
    logger.info("Analyzing spatial correlation...")
    if data.sigma_dh is None:
        raise ValueError(
            "Error raster not computed. Call analyze_heteroscedasticity first."
        )

    # Compute standardized elevation difference if requested, otherwise use original
    z_dh = data.dh / data.sigma_dh if standardize else data.dh.copy()

    # Remove values on unstable terrain and large outliers
    z_dh.data[~data.stable_mask.data] = np.nan
    z_dh.data[np.abs(z_dh.data) > 4] = np.nan

    # Sample empirical variogram
    df_vgm = xdem.spatialstats.sample_empirical_variogram(
        values=z_dh,
        subsample=n_samples,
        subsample_method=subsample_method,
        n_variograms=n_variograms,
        estimator=estimator,
        random_state=random_state,
        n_jobs=n_variograms,
        **(sample_kwargs or {}),
    )

    # Fit variogram model
    if list_models is None:
        list_models = ["Spherical", "Spherical"]
    func_sum_vgm, params_vgm = xdem.spatialstats.fit_sum_model_variogram(
        list_models,
        empirical_variogram=df_vgm,
        **(fit_kwargs or {}),
    )

    data.variogram_data = df_vgm
    data.variogram_function = func_sum_vgm
    data.variogram_params = params_vgm

    logger.info("Spatial correlation analysis completed.")
    return func_sum_vgm, params_vgm


def compute_uncertainty_for_area(
    data,
    area_vector=None,
    column_name="rgi_id",
    area_name=None,
    min_area_fraction=0.05,
    neff_args=None,
    n_jobs=1,
):
    """
    Compute uncertainty for specific area(s) (e.g., glaciers).

    Args:
        data: UncertaintyData container
        area_vector: Vector with areas of interest
        column_name: Column name for area identification
        area_name: Name(s) for area(s) to analyze
        min_area_fraction: Minimum area percentage
        neff_args: Arguments for effective samples calculation
        n_jobs: Number of parallel jobs

    Returns:
        pd.DataFrame: Results for each area
    """

    def _process_single_area(params):
        """Process a single area to compute uncertainty.

        Args:
            params (dict): Dictionary containing all required parameters:
                - idx_row_tuple: Tuple containing (index, row) for the area
                - work_vector: Vector with all areas
                - column_name: Column name for area identification
                - ref_dem: Reference DEM
                - dh: Elevation difference DEM
                - sigma_dh: Error/uncertainty map
                - variogram_params: Parameters for variogram model
                - min_area_fraction: Minimum area fraction to process
                - neff_args: Arguments for effective number of samples calculation
                - dem_path_name: DEM path name for logging

        Returns:
            tuple or None: Tuple containing area_id and result dict, or None if skipped
        """

        logger = logging.getLogger("dempp")

        # Extract parameters
        idx_row_tuple = params["idx_row_tuple"]
        work_vector = params["work_vector"]
        column_name = params["column_name"]
        ref_dem = params["ref_dem"]
        dh = params["dh"]
        sigma_dh = params["sigma_dh"]
        variogram_params = params["variogram_params"]
        min_area_fraction = params["min_area_fraction"]
        neff_args = params.get("neff_args", {})
        dem_path_name = params.get("dem_path_name", "DEM")

        # Extract area information
        idx, row = idx_row_tuple
        area_id = row[column_name]
        logger.info(f"Processing area: {area_id}")

        # Extract the geometry without copying the entire vector
        geom = shape(work_vector.ds.iloc[idx].geometry)

        # Create a minimal single geometry Vector with just what's needed
        gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs=work_vector.crs)
        single_geom = Vector(gdf)

        # Create area mask
        area_mask = single_geom.create_mask(ref_dem)

        # Check if the mask contains any valid pixels
        if not np.any(area_mask):
            logger.warning(f"Area {area_id} has no valid pixels - skipping")
            return None

        # Check if the DEM covers at least min_area_fraction of the area
        dem_px_in_area = len(dh[area_mask].compressed())
        area_coverage = dem_px_in_area / np.sum(area_mask)
        if area_coverage < min_area_fraction:
            logger.info(
                f"{dem_path_name} covers only {area_coverage:.2%} of the area {area_id}. "
                f"It's less than the minimum fraction {min_area_fraction:.2%} - skipping"
            )
            return None

        # Compute the mean elevation difference in the area and the mean sigma
        dh_area = np.nanmean(dh[area_mask])
        mean_sig = np.nanmean(sigma_dh[area_mask])

        # Calculate effective number of samples
        n_eff = xdem.spatialstats.number_effective_samples(
            area=single_geom, params_variogram_model=variogram_params, **neff_args
        )

        # Rescale the standard deviation of the mean elevation difference with the effective number of samples
        sig_dh_area = (
            mean_sig / np.sqrt(n_eff) if n_eff > 0 else np.nan
        )  # Avoid division by zero or sqrt of negative

        err_perc = (
            sig_dh_area / abs(dh_area) * 100
            if dh_area != 0 and not np.isnan(sig_dh_area)
            else float("inf")
        )

        logger.info(
            f"Area {area_id}: Mean elevation difference: {dh_area:.2f} ± {sig_dh_area:.2f} m "
            f"({err_perc:.2f}%) - neff samples: {n_eff:.2f}."
        )

        logger.info(f"Completed uncertainty calculation for area {area_id}")

        return area_id, {
            "mean_elevation_diff": dh_area,
            "mean_uncertainty_unscaled": mean_sig,
            "area": single_geom.area.iloc[0],
            "effective_samples": n_eff,
            "uncertainty": sig_dh_area,
        }

    logger.info("Computing uncertainty for area(s)...")
    if not hasattr(data, "variogram_params") or data.variogram_params is None:
        raise ValueError(
            "Spatial correlation not analyzed. Call analyze_spatial_correlation first."
        )

    # Determine which vector to use (input or stored glacier outlines)
    if area_vector is None:
        if data.glacier_outlines is None:
            raise ValueError("No area vector provided and glacier outlines not loaded")
        work_vector = data.glacier_outlines.copy()
        logger.debug("Using stored glacier outlines")
    else:
        if isinstance(area_vector, (str, Path)):
            try:
                area_vector = gu.Vector(area_vector)
            except Exception as e:
                raise ValueError(
                    f"Failed to load area vector from {area_vector}: {e}"
                ) from e
        elif not isinstance(area_vector, gu.Vector):
            raise ValueError(
                "area_vector must be a geoutils.Vector or a path to a vector file."
            )
        work_vector = area_vector.copy()
        logger.debug("Using provided area vector")

    # Check if column_name exists in the vector
    if column_name not in work_vector.ds.columns:
        available_columns = work_vector.ds.columns.tolist()
        raise ValueError(
            f"Column '{column_name}' not found in vector dataset. "
            f"Available columns: {available_columns}"
        )

    # If area_name is provided, filter the vector to include only matching areas
    if area_name is not None:
        # Filter vector by area name(s)
        if isinstance(area_name, str):
            area_name = [area_name]
        filtered_vector = work_vector[work_vector.ds[column_name].isin(area_name)]

        if filtered_vector.ds.empty:
            available_ids = work_vector.ds[column_name].unique().tolist()
            raise ValueError(
                f"Area name(s) {area_name} not found in column '{column_name}'. "
                f"Available IDs: {available_ids[:5]}{'...' if len(available_ids) > 5 else ''}"
            )

        work_vector = filtered_vector
        logger.debug(
            f"Filtered vector to {len(work_vector.ds)} geometries matching {area_name}"
        )

    # Ensure vector is in a projected CRS
    if work_vector.crs.is_geographic:
        logger.debug("Converting vector from geographic to projected CRS")
        work_vector = work_vector.to_crs(crs=work_vector.ds.estimate_utm_crs())

    # Process each geometry in the vector in parallel
    # Prepare tasks for parallel processing with all required parameters
    tasks = []
    for idx, row in work_vector.ds.iterrows():
        params = {
            "idx_row_tuple": (idx, row),
            "work_vector": work_vector,
            "column_name": column_name,
            "ref_dem": data.ref_dem,
            "dh": data.dh,
            "sigma_dh": data.sigma_dh,
            "variogram_params": data.variogram_params,
            "min_area_fraction": min_area_fraction,
            "neff_args": neff_args or {},
            "dem_path_name": str(data.dem_path.name),
        }
        tasks.append(params)

    try:
        logger.info("Processing areas in parallel...")
        processed_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_process_single_area)(task) for task in tasks
        )
    except MemoryError as e:
        logger.error(f"Memory error during parallel processing: {e}")
        logger.info(
            "Switching to single-threaded processing due to memory constraints."
        )
        processed_results = []
        for task in tqdm(tasks):
            # Force a lower resolution for rasterization
            task["neff_args"] = {"rasterize_resolution": 50}
            processed_results.append(_process_single_area(task))

    # Filter out None results (skipped areas) and create a dataframe
    results = {}
    for res_tuple in processed_results:
        if res_tuple is not None:
            area_id, data_dict = res_tuple
            results[area_id] = data_dict

    results_df = pd.DataFrame.from_dict(results, orient="index")

    # Ensure correct dtypes if dataframe is not empty but some columns might be all NaN
    if not results_df.empty:
        expected_columns = {
            "mean_elevation_diff": float,
            "mean_uncertainty_unscaled": float,
            "area": float,
            "effective_samples": float,
            "uncertainty": float,
        }
        for col, dtype in expected_columns.items():
            if col in results_df.columns:
                results_df[col] = results_df[col].astype(dtype)
            else:  # if a column is missing because all areas were skipped
                results_df[col] = pd.Series(dtype=dtype)
    else:  # create an empty dataframe with the same columns
        results_df = pd.DataFrame(
            columns=[
                "mean_elevation_diff",
                "mean_uncertainty_unscaled",
                "area",
                "effective_samples",
                "uncertainty",
            ]
        )

    return results_df


def plot_error_vs_predictors_1D(data, statistics="nmad", path=None, axes=None):
    """
    Plot error dependence on predictors (1D).

    Args:
        data: UncertaintyData container
        statistics: Statistic to plot
        path: Path to save plot
        axes: Axes to plot on

    Returns:
        tuple: (fig, axes)
    """

    logger.info("Plotting error vs predictors (1D)...")
    if data.df_binned is None:
        raise ValueError(
            "Binned data not available. Call analyze_heteroscedasticity first."
        )

    n_pred = len(data.predictors)

    # Create a figure with 1-D analysis for each predictor
    if axes is None:
        fig, axes = plt.subplots(1, n_pred, figsize=(n_pred * 4, 6))
    else:
        fig = None
        if len(axes) != n_pred:
            raise ValueError("Number of axes does not match number of predictors.")

    axes = [axes] if n_pred == 1 else axes.flatten()

    # Loop through each predictor and plot
    for i, key in enumerate(data.predictors.keys()):
        # Plot the binned data
        xdem.spatialstats.plot_1d_binning(
            data.df_binned,
            var_name=key,
            statistic_name=statistics,
            label_var=key,
            label_statistic=f"{statistics} of dh (m)",
            ax=axes[i],
        )
    plt.tight_layout()

    # If path is provided, save the figure
    if path is not None and fig is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    logger.info("1D error vs predictors plot created.")
    return fig, axes


def plot_error_vs_predictors_nD(data, statistics="nmad", path=None, axes=None):
    """
    Plot error dependence on predictors (nD).

    Args:
        data: UncertaintyData container
        statistics: Statistic to plot
        path: Path to save plot
        axes: Axes to plot on

    Returns:
        tuple: (fig, axes)
    """

    logger.info("Plotting error vs predictors (nD)...")
    if data.df_binned is None:
        raise ValueError(
            "Binned data not available. Call analyze_heteroscedasticity first."
        )

    n_pred = len(data.predictors)
    if n_pred < 2:
        raise ValueError("At least two predictors are required for nD analysis.")

    # Create a figure with n-D analysis for each pair of predictors
    if axes is None:
        if n_pred == 2:
            n_plots = 1
        elif n_pred == 3:
            n_plots = 3
        else:
            raise NotImplementedError(
                "plotting nD analysis for more than 3 predictors is not implemented yet."
            )
        fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 4, 6))
    else:
        fig = None
        if len(axes) != len(data.predictors):
            raise ValueError("Number of axes does not match number of predictors.")

    axes = [axes] if isinstance(axes, plt.Axes) else axes.flatten()

    # Create a list with all the combinations of predictors
    pred_pairs = list(combinations(data.predictors.keys(), 2))

    # For each pair of predictors, plot the binned data to show the covariance
    for i, (key1, key2) in enumerate(pred_pairs):
        xdem.spatialstats.plot_2d_binning(
            df=data.df_binned,
            var_name_1=key1,
            var_name_2=key2,
            statistic_name=statistics,
            label_var_name_1=key1,
            label_var_name_2=key2,
            label_statistic=f"{statistics} of dh (m)",
            ax=axes[i],
        )
    plt.tight_layout()

    # If path is provided, save the figure
    if path is not None and fig is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    logger.info("nD error vs predictors plot created.")
    return fig, axes


def plot_error_map(data, vmin=None, vmax=None, path=None, ax=None):
    """
    Plot map of estimated errors.

    Args:
        data: UncertaintyData container
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        path: Path to save plot
        ax: Axes to plot on

    Returns:
        tuple: (fig, ax)
    """
    logger.info("Plotting error map...")
    if data.sigma_dh is None:
        raise ValueError(
            "Error raster not computed. Call analyze_heteroscedasticity first."
        )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    else:
        fig = None

    data.sigma_dh.plot(
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
        cbar_title=r"Elevation error ($1\sigma$, m)",
        ax=ax,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    if path is not None and fig is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    logger.info("Error map plot created.")
    return fig, ax


def plot_variogram(
    data, xscale_range_split=None, list_fit_fun_label=None, path=None, ax=None, **kwargs
):
    """
    Plot the empirical variogram and fitted model.

    Args:
        data: UncertaintyData container
        xscale_range_split: List of x-scale range splits
        list_fit_fun_label: Labels for fitted functions
        path: Path to save plot
        ax: Axes to plot on
        **kwargs: Additional plotting arguments
    """
    from pathlib import Path

    import xdem

    logger.info("Plotting variogram...")
    if data.variogram_data is None:
        raise ValueError(
            "Variogram data not available. Call analyze_spatial_correlation first."
        )
    if data.variogram_function is None:
        raise ValueError(
            "Variogram function not available. Call analyze_spatial_correlation first."
        )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    else:
        fig = None

    if xscale_range_split is None:
        xscale_range_split = [200, 500, 2000, 10000]
    if list_fit_fun_label is None:
        list_fit_fun_label = ["Variogram"]

    # Plot empirical variogram
    xdem.spatialstats.plot_variogram(
        data.variogram_data,
        xscale_range_split=xscale_range_split,
        list_fit_fun=[data.variogram_function],
        list_fit_fun_label=list_fit_fun_label,
        ax=ax,
        **kwargs,
    )
    logger.info("Variogram plot created.")

    if path is not None and fig is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig, ax


class UncertaintyAnalyzer(UncertaintyData):
    """
    Legacy class for backward compatibility.
    Provides the same interface as the original UncertaintyAnalyzer.
    """

    def load_data(self, *args, **kwargs):
        data = load_uncertainty_data(*args, **kwargs)
        # Copy all attributes from data to self
        for attr, value in vars(data).items():
            setattr(self, attr, value)
        return self

    def compute_elevation_difference(self, *args, **kwargs):
        return compute_elevation_difference(self, *args, **kwargs)

    def extract_terrain_features(self, *args, **kwargs):
        return extract_terrain_features(self, *args, **kwargs)

    def analyze_heteroscedasticity(self, *args, **kwargs):
        return analyze_heteroscedasticity(self, *args, **kwargs)

    def analyze_spatial_correlation(self, *args, **kwargs):
        return analyze_spatial_correlation(self, *args, **kwargs)

    def compute_uncertainty_for_area(self, *args, **kwargs):
        return compute_uncertainty_for_area(self, *args, **kwargs)

    def plot_error_vs_predictors_1D(self, *args, **kwargs):
        return plot_error_vs_predictors_1D(self, *args, **kwargs)

    def plot_error_vs_predictors_nD(self, *args, **kwargs):
        return plot_error_vs_predictors_nD(self, *args, **kwargs)

    def plot_error_map(self, *args, **kwargs):
        return plot_error_map(self, *args, **kwargs)

    def plot_variogram(self, *args, **kwargs):
        return plot_variogram(self, *args, **kwargs)


# def analyze_dem_uncertainty(
#     ref_dem_path: str | Path,
#     dem_path: str | Path,
#     stable_mask_path: str | Path,
#     glacier_outlines_path: str | Path,
#     compute_terrain_features: bool = True,
#     attibute_list: list[str] = None,
#     additional_predictors: dict[str, Path | str] = None,
#     subsample_res: int = None,
#     output_dir: str | Path = None,
#     area_vector: gu.Vector = None,
#     column_name: str = "rgi_id",
#     area_name: str | list[str] = None,
#     min_area_fraction: float = 0.05,
#     neff_args: dict | None = None,
#     save_intermediate: bool = False,
# ) -> dict:
#     """Analyze DEM uncertainty and return results.

#     This function analyzes the uncertainty in a DEM by comparing it to a reference DEM
#     over stable terrain and creates an error model that can be applied to non-stable terrain.

#     Args:
#         ref_dem_path: Path to reference DEM.
#         dem_path: Path to DEM to analyze.
#         stable_mask_path: Path to stable terrain mask raster.
#         glacier_outlines_path: Path to glacier outlines vector file.
#         additional_predictors: Dictionary mapping predictor names to file paths. These predictors will be used in the uncertainty analysis.
#         subsample_res: Resolution to subsample to for analysis. If None, original resolution is used.
#         plot_dir: Directory to save plots. If None, plots are not saved.
#         column_name (str, optional): Column name in the vector dataset to use for area identification. Defaults to "rgi_id".
#         area_name (str | list[str], optional): Name/ID(s) for the area(s) to analyze. If provided, only areas matching these IDs will be processed. Otherwise, all areas in the vector will be processed.
#         min_area_fraction (float, optional): Minimum area percentage to consider for uncertainty analysis. Defaults to 0.05 (5% of the area).

#     Returns:
#         Dict with the following keys:
#             - analyzer: The UncertaintyAnalyzer instance
#             - area_result: DataFrame with uncertainty results for each area, if area_vector is provided, otherwise None.
#     """
#     # Initialize analyzer
#     analyzer = UncertaintyAnalyzer()

#     # Load data
#     analyzer.load_data(
#         ref_dem_path=ref_dem_path,
#         dem_path=dem_path,
#         stable_mask_path=stable_mask_path,
#         glacier_outlines_path=glacier_outlines_path,
#         subsample_res=subsample_res,
#         additional_predictors=additional_predictors,
#     )

#     # Compute elevation difference
#     analyzer.compute_elevation_difference(get_statistics=True)

#     # Extract terrain features
#     if compute_terrain_features:
#         if attibute_list is None:
#             attibute_list = ["slope", "maximum_curvature"]
#         analyzer.extract_terrain_features(
#             attribute=attibute_list, compute_abs_maxc=True
#         )

#     # Analyze heteroscedasticity
#     analyzer.analyze_heteroscedasticity()

#     if output_dir is not None:
#         plot_dir = Path(output_dir)
#         plot_dir.mkdir(parents=True, exist_ok=True)
#         analyzer.plot_error_vs_predictors_1D(path=output_dir / "1D binning.png")
#         analyzer.plot_error_vs_predictors_nD(path=output_dir / "nD binning.png")
#         analyzer.plot_error_map(path=output_dir / "predicted_error_map.png")
#         analyzer.sigma_dh.save(output_dir / "predicted_error.tif")

#     if save_intermediate and output_dir is not None:
#         analyzer.to_pickle(output_dir / "analyzer_state_intermediate.pkl")

#     # Analyze spatial correlation
#     analyzer.analyze_spatial_correlation()
#     if output_dir is not None:
#         analyzer.plot_variogram(path=output_dir / "variogram.png")
#         analyzer.variogram_params.to_csv(
#             path_or_buf=output_dir / "variogram_params.csv", index=False, header=True
#         )
#     if save_intermediate and output_dir is not None:
#         analyzer.to_pickle(output_dir / "analyzer_state_intermediate.pkl")

#     # Compute uncertainty for specific area
#     if area_vector is not None:
#         areas_uncertainty = analyzer.compute_uncertainty_for_area(
#             area_vector=area_vector,
#             column_name=column_name,
#             area_name=area_name,
#             min_area_fraction=min_area_fraction,
#             neff_args=neff_args,
#             n_jobs=1,
#         )
#         if output_dir is not None and areas_uncertainty is not None:
#             areas_uncertainty.to_csv(
#                 path_or_buf=output_dir / "areas_uncertainty.csv",
#                 index=True,
#                 header=True,
#             )
#     else:
#         areas_uncertainty = None

#     # Save the analyzer state to a pickle file
#     if output_dir is not None:
#         analyzer.to_pickle(output_dir / "uncertainty_analyzer_state.pkl")
#         if save_intermediate:
#             (output_dir / "analyzer_state_intermediate.pkl").unlink(missing_ok=True)

#     return {
#         "analyzer": analyzer,
#         "area_result": areas_uncertainty,
#     }


# class UncertaintyAnalyzer:
#     """
#     A class to analyze uncertainty in Digital Elevation Models.
#     """

#     def __init__(self):
#         """
#         Initialize the uncertainty analyzer with paths to required data.
#         """
#         # Data containers
#         self.ref_dem: xdem.DEM | None = None
#         self.dem: xdem.DEM | None = None
#         self.dh: xdem.DEM | None = None
#         self.stable_mask: gu.Mask | None = None
#         self.glacier_outlines: gu.Vector | None = None
#         self.glacier_mask: gu.Mask | None = None

#         # Heteroscedasticity analysis
#         self.predictors = {}
#         self.df_binned = None
#         self.zscores = None
#         self.dh_err_fun = None

#         # Spatial correlation analysis
#         self.sigma_dh = None
#         self.variogram_model = None

#     def to_pickle(self, path: Path | str) -> bool:
#         """
#         Save the current state of the analyzer to a pickle file.

#         Args:
#             path (Path | str): Path to save the pickle file.

#         Returns:
#             bool: True if saved successfully, False otherwise.
#         """
#         path = Path(path)
#         path.parent.mkdir(parents=True, exist_ok=True)
#         try:
#             with open(path, "wb") as f:
#                 cloudpickle.dump(self, f)
#             return True
#         except Exception as e:
#             logger.error(f"Failed to save to {path}: {e}")
#             return False

#     @classmethod
#     def from_pickle(cls, path: Path | str) -> "UncertaintyAnalyzer":
#         """
#         Load the analyzer state from a pickle file.

#         Args:
#             path (Path | str): Path to the pickle file.

#         Returns:
#             UncertaintyAnalyzer: Loaded instance of the analyzer.
#         """
#         path = Path(path)
#         if not path.exists():
#             raise FileNotFoundError(f"File {path} does not exist.")

#         try:
#             with open(path, "rb") as f:
#                 return cloudpickle.load(f)
#         except Exception as e:
#             logger.error(f"Failed to load from {path}: {e}")
#             raise

#     def load_data(
#         self,
#         ref_dem_path: str | Path,
#         dem_path: str | Path,
#         stable_mask_path: str | Path = None,
#         glacier_outlines_path: str | Path = None,
#         subsample_res: int = None,
#         no_data_value: float = -9999,
#         additional_predictors: dict[str, Path | str] = None,
#     ) -> "UncertaintyAnalyzer":
#         """
#         Load and prepare all necessary data.

#         Args:
#             ref_dem_path (str | Path): Path to reference DEM.
#             dem_path (str | Path): Path to DEM to analyze.
#             stable_mask_path (str | Path, optional): Path to stable terrain mask.
#             glacier_outlines_path (str | Path, optional): Path to glacier outlines vector file.
#             subsample_res (int, optional): Resolution to subsample to for analysis.
#             no_data_value (float, optional): No data value for the DEMs.
#             additional_predictors (dict[str, Path | str], optional): Additional predictors.

#         Returns:
#             UncertaintyAnalyzer: The instance of the analyzer.
#         """
#         logger.info("Loading data...")
#         self.ref_dem_path = Path(ref_dem_path)
#         self.dem_path = Path(dem_path)
#         self.stable_mask_path = Path(stable_mask_path) if stable_mask_path else None
#         self.glacier_outlines_path = (
#             Path(glacier_outlines_path) if glacier_outlines_path else None
#         )
#         self.subsample_res = subsample_res
#         self.no_data_value = no_data_value

#         # Load reference DEM
#         self.ref_dem = xdem.DEM(self.ref_dem_path)
#         self.ref_dem.set_area_or_point("Area")
#         self.ref_dem.set_nodata(self.no_data_value)

#         # Load DEM to analyze
#         self.dem = xdem.DEM(self.dem_path)
#         self.dem.set_area_or_point("Area")
#         self.dem.set_nodata(self.no_data_value)

#         # Subsample if resolution specified
#         if self.subsample_res:
#             self.dem.reproject(
#                 res=self.subsample_res, resampling="bilinear", inplace=True
#             )

#         # Reproject the reference DEM to the DEM
#         self.ref_dem.reproject(self.dem, resampling="bilinear", inplace=True)

#         # Load glacier outlines if provided
#         if self.glacier_outlines_path:
#             self.glacier_outlines = gu.Vector(self.glacier_outlines_path).crop(self.dem)
#             self.glacier_mask = self.glacier_outlines.create_mask(self.dem)

#         # Load stable area mask if provided
#         if self.stable_mask_path:
#             stable_mask_raster = gu.Raster(self.stable_mask_path)
#             stable_mask_raster.set_nodata(255)
#             stable_mask_raster.reproject(self.dem, resampling="nearest", inplace=True)
#             self.stable_mask = stable_mask_raster == 1

#         # Load other data if provided (e.g., correlation value map)
#         if additional_predictors:
#             for key, value in additional_predictors.items():
#                 self.predictors[key] = gu.Raster(value).reproject(
#                     self.dem, resampling="bilinear"
#                 )

#         logger.info("Data loaded successfully.")
#         return self

#     def compute_elevation_difference(
#         self, get_statistics: bool = True
#     ) -> tuple[xdem.DEM, dict]:
#         """
#         Compute the elevation difference between reference and analyzed DEM.
#         Optionally subsample the difference for more efficient analysis.

#         Args:
#             get_statistics (bool, optional): Whether to get statistics of the elevation difference.

#         Returns:
#             xdem.DEM: The elevation difference DEM.
#             dict: Statistics of the elevation difference. None if get_statistics is False.
#         """
#         logger.info("Computing elevation difference...")
#         self.dh = self.ref_dem - self.dem

#         if get_statistics:
#             stats = self.dh.get_stats(inlier_mask=self.stable_mask)
#             logger.info("Elevation difference statistics:")
#             for key, value in stats.items():
#                 logger.info(f"\t{key}: {value:.2f}")
#         else:
#             stats = None

#         logger.info("Elevation difference computed.")
#         return self.dh, stats

#     def extract_terrain_features(
#         self,
#         attribute: list[str] = None,
#         compute_abs_maxc: bool = True,
#     ) -> "UncertaintyAnalyzer":
#         """
#         Extract terrain features to be used as predictors for uncertainty.

#         Args:
#             attribute (list[str], optional): List of terrain attributes to extract.
#             compute_abs_maxc (bool, optional): Whether to compute absolute value of the maximum curvature.

#         Returns:
#             UncertaintyAnalyzer: The instance of the analyzer.
#         """
#         logger.info(f"Extracting terrain features: {attribute}...")
#         if attribute is None:
#             attribute = ["slope", "maximum_curvature"]

#         # Compute slope and maximum curvature
#         outs = xdem.terrain.get_terrain_attribute(self.ref_dem, attribute=attribute)

#         # Stores the terrain features in the predictors dictionary
#         for key, data in zip(attribute, outs, strict=False):
#             # If requested, compute absolute value of the maximum curvature
#             if key == "maximum_curvature" and compute_abs_maxc:
#                 logger.debug("Computing absolute values of the maximum curvature")
#                 data = np.abs(data)
#             self.predictors[key] = data

#         logger.info(f"Terrain features extracted successfully: {attribute}")
#         return self

#     def prepare_stable_terrain_data(
#         self, nmad_factor: int = 5
#     ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
#         """
#         Extract data from stable terrain and filter outliers.

#         Args:
#             nmad_factor (int, optional): Factor to multiply NMAD for outlier filtering.

#         Returns:
#             tuple: Tuple containing the elevation difference array and dictionary of predictor arrays.
#         """
#         logger.info("Preparing stable terrain data...")
#         if self.stable_mask is None:
#             raise ValueError("Stable mask not loaded. Call load_data first.")
#         if self.dh is None:
#             raise ValueError(
#                 "Elevation difference not computed. Call compute_elevation_difference first."
#             )
#         if not self.predictors:
#             raise ValueError(
#                 "Predictors not computed. Call extract_terrain_features first."
#             )

#         # Apply mask to the elevation difference
#         dh_arr = self.dh[self.stable_mask].filled(np.nan)

#         # Filter outliers in the elevation difference using NMAD
#         arr_lim = nmad_factor * nmad(dh_arr)
#         logger.debug(f"Filtering outliers with limit: {arr_lim}")
#         outlier_mask = np.abs(dh_arr) > arr_lim
#         dh_arr[outlier_mask] = np.nan

#         # Extract arrays for stable terrain for each predictor
#         predictors_arrays = {}
#         for key, data in self.predictors.items():
#             # Apply masks to the data (set masked values to NaN)
#             arr = data[self.stable_mask].filled(np.nan)
#             arr[outlier_mask] = np.nan
#             predictors_arrays[key] = arr

#         logger.info("Stable terrain data prepared.")
#         return dh_arr, predictors_arrays

#     def analyze_heteroscedasticity(
#         self,
#         nmad_factor: int = 5,
#         list_var_bins: list[float] = None,
#         list_ranges: list[float] = None,
#         statistics: list[str] = None,
#         nd_binning_statistic: str = "nmad",
#         nd_binning_min_count: int = 100,
#     ) -> tuple:
#         """
#         Analyze error heteroscedasticity and create error function.

#         Args:
#             nmad_factor (int, optional): Factor for outlier removal.
#             statistics (list[str], optional): List of statistics to compute.
#             list_var_bins (list[float], optional): List of bin limits for each predictor.
#             list_ranges (list[float], optional): List of ranges for each predictor.

#         Returns:
#             tuple: Tuple containing the binned data, z-scores, error function, and sigma_dh.
#         """
#         logger.info("Analyzing heteroscedasticity...")
#         # Get stable terrain data
#         dh_arr, stable_data = self.prepare_stable_terrain_data(nmad_factor=nmad_factor)

#         # Prepare data for n-d binning
#         var_names = list(stable_data.keys())
#         predictor_list = [stable_data[key] for key in var_names]

#         # Compute the bin limits for each predictor
#         if list_var_bins is None:
#             list_var_bins = [
#                 np.linspace(
#                     np.floor(np.nanpercentile(d, 0.5)),
#                     np.ceil(np.nanpercentile(d, 99.5)),
#                     10,
#                 )
#                 for d in predictor_list
#             ]

#         # Set the statistics to compute
#         if statistics is None:
#             statistics = ["count", np.nanmedian, xdem.spatialstats.nmad]

#         # Compute n-d binning with all the predictors
#         logger.info("Computing n-d binning of the predictors...")
#         df = xdem.spatialstats.nd_binning(
#             values=dh_arr,
#             list_var=predictor_list,
#             list_var_names=var_names,
#             statistics=statistics,
#             list_var_bins=list_var_bins,
#             list_ranges=list_ranges,
#         )

#         # Fit error model
#         logger.info("Fitting error model on the predictors...")
#         unscaled_dh_err_fun = xdem.spatialstats.interp_nd_binning(
#             df,
#             list_var_names=var_names,
#             statistic=nd_binning_statistic,
#             min_count=nd_binning_min_count,
#         )

#         # Compute the mean predicted elevation error on the stable terrain
#         dh_err_stable = unscaled_dh_err_fun(
#             tuple([stable_data[key] for key in var_names])
#         )
#         logger.info(
#             f"The spread of elevation difference is {xdem.spatialstats.nmad(dh_arr):.2f} compared to a mean predicted elevation error of {np.nanmean(dh_err_stable):.2f}."
#         )

#         # Two-step standardization
#         logger.info("Standardizing the predicted errors...")
#         zscores, dh_err_fun = xdem.spatialstats.two_step_standardization(
#             dvalues=dh_arr,
#             list_var=predictor_list,
#             unscaled_error_fun=unscaled_dh_err_fun,
#         )

#         # Compute the error function for the whole DEM
#         logger.info("Computing the error function for the whole DEM...")
#         sigma_dh = self.dh.copy(
#             new_array=dh_err_fun(
#                 tuple([self.predictors[key].data for key in var_names])
#             )
#         )
#         sigma_dh.set_mask(self.dem.data.mask)

#         # Save the results
#         self.df_binned = df
#         self.zscores = zscores
#         self.dh_err_fun = dh_err_fun
#         self.sigma_dh = sigma_dh

#         logger.info("Heteroscedasticity analysis completed.")
#         return df, zscores, dh_err_fun, sigma_dh

#     def plot_error_vs_predictors_1D(
#         self,
#         statistics: str = "nmad",
#         path: Path = None,
#         axes: plt.Axes = None,
#     ) -> tuple[plt.Figure, plt.Axes]:
#         """
#         Plot error dependence on predictors.

#         Args:
#             statistics (str, optional): Statistic to plot.
#             path (Path, optional): Path to save the plot.
#             axes (plt.Axes, optional): Axes to plot on.

#         Returns:
#             tuple: Tuple containing the figure and axes.
#         """
#         logger.info("Plotting error vs predictors (1D)...")
#         if self.df_binned is None:
#             raise ValueError(
#                 "Binned data not available. Call analyze_heteroscedasticity first."
#             )

#         n_pred = len(self.predictors)

#         # Create a figure with 1-D analysis for each predictor
#         if axes is None:
#             fig, axes = plt.subplots(1, n_pred, figsize=(n_pred * 4, 6))
#         else:
#             fig = None
#             if len(axes) != n_pred:
#                 raise ValueError("Number of axes does not match number of predictors.")

#         axes = [axes] if len(axes) == 1 else axes.flatten()

#         # Loop through each predictor and plot
#         for i, key in enumerate(self.predictors.keys()):
#             # Plot the binned data
#             xdem.spatialstats.plot_1d_binning(
#                 self.df_binned,
#                 var_name=key,
#                 statistic_name=statistics,
#                 label_var=key,
#                 label_statistic=f"{statistics} of dh (m)",
#                 ax=axes[i],
#             )
#         plt.tight_layout()

#         # If path is provided, save the figure
#         if path is not None and fig is not None:
#             path = Path(path)
#             path.parent.mkdir(parents=True, exist_ok=True)
#             fig.savefig(path, dpi=300, bbox_inches="tight")

#         logger.info("1D error vs predictors plot created.")
#         return fig, axes

#     def plot_error_vs_predictors_nD(
#         self,
#         statistics: str = "nmad",
#         path: Path = None,
#         axes: plt.Axes = None,
#     ) -> tuple[plt.Figure, plt.Axes]:
#         """
#         Plot error dependence on predictors.

#         Args:
#             statistics (str, optional): Statistic to plot.
#             path (Path, optional): Path to save the plot.
#             axes (plt.Axes, optional): Axes to plot on.

#         Returns:
#             tuple: Tuple containing the figure and axes.
#         """
#         logger.info("Plotting error vs predictors (nD)...")
#         from itertools import combinations

#         if self.df_binned is None:
#             raise ValueError(
#                 "Binned data not available. Call analyze_heteroscedasticity first."
#             )

#         n_pred = len(self.predictors)
#         if n_pred < 2:
#             raise ValueError("At least two predictors are required for nD analysis.")

#         # Create a figure with n-D analysis for each pair of predictors
#         if axes is None:
#             if len(self.predictors) == 2:
#                 n_plots = 1
#             elif len(self.predictors) == 3:
#                 n_plots = 3
#             else:
#                 raise NotImplementedError(
#                     "plotting nD analysis for more than 3 predictors is not implemented yet."
#                 )
#             fig, axes = plt.subplots(
#                 1,
#                 n_plots,
#                 figsize=(n_plots * 4, 6),
#             )
#         else:
#             fig = None
#             if len(axes) != len(self.predictors):
#                 raise ValueError("Number of axes does not match number of predictors.")

#         axes = [axes] if len(axes) == 1 else axes.flatten()

#         # Create a list with all the combinations of predictors
#         pred_pairs = list(combinations(self.predictors.keys(), 2))

#         # For each pair of predictors, plot the binned data to show the covariance
#         for i, (key1, key2) in enumerate(pred_pairs):
#             xdem.spatialstats.plot_2d_binning(
#                 df=self.df_binned,
#                 var_name_1=key1,
#                 var_name_2=key2,
#                 statistic_name=statistics,
#                 label_var_name_1=key1,
#                 label_var_name_2=key2,
#                 label_statistic=f"{statistics} of dh (m)",
#                 ax=axes[i],
#             )
#         plt.tight_layout()

#         # If path is provided, save the figure
#         if path is not None and fig is not None:
#             path = Path(path)
#             path.parent.mkdir(parents=True, exist_ok=True)
#             fig.savefig(path, dpi=300, bbox_inches="tight")

#         logger.info("nD error vs predictors plot created.")
#         return fig, axes

#     def plot_error_map(
#         self,
#         vmin: float = None,
#         vmax: float = None,
#         path: Path = None,
#         ax: plt.Axes = None,
#     ) -> tuple[plt.Figure, plt.Axes]:
#         """
#         Plot map of estimated errors.

#         Args:
#             vmin (float, optional): Minimum value for color scale.
#             vmax (float, optional): Maximum value for color scale.
#             path (Path, optional): Path to save the plot.
#             ax (plt.Axes, optional): Axes to plot on.

#         Returns:
#             tuple: Tuple containing the figure and axes.
#         """
#         logger.info("Plotting error map...")
#         if self.sigma_dh is None:
#             raise ValueError(
#                 "Error raster not computed. Call analyze_heteroscedasticity first."
#             )
#         if ax is None:
#             fig, ax = plt.subplots(1, 1, figsize=(12, 6))
#         else:
#             fig = None
#         self.sigma_dh.plot(
#             cmap="Reds",
#             vmin=vmin,
#             vmax=vmax,
#             cbar_title=r"Elevation error ($1\sigma$, m)",
#             ax=ax,
#         )
#         ax.set_xticks([])
#         ax.set_yticks([])

#         if path is not None and fig is not None:
#             path = Path(path)
#             path.parent.mkdir(parents=True, exist_ok=True)
#             fig.savefig(path, dpi=300, bbox_inches="tight")

#         logger.info("Error map plot created.")
#         return fig, ax

#     def analyze_spatial_correlation(
#         self,
#         standardize: bool = True,
#         n_samples: int = 1000,
#         subsample_method: str = "cdist_equidistant",
#         n_variograms: int = 3,
#         estimator: str = "dowd",
#         random_state: int = None,
#         list_models: list[str] = None,
#         sample_kwargs: dict = None,
#         fit_kwargs: dict = None,
#     ) -> tuple:
#         """
#         Analyze spatial correlation of standardized errors.

#         Args:
#             standardize (bool, optional): Whether to standardize the elevation difference.
#             n_samples (int, optional): Number of samples for variogram computation.
#             subsample_method (str, optional): Method for subsampling.
#             n_variograms (int, optional): Number of variogram realizations.
#             estimator (str, optional): Estimator type for variogram calculation.
#             random_state (int, optional): Random seed.
#             list_models (list[str], optional): List of models for variogram fitting.
#             sample_kwargs (dict, optional): Additional arguments for sampling.
#             fit_kwargs (dict, optional): Additional arguments for fitting.

#         Returns:
#             tuple: Tuple containing the variogram function and parameters.
#         """
#         logger.info("Analyzing spatial correlation...")
#         if self.sigma_dh is None:
#             raise ValueError(
#                 "Error raster not computed. Call analyze_heteroscedasticity first."
#             )

#         # Compute standardized elevation difference if requested, otherwise use original
#         z_dh = self.dh / self.sigma_dh if standardize else self.dh.copy()

#         # Remove values on unstable terrain and large outliers
#         z_dh.data[~self.stable_mask.data] = np.nan
#         z_dh.data[np.abs(z_dh.data) > 4] = np.nan

#         # Sample empirical variogram
#         df_vgm = xdem.spatialstats.sample_empirical_variogram(
#             values=z_dh,
#             subsample=n_samples,
#             subsample_method=subsample_method,
#             n_variograms=n_variograms,
#             estimator=estimator,
#             random_state=random_state,
#             n_jobs=n_variograms,
#             **(sample_kwargs or {}),
#         )

#         # Fit variogram model
#         if list_models is None:
#             list_models = ["Spherical", "Spherical"]
#         func_sum_vgm, params_vgm = xdem.spatialstats.fit_sum_model_variogram(
#             list_models,
#             empirical_variogram=df_vgm,
#             **(fit_kwargs or {}),
#         )

#         self.variogram_data = df_vgm
#         self.variogram_function = func_sum_vgm
#         self.variogram_params = params_vgm

#         logger.info("Spatial correlation analysis completed.")
#         return func_sum_vgm, params_vgm

#     def plot_variogram(
#         self,
#         xscale_range_split: list[float] = None,
#         list_fit_fun_label: list[str] = None,
#         path: Path = None,
#         ax: plt.Axes = None,
#         **kwargs,
#     ) -> None:
#         """
#         Plot the empirical variogram and fitted model.

#         Args:
#             xscale_range_split (list[float], optional): List of x-scale range splits.
#             list_fit_fun_label (list[str], optional): List of labels for fitted functions.
#             path (Path, optional): Path to save the plot.
#             ax (plt.Axes, optional): Axes to plot on.
#             **kwargs: Additional arguments for plotting.
#         """
#         logger.info("Plotting variogram...")
#         if self.variogram_data is None:
#             raise ValueError(
#                 "Variogram data not available. Call analyze_spatial_correlation first."
#             )
#         if self.variogram_function is None:
#             raise ValueError(
#                 "Variogram function not available. Call analyze_spatial_correlation first."
#             )

#         if ax is None:
#             fig, ax = plt.subplots(1, 1, figsize=(12, 6))
#         else:
#             fig = None
#         if xscale_range_split is None:
#             xscale_range_split = [200, 500, 2000, 10000]
#         if list_fit_fun_label is None:
#             list_fit_fun_label = ["Variogram"]

#         # Plot empirical variogram
#         xdem.spatialstats.plot_variogram(
#             self.variogram_data,
#             xscale_range_split=xscale_range_split,
#             list_fit_fun=[self.variogram_function],
#             list_fit_fun_label=list_fit_fun_label,
#             ax=ax,
#             **kwargs,
#         )
#         logger.info("Variogram plot created.")

#         if path is not None and fig is not None:
#             path = Path(path)
#             path.parent.mkdir(parents=True, exist_ok=True)
#             fig.savefig(path, dpi=300, bbox_inches="tight")

#     def compute_uncertainty_for_area(
#         self,
#         area_vector: gu.Vector | Path | None = None,
#         column_name: str = "rgi_id",
#         area_name: str | list[str] | None = None,
#         min_area_fraction: float = 0.05,
#         neff_args: dict | None = None,
#         n_jobs: int = 1,
#     ) -> pd.DataFrame:
#         """Compute uncertainty for specific area(s) (e.g., glaciers).

#         This function computes the uncertainty for one or multiple areas. It calculates
#         the mean elevation difference, the mean uncertainty, and the effective number
#         of samples for each area, accounting for spatial correlation.

#         Args:
#             area_vector (gu.Vector, optional): Vector defining the area(s) of interest. If not provided, the glacier outlines stored in the class will be used.
#             column_name (str, optional): Column name in the vector dataset to use for area identification. Defaults to "rgi_id".
#             area_name (str | list[str], optional): Name/ID(s) for the area(s) to analyze. If provided, only areas matching these IDs will be processed. Otherwise, all areas in the vector will be processed.
#             min_area_fraction (float, optional): Minimum area percentage to consider for uncertainty analysis. Defaults to 0.05 (5% of the area).
#             neff_args (dict | None, optional): Additional arguments for the effective number of samples calculation to pass to the number_effective_samples() function. Check xdem documentation for more details. If None, default values are used.
#             n_jobs (int, optional): Number of parallel jobs to use for processing. If 1, processing is done sequentially. Defaults to 1.

#         Returns:
#             pd.DataFrame: DataFrame containing the results for each area, including mean elevation difference, mean uncertainty, and effective number of samples.

#         Raises:
#             ValueError: If spatial correlation has not been analyzed yet.
#             ValueError: If neither area_vector nor stored glacier outlines are available.
#             ValueError: If the specified column_name does not exist in the vector dataset.
#             ValueError: If area_name is provided but not found in the vector dataset.
#         """
#         logger.info("Computing uncertainty for area(s)...")
#         if not hasattr(self, "variogram_params"):
#             raise ValueError(
#                 "Spatial correlation not analyzed. Call analyze_spatial_correlation first."
#             )

#         # Determine which vector to use (input or stored glacier outlines)
#         if area_vector is None:
#             if self.glacier_outlines is None:
#                 raise ValueError(
#                     "No area vector provided and glacier outlines not loaded"
#                 )
#             work_vector = self.glacier_outlines.copy()
#             logger.debug("Using stored glacier outlines")
#         else:
#             if isinstance(area_vector, str | Path):
#                 try:
#                     area_vector = gu.Vector(area_vector)
#                 except Exception as e:
#                     raise ValueError(
#                         f"Failed to load area vector from {area_vector}: {e}"
#                     ) from e
#             elif not isinstance(area_vector, gu.Vector):
#                 raise ValueError(
#                     "area_vector must be a geoutils.Vector or a path to a vector file."
#                 )
#             work_vector = area_vector.copy()
#             logger.debug("Using provided area vector")

#         # Check if column_name exists in the vector
#         if column_name not in work_vector.ds.columns:
#             available_columns = work_vector.ds.columns.tolist()
#             raise ValueError(
#                 f"Column '{column_name}' not found in vector dataset. "
#                 f"Available columns: {available_columns}"
#             )

#         # If area_name is provided, filter the vector to include only matching areas
#         if area_name is not None:
#             # Filter vector by area name(s)
#             if isinstance(area_name, str):
#                 area_name = [area_name]
#             filtered_vector = work_vector[work_vector.ds[column_name].isin(area_name)]

#             if filtered_vector.ds.empty:
#                 available_ids = work_vector.ds[column_name].unique().tolist()
#                 raise ValueError(
#                     f"Area name(s) {area_name} not found in column '{column_name}'. "
#                     f"Available IDs: {available_ids[:5]}{'...' if len(available_ids) > 5 else ''}"
#                 )

#             work_vector = filtered_vector
#             logger.debug(
#                 f"Filtered vector to {len(work_vector.ds)} geometries matching {area_name}"
#             )

#         # Ensure vector is in a projected CRS
#         if work_vector.crs.is_geographic:
#             logger.debug("Converting vector from geographic to projected CRS")
#             work_vector = work_vector.to_crs(crs=work_vector.ds.estimate_utm_crs())

#         # Process each geometry in the vector in parallel
#         # Prepare tasks for parallel processing with all required parameters
#         tasks = []
#         for idx, row in work_vector.ds.iterrows():
#             params = {
#                 "idx_row_tuple": (idx, row),
#                 "work_vector": work_vector,
#                 "column_name": column_name,
#                 "ref_dem": self.ref_dem,
#                 "dh": self.dh,
#                 "sigma_dh": self.sigma_dh,
#                 "variogram_params": self.variogram_params,
#                 "min_area_fraction": min_area_fraction,
#                 "neff_args": neff_args or {},
#                 "dem_path_name": str(self.dem_path.name),
#             }
#             tasks.append(params)

#         try:
#             logger.info("Processing areas in parallel...")
#             processed_results = Parallel(n_jobs=n_jobs, verbose=10)(
#                 delayed(_process_single_area)(task) for task in tasks
#             )
#         except MemoryError as e:
#             logger.error(f"Memory error during parallel processing: {e}")
#             logger.info(
#                 "Switching to single-threaded processing due to memory constraints."
#             )
#             processed_results = []
#             for task in tqdm(tasks):
#                 # Force a lower resolution for rasterization
#                 task["neff_args"] = {"rasterize_resolution": 50}
#                 processed_results.append(_process_single_area(task))

#         # Filter out None results (skipped areas) and create a dataframe
#         results = {}
#         for res_tuple in processed_results:
#             if res_tuple is not None:
#                 area_id, data = res_tuple
#                 results[area_id] = data
#         results_df = pd.DataFrame.from_dict(results, orient="index")

#         # Ensure correct dtypes if dataframe is not empty but some columns might be all NaN
#         if not results_df.empty:
#             expected_columns = {
#                 "mean_elevation_diff": float,
#                 "mean_uncertainty_unscaled": float,
#                 "area": float,
#                 "effective_samples": float,
#                 "uncertainty": float,
#             }
#             for col, dtype in expected_columns.items():
#                 if col in results_df.columns:
#                     results_df[col] = results_df[col].astype(dtype)
#                 else:  # if a column is missing because all areas were skipped before its calculation
#                     results_df[col] = pd.Series(dtype=dtype)
#         else:  # create an empty dataframe with the same columns
#             results_df = pd.DataFrame(
#                 columns=[
#                     "mean_elevation_diff",
#                     "mean_uncertainty_unscaled",
#                     "area",
#                     "effective_samples",
#                     "uncertainty",
#                 ]
#             )

#         return results_df

#     def plot_area_result(self, area_result, figsize=(12, 6), vmin=-30, vmax=30):
#         """
#         Plot elevation difference with uncertainty for a specific area.

#         Parameters
#         ----------
#         area_result : dict
#             Result from compute_uncertainty_for_area
#         figsize : tuple, optional
#             Figure size
#         vmin, vmax : float, optional
#             Limits for color scale

#         Returns
#         -------
#         matplotlib.figure.Figure
#             Figure with plot
#         """
#         fig, ax = plt.subplots(1, 1, figsize=figsize)
#         self.dh.plot(
#             cmap="RdYlBu",
#             cbar_title="Elevation differences (m)",
#             ax=ax,
#             vmin=vmin,
#             vmax=vmax,
#         )
#         area_result["area"].plot(self.dh, fc="none", ec="black", lw=2)

#         # Add text with results
#         plt.text(
#             area_result["area"].ds.centroid.x.values[0] - 10000,
#             area_result["area"].ds.centroid.y.values[0] - 10000,
#             f"{area_result['mean_elevation_diff']:.2f} \n$\\pm$ {area_result['standard_error']:.2f} m",
#             color="black",
#             fontweight="bold",
#             va="top",
#             ha="center",
#         )

#         return fig

#     def create_error_mask(self, max_percent=0.90, on_glacier_only=False):
#         """
#         Create a mask for areas with acceptable error levels.

#         Parameters
#         ----------
#         max_percent : float, optional
#             Percentile threshold for error filtering
#         on_glacier_only : bool, optional
#             Whether to compute percentile only on glacier areas

#         Returns
#         -------
#         gu.Mask
#             Mask of areas with acceptable error
#         """
#         if self.sigma_dh is None:
#             self.compute_error_raster()

#         if on_glacier_only:
#             if self.glacier_mask is None:
#                 raise ValueError(
#                     "Glacier mask not available, can't filter on glacier areas only"
#                 )

#             # Compute percentile on glacier areas only
#             max_error = np.nanquantile(
#                 self.sigma_dh[self.glacier_mask].compressed(), max_percent
#             )
#             valid = self.sigma_dh.data < max_error
#             error_mask = gu.Mask.from_array(
#                 valid, self.sigma_dh.transform, self.sigma_dh.crs
#             )
#             error_mask.set_mask(~self.glacier_mask)
#         else:
#             # Compute percentile on all areas
#             max_error = np.nanquantile(self.sigma_dh.data.compressed(), max_percent)
#             error_mask = self.sigma_dh < max_error

#         return error_mask


if __name__ == "__main__":
    data_dir = Path.cwd() / "data"
    stereo_dir = Path.cwd() / "outputs/proc/009_003-009_S5_054-256-0_2003-11-15"

    reference_dem_path = data_dir / "swissalti3d_aletsch_32632_hell_10m.tif"
    mask_path = data_dir / "stable_mask_32632_10m.tif"
    rgi_path = data_dir / "RGI2000-v7.0-G-11_central_europe.geojson"
    dem_path = (
        stereo_dir / "opals/stereo-DEM_transLSM_robMovingPlanes_10m_filled_adaptive.tif"
    )
    ncc_map_path = stereo_dir / "stereo-ncc.tif"

    additional_predictors = {"ncc": ncc_map_path}

    results = analyze_dem_uncertainty(
        ref_dem_path=reference_dem_path,
        dem_path=dem_path,
        stable_mask_path=mask_path,
        glacier_outlines_path=rgi_path,
        additional_predictors=additional_predictors,
        subsample_res=50,  # Subsample to 30m for faster analysis
        area_name="RGI2000-v7.0-G-11-02596",  # Aletsch glacier
        plot_dir=Path.cwd() / "figs",  # Directory to save plots
    )

    print("Done.")

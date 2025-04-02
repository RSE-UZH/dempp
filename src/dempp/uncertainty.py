import logging
import pickle
from pathlib import Path

import geoutils as gu
import numpy as np
import pandas as pd
import xdem
from matplotlib import pyplot as plt
from xdem.spatialstats import nmad

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


def analyze_dem_uncertainty(
    ref_dem_path: str | Path,
    dem_path: str | Path,
    stable_mask_path: str | Path,
    glacier_outlines_path: str | Path,
    additional_predictors: dict[str, Path | str] = None,
    subsample_res: int = None,
    output_dir: str | Path = None,
    area_vector: gu.Vector = None,
    column_name: str = "rgi_id",
    area_name: str | list[str] = None,
    min_area_fraction: float = 0.05,
) -> dict:
    """Analyze DEM uncertainty and return results.

    This function analyzes the uncertainty in a DEM by comparing it to a reference DEM
    over stable terrain and creates an error model that can be applied to non-stable terrain.

    Args:
        ref_dem_path: Path to reference DEM.
        dem_path: Path to DEM to analyze.
        stable_mask_path: Path to stable terrain mask raster.
        glacier_outlines_path: Path to glacier outlines vector file.
        additional_predictors: Dictionary mapping predictor names to file paths. These predictors will be used in the uncertainty analysis.
        subsample_res: Resolution to subsample to for analysis. If None, original resolution is used.
        plot_dir: Directory to save plots. If None, plots are not saved.
        column_name (str, optional): Column name in the vector dataset to use for area identification. Defaults to "rgi_id".
        area_name (str | list[str], optional): Name/ID(s) for the area(s) to analyze. If provided, only areas matching these IDs will be processed. Otherwise, all areas in the vector will be processed.
        min_area_fraction (float, optional): Minimum area percentage to consider for uncertainty analysis. Defaults to 0.05 (5% of the area).

    Returns:
        Dict with the following keys:
            - analyzer: The DEMUncertaintyAnalyzer instance
            - dh: The elevation difference raster
            - sigma_dh: The estimated uncertainty raster
            - area_result: Results for specific area (if area_name was provided)
    """
    # Initialize analyzer
    analyzer = DEMUncertaintyAnalyzer()

    # Load data
    analyzer.load_data(
        ref_dem_path=ref_dem_path,
        dem_path=dem_path,
        stable_mask_path=stable_mask_path,
        glacier_outlines_path=glacier_outlines_path,
        subsample_res=subsample_res,
        additional_predictors=additional_predictors,
    )

    # Compute elevation difference
    analyzer.compute_elevation_difference(get_statistics=True)

    # Extract terrain features
    analyzer.extract_terrain_features(
        attribute=["slope", "maximum_curvature"], compute_abs_maxc=True
    )

    # Analyze heteroscedasticity
    analyzer.analyze_heteroscedasticity()

    if output_dir is not None:
        plot_dir = Path(output_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        analyzer.plot_error_vs_predictors_1D(path=output_dir / "1D binning.png")
        analyzer.plot_error_vs_predictors_nD(path=output_dir / "nD binning.png")
        analyzer.plot_error_map(path=output_dir / "predicted_error_map.png")
        analyzer.sigma_dh.save(output_dir / "predicted_error.tif")

    # Analyze spatial correlation
    analyzer.analyze_spatial_correlation()
    if output_dir is not None:
        analyzer.plot_variogram(path=output_dir / "variogram.png")
        analyzer.variogram_params.to_csv(
            path_or_buf=output_dir / "variogram_params.csv", index=False, header=True
        )

    # Compute uncertainty for specific area
    areas_uncertainty = analyzer.compute_uncertainty_for_area(
        area_vector=area_vector,
        column_name=column_name,
        area_name=area_name,
        min_area_fraction=min_area_fraction,
    )
    if output_dir is not None and areas_uncertainty is not None:
        areas_uncertainty.to_csv(
            path_or_buf=output_dir / "areas_uncertainty.csv", index=False, header=True
        )

    # Save the analyzer state to a pickle file
    if output_dir is not None:
        analyzer.to_pickle(output_dir / "analyzer_state.pkl")

    return {
        "analyzer": analyzer,
        "dh": analyzer.dh,
        "sigma_dh": analyzer.sigma_dh,
        "area_result": areas_uncertainty,
    }


class DEMUncertaintyAnalyzer:
    """
    A class to analyze uncertainty in Digital Elevation Models.
    """

    def __init__(self):
        """
        Initialize the uncertainty analyzer with paths to required data.
        """
        # Data containers
        self.ref_dem = None
        self.dem = None
        self.dh = None
        self.stable_mask = None
        self.glacier_outlines = None
        self.glacier_mask = None

        # Heteroscedasticity analysis
        self.predictors = {}
        self.df_binned = None
        self.zscores = None
        self.dh_err_fun = None

        # Spatial correlation analysis
        self.sigma_dh = None
        self.variogram_model = None

    def to_pickle(self, path: Path | str) -> bool:
        """
        Save the current state of the analyzer to a pickle file.

        Args:
            path (Path | str): Path to save the pickle file.

        Returns:
            bool: True if saved successfully, False otherwise.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f)
            return True
        except Exception as e:
            logger.error(f"Failed to save to {path}: {e}")
            return False

    @classmethod
    def from_pickle(cls, path: Path | str) -> "DEMUncertaintyAnalyzer":
        """
        Load the analyzer state from a pickle file.

        Args:
            path (Path | str): Path to the pickle file.

        Returns:
            DEMUncertaintyAnalyzer: Loaded instance of the analyzer.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist.")

        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load from {path}: {e}")
            raise

    def load_data(
        self,
        ref_dem_path: str | Path,
        dem_path: str | Path,
        stable_mask_path: str | Path = None,
        glacier_outlines_path: str | Path = None,
        subsample_res: int = None,
        no_data_value: float = -9999,
        additional_predictors: dict[str, Path | str] = None,
    ) -> "DEMUncertaintyAnalyzer":
        """
        Load and prepare all necessary data.

        Args:
            ref_dem_path (str | Path): Path to reference DEM.
            dem_path (str | Path): Path to DEM to analyze.
            stable_mask_path (str | Path, optional): Path to stable terrain mask.
            glacier_outlines_path (str | Path, optional): Path to glacier outlines vector file.
            subsample_res (int, optional): Resolution to subsample to for analysis.
            no_data_value (float, optional): No data value for the DEMs.
            additional_predictors (dict[str, Path | str], optional): Additional predictors.

        Returns:
            DEMUncertaintyAnalyzer: The instance of the analyzer.
        """
        logger.info("Loading data...")
        self.ref_dem_path = Path(ref_dem_path)
        self.dem_path = Path(dem_path)
        self.stable_mask_path = Path(stable_mask_path) if stable_mask_path else None
        self.glacier_outlines_path = (
            Path(glacier_outlines_path) if glacier_outlines_path else None
        )
        self.subsample_res = subsample_res
        self.no_data_value = no_data_value

        # Load reference DEM
        self.ref_dem = xdem.DEM(self.ref_dem_path)
        self.ref_dem.set_area_or_point("area")
        self.ref_dem.set_nodata(self.no_data_value)

        # Load DEM to analyze
        self.dem = xdem.DEM(self.dem_path)
        self.dem.set_area_or_point("area")
        self.dem.set_nodata(self.no_data_value)

        # Subsample if resolution specified
        if self.subsample_res:
            self.ref_dem = self.ref_dem.reproject(res=self.subsample_res)

        # Reproject the DEM to the reference DEM
        self.dem = self.dem.reproject(self.ref_dem, resampling="bilinear")

        # Load glacier outlines if provided
        if self.glacier_outlines_path:
            self.glacier_outlines = gu.Vector(self.glacier_outlines_path).crop(
                self.ref_dem
            )
            self.glacier_mask = self.glacier_outlines.create_mask(self.ref_dem)

        # Load stable area mask if provided
        if self.stable_mask_path:
            stable_mask_raster = gu.Raster(self.stable_mask_path)
            stable_mask_raster.set_nodata(255)
            stable_mask_raster.reproject(
                self.ref_dem, resampling="nearest", inplace=True
            )
            self.stable_mask = stable_mask_raster == 1

        # Load other data if provided (e.g., correlation value map)
        if additional_predictors:
            for key, value in additional_predictors.items():
                self.predictors[key] = gu.Raster(value).reproject(self.ref_dem)

        logger.info("Data loaded successfully.")
        return self

    def compute_elevation_difference(
        self, get_statistics: bool = True
    ) -> tuple[xdem.DEM, dict]:
        """
        Compute the elevation difference between reference and analyzed DEM.
        Optionally subsample the difference for more efficient analysis.

        Args:
            get_statistics (bool, optional): Whether to get statistics of the elevation difference.

        Returns:
            xdem.DEM: The elevation difference DEM.
            dict: Statistics of the elevation difference. None if get_statistics is False.
        """
        logger.info("Computing elevation difference...")
        self.dh = self.ref_dem - self.dem

        if get_statistics:
            stats = self.dh.get_stats(inlier_mask=self.stable_mask)
            logger.info("Elevation difference statistics:")
            for key, value in stats.items():
                logger.info(f"\t{key}: {value}")
        else:
            stats = None

        logger.info("Elevation difference computed.")
        return self.dh, stats

    def extract_terrain_features(
        self,
        attribute: list[str] = None,
        compute_abs_maxc: bool = True,
    ) -> "DEMUncertaintyAnalyzer":
        """
        Extract terrain features to be used as predictors for uncertainty.

        Args:
            attribute (list[str], optional): List of terrain attributes to extract.
            compute_abs_maxc (bool, optional): Whether to compute absolute value of the maximum curvature.

        Returns:
            DEMUncertaintyAnalyzer: The instance of the analyzer.
        """
        logger.info("Extracting terrain features...")
        if attribute is None:
            attribute = ["slope", "maximum_curvature"]

        # Compute slope and maximum curvature
        logger.debug(f"Computing terrain attributes: {attribute}")
        outs = xdem.terrain.get_terrain_attribute(self.ref_dem, attribute=attribute)

        # Stores the terrain features in the predictors dictionary
        for key, data in zip(attribute, outs, strict=False):
            # If requested, compute absolute value of the maximum curvature
            if key == "maximum_curvature" and compute_abs_maxc:
                logger.debug("Computing absolute values of the maximum curvature")
                data = np.abs(data)
            self.predictors[key] = data

        logger.info("Terrain features extracted.")
        return self

    def prepare_stable_terrain_data(
        self, nmad_factor: int = 5
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Extract data from stable terrain and filter outliers.

        Args:
            nmad_factor (int, optional): Factor to multiply NMAD for outlier filtering.

        Returns:
            tuple: Tuple containing the elevation difference array and dictionary of predictor arrays.
        """
        logger.info("Preparing stable terrain data...")
        if self.stable_mask is None:
            raise ValueError("Stable mask not loaded. Call load_data first.")
        if self.dh is None:
            raise ValueError(
                "Elevation difference not computed. Call compute_elevation_difference first."
            )
        if not self.predictors:
            raise ValueError(
                "Predictors not computed. Call extract_terrain_features first."
            )

        # Apply mask to the elevation difference
        dh_arr = self.dh[self.stable_mask].filled(np.nan)

        # Filter outliers in the elevation difference using NMAD
        arr_lim = nmad_factor * nmad(dh_arr)
        logger.debug(f"Filtering outliers with limit: {arr_lim}")
        outlier_mask = np.abs(dh_arr) > arr_lim
        dh_arr[outlier_mask] = np.nan

        # Extract arrays for stable terrain for each predictor
        predictors_arrays = {}
        for key, data in self.predictors.items():
            # Apply masks to the data (set masked values to NaN)
            arr = data[self.stable_mask].filled(np.nan)
            arr[outlier_mask] = np.nan
            predictors_arrays[key] = arr

        logger.info("Stable terrain data prepared.")
        return dh_arr, predictors_arrays

    def analyze_heteroscedasticity(
        self,
        nmad_factor: int = 5,
        statistics: list[str] = None,
        list_var_bins: list[float] = None,
        list_ranges: list[float] = None,
    ) -> tuple:
        """
        Analyze error heteroscedasticity and create error function.

        Args:
            nmad_factor (int, optional): Factor for outlier removal.
            statistics (list[str], optional): List of statistics to compute.
            list_var_bins (list[float], optional): List of bin limits for each predictor.
            list_ranges (list[float], optional): List of ranges for each predictor.

        Returns:
            tuple: Tuple containing the binned data, z-scores, error function, and sigma_dh.
        """
        logger.info("Analyzing heteroscedasticity...")
        # Get stable terrain data
        dh_arr, stable_data = self.prepare_stable_terrain_data(nmad_factor=nmad_factor)

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
        df = xdem.spatialstats.nd_binning(
            values=dh_arr,
            list_var=predictor_list,
            list_var_names=var_names,
            statistics=statistics,
            list_var_bins=list_var_bins,
            list_ranges=list_ranges,
        )

        # Fit error model
        unscaled_dh_err_fun = xdem.spatialstats.interp_nd_binning(
            df, list_var_names=var_names, statistic="nmad", min_count=50
        )

        # Compute the mean predicted elevation error on the stable terrain
        dh_err_stable = unscaled_dh_err_fun(
            tuple([stable_data[key] for key in var_names])
        )
        print(
            f"The spread of elevation difference is {xdem.spatialstats.nmad(dh_arr):.2f} compared to a mean predicted elevation error of {np.nanmean(dh_err_stable):.2f}."
        )

        # Two-step standardization
        zscores, dh_err_fun = xdem.spatialstats.two_step_standardization(
            dvalues=dh_arr,
            list_var=predictor_list,
            unscaled_error_fun=unscaled_dh_err_fun,
        )

        # Compute the error function for the whole DEM
        sigma_dh = self.dh.copy(
            new_array=dh_err_fun(
                tuple([self.predictors[key].data for key in var_names])
            )
        )

        # Save the results
        self.df_binned = df
        self.zscores = zscores
        self.dh_err_fun = dh_err_fun
        self.sigma_dh = sigma_dh

        logger.info("Heteroscedasticity analysis completed.")
        return df, zscores, dh_err_fun, sigma_dh

    def plot_error_vs_predictors_1D(
        self,
        statistics: str = "nmad",
        path: Path = None,
        axes: plt.Axes = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot error dependence on predictors.

        Args:
            statistics (str, optional): Statistic to plot.
            path (Path, optional): Path to save the plot.
            axes (plt.Axes, optional): Axes to plot on.

        Returns:
            tuple: Tuple containing the figure and axes.
        """
        logger.info("Plotting error vs predictors (1D)...")
        if self.df_binned is None:
            raise ValueError(
                "Binned data not available. Call analyze_heteroscedasticity first."
            )

        n_pred = len(self.predictors)

        # Create a figure with 1-D analysis for each predictor
        if axes is None:
            fig, axes = plt.subplots(1, n_pred, figsize=(n_pred * 4, 6))
        else:
            fig = None
            if len(axes) != n_pred:
                raise ValueError("Number of axes does not match number of predictors.")

        axes = [axes] if len(axes) == 1 else axes.flatten()

        # Loop through each predictor and plot
        for i, key in enumerate(self.predictors.keys()):
            # Plot the binned data
            xdem.spatialstats.plot_1d_binning(
                self.df_binned,
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

    def plot_error_vs_predictors_nD(
        self,
        statistics: str = "nmad",
        path: Path = None,
        axes: plt.Axes = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot error dependence on predictors.

        Args:
            statistics (str, optional): Statistic to plot.
            path (Path, optional): Path to save the plot.
            axes (plt.Axes, optional): Axes to plot on.

        Returns:
            tuple: Tuple containing the figure and axes.
        """
        logger.info("Plotting error vs predictors (nD)...")
        from itertools import combinations

        if self.df_binned is None:
            raise ValueError(
                "Binned data not available. Call analyze_heteroscedasticity first."
            )

        n_pred = len(self.predictors)
        if n_pred < 2:
            raise ValueError("At least two predictors are required for nD analysis.")

        # Create a figure with n-D analysis for each pair of predictors
        if axes is None:
            if len(self.predictors) == 2:
                n_plots = 1
            elif len(self.predictors) == 3:
                n_plots = 3
            else:
                raise NotImplementedError(
                    "plotting nD analysis for more than 3 predictors is not implemented yet."
                )
            fig, axes = plt.subplots(
                1,
                n_plots,
                figsize=(n_plots * 4, 6),
            )
        else:
            fig = None
            if len(axes) != len(self.predictors):
                raise ValueError("Number of axes does not match number of predictors.")

        axes = [axes] if len(axes) == 1 else axes.flatten()

        # Create a list with all the combinations of predictors
        pred_pairs = list(combinations(self.predictors.keys(), 2))

        # For each pair of predictors, plot the binned data to show the covariance
        for i, (key1, key2) in enumerate(pred_pairs):
            xdem.spatialstats.plot_2d_binning(
                df=self.df_binned,
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

    def plot_error_map(
        self,
        vmin: float = None,
        vmax: float = None,
        path: Path = None,
        ax: plt.Axes = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot map of estimated errors.

        Args:
            vmin (float, optional): Minimum value for color scale.
            vmax (float, optional): Maximum value for color scale.
            path (Path, optional): Path to save the plot.
            ax (plt.Axes, optional): Axes to plot on.

        Returns:
            tuple: Tuple containing the figure and axes.
        """
        logger.info("Plotting error map...")
        if self.sigma_dh is None:
            raise ValueError(
                "Error raster not computed. Call analyze_heteroscedasticity first."
            )
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        else:
            fig = None
        self.sigma_dh.plot(
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

    def analyze_spatial_correlation(
        self,
        standardize: bool = True,
        n_samples: int = 5000,
        subsample_method: str = "cdist_equidistant",
        n_variograms: int = 5,
        estimator: str = "dowd",
        random_state: int = None,
        list_models: list[str] = None,
        sample_kwargs: dict = None,
        fit_kwargs: dict = None,
    ) -> tuple:
        """
        Analyze spatial correlation of standardized errors.

        Args:
            standardize (bool, optional): Whether to standardize the elevation difference.
            n_samples (int, optional): Number of samples for variogram computation.
            subsample_method (str, optional): Method for subsampling.
            n_variograms (int, optional): Number of variogram realizations.
            estimator (str, optional): Estimator type for variogram calculation.
            random_state (int, optional): Random seed.
            list_models (list[str], optional): List of models for variogram fitting.
            sample_kwargs (dict, optional): Additional arguments for sampling.
            fit_kwargs (dict, optional): Additional arguments for fitting.

        Returns:
            tuple: Tuple containing the variogram function and parameters.
        """
        logger.info("Analyzing spatial correlation...")
        if self.sigma_dh is None:
            raise ValueError(
                "Error raster not computed. Call analyze_heteroscedasticity first."
            )

        # Compute standardized elevation difference if requested, otherwise use original
        z_dh = self.dh / self.sigma_dh if standardize else self.dh.copy()

        # Remove values on unstable terrain and large outliers
        z_dh.data[~self.stable_mask.data] = np.nan
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

        self.variogram_data = df_vgm
        self.variogram_function = func_sum_vgm
        self.variogram_params = params_vgm

        logger.info("Spatial correlation analysis completed.")
        return func_sum_vgm, params_vgm

    def plot_variogram(
        self,
        xscale_range_split: list[float] = None,
        list_fit_fun_label: list[str] = None,
        path: Path = None,
        ax: plt.Axes = None,
        **kwargs,
    ) -> None:
        """
        Plot the empirical variogram and fitted model.

        Args:
            xscale_range_split (list[float], optional): List of x-scale range splits.
            list_fit_fun_label (list[str], optional): List of labels for fitted functions.
            path (Path, optional): Path to save the plot.
            ax (plt.Axes, optional): Axes to plot on.
            **kwargs: Additional arguments for plotting.
        """
        logger.info("Plotting variogram...")
        if self.variogram_data is None:
            raise ValueError(
                "Variogram data not available. Call analyze_spatial_correlation first."
            )
        if self.variogram_function is None:
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
            self.variogram_data,
            xscale_range_split=xscale_range_split,
            list_fit_fun=[self.variogram_function],
            list_fit_fun_label=list_fit_fun_label,
            ax=ax,
            **kwargs,
        )
        logger.info("Variogram plot created.")

        if path is not None and fig is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=300, bbox_inches="tight")

    def compute_uncertainty_for_area(
        self,
        area_vector: gu.Vector | Path = None,
        column_name: str = "rgi_id",
        area_name: str | list[str] = None,
        min_area_fraction: float = 0.05,
    ) -> dict:
        """Compute uncertainty for specific area(s) (e.g., glaciers).

        This function computes the uncertainty for one or multiple areas. It calculates
        the mean elevation difference, the mean uncertainty, and the effective number
        of samples for each area, accounting for spatial correlation.

        Args:
            area_vector (gu.Vector, optional): Vector defining the area(s) of interest. If not provided, the glacier outlines stored in the class will be used.
            column_name (str, optional): Column name in the vector dataset to use for area identification. Defaults to "rgi_id".
            area_name (str | list[str], optional): Name/ID(s) for the area(s) to analyze. If provided, only areas matching these IDs will be processed. Otherwise, all areas in the vector will be processed.
            min_area_fraction (float, optional): Minimum area percentage to consider for uncertainty analysis. Defaults to 0.05 (5% of the area).

        Returns:
            dict: Dictionary with area names/IDs as keys and uncertainty results as values.
                Each result contains:
                - area: The area vector
                - mean_elevation_diff: Mean elevation difference in the area
                - mean_uncertainty_unscaled: Mean uncertainty in the area
                - effective_samples: Effective number of samples in the area
                - uncertainty: Standard error of the mean elevation difference

        Raises:
            ValueError: If spatial correlation has not been analyzed yet.
            ValueError: If neither area_vector nor stored glacier outlines are available.
            ValueError: If the specified column_name does not exist in the vector dataset.
            ValueError: If area_name is provided but not found in the vector dataset.
        """
        logger.info("Computing uncertainty for area(s)...")
        if not hasattr(self, "variogram_params"):
            raise ValueError(
                "Spatial correlation not analyzed. Call analyze_spatial_correlation first."
            )

        # Determine which vector to use (input or stored glacier outlines)
        if area_vector is None:
            if self.glacier_outlines is None:
                raise ValueError(
                    "No area vector provided and glacier outlines not loaded"
                )
            work_vector = self.glacier_outlines.copy()
            logger.debug("Using stored glacier outlines")
        else:
            if isinstance(area_vector, (str, Path)):
                try:
                    area_vector = gu.Vector(area_vector)
                except Exception as e:
                    raise ValueError(
                        f"Failed to load area vector from {area_vector}: {e}"
                    )
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

        # Initialize results dictionary
        results = {}

        # Process each geometry in the vector
        for idx, row in work_vector.ds.iterrows():
            # Get area identifier from column
            area_id = row[column_name]
            logger.info(f"Processing area: {area_id}")

            # Extract single geometry
            single_geom = work_vector.copy()
            single_geom.ds = single_geom.ds.iloc[[idx]]

            # Create mask for the current geometry
            area_mask = single_geom.create_mask(self.ref_dem)

            # Check if the mask contains any valid pixels
            if not np.any(area_mask):
                logger.warning(f"Area {area_id} has no valid pixels - skipping")
                continue

            # Check if the DEM covers at least min_area_fraction of the area
            dem_px_in_area = len(self.dh[area_mask].compressed())
            area_coverage = dem_px_in_area / np.sum(area_mask)
            if area_coverage < min_area_fraction:
                logger.info(
                    f"DEM {self.dem_path.name} covers only {area_coverage:.2%} of the area {area_id}. It's less than the minimum fraction {min_area_fraction:.2%} - skipping"
                )
                continue

            # Compute the mean elevation difference in the area and the mean error
            dh_area = np.nanmean(self.dh[area_mask])
            mean_sig = np.nanmean(self.sigma_dh[area_mask])
            logger.info(
                f"Area {area_id}: Mean elevation difference: {dh_area:.2f} ± {mean_sig:.2f} m"
            )

            # Calculate effective number of samples
            n_eff = xdem.spatialstats.number_effective_samples(
                area=single_geom,
                params_variogram_model=self.variogram_params,
                # rasterize_resolution=10,
            )
            logger.info(f"Area {area_id}: Effective number of samples: {n_eff:.2f}")

            # Rescale the standard deviation of the mean elevation difference with the effective number of samples
            sig_dh_area = mean_sig / np.sqrt(n_eff)
            err_perc = (
                sig_dh_area / abs(dh_area) * 100 if dh_area != 0 else float("inf")
            )
            logger.info(
                f"Area {area_id}: Random error for mean elevation change: {sig_dh_area:.2f} m ({err_perc:.2f}%)"
            )

            # Store results for this geometry using the area_id as key
            results[area_id] = {
                "mean_elevation_diff": dh_area,
                "mean_uncertainty_unscaled": mean_sig,
                "effective_samples": n_eff,
                "uncertainty": sig_dh_area,
            }

            logger.info(f"Completed uncertainty calculation for {len(results)} area(s)")

        # Convert the results to a dataframe
        results_df = pd.DataFrame.from_dict(results, orient="index")

        # if the dataframe is empty, create an empty dataframe with the same columns
        if results_df.empty:
            results_df = pd.DataFrame(
                columns=[
                    "mean_elevation_diff",
                    "mean_uncertainty_unscaled",
                    "effective_samples",
                    "uncertainty",
                ]
            )

        return results_df

    def plot_area_result(self, area_result, figsize=(12, 6), vmin=-30, vmax=30):
        """
        Plot elevation difference with uncertainty for a specific area.

        Parameters
        ----------
        area_result : dict
            Result from compute_uncertainty_for_area
        figsize : tuple, optional
            Figure size
        vmin, vmax : float, optional
            Limits for color scale

        Returns
        -------
        matplotlib.figure.Figure
            Figure with plot
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        self.dh.plot(
            cmap="RdYlBu",
            cbar_title="Elevation differences (m)",
            ax=ax,
            vmin=vmin,
            vmax=vmax,
        )
        area_result["area"].plot(self.dh, fc="none", ec="black", lw=2)

        # Add text with results
        plt.text(
            area_result["area"].ds.centroid.x.values[0] - 10000,
            area_result["area"].ds.centroid.y.values[0] - 10000,
            f"{area_result['mean_elevation_diff']:.2f} \n$\\pm$ {area_result['standard_error']:.2f} m",
            color="black",
            fontweight="bold",
            va="top",
            ha="center",
        )

        return fig

    def create_error_mask(self, max_percent=0.90, on_glacier_only=False):
        """
        Create a mask for areas with acceptable error levels.

        Parameters
        ----------
        max_percent : float, optional
            Percentile threshold for error filtering
        on_glacier_only : bool, optional
            Whether to compute percentile only on glacier areas

        Returns
        -------
        gu.Mask
            Mask of areas with acceptable error
        """
        if self.sigma_dh is None:
            self.compute_error_raster()

        if on_glacier_only:
            if self.glacier_mask is None:
                raise ValueError(
                    "Glacier mask not available, can't filter on glacier areas only"
                )

            # Compute percentile on glacier areas only
            max_error = np.nanquantile(
                self.sigma_dh[self.glacier_mask].compressed(), max_percent
            )
            valid = self.sigma_dh.data < max_error
            error_mask = gu.Mask.from_array(
                valid, self.sigma_dh.transform, self.sigma_dh.crs
            )
            error_mask.set_mask(~self.glacier_mask)
        else:
            # Compute percentile on all areas
            max_error = np.nanquantile(self.sigma_dh.data.compressed(), max_percent)
            error_mask = self.sigma_dh < max_error

        return error_mask


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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.genmod.families import NegativeBinomial, Poisson
from statsmodels.genmod.generalized_linear_model import GLM
from typing import Dict, List, Tuple, Optional, Union


class ClimateHealthPipeline:
    """
    A modular pipeline for analyzing the relationship between weather patterns
    and healthcare service utilization.
    """

    def __init__(self, config: Dict):
        """
        Initialize the pipeline with configuration parameters.

        Args:
            config: Dictionary containing pipeline configuration
        """
        self.config = config
        self.service_type = config.get('service_type', 'ANC')
        self.min_year_for_analysis = config.get('min_year_for_analysis', 2012)
        self.absolute_min_year = config.get('absolute_min_year', 2011)
        self.mask_threshold = config.get('mask_threshold', -np.inf)
        self.covid_months = config.get('covid_months', range(96, 116))
        self.log_y = config.get('log_y', True)
        self.poisson = config.get('poisson', False)

        # Initialize storage for models and data
        self.baseline_model = None
        self.weather_model = None
        self.scaler = StandardScaler()
        self.facility_info = None
        self.has_monthly_weather = False
        self.has_daily_weather = False

    def load_appointment_data(self, filepath: str) -> pd.DataFrame:
        """
        Load and preprocess appointment/service utilization data.

        Args:
            filepath: Path to the appointment data CSV file

        Returns:
            Preprocessed appointment data DataFrame
        """
        monthly_reporting_by_facility = pd.read_csv(
            "/Users/rem76/Desktop/Climate_change_health/Data/Temperature_data/WBGT/monthly_reporting_ANC_by_smaller_facility_lm_wbgt.csv",
            index_col=0)
        zero_sum_columns = monthly_reporting_by_facility.columns[(monthly_reporting_by_facility.sum(axis=0) == 0)]
        data = monthly_reporting_by_facility.drop(columns=zero_sum_columns)


        # Mask COVID months if specified
        if hasattr(self, 'covid_months') and self.covid_months is not None:
            data.iloc[self.covid_months, :] = np.nan
            print(f"Masked {len(self.covid_months)} COVID-affected months")

        # Filter by year range
        start_idx = (self.min_year_for_analysis - self.absolute_min_year) * 12
        data = data.iloc[start_idx:]

        return data, zero_sum_columns

    def load_weather_data(self, monthly_filepath: Optional[str] = None,
                          daily_filepath: Optional[str] = None) -> Tuple[
        Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load and preprocess weather data. Can load either monthly, daily, or both.

        Args:
            monthly_filepath: Optional path to monthly weather data
            daily_filepath: Optional path to daily weather data for extreme events

        Returns:
            Tuple of (monthly_weather_df, daily_weather_df), where either can be None
        """
        if monthly_filepath is None and daily_filepath is None:
            print("WARNING: No weather data paths provided. Pipeline will run without weather features.")
            return None, None

        monthly_data = None
        daily_data = None
        monthly_data_all_years = None
        daily_data_all_years = None
        start_idx = (self.min_year_for_analysis - self.absolute_min_year) * 12

        if monthly_filepath:
            print(f"Loading monthly weather data from {monthly_filepath}")
            monthly_data_all_years = pd.read_csv(monthly_filepath, index_col=0)
            monthly_data = monthly_data_all_years.iloc[start_idx:]
            self.has_monthly_weather = True

        if daily_filepath:
            print(f"Loading daily weather data from {daily_filepath}")
            daily_data_all_years = pd.read_csv(daily_filepath, index_col=0)
            daily_data = daily_data_all_years.iloc[start_idx:]
            self.has_daily_weather = True

        return monthly_data_all_years, monthly_data, daily_data_all_years, daily_data

    def load_facility_info(self, filepath: str) -> pd.DataFrame:
        """
        Load facility metadata (location, type, etc.).

        Args:
            filepath: Path to facility information CSV

        Returns:
            Facility information DataFrame
        """
        facility_info = pd.read_csv(filepath, index_col=0)
        self.facility_info = facility_info
        return facility_info

    def create_lagged_features(self, weather_data_all_years: pd.DataFrame, lags: List[int] = [1, 2, 3, 4, 9]) -> Dict[
        str, np.ndarray]:
        """
        Create lagged weather features.

        Args:
            weather_data: Weather data DataFrame
            lags: List of lag periods (in months)

        Returns:
            Dictionary of lagged features
        """
        lagged_features = {}

        for lag in lags:
            lagged_data = weather_data_all_years.shift(lag).values
            start_idx = (self.min_year_for_analysis - self.absolute_min_year) * 12
            lagged_features[f'lag_{lag}'] = lagged_data[start_idx:].flatten()

        return lagged_features

    def prepare_features(self, appointment_data: pd.DataFrame,
                         weather_data: Optional[pd.DataFrame] = None,
                         weather_data_all_years: Optional[pd.DataFrame] = None,
                         daily_weather_data: Optional[pd.DataFrame] = None,
                         facility_info: Optional[pd.DataFrame] = None,
                         zero_sum_columns: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Prepare feature matrix and target variable. Works with any combination of weather data.

        Args:
            appointment_data: Healthcare appointment data
            weather_data: Optional monthly weather data
            daily_weather_data: Optional daily weather data
            facility_info: Facility metadata

        Returns:
            Tuple of (feature_matrix, target_variable, num_weather_features)
        """
        # Create time variables
        year_range = range(self.min_year_for_analysis, 2025)
        num_facilities = len(appointment_data.columns)

        # format facility information
        facility_info = facility_info.drop(columns=zero_sum_columns)
        #facility_info = facility_info[common_columns]
        facility_info = facility_info.T.reindex(columns=facility_info.index)
        # Create flattened time series
        year_flattened = []
        month_flattened = []
        facility_flattened = []

        for year in year_range:
            for month in range(1, 13):
                for facility in appointment_data.columns:
                    year_flattened.append(year)
                    month_flattened.append(month)
                    facility_flattened.append(facility)

        # Trim to match data length
        data_length = len(appointment_data) * num_facilities
        year_flattened = year_flattened[:data_length]
        month_flattened = month_flattened[:data_length]
        facility_flattened = facility_flattened[:data_length]

        # Flatten target variable
        y = appointment_data.values.flatten()
        if np.nanmin(y[~np.isnan(y)]) < 1:
            y = y + 1

        # Remove extreme outliers
        y[y > 4000] = np.nan

        # Create weather features
        weather_features = []
        num_weather_features = 0

        weather_data = weather_data.drop(columns=zero_sum_columns)
        weather_data_all_years = weather_data_all_years.drop(columns=zero_sum_columns)
        # Monthly weather features
        if weather_data is not None:
            weather_monthly = weather_data.values.flatten()
            weather_features.append(weather_monthly)
            num_weather_features += 1

            # Polynomial features
            weather_features.append(weather_monthly ** 2)
            weather_features.append(weather_monthly ** 3)
            num_weather_features += 2

            # Lagged features
            lagged_features = self.create_lagged_features(weather_data_all_years)
            for lag_name, lag_data in lagged_features.items():
                weather_features.append(lag_data)
                num_weather_features += 1

        # Daily weather features
        if daily_weather_data is not None:
            daily_flattened = daily_weather_data.values.flatten()
            weather_features.append(daily_flattened)
            num_weather_features += 1

            weather_features.append(daily_flattened ** 2)
            num_weather_features += 1

            # Interaction with monthly weather if available
            if weather_data is not None and len(weather_features) >= 2:
                weather_monthly = weather_data.values.flatten()
                weather_features.append(weather_monthly * daily_flattened)
                num_weather_features += 1

        # Build feature matrix
        continuous_features = [
            np.array(year_flattened),
            np.array(month_flattened)
        ]

        # Add all weather features
        continuous_features.extend(weather_features)
        # Add facility information if available
        categorical_features = []
        if facility_info is not None:
            categorical_features = self._create_facility_features(
                facility_flattened, facility_info, len(year_range)
            )
            # Add continuous facility features
            if 'A109__Altitude' in facility_info.columns:
                altitude = self._repeat_facility_info(
                    facility_info['A109__Altitude'], len(year_range)
                )
                # Convert to numeric, coercing errors to NaN
                altitude = pd.to_numeric(altitude, errors='coerce')
                altitude = np.array(altitude)
                altitude = np.where(altitude < 0, np.nan, altitude)
                mean_altitude = round(np.nanmean(altitude))
                altitude = np.where(np.isnan(altitude), float(mean_altitude), altitude)
                altitude = np.nan_to_num(altitude, nan=mean_altitude, posinf=mean_altitude,
                                                  neginf=mean_altitude)
                altitude = list(altitude)
                continuous_features.append(np.array(altitude))

            if 'minimum_distance' in facility_info.columns:
                min_dist = self._repeat_facility_info(
                    facility_info['minimum_distance'], len(year_range)
                )
                min_dist = np.nan_to_num(min_dist, nan=np.nan, posinf=np.nan,
                                                          neginf=np.nan)
                min_dist = pd.to_numeric(min_dist, errors='coerce')
                print(min_dist)
                continuous_features.append(np.array(min_dist))


        # Combine all features
        continuous_features = [arr.astype(float) for arr in continuous_features]

        X_continuous = np.column_stack(continuous_features)

        if categorical_features:
            X_categorical = np.column_stack(categorical_features)
            X = np.column_stack([X_continuous, X_categorical])
        else:
            X = X_continuous

        return X, y, num_weather_features

    def _create_facility_features(self, facility_flattened: List,
                                  facility_info: pd.DataFrame,
                                  year_range_len: int) -> List[np.ndarray]:
        """Create categorical facility features."""
        categorical_features = []

        categorical_columns = ['Zonename', 'Dist', 'Resid', 'A105', 'Ftype']

        for col in categorical_columns:
            if col in facility_info.columns:
                feature_values = self._repeat_facility_info(
                    facility_info[col].values,  year_range_len
                )
                encoded_feature = pd.get_dummies(feature_values, drop_first=True)
                categorical_features.extend([encoded_feature.iloc[:, i].values
                                             for i in range(encoded_feature.shape[1])])
                for i in range(encoded_feature.shape[1]):
                    feature_array = encoded_feature.iloc[:, i].values.astype(float)
                    categorical_features.append(feature_array)

        return categorical_features

    def _repeat_facility_info(self, info_series: pd.Series,
                              year_range_len: int) -> List:
        """Repeat facility information for each time point."""
        repeated_info = []
        for _ in range(year_range_len):
            for _ in range(12):
                repeated_info.extend(info_series.tolist())
        return repeated_info

    def stepwise_selection(self, X: np.ndarray, y: np.ndarray,
                           alpha_enter: float = 0.05,
                           alpha_remove: float = 0.10) -> Tuple[np.ndarray, object, np.ndarray, np.ndarray]:
        """
        Perform stepwise feature selection.

        Args:
            X: Feature matrix
            y: Target variable
            alpha_enter: P-value threshold for entering variables
            alpha_remove: P-value threshold for removing variables

        Returns:
            Tuple of (included_features, model_results, predictions, data_mask)
        """
        print("Performing stepwise feature selection...")

        mask = (~np.isnan(X).any(axis=1) & ~np.isnan(y) & (y <= 1e4))

        X_clean = X[mask]
        y_clean = y[mask]


        n_features = X_clean.shape[1]
        included = np.zeros(n_features, dtype=bool)

        for step in range(n_features):
            candidates = ~included
            if not candidates.any():
                break

            best_pval = float('inf')
            best_feature = None

            for feature in np.where(candidates)[0]:
                test_included = included.copy()
                test_included[feature] = True

                X_subset = X_clean[:, test_included]
                X_subset = sm.add_constant(X_subset)

                if self.log_y:
                    y_model = np.log(y_clean)
                    print(X_subset)
                    model = sm.OLS(y_model, X_subset).fit()
                else:
                    if self.poisson:
                        model = GLM(y_clean, X_subset, family=Poisson()).fit()
                    else:
                        model = sm.OLS(y_clean, X_subset).fit()

                if len(model.pvalues) > 1:
                    pval = model.pvalues.iloc[-1] if hasattr(model.pvalues, 'iloc') else model.pvalues[-1]
                    if pval < best_pval:
                        best_pval = pval
                        best_feature = feature

            if best_feature is not None and best_pval < alpha_enter:
                included[best_feature] = True
                print(f"Added feature {best_feature} (p={best_pval:.4f})")
            else:
                break

        # Fit final model
        if included.any():
            X_final = X_clean[:, included]
            X_final = sm.add_constant(X_final)

            if self.log_y:
                y_model = np.log(y_clean)
                final_model = sm.OLS(y_model, X_final).fit()
            else:
                if self.poisson:
                    final_model = GLM(y_clean, X_final, family=Poisson()).fit()
                else:
                    final_model = sm.OLS(y_clean, X_final).fit()

            predictions = final_model.predict(X_final)
        else:
            X_final = sm.add_constant(np.ones((len(y_clean), 1)))
            if self.log_y:
                y_model = np.log(y_clean)
                final_model = sm.OLS(y_model, X_final).fit()
            else:
                final_model = sm.OLS(y_clean, X_final).fit()
            predictions = final_model.predict(X_final)

        return included, final_model, predictions, mask

    def fit_baseline_model(self, X: np.ndarray, y: np.ndarray, num_weather_features: int) -> object:
        """
        Fit baseline model without weather variables.

        Args:
            X: Feature matrix (including weather)
            y: Target variable
            num_weather_features: Number of weather features to exclude

        Returns:
            Fitted baseline model
        """
        print("Fitting baseline model (excluding weather features)...")

        # Exclude weather features (assuming they come after year and month)
        # Year and month are the first 2 features
        weather_start_idx = 2
        weather_end_idx = weather_start_idx + num_weather_features

        # Keep year, month, and all non-weather features
        baseline_indices = list(range(2)) + list(range(weather_end_idx, X.shape[1]))
        X_baseline = X[:, baseline_indices]

        print(
            f"Baseline model using {X_baseline.shape[1]} features (excluding {num_weather_features} weather features)")

        included, model, predictions, mask = self.stepwise_selection(X_baseline, y)

        self.baseline_model = {
            'model': model,
            'included_features': included,
            'predictions': predictions,
            'mask': mask
        }
        print("baseline model", included)

        print("Baseline model fitted successfully")
        return self.baseline_model

    def fit_weather_model(self, X: np.ndarray, y: np.ndarray) -> object:
        """
        Fit full model with weather variables.

        Args:
            X: Full feature matrix (including weather)
            y: Target variable

        Returns:
            Fitted weather model
        """
        print("Fitting weather model (including all features)...")

        included, model, predictions, mask = self.stepwise_selection(X, y)
        print("weather model", included)
        self.weather_model = {
            'model': model,
            'included_features': included,
            'predictions': predictions,
            'mask': mask
        }

        print("Weather model fitted successfully")
        return self.weather_model

    def calculate_weather_impact(self) -> pd.DataFrame:
        """
        Calculate the impact of weather on healthcare utilization,
        accounting for differing masks and prediction lengths.

        Returns:
            DataFrame with weather impact analysis
        """
        if self.baseline_model is None or self.weather_model is None:
            raise ValueError("Both baseline and weather models must be fitted first")

        print("Calculating weather impact with alignment of masks...")

        baseline_pred = self.baseline_model['predictions']
        weather_pred = self.weather_model['predictions']

        baseline_mask = self.baseline_model['mask']
        weather_mask = self.weather_model['mask']

        # Determine the combined valid mask
        combined_mask = baseline_mask & weather_mask
        print(f"Number of valid points after mask alignment: {np.sum(combined_mask)}")

        # Create full-length arrays with NaNs where predictions were masked
        full_baseline_pred = np.full(baseline_mask.shape, np.nan)
        full_baseline_pred[baseline_mask] = baseline_pred

        full_weather_pred = np.full(weather_mask.shape, np.nan)
        full_weather_pred[weather_mask] = weather_pred

        # Align using combined mask
        aligned_baseline = full_baseline_pred[combined_mask]
        aligned_weather = full_weather_pred[combined_mask]

        if self.log_y:
            difference = np.exp(aligned_weather) - np.exp(aligned_baseline)
        else:
            difference = aligned_weather - aligned_baseline

        results = pd.DataFrame({
            'baseline_prediction': np.exp(aligned_baseline) if self.log_y else aligned_baseline,
            'weather_prediction': np.exp(aligned_weather) if self.log_y else aligned_weather,
            'weather_impact': difference
        })

        return results

    def likelihood_ratio_test(self) -> Dict:
        """
        Perform likelihood ratio test between baseline and weather models.

        Returns:
            Dictionary with test statistics
        """
        if self.baseline_model is None or self.weather_model is None:
            raise ValueError("Both models must be fitted first")

        ll_baseline = self.baseline_model['model'].llf
        ll_weather = self.weather_model['model'].llf

        lr_stat = -2 * (ll_baseline - ll_weather)
        df = len(self.weather_model['model'].params) - len(self.baseline_model['model'].params)
        p_value = 1 - stats.chi2.cdf(lr_stat, df)

        results = {
            'lr_statistic': lr_stat,
            'degrees_of_freedom': df,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

        print(f"Likelihood Ratio Test: LR={lr_stat:.4f}, df={df}, p={p_value:.4f}")

        return results

    def plot_results(self, weather_impact: pd.DataFrame,
                     monthly_weather_data: Optional[pd.DataFrame] = None,
                     save_path: Optional[str] = None) -> None:
        """
        Create visualization of results.

        Args:
            weather_impact: DataFrame from calculate_weather_impact()
            monthly_weather_data: Optional monthly weather data for x-axis
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Predictions comparison
        ax1.scatter(range(len(weather_impact)), weather_impact['baseline_prediction'],
                    alpha=0.5, label='Baseline Model', color='blue')
        ax1.scatter(range(len(weather_impact)), weather_impact['weather_prediction'],
                    alpha=0.5, label='Weather Model', color='red')
        ax1.set_xlabel('Observation')
        ax1.set_ylabel(f'{self.service_type} Visits')
        ax1.set_title('Model Predictions Comparison')
        ax1.legend()

        # Weather impact plot
        if monthly_weather_data is not None:
            x_axis = monthly_weather_data.values.flatten()[:len(weather_impact)]
        else:
            x_axis = range(len(weather_impact))

        ax2.scatter(x_axis, weather_impact['weather_impact'],
                    alpha=0.5, color='green')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Monthly Weather')
        ax2.set_ylabel('Weather Impact on Visits')
        ax2.set_title('Impact of Weather on Healthcare Utilization')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def save_results(self, output_dir: str) -> None:
        """
        Save all model results and predictions.

        Args:
            output_dir: Directory to save results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        if self.baseline_model:
            with open(f"{output_dir}/baseline_model_summary.txt", 'w') as f:
                f.write(str(self.baseline_model['model'].summary()))

        if self.weather_model:
            with open(f"{output_dir}/weather_model_summary.txt", 'w') as f:
                f.write(str(self.weather_model['model'].summary()))

        if self.baseline_model and self.weather_model:
            lr_results = self.likelihood_ratio_test()
            lr_df = pd.DataFrame([lr_results])
            lr_df.to_csv(f"{output_dir}/likelihood_ratio_test.csv", index=False)

        print(f"Results saved to {output_dir}")


def run_pipeline(config: Dict) -> ClimateHealthPipeline:
    """
    Main function to run the complete pipeline.

    Args:
        config: Configuration dictionary

    Returns:
        Fitted pipeline object
    """
    pipeline = ClimateHealthPipeline(config)

    # Load data
    appointment_data, zero_sum_columns = pipeline.load_appointment_data(config['appointment_data_path'])

    # Load weather data (either, both, or neither)
    monthly_weather_all_years, monthly_weather, daily_weather_all_years, daily_weather = pipeline.load_weather_data(
        config.get('monthly_weather_path'),
        config.get('daily_weather_path')
    )
    facility_info = None
    if 'facility_info_path' in config:
        facility_info = pipeline.load_facility_info(config['facility_info_path'])

    # Prepare features
    X, y, num_weather_features = pipeline.prepare_features(
        appointment_data, monthly_weather, monthly_weather_all_years, daily_weather, facility_info, zero_sum_columns
    )

    # Fit models
    pipeline.fit_baseline_model(X, y, num_weather_features)
    pipeline.fit_weather_model(X, y)

    # Calculate impact and perform tests (only if weather features exist)
    if num_weather_features > 0:
        weather_impact = pipeline.calculate_weather_impact()
        lr_results = pipeline.likelihood_ratio_test()
        pipeline.plot_results(weather_impact, monthly_weather_data=monthly_weather)
    else:
        print("No weather features available. Skipping weather impact analysis.")

    # Save results
    if 'output_dir' in config:
        pipeline.save_results(config['output_dir'])

    return pipeline


# Example usage
if __name__ == "__main__":

    config_monthly = {
        'service_type': 'ANC',
        'min_year_for_analysis': 2012,
        'absolute_min_year': 2011,
        'appointment_data_path': '/Users/rem76/Desktop/Climate_change_health/Data/ANC_data/ANC_data_2011_2024.csv',
        'monthly_weather_path': '/Users/rem76/Desktop/Climate_change_health/Data/Temperature_data/WBGT/Historical/historical_wbgt_by_smaller_facilities_with_ANC_lm.csv',
        'facility_info_path': '/Users/rem76/Desktop/Climate_change_health/Data/Temperature_data/WBGT/expanded_facility_info_wbgt_by_smaller_facility_lm_with_ANC.csv',
        'daily_weather_path':  None, # for WBGT, unsure which index to use - as sustained does make a difference...
        'covid_months': range(96, 116),
        'output_dir': '/Users/rem76/Desktop/Climate_change_health/Results/WBGT',
        'log_y': True
    }

    # Run pipeline with desired configuration
    pipeline = run_pipeline(config_monthly)

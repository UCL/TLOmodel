from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from tlo import Date, DateOffset, Module, Parameter, Population, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent, Priority

from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.hsi_event import (
    LABEL_FOR_MERGED_FACILITY_LEVELS_1B_AND_2,
    FacilityInfo,
    HSI_Event,
    HSIEventDetails,
    HSIEventQueueItem,
    HSIEventWrapper,
)
from tlo.util import read_csv_files

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger_summary = logging.getLogger(f"{__name__}.summary")
logger_summary.setLevel(logging.INFO)

class WeatherDisruptions(Module):
    """
    Module to handle climate-mediated disruptions to healthcare services.
    Manages weather-related service interruptions and their impact on
    healthcare seeking behavior and appointment scheduling.

    """

    # Declare dependencies
    INIT_DEPENDENCIES = {"Demography", "HealthSeekingBehaviour"}

    OPTIONAL_INIT_DEPENDENCIES = {"HealthSystem"}

    # Declare Metadata
    METADATA = {Metadata.USES_HEALTHSYSTEM}

    # Define module-level parameters

    PARAMETERS = {

        # Disruption-related parameters
        "projected_precip_disruptions": Parameter(
            Types.REAL, "Probabilities of precipitation-mediated " "disruptions to services by month, year, and clinic."
        ),
        "scale_factor_prob_disruption": Parameter(
            Types.REAL,
            "Due to uknown behaviours (from patient and health practiciion), broken chains of events, etc, which cause discrepencies  "
            "between the estimated disruptions and those modelled in TLO, rescale the original probability of disruption.",
        ),
        # Scenario-related parameters
        "climate_ssp": Parameter(
            Types.STRING,
            "Which future shared socioeconomic pathway (determines degree of "
            "warming) is under consideration."
            "Options are ssp126, ssp245, and ssp585, in terms of increasing "
            "severity.",
        ),
        "climate_model_ensemble_model": Parameter(
            Types.STRING,
            "Which model from the model ensemble for each climate ssp is under consideration."
            "Options are lowest, mean, and highest, based on total precipitation between 2025 and 2070.",
        ),
        "year_effective_climate_disruptions": Parameter(Types.INT,
                                                        "Mimimum year from which there can be climate disruptions. Minimum is 2025"),

        "services_affected_precip": Parameter(
            Types.STRING, "Which modelled services can be affected by weather. Options are all, none"
        ),

        # Rescheduling parameters
        "delay_in_seeking_care_weather": Parameter(
            Types.REAL,
            "If faced with a climate disruption, and it is determined the individual will "
            "reseek healthcare, the number of days of delay in seeking healthcare."
            "Scale factor makes it proportional to the urgency.",
        ),
        "scale_factor_reseeking_healthcare_post_disruption": Parameter(
            Types.REAL,
            "If faced with a climate disruption, and it is determined the individual will "
            "reseek healthcare, scaling of their original probability of seeking care.",
        ),

        "scale_factor_appointment_urgency": Parameter(
            Types.REAL,
            "Scale factor in seeking healthcare for how urgent a HSI is."
        ),
        "scale_factor_severity_disruption_and_delay": Parameter(
            Types.REAL,
            "Scale factor that changes the delay in reseeking healthcare to the severity of disruption (as measured by probability of disruption)",
        ),
        "prop_supply_side_disruptions": Parameter(
            Types.REAL,
            "Probability that a climate disruption is supply-side (consumes capabilities in mode 2) "
            "vs demand-side (frees up capabilities in mode 2)."
        ),

        # Baseline model, no precipitation coefficients
        "baseline_coef_intercept": Parameter(Types.REAL, "Baseline model intercept"),
        "baseline_coef_year": Parameter(Types.REAL, "Baseline: year"),
        "baseline_coef_month": Parameter(Types.REAL, "Baseline: month"),
        "baseline_coef_min_distance": Parameter(Types.REAL, "Baseline: min distance to facility"),
        "baseline_coef_altitude": Parameter(Types.REAL, "Baseline: altitude of facility"),
        "baseline_coef_urban": Parameter(Types.REAL, "Baseline: urban vs rural"),
        "baseline_coef_central_west": Parameter(Types.REAL, "Baseline: Central West zone"),
        "baseline_coef_northern": Parameter(Types.REAL, "Baseline: Northern zone"),
        "baseline_coef_south_east": Parameter(Types.REAL, "Baseline: South East zone"),
        "baseline_coef_south_west": Parameter(Types.REAL, "Baseline: South West zone"),
        "baseline_coef_government": Parameter(Types.REAL, "Baseline: Government ownership"),
        "baseline_coef_private": Parameter(Types.REAL, "Baseline: Private ownership"),

        # Precipitation model coefficients
        "precipitation_coef_intercept": Parameter(Types.REAL, "Precipitation model intercept"),
        "precipitation_coef_year": Parameter(Types.REAL, "Precipitation: year"),
        "precipitation_coef_month": Parameter(Types.REAL, "Precipitation: month"),
        "precipitation_coef_min_distance": Parameter(Types.REAL, "Precipitation: min distance"),
        "precipitation_coef_altitude": Parameter(Types.REAL, "Precipitation: altitude of facility"),
        "precipitation_coef_urban": Parameter(Types.REAL, "Precipitation: urban vs rural"),
        "precipitation_coef_central_west": Parameter(Types.REAL, "Precipitation: Central West zone"),
        "precipitation_coef_northern": Parameter(Types.REAL, "Precipitation: Northern zone"),
        "precipitation_coef_south_east": Parameter(Types.REAL, "Precipitation: South East zone"),
        "precipitation_coef_south_west": Parameter(Types.REAL, "Baseline: South East zone"),
        "precipitation_coef_government": Parameter(Types.REAL, "Precipitation: Government ownership"),
        "precipitation_coef_private": Parameter(Types.REAL, "Precipitation: Private ownership"),
        "precipitation_coef_precip_monthly": Parameter(Types.REAL, "Precipitation: monthly cumulative precip"),
        "precipitation_coef_precip_5day": Parameter(Types.REAL, "Precipitation: 5-day cumulative max precip"),
        "precipitation_coef_lag_4month": Parameter(Types.REAL, "Precipitation: monthly cumulative precip 4 month lag"),
        "precipitation_coef_lag_9month": Parameter(Types.REAL, "Precipitation: monthly cumulative precip 9 month lag"),
        "precipitation_coef_lag_1_5day": Parameter(Types.REAL, "Precipitation: 5-day max precip 1 month lag"),

    }

    # Define properties of individuals (none)
    PROPERTIES = {
    }

    def __init__(
        self,
        name: Optional[str] = None,
        resourcefilepath: Optional[Path] = None,
        climate_ssp: Optional[str] = None,
        climate_model_ensemble_model: Optional[str] = None,
        year_effective_climate_disruptions: Optional[int] = None,
        services_affected_precip: Optional[str] = None,
        response_to_disruption: Optional[str] = None,
        delay_in_seeking_care_weather: Optional[float] = None,
        scale_factor_reseeking_healthcare_post_disruption: Optional[float] = None,
        scale_factor_prob_disruption: Optional[float] = None,
        scale_factor_severity_disruption_and_delay: Optional[float] = None,
        prop_supply_side_disruptions: Optional[float] = None,
    ):
        """
        Initialize the ClimateDisruptions module.

        :param name: Name to use for this module
        :param resourcefilepath: Path to directory containing resource files
        :param climate_ssp: Climate scenario. Options are ssp126, ssp245, ssp585
        :param climate_model_ensemble_model: Which model from the model ensemble for each climate ssp
            is under consideration. Options are 'lowest', 'mean', and 'highest', based on total
            precipitation between 2025 and 2070.
        :param year_effective_climate_disruptions: Minimum year from which climate disruptions occur.
            Minimum is 2025.
        :param services_affected_precip: Which modelled services can be affected by weather.
            Options are 'all', 'none'.
        :param response_to_disruption: How an appointment that is determined to be affected by weather
            will be handled. Options are 'delay', 'cancel'.
        :param delay_in_seeking_care_weather: The scale factor on number of days delay in reseeking
            healthcare after an appointment has been delayed by weather. Unit is day.
        :param scale_factor_reseeking_healthcare_post_disruption: Rescaling of probability of seeking
            care after a disruption has occurred.
        :param scale_factor_prob_disruption: To account for structural/behavioural assumptions in the TLO
            and limitations of DHIS2 dataset.
        :param scale_factor_severity_disruption_and_delay: Scale on the delay in reseeking healthcare
            based on the "severity" of disruption.
        :param prop_supply_side_disruptions: Probability that the climate-mediated disruptions to
            healthcare are from the supply-side.
        """
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # Store parameters
        self.arg_climate_ssp = climate_ssp
        self.arg_climate_model_ensemble_model = climate_model_ensemble_model
        self.arg_year_effective = year_effective_climate_disruptions
        self.arg_services_affected_precip = services_affected_precip
        self.arg_response_to_disruption = response_to_disruption
        self.arg_delay_in_seeking_care = delay_in_seeking_care_weather
        self.arg_scale_factor_reseeking = scale_factor_reseeking_healthcare_post_disruption
        self.arg_scale_factor_prob_disruption = scale_factor_prob_disruption
        self.arg_scale_factor_severity = scale_factor_severity_disruption_and_delay
        self.arg_prop_supply_side = prop_supply_side_disruptions

        # Counters for logging
        self._disruptions_cancelled_count = 0
        self._disruptions_delayed_count = 0
        self._disruptions_checked_count = 0
        self._supply_side_disruptions_count = 0
        self._demand_side_disruptions_count = 0

        self._disruptions_by_district = {}
        self._cancelled_by_district = {}
        self._delayed_by_district = {}
        self._supply_side_by_district = {}
        self._demand_side_by_district = {}

    def read_parameters(self, resourcefilepath: str | Path) -> None:
        p = self.parameters
        #Override with values from scenario files
        if self.arg_climate_ssp is not None:
            self.parameters["climate_ssp"] = self.arg_climate_ssp
        if self.arg_climate_model_ensemble_model is not None:
            self.parameters["climate_model_ensemble_model"] = self.arg_climate_model_ensemble_model
        if self.arg_year_effective is not None:
            self.parameters["year_effective_climate_disruptions"] = self.arg_year_effective
        if self.arg_services_affected_precip is not None:
            self.parameters["services_affected_precip"] = self.arg_services_affected_precip
        if self.arg_response_to_disruption is not None:
            self.parameters["response_to_disruption"] = self.arg_response_to_disruption
        if self.arg_delay_in_seeking_care is not None:
            self.parameters["delay_in_seeking_care_weather"] = self.arg_delay_in_seeking_care
        if self.arg_scale_factor_reseeking is not None:
            self.parameters["scale_factor_reseeking_healthcare_post_disruption"] = self.arg_scale_factor_reseeking
        if self.arg_scale_factor_prob_disruption is not None:
            self.parameters["scale_factor_prob_disruption"] = self.arg_scale_factor_prob_disruption
        if self.arg_scale_factor_severity is not None:
            self.parameters["scale_factor_severity_disruption_and_delay"] = self.arg_scale_factor_severity
        if self.arg_prop_supply_side is not None:
            self.parameters["prop_supply_side_disruptions"] = self.arg_prop_supply_side

        # Read in parameters
        self.load_parameters_from_dataframe(
            read_csv_files(resourcefilepath / 'ResourceFile_WeatherDisruption', files='parameter_values')
        )

        # Validate year
        if self.parameters["year_effective_climate_disruptions"] < 2025:
            logger.warning(
                key="message",
                data=f"year_effective set to {self.parameters['year_effective_climate_disruptions']}, minimum is 2025. Setting to 2025."
            )
            self.parameters["year_effective_climate_disruptions"] = 2025

        # Load precipitation data

        ssp = self.parameters["climate_ssp"]
        model = self.parameters["climate_model_ensemble_model"]
        service = self.parameters["services_affected_precip"]

        # Load five-day and monthly precipitation
        precip_5day = read_csv_files(resourcefilepath / 'ResourceFile_WeatherDisruption', files= f"ResourceFile_Precipitation_Disruptions_{ssp}_{model}_window_prediction_weather_by_facility.csv")
        monthly_file = read_csv_files(resourcefilepath / 'ResourceFile_WeatherDisruption', files= f"ResourceFile_Precipitation_Disruptions_{ssp}_{model}_monthly_prediction_weather_by_facility.csv")

        precip_5day = precip_5day.iloc[:, 1:]
        precip_monthly = monthly_file.iloc[:, 1:]

        # Remove zero-sum columns
        zero_cols_monthly = precip_monthly.columns[precip_monthly.sum() == 0]
        zero_cols_5day = precip_5day.columns[precip_5day.sum() == 0]

        precip_monthly = precip_monthly.drop(columns=zero_cols_monthly)
        precip_5day = precip_5day.drop(columns=zero_cols_5day)

        self.parameters["precipitation_data_monthly"] = precip_monthly
        self.parameters["precipitation_data_five_day"] = precip_5day

        # Load facility characteristics
        self.parameters["facility_characteristics"] = read_csv_files(resourcefilepath / 'ResourceFile_WeatherDisruption', files= f"ResourceFile_Facility_Characteristics.csv")

    def pre_initialise_population(self):
        """Pre-initialization (not needed for this module)."""
        pass

    def initialise_population(self, population):
        """Population initialization (no properties added)."""
        super().initialise_population(population=population)


    def initialise_simulation(self, sim):
        """Build disruption predictions at simulation start."""
        # Calculate disruptions
        self.build_disruption_probabilities()

        # Log configuration

        # Schedule monthly logger
        first_log_date = (sim.date + DateOffset(months=1)).replace(day=1) - DateOffset(days=1)
        sim.schedule_event(WeatherDisruptionsMonthlyLogger(self), first_log_date)


    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        """
        pass


    def on_simulation_end(self):
        pass


    def build_linear_models(self):
        """Build the baseline and precipitation linear models using TLO's LinearModel class."""
        p = self.parameters
        print(type(p['baseline_coef_min_distance']))
        print(p['baseline_coef_min_distance'])
        # ===== BASELINE MODEL (no weather) =====
        self.lm_baseline = LinearModel(
            LinearModelType.ADDITIVE,
            0,
            Predictor('year', external=True).apply(lambda x: p['baseline_coef_year']*x),
            #Predictor('month', external=True).apply(lambda x: p['baseline_coef_month']*x),
            Predictor('min_distance_to_clinic', external=True).apply(lambda x: p['baseline_coef_min_distance']* np.array(x)),
            Predictor('altitude', external=True).apply(lambda x: p['baseline_coef_altitude']*x),
            Predictor('urban_rural').when('urban', p['baseline_coef_urban']).otherwise(0.0),
            Predictor('zone', conditions_are_mutually_exclusive=True)
            .when('Central West', p['baseline_coef_central_west'])
            .when('Northern', p['baseline_coef_northern'])
            .when('South East', p['baseline_coef_south_east'])
            .when('South West', p['baseline_coef_south_west'])
            .otherwise(0.0),  # Central East is reference
            Predictor('ownership', conditions_are_mutually_exclusive=True)
            .when('Government', p['baseline_coef_government'])
            .when('Private', p['baseline_coef_private'])
            .otherwise(0.0),  # CHAM is reference
        )

        # ===== PRECIPITATION MODEL (with weather) =====
        self.lm_precipitation = LinearModel(
            LinearModelType.ADDITIVE,
            0,
            Predictor('year', external=True).apply(lambda x: p['precipitation_coef_year']*x),
            #Predictor('month', external=True).apply(lambda x: p['precipitation_coef_month']*x),
            Predictor('min_distance_to_clinic', external=True).apply(lambda x: p['precipitation_coef_min_distance']*x),
            Predictor('altitude', external=True).apply(lambda x: p['precipitation_coef_altitude']*x),
            Predictor('urban_rural').when('urban', p['precipitation_coef_urban']).otherwise(0.0),
            Predictor('zone', conditions_are_mutually_exclusive=True)
            .when('Central West', p['precipitation_coef_central_west'])
            .when('Northern', p['precipitation_coef_northern'])
            .when('South East', p['precipitation_coef_south_east'])
            .when('South West', p['precipitation_coef_south_west'])
            .otherwise(0.0),
            Predictor('ownership', conditions_are_mutually_exclusive=True)
            .when('Government', p['precipitation_coef_government'])
            .when('Private', p['precipitation_coef_private'])
            .otherwise(0.0),
            # Precip variables
            Predictor('precip_monthly', external=True).apply(lambda x: p['precipitation_coef_precip_monthly']*x),
            Predictor('precip_5day', external=True).apply(lambda x: p['precipitation_coef_precip_5day']*x),
            Predictor('lag_4month', external=True).apply(lambda x: p['precipitation_coef_lag_4month']*x),
            Predictor('lag_9month', external=True).apply(lambda x: p['precipitation_coef_lag_9month']*x),
            Predictor('lag_1_5day', external=True).apply(lambda x: p['precipitation_coef_lag_1_5day']*x),
        )

    def build_disruption_probabilities(self):
        """
        Calculate service deficits by comparing baseline and precipitation models.
        Uses TLO's LinearModel for consistency with other modules.

        Deficit = exp(Baseline) - exp(Precipitation)
        Positive deficit = appointments lost due to precipitation
        """
        p = self.parameters

        # Build the linear models
        self.build_linear_models()

        # Get data (includes 2024 for lag calculation)
        precip_monthly_full = p["precipitation_data_monthly"]
        precip_5day_full = p["precipitation_data_five_day"]

        # Calculate lag variables on FULL data (including 2024)
        lag_4month_monthly = precip_monthly_full.shift(4)
        lag_9month_monthly = precip_monthly_full.shift(9)
        lag_1_5day = precip_5day_full.shift(1)

        # Skip 2024 (first 12 rows) for predictions - keep only 2025 onwards
        start_idx = 12  # Skip 2024
        precip_monthly = precip_monthly_full.iloc[start_idx:].reset_index(drop=True)
        precip_5day = precip_5day_full.iloc[start_idx:].reset_index(drop=True)
        lag_4month_monthly = lag_4month_monthly.iloc[start_idx:].reset_index(drop=True)
        lag_9month_monthly = lag_9month_monthly.iloc[start_idx:].reset_index(drop=True)
        lag_1_5day = lag_1_5day.iloc[start_idx:].reset_index(drop=True)

        facility_chars = p["facility_characteristics"]
        facility_chars = facility_chars.set_index(facility_chars.columns[0])
        facility_chars = facility_chars.T  # First column is facility ID

        facilities = precip_monthly.columns.tolist()
        n_time = len(precip_monthly)

        # Create time index starting from 2025
        year = 2025
        month = 1
        time_index = []
        for _ in range(n_time):
            time_index.append((year, month))
            month += 1
            if month > 12:
                month, year = 1, year + 1

        # Build list to store all facility-month combinations
        rows = []

        for t_idx, (year, month) in enumerate(time_index):
            for fac_idx, fac in enumerate(facilities):
                if fac not in facility_chars.index:
                    continue

                # Get precipitation values
                precip_m = precip_monthly.iloc[t_idx, fac_idx]
                precip_5d = precip_5day.iloc[t_idx, fac_idx]

                # Get lag values (now properly calculated from 2024 data)
                lag_4m = lag_4month_monthly.iloc[t_idx, fac_idx] if not pd.isna(
                    lag_4month_monthly.iloc[t_idx, fac_idx]) else 0.0
                lag_9m = lag_9month_monthly.iloc[t_idx, fac_idx] if not pd.isna(
                    lag_9month_monthly.iloc[t_idx, fac_idx]) else 0.0
                lag_1_5d = lag_1_5day.iloc[t_idx, fac_idx] if not pd.isna(lag_1_5day.iloc[t_idx, fac_idx]) else 0.0

                # Get facility characteristics
                dist = facility_chars.at[fac, "min_distance_to_clinic"]
                altitude = facility_chars.at[fac, "altitude"]
                urban = facility_chars.at[fac, "urban_rural"]
                zone = facility_chars.at[fac, "zone"]
                owner = facility_chars.at[fac, "ownership"]

                rows.append({
                    'RealFacility_ID': fac,
                    'year': year,
                    'month': month,
                    'min_distance_to_clinic': dist,
                    'altitude': altitude,
                    'urban_rural': urban,
                    'zone': zone,
                    'ownership': owner,
                    'precip_monthly': precip_m,
                    'precip_5day': precip_5d,
                    'lag_4month': lag_4m,
                    'lag_9month': lag_9m,
                    'lag_1_5day': lag_1_5d,
                })

        # Create dataframe with all facility-month-year combinations
        facility_month_df = pd.DataFrame(rows)
        # ===== PREDICT USING LINEAR MODELS =====
        facility_month_df['min_distance_to_clinic'] = pd.to_numeric(
            facility_month_df['min_distance_to_clinic'], errors='coerce'
        )
        facility_month_df['altitude'] = pd.to_numeric(
            facility_month_df['altitude'], errors='coerce'
        )
        # Baseline predictions (no weather)
        log_pred_baseline = self.lm_baseline.predict(
            facility_month_df,
            rng=None,
            year=facility_month_df['year'].values,
            month=facility_month_df['month'].values,
            min_distance_to_clinic=facility_month_df['min_distance_to_clinic'].values,
            altitude=facility_month_df['altitude'].values,
            urban=facility_month_df['urban_rural'].values,
            zone=facility_month_df['zone'].values,
            ownership=facility_month_df['ownership'].values
        )

        # Precipitation predictions (with all weather variables)
        log_pred_precip = self.lm_precipitation.predict(
            facility_month_df,
            rng=None,
            year=facility_month_df['year'].values,
            month=facility_month_df['month'].values,
            min_distance_to_clinic=facility_month_df['min_distance_to_clinic'].values,
            altitude=facility_month_df['altitude'].values,
            precip_monthly=facility_month_df['precip_monthly'].values,
            precip_5day=facility_month_df['precip_5day'].values,
            lag_4month=facility_month_df['lag_4month'].values,
            lag_9month=facility_month_df['lag_9month'].values,
            lag_1_5day=facility_month_df['lag_1_5day'].values,
            urban=facility_month_df['urban_rural'].values,
            zone=facility_month_df['zone'].values,
            ownership=facility_month_df['ownership'].values
        )

        # Convert from log scale
        pred_baseline = np.exp(log_pred_baseline)
        pred_precip = np.exp(log_pred_precip)

        # Calculate deficit (positive = appointments lost)
        deficit = pred_baseline - pred_precip
        # Convert deficit to probability of disruption
        prob_disruption = np.where(
            pred_baseline > 0,
            np.clip(deficit / pred_baseline, 0, 1),
            0
        )
        print(prob_disruption)
        # Add predictions to dataframe
        facility_month_df['service'] = p["services_affected_precip"]
        facility_month_df['disruption'] = prob_disruption
        facility_month_df['pred_baseline'] = pred_baseline
        facility_month_df['pred_precip'] = pred_precip

        # Keep only the columns needed for lookup
        self.parameters["projected_precip_disruptions"] = facility_month_df[[
            'RealFacility_ID', 'year', 'month', 'service', 'disruption',
            'pred_baseline', 'pred_precip'
        ]].copy()



class WeatherDisruptionsMonthlyLogger(RegularEvent, PopulationScopeEventMixin):
        """Monthly logger for weather disruptions."""

        def __init__(self, module: WeatherDisruptions):
            super().__init__(module, frequency=DateOffset(months=1), priority=Priority.END_OF_DAY)

        def apply(self, population):
            """Log monthly statistics overall and by district."""

            # Overall statistics
            logger_summary.info(
                key="weather_disruptions_monthly",
                data={
                    "year": self.sim.date.year,
                    "month": self.sim.date.month,
                    "checked": self.module._disruptions_checked_count,
                    "cancelled": self.module._disruptions_cancelled_count,
                    "delayed": self.module._disruptions_delayed_count,
                    "supply_side": self.module._supply_side_disruptions_count,
                    "demand_side": self.module._demand_side_disruptions_count,
                }
            )

            # District-level statistics
            logger_summary.info(
                key="weather_disruptions_monthly_by_district",
                data={
                    "year": self.sim.date.year,
                    "month": self.sim.date.month,
                    "checked_by_district": dict(self.module._disruptions_by_district),
                    "cancelled_by_district": dict(self.module._cancelled_by_district),
                    "delayed_by_district": dict(self.module._delayed_by_district),
                    "supply_side_by_district": dict(self.module._supply_side_by_district),
                    "demand_side_by_district": dict(self.module._demand_side_by_district),
                }
            )

            self.module.reset_monthly_counters()



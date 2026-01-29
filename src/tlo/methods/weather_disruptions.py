from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from tlo import Date, DateOffset, Module, Parameter, Population, Property, Types, logging
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
    super().__init__(name)
    self.resourcefilepath = resourcefilepath

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

        # Baseline model (no weather) coefficients
        "baseline_coef_intercept": Parameter(Types.REAL, "Baseline model intercept"),
        "baseline_coef_year": Parameter(Types.REAL, "Baseline: year"),
        "baseline_coef_month": Parameter(Types.REAL, "Baseline: month"),
        "baseline_coef_altitude": Parameter(Types.REAL, "Baseline: altitude"),
        "baseline_coef_min_distance": Parameter(Types.REAL, "Baseline: min distance to clinic"),
        "baseline_coef_urban": Parameter(Types.REAL, "Baseline: urban vs rural"),
        "baseline_coef_central_west": Parameter(Types.REAL, "Baseline: Central West zone"),
        "baseline_coef_northern": Parameter(Types.REAL, "Baseline: Northern zone"),
        "baseline_coef_south_east": Parameter(Types.REAL, "Baseline: South East zone"),
        "baseline_coef_south_west": Parameter(Types.REAL, "Baseline: South West zone"),
        "baseline_coef_government": Parameter(Types.REAL, "Baseline: Government ownership"),
        "baseline_coef_private": Parameter(Types.REAL, "Baseline: Private ownership"),

        # Weather model coefficients (includes all baseline + weather variables)
        "precip_coef_intercept": Parameter(Types.REAL, "Weather model intercept"),
        "precip_coef_year": Parameter(Types.REAL, "Weather: year"),
        "precip_coef_month": Parameter(Types.REAL, "Weather: month"),
        "precip_coef_altitude": Parameter(Types.REAL, "Weather: altitude"),
        "precip_coef_min_distance": Parameter(Types.REAL, "Weather: min distance"),
        "precip_coef_urban": Parameter(Types.REAL, "Weather: urban vs rural"),
        "precip_coef_central_west": Parameter(Types.REAL, "Weather: Central West zone"),
        "precip_coef_northern": Parameter(Types.REAL, "Weather: Northern zone"),
        "precip_coef_south_east": Parameter(Types.REAL, "Weather: South East zone"),
        "precip_coef_south_west": Parameter(Types.REAL, "Weather: South West zone"),
        "precip_coef_government": Parameter(Types.REAL, "Weather: Government ownership"),
        "precip_coef_private": Parameter(Types.REAL, "Weather: Private ownership"),
        "precip_coef_precip_monthly": Parameter(Types.REAL, "Weather: monthly cumulative precip"),
        "precip_coef_precip_5day": Parameter(Types.REAL, "Weather: 5-day max precip"),
        "precip_coef_precip_monthly_sq": Parameter(Types.REAL, "Weather: monthly precip squared"),
        "precip_coef_precip_5day_sq": Parameter(Types.REAL, "Weather: 5-day precip squared"),
        "precip_coef_precip_monthly_cube": Parameter(Types.REAL, "Weather: monthly precip cubed"),
        "precip_coef_precip_5day_cube": Parameter(Types.REAL, "Weather: 5-day precip cubed"),
        "precip_coef_precip_interaction": Parameter(Types.REAL, "Weather: monthly * 5day interaction"),
        "precip_coef_lag_1month": Parameter(Types.REAL, "Weather: 1-month lag"),
        "precip_coef_lag_2month": Parameter(Types.REAL, "Weather: 2-month lag"),
        "precip_coef_lag_3month": Parameter(Types.REAL, "Weather: 3-month lag"),
        "precip_coef_lag_4month": Parameter(Types.REAL, "Weather: 4-month lag"),
        "precip_coef_lag_9month": Parameter(Types.REAL, "Weather: 9-month lag"),
        "precip_coef_lag_1_5day": Parameter(Types.REAL, "Weather: 1-month lag 5-day max"),
        "precip_coef_lag_2_5day": Parameter(Types.REAL, "Weather: 2-month lag 5-day max"),
        "precip_coef_lag_3_5day": Parameter(Types.REAL, "Weather: 3-month lag 5-day max"),
        "precip_coef_lag_4_5day": Parameter(Types.REAL, "Weather: 4-month lag 5-day max"),
        "precip_coef_lag_9_5day": Parameter(Types.REAL, "Weather: 9-month lag 5-day max"),

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


    def read_parameters(self, resourcefilepath: str | Path) -> None:
        p = self.parameters
        # Read in climate disruption files
        self.parameters["projected_precip_disruptions"] = pd.read_csv(
            resourcefilepath
            / f'ResourceFile_Precipitation_Disruptions_{self.parameters["climate_ssp"]}_{self.parameters["climate_model_ensemble_model"]}.csv'
        )

        self.load_parameters_from_dataframe(
            read_csv_files(resourcefilepath / 'ResourceFile_Climate_Disruptions', files='parameter_values')
        )

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

        # Validate year
        if self.parameters["year_effective_climate_disruptions"] < 2025:

            self.parameters["year_effective_climate_disruptions"] = 2025

        # Read in precip files etc to later be used in linear model

        # Validate year
        if self.parameters["year_effective_climate_disruptions"] < 2025:
            logger.warning(
                key="message",
                data=f"year_effective set to {self.parameters['year_effective_climate_disruptions']}, minimum is 2025. Setting to 2025."
            )
            self.parameters["year_effective_climate_disruptions"] = 2025

        # Load precipitation data

        model = self.parameters["climate_model_ensemble_model"]
        service = self.parameters["services_affected_precip"]

        # Load five-day and monthly precipitation
        five_day_file = self.resourcefilepath / f"{model}_window_prediction_weather_by_facility_{service}.csv"
        monthly_file = self.resourcefilepath / f"{model}_monthly_prediction_weather_by_facility_{service}.csv"

        precip_5day = pd.read_csv(five_day_file).drop(columns=[pd.read_csv(five_day_file).columns[0]])
        precip_monthly = pd.read_csv(monthly_file).drop(columns=[pd.read_csv(monthly_file).columns[0]])

        # Remove zero-sum columns
        zero_cols_monthly = precip_monthly.columns[precip_monthly.sum() == 0]
        zero_cols_5day = precip_5day.columns[precip_5day.sum() == 0]

        precip_monthly = precip_monthly.drop(columns=zero_cols_monthly)
        precip_5day = precip_5day.drop(columns=zero_cols_5day)

        self.parameters["precipitation_data_monthly"] = precip_monthly
        self.parameters["precipitation_data_five_day"] = precip_5day

        # Load facility characteristics
        self.parameters["facility_characteristics"] = pd.read_csv( self.resourcefilepath / "ResourceFile_Facility_Characteristics.csv"
        )


    def pre_initialise_population(self) -> None:
        pass

    def initialise_population(self, population):


        pass

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.

        :param mother_id: The person ID of the mother
        :param child_id: The person ID of the newborn
        """
        pass


    def on_simulation_end(self):
        pass

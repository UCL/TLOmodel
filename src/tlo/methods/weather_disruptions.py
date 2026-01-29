from pathlib import Path
from tlo.methods.hsi_event import (
    LABEL_FOR_MERGED_FACILITY_LEVELS_1B_AND_2,
    FacilityInfo,
    HSI_Event,
    HSIEventDetails,
    HSIEventQueueItem,
    HSIEventWrapper,
)
from typing import Dict, Optional

from tlo.util import read_csv_files
from tlo import Date, DateOffset, Module, Parameter, Population, Property, Types, logging

import pandas as pd

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

        # Declare dependencies
        INIT_DEPENDENCIES = {"Demography", "HealthSeekingBehaviour"}

        OPTIONAL_INIT_DEPENDENCIES = {"HealthSystem"}

        # Declare Metadata
        METADATA = {}

        # Define module-level parameters

        PARAMETERS = {"projected_precip_disruptions": Parameter(
                Types.REAL, "Probabilities of precipitation-mediated " "disruptions to services by month, year, and clinic."
            ),
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
            "year_effective_climate_disruptions": Parameter(Types.INT, "Mimimum year from which there can be climate disruptions. Minimum is 2025"),
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
            "scale_factor_prob_disruption": Parameter(
                Types.REAL,
                "Due to uknown behaviours (from patient and health practiciion), broken chains of events, etc, which cause discrepencies  "
                "between the estimated disruptions and those modelled in TLO, rescale the original probability of disruption.",
            ),
            "scale_factor_appointment_urgency": Parameter(
                Types.REAL,
                "Scale factor in seeking healthcare for how urgent a HSI is."
            ),
            "services_affected_precip": Parameter(
                Types.STRING, "Which modelled services can be affected by weather. Options are all, none"
            ),
            "scale_factor_severity_disruption_and_delay": Parameter(
                Types.REAL,
                "Scale factor that changes the delay in reseeking healthcare to the severity of disruption (as measured by probability of disruption)",
            ),
            "prop_supply_side_disruptions": Parameter(
                Types.REAL,
                "Probability that a climate disruption is supply-side (consumes capabilities in mode 2) "
                "vs demand-side (frees up capabilities in mode 2)."
            )}

        # Define properties of individuals (none)
        PROPERTIES = {}

    def read_parameters(self, resourcefilepath: str | Path) -> None:
        p = self.parameters
        # Read in climate disruption files
        path_to_resourcefiles_for_climate = resourcefilepath / "climate_change_impacts"
        self.parameters["projected_precip_disruptions"] = pd.read_csv(
            resourcefilepath
            / f'ResourceFile_Precipitation_Disruptions_{self.parameters["climate_ssp"]}_{self.parameters["climate_model_ensemble_model"]}.csv'
        )

        self.load_parameters_from_dataframe(
            read_csv_files(resourcefilepath / 'ResourceFile_Climate_Disruptions', files='parameter_values')
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

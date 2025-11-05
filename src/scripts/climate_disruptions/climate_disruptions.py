from pathlib import Path
from tlo import Parameter, Types, logging


import numpy as np
import pandas as pd

from tlo import logging

logger = logging.getLogger("tlo.methods.healthsystem")
logger_summary = logging.getLogger("tlo.methods.healthsystem.summary")


class Climate_Disruptions:
    """This is the Climate Disruptions Class. It determines whether a particular HSI event is delayed or cancelled due to
    weather.

    :param climate_ssp: Which future shared socioeconomic pathway (determines degree of warming) is under consideration.
                Options are ssp126, ssp245, and ssp585, in terms of increasing severity.

    :param climate_model_ensemble_model: Which model from the model ensemble for each climate ssp is under consideratin.
                Options are 'lowest', 'mean', and 'highest', based on total precipitation between 2025 and 2070.

    :param services_affected_precip: Which modelled services can be affected by weather. Options are 'all', 'none'

    :param response_to_disruption: How an appointment that is determined to be affected by weather will be handled. Options are 'delay', 'cancel'

    :param delay_in_seeking_care_weather: The number of weeks' delay in reseeking healthcare after an appointmnet has been delayed by weather. Unit is week.
    """

    PARAMETERS = {
        # Probability of climate disruption
        "projected_precip_disruptions": Parameter(
            Types.REAL, "Probabilities of precipitation-mediated " "disruptions to services by month, year, and clinic."
        ),
    }

    def __init__(
        self,
        climate_ssp: str = None,
        climate_model_ensemble_model: pd.DataFrame = None,
        services_affected_precip: str = "default",
        delay_in_seeking_care_weather: int = 4,
    ) -> None:
        self._climate_ssp = {"none", "ssp126", "ssp245", "ssp585"}
        self._climate_model_ensemble_model = {"low", "medium", "high"}

        self._services_affected_precip = {"none", "all"}

        # Create internal items:
        self._climate_ssp = climate_ssp
        self._climate_model_ensemble_model = climate_model_ensemble_model
        self._services_affected_precip = services_affected_precip
        self._delay_in_seeking_care_weather = delay_in_seeking_care_weather

        path_to_resourcefiles_for_climate = Path(self.resourcefilepath) / "climate_change_impacts"
        self.parameters["projected_precip_disruptions"] = pd.read_csv(
            path_to_resourcefiles_for_climate
            / f'ResourceFile_Precipitation_Disruptions_{self.parameters["climate_ssp"]}_{self.parameters["climate_model_ensemble_model"]}.csv'
        )

    def check_if_hsi_experiences_climate_disruption(self, sim, hsi_item):
        year = sim.date.year
        month = sim.date.month
        if (
            year > 2025
            and self.module.parameters["services_affected_precip"] != "none"
            and self.module.parameters["services_affected_precip"] is not None
        ):
            assert self.module.parameters["services_affected_precip"] == "all"
            fac_level = hsi_item.hsi_event.facility_info.level
            facility_used = sim.population.props.at[hsi_item.hsi_event.target, f"level_{fac_level}"]
            if facility_used in self.module.parameters["projected_precip_disruptions"]["RealFacility_ID"].values:
                prob_disruption = self.module.parameters["projected_precip_disruptions"].loc[
                    (self.module.parameters["projected_precip_disruptions"]["RealFacility_ID"] == facility_used)
                    & (self.module.parameters["projected_precip_disruptions"]["year"] == year)
                    & (self.module.parameters["projected_precip_disruptions"]["month"] == month)
                    & (
                        self.module.parameters["projected_precip_disruptions"]["service"]
                        == self.module.parameters["services_affected_precip"]
                    ),
                    "disruption",
                ]
                prob_disruption = pd.DataFrame(prob_disruption)
                prob_disruption = float(prob_disruption.iloc[0])
                if np.random.binomial(1, prob_disruption) == 1:
                    climate_disrupted = True
        return climate_disrupted

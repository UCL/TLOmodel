from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from tlo import Date, DateOffset, Module, Parameter, Types, logging
from tlo.events import PopulationScopeEventMixin, Priority, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.hsi_event import HSIEventQueueItem
from tlo.util import read_csv_files

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger_summary = logging.getLogger(f"{__name__}.summary")
logger_summary.setLevel(logging.INFO)

# HSI modules with hard clinical deadlines — delay must never push topen past tclose
TIME_SENSITIVE_MODULES = {'Labour', 'PostnatalCare', 'PregnancySupervisor', 'CareOfWomenDuringPregnancy'}


class WeatherDisruptions(Module):
    """
    Module to handle climate-mediated disruptions to healthcare services.
    Manages weather-related service interruptions and their impact on
    healthcare seeking behavior and appointment scheduling.
    """

    INIT_DEPENDENCIES = {"Demography", "HealthSeekingBehaviour"}
    OPTIONAL_INIT_DEPENDENCIES = {"HealthSystem"}
    METADATA = {Metadata.USES_HEALTHSYSTEM}

    PARAMETERS = {
        "projected_precip_disruptions": Parameter(
            Types.REAL, "Probabilities of precipitation-mediated disruptions to services by month, year, and clinic."
        ),
        "scale_factor_prob_disruption": Parameter(
            Types.REAL,
            "Due to unknown behaviours (from patient and health practitioner), broken chains of events, etc, which "
            "cause discrepancies between the estimated disruptions and those modelled in TLO, rescale the original "
            "probability of disruption.",
        ),
        "climate_ssp": Parameter(
            Types.STRING,
            "Which future shared socioeconomic pathway (determines degree of warming) is under consideration. "
            "Options are ssp126, ssp245, and ssp585, in terms of increasing severity.",
        ),
        "climate_model_ensemble_model": Parameter(
            Types.STRING,
            "Which model from the model ensemble for each climate ssp is under consideration. "
            "Options are lowest, mean, and highest, based on total precipitation between 2025 and 2070.",
        ),
        "year_effective_climate_disruptions": Parameter(
            Types.INT, "Minimum year from which there can be climate disruptions. Minimum is 2025"
        ),
        "services_affected_precip": Parameter(
            Types.STRING, "Which modelled services can be affected by weather. Options are all, none"
        ),
        "delay_in_seeking_care_weather": Parameter(
            Types.REAL,
            "If faced with a climate disruption, and it is determined the individual will "
            "reseek healthcare, the number of days of delay in seeking healthcare. "
            "Scale factor makes it proportional to the urgency.",
        ),
        "scale_factor_reseeking_healthcare_post_disruption": Parameter(
            Types.REAL,
            "If faced with a climate disruption, and it is determined the individual will "
            "reseek healthcare, scaling of their original probability of seeking care.",
        ),
        "scale_factor_appointment_urgency": Parameter(
            Types.REAL, "Scale factor in seeking healthcare for how urgent a HSI is."
        ),
        "scale_factor_severity_disruption_and_delay": Parameter(
            Types.REAL,
            "Scale factor that changes the delay in reseeking healthcare to the severity of disruption "
            "(as measured by probability of disruption)",
        ),
        "prop_supply_side_disruptions": Parameter(
            Types.REAL,
            "Probability that a climate disruption is supply-side (consumes capabilities in mode 2) "
            "vs demand-side (frees up capabilities in mode 2)."
        ),
        # Baseline model coefficients
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
        "precipitation_coef_south_west": Parameter(Types.REAL, "Precipitation: South West zone"),
        "precipitation_coef_government": Parameter(Types.REAL, "Precipitation: Government ownership"),
        "precipitation_coef_private": Parameter(Types.REAL, "Precipitation: Private ownership"),
        "precipitation_coef_precip_monthly": Parameter(Types.REAL, "Precipitation: monthly cumulative precip"),
        "precipitation_coef_precip_5day": Parameter(Types.REAL, "Precipitation: 5-day cumulative max precip"),
        "precipitation_coef_lag_4month": Parameter(Types.REAL, "Precipitation: monthly cumulative precip 4 month lag"),
        "precipitation_coef_lag_9month": Parameter(Types.REAL, "Precipitation: monthly cumulative precip 9 month lag"),
        "precipitation_coef_lag_1_5day": Parameter(Types.REAL, "Precipitation: 5-day max precip 1 month lag"),
    }

    PROPERTIES = {}

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
        scale_factor_appointment_urgency: Optional[float] = None,  # ← add
        prop_supply_side_disruptions: Optional[float] = None,
    ):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # Constructor arguments are parked here so they can override CSV defaults in read_parameters
        self.arg_climate_ssp = climate_ssp
        self.arg_climate_model_ensemble_model = climate_model_ensemble_model
        self.arg_year_effective = year_effective_climate_disruptions
        self.arg_services_affected_precip = services_affected_precip
        self.arg_response_to_disruption = response_to_disruption
        self.arg_delay_in_seeking_care = delay_in_seeking_care_weather
        self.arg_scale_factor_reseeking = scale_factor_reseeking_healthcare_post_disruption
        self.arg_scale_factor_prob_disruption = scale_factor_prob_disruption
        self.arg_scale_factor_severity = scale_factor_severity_disruption_and_delay
        self.arg_scale_factor_appointment_urgency = scale_factor_appointment_urgency
        self.arg_prop_supply_side = prop_supply_side_disruptions

        self._reset_monthly_counters_internal()

    def _reset_monthly_counters_internal(self):
        """Initialise or reset all monthly counters (overall, per-district, per-facility×treatment)."""
        # Overall scalars
        self._disruptions_cancelled_count = 0
        self._disruptions_delayed_count = 0
        self._disruptions_hsi_total_count = 0
        self._supply_side_disruptions_count = 0
        self._demand_side_disruptions_count = 0

        # Per-district counters (district_name -> int)
        self._hsi_total_by_district: Dict[str, int] = {}
        self._cancelled_by_district: Dict[str, int] = {}
        self._delayed_by_district: Dict[str, int] = {}
        self._supply_side_by_district: Dict[str, int] = {}
        self._demand_side_by_district: Dict[str, int] = {}

        # Per-facility × per-treatment counters ("facility_id|TREATMENT_ID" -> int)
        self._hsi_total_by_facility_treatment: Dict[str, int] = {}
        self._cancelled_by_facility_treatment: Dict[str, int] = {}
        self._delayed_by_facility_treatment: Dict[str, int] = {}
        self._supply_side_by_facility_treatment: Dict[str, int] = {}
        self._demand_side_by_facility_treatment: Dict[str, int] = {}

    def _increment_district_counter(self, counter_dict: Dict, district: str, n: int = 1):
        counter_dict[district] = counter_dict.get(district, 0) + n

    def _increment_facility_treatment_counter(
        self, counter_dict: Dict, facility_id: str, treatment_id: str, n: int = 1
    ):
        key = f"{facility_id}|{treatment_id}"
        counter_dict[key] = counter_dict.get(key, 0) + n

    def read_parameters(self, resourcefilepath) -> None:
        # Constructor arguments override CSV defaults where provided
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
        if self.arg_scale_factor_appointment_urgency is not None:
            self.parameters["scale_factor_appointment_urgency"] = self.arg_scale_factor_appointment_urgency
        if self.arg_prop_supply_side is not None:
            self.parameters["prop_supply_side_disruptions"] = self.arg_prop_supply_side

        self.load_parameters_from_dataframe(
            read_csv_files(resourcefilepath / 'ResourceFile_WeatherDisruption', files='parameter_values')
        )

        if self.parameters["year_effective_climate_disruptions"] < 2025:
            logger.warning(
                key="message",
                data=f"year_effective set to {self.parameters['year_effective_climate_disruptions']}, "
                     f"minimum is 2025. Setting to 2025."
            )
            self.parameters["year_effective_climate_disruptions"] = 2025

        ssp = self.parameters["climate_ssp"]
        model = self.parameters["climate_model_ensemble_model"]

        precip_5day = read_csv_files(
            resourcefilepath / 'ResourceFile_WeatherDisruption',
            files=f"ResourceFile_Precipitation_Disruptions_{ssp}_{model}_window_prediction_weather_by_facility.csv"
        )
        monthly_file = read_csv_files(
            resourcefilepath / 'ResourceFile_WeatherDisruption',
            files=f"ResourceFile_Precipitation_Disruptions_{ssp}_{model}_monthly_prediction_weather_by_facility.csv"
        )

        precip_5day = precip_5day.iloc[:, 1:]
        precip_monthly = monthly_file.iloc[:, 1:]

        precip_monthly = precip_monthly.drop(columns=precip_monthly.columns[precip_monthly.sum() == 0])
        precip_5day = precip_5day.drop(columns=precip_5day.columns[precip_5day.sum() == 0])

        self.parameters["precipitation_data_monthly"] = precip_monthly
        self.parameters["precipitation_data_five_day"] = precip_5day

        self.parameters["facility_characteristics"] = read_csv_files(
            resourcefilepath / 'demography',
            files="ResourceFile_Facility_Characteristics.csv"
        )

    def pre_initialise_population(self):
        pass

    def initialise_population(self, population):
        super().initialise_population(population=population)

    def initialise_simulation(self, sim):
        self.build_disruption_probabilities()
        sim.schedule_event(
            WeatherDisruptionsMonthlyLogger(self),
            Date(sim.date.year, 1, 1) + DateOffset(months=1)
        )
    def on_birth(self, mother_id, child_id):
        pass

    def on_simulation_end(self):
        logger_summary.info(
            key="weather_disruptions_final",
            data={
                "year": self.sim.date.year,
                "month": self.sim.date.month,
                "hsi_total": self._disruptions_hsi_total_count,
                "cancelled": self._disruptions_cancelled_count,
                "delayed": self._disruptions_delayed_count,
                "supply_side": self._supply_side_disruptions_count,
                "demand_side": self._demand_side_disruptions_count,
            }
        )
        logger_summary.info(
            key="weather_disruptions_final_by_district",
            data={
                "year": self.sim.date.year,
                "month": self.sim.date.month,
                "hsi_total_by_district": dict(self._hsi_total_by_district),
                "cancelled_by_district": dict(self._cancelled_by_district),
                "delayed_by_district": dict(self._delayed_by_district),
                "supply_side_by_district": dict(self._supply_side_by_district),
                "demand_side_by_district": dict(self._demand_side_by_district),
            }
        )
        logger_summary.info(
            key="weather_disruptions_final_by_facility_treatment",
            data={
                "year": self.sim.date.year,
                "month": self.sim.date.month,
                "hsi_total_by_facility_treatment": dict(self._hsi_total_by_facility_treatment),
                "cancelled_by_facility_treatment": dict(self._cancelled_by_facility_treatment),
                "delayed_by_facility_treatment": dict(self._delayed_by_facility_treatment),
                "supply_side_by_facility_treatment": dict(self._supply_side_by_facility_treatment),
                "demand_side_by_facility_treatment": dict(self._demand_side_by_facility_treatment),
            }
        )

    def reset_monthly_counters(self):
        self._reset_monthly_counters_internal()

    # -------------------------------------------------------------------------
    # Linear model construction
    # -------------------------------------------------------------------------

    def build_linear_models(self):
        p = self.parameters

        self.lm_baseline = LinearModel(
            LinearModelType.ADDITIVE,
            0,
            Predictor('year', external=True).apply(lambda x: p['baseline_coef_year'] * x),
            Predictor('min_distance_to_clinic', external=True).apply(
                lambda x: p['baseline_coef_min_distance'] * np.array(x)),
            Predictor('altitude', external=True).apply(lambda x: p['baseline_coef_altitude'] * x),
            Predictor('urban_rural').when('urban', p['baseline_coef_urban']).otherwise(0.0),
            Predictor('zone', conditions_are_mutually_exclusive=True)
            .when('Central West', p['baseline_coef_central_west'])
            .when('Northern', p['baseline_coef_northern'])
            .when('South East', p['baseline_coef_south_east'])
            .when('South West', p['baseline_coef_south_west'])
            .otherwise(0.0),
            Predictor('ownership', conditions_are_mutually_exclusive=True)
            .when('Government', p['baseline_coef_government'])
            .when('Private', p['baseline_coef_private'])
            .otherwise(0.0),
        )

        self.lm_precipitation = LinearModel(
            LinearModelType.ADDITIVE,
            0,
            Predictor('year', external=True).apply(lambda x: p['precipitation_coef_year'] * x),
            Predictor('min_distance_to_clinic', external=True).apply(
                lambda x: p['precipitation_coef_min_distance'] * x),
            Predictor('altitude', external=True).apply(lambda x: p['precipitation_coef_altitude'] * x),
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
            Predictor('precip_monthly', external=True).apply(lambda x: p['precipitation_coef_precip_monthly'] * x),
            Predictor('precip_5day', external=True).apply(lambda x: p['precipitation_coef_precip_5day'] * x),
            Predictor('lag_4month', external=True).apply(lambda x: p['precipitation_coef_lag_4month'] * x),
            Predictor('lag_9month', external=True).apply(lambda x: p['precipitation_coef_lag_9month'] * x),
            Predictor('lag_1_5day', external=True).apply(lambda x: p['precipitation_coef_lag_1_5day'] * x),
        )

    def build_disruption_probabilities(self):
        p = self.parameters
        self.build_linear_models()

        precip_monthly_full = p["precipitation_data_monthly"]
        precip_5day_full = p["precipitation_data_five_day"]

        lag_4month_monthly = precip_monthly_full.shift(4)
        lag_9month_monthly = precip_monthly_full.shift(9)
        lag_1_5day = precip_5day_full.shift(1)

        start_idx = 12  # Skip 2024 rows; predictions run from 2025 onwards
        precip_monthly = precip_monthly_full.iloc[start_idx:].reset_index(drop=True)
        precip_5day = precip_5day_full.iloc[start_idx:].reset_index(drop=True)
        lag_4month_monthly = lag_4month_monthly.iloc[start_idx:].reset_index(drop=True)
        lag_9month_monthly = lag_9month_monthly.iloc[start_idx:].reset_index(drop=True)
        lag_1_5day = lag_1_5day.iloc[start_idx:].reset_index(drop=True)

        facility_chars = p["facility_characteristics"].copy()
        facility_chars.index = [
            'zone', 'urban_rural', 'district', 'ownership', 'altitude',
            'facility_type', 'Latitude', 'Longitude', 'min_distance_to_clinic'
        ]
        facility_chars = facility_chars.T
        facilities = precip_monthly.columns.tolist()
        n_time = len(precip_monthly)

        year, month = 2025, 1
        time_index = []
        for _ in range(n_time):
            time_index.append((year, month))
            month += 1
            if month > 12:
                month, year = 1, year + 1

        rows = []
        for t_idx, (year, month) in enumerate(time_index):
            for fac_idx, fac in enumerate(facilities):
                if fac not in facility_chars.index:
                    continue

                precip_m = precip_monthly.iloc[t_idx, fac_idx]
                precip_5d = precip_5day.iloc[t_idx, fac_idx]
                lag_4m = lag_4month_monthly.iloc[t_idx, fac_idx]
                lag_9m = lag_9month_monthly.iloc[t_idx, fac_idx]
                lag_1_5d = lag_1_5day.iloc[t_idx, fac_idx]

                rows.append({
                    'RealFacility_ID': fac,
                    'year': year,
                    'month': month,
                    'min_distance_to_clinic': facility_chars.at[fac, "min_distance_to_clinic"],
                    'altitude': facility_chars.at[fac, "altitude"],
                    'urban_rural': facility_chars.at[fac, "urban_rural"],
                    'zone': facility_chars.at[fac, "zone"],
                    'ownership': facility_chars.at[fac, "ownership"],
                    'precip_monthly': precip_m,
                    'precip_5day': precip_5d,
                    'lag_4month': 0.0 if pd.isna(lag_4m) else lag_4m,
                    'lag_9month': 0.0 if pd.isna(lag_9m) else lag_9m,
                    'lag_1_5day': 0.0 if pd.isna(lag_1_5d) else lag_1_5d,
                })

        facility_month_df = pd.DataFrame(rows)
        facility_month_df['min_distance_to_clinic'] = pd.to_numeric(
            facility_month_df['min_distance_to_clinic'], errors='coerce')
        facility_month_df['altitude'] = pd.to_numeric(facility_month_df['altitude'], errors='coerce')

        log_pred_baseline = self.lm_baseline.predict(
            facility_month_df,
            rng=None,
            year=facility_month_df['year'].values,
            month=facility_month_df['month'].values,
            min_distance_to_clinic=facility_month_df['min_distance_to_clinic'].values,
            altitude=facility_month_df['altitude'].values,
            urban=facility_month_df['urban_rural'].values,
            zone=facility_month_df['zone'].values,
            ownership=facility_month_df['ownership'].values,
        )

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
            ownership=facility_month_df['ownership'].values,
        )

        pred_baseline = np.exp(log_pred_baseline)
        pred_precip = np.exp(log_pred_precip)

        deficit = pred_baseline - pred_precip
        prob_disruption = np.where(
            pred_baseline > 0,
            np.clip(deficit / pred_baseline, 0, 1),
            0,
        )

        facility_month_df['service'] = p["services_affected_precip"]
        facility_month_df['disruption'] = prob_disruption
        facility_month_df['pred_baseline'] = pred_baseline
        facility_month_df['pred_precip'] = pred_precip

        self.parameters["projected_precip_disruptions"] = facility_month_df[[
            'RealFacility_ID', 'year', 'month', 'service', 'disruption',
            'pred_baseline', 'pred_precip'
        ]].copy()

        print(self.parameters["projected_precip_disruptions"])


    def check_hsi_for_disruption(self, hsi_event_item: HSIEventQueueItem, current_date: Date) -> bool:
        """
        Check if an HSI event should be disrupted due to climate/weather conditions.
        Returns True if the event was disrupted (and should not proceed), False otherwise.
        """
        year = current_date.year
        month = current_date.month

        if year < self.parameters["year_effective_climate_disruptions"]:
            return False, False

        if self.parameters["services_affected_precip"] in ("none", None):
            return False, False

        fac_level = hsi_event_item.hsi_event.facility_info.level
        person_id = hsi_event_item.hsi_event.target
        facility_used = self.sim.population.props.at[person_id, f"level_{fac_level}"]

        if facility_used not in self.parameters["projected_precip_disruptions"]["RealFacility_ID"].values:
            return False, False

        prob_disruption = self._get_disruption_probability(
            facility_id=facility_used,
            year=year,
            month=month,
            service=self.parameters["services_affected_precip"],
        )

        if prob_disruption == 0.0 or self.rng.binomial(1, prob_disruption) == 0:
            return False, False

        district = self.sim.population.props.at[person_id, "district_of_residence"]
        treatment_id = getattr(hsi_event_item.hsi_event, "TREATMENT_ID", "unknown")

        is_supply_side = self._handle_disruption(
            hsi_event_item=hsi_event_item,
            prob_disruption=prob_disruption,
            current_date=current_date,
            facility_id=str(facility_used),
            district=str(district),
            treatment_id=str(treatment_id),
        )

        return True, is_supply_side

    def _get_disruption_probability(self, facility_id, year: int, month: int, service: str) -> float:
        """Look up and scale the disruption probability for a given facility/year/month/service."""
        disruption_probs = self.parameters["projected_precip_disruptions"]
        mask = (
            (disruption_probs["RealFacility_ID"] == facility_id)
            & (disruption_probs["year"] == year)
            & (disruption_probs["month"] == month)
            & (disruption_probs["service"] == service)
        )
        prob = disruption_probs.loc[mask, "disruption"]
        if prob.empty:
            return 0.0
        return min(float(prob.iloc[0]) * self.parameters["scale_factor_prob_disruption"], 1.0)

    def _handle_disruption(
        self,
        hsi_event_item: HSIEventQueueItem,
        prob_disruption: float,
        current_date: Date,
        facility_id: str,
        district: str,
        treatment_id: str,
    ):
        """
        Handle a confirmed disruption: determine supply/demand side, reschedule or cancel,
        and update all counters (overall, district, facility×treatment).
        """
        is_supply_side = bool(self.rng.binomial(1, self.parameters["prop_supply_side_disruptions"]))

        if is_supply_side and self.sim.modules["HealthSystem"].mode_appt_constraints == 2:
            clinic = hsi_event_item.clinic_eligibility
            footprint = hsi_event_item.hsi_event.expected_time_requests
            self.sim.modules["HealthSystem"].running_total_footprint[clinic].update(footprint)
            self._supply_side_disruptions_count += 1
            self._increment_district_counter(self._supply_side_by_district, district)
            self._increment_facility_treatment_counter(self._supply_side_by_facility_treatment, facility_id,
                                                       treatment_id)
        else:
            self._demand_side_disruptions_count += 1
            self._increment_district_counter(self._demand_side_by_district, district)
            self._increment_facility_treatment_counter(self._demand_side_by_facility_treatment, facility_id,
                                                       treatment_id)

        will_reschedule = self._determine_rescheduling(hsi_event_item)

        if will_reschedule:
            delay_days = self._calculate_delay(hsi_event_item, prob_disruption, current_date)
            new_topen = current_date + DateOffset(days=delay_days)

            self.sim.modules["HealthSystem"]._add_hsi_event_queue_item_to_hsi_event_queue(
                priority=hsi_event_item.priority,
                clinic_eligibility=hsi_event_item.clinic_eligibility,
                topen=new_topen,
                tclose=new_topen,  # point-in-time reschedule, matching healthsystem.py convention
                hsi_event=hsi_event_item.hsi_event,
            )
            self._log_delayed_hsi(hsi_event_item, facility_id=facility_id)
            self._disruptions_delayed_count += 1
            self._increment_district_counter(self._delayed_by_district, district)
            self._increment_facility_treatment_counter(self._delayed_by_facility_treatment, facility_id, treatment_id)
        else:
            self._log_cancelled_hsi(hsi_event_item, facility_id=facility_id)
            self._disruptions_cancelled_count += 1
            self._increment_district_counter(self._cancelled_by_district, district)
            self._increment_facility_treatment_counter(self._cancelled_by_facility_treatment, facility_id, treatment_id)

        # Overall hsi_total counter — incremented once per disruption regardless of outcome
        self._disruptions_hsi_total_count += 1
        self._increment_district_counter(self._hsi_total_by_district, district)
        self._increment_facility_treatment_counter(self._hsi_total_by_facility_treatment, facility_id, treatment_id)

        return is_supply_side

    def _determine_rescheduling(self, hsi_event_item: HSIEventQueueItem) -> bool:
        """Determine if a person will reschedule their appointment after a disruption."""
        if self.sim.modules["HealthSeekingBehaviour"].force_any_symptom_to_lead_to_healthcareseeking:
            return True

        person_id = hsi_event_item.hsi_event.target
        patient = self.sim.population.props.loc[[person_id]]

        if patient.age_years.iloc[0] < 15:
            subgroup_name = "children"
            care_seeking_odds_ratios = self.sim.modules["HealthSeekingBehaviour"].odds_ratio_health_seeking_in_children
            hsb_model = self.sim.modules["HealthSeekingBehaviour"].hsb_linear_models["children"]
        else:
            subgroup_name = "adults"
            care_seeking_odds_ratios = self.sim.modules["HealthSeekingBehaviour"].odds_ratio_health_seeking_in_adults
            hsb_model = self.sim.modules["HealthSeekingBehaviour"].hsb_linear_models["adults"]

        will_seek_care_prob = min(
            self.parameters["scale_factor_reseeking_healthcare_post_disruption"]
            * hsb_model.predict(
                df=patient,
                subgroup=subgroup_name,
                care_seeking_odds_ratios=care_seeking_odds_ratios,
            ).iloc[0],
            1.0,
        )
        return self.rng.random() < will_seek_care_prob

    def _calculate_delay(
        self,
        hsi_event_item: HSIEventQueueItem,
        prob_disruption: float,
        current_date: Date,
    ) -> int:
        """
        Calculate the delay in days for rescheduling after a disruption.

        Two adjustments are made relative to the basic urgency×severity calculation:

        1. Window width is added — the rescheduled topen is pushed past the end of the
           original appointment window, avoiding overlap with the disrupted slot.

        2. For time-sensitive modules (Labour, PostnatalCare, PregnancySupervisor,
           CareOfWomenDuringPregnancy), the delay is capped so that the new topen
           never exceeds the original tclose. Rescheduling beyond a clinical deadline
           would be meaningless and could cause simulation errors.
        """
        base_delay = int(
            max(self.parameters["scale_factor_appointment_urgency"] * hsi_event_item.priority, 1)
            * prob_disruption
            * self.parameters["scale_factor_severity_disruption_and_delay"]
            * self.parameters["delay_in_seeking_care_weather"]
        )

        # Add the original appointment window width so the reschedule opens after the
        # disrupted window closes rather than potentially overlapping with it
        window_days = (hsi_event_item.tclose - hsi_event_item.topen).days
        delay_days = max(0, base_delay + window_days)

        # For time-sensitive clinical events, cap delay so topen never exceeds tclose
        module_name = hsi_event_item.hsi_event.module.__class__.__name__
        if module_name in TIME_SENSITIVE_MODULES:
            max_allowable_days = max(0, (hsi_event_item.tclose - current_date).days)
            delay_days = min(delay_days, max_allowable_days)

        return delay_days

    def _log_delayed_hsi(self, hsi_event_item: HSIEventQueueItem, facility_id: Optional[str] = None):
        """Log an HSI that was delayed due to weather.
        The event has already been rescheduled — call did_not_run()
        """
        hsi_event_item.hsi_event.did_not_run()

        logger_summary.info(
            key="Weather_delayed_HSI_Event_full_info",
            data={
                "TREATMENT_ID": getattr(hsi_event_item.hsi_event, "TREATMENT_ID", "unknown"),
                "Person_ID": hsi_event_item.hsi_event.target,
                "priority": hsi_event_item.priority,
                "RealFacility_ID": facility_id if facility_id is not None else "unknown",
            },
            description="record of each HSI event delayed due to weather",
        )

    def _log_cancelled_hsi(self, hsi_event_item: HSIEventQueueItem, facility_id: Optional[str] = None):
        """Log an HSI that was cancelled due to weather.
        The event will not be rescheduled — call never_ran().
        """
        person_id = hsi_event_item.hsi_event.target
        if not self.sim.population.props.at[person_id, 'is_alive']:
            return

        hsi_event_item.hsi_event.never_ran()

        logger_summary.info(
            key="Weather_cancelled_HSI_Event_full_info",
            data={
                "TREATMENT_ID": getattr(hsi_event_item.hsi_event, "TREATMENT_ID", "unknown"),
                "Person_ID": person_id,
                "priority": hsi_event_item.priority,
                "RealFacility_ID": facility_id if facility_id is not None else "unknown",
            },
            description="record of each HSI event cancelled due to weather",
        )

class WeatherDisruptionsMonthlyLogger(RegularEvent, PopulationScopeEventMixin):
    """Monthly logger for weather disruptions — overall, by district, and by facility×treatment."""

    def __init__(self, module: WeatherDisruptions):
        super().__init__(module, frequency=DateOffset(months=1), priority=Priority.END_OF_DAY)

    def apply(self, population):
        m = self.module

        # Overall counts
        logger_summary.info(
            key="weather_disruptions_monthly",
            data={
                "year": self.sim.date.year,
                "month": self.sim.date.month,
                "hsi_total": m._disruptions_hsi_total_count,
                "cancelled": m._disruptions_cancelled_count,
                "delayed": m._disruptions_delayed_count,
                "supply_side": m._supply_side_disruptions_count,
                "demand_side": m._demand_side_disruptions_count,
            }
        )

        # By district
        logger_summary.info(
            key="weather_disruptions_monthly_by_district",
            data={
                "year": self.sim.date.year,
                "month": self.sim.date.month,
                "hsi_total_by_district": dict(m._hsi_total_by_district),
                "cancelled_by_district": dict(m._cancelled_by_district),
                "delayed_by_district": dict(m._delayed_by_district),
                "supply_side_by_district": dict(m._supply_side_by_district),
                "demand_side_by_district": dict(m._demand_side_by_district),
            }
        )

        # By facility × treatment (keys: "facility_id|TREATMENT_ID")
        logger_summary.info(
            key="weather_disruptions_monthly_by_facility_treatment",
            data={
                "year": self.sim.date.year,
                "month": self.sim.date.month,
                "hsi_total_by_facility_treatment": dict(m._hsi_total_by_facility_treatment),
                "cancelled_by_facility_treatment": dict(m._cancelled_by_facility_treatment),
                "delayed_by_facility_treatment": dict(m._delayed_by_facility_treatment),
                "supply_side_by_facility_treatment": dict(m._supply_side_by_facility_treatment),
                "demand_side_by_facility_treatment": dict(m._demand_side_by_facility_treatment),
            }
        )

        m.reset_monthly_counters()

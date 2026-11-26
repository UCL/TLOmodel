from pathlib import Path
from typing import Dict, Optional

import re

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
TIME_SENSITIVE_MODULES = {'Labour', 'PregnancySupervisor', 'CareOfWomenDuringPregnancy', 'NewbornOutcomes'}

# Seed TREATMENT_ID -> weather service mapping (overridable via parameter/constructor).
DEFAULT_TREATMENT_SERVICE_MAPPING = {
    "AntenatalCare_Outpatient": "ANC",
    "PostnatalCare_Maternal": "PNC",
    "Contraception_Routine": "FP",
    "FirstAttendance_NonEmergency": "OPD",
    "CervicalCancer_Screening_VIA": "CervicalScreening",
    "CervicalCancer_Screening_Xpert": "CervicalScreening",
}
DEFAULT_WBGT_SERVICE_INDICATORS = {
    "ANC": "anc_total_visits",
    "PNC": "pnc_within_2wks",
    "FP":  "fp_total_clients",
    "OPD": "opd_attendance",
    "CervicalScreening": "cervical_screening_total",
}
WBGT_VAR = "wbgt5x_day"
WBGT_PRECIP_VAR = "precip_month"
WBGT_REFERENCE_YEAR = 2025  # zone_year is held at 2025, as in the projections
MONTH_FE_NAME = "C(month)"
ZONE_YEAR_FE_NAME = "C(zone_year)"

# column name in the fitted model -> column name in ResourceFile_Facility_Characteristics
WBGT_CHARACTERISTIC_COLUMNS = {
    "A105": "ownership",
    "Resid": "urban_rural",
    "A109__Altitude": "altitude",
    "distance_nearest_clinic": "min_distance_to_clinic",
}
TIER_TO_MODEL = {
    "lowest": "MPI-ESM1-2-HR",
    "median": "MIROC6",
    "highest": "CanESM5",
}


def _safe_name(s):
    # same cleaning as in the fitting script, so category names match the coefficient names
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(s)).strip("_")


class WeatherDisruptions(Module):
    """
    Module to handle climate-mediated disruptions to healthcare services.

    Two interchangeable disruption drivers, selected by ``weather_event``:
      * "precip" — baseline and precipitation linear models evaluated from coefficients.
      * "wbgt"   — a per-service WBGT->service response (spline lookup + lag coefficients)
        applied to a WBGT exposure series.
    Both populate ``projected_precip_disruptions`` (keyed by facility, year, month, service),
    so all runtime handling is shared across drivers.
    """

    INIT_DEPENDENCIES = {"Demography", "HealthSeekingBehaviour"}
    OPTIONAL_INIT_DEPENDENCIES = {"HealthSystem"}
    METADATA = {Metadata.USES_HEALTHSYSTEM}

    PARAMETERS = {
        "projected_precip_disruptions": Parameter(
            Types.REAL, "Disruption probabilities by facility, year, month and service."
        ),
        "weather_event": Parameter(
            Types.STRING, "Which climate driver generates disruptions. Options: 'precip', 'wbgt'."
        ),
        "wbgt_response_indicator": Parameter(
            Types.STRING,
            "For the 'wbgt' driver: default fitted indicator's exported WBGT response to apply "
            "when no per-service mapping is given, e.g. 'opd_attendance'."
        ),
        "detailed_disruption_logging": Parameter(
            Types.BOOL,
            "If True, each delayed/cancelled HSI is logged with district, weather_event, supply/demand "
            "flag, the scaled disruption probability, and the weather value and model outputs."
        ),
        "scale_factor_prob_disruption": Parameter(
            Types.REAL, "Rescaling of the original probability of disruption."
        ),
        "climate_ssp": Parameter(
            Types.STRING, "Future SSP under consideration. Options: ssp126, ssp245, ssp585."
        ),
        "climate_model_ensemble_model": Parameter(
            Types.STRING, "Which ensemble member. Options: lowest, mean/median, highest."
        ),
        "year_effective_climate_disruptions": Parameter(
            Types.INT, "Minimum year from which there can be climate disruptions. Minimum is 2026"
        ),
        "services_affected_precip": Parameter(
            Types.STRING, "Master switch for whether services can be affected by weather. Options: all, none"
        ),
        "delay_in_seeking_care_weather": Parameter(
            Types.REAL, "Base number of days of delay in reseeking care after a disruption."
        ),
        "scale_factor_reseeking_healthcare_post_disruption": Parameter(
            Types.REAL, "Scaling of the original probability of seeking care after a disruption."
        ),
        "scale_factor_appointment_urgency": Parameter(
            Types.REAL, "Scale factor in seeking healthcare for how urgent a HSI is."
        ),
        "scale_factor_severity_disruption_and_delay": Parameter(
            Types.REAL, "Scales the delay in reseeking care to the severity of disruption."
        ),
        "prop_supply_side_disruptions": Parameter(
            Types.REAL,
            "Probability that a climate disruption is supply-side (consumes capabilities in mode 2) "
            "vs demand-side (frees up capabilities in mode 2)."
        ),
        "treatment_service_mapping": Parameter(
            Types.DICT, "Maps HSI TREATMENT_ID -> weather service label used for disruption lookup."
        ),
        "wbgt_service_indicators": Parameter(
            Types.DICT,
            "For the 'wbgt' driver: {service_label: exported indicator name}, selecting the "
            "tlo_spline_/tlo_parameters_ files to apply for each service."
        ),
        "default_disruption_service": Parameter(
            Types.STRING,
            "Service label applied to any TREATMENT_ID not in treatment_service_mapping. "
            "If empty/None, unmapped treatments are never disrupted."
        ),
        # Baseline model coefficients (precip driver)
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
        # Precipitation model coefficients (precip driver)
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
        weather_event: Optional[str] = None,
        wbgt_response_indicator: Optional[str] = None,
        detailed_disruption_logging: Optional[bool] = None,
        climate_ssp: Optional[str] = None,
        climate_model_ensemble_model: Optional[str] = None,
        year_effective_climate_disruptions: Optional[int] = None,
        services_affected_precip: Optional[str] = None,
        response_to_disruption: Optional[str] = None,
        delay_in_seeking_care_weather: Optional[float] = None,
        scale_factor_reseeking_healthcare_post_disruption: Optional[float] = None,
        scale_factor_prob_disruption: Optional[float] = None,
        scale_factor_severity_disruption_and_delay: Optional[float] = None,
        scale_factor_appointment_urgency: Optional[float] = None,
        prop_supply_side_disruptions: Optional[float] = None,
        treatment_service_mapping: Optional[dict] = None,
        wbgt_service_indicators: Optional[dict] = None,
        default_disruption_service: Optional[str] = None,
    ):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        self.arg_weather_event = weather_event
        self.arg_wbgt_response_indicator = wbgt_response_indicator
        self.arg_detailed_logging = detailed_disruption_logging
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
        self.arg_treatment_service_mapping = treatment_service_mapping
        self.arg_wbgt_service_indicators = wbgt_service_indicators
        self.arg_default_disruption_service = default_disruption_service

        self._reset_monthly_counters_internal()

    def _reset_monthly_counters_internal(self):
        """Initialise or reset all monthly counters."""
        self._disruptions_cancelled_count = 0
        self._disruptions_delayed_count = 0
        self._disruptions_hsi_total_count = 0
        self._supply_side_disruptions_count = 0
        self._demand_side_disruptions_count = 0

        self._hsi_total_by_district: Dict[str, int] = {}
        self._cancelled_by_district: Dict[str, int] = {}
        self._delayed_by_district: Dict[str, int] = {}
        self._supply_side_by_district: Dict[str, int] = {}
        self._demand_side_by_district: Dict[str, int] = {}

        self._hsi_total_by_facility_treatment: Dict[str, int] = {}
        self._cancelled_by_facility_treatment: Dict[str, int] = {}
        self._delayed_by_facility_treatment: Dict[str, int] = {}
        self._supply_side_by_facility_treatment: Dict[str, int] = {}
        self._demand_side_by_facility_treatment: Dict[str, int] = {}

        self._hsi_checked_count = 0
        self._hsi_checked_unique_count = 0

        self._hsi_checked_by_district: Dict[str, int] = {}
        self._hsi_checked_unique_by_district: Dict[str, int] = {}

        self._hsi_checked_by_facility_treatment: Dict[str, int] = {}
        self._hsi_checked_unique_by_facility_treatment: Dict[str, int] = {}

    def _increment_district_counter(self, counter_dict: Dict, district: str, n: int = 1):
        counter_dict[district] = counter_dict.get(district, 0) + n

    def _increment_facility_treatment_counter(
        self, counter_dict: Dict, facility_id: str, treatment_id: str, n: int = 1
    ):
        key = f"{facility_id}|{treatment_id}"
        counter_dict[key] = counter_dict.get(key, 0) + n

    def read_parameters(self, resourcefilepath) -> None:
        self.load_parameters_from_dataframe(
            read_csv_files(resourcefilepath / 'ResourceFile_WeatherDisruption', files='parameter_values')
        )

        if self.arg_weather_event is not None:
            self.parameters["weather_event"] = self.arg_weather_event
        if self.arg_wbgt_service_indicators is not None:
            self.parameters["wbgt_service_indicators"] = self.arg_wbgt_service_indicators
        elif not self.parameters.get("wbgt_service_indicators"):
            self.parameters["wbgt_service_indicators"] = dict(DEFAULT_WBGT_SERVICE_INDICATORS)
        if self.arg_detailed_logging is not None:
            self.parameters["detailed_disruption_logging"] = self.arg_detailed_logging
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

        if self.arg_treatment_service_mapping is not None:
            self.parameters["treatment_service_mapping"] = self.arg_treatment_service_mapping
        elif not self.parameters.get("treatment_service_mapping"):
            self.parameters["treatment_service_mapping"] = dict(DEFAULT_TREATMENT_SERVICE_MAPPING)

        if self.arg_wbgt_service_indicators is not None:
            self.parameters["wbgt_service_indicators"] = self.arg_wbgt_service_indicators

        if self.arg_default_disruption_service is not None:
            self.parameters["default_disruption_service"] = self.arg_default_disruption_service

        self._treatment_to_service = dict(self.parameters["treatment_service_mapping"])
        _d = self.parameters.get("default_disruption_service")
        self._default_disruption_service = None if _d in ("", None) else _d
        self._services = sorted(
            set(self._treatment_to_service.values())
            | ({self._default_disruption_service} if self._default_disruption_service else set())
        )

        year_param = self.parameters.get("year_effective_climate_disruptions")
        if year_param is None or pd.isna(year_param) or year_param < 2026:
            self.parameters["year_effective_climate_disruptions"] = 2026

        weather_event = self.parameters.get("weather_event", "precip")
        if weather_event not in ("precip", "wbgt"):
            raise ValueError(f"weather_event must be 'precip' or 'wbgt', got {weather_event!r}")

        if weather_event == "precip":
            self._read_precip_inputs(resourcefilepath)
        else:
            self._read_wbgt_inputs(resourcefilepath)

    def _read_precip_inputs(self, resourcefilepath) -> None:
        """Load precipitation coefficients, exposure files and facility characteristics."""
        self.load_parameters_from_dataframe(
            read_csv_files(
                resourcefilepath / 'ResourceFile_WeatherDisruption',
                files='ResourceFile_Precipitation_Coefficients',
            )
        )

        ssp = self.parameters["climate_ssp"]
        model = self.parameters["climate_model_ensemble_model"]

        precip_5day = read_csv_files(
            resourcefilepath / 'ResourceFile_WeatherDisruption',
            files=f"ResourceFile_Precipitation_Disruptions_{ssp}_{model}_window_prediction_weather_by_facility.csv"
        )
        precip_monthly = read_csv_files(
            resourcefilepath / 'ResourceFile_WeatherDisruption',
            files=f"ResourceFile_Precipitation_Disruptions_{ssp}_{model}_monthly_prediction_weather_by_facility.csv"
        )

        precip_monthly = precip_monthly.drop(columns=precip_monthly.columns[precip_monthly.sum() == 0])
        precip_5day = precip_5day.drop(columns=precip_5day.columns[precip_5day.sum() == 0])

        self.parameters["precipitation_data_monthly"] = precip_monthly
        self.parameters["precipitation_data_five_day"] = precip_5day

        self.parameters["facility_characteristics"] = read_csv_files(
            resourcefilepath / 'demography',
            files="ResourceFile_Facility_Characteristics.csv"
        )

    def _read_wbgt_inputs(self, resourcefilepath) -> None:
        """Load the exported per-service WBGT responses and the WBGT exposure series."""
        ssp = self.parameters["climate_ssp"]
        tier = self.parameters["climate_model_ensemble_model"]

        rf_dir = resourcefilepath / "ResourceFile_WeatherDisruption"

        service_indicators = self.parameters.get("wbgt_service_indicators") or {
            s: self.parameters["wbgt_response_indicator"] for s in self._services
        }

        # one file per indicator, written by export_two_model_for_tlo in the fitting script
        self._wbgt_two_model = {}
        for service, indicator in service_indicators.items():
            path = rf_dir / f"ResourceFile_WBGT_TwoModel_{indicator}.csv"
            if not path.exists():
                raise FileNotFoundError(f"WBGT two-model file not found for {service}: {path}")
            self._wbgt_two_model[service] = pd.read_csv(path)
        self._wbgt_service_indicators = service_indicators

        # facility characteristics (same file as the precip driver)
        self.parameters["facility_characteristics"] = read_csv_files(
            resourcefilepath / 'demography',
            files="ResourceFile_Facility_Characteristics.csv"
        )

        model_id = TIER_TO_MODEL.get(tier, tier)
        exposure_path = rf_dir / f"wbgt_extreme_indices_facility_{model_id}_{ssp}.csv"
        if not exposure_path.exists():
            raise FileNotFoundError(
                f"WBGT exposure file not found: {exposure_path} (ssp={ssp}, tier={tier} -> model={model_id})"
            )
        self.parameters["wbgt_exposure_data"] = pd.read_csv(exposure_path, parse_dates=["date"])

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
                "hsi_checked": self._hsi_checked_count,
                "hsi_checked_unique": self._hsi_checked_unique_count,
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
                "hsi_checked_by_district": dict(self._hsi_checked_by_district),
                "hsi_checked_unique_by_district": dict(self._hsi_checked_unique_by_district),
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
                "hsi_checked_by_facility_treatment": dict(self._hsi_checked_by_facility_treatment),
                "hsi_checked_unique_by_facility_treatment": dict(self._hsi_checked_unique_by_facility_treatment),
                "hsi_total_by_facility_treatment": dict(self._hsi_total_by_facility_treatment),
                "cancelled_by_facility_treatment": dict(self._cancelled_by_facility_treatment),
                "delayed_by_facility_treatment": dict(self._delayed_by_facility_treatment),
                "supply_side_by_facility_treatment": dict(self._supply_side_by_facility_treatment),
                "demand_side_by_facility_treatment": dict(self._demand_side_by_facility_treatment),
            }
        )

    def reset_monthly_counters(self):
        self._reset_monthly_counters_internal()

    # ----------------------------------------------------------------------
    # Disruption-probability build (dispatches on weather_event)
    # ----------------------------------------------------------------------
    def build_disruption_probabilities(self):
        weather_event = self.parameters.get("weather_event", "precip")
        if weather_event == "precip":
            self._build_precip_disruptions()
        else:
            self._build_wbgt_disruptions()
        t = self.parameters["projected_precip_disruptions"]
        if "all" not in set(t["service"]):
            keys = ["RealFacility_ID", "year", "month"]
            all_rows = t.groupby(keys, as_index=False).agg(
                disruption=("disruption", "mean"),
                weather_event_value=("weather_event_value", "first"),  # same exposure for every service
            )
            all_rows["service"] = "all"
            all_rows["pred_baseline"] = np.nan
            all_rows["pred_precip"] = np.nan
            self.parameters["projected_precip_disruptions"] = pd.concat([t, all_rows[t.columns]], ignore_index=True)

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

    def _build_precip_disruptions(self):
        """Precip driver: single response replicated across all services for a uniform lookup."""
        p = self.parameters
        self.build_linear_models()

        precip_monthly_full = p["precipitation_data_monthly"]
        precip_5day_full = p["precipitation_data_five_day"]

        lag_4month_monthly = precip_monthly_full.shift(4)
        lag_9month_monthly = precip_monthly_full.shift(9)
        lag_1_5day = precip_5day_full.shift(1)

        start_idx = 12  # Skip  2025 rows; predictions run from 2026 onwards **UNTIL NASA FIX**
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

        year, month = 2026, 1
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
                precip_5d = precip_5day.at[t_idx, fac]
                lag_1_5d = lag_1_5day.at[t_idx, fac]
                lag_4m = lag_4month_monthly.iloc[t_idx, fac_idx]
                lag_9m = lag_9month_monthly.iloc[t_idx, fac_idx]

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
            np.clip(deficit / pred_baseline, 0, 1),  # deficits only
            0,
        )

        base = facility_month_df[["RealFacility_ID", "year", "month"]].copy()
        base["disruption"] = prob_disruption
        base["pred_baseline"] = pred_baseline
        base["pred_precip"] = pred_precip
        base["weather_event_value"] = facility_month_df["precip_monthly"].values

        services = self._services or [p["services_affected_precip"]]
        self.parameters["projected_precip_disruptions"] = pd.concat(
            [base.assign(service=s) for s in services], ignore_index=True
        )[[
            "RealFacility_ID", "year", "month", "service", "disruption",
            "pred_baseline", "pred_precip", "weather_event_value"
        ]]

    def _build_wbgt_disruptions(self):
        """Disruption = (baseline - wbgt) / baseline, using the two fitted models.
        Only reductions count, so it is clipped to [0, 1]."""
        p = self.parameters

        exposure = p["wbgt_exposure_data"].copy()
        for col in ["facility", "date", WBGT_VAR, WBGT_VAR + "_lag1", WBGT_PRECIP_VAR]:
            if col not in exposure.columns:
                raise KeyError(f"WBGT exposure file is missing column {col}")
        exposure["facility"] = exposure["facility"].astype(str)  # keep names exactly as in the characteristics file
        exposure["year"] = exposure["date"].dt.year
        exposure["month"] = exposure["date"].dt.month
        exposure = exposure[exposure["year"] >= p["year_effective_climate_disruptions"]]
        exposure = exposure.reset_index(drop=True)

        # the first month of the series has no lag1 (there is no earlier month in the file),
        # so those facility-months are left out and will not be disrupted
        no_lag = exposure[WBGT_VAR + "_lag1"].isna()
        if no_lag.any():
            months = sorted(exposure.loc[no_lag, "date"].dt.strftime("%Y-%m").unique())
            logger.warning(key="message",
                           data=f"WBGT: {no_lag.sum()} facility-months have no lag1 ({months}), not disrupted")
            exposure = exposure[~no_lag].reset_index(drop=True)

        # add each facility's characteristics to the exposure rows
        chars = p["facility_characteristics"].copy()
        chars.index = ['zone', 'urban_rural', 'district', 'ownership', 'altitude',
                       'facility_type', 'Latitude', 'Longitude', 'min_distance_to_clinic']
        chars = chars.T
        missing = [f for f in exposure["facility"].unique() if f not in chars.index]
        if missing:
            raise KeyError(f"No facility characteristics for {len(missing)} facilities, e.g. {missing[:10]}")
        exposure = exposure.join(chars, on="facility")

        results = []
        for service, table in self._wbgt_two_model.items():
            self._log_reference_categories(service, table, exposure)

            mu_baseline = np.exp(self._wbgt_log_mu(exposure, table, "baseline"))
            mu_wbgt = np.exp(self._wbgt_log_mu(exposure, table, "wbgt"))
            disruption = np.clip((mu_baseline - mu_wbgt) / mu_baseline, 0, 1)
            if np.isnan(disruption).any():
                cols = ["facility", "date", WBGT_VAR, WBGT_VAR + "_lag1", WBGT_PRECIP_VAR, "month", "zone"]
                bad = exposure.loc[np.isnan(disruption), cols]
                raise ValueError(f"NaN in WBGT disruption for {service} in {len(bad)} rows "
                                 f"({bad['facility'].nunique()} facilities):\n{bad.head(10)}")

            out = pd.DataFrame({
                "RealFacility_ID": exposure["facility"],
                "year": exposure["year"],
                "month": exposure["month"],
                "service": service,
                "disruption": disruption,
                "pred_baseline": mu_baseline,
                "pred_precip": mu_wbgt,
                "weather_event_value": exposure[WBGT_VAR],
            })
            results.append(out)

        p["projected_precip_disruptions"] = pd.concat(results, ignore_index=True)

    def _wbgt_log_mu(self, df, table, model):
        """log(mu) for every row of df, for one model ("baseline" or "wbgt")."""
        shared = table[table["model"] == "shared"]
        shared = dict(zip(shared["parameter_name"], shared["value"].astype(float)))
        rows = table[table["model"] == model]

        log_mu = np.zeros(len(df))

        # month fixed effect
        month_rows = rows[rows["parameter_name"] == MONTH_FE_NAME]
        month_fe = dict(zip(month_rows["level"].astype(float).astype(int), month_rows["value"]))
        log_mu += df["month"].map(month_fe).values

        # zone fixed effect, using 2025
        zone_rows = rows[rows["parameter_name"] == ZONE_YEAR_FE_NAME]
        zone_fe = dict(zip(zone_rows["level"].astype(str), zone_rows["value"]))
        zone_values = (df["zone"].astype(str) + "_" + str(WBGT_REFERENCE_YEAR)).map(zone_fe)
        if zone_values.isna().any():
            raise KeyError(f"No zone fixed effect for zones {sorted(df.loc[zone_values.isna(), 'zone'].unique())}")
        log_mu += zone_values.values

        # coefficients
        wbgt_lo = shared["wbgt_clip_lo"]
        wbgt_hi = shared["wbgt_clip_hi"]
        coefs = rows[rows["level"].isna()]
        for term, b in zip(coefs["parameter_name"], coefs["value"].astype(float)):
            if term.startswith("year_fe_") or term == "covid":
                continue  # these are 0 in 2025
            elif term.startswith(WBGT_VAR + "_s"):
                continue  # spline columns, added below from the lookup
            elif term == "precip_c":
                log_mu += b * (df[WBGT_PRECIP_VAR].values - shared["precip_shift"])
            elif term.startswith(WBGT_VAR + "_lag"):
                lag_col = term[:-2]  # e.g. wbgt5x_day_lag1_c -> wbgt5x_day_lag1
                lag = np.clip(df[lag_col].values.astype(float), wbgt_lo, wbgt_hi)
                log_mu += b * (lag - shared["wbgt_shift"])
            elif term.startswith("me_"):
                log_mu += b * self._wbgt_characteristic(df, term, shared)
            else:
                raise KeyError(f"Unknown term in WBGT file: {term}")

        # spline (wbgt model only)
        if model == "wbgt":
            spline = rows[rows["parameter_name"] == "spline"].copy()
            spline["level"] = spline["level"].astype(float)
            spline = spline.sort_values("level")
            wbgt = np.clip(df[WBGT_VAR].values.astype(float), wbgt_lo, wbgt_hi)
            log_mu += np.interp(wbgt, spline["level"], spline["value"])

        return log_mu

    def _wbgt_characteristic(self, df, term, shared):
        """Values of one facility characteristic term (me_...), coded the same way as in the fit."""
        for model_col, tlo_col in WBGT_CHARACTERISTIC_COLUMNS.items():
            # continuous: centred, negative altitude treated as missing
            if term == "me_" + model_col:
                x = pd.to_numeric(df[tlo_col], errors="coerce")
                if model_col == "A109__Altitude":
                    x = x.where(x >= 0)
                center = shared["center_" + model_col]
                return (x.fillna(center) - center).values

            # categorical: 1 if the facility has this level
            if term.startswith("me_" + model_col + "_"):
                level = term[len("me_" + model_col + "_"):].lower()
                return np.array([1.0 if _safe_name(v).lower() == level else 0.0 for v in df[tlo_col]])

        raise KeyError(f"No TLO column for {term}")

    def _log_reference_categories(self, service, table, df):
        # categories with no coefficient are treated as the reference level,
        # so this should only list the reference levels (e.g. Government, Rural)
        terms = list(table.loc[table["model"] == "wbgt", "parameter_name"].astype(str))
        for model_col, tlo_col in WBGT_CHARACTERISTIC_COLUMNS.items():
            prefix = "me_" + model_col + "_"
            levels = [t[len(prefix):].lower() for t in terms if t.startswith(prefix)]
            if levels:
                not_matched = sorted(set(str(v) for v in df[tlo_col] if _safe_name(v).lower() not in levels))
                logger.warning(key="message",
                               data=f"WBGT {service}: {tlo_col} treated as reference: {not_matched}")

    # ----------------------------------------------------------------------
    # Runtime, shared across drivers
    # ----------------------------------------------------------------------
    def check_hsi_for_disruption(self, hsi_event_item: HSIEventQueueItem, current_date: Date):
        """
        Check if an HSI event should be disrupted due to climate/weather conditions.
        Returns (disrupted, is_supply_side). disrupted is True if the event was disrupted
        (and should not proceed); is_supply_side distinguishes supply- vs demand-side (mode 2).
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
        district = self.sim.population.props.at[person_id, "district_of_residence"]
        treatment_id = getattr(hsi_event_item.hsi_event, "TREATMENT_ID", "unknown")

        service = self._treatment_to_service.get(treatment_id, self._default_disruption_service)
        if service is None:
            return False, False  # treatment not weather-affected

        if facility_used not in self.parameters["projected_precip_disruptions"]["RealFacility_ID"].values:
            return False, False

        fac_str, trt_str, dist_str = str(facility_used), str(treatment_id), str(district)

        # every check (a rescheduled HSI is counted again when it comes back)
        self._hsi_checked_count += 1
        self._increment_district_counter(self._hsi_checked_by_district, dist_str)
        self._increment_facility_treatment_counter(self._hsi_checked_by_facility_treatment, fac_str, trt_str)

        # first check of each HSI only
        hsi = hsi_event_item.hsi_event
        if not getattr(hsi, "_weather_checked", False):
            hsi._weather_checked = True
            self._hsi_checked_unique_count += 1
            self._increment_district_counter(self._hsi_checked_unique_by_district, dist_str)
            self._increment_facility_treatment_counter(
                self._hsi_checked_unique_by_facility_treatment, fac_str, trt_str
            )

        prob_disruption = self._get_disruption_probability(
            facility_id=facility_used, year=year, month=month, service=service,
        )

        if self.rng.binomial(1, prob_disruption) == 0:
            return False, False

        event_detail = self._get_event_detail(facility_used, year, month, service)

        is_supply_side = self._handle_disruption(
            hsi_event_item=hsi_event_item,
            prob_disruption=prob_disruption,
            current_date=current_date,
            facility_id=str(facility_used),
            district=str(district),
            treatment_id=str(treatment_id),
            event_detail=event_detail,
        )

        return True, is_supply_side

    def _get_disruption_probability(self, facility_id, year: int, month: int, service: str) -> float:
        """Look up and scale the disruption probability for a facility/year/month/service."""
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
        p = float(prob.iloc[0]) * self.parameters["scale_factor_prob_disruption"]
        if np.isnan(p) or p < 0:
            raise ValueError(f"Bad disruption probability {p} for {facility_id}, {year}-{month}, {service}")
        return min(p, 1.0)

    def _get_event_detail(self, facility_id, year: int, month: int, service: str) -> Dict:
        """Extra per-event fields for detailed logging. Returns {} if off or row absent."""
        if not self.parameters.get("detailed_disruption_logging", False):
            return {}
        t = self.parameters["projected_precip_disruptions"]
        mask = (
            (t["RealFacility_ID"] == facility_id)
            & (t["year"] == year)
            & (t["month"] == month)
            & (t["service"] == service)
        )
        sub = t.loc[mask]
        if sub.empty:
            return {}
        r = sub.iloc[0]
        return {
            "weather_event": self.parameters.get("weather_event", "precip"),
            "weather_event_value": float(r["weather_event_value"]) if "weather_event_value" in r else float("nan"),
            "pred_baseline": float(r["pred_baseline"]) if "pred_baseline" in r else float("nan"),
            "pred_precip": float(r["pred_precip"]) if "pred_precip" in r else float("nan"),
            "base_disruption": float(r["disruption"]),
        }

    def _handle_disruption(
        self,
        hsi_event_item: HSIEventQueueItem,
        prob_disruption: float,
        current_date: Date,
        facility_id: str,
        district: str,
        treatment_id: str,
        event_detail: Optional[Dict] = None,
    ):
        """Handle a disruption: determine supply/demand side, then reschedule or cancel."""
        event_detail = dict(event_detail or {})
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

        if event_detail:
            event_detail.update({
                "district": district,
                "is_supply_side": bool(is_supply_side),
                "prob_disruption": float(prob_disruption),
            })

        will_reschedule = self._determine_rescheduling(hsi_event_item)
        reschedule_refused = will_reschedule and (hsi_event_item.hsi_event.did_not_run() is False)

        if will_reschedule and not reschedule_refused:
            delay_days = self._calculate_delay(hsi_event_item, prob_disruption, current_date)
            new_topen = current_date + DateOffset(days=delay_days)

            self.sim.modules["HealthSystem"]._add_hsi_event_queue_item_to_hsi_event_queue(
                priority=hsi_event_item.priority,
                clinic_eligibility=hsi_event_item.clinic_eligibility,
                topen=new_topen,
                tclose=new_topen,
                hsi_event=hsi_event_item.hsi_event,
            )
            self._log_delayed_hsi(hsi_event_item, facility_id=facility_id, event_detail=event_detail)
            self._disruptions_delayed_count += 1
            self._increment_district_counter(self._delayed_by_district, district)
            self._increment_facility_treatment_counter(self._delayed_by_facility_treatment, facility_id, treatment_id)
        else:
            self._log_cancelled_hsi(
                hsi_event_item, facility_id=facility_id, call_never_ran=not reschedule_refused,
                event_detail=event_detail,
            )
            self._disruptions_cancelled_count += 1
            self._increment_district_counter(self._cancelled_by_district, district)
            self._increment_facility_treatment_counter(self._cancelled_by_facility_treatment, facility_id, treatment_id)

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
        """Days of delay for rescheduling. For time-sensitive modules the delay is capped
        so the new topen never exceeds the original tclose."""
        base_delay = int(
            max(self.parameters["scale_factor_appointment_urgency"] * hsi_event_item.priority, 1)
            * prob_disruption
            * self.parameters["scale_factor_severity_disruption_and_delay"]
            * self.parameters["delay_in_seeking_care_weather"]
        )

        window_days = (hsi_event_item.tclose - hsi_event_item.topen).days
        delay_days = max(0, base_delay + window_days)

        module_name = hsi_event_item.hsi_event.module.__class__.__name__
        if module_name in TIME_SENSITIVE_MODULES:
            max_allowable_days = max(0, (hsi_event_item.tclose - current_date).days)
            delay_days = min(delay_days, max_allowable_days)

        return delay_days

    def _log_delayed_hsi(self, hsi_event_item: HSIEventQueueItem, facility_id: Optional[str] = None,
                         event_detail: Optional[Dict] = None):
        """Log an HSI that was delayed due to weather (already rescheduled)."""
        data = {
            "TREATMENT_ID": getattr(hsi_event_item.hsi_event, "TREATMENT_ID", "unknown"),
            "Person_ID": hsi_event_item.hsi_event.target,
            "priority": hsi_event_item.priority,
            "RealFacility_ID": facility_id if facility_id is not None else "unknown",
        }
        if self.parameters.get("detailed_disruption_logging", False) and event_detail:
            data.update(event_detail)
        logger_summary.info(
            key="Weather_delayed_HSI_Event_full_info",
            data=data,
            description="record of each HSI event delayed due to weather",
        )

    def _log_cancelled_hsi(self, hsi_event_item: HSIEventQueueItem, facility_id: Optional[str] = None,
                           call_never_ran: bool = True, event_detail: Optional[Dict] = None):
        """Log an HSI that was cancelled due to weather (will not be rescheduled)."""
        person_id = hsi_event_item.hsi_event.target
        if not self.sim.population.props.at[person_id, 'is_alive']:
            return

        if call_never_ran:
            self.sim.modules["HealthSystem"].call_and_record_never_ran_hsi_event(
                hsi_event=hsi_event_item.hsi_event,
                priority=hsi_event_item.priority,
                clinic=hsi_event_item.clinic_eligibility,
            )
        data = {
            "TREATMENT_ID": getattr(hsi_event_item.hsi_event, "TREATMENT_ID", "unknown"),
            "Person_ID": person_id,
            "priority": hsi_event_item.priority,
            "RealFacility_ID": facility_id if facility_id is not None else "unknown",
        }
        if self.parameters.get("detailed_disruption_logging", False) and event_detail:
            data.update(event_detail)
        logger_summary.info(
            key="Weather_cancelled_HSI_Event_full_info",
            data=data,
            description="record of each HSI event cancelled due to weather",
        )


class WeatherDisruptionsMonthlyLogger(RegularEvent, PopulationScopeEventMixin):
    """Monthly logger for weather disruptions — overall, by district, and by facility×treatment."""

    def __init__(self, module: WeatherDisruptions):
        super().__init__(module, frequency=DateOffset(months=1), priority=Priority.END_OF_DAY)

    def apply(self, population):
        m = self.module

        logger_summary.info(
            key="weather_disruptions_monthly",
            data={
                "year": self.sim.date.year,
                "month": self.sim.date.month,
                "hsi_checked": m._hsi_checked_count,
                "hsi_checked_unique": m._hsi_checked_unique_count,
                "hsi_total": m._disruptions_hsi_total_count,
                "cancelled": m._disruptions_cancelled_count,
                "delayed": m._disruptions_delayed_count,
                "supply_side": m._supply_side_disruptions_count,
                "demand_side": m._demand_side_disruptions_count,
            }
        )
        logger_summary.info(
            key="weather_disruptions_monthly_by_district",
            data={
                "year": self.sim.date.year,
                "month": self.sim.date.month,
                "hsi_checked_by_district": dict(m._hsi_checked_by_district),
                "hsi_checked_unique_by_district": dict(m._hsi_checked_unique_by_district),
                "hsi_total_by_district": dict(m._hsi_total_by_district),
                "cancelled_by_district": dict(m._cancelled_by_district),
                "delayed_by_district": dict(m._delayed_by_district),
                "supply_side_by_district": dict(m._supply_side_by_district),
                "demand_side_by_district": dict(m._demand_side_by_district),
            }
        )
        logger_summary.info(
            key="weather_disruptions_monthly_by_facility_treatment",
            data={
                "year": self.sim.date.year,
                "month": self.sim.date.month,
                "hsi_checked_by_facility_treatment": dict(m._hsi_checked_by_facility_treatment),
                "hsi_checked_unique_by_facility_treatment": dict(m._hsi_checked_unique_by_facility_treatment),
                "hsi_total_by_facility_treatment": dict(m._hsi_total_by_facility_treatment),
                "cancelled_by_facility_treatment": dict(m._cancelled_by_facility_treatment),
                "delayed_by_facility_treatment": dict(m._delayed_by_facility_treatment),
                "supply_side_by_facility_treatment": dict(m._supply_side_by_facility_treatment),
                "demand_side_by_facility_treatment": dict(m._demand_side_by_facility_treatment),
            }
        )

        m.reset_monthly_counters()

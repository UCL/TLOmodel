import datetime
import heapq as hp
import itertools
import re
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import repeat
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from tlo import Date, DateOffset, Module, Parameter, Population, Property, Types, logging
from tlo.analysis.utils import (  # get_filtered_treatment_ids,
    flatten_multi_index_series_into_dict_for_logging,
)
from tlo.events import Event, PopulationScopeEventMixin, Priority, RegularEvent
from tlo.methods import Metadata
from tlo.methods.bed_days import BedDays
from tlo.methods.consumables import (
    Consumables,
    get_item_code_from_item_name,
    get_item_codes_from_package_name,
)
from tlo.methods.dxmanager import DxManager
from tlo.methods.equipment import Equipment
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

# Declare the assumption for the availability of consumables at the merged levels '1b' and '2'. This can be a
#  list of facility_levels over which an average is taken (within a district): e.g. ['1b', '2'].
AVAILABILITY_OF_CONSUMABLES_AT_MERGED_LEVELS_1B_AND_2 = ["1b"]  # <-- Implies that availability at merged level '1b & 2'
#                                                                     is equal to availability at level '1b'. This is
#                                                                     reasonable because the '1b' are more numerous than
#                                                                     those of '2' and have more overall capacity, so
#                                                                     probably account for the majority of the
#                                                                     interactions.
# Note that, as of PR #1743, this should not be changed, as the availability of consumables at level 1b is now
# encoded to reflect a (weighted) average of the availability of levels '1b' and '2'.

def pool_capabilities_at_levels_1b_and_2(df_original: pd.DataFrame) -> pd.DataFrame:
    """Return a modified version of the imported capabilities DataFrame to reflect that the capabilities of level 1b
    are pooled with those of level 2, and all labelled as level 2."""

    # Find total minutes and staff count after the re-allocation of capabilities from '1b' to '2'
    tots_after_reallocation = (
        df_original.assign(
            Facility_Level=lambda df: df.Facility_Level.replace(
                {"1b": LABEL_FOR_MERGED_FACILITY_LEVELS_1B_AND_2, "2": LABEL_FOR_MERGED_FACILITY_LEVELS_1B_AND_2}
            )
        )
        .groupby(by=["Facility_Level", "District", "Region", "Officer_Category"], dropna=False)[
            ["Total_Mins_Per_Day", "Staff_Count"]
        ]
        .sum()
        .reset_index()
    )

    # Construct a new version of the dataframe that uses the new totals
    df_updated = (
        df_original.drop(columns=["Total_Mins_Per_Day", "Staff_Count"])
        .merge(
            tots_after_reallocation,
            on=["Facility_Level", "District", "Region", "Officer_Category"],
            how="left",
        )
        .assign(
            Total_Mins_Per_Day=lambda df: df.Total_Mins_Per_Day.fillna(0.0),
            Staff_Count=lambda df: df.Staff_Count.fillna(0.0),
        )
    )

    ## Drop the index because the updated dataframe does not have an index and that breaks the assertion
    df_original = df_original.reset_index(drop=True)
    # Check that the *total* number of minutes per officer in each district/region is the same as before the change
    assert_series_equal(
        df_updated.groupby(by=["District", "Region", "Officer_Category"], dropna=False)["Total_Mins_Per_Day"].sum(),
        df_original.groupby(by=["District", "Region", "Officer_Category"], dropna=False)["Total_Mins_Per_Day"].sum(),
    )

    df_updated.groupby("Facility_Level")["Total_Mins_Per_Day"].sum()

    # Check size/shape of the updated dataframe is as expected
    assert df_updated.shape == df_original.shape
    assert (df_updated.dtypes == df_original.dtypes).all()

    for _level in ["0", "1a", "3", "4"]:
        assert df_original.loc[df_original.Facility_Level == _level].equals(
            df_updated.loc[df_updated.Facility_Level == _level]
        )

    assert np.isclose(
        df_updated.loc[
            df_updated.Facility_Level == LABEL_FOR_MERGED_FACILITY_LEVELS_1B_AND_2, "Total_Mins_Per_Day"
        ].sum(),
        df_updated.loc[df_updated.Facility_Level.isin(["1b", "2"]), "Total_Mins_Per_Day"].sum(),
    )

    return df_updated


class AppointmentSubunit(NamedTuple):
    """Component of an appointment relating to a specific officer type."""

    officer_type: str
    time_taken: float


def _accepts_argument(function: callable, argument: str) -> bool:
    """Helper to test if callable object accepts an argument with a given name.

    Compared to using `inspect.signature` or `inspect.getfullargspec` the approach here
    has significantly less overhead (as a full `Signature` or `FullArgSpec` object
     does not need to constructed) but is also less readable hence why it has been
    wrapped as a helper function despite being only one-line to make its functionality
    more obvious.

    :param function: Callable object to check if argument is present in.
    :param argument: Name of argument to check.
    :returns: ``True`` is ``argument`` is an argument of ``function`` else ``False``.
    """
    # co_varnames include both arguments to function and any internally defined variable
    # names hence we check only in the first `co_argcount` items which correspond to
    # just the arguments
    return argument in function.__code__.co_varnames[: function.__code__.co_argcount]


class HealthSystem(Module):
    """
    This is the Health System Module.
    The execution of all health systems interactions are controlled through this module.
    """

    INIT_DEPENDENCIES = {"Demography"}

    PARAMETERS = {
        # Organization of the HealthSystem
        "Master_Facilities_List": Parameter(Types.DATA_FRAME, "Listing of all health facilities."),
        # Definitions of the officers and appointment types
        "Officer_Types_Table": Parameter(Types.DATA_FRAME, 'The names of the types of health workers ("officers")'),
        "Appt_Types_Table": Parameter(Types.DATA_FRAME, "The names of the type of appointments with the health system"),
        "Appt_Offered_By_Facility_Level": Parameter(
            Types.DATA_FRAME, "Table indicating whether or not each appointment is offered at each facility level."
        ),
        "Appt_Time_Table": Parameter(
            Types.DATA_FRAME, "The time taken for each appointment, according to officer and facility type."
        ),
        # Capabilities of the HealthSystem (under alternative assumptions)
        "Daily_Capabilities_actual": Parameter(
            Types.DATA_FRAME,
            "The capabilities (minutes of time available of each type of officer in each facility) "
            "based on the _estimated current_ number and distribution of staff estimated.",
        ),
        "Daily_Capabilities_funded": Parameter(
            Types.DATA_FRAME,
            "The capabilities (minutes of time available of each type of officer in each facility) "
            "based on the _potential_ number and distribution of staff estimated (i.e. those "
            "positions that can be funded).",
        ),
        "Daily_Capabilities_funded_plus": Parameter(
            Types.DATA_FRAME,
            "The capabilities (minutes of time available of each type of officer in each facility) "
            "based on the _potential_ number and distribution of staff estimated, with adjustments "
            "to permit each appointment type that should be run at facility level to do so in every "
            "district.",
        ),
        "use_funded_or_actual_staffing": Parameter(
            Types.STRING,
            "If `actual`, then use the numbers and distribution of staff estimated to be available"
            " currently; If `funded`, then use the numbers and distribution of staff that are "
            "potentially available. If `funded_plus`, then use a dataset in which the allocation of "
            "staff to facilities is tweaked so as to allow each appointment type to run at each "
            "facility_level in each district for which it is defined. N.B. This parameter is "
            "over-ridden if an argument is provided to the module initialiser.",
        ),
        # Consumables
        "item_and_package_code_lookups": Parameter(
            Types.DATA_FRAME, "Data imported from the OneHealth Tool on consumable items, packages and costs."
        ),
        "consumables_item_designations": Parameter(
            Types.DATA_FRAME,
            "Look-up table for the designations of consumables (whether diagnostic, medicine, or other",
        ),
        "data_source_for_cons_availability_estimates": Parameter(
            Types.STRING, "Source of data on consumable availability. Options are: `original` or `updated`."
                          "The original source was used in the calibration and presented in the overview paper. The "
                          "updated source introduced in PR #1743 and better reflects the average availability of "
                          "consumables in the merged 1b/2 facility level."
        ),
        "availability_estimates": Parameter(
            Types.DICT, "Estimated availability of consumables in the LMIS dataset. Dict contains all the databases "
                        "that might be selected using `data_source_for_cons_availability_estimates`"
        ),
        "cons_availability": Parameter(
            Types.STRING,
            "Availability of consumables. If 'default' then use the availability specified in the ResourceFile; if "
            "'none', then let no consumable be  ever be available; if 'all', then all consumables are always available."
            " When using 'all' or 'none', requests for consumables are not logged. NB. This parameter is over-ridden"
            "if an argument is provided to the module initialiser."
            "Note that other options are also available: see the `Consumables` class.",
        ),
        "cons_override_treatment_ids": Parameter(
            Types.LIST,
            "Consumable availability within any treatment ids listed in this parameter will be set at to a "
            "given probabilty stored in override_treatment_ids_avail. By default this list is empty",
        ),
        "cons_override_treatment_ids_prob_avail": Parameter(
            Types.REAL,
            "Probability that consumables for treatment ids listed in cons_override_treatment_ids will be available",
        ),
        # Infrastructure and Equipment
        "BedCapacity": Parameter(Types.DATA_FRAME, "Data on the number of beds available of each type by facility_id"),
        "beds_availability": Parameter(
            Types.STRING,
            "Availability of beds. If 'default' then use the availability specified in the ResourceFile; if "
            "'none', then let no beds be  ever be available; if 'all', then all beds are always available. NB. This "
            "parameter is over-ridden if an argument is provided to the module initialiser.",
        ),
        "EquipmentCatalogue": Parameter(Types.DATA_FRAME, "Data on equipment items and packages."),
        "equipment_availability_estimates": Parameter(
            Types.DATA_FRAME, "Data on the availability of equipment items and packages."
        ),
        "equip_availability": Parameter(
            Types.STRING,
            "What to assume about the availability of equipment. If 'default' then use the availability specified in "
            "the ResourceFile; if 'none', then let no equipment ever be available; if 'all', then all equipment is "
            "always available. NB. This parameter is over-ridden if an argument is provided to the module initialiser.",
        ),
        "equip_availability_postSwitch": Parameter(
            Types.STRING,
            "What to assume about the availability of equipment after the switch (see `year_equip_availability_switch`"
            "). The options for this are the same as `equip_availability`.",
        ),
        "year_equip_availability_switch": Parameter(
            Types.INT,
            "Year in which the assumption for `equip_availability` changes (The change happens on 1st January of that "
            "year.)",
        ),
        # Service Availability
        "Service_Availability": Parameter(
            Types.LIST,
            "List of services to be available. NB. This parameter is over-ridden if an argument is provided"
            " to the module initialiser.",
        ),
        "policy_name": Parameter(Types.STRING, "Name of priority policy adopted"),
        "year_mode_switch": Parameter(Types.INT, "Year in which mode switch is enforced"),
        "scale_to_effective_capabilities": Parameter(
            Types.BOOL,
            "In year in which mode switch takes place, will rescale available capabilities to match those"
            "that were effectively used (on average) in the past year if this is set to True. This way,"
            "we can approximate overtime and rushing of appts even in mode 2.",
        ),
        "year_cons_availability_switch": Parameter(
            Types.INT,
            "Year in which consumable availability switch is enforced. The change happenson 1st January of that year.)",
        ),
        "year_use_funded_or_actual_staffing_switch": Parameter(
            Types.INT,
            "Year in which switch for `use_funded_or_actual_staffing` is enforced. (The change happens"
            "on 1st January of that year.)",
        ),
        "priority_rank": Parameter(
            Types.DICT,
            "Data on the priority ranking of each of the Treatment_IDs to be adopted by "
            " the queueing system under different policies, where the lower the number the higher"
            " the priority, and on which categories of individuals classify for fast-tracking "
            " for specific treatments",
        ),
        "HR_scaling_by_level_and_officer_type_table": Parameter(
            Types.DICT,
            "Factors by which capabilities of medical officer types at different levels will be"
            "scaled at the start of the year specified by `year_HR_scaling_by_level_and_officer_type`. This"
            "serves to simulate a number of effects (e.g. absenteeism, boosting capabilities of specific "
            "medical cadres, etc). This is the imported from an Excel workbook: keys are the worksheet "
            "names and values are the worksheets in the format of pd.DataFrames. Additional scenarios can "
            "be added by adding worksheets to this workbook: the value of "
            "`HR_scaling_by_level_and_officer_type_mode` indicates which sheet is used.",
        ),
        "year_HR_scaling_by_level_and_officer_type": Parameter(
            Types.INT,
            "Year in which one-off constant HR scaling will take place. (The change happens"
            "on 1st January of that year.)",
        ),
        "HR_scaling_by_level_and_officer_type_mode": Parameter(
            Types.STRING,
            "Mode of HR scaling considered at the start of the simulation. This corresponds to the name"
            "of the worksheet in `ResourceFile_HR_scaling_by_level_and_officer_type.xlsx` that should be"
            " used. Options are: `default` (capabilities are scaled by a constaint factor of 1); `data` "
            "(factors informed by survey data); and, `custom` (user can freely set these factors as "
            "parameters in the analysis).",
        ),
        "HR_scaling_by_district_table": Parameter(
            Types.DICT,
            "Factors by which daily capabilities in different districts will be"
            "scaled at the start of the year specified by year_HR_scaling_by_district to simulate"
            "(e.g., through catastrophic event disrupting delivery of services in particular district(s))."
            "This is the import of an Excel workbook: keys are the worksheet names and values are the "
            "worksheets in the format of pd.DataFrames. Additional scenarios can be added by adding "
            "worksheets to this workbook: the value of `HR_scaling_by_district_mode` indicates which"
            "sheet is used.",
        ),
        "year_HR_scaling_by_district": Parameter(
            Types.INT,
            "Year in which scaling of daily capabilities by district will take place. (The change happens"
            "on 1st January of that year.)",
        ),
        "HR_scaling_by_district_mode": Parameter(
            Types.STRING,
            "Mode of scaling of daily capabilities by district. This corresponds to the name of the "
            "worksheet in the file `ResourceFile_HR_scaling_by_district.xlsx`.",
        ),
        "yearly_HR_scaling": Parameter(
            Types.DICT,
            "Factors by which HR capabilities are scaled. "
            "Each sheet specifies a 'mode' for dynamic HR scaling. The mode to use is determined by the "
            "parameter `yearly_HR_scaling_mode`. Each sheet must have the same format, including the same "
            "column headers. On each sheet, the first row (for `2010`, when the simulation starts) "
            "specifies the initial configuration: `dynamic_HR_scaling_factor` (float) is the factor by "
            "which all human resoucres capabilities and multiplied; `scale_HR_by_popsize` (bool) specifies "
            "whether the capabilities should (also) grow by the factor by which the population has grown in"
            " the last year. Each subsequent row specifies a year where there should be a CHANGE in the "
            "configuration. If there are no further rows, then there is no change. But, for example, an"
            " additional row of the form ```2015, 1.05, TRUE``` would mean that on 1st January of 2015, "
            "2016, 2017, ....(and the rest of the simulation), the capabilities would increase by the "
            "product of 1.05 and by the ratio of the population size to that in the year previous.",
        ),
        "yearly_HR_scaling_mode": Parameter(
            Types.STRING,
            "Specifies which of the policies in yearly_HR_scaling should be adopted. This corresponds to"
            "a worksheet of the file `ResourceFile_dynamic_HR_scaling.xlsx`.",
        ),
        "tclose_overwrite": Parameter(
            Types.INT, "Decide whether to overwrite tclose variables assigned by disease modules"
        ),
        "tclose_days_offset_overwrite": Parameter(
            Types.INT,
            "Offset in days from topen at which tclose will be set by the healthsystem for all HSIs"
            "if tclose_overwrite is set to True.",
        ),
        # Mode Appt Constraints
        "mode_appt_constraints": Parameter(
            Types.INT,
            "Integer code in `{1, 2}` determining mode of constraints with regards to officer numbers "
            "and time - 1: elastic constraints, all HSI events run, provided "
            "officers required to deliver the HSI have capabilities > 0"
            "2: hard constraints, only HSI events for which capabilities are available run. N.B. This parameter"
            "is over-ridden if an argument is provided to the module initialiser.",
        ),
        "mode_appt_constraints_postSwitch": Parameter(
            Types.INT, "Mode considered after a mode switch in year_mode_switch."
        ),
        "cons_availability_postSwitch": Parameter(
            Types.STRING,
            "Consumables availability after switch in `year_cons_availability_switch`. Acceptable values"
            "are the same as those for Parameter `cons_availability`.",
        ),
        "use_funded_or_actual_staffing_postSwitch": Parameter(
            Types.STRING,
            "Staffing availability after switch in `year_use_funded_or_actual_staffing_switch`. "
            "Acceptable values are the same as those for Parameter `use_funded_or_actual_staffing`.",
        ),
        "clinic_configuration_name": Parameter(Types.STRING, "Name of configuration of clinics to use."),

        # Climate disruptions
        "projected_precip_disruptions": Parameter(
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
        ),
    }

    PROPERTIES = {
        "hs_is_inpatient": Property(
            Types.BOOL, "Whether or not the person is currently an in-patient at any medical facility"
        ),
    }

    def __init__(
        self,
        name: Optional[str] = None,
        service_availability: Optional[List[str]] = None,
        mode_appt_constraints: Optional[int] = None,
        cons_availability: Optional[str] = None,
        beds_availability: Optional[str] = None,
        equip_availability: Optional[str] = None,
        randomise_queue: bool = True,
        ignore_priority: bool = False,
        policy_name: Optional[str] = None,
        capabilities_coefficient: Optional[float] = None,
        use_funded_or_actual_staffing: Optional[str] = None,
        disable: bool = False,
        disable_and_reject_all: bool = False,
        hsi_event_count_log_period: Optional[str] = "month",
        projected_precip_disruptions: Optional[List[str]] = None,
        climate_ssp: Optional[str] = "ssp245",
        climate_model_ensemble_model: Optional[str] = "mean",
        year_effective_climate_disruptions: Optional[int] = 2025,
        services_affected_precip: Optional[str] = "none",
        scale_factor_appointment_urgency: Optional[str] = 1,
        delay_in_seeking_care_weather: Optional[float] = 4,
        scale_factor_severity_disruption_and_delay: Optional[float] = None,
        prop_supply_side_disruptions: Optional[float] = 0.5,
    ):
        super().__init__(name)

        assert isinstance(disable, bool)
        assert isinstance(disable_and_reject_all, bool)
        assert not (disable and disable_and_reject_all), "Cannot have both disable and disable_and_reject_all selected"
        assert not (ignore_priority and policy_name is not None), (
            "Cannot adopt a priority policy if the priority will be then ignored"
        )

        self.disable = disable
        self.disable_and_reject_all = disable_and_reject_all

        self.mode_appt_constraints = None
        if mode_appt_constraints is not None:
            assert mode_appt_constraints in {1, 2}
        self.arg_mode_appt_constraints = mode_appt_constraints

        self.rng_for_hsi_queue = None
        self.rng_for_dx = None

        self.randomise_queue = randomise_queue
        self.ignore_priority = ignore_priority
        self.lowest_priority_considered = 2
        self.priority_policy = None

        if policy_name is not None:
            assert policy_name in [
                "",
                "Default",
                "Test",
                "Test Mode 1",
                "Random",
                "Naive",
                "RMNCH",
                "VerticalProgrammes",
                "ClinicallyVulnerable",
                "EHP_III",
                "LCOA_EHP",
            ]
        self.arg_policy_name = policy_name

        self.tclose_overwrite = None
        self.tclose_days_offset_overwrite = None
        self.list_fasttrack = []
        self.arg_service_availability = service_availability
        self.service_availability = ["*"]

        if capabilities_coefficient is not None:
            assert capabilities_coefficient >= 0
            assert isinstance(capabilities_coefficient, float)
        self.capabilities_coefficient = capabilities_coefficient

        self.arg_use_funded_or_actual_staffing = use_funded_or_actual_staffing
        self._use_funded_or_actual_staffing = None

        self.recognised_modules_names = []
        self.HSI_EVENT_QUEUE = []
        self.hsi_event_queue_counter = 0

        assert cons_availability in (None, "default", "all", "none")
        self.arg_cons_availability = cons_availability

        assert beds_availability in (None, "default", "all", "none")
        self.arg_beds_availability = beds_availability

        assert equip_availability in (None, "default", "all", "none")
        self.arg_equip_availability = equip_availability

        self.dx_manager = DxManager(self)
        self.bed_days = None
        self.consumables = None
        self.healthsystemscheduler = None
        self._summary_counter = HealthSystemSummaryCounter()
        self.running_total_footprint = defaultdict(Counter)

        self._hsi_event_count_log_period = hsi_event_count_log_period
        self._hsi_event_counts_by_facility_monthly = Counter()

        if hsi_event_count_log_period in {"day", "month", "year", "simulation"}:
            self._hsi_event_counts_log_period = Counter()
            self._hsi_event_counts_cumulative = Counter()
            self._hsi_event_counts_by_facility_monthly = Counter()
            self._hsi_event_details = dict()

            self._never_ran_hsi_event_counts_log_period = Counter()
            self._never_ran_hsi_event_counts_cumulative = Counter()
            self._weather_cancelled_hsi_event_counts_log_period = Counter()
            self._weather_cancelled_hsi_event_counts_cumulative = Counter()
            self._weather_delayed_hsi_event_counts_log_period = Counter()
            self._weather_delayed_hsi_event_counts_cumulative = Counter()

            self._never_ran_hsi_event_details = dict()
            self._weather_cancelled_hsi_event_details = dict()
            self._weather_delayed_hsi_event_details = dict()

        elif hsi_event_count_log_period is not None:
            raise ValueError(
                "hsi_event_count_log_period argument should be one of 'day', 'month' 'year', 'simulation' or None."
            )

    def read_parameters(self, resourcefilepath: Optional[Path] = None):
        path_to_resourcefiles_for_healthsystem = resourcefilepath / "healthsystem"

        self.load_parameters_from_dataframe(
            pd.read_csv(path_to_resourcefiles_for_healthsystem / "ResourceFile_HealthSystem_parameters.csv")
        )

        self.parameters["Master_Facilities_List"] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem / "organisation" / "ResourceFile_Master_Facilities_List.csv"
        )

        filepath = (
            path_to_resourcefiles_for_healthsystem
            / "human_resources"
            / "clinics"
            / "ResourceFile_ClinicConfigurations"
            / f"{self.parameters['clinic_configuration_name']}.csv"
        )

        self._clinic_configuration = pd.read_csv(filepath)
        filepath = (
            path_to_resourcefiles_for_healthsystem
            / "human_resources"
            / "clinics"
            / "ResourceFile_ClinicMappings"
            / f"{self.parameters['clinic_configuration_name']}.csv"
        )

        self._clinic_mapping = pd.read_csv(filepath)
        self._clinic_names = self._clinic_configuration.columns.difference(
            ["Facility_ID", "Officer_Type_Code"]
        )
        self.validate_clinic_configuration(self._clinic_configuration)

        self.parameters["Officer_Types_Table"] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem
            / "human_resources"
            / "definitions"
            / "ResourceFile_Officer_Types_Table.csv"
        )
        self.parameters["Appt_Types_Table"] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem
            / "human_resources"
            / "definitions"
            / "ResourceFile_Appt_Types_Table.csv"
        )
        self.parameters["Appt_Offered_By_Facility_Level"] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem
            / "human_resources"
            / "definitions"
            / "ResourceFile_ApptType_By_FacLevel.csv"
        )
        self.parameters["Appt_Time_Table"] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem
            / "human_resources"
            / "definitions"
            / "ResourceFile_Appt_Time_Table.csv"
        )

        for _i in ["actual", "funded", "funded_plus"]:
            self.parameters[f"Daily_Capabilities_{_i}"] = pd.read_csv(
                path_to_resourcefiles_for_healthsystem
                / "human_resources"
                / f"{_i}"
                / "ResourceFile_Daily_Capabilities.csv"
            )

        self.parameters["item_and_package_code_lookups"] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem / "consumables" / "ResourceFile_Consumables_Items_and_Packages.csv"
        )
        self.parameters["consumables_item_designations"] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem / "consumables" / "ResourceFile_Consumables_Item_Designations.csv",
            dtype={"Item_Code": int, "is_diagnostic": bool, "is_medicine": bool, "is_other": bool},
        ).set_index("Item_Code")

        def read_consumables(filename):
            return pd.read_csv(path_to_resourcefiles_for_healthsystem / "consumables" / filename)
        self.parameters["availability_estimates"] = {
            "original": read_consumables("ResourceFile_Consumables_availability_small_original.csv"),
            "updated": read_consumables("ResourceFile_Consumables_availability_small.csv"),
        }

        self.parameters["BedCapacity"] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem / "infrastructure_and_equipment" / "ResourceFile_Bed_Capacity.csv"
        )

        self.parameters["EquipmentCatalogue"] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem
            / "infrastructure_and_equipment"
            / "ResourceFile_EquipmentCatalogue.csv"
        )
        self.parameters["equipment_availability_estimates"] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem
            / "infrastructure_and_equipment"
            / "ResourceFile_Equipment_Availability_Estimates.csv"
        )

        self.parameters["priority_rank"] = read_csv_files(
            path_to_resourcefiles_for_healthsystem / "priority_policies" / "ResourceFile_PriorityRanking_ALLPOLICIES",
            files=None,
        )

        self.parameters["HR_scaling_by_level_and_officer_type_table"]: Dict = read_csv_files(
            path_to_resourcefiles_for_healthsystem
            / "human_resources"
            / "scaling_capabilities"
            / "ResourceFile_HR_scaling_by_level_and_officer_type",
            files=None,
        )
        assert (
            self.parameters["HR_scaling_by_level_and_officer_type_mode"]
            in self.parameters["HR_scaling_by_level_and_officer_type_table"]
        ), (
            f"Value of `HR_scaling_by_level_and_officer_type_mode` not recognised: "
            f"{self.parameters['HR_scaling_by_level_and_officer_type_mode']}"
        )

        self.parameters["HR_scaling_by_district_table"]: Dict = read_csv_files(
            path_to_resourcefiles_for_healthsystem
            / "human_resources"
            / "scaling_capabilities"
            / "ResourceFile_HR_scaling_by_district",
            files=None,
        )
        assert self.parameters["HR_scaling_by_district_mode"] in self.parameters["HR_scaling_by_district_table"], (
            f"Value of `HR_scaling_by_district_mode` not recognised: {self.parameters['HR_scaling_by_district_mode']}"
        )

        self.parameters["yearly_HR_scaling"]: Dict = read_csv_files(
            path_to_resourcefiles_for_healthsystem
            / "human_resources"
            / "scaling_capabilities"
            / "ResourceFile_dynamic_HR_scaling",
            files=None,
            dtype={
                "year": int,
                "dynamic_HR_scaling_factor": float,
                "scale_HR_by_popsize": bool,
            },
        )
        assert self.parameters["yearly_HR_scaling_mode"] in self.parameters["yearly_HR_scaling"], (
            f"Value of `yearly_HR_scaling` not recognised: {self.parameters['yearly_HR_scaling_mode']}"
        )
        assert all(2010 in sheet["year"].values for sheet in self.parameters["yearly_HR_scaling"].values())

        path_to_resourcefiles_for_climate = resourcefilepath / "climate_change_impacts"
        self.parameters["projected_precip_disruptions"] = pd.read_csv(
            path_to_resourcefiles_for_climate
            / f'ResourceFile_Precipitation_Disruptions_{self.parameters["climate_ssp"]}_{self.parameters["climate_model_ensemble_model"]}.csv'
        )

    def validate_clinic_configuration(self, clinic_capabilities_df: pd.DataFrame):
        if clinic_capabilities_df.shape[0] == 0:
            return

        id_cols = ["Facility_ID", "Officer_Type_Code"]
        data = clinic_capabilities_df.drop(columns=id_cols)
        row_sums = data.sum(axis=1)
        mask = ~np.isclose(row_sums, 1.0, rtol=1e-5, atol=1e-8)
        if mask.any():
            raise ValueError(
                f"Row(s) {clinic_capabilities_df[mask][id_cols].values} in the clinics file do not sum to 1.0."
                "Please ensure that the fractions for clinic types sum to 1.0."
            )

        all_valid = self._clinic_mapping["Clinic"].isin(self._clinic_names).all()
        if not all_valid:
            raise ValueError(
                "The clinic mapping file contains at least one clinic name that is not present in the "
                "clinic configuration file. Please ensure that all clinic names in the mapping file "
                "are also present in the configuration file."
            )

    def pre_initialise_population(self):
        self.rng_for_hsi_queue = np.random.RandomState(self.rng.randint(2**31 - 1))
        self.rng_for_dx = np.random.RandomState(self.rng.randint(2**31 - 1))
        rng_for_consumables = np.random.RandomState(self.rng.randint(2**31 - 1))
        rng_for_equipment = np.random.RandomState(self.rng.randint(2**31 - 1))

        self.mode_appt_constraints = self.get_mode_appt_constraints()

        if self.mode_appt_constraints == 1:
            self.ignore_priority = True

        self.service_availability = self.get_service_availability()
        self.process_healthsystem_organisation_files()

        self.use_funded_or_actual_staffing = (
            self.parameters["use_funded_or_actual_staffing"]
            if self.arg_use_funded_or_actual_staffing is None
            else self.arg_use_funded_or_actual_staffing
        )

        self.bed_days = BedDays(hs_module=self, availability=self.get_beds_availability())
        self.bed_days.pre_initialise_population()

        _availability_data = self.update_consumables_availability_to_represent_merging_of_levels_1b_and_2(
                self.parameters["availability_estimates"][self.parameters["data_source_for_cons_availability_estimates"]]
            )

        self.consumables = Consumables(
            availability_data=_availability_data,
            item_code_designations=self.parameters["consumables_item_designations"],
            rng=rng_for_consumables,
            availability=self.get_cons_availability(),
            treatment_ids_overridden=self.parameters["cons_override_treatment_ids"],
            treatment_ids_overridden_avail=self.parameters["cons_override_treatment_ids_prob_avail"],
        )
        del self.parameters["availability_estimates"]

        self.equipment = Equipment(
            catalogue=self.parameters["EquipmentCatalogue"],
            data_availability=self.parameters["equipment_availability_estimates"],
            rng=rng_for_equipment,
            master_facilities_list=self.parameters["Master_Facilities_List"],
            availability=self.get_equip_availability(),
        )
        self.in_patient_equipment_package: set[int] = self.equipment.from_pkg_names("In-patient")

        self.tclose_overwrite = self.parameters["tclose_overwrite"]
        self.tclose_days_offset_overwrite = self.parameters["tclose_days_offset_overwrite"]

        assert self.parameters["policy_name"] in self.parameters["priority_rank"]
        self.setup_priority_policy()

    def initialise_population(self, population):
        self.bed_days.initialise_population(population.props)

    def initialise_simulation(self, sim):
        if self.capabilities_coefficient is None:
            self.capabilities_coefficient = self.sim.modules["Demography"].initial_model_to_data_popsize_ratio

        self.bed_days.initialise_beddays_tracker(
            model_to_data_popsize_ratio=self.sim.modules["Demography"].initial_model_to_data_popsize_ratio
        )

        self.consumables.on_start_of_day(sim.date)

        self.recognised_modules_names = [
            m.name for m in self.sim.modules.values() if Metadata.USES_HEALTHSYSTEM in m.METADATA
        ]

        df = self.sim.population.props
        districts_of_residence = set(df.loc[df.is_alive, "district_of_residence"].cat.categories)
        assert all(
            districts_of_residence.issubset(per_level_facilities.keys())
            for per_level_facilities in self._facilities_for_each_district.values()
        ), (
            "At least one district_of_residence value in population not present in "
            "self._facilities_for_each_district resource file"
        )

        if not (self.disable or self.disable_and_reject_all):
            self.healthsystemscheduler = HealthSystemScheduler(self)
            sim.schedule_event(self.healthsystemscheduler, sim.date)

        sim.schedule_event(HealthSystemChangeMode(self), Date(self.parameters["year_mode_switch"], 1, 1))

        if self.parameters["cons_availability_postSwitch"] not in self.consumables._options_for_availability:
            raise ValueError(
                f"Value for `cons_availability_postSwitch` is not within defined options: "
                f"{self.parameters['cons_availability_postSwitch']}"
            )

        sim.schedule_event(
            HealthSystemChangeParameters(self, parameters_to_change=["cons_availability"]),
            Date(self.parameters["year_cons_availability_switch"], 1, 1),
        )

        sim.schedule_event(
            HealthSystemChangeParameters(self, parameters_to_change=["equip_availability"]),
            Date(self.parameters["year_equip_availability_switch"], 1, 1),
        )

        sim.schedule_event(
            HealthSystemChangeParameters(self, parameters_to_change=["use_funded_or_actual_staffing"]),
            Date(self.parameters["year_use_funded_or_actual_staffing_switch"], 1, 1),
        )

        sim.schedule_event(
            ConstantRescalingHRCapabilities(self),
            Date(self.parameters["year_HR_scaling_by_level_and_officer_type"], 1, 1),
        )

        sim.schedule_event(
            RescaleHRCapabilities_ByDistrict(self), Date(self.parameters["year_HR_scaling_by_district"], 1, 1)
        )

        sim.schedule_event(DynamicRescalingHRCapabilities(self), Date(sim.date))
        sim.schedule_event(HealthSystemLogger(self), Date(sim.date.year, 1, 1))

    def on_birth(self, mother_id, child_id):
        self.bed_days.on_birth(self.sim.population.props, mother_id, child_id)

    def on_simulation_end(self):
        self.bed_days.on_simulation_end()
        self.consumables.on_simulation_end()
        self.equipment.on_simulation_end()

        if self._hsi_event_count_log_period == "simulation":
            self._write_hsi_event_counts_to_log_and_reset()
            self._write_never_ran_hsi_event_counts_to_log_and_reset()
            self._write_weather_cancelled_hsi_event_counts_to_log_and_reset()
            self._write_weather_delayed_hsi_event_counts_to_log_and_reset()
        if self._hsi_event_count_log_period is not None:
            logger_summary.info(
                key="hsi_event_details",
                description="Map from integer keys to hsi event detail dictionaries",
                data={"hsi_event_key_to_event_details": {k: d._asdict() for d, k in self._hsi_event_details.items()}},
            )
            logger_summary.info(
                key="never_ran_hsi_event_details",
                description="Map from integer keys to never ran hsi event detail dictionaries",
                data={
                    "never_ran_hsi_event_key_to_event_details": {
                        k: d._asdict() for d, k in self._never_ran_hsi_event_details.items()
                    }
                },
            )
            logger_summary.info(
                key="weather_delayed_hsi_event_details",
                description="Map from integer keys to weather delayed hsi event detail dictionaries",
                data={
                    "weather_delayed_hsi_event_key_to_event_details": {
                        k: d._asdict() for d, k in self._weather_delayed_hsi_event_details.items()
                    }
                },
            )
            logger_summary.info(
                key="weather_cancelled_hsi_event_details",
                description="Map from integer keys to weather cancelled hsi event detail dictionaries",
                data={
                    "weather_cancelled_hsi_event_key_to_event_details": {
                        k: d._asdict() for d, k in self._weather_cancelled_hsi_event_details.items()
                    }
                },
            )

    def setup_priority_policy(self):
        self.priority_policy = self.get_priority_policy_initial()
        self.load_priority_policy(self.priority_policy)

        self.list_fasttrack.append(("age_exact_years", "FT_if_5orUnder"))
        if "Contraception" in self.sim.modules or "SimplifiedBirths" in self.sim.modules:
            self.list_fasttrack.append(("is_pregnant", "FT_if_pregnant"))
        if "Hiv" in self.sim.modules:
            self.list_fasttrack.append(("hv_diagnosed", "FT_if_Hivdiagnosed"))
        if "Tb" in self.sim.modules:
            self.list_fasttrack.append(("tb_diagnosed", "FT_if_tbdiagnosed"))

    def process_healthsystem_organisation_files(self):
        self._facility_levels = set(self.parameters["Master_Facilities_List"]["Facility_Level"]) - {"5"}
        assert self._facility_levels == {"0", "1a", "1b", "2", "3", "4"}

        self._appointment_types = set(self.parameters["Appt_Types_Table"]["Appt_Type_Code"])

        appt_time_data = self.parameters["Appt_Time_Table"]
        appt_times_per_level_and_type = {
            _facility_level: defaultdict(list) for _facility_level in self._facility_levels
        }
        for appt_time_tuple in appt_time_data.itertuples():
            appt_times_per_level_and_type[appt_time_tuple.Facility_Level][appt_time_tuple.Appt_Type_Code].append(
                AppointmentSubunit(
                    officer_type=appt_time_tuple.Officer_Category, time_taken=appt_time_tuple.Time_Taken_Mins
                )
            )
        assert sum(
            len(appt_info_list)
            for level in self._facility_levels
            for appt_info_list in appt_times_per_level_and_type[level].values()
        ) == len(appt_time_data)
        self._appt_times = appt_times_per_level_and_type

        appt_type_per_level_data = self.parameters["Appt_Offered_By_Facility_Level"]
        self._appt_type_by_facLevel = {
            _facility_level: set(
                appt_type_per_level_data["Appt_Type_Code"][
                    appt_type_per_level_data[f"Facility_Level_{_facility_level}"]
                ]
            )
            for _facility_level in self._facility_levels
        }

        districts_in_region = self.sim.modules["Demography"].parameters["districts_in_region"]
        all_districts = set(self.sim.modules["Demography"].parameters["district_num_to_district_name"].values())

        facilities_per_level_and_district = {_facility_level: {} for _facility_level in self._facility_levels}
        facilities_by_facility_id = dict()
        for facility_tuple in self.parameters["Master_Facilities_List"].itertuples():
            _facility_info = FacilityInfo(
                id=facility_tuple.Facility_ID,
                name=facility_tuple.Facility_Name,
                level=facility_tuple.Facility_Level,
                region=facility_tuple.Region,
            )

            facilities_by_facility_id[facility_tuple.Facility_ID] = _facility_info

            if pd.notnull(facility_tuple.District):
                facilities_per_level_and_district[facility_tuple.Facility_Level][facility_tuple.District] = (
                    _facility_info
                )
            elif pd.isnull(facility_tuple.District) and pd.notnull(facility_tuple.Region):
                for _district in districts_in_region[facility_tuple.Region]:
                    facilities_per_level_and_district[facility_tuple.Facility_Level][_district] = _facility_info
            elif (
                pd.isnull(facility_tuple.District)
                and pd.isnull(facility_tuple.Region)
                and (facility_tuple.Facility_Level != "5")
            ):
                for _district in all_districts:
                    facilities_per_level_and_district[facility_tuple.Facility_Level][_district] = _facility_info

        assert all(
            all_districts == facilities_per_level_and_district[_facility_level].keys()
            for _facility_level in self._facility_levels
        ), "There is not one of each facility type available to each district."

        self._facility_by_facility_id = facilities_by_facility_id
        self._facilities_for_each_district = facilities_per_level_and_district

    def setup_daily_capabilities(self, use_funded_or_actual_staffing):
        capabilities = self.parameters[f"Daily_Capabilities_{use_funded_or_actual_staffing}"]
        capabilities = capabilities.set_index(
            "FacilityID_"
            + capabilities["Facility_ID"].astype("Int64").astype(str)
            + "_Officer_"
            + capabilities["Officer_Category"]
        )

        self._clinic_configuration = self.format_clinic_capabilities()
        self._daily_capabilities = {}
        self._daily_capabilities_per_staff = {}
        self._officers_with_availability = {}
        for clinic in self._clinic_names:
            multiplier = self._clinic_configuration[clinic]
            updated_capabilities = capabilities.copy()
            updated_capabilities["Total_Mins_Per_Day"] = updated_capabilities["Total_Mins_Per_Day"] * multiplier
            updated_capabilities["Staff_Count"] = updated_capabilities["Staff_Count"] * multiplier
            self._daily_capabilities[clinic], self._daily_capabilities_per_staff[clinic] = (
                self.format_daily_capabilities(updated_capabilities, use_funded_or_actual_staffing)
            )
            self._officers_with_availability[clinic] = {k for k, v in self._daily_capabilities[clinic].items() if v > 0}

    def get_clinic_eligibility(self, treatment_id: str) -> str:
        eligible_treatment_ids = self._clinic_mapping.loc[
            self._clinic_mapping["Treatment"] == treatment_id, "Clinic"
        ]
        clinic = eligible_treatment_ids.iloc[0] if not eligible_treatment_ids.empty else "GenericClinic"
        return clinic

    def format_daily_capabilities(
        self, capabilities, use_funded_or_actual_staffing: str
    ) -> tuple[pd.Series, pd.Series]:
        capabilities = pool_capabilities_at_levels_1b_and_2(capabilities)
        capabilities = capabilities.rename(columns={"Officer_Category": "Officer_Type_Code"})
        capabilities["Mins_Per_Day_Per_Staff"] = capabilities["Total_Mins_Per_Day"] / capabilities["Staff_Count"]

        facility_ids = self.parameters["Master_Facilities_List"]["Facility_ID"].values
        officer_type_codes = set(self.parameters["Officer_Types_Table"]["Officer_Category"].values)

        facs = list()
        officers = list()
        for f in facility_ids:
            for o in officer_type_codes:
                facs.append(f)
                officers.append(o)

        capabilities_ex = pd.DataFrame(data={"Facility_ID": facs, "Officer_Type_Code": officers})

        mfl = self.parameters["Master_Facilities_List"]
        capabilities_ex = capabilities_ex.merge(mfl, on="Facility_ID", how="left")

        capabilities_per_staff_ex = capabilities_ex.copy()

        capabilities_ex = capabilities_ex.merge(
            capabilities[["Facility_ID", "Officer_Type_Code", "Total_Mins_Per_Day"]],
            on=["Facility_ID", "Officer_Type_Code"],
            how="left",
        )
        capabilities_ex = capabilities_ex.fillna(0)

        capabilities_per_staff_ex = capabilities_per_staff_ex.merge(
            capabilities[["Facility_ID", "Officer_Type_Code", "Mins_Per_Day_Per_Staff"]],
            on=["Facility_ID", "Officer_Type_Code"],
            how="left",
        )
        capabilities_per_staff_ex = capabilities_per_staff_ex.fillna(0)

        capabilities_ex = capabilities_ex.set_index(
            "FacilityID_"
            + capabilities_ex["Facility_ID"].astype(str)
            + "_Officer_"
            + capabilities_ex["Officer_Type_Code"]
        )

        capabilities_per_staff_ex = capabilities_per_staff_ex.set_index(
            "FacilityID_"
            + capabilities_ex["Facility_ID"].astype(str)
            + "_Officer_"
            + capabilities_ex["Officer_Type_Code"]
        )

        capabilities_ex = capabilities_ex.rename(columns={"Total_Mins_Per_Day": "Total_Minutes_Per_Day"})

        assert abs(capabilities_ex["Total_Minutes_Per_Day"].sum() - capabilities["Total_Mins_Per_Day"].sum()) < 1e-7
        assert len(capabilities_ex) == len(facility_ids) * len(officer_type_codes)
        assert len(capabilities_per_staff_ex) == len(facility_ids) * len(officer_type_codes)

        return capabilities_ex["Total_Minutes_Per_Day"].to_dict(), capabilities_per_staff_ex[
            "Mins_Per_Day_Per_Staff"
        ].to_dict()

    def format_clinic_capabilities(self) -> pd.DataFrame:
        capabilities_cl = self._clinic_configuration
        facility_ids = set(self._facility_by_facility_id.keys())
        officer_type_codes = set(self.parameters["Officer_Types_Table"]["Officer_Category"].values)
        facs = list()
        officers = list()
        for f in facility_ids:
            for o in officer_type_codes:
                facs.append(f)
                officers.append(o)

        capabilities_ex = pd.DataFrame(data={"Facility_ID": facs, "Officer_Type_Code": officers})

        mfl = self.parameters["Master_Facilities_List"]
        capabilities_ex = capabilities_ex.merge(mfl, on="Facility_ID", how="left")
        capabilities_ex = capabilities_ex.merge(
            capabilities_cl,
            on=["Facility_ID", "Officer_Type_Code"],
            how="left",
        )
        capabilities_ex["GenericClinic"] = capabilities_ex["GenericClinic"].fillna(1)
        other_cols = capabilities_ex.columns.difference(["Facility_ID", "Officer_Type_Code", "GenericClinic"])
        capabilities_ex[other_cols] = capabilities_ex[other_cols].fillna(0)

        capabilities_ex = capabilities_ex.set_index(
            "FacilityID_"
            + capabilities_ex["Facility_ID"].astype(str)
            + "_Officer_"
            + capabilities_ex["Officer_Type_Code"]
        )

        assert len(capabilities_ex) == len(facility_ids) * len(officer_type_codes)
        return capabilities_ex

    def _rescale_capabilities_to_capture_effective_capability(self):
        for clinic, clinic_cl in self._daily_capabilities.items():
            for facID_and_officer in clinic_cl.keys():
                rescaling_factor = self._summary_counter.frac_time_used_by_facID_and_officer(
                    facID_and_officer=facID_and_officer, clinic=clinic
                )

                if rescaling_factor > 1 and rescaling_factor != float("inf"):
                    self._daily_capabilities[clinic][facID_and_officer] *= rescaling_factor
                    self._daily_capabilities_per_staff[clinic][facID_and_officer] *= rescaling_factor

    def update_consumables_availability_to_represent_merging_of_levels_1b_and_2(self, df_original):
        mfl = self.parameters["Master_Facilities_List"]

        dfx = df_original.merge(
            mfl[["Facility_ID", "District", "Facility_Level"]],
            left_on="Facility_ID",
            right_on="Facility_ID",
            how="left",
        )

        availability_columns = list(filter(lambda x: x.startswith("available_prop"), dfx.columns))

        availability_at_1b_and_2 = (
            dfx.drop(dfx.index[~dfx["Facility_Level"].isin(AVAILABILITY_OF_CONSUMABLES_AT_MERGED_LEVELS_1B_AND_2)])
            .groupby(by=["District", "month", "item_code"])[availability_columns]
            .mean()
            .reset_index()
            .assign(Facility_Level=LABEL_FOR_MERGED_FACILITY_LEVELS_1B_AND_2)
        )

        availability_at_1b_and_2 = availability_at_1b_and_2.merge(
            mfl[["Facility_ID", "District", "Facility_Level"]],
            left_on=["District", "Facility_Level"],
            right_on=["District", "Facility_Level"],
            how="left",
        )

        df_updated = (
            pd.concat(
                [
                    dfx.drop(dfx.index[dfx["Facility_Level"] == LABEL_FOR_MERGED_FACILITY_LEVELS_1B_AND_2]),
                    availability_at_1b_and_2[dfx.columns],
                ]
            )
            .drop(columns=["Facility_Level", "District"])
            .sort_values(["Facility_ID", "month", "item_code"])
            .reset_index(drop=True)
        )

        assert df_updated.shape == df_original.shape
        assert (df_updated.columns == df_original.columns).all()
        assert (df_updated.dtypes == df_original.dtypes).all()

        facilities_with_any_differences = set(
            df_updated.loc[
                ~(
                    df_original.sort_values(["Facility_ID", "month", "item_code"]).reset_index(drop=True) == df_updated
                ).all(axis=1),
                "Facility_ID",
            ]
        )
        updated_facilities = set(
            mfl.loc[mfl['Facility_Level'] == LABEL_FOR_MERGED_FACILITY_LEVELS_1B_AND_2, 'Facility_ID']
        )
        assert facilities_with_any_differences.issubset(updated_facilities)

        return df_updated

    def get_service_availability(self) -> List[str]:
        if self.arg_service_availability is None:
            service_availability = self.parameters["Service_Availability"]
        else:
            service_availability = self.arg_service_availability

        assert isinstance(service_availability, list)

        logger.info(
            key="message",
            data=f"Running Health System With the Following Service Availability: {self.service_availability}",
        )
        return service_availability

    def get_cons_availability(self) -> str:
        if self.arg_cons_availability is None:
            _cons_availability = self.parameters["cons_availability"]
        else:
            _cons_availability = self.arg_cons_availability

        logger.info(
            key="message",
            data=f"Running Health System With the Following Consumables Availability: {_cons_availability}",
        )
        return _cons_availability

    def get_beds_availability(self) -> str:
        if self.arg_beds_availability is None:
            _beds_availability = self.parameters["beds_availability"]
        else:
            _beds_availability = self.arg_beds_availability

        if self.disable:
            _beds_availability = "all"

        logger.info(
            key="message", data=f"Running Health System With the Following Beds Availability: {_beds_availability}"
        )
        return _beds_availability

    def get_equip_availability(self) -> str:
        if self.arg_equip_availability is None:
            _equip_availability = self.parameters["equip_availability"]
        else:
            _equip_availability = self.arg_equip_availability

        logger.info(
            key="message",
            data=f"Running Health System With the Following Equipment Availability: {_equip_availability}",
        )
        return _equip_availability

    def schedule_to_call_never_ran_on_date(self, hsi_event: "HSI_Event", tdate: datetime.datetime):
        self.sim.schedule_event(HSIEventWrapper(hsi_event=hsi_event, run_hsi=False), tdate)

    def get_mode_appt_constraints(self) -> int:
        return (
            self.parameters["mode_appt_constraints"]
            if self.arg_mode_appt_constraints is None
            else self.arg_mode_appt_constraints
        )

    @property
    def use_funded_or_actual_staffing(self) -> str:
        return self._use_funded_or_actual_staffing

    @use_funded_or_actual_staffing.setter
    def use_funded_or_actual_staffing(self, use_funded_or_actual_staffing) -> str:
        assert use_funded_or_actual_staffing in ["actual", "funded", "funded_plus"]
        self._use_funded_or_actual_staffing = use_funded_or_actual_staffing
        self.setup_daily_capabilities(self._use_funded_or_actual_staffing)

    def get_priority_policy_initial(self) -> str:
        return self.parameters["policy_name"] if self.arg_policy_name is None else self.arg_policy_name

    def load_priority_policy(self, policy):
        if policy != "":
            Policy_df = self.parameters["priority_rank"][policy]
            self.lowest_priority_considered = Policy_df.loc[
                Policy_df["Treatment"] == "lowest_priority_considered", "Priority"
            ].iloc[0]
            self.priority_rank_dict = (
                Policy_df.set_index("Treatment", drop=True)
                .convert_dtypes()
                .to_dict(orient="index")
            )
            del self.priority_rank_dict["lowest_priority_considered"]

    def schedule_hsi_event(
        self,
        hsi_event: "HSI_Event",
        priority: int,
        topen: datetime.datetime,
        tclose: Optional[datetime.datetime] = None,
        do_hsi_event_checks: bool = True,
    ):
        DEFAULT_DAYS_OFFSET_VALUE_FOR_TCLOSE_IF_NONE_SPECIFIED = 7

        if hsi_event.module.name in ("CareOfWomenDuringPregnancy", "Labour", "PostnatalSupervisor", "NewbornOutcomes"):
            if tclose is None:
                tclose = topen + DateOffset(days=DEFAULT_DAYS_OFFSET_VALUE_FOR_TCLOSE_IF_NONE_SPECIFIED)
        else:
            if self.tclose_overwrite == 1:
                tclose = topen + pd.to_timedelta(self.tclose_days_offset_overwrite, unit="D")
            elif tclose is None:
                tclose = topen + DateOffset(days=DEFAULT_DAYS_OFFSET_VALUE_FOR_TCLOSE_IF_NONE_SPECIFIED)

        assert topen >= self.sim.date
        assert topen < tclose

        if self.ignore_priority:
            priority = 0
        elif self.priority_policy != "":
            priority = self.enforce_priority_policy(hsi_event=hsi_event)

        assert priority >= 0

        if (self.mode_appt_constraints == 2) and (priority > self.lowest_priority_considered):
            self.schedule_to_call_never_ran_on_date(hsi_event=hsi_event, tdate=tclose)
            return

        if self.disable and (not self.disable_and_reject_all):
            self.sim.schedule_event(HSIEventWrapper(hsi_event=hsi_event, run_hsi=True), topen)
            return

        if self.disable_and_reject_all:
            self.schedule_to_call_never_ran_on_date(hsi_event=hsi_event, tdate=tclose)
            return

        if do_hsi_event_checks:
            self.check_hsi_event_is_valid(hsi_event)

        if not self.is_treatment_id_allowed(hsi_event.TREATMENT_ID, self.service_availability):
            self.sim.schedule_event(HSIEventWrapper(hsi_event=hsi_event, run_hsi=False), tclose)
        else:
            hsi_event.initialise()
            clinic_eligibility = self.get_clinic_eligibility(hsi_event.TREATMENT_ID)

            for officer in hsi_event.expected_time_requests:
                if officer not in self._daily_capabilities[clinic_eligibility]:
                    logger.warning(
                        key="message",
                        data=f"Unknown officer '{officer}' requested by "
                             f"{hsi_event.__class__.__name__} at time of scheduling."
                    )

            self._add_hsi_event_queue_item_to_hsi_event_queue(
                clinic_eligibility=clinic_eligibility,
                priority=priority,
                topen=topen,
                tclose=tclose,
                hsi_event=hsi_event,
            )

    def _add_hsi_event_queue_item_to_hsi_event_queue(
        self, clinic_eligibility, priority, topen, tclose, hsi_event
    ) -> None:
        self.hsi_event_queue_counter += 1

        if self.randomise_queue:
            rand_queue = self.rng_for_hsi_queue.randint(0, 1000000)
        else:
            rand_queue = self.hsi_event_queue_counter

        _new_item: HSIEventQueueItem = HSIEventQueueItem(
            clinic_eligibility, priority, topen, rand_queue, self.hsi_event_queue_counter, tclose, hsi_event
        )
        hp.heappush(self.HSI_EVENT_QUEUE, _new_item)

    def enforce_priority_policy(self, hsi_event) -> int:
        priority_ranking = self.priority_rank_dict

        if hsi_event.TREATMENT_ID not in priority_ranking:
            warnings.warn(UserWarning(f"Couldn't find priority ranking for TREATMENT_ID {hsi_event.TREATMENT_ID}"))
            return self.lowest_priority_considered

        df = self.sim.population.props
        treatment_ranking = priority_ranking[hsi_event.TREATMENT_ID]
        for attribute, fasttrack_code in self.list_fasttrack:
            if treatment_ranking[fasttrack_code] > -1:
                if attribute == "age_exact_years":
                    if df.at[hsi_event.target, attribute] <= 5:
                        return treatment_ranking[fasttrack_code]
                else:
                    if df.at[hsi_event.target, attribute]:
                        return treatment_ranking[fasttrack_code]

        return treatment_ranking["Priority"]

    def check_hsi_event_is_valid(self, hsi_event):
        assert isinstance(hsi_event, HSI_Event)
        assert hsi_event.TREATMENT_ID != ""
        assert not isinstance(hsi_event.target, Population)

        assert self.appt_footprint_is_valid(hsi_event.EXPECTED_APPT_FOOTPRINT), (
            f"the incorrectly formatted appt_footprint is {hsi_event.EXPECTED_APPT_FOOTPRINT}"
        )

        assert hsi_event.ACCEPTED_FACILITY_LEVEL in self._facility_levels, (
            f"In the HSI with TREATMENT_ID={hsi_event.TREATMENT_ID}, the ACCEPTED_FACILITY_LEVEL (="
            f"{hsi_event.ACCEPTED_FACILITY_LEVEL}) is not recognised."
        )

        self.bed_days.check_beddays_footprint_format(hsi_event.BEDDAYS_FOOTPRINT)

        appt_type_to_check_list = hsi_event.EXPECTED_APPT_FOOTPRINT.keys()
        facility_appt_types = self._appt_type_by_facLevel[hsi_event.ACCEPTED_FACILITY_LEVEL]
        assert facility_appt_types.issuperset(appt_type_to_check_list), (
            f"An appointment type has been requested at a facility level for "
            f"which it is not possible: TREATMENT_ID={hsi_event.TREATMENT_ID}"
        )

    @staticmethod
    def is_treatment_id_allowed(treatment_id: str, service_availability: list) -> bool:
        def _treatment_matches_pattern(_treatment_id, _service_availability):
            def _matches_this_pattern(_treatment_id, _s):
                if "*" in _s:
                    assert _s[-1] == "*", f"Component of service_availability has an asteriks not at the end: {_s}"
                    _s_split = _s.split("_")
                    _treatment_id_split = _treatment_id.split("_", len(_s_split) - 1)
                    return all(
                        [(a == b) or (b == "*") for a, b in itertools.zip_longest(_treatment_id_split, _s_split)]
                    )
                else:
                    return _treatment_id == _s

            for _s in service_availability:
                if _matches_this_pattern(_treatment_id, _s):
                    return True
            return False

        if not service_availability:
            return False
        if service_availability == ["*"]:
            return True
        elif treatment_id in service_availability:
            return True
        elif treatment_id.startswith("FirstAttendance_"):
            return True
        else:
            if _treatment_matches_pattern(treatment_id, service_availability):
                return True
        return False

    def schedule_batch_of_individual_hsi_events(
        self, hsi_event_class, person_ids, priority, topen, tclose=None, **event_kwargs
    ):
        priorities = priority if isinstance(priority, Iterable) else repeat(priority)
        topens = topen if isinstance(topen, Iterable) else repeat(topen)
        tcloses = tclose if isinstance(tclose, Iterable) else repeat(tclose)
        for i, (person_id, priority, topen, tclose) in enumerate(zip(person_ids, priorities, topens, tcloses)):
            self.schedule_hsi_event(
                hsi_event=hsi_event_class(person_id=person_id, **event_kwargs),
                priority=priority,
                topen=topen,
                tclose=tclose,
                do_hsi_event_checks=(i == 0),
            )

    def appt_footprint_is_valid(self, appt_footprint):
        return isinstance(appt_footprint, dict) and all(
            k in self._appointment_types and v >= 0 for k, v in appt_footprint.items()
        )

    @property
    def capabilities_today(self) -> dict:
        scaled = {
            clinic_name: {fid: cl * self.capabilities_coefficient for fid, cl in clinic_cl.items()}
            for clinic_name, clinic_cl in self._daily_capabilities.items()
        }
        return scaled

    def get_blank_appt_footprint(self):
        return Counter()

    def get_facility_info(self, hsi_event) -> FacilityInfo:
        the_district = self.sim.population.props.at[hsi_event.target, "district_of_residence"]
        the_level = hsi_event.ACCEPTED_FACILITY_LEVEL
        return self._facilities_for_each_district[the_level][the_district]

    def get_appt_footprint_as_time_request(self, facility_info: FacilityInfo, appt_footprint: dict):
        appt_footprint_times = Counter()
        for appt_type in appt_footprint:
            try:
                appt_info_list = self._appt_times[facility_info.level][appt_type]
            except KeyError as e:
                raise KeyError(
                    f"The time needed for an appointment is not defined for the specified facility level: "
                    f"appt_type={appt_type}, "
                    f"facility_level={facility_info.level}."
                ) from e

            for appt_info in appt_info_list:
                appt_footprint_times[f"FacilityID_{facility_info.id}_Officer_{appt_info.officer_type}"] += (
                    appt_info.time_taken
                )
        return appt_footprint_times

    def record_hsi_event(
        self, hsi_event, actual_appt_footprint=None, did_run=True, priority=None, clinic=None
    ):
        real_facility_id = None
        if hsi_event.target is not None and hsi_event.facility_info is not None:
            fac_level = hsi_event.facility_info.level
            try:
                real_facility_id = self.sim.population.props.at[hsi_event.target, f"level_{fac_level}"]
            except (KeyError, TypeError):
                real_facility_id = None

        self.write_to_hsi_log(
            event_details=hsi_event.as_namedtuple(actual_appt_footprint),
            person_id=hsi_event.target,
            facility_id=hsi_event.facility_info.id,
            did_run=did_run,
            priority=priority,
            clinic=clinic,
            real_facility_id=real_facility_id,
        )

    def write_to_hsi_log(
        self,
        event_details: HSIEventDetails,
        person_id: int,
        facility_id: Optional[int],
        did_run: bool,
        priority: int,
        clinic: str,
        real_facility_id: Optional[str] = None,
    ):
        hsi_record = {
            "Event_Name": event_details.event_name,
            "TREATMENT_ID": event_details.treatment_id,
            "Number_By_Appt_Type_Code": dict(event_details.appt_footprint),
            "Person_ID": person_id,
            "Squeeze_Factor": 0.0,
            "priority": priority,
            "did_run": did_run,
            "Facility_Level": event_details.facility_level if event_details.facility_level is not None else -99,
            "Facility_ID": facility_id if facility_id is not None else -99,
            "RealFacility_ID": real_facility_id if real_facility_id is not None else "unknown",
            "Equipment": sorted(event_details.equipment),
            "Clinic": clinic if clinic is not None else "None",
        }

        logger.debug(key="HSI_Event", data=hsi_record, description="record of each HSI event")
        if did_run:
            if self._hsi_event_count_log_period is not None:
                event_details_key = self._hsi_event_details.setdefault(event_details, len(self._hsi_event_details))
                self._hsi_event_counts_log_period[event_details_key] += 1

                if real_facility_id is not None and real_facility_id != 'unknown':
                    facility_key = f"{real_facility_id}:{event_details.treatment_id}"
                    self._hsi_event_counts_by_facility_monthly[facility_key] += 1

            # *** CHANGE: pass real_facility_id to summary counter ***
            self._summary_counter.record_hsi_event(
                treatment_id=event_details.treatment_id,
                hsi_event_name=event_details.event_name,
                appt_footprint=event_details.appt_footprint,
                level=event_details.facility_level,
                real_facility_id=real_facility_id,
            )

    def call_and_record_never_ran_hsi_event(self, hsi_event, priority=None, clinic=None):
        hsi_event.never_ran()

        if hsi_event.facility_info is not None:
            self.write_to_never_ran_hsi_log(
                event_details=hsi_event.as_namedtuple(),
                person_id=hsi_event.target,
                facility_id=hsi_event.facility_info.id,
                priority=priority,
                clinic=clinic,
            )
        else:
            self.write_to_never_ran_hsi_log(
                event_details=hsi_event.as_namedtuple(), person_id=-1, facility_id=-1, priority=priority, clinic=clinic
            )

    def call_and_record_weather_cancelled_hsi_event(self, hsi_event, priority=None, real_facility_id=None):
        hsi_event.never_ran()
        if hsi_event.facility_info is not None:
            self.write_to_weather_cancelled_hsi_log(
                event_details=hsi_event.as_namedtuple(),
                person_id=hsi_event.target,
                facility_id=hsi_event.facility_info.id,
                priority=priority,
                real_facility_id=real_facility_id,
            )
        else:
            self.write_to_weather_cancelled_hsi_log(
                event_details=hsi_event.as_namedtuple(),
                person_id=-1,
                facility_id=-1,
                priority=priority,
                real_facility_id=real_facility_id,
            )

    def call_and_record_weather_delayed_hsi_event(self, hsi_event, priority=None, real_facility_id=None):
        if "HSI_CardioMetabolicDisorders_Refill_Medication" in hsi_event.as_namedtuple():
            hsi_event.did_not_run_weather_event()
        else:
            hsi_event.did_not_run()
        if hsi_event.facility_info is not None:
            self.write_to_weather_delayed_hsi_log(
                event_details=hsi_event.as_namedtuple(),
                person_id=hsi_event.target,
                facility_id=hsi_event.facility_info.id,
                priority=priority,
                real_facility_id=real_facility_id,
            )
        else:
            self.write_to_weather_delayed_hsi_log(
                event_details=hsi_event.as_namedtuple(),
                person_id=-1,
                facility_id=-1,
                priority=priority,
                real_facility_id=real_facility_id,
            )

    def write_to_never_ran_hsi_log(
        self, event_details: HSIEventDetails, person_id: int, facility_id: Optional[int], priority: int, clinic: str
    ):
        logger.debug(
            key="Never_ran_HSI_Event",
            data={
                "Event_Name": event_details.event_name,
                "TREATMENT_ID": event_details.treatment_id,
                "Number_By_Appt_Type_Code": dict(event_details.appt_footprint),
                "Person_ID": person_id,
                "priority": priority,
                "Facility_Level": event_details.facility_level if event_details.facility_level is not None else "-99",
                "Facility_ID": facility_id if facility_id is not None else -99,
                "Clinic": clinic,
            },
            description="record of each HSI event that never ran",
        )
        if self._hsi_event_count_log_period is not None:
            event_details_key = self._never_ran_hsi_event_details.setdefault(
                event_details, len(self._never_ran_hsi_event_details)
            )
            self._never_ran_hsi_event_counts_log_period[event_details_key] += 1
        self._summary_counter.record_never_ran_hsi_event(
            treatment_id=event_details.treatment_id,
            hsi_event_name=event_details.event_name,
            appt_footprint=event_details.appt_footprint,
            level=event_details.facility_level,
        )

    def write_to_weather_cancelled_hsi_log(
        self,
        event_details: HSIEventDetails,
        person_id: int,
        facility_id: Optional[int],
        priority: int,
        real_facility_id: Optional[str] = None,
    ):
        logger_summary.info(
            key="Weather_cancelled_HSI_Event_full_info",
            data={
                "Event_Name": event_details.event_name,
                "TREATMENT_ID": event_details.treatment_id,
                "Number_By_Appt_Type_Code": dict(event_details.appt_footprint),
                "Person_ID": person_id,
                "priority": priority,
                "Facility_Level": event_details.facility_level if event_details.facility_level is not None else "-99",
                "Facility_ID": facility_id if facility_id is not None else -99,
                "RealFacility_ID": real_facility_id if real_facility_id is not None else "unknown",
            },
            description="record of each HSI event that was cancelled due to weather",
        )
        if self._hsi_event_count_log_period is not None:
            event_details_key = self._weather_cancelled_hsi_event_details.setdefault(
                event_details, len(self._weather_cancelled_hsi_event_details)
            )
            self._weather_cancelled_hsi_event_counts_log_period[event_details_key] += 1
        # *** CHANGE: pass real_facility_id to summary counter ***
        self._summary_counter.record_weather_cancelled_hsi_event(
            treatment_id=event_details.treatment_id,
            hsi_event_name=event_details.event_name,
            appt_footprint=event_details.appt_footprint,
            level=event_details.facility_level,
            real_facility_id=real_facility_id,
        )

    def write_to_weather_delayed_hsi_log(
        self,
        event_details: HSIEventDetails,
        person_id: int,
        facility_id: Optional[int],
        priority: int,
        real_facility_id: Optional[str] = None,
    ):
        logger_summary.info(
            key="Weather_delayed_HSI_Event_full_info",
            data={
                "Event_Name": event_details.event_name,
                "TREATMENT_ID": event_details.treatment_id,
                "Number_By_Appt_Type_Code": dict(event_details.appt_footprint),
                "Person_ID": person_id,
                "priority": priority,
                "Facility_Level": event_details.facility_level if event_details.facility_level is not None else "-99",
                "Facility_ID": facility_id if facility_id is not None else -99,
                "RealFacility_ID": real_facility_id if real_facility_id is not None else "unknown",
            },
            description="record of each HSI event that was delayed due to weather",
        )
        if self._hsi_event_count_log_period is not None:
            event_details_key = self._weather_delayed_hsi_event_details.setdefault(
                event_details, len(self._weather_delayed_hsi_event_details)
            )
            self._weather_delayed_hsi_event_counts_log_period[event_details_key] += 1
        # *** CHANGE: pass real_facility_id to summary counter ***
        self._summary_counter.record_weather_delayed_hsi_event(
            treatment_id=event_details.treatment_id,
            hsi_event_name=event_details.event_name,
            appt_footprint=event_details.appt_footprint,
            level=event_details.facility_level,
            real_facility_id=real_facility_id,
        )

    def log_current_capabilities_and_usage(self):
        for clinic_name in self._clinic_names:
            self.log_clinic_current_capabilities_and_usage(clinic_name)

    def log_current_capabilities_and_usage(self, clinic_name):
        current_capabilities = self.capabilities_today[clinic_name]
        total_footprint = self.running_total_footprint[clinic_name]

        comparison = pd.DataFrame(index=current_capabilities.keys())
        comparison["Total_Minutes_Per_Day"] = current_capabilities.values()
        comparison["Minutes_Used"] = pd.Series(total_footprint, dtype="float64")
        comparison["Minutes_Used"] = comparison["Minutes_Used"].fillna(0.0)
        assert len(comparison) == len(current_capabilities)

        total_available = comparison["Total_Minutes_Per_Day"].sum()
        fraction_time_used_overall = comparison["Minutes_Used"].sum() / total_available if total_available > 0 else 0.0

        facility_id = [_f.split("_")[1] for _f in comparison.index]
        summary_by_fac_id = comparison.groupby(by=facility_id)[["Total_Minutes_Per_Day", "Minutes_Used"]].sum()
        summary_by_fac_id["Fraction_Time_Used"] = (
            summary_by_fac_id["Minutes_Used"] / summary_by_fac_id["Total_Minutes_Per_Day"]
        ).replace([np.inf, -np.inf, np.nan], 0.0)

        fraction_time_used_by_facID_and_officer = (
            comparison["Minutes_Used"] / comparison["Total_Minutes_Per_Day"]
        ).replace([np.inf, -np.inf, np.nan], 0.0)

        officer = [_f.rsplit("Officer_")[1] for _f in comparison.index]
        level = [self._facility_by_facility_id[int(_fac_id)].level for _fac_id in facility_id]
        level = list(map(lambda x: x.replace("1b", "2"), level))
        summary_by_officer = comparison.groupby(by=[officer, level])[["Total_Minutes_Per_Day", "Minutes_Used"]].sum()
        summary_by_officer["Fraction_Time_Used"] = (
            summary_by_officer["Minutes_Used"] / summary_by_officer["Total_Minutes_Per_Day"]
        ).replace([np.inf, -np.inf, np.nan], 0.0)
        summary_by_officer.index.names = ["Officer_Type", "Facility_Level"]

        logger.info(
            key="Capacity",
            data={
                "Clinic": clinic_name,
                "Frac_Time_Used_Overall": fraction_time_used_overall,
                "Frac_Time_Used_By_Facility_ID": summary_by_fac_id["Fraction_Time_Used"].to_dict(),
                "Frac_Time_Used_By_OfficerType": flatten_multi_index_series_into_dict_for_logging(
                    summary_by_officer["Fraction_Time_Used"]
                ),
            },
            description="daily summary of utilisation and capacity of health system resources",
        )

        self._summary_counter.record_hs_status(
            fraction_time_used_across_all_facilities_in_this_clinic=fraction_time_used_overall,
            fraction_time_used_by_facID_and_officer_in_this_clinic=fraction_time_used_by_facID_and_officer.to_dict(),
            clinic=clinic_name
        )

    def remove_beddays_footprint(self, person_id):
        self.bed_days.remove_beddays_footprint(person_id=person_id)

    def find_events_for_person(self, person_id: int):
        list_of_events = list()
        for ev_tuple in self.HSI_EVENT_QUEUE:
            date = ev_tuple.topen
            event = ev_tuple.hsi_event
            if isinstance(event.target, (int, np.integer)):
                if event.target == person_id:
                    list_of_events.append((date, event))
        return list_of_events

    def reset_queue(self):
        self.HSI_EVENT_QUEUE = []
        self.hsi_event_queue_counter = 0

    def get_item_codes_from_package_name(self, package: str) -> dict:
        return get_item_codes_from_package_name(self.parameters["item_and_package_code_lookups"], package)

    def get_item_code_from_item_name(self, item: str) -> int:
        return get_item_code_from_item_name(self.parameters["item_and_package_code_lookups"], item)

    def override_availability_of_consumables(self, item_codes) -> None:
        self.consumables.override_availability(item_codes)

    def override_cons_availability_for_treatment_ids(
        self, treatment_ids: list = None, prob_available: float = None
    ) -> None:
        self.consumables.treatment_ids_overridden = treatment_ids if treatment_ids is not None else []
        if (treatment_ids is not None) and (len(treatment_ids) > 0):
            assert prob_available is not None, "If treatment_ids is provided, prob_available must be provided"
        self.consumables.treatment_ids_overridden_avail = prob_available if prob_available is not None else 0.0

    def _write_hsi_event_counts_to_log_and_reset(self):
        logger_summary.info(
            key="hsi_event_counts",
            description=(
                f"Counts of the HSI events that have run in this "
                f"{self._hsi_event_count_log_period} with keys corresponding to integer"
                f" keys recorded in dictionary in hsi_event_details log entry."
            ),
            data={"hsi_event_key_to_counts": dict(self._hsi_event_counts_log_period)},
        )
        self._hsi_event_counts_cumulative += self._hsi_event_counts_log_period
        self._hsi_event_counts_log_period.clear()

    def _write_never_ran_hsi_event_counts_to_log_and_reset(self):
        logger_summary.info(
            key="never_ran_hsi_event_counts",
            description=(
                f"Counts of the HSI events that never ran in this "
                f"{self._hsi_event_count_log_period} with keys corresponding to integer"
                f" keys recorded in dictionary in hsi_event_details log entry."
            ),
            data={"never_ran_hsi_event_key_to_counts": dict(self._never_ran_hsi_event_counts_log_period)},
        )
        self._never_ran_hsi_event_counts_cumulative += self._never_ran_hsi_event_counts_log_period
        self._never_ran_hsi_event_counts_log_period.clear()

    def _write_weather_cancelled_hsi_event_counts_to_log_and_reset(self):
        logger_summary.info(
            key="weather_cancelled_hsi_event_counts",
            description=(
                f"Counts of the HSI events that were cancelled due to weather ran "
                f"{self._hsi_event_count_log_period} with keys corresponding to integer"
                f" keys recorded in dictionary in hsi_event_details log entry."
            ),
            data={
                "weather_cancelled_hsi_event_key_to_counts": dict(self._weather_cancelled_hsi_event_counts_log_period)
            },
        )
        self._weather_cancelled_hsi_event_counts_cumulative += self._weather_cancelled_hsi_event_counts_log_period
        self._weather_cancelled_hsi_event_counts_log_period.clear()

    def _write_weather_delayed_hsi_event_counts_to_log_and_reset(self):
        logger_summary.info(
            key="weather_delayed_hsi_event_counts",
            description=(
                f"Counts of the HSI events that were delayed due to weather ran "
                f"{self._hsi_event_count_log_period} with keys corresponding to integer"
                f" keys recorded in dictionary in hsi_event_details log entry."
            ),
            data={"weather_delayed_hsi_event_key_to_counts": dict(self._weather_delayed_hsi_event_counts_log_period)},
        )
        self._weather_delayed_hsi_event_counts_cumulative += self._weather_delayed_hsi_event_counts_log_period
        self._weather_delayed_hsi_event_counts_log_period.clear()

    def _write_hsi_event_counts_by_facility_to_log_and_reset(self):
        logger_summary.info(
            key="hsi_event_counts_by_facility_monthly",
            description=(
                "Monthly counts of HSI events by facility_id and treatment_id. "
                "Keys are in format 'facility_id:treatment_id'."
            ),
            data={
                "counts": dict(self._hsi_event_counts_by_facility_monthly)
            },
        )
        self._hsi_event_counts_by_facility_monthly.clear()

    def on_end_of_day(self) -> None:
        self.bed_days.on_end_of_day()
        if self._hsi_event_count_log_period == "day":
            self._write_hsi_event_counts_to_log_and_reset()
            self._write_never_ran_hsi_event_counts_to_log_and_reset()
            self._write_weather_cancelled_hsi_event_counts_to_log_and_reset()
            self._write_weather_delayed_hsi_event_counts_to_log_and_reset()

    def on_end_of_month(self) -> None:
        self._write_hsi_event_counts_by_facility_to_log_and_reset()

        if self._hsi_event_count_log_period == "month":
            self._write_hsi_event_counts_to_log_and_reset()
            self._write_never_ran_hsi_event_counts_to_log_and_reset()
            self._write_weather_cancelled_hsi_event_counts_to_log_and_reset()
            self._write_weather_delayed_hsi_event_counts_to_log_and_reset()

    def on_end_of_year(self) -> None:
        if (self.sim.date.year == self.parameters["year_mode_switch"] - 1) and self.parameters[
            "scale_to_effective_capabilities"
        ]:
            self._rescale_capabilities_to_capture_effective_capability()
        self._summary_counter.write_to_log_and_reset_counters()
        self.consumables.on_end_of_year()
        self.bed_days.on_end_of_year()
        if self._hsi_event_count_log_period == "year":
            self._write_hsi_event_counts_to_log_and_reset()
            self._write_never_ran_hsi_event_counts_to_log_and_reset()
            self._write_weather_cancelled_hsi_event_counts_to_log_and_reset()
            self._write_weather_delayed_hsi_event_counts_to_log_and_reset()

        self._record_general_equipment_usage_for_year()

    def do_all_required_officers_have_nonzero_capabilities(self, expected_time_requests, clinic) -> bool:
        clinic_capabilities = self.capabilities_today[clinic]
        for officer in expected_time_requests:
            if clinic_capabilities.get(officer, 0.0) == 0.0:
                return False
        return True

    def run_individual_level_events_in_mode_1(
        self, _list_of_individual_hsi_event_tuples: List[HSIEventQueueItem]
    ) -> List:
        _to_be_held_over = list()
        assert self.mode_appt_constraints == 1

        if _list_of_individual_hsi_event_tuples:
            for ev_num, event in enumerate(_list_of_individual_hsi_event_tuples):
                _priority = event.priority
                clinic = event.clinic_eligibility
                event = event.hsi_event
                _appt_footprint_before_running = event.EXPECTED_APPT_FOOTPRINT

                ok_to_run = True
                if event.expected_time_requests:
                    ok_to_run = self.do_all_required_officers_have_nonzero_capabilities(
                                    event.expected_time_requests, clinic=clinic)
                if ok_to_run:
                    if sum(event.BEDDAYS_FOOTPRINT.values()):
                        event._received_info_about_bed_days = self.bed_days.issue_bed_days_according_to_availability(
                            facility_id=self.bed_days.get_facility_id_for_beds(persons_id=event.target),
                            footprint=event.BEDDAYS_FOOTPRINT,
                        )

                    assert event.facility_info is not None, (
                        f"Cannot run HSI {event.TREATMENT_ID} without facility_info being defined."
                    )

                    actual_appt_footprint = event.run(squeeze_factor=0.0)

                    if actual_appt_footprint is not None:
                        assert self.appt_footprint_is_valid(actual_appt_footprint)
                    else:
                        actual_appt_footprint = _appt_footprint_before_running

                    actual_call = self.get_appt_footprint_as_time_request(
                        facility_info=event.facility_info, appt_footprint=actual_appt_footprint
                    )
                    self.running_total_footprint[clinic].update(actual_call)

                    self.record_hsi_event(
                        hsi_event=event,
                        actual_appt_footprint=actual_appt_footprint,
                        did_run=True,
                        priority=_priority,
                    )
                else:
                    rtn_from_did_not_run = event.did_not_run()

                    if rtn_from_did_not_run is not False:
                        hp.heappush(_to_be_held_over, _list_of_individual_hsi_event_tuples[ev_num])

                    self.record_hsi_event(
                        hsi_event=event,
                        actual_appt_footprint=event.EXPECTED_APPT_FOOTPRINT,
                        did_run=False,
                        priority=_priority,
                    )

        return _to_be_held_over

    def _record_general_equipment_usage_for_year(self):
        general_equipment_by_facility_level = {
            "1a": self.equipment.from_pkg_names("General_FacilityLevel_1a_and_1b"),
            "1b": self.equipment.from_pkg_names("General_FacilityLevel_1a_and_1b"),
            "2": self.equipment.from_pkg_names("General_FacilityLevel_2"),
        }

        for fac in self._facility_by_facility_id.values():
            self.equipment.record_use_of_equipment(
                facility_id=fac.id, item_codes=general_equipment_by_facility_level.get(fac.level, set())
            )

    @property
    def hsi_event_counts(self) -> Counter:
        if self._hsi_event_count_log_period is None:
            return Counter()
        else:
            total_hsi_event_counts = self._hsi_event_counts_cumulative + self._hsi_event_counts_log_period
            return Counter(
                {
                    event_details: total_hsi_event_counts[event_details_key]
                    for event_details, event_details_key in self._hsi_event_details.items()
                }
            )

    @property
    def never_ran_hsi_event_counts(self) -> Counter:
        if self._hsi_event_count_log_period is None:
            return Counter()
        else:
            total_never_ran_hsi_event_counts = (
                self._never_ran_hsi_event_counts_cumulative + self._never_ran_hsi_event_counts_log_period
            )
            return Counter(
                {
                    event_details: total_never_ran_hsi_event_counts[event_details_key]
                    for event_details, event_details_key in self._never_ran_hsi_event_details.items()
                }
            )

    @property
    def weather_cancelled_hsi_event_counts(self) -> Counter:
        if self._hsi_event_count_log_period is None:
            return Counter()
        else:
            total_weather_cancelled_hsi_event_counts = (
                self._weather_cancelled_hsi_event_counts_cumulative
                + self._weather_cancelled_hsi_event_counts_log_period
            )
            return Counter(
                {
                    event_details: total_weather_cancelled_hsi_event_counts[event_details_key]
                    for event_details, event_details_key in self._weather_cancelled_hsi_event_details.items()
                }
            )

    @property
    def weather_delayed_hsi_event_counts(self) -> Counter:
        if self._hsi_event_count_log_period is None:
            return Counter()
        else:
            total_weather_delayed_hsi_event_counts = (
                self._weather_delayed_hsi_event_counts_cumulative + self._weather_delayed_hsi_event_counts_log_period
            )
            return Counter(
                {
                    event_details: total_weather_delayed_hsi_event_counts[event_details_key]
                    for event_details, event_details_key in self._weather_delayed_hsi_event_details.items()
                }
            )


class HealthSystemScheduler(RegularEvent, PopulationScopeEventMixin):
    """
    This is the HealthSystemScheduler. It is an event that occurs every day and must be the LAST event of the day.
    """

    def __init__(self, module: HealthSystem):
        super().__init__(module, frequency=DateOffset(days=1), priority=Priority.END_OF_DAY)

    @staticmethod
    def _is_last_day_of_the_year(date):
        return (date.month == 12) and (date.day == 31)

    @staticmethod
    def _is_last_day_of_the_month(date):
        return date.month != (date + pd.DateOffset(days=1)).month

    def _get_events_due_today(self) -> List:
        due_today = list()
        is_alive = self.sim.population.props.is_alive

        while len(self.module.HSI_EVENT_QUEUE) > 0:
            event = hp.heappop(self.module.HSI_EVENT_QUEUE)

            if self.sim.date > event.tclose:
                self.module.call_and_record_never_ran_hsi_event(hsi_event=event.hsi_event, priority=event.priority)
            elif not is_alive[event.hsi_event.target]:
                continue
            elif self.sim.date < event.topen:
                hp.heappush(self.module.HSI_EVENT_QUEUE, event)
                break
            else:
                due_today.append(event)

        return due_today

    def _check_climate_disruption(self, item: HSIEventQueueItem, hold_over: List[HSIEventQueueItem]) -> bool:
        year = self.sim.date.year
        month = self.sim.date.month

        climate_disrupted = False

        if (
            year >= self.module.parameters["year_effective_climate_disruptions"]
            and self.module.parameters["services_affected_precip"] != "none"
            and self.module.parameters["services_affected_precip"] is not None
        ):
            fac_level = item.hsi_event.facility_info.level
            facility_used = self.sim.population.props.at[item.hsi_event.target, f"level_{fac_level}"]
            if (
                facility_used
                in self.module.parameters["projected_precip_disruptions"]["RealFacility_ID"].values
            ):
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
                prob_disruption = min(
                    float(prob_disruption.iloc[0]) * self.module.parameters["scale_factor_prob_disruption"], 1
                )
                if np.random.binomial(1, prob_disruption) == 1:
                    climate_disrupted = True
                    if np.random.binomial(1, self.module.parameters["prop_supply_side_disruptions"]) and self.module.parameters["mode_appt_constraints"] == 2:
                        footprint = item.hsi_event.expected_time_requests
                        self.module.running_total_footprint.update(footprint)
                    if self.sim.modules[
                        "HealthSeekingBehaviour"
                    ].force_any_symptom_to_lead_to_healthcareseeking:
                        self.sim.modules["HealthSystem"]._add_hsi_event_queue_item_to_hsi_event_queue(
                            priority=item.priority,
                            clinic_eligibility=item.clinic_eligibility,
                            topen=self.sim.date
                                  + DateOffset(
                                days=(
                                    int(
                                        max( self.module.parameters["scale_factor_appointment_urgency"] * item.priority, 1)
                                        * prob_disruption
                                        * self.module.parameters["scale_factor_severity_disruption_and_delay"]
                                        * self.module.parameters["delay_in_seeking_care_weather"]
                                    )
                                )
                            ),
                            tclose=self.sim.date
                                   + DateOffset(
                                days=(
                                    int(
                                        max( self.module.parameters["scale_factor_appointment_urgency"] * item.priority, 1)
                                        * prob_disruption
                                        * self.module.parameters["scale_factor_severity_disruption_and_delay"]
                                        * self.module.parameters["delay_in_seeking_care_weather"]
                                    )
                                )
                            )
                                   + DateOffset((item.topen - item.tclose).days),
                            hsi_event=item.hsi_event
                        )
                        self.module.call_and_record_weather_delayed_hsi_event(
                            hsi_event=item.hsi_event, priority=item.priority, real_facility_id=facility_used
                        )
                    else:
                        patient = self.sim.population.props.loc[[item.hsi_event.target]]
                        if patient.age_years.iloc[0] < 15:
                            subgroup_name = "children"
                            care_seeking_odds_ratios = self.sim.modules[
                                "HealthSeekingBehaviour"
                            ].odds_ratio_health_seeking_in_children
                            hsb_model = self.sim.modules["HealthSeekingBehaviour"].hsb_linear_models["children"]
                        else:
                            subgroup_name = "adults"
                            care_seeking_odds_ratios = self.sim.modules[
                                "HealthSeekingBehaviour"
                            ].odds_ratio_health_seeking_in_adults
                            hsb_model = self.sim.modules["HealthSeekingBehaviour"].hsb_linear_models["adults"]

                        will_seek_care_prob = min(
                            self.module.parameters["scale_factor_reseeking_healthcare_post_disruption"]
                            * hsb_model.predict(
                                df=patient,
                                subgroup=subgroup_name,
                                care_seeking_odds_ratios=care_seeking_odds_ratios,
                            ).iloc[0],
                            1,
                        )

                        will_seek_care = 0
                        if np.random.random() < will_seek_care_prob:
                            will_seek_care = 1
                        if will_seek_care:
                            self.sim.modules["HealthSystem"]._add_hsi_event_queue_item_to_hsi_event_queue(
                                priority=item.priority,
                                clinic_eligibility=item.clinic_eligibility,
                                topen=self.sim.date
                                      + DateOffset(
                                    days=(
                                        int(
                                            max(self.module.parameters[
                                                    "scale_factor_appointment_urgency"] * item.priority, 1)
                                            * prob_disruption
                                            * self.module.parameters["scale_factor_severity_disruption_and_delay"]
                                            * self.module.parameters["delay_in_seeking_care_weather"]
                                        )
                                    )
                                ),
                                tclose=self.sim.date
                                       + DateOffset(
                                    days=(
                                        int(
                                            max(self.module.parameters[
                                                    "scale_factor_appointment_urgency"] * item.priority, 1)
                                            * prob_disruption
                                            * self.module.parameters["scale_factor_severity_disruption_and_delay"]
                                            * self.module.parameters["delay_in_seeking_care_weather"]
                                        )
                                    )
                                )
                                       + DateOffset((item.topen - item.tclose).days),
                                hsi_event=item.hsi_event,
                            )
                            self.module.call_and_record_weather_delayed_hsi_event(
                                hsi_event=item.hsi_event, priority=item.priority, real_facility_id=facility_used
                            )
                        else:
                            self.module.call_and_record_weather_cancelled_hsi_event(
                                hsi_event=item.hsi_event, priority=item.priority, real_facility_id=facility_used,
                            )

        return climate_disrupted

    def process_events_mode_1(self, hold_over: List[HSIEventQueueItem]) -> None:
        while True:
            list_of_individual_hsi_event_tuples_due_today = self._get_events_due_today()

            if not list_of_individual_hsi_event_tuples_due_today:
                break

            list_of_individual_hsi_event_tuples_due_today_that_meet_all_conditions = []

            for item in list_of_individual_hsi_event_tuples_due_today:
                climate_disrupted = self._check_climate_disruption(item, hold_over)
                if not climate_disrupted:
                    equipment_available = True
                    if not item.hsi_event.is_all_declared_equipment_available:
                        self.module.call_and_record_never_ran_hsi_event(
                            hsi_event=item.hsi_event, priority=item.priority
                        )
                        equipment_available = False

                    if equipment_available:
                        list_of_individual_hsi_event_tuples_due_today_that_meet_all_conditions.append(item)

            _to_be_held_over = self.module.run_individual_level_events_in_mode_1(
                list_of_individual_hsi_event_tuples_due_today_that_meet_all_conditions,
            )
            hold_over.extend(_to_be_held_over)

    def process_events_mode_2(self, hold_over: List[HSIEventQueueItem]) -> None:
        capabilities_monitor = {
            clinic: Counter(clinic_cl) for clinic, clinic_cl in self.module.capabilities_today.items()
        }
        set_capabilities_still_available = defaultdict(set)

        for clinic_name, clinic_val in capabilities_monitor.items():
            for facility_officer_id, facility_officer_id_capabilities in clinic_val.items():
                if facility_officer_id_capabilities > 0:
                    set_capabilities_still_available[clinic_name].add(facility_officer_id)

        alive_persons = set(self.sim.population.props.index[self.sim.population.props.is_alive].to_list())

        list_of_events_not_due_today = list()

        while len(self.module.HSI_EVENT_QUEUE) > 0:
            if len(set_capabilities_still_available) > 0:
                next_event_tuple = hp.heappop(self.module.HSI_EVENT_QUEUE)

                event = next_event_tuple.hsi_event
                event_clinic = next_event_tuple.clinic_eligibility
                capabilities_still_available = set_capabilities_still_available[event_clinic]

                if self.sim.date > next_event_tuple.tclose:
                    self.module.call_and_record_never_ran_hsi_event(
                        hsi_event=event,
                        priority=next_event_tuple.priority
                    )
                elif event.target not in alive_persons:
                    pass
                elif self.sim.date < next_event_tuple.topen:
                    hp.heappush(list_of_events_not_due_today, next_event_tuple)
                    if next_event_tuple.priority == self.module.lowest_priority_considered:
                        break
                else:
                    original_call = next_event_tuple.hsi_event.expected_time_requests
                    _priority = next_event_tuple.priority
                    climate_disrupted = self._check_climate_disruption(next_event_tuple, hold_over)

                    if not climate_disrupted:
                        out_of_resources = False
                        for officer, call in original_call.items():
                            if officer not in capabilities_still_available:
                                out_of_resources = True

                        if out_of_resources:
                            rtn_from_did_not_run = event.did_not_run()
                            if rtn_from_did_not_run is not False:
                                hp.heappush(hold_over, next_event_tuple)

                        self.module.record_hsi_event(
                            hsi_event=event,
                            actual_appt_footprint=event.EXPECTED_APPT_FOOTPRINT,
                            did_run=False,
                            priority=_priority,
                            clinic=event_clinic,
                        )
                    else:
                        if sum(event.BEDDAYS_FOOTPRINT.values()):
                            event._received_info_about_bed_days = (
                                self.module.bed_days.issue_bed_days_according_to_availability(
                                    facility_id=self.module.bed_days.get_facility_id_for_beds(
                                        persons_id=event.target),
                                    footprint=event.BEDDAYS_FOOTPRINT,
                                )
                            )

                        assert event.facility_info is not None, (
                            f"Cannot run HSI {event.TREATMENT_ID} without facility_info being defined."
                        )

                        if not event.is_all_declared_equipment_available:
                            self.module.call_and_record_never_ran_hsi_event(
                                hsi_event=event,
                                priority=next_event_tuple.priority,
                                clinic=next_event_tuple.clinic_eligibility,
                            )
                            continue

                        _appt_footprint_before_running = event.EXPECTED_APPT_FOOTPRINT
                        actual_appt_footprint = event.run(squeeze_factor=0.0)

                        if actual_appt_footprint is not None:
                            assert self.module.appt_footprint_is_valid(actual_appt_footprint)
                            updated_call = self.module.get_appt_footprint_as_time_request(
                                facility_info=event.facility_info,
                                appt_footprint=actual_appt_footprint
                            )
                        else:
                            actual_appt_footprint = _appt_footprint_before_running
                            updated_call = original_call

                        capabilities_monitor[event_clinic].subtract(updated_call)

                        for officer, call in updated_call.items():
                            if capabilities_monitor[event_clinic][officer] <= 0:
                                if officer in capabilities_still_available:
                                    capabilities_still_available.remove(officer)
                                else:
                                    logger.warning(
                                        key="message",
                                        data=(
                                            f"{event.TREATMENT_ID} actual_footprint requires different"
                                            f"officers than expected_footprint."
                                        ),
                                    )

                        self.module.running_total_footprint[event_clinic].update(updated_call)

                        self.module.record_hsi_event(
                            hsi_event=event,
                            actual_appt_footprint=actual_appt_footprint,
                            did_run=True,
                            priority=_priority,
                            clinic=event_clinic,
                        )
            else:
                break

        while len(self.module.HSI_EVENT_QUEUE) > 0:
            next_event_tuple = hp.heappop(self.module.HSI_EVENT_QUEUE)
            event = next_event_tuple.hsi_event

            if next_event_tuple.priority > self.module.lowest_priority_considered:
                self.module.schedule_to_call_never_ran_on_date(hsi_event=event, tdate=next_event_tuple.tclose)
            elif self.sim.date > next_event_tuple.tclose:
                self.module.call_and_record_never_ran_hsi_event(
                    hsi_event=event,
                    priority=next_event_tuple.priority,
                    clinic=next_event_tuple.clinic_eligibility
                )
            elif event.target not in alive_persons:
                pass
            elif self.sim.date < next_event_tuple.topen:
                hp.heappush(list_of_events_not_due_today, next_event_tuple)
            else:
                rtn_from_did_not_run = event.did_not_run()
                if rtn_from_did_not_run is not False:
                    hp.heappush(hold_over, next_event_tuple)

                self.module.record_hsi_event(
                    hsi_event=event,
                    actual_appt_footprint=event.EXPECTED_APPT_FOOTPRINT,
                    did_run=False,
                    priority=next_event_tuple.priority,
                    clinic=next_event_tuple.clinic_eligibility,
                )

        while len(list_of_events_not_due_today) > 0:
            hp.heappush(self.module.HSI_EVENT_QUEUE, hp.heappop(list_of_events_not_due_today))

    def apply(self, population):
        self.module.bed_days.on_start_of_day()
        self.module.consumables.on_start_of_day(self.sim.date)

        inpatient_appts = self.module.bed_days.get_inpatient_appts()
        inpatient_footprints = Counter()
        for _fac_id, _footprint in inpatient_appts.items():
            inpatient_footprints.update(
                self.module.get_appt_footprint_as_time_request(
                    facility_info=self.module._facility_by_facility_id[_fac_id], appt_footprint=_footprint
                )
            )

        if len(inpatient_appts):
            for _fac_id, _inpatient_appts in inpatient_appts.items():
                self.module.write_to_hsi_log(
                    event_details=HSIEventDetails(
                        event_name="Inpatient_Care",
                        module_name="HealthSystem",
                        treatment_id="Inpatient_Care",
                        facility_level=self.module._facility_by_facility_id[_fac_id].level,
                        appt_footprint=tuple(sorted(_inpatient_appts.items())),
                        beddays_footprint=(),
                        equipment=tuple(),
                    ),
                    person_id=-1,
                    facility_id=_fac_id,
                    priority=-1,
                    clinic=str(None),
                    did_run=True,
                )

        self.module.running_total_footprint["GenericClinic"] = inpatient_footprints
        hold_over = list()

        if self.module.mode_appt_constraints == 1:
            self.process_events_mode_1(hold_over)
        elif self.module.mode_appt_constraints == 2:
            self.process_events_mode_2(hold_over)

        while len(hold_over) > 0:
            hp.heappush(self.module.HSI_EVENT_QUEUE, hp.heappop(hold_over))

        for clinic in self.module._clinic_names:
            if clinic not in self.module.capabilities_today:
                continue
            self.module.log_current_capabilities_and_usage(clinic)

        self.module.on_end_of_day()

        if self._is_last_day_of_the_month(self.sim.date):
            self.module.on_end_of_month()

        if self._is_last_day_of_the_year(self.sim.date):
            self.module.on_end_of_year()


# ---------------------------------------------------------------------------
#   Logging
# ---------------------------------------------------------------------------


class HealthSystemSummaryCounter:
    """Helper class to keep running counts of HSI and the state of the HealthSystem and logging summaries."""

    def __init__(self):
        self._reset_internal_stores()

    def _reset_internal_stores(self) -> None:
        """Create empty versions of the data structures used to store a running records."""

        self._treatment_ids = defaultdict(int)
        self._appts = defaultdict(int)
        self._appts_by_level = {_level: defaultdict(int) for _level in ("0", "1a", "1b", "2", "3", "4")}

        self._no_blank_appt_treatment_ids = defaultdict(int)
        self._no_blank_appt_appts = defaultdict(int)
        self._no_blank_appt_by_level = {_level: defaultdict(int) for _level in ("0", "1a", "1b", "2", "3", "4")}

        self._never_ran_treatment_ids = defaultdict(int)
        self._never_ran_appts = defaultdict(int)
        self._never_ran_appts_by_level = {_level: defaultdict(int) for _level in ("0", "1a", "1b", "2", "3", "4")}

        self._frac_time_used_overall = defaultdict(list)
        self._sum_of_daily_frac_time_used_by_facID_and_officer = defaultdict(Counter)

        self._weather_cancelled_treatment_ids = defaultdict(int)
        self._weather_cancelled_appts = defaultdict(int)
        self._weather_cancelled_appts_by_level = {
            _level: defaultdict(int) for _level in ("0", "1a", "1b", "2", "3", "4")
        }

        self._weather_delayed_treatment_ids = defaultdict(int)
        self._weather_delayed_appts = defaultdict(int)
        self._weather_delayed_appts_by_level = {_level: defaultdict(int) for _level in ("0", "1a", "1b", "2", "3", "4")}

        self._squeeze_factor_by_hsi_event_name = defaultdict(list)

        # *** NEW: per-facility counts ***
        self._hsi_by_real_facility = defaultdict(int)
        self._weather_cancelled_by_real_facility = defaultdict(int)
        self._weather_delayed_by_real_facility = defaultdict(int)

    def record_hsi_event(
        self, treatment_id: str, hsi_event_name: str, appt_footprint: Counter, level: str,
        real_facility_id: Optional[str] = None
    ) -> None:
        """Add information about an `HSI_Event` to the running summaries."""
        self._treatment_ids[treatment_id] += 1
        for appt_type, number in appt_footprint:
            self._appts[appt_type] += number
            self._appts_by_level[level][appt_type] += number
        if len(appt_footprint):
            self._no_blank_appt_treatment_ids[treatment_id] += 1
            for appt_type, number in appt_footprint:
                self._no_blank_appt_appts[appt_type] += number
                self._no_blank_appt_by_level[level][appt_type] += number
        # *** NEW ***
        if real_facility_id and real_facility_id != 'unknown':
            self._hsi_by_real_facility[real_facility_id] += 1

    def record_never_ran_hsi_event(
        self, treatment_id: str, hsi_event_name: str, appt_footprint: Counter, level: str
    ) -> None:
        """Add information about a never-ran `HSI_Event` to the running summaries."""
        self._never_ran_treatment_ids[treatment_id] += 1
        for appt_type, number in appt_footprint:
            self._never_ran_appts[appt_type] += number
            self._never_ran_appts_by_level[level][appt_type] += number

    def record_weather_cancelled_hsi_event(
        self, treatment_id: str, hsi_event_name: str, appt_footprint: Counter, level: str,
        real_facility_id: Optional[str] = None
    ) -> None:
        """Add information about a weather-cancelled `HSI_Event` to the running summaries."""
        self._weather_cancelled_treatment_ids[treatment_id] += 1
        for appt_type, number in appt_footprint:
            self._weather_cancelled_appts[appt_type] += number
            self._weather_cancelled_appts_by_level[level][appt_type] += number
        # *** NEW ***
        if real_facility_id and real_facility_id != 'unknown':
            self._weather_cancelled_by_real_facility[real_facility_id] += 1

    def record_weather_delayed_hsi_event(
        self, treatment_id: str, hsi_event_name: str, appt_footprint: Counter, level: str,
        real_facility_id: Optional[str] = None
    ) -> None:
        """Add information about a weather-delayed `HSI_Event` to the running summaries."""
        self._weather_delayed_treatment_ids[treatment_id] += 1
        for appt_type, number in appt_footprint:
            self._weather_delayed_appts[appt_type] += number
            self._weather_delayed_appts_by_level[level][appt_type] += number
        # *** NEW ***
        if real_facility_id and real_facility_id != 'unknown':
            self._weather_delayed_by_real_facility[real_facility_id] += 1

    def record_hs_status(
        self,
        fraction_time_used_across_all_facilities_in_this_clinic: float,
        fraction_time_used_by_facID_and_officer_in_this_clinic: Dict[str, float],
        clinic: str
    ) -> None:
        """Record a current status metric of the HealthSystem."""
        self._frac_time_used_overall[clinic].append(fraction_time_used_across_all_facilities_in_this_clinic)

        for facID_and_officer, fraction_time in fraction_time_used_by_facID_and_officer_in_this_clinic.items():
            self._sum_of_daily_frac_time_used_by_facID_and_officer[clinic][facID_and_officer] += fraction_time

    def write_to_log_and_reset_counters(self):
        """Log summary statistics reset the data structures. This usually occurs at the end of the year."""
        logger_summary.info(
            key="HSI_Event",
            description="Counts of the HSI_Events that have occurred in this calendar year by TREATMENT_ID, "
            "and counts of the 'Appt_Type's that have occurred in this calendar year."
            "Squeeze factors are always assumed to be 0.",
            data={
                "TREATMENT_ID": self._treatment_ids,
                "Number_By_Appt_Type_Code": self._appts,
                "Number_By_Appt_Type_Code_And_Level": self._appts_by_level,
                "squeeze_factor": {t_id: 0.0 for t_id, v in self._treatment_ids.items()},
                "Number_By_RealFacility_ID": self._hsi_by_real_facility,  # *** NEW ***
            },
        )
        logger_summary.info(
            key="HSI_Event_non_blank_appt_footprint",
            description="Same as for key 'HSI_Event' but limited to HSI_Event that have non-blank footprints",
            data={
                "TREATMENT_ID": self._no_blank_appt_treatment_ids,
                "Number_By_Appt_Type_Code": self._no_blank_appt_appts,
                "Number_By_Appt_Type_Code_And_Level": self._no_blank_appt_by_level,
            },
        )

        logger_summary.info(
            key="Never_ran_HSI_Event",
            description="Counts of the HSI_Events that never ran in this calendar year by TREATMENT_ID, "
            "and the respective 'Appt_Type's that have not occurred in this calendar year.",
            data={
                "TREATMENT_ID": self._never_ran_treatment_ids,
                "Number_By_Appt_Type_Code": self._never_ran_appts,
                "Number_By_Appt_Type_Code_And_Level": self._never_ran_appts_by_level,
            },
        )
        logger_summary.info(
            key="Weather_cancelled_HSI_Event",
            description="Counts of the HSI_Events that were cancelled due to weather in this calendar year by TREATMENT_ID, "
                        "and the respective 'Appt_Type's that were cancelled in this calendar year.",
            data={
                "TREATMENT_ID": self._weather_cancelled_treatment_ids,
                "Number_By_Appt_Type_Code": self._weather_cancelled_appts,
                "Number_By_Appt_Type_Code_And_Level": self._weather_cancelled_appts_by_level,
                "Number_By_RealFacility_ID": self._weather_cancelled_by_real_facility,  # *** NEW ***
            },
        ),
        logger_summary.info(
            key="Weather_delayed_HSI_Event",
            description="Counts of the HSI_Events that were delayed due to weather in this calendar year by TREATMENT_ID, "
                        "and the respective 'Appt_Type's that were delayed in this calendar year.",
            data={
                "TREATMENT_ID": self._weather_delayed_treatment_ids,
                "Number_By_Appt_Type_Code": self._weather_delayed_appts,
                "Number_By_Appt_Type_Code_And_Level": self._weather_delayed_appts_by_level,
                "Number_By_RealFacility_ID": self._weather_delayed_by_real_facility,  # *** NEW ***
            },
        ),

        logger_summary.info(
            key="Capacity",
            description="The fraction of all the healthcare worker time that is used each day, averaged over this "
            "calendar year.",
            data={
                "average_Frac_Time_Used_Overall": {
                    clinic: np.mean(values) for clinic, values in self._frac_time_used_overall.items()
                },
            },
        )

        for clinic in self._frac_time_used_overall.keys():
            logger_summary.info(
               key="Capacity_By_FacID_and_Officer",
                description="The fraction of healthcare worker time that is used each day, averaged over this "
                "calendar year, for each officer type at each facility.",
                data=flatten_multi_index_series_into_dict_for_logging(self.frac_time_used_by_facID_and_officer(clinic))
            )

        self._reset_internal_stores()

    def frac_time_used_by_facID_and_officer(
        self,
        clinic: str,
        facID_and_officer: Optional[str] = None,
    ) -> Union[float, pd.Series]:
        if facID_and_officer is not None:
            return (
                self._sum_of_daily_frac_time_used_by_facID_and_officer[clinic][facID_and_officer]
                / len(self._frac_time_used_overall[clinic])
            )
        else:
            mean_frac_time_used = {
                (_facID_and_officer): v / len(self._frac_time_used_overall[clinic])
                for (_facID_and_officer), v in self._sum_of_daily_frac_time_used_by_facID_and_officer[clinic].items()
            }
            return pd.Series(
                index=pd.MultiIndex.from_tuples([(clinic, key) for key in mean_frac_time_used.keys()],
                                                names=["clinic", "facID_and_officer"]),
                data=mean_frac_time_used.values(),
            ).sort_index()


class HealthSystemChangeParameters(Event, PopulationScopeEventMixin):
    """Event that causes certain internal parameters of the HealthSystem to be changed."""

    def __init__(self, module: HealthSystem, parameters_to_change: List):
        super().__init__(module)
        assert isinstance(module, HealthSystem)

        self.supported_parameters = ["cons_availability", "equip_availability", "use_funded_or_actual_staffing"]
        if not all(param in self.supported_parameters for param in parameters_to_change):
            raise ValueError(
                f"parameters_to_change can only contain the following values: {self.supported_parameters}. "
                f"Received: {parameters_to_change}"
            )

        self.parameters_to_change = parameters_to_change

    def apply(self, population):
        p = self.module.parameters

        if "cons_availability" in self.parameters_to_change:
            self.module.consumables.availability = p["cons_availability_postSwitch"]

        if "equip_availability" in self.parameters_to_change:
            self.module.equipment.availability = p["equip_availability_postSwitch"]

        if "use_funded_or_actual_staffing" in self.parameters_to_change:
            self.module.use_funded_or_actual_staffing = p["use_funded_or_actual_staffing_postSwitch"]


class DynamicRescalingHRCapabilities(RegularEvent, PopulationScopeEventMixin):
    """This event exists to scale the daily capabilities assumed at fixed time intervals"""

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(years=1))
        self.last_year_pop_size = self.current_pop_size

        self.scaling_values = (
            self.module.parameters["yearly_HR_scaling"][self.module.parameters["yearly_HR_scaling_mode"]]
            .set_index("year")
            .to_dict("index")
        )

    @property
    def current_pop_size(self) -> float:
        df = self.sim.population.props
        return df.is_alive.sum()

    def _get_most_recent_year_specified_for_a_change_in_configuration(self) -> int:
        years = np.array(list(self.scaling_values.keys()))
        return years[years <= self.sim.date.year].max()

    def apply(self, population):
        this_year_pop_size = self.current_pop_size
        config = self.scaling_values.get(self._get_most_recent_year_specified_for_a_change_in_configuration())

        for clinic_name, clinic_cl in self.module._daily_capabilities.items():
            for cl in clinic_cl:
                clinic_cl[cl] *= config["dynamic_HR_scaling_factor"]

        if config["scale_HR_by_popsize"]:
            for clinic_name, clinic_cl in self.module._daily_capabilities.items():
                for cl in clinic_cl:
                    clinic_cl[cl] *= this_year_pop_size / self.last_year_pop_size

        self.last_year_pop_size = this_year_pop_size


class ConstantRescalingHRCapabilities(Event, PopulationScopeEventMixin):
    """This event exists to scale the daily capabilities, with a factor for each Officer Type at each Facility_Level."""

    def __init__(self, module):
        super().__init__(module)

    def apply(self, population):
        HR_scaling_by_level_and_officer_type_factor = self.module.parameters[
            "HR_scaling_by_level_and_officer_type_table"
        ][self.module.parameters["HR_scaling_by_level_and_officer_type_mode"]].set_index("Officer_Category")

        pattern = r"FacilityID_(\w+)_Officer_(\w+)"

        for clinic, clinic_cl in self.module._daily_capabilities.items():
            for officer in clinic_cl.keys():
                matches = re.match(pattern, officer)
                facility_id = int(matches.group(1))
                officer_type = matches.group(2)
                level = self.module._facility_by_facility_id[facility_id].level
                self.module._daily_capabilities[clinic][officer] *= HR_scaling_by_level_and_officer_type_factor.at[
                    officer_type, f"L{level}_factor"
                ]


class RescaleHRCapabilities_ByDistrict(Event, PopulationScopeEventMixin):
    """This event exists to scale the daily capabilities, with a factor for each district."""

    def __init__(self, module):
        super().__init__(module)

    def apply(self, population):
        HR_scaling_factor_by_district = (
            self.module.parameters["HR_scaling_by_district_table"][
                self.module.parameters["HR_scaling_by_district_mode"]
            ]
            .set_index("District")
            .to_dict()
        )

        pattern = r"FacilityID_(\w+)_Officer_(\w+)"
        for clinic, clinic_cl in self.module._daily_capabilities.items():
            for officer in clinic_cl.keys():
                matches = re.match(pattern, officer)
                facility_id = int(matches.group(1))
                district = self.module._facility_by_facility_id[facility_id].district
                if district in HR_scaling_factor_by_district:
                    self.module._daily_capabilities[clinic][officer] *= HR_scaling_factor_by_district[district]


class HealthSystemChangeMode(RegularEvent, PopulationScopeEventMixin):
    """This event exists to change the priority policy adopted by the HealthSystem at a given year."""

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(years=100))

    def apply(self, population):
        health_system: HealthSystem = self.module
        preswitch_mode = health_system.mode_appt_constraints

        health_system.mode_appt_constraints = health_system.parameters["mode_appt_constraints_postSwitch"]

        if preswitch_mode == 1 and health_system.mode_appt_constraints == 2:
            updated_events: List[HSIEventQueueItem | None] = [None] * len(health_system.HSI_EVENT_QUEUE)
            offset = 0

            while health_system.HSI_EVENT_QUEUE:
                event = hp.heappop(health_system.HSI_EVENT_QUEUE)
                clinic_eligibility = health_system.get_clinic_eligibility(event.hsi_event.TREATMENT_ID)
                enforced_priority = health_system.enforce_priority_policy(event.hsi_event)

                if event.priority != enforced_priority:
                    event = HSIEventQueueItem(
                        clinic_eligibility,
                        enforced_priority,
                        event.topen,
                        event.rand_queue_counter,
                        event.queue_counter,
                        event.tclose,
                        event.hsi_event,
                    )

                updated_events[offset] = event
                offset += 1

            while updated_events:
                hp.heappush(health_system.HSI_EVENT_QUEUE, updated_events.pop())

            del updated_events

        logger.info(
            key="message",
            data=f"Switched mode at sim date: {self.sim.date}Now using mode: {self.module.mode_appt_constraints}",
        )


class HealthSystemLogger(RegularEvent, PopulationScopeEventMixin):
    """This event runs at the start of each year and does any logging jobs for the HealthSystem module."""

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(years=1))

    def apply(self, population):
        self.log_number_of_staff()

    def log_number_of_staff(self):
        hs = self.module

        current_staff_count = {}
        for clinic in sorted(hs._daily_capabilities):
            current_staff_count[clinic] = {}
            for fid in sorted(hs._daily_capabilities[clinic]):
                denom = hs._daily_capabilities_per_staff[clinic][fid]
                if denom == 0:
                    current_staff_count[clinic][fid] = 0
                else:
                    current_staff_count[clinic][fid] = hs._daily_capabilities[clinic][fid] / denom

        logger_summary.info(
            key="number_of_hcw_staff",
            description="The number of hcw_staff this year",
            data=current_staff_count,
        )

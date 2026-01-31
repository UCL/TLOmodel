"""
The HIV Module
Overview:
HIV infection ---> AIDS onset Event (defined by the presence of AIDS symptoms) --> AIDS Death Event
Testing is spontaneously taken-up and can lead to accessing intervention services (ART, VMMC, PrEP).
AIDS symptoms can also lead to care-seeking and there is routine testing for HIV at all non-emergency Generic HSI
 events.
Persons can be on ART -
    - with viral suppression: when the person with not develop AIDS, or if they have already, it is relieved and they
        will not die of AIDS; and the person is not infectious
    - without viral suppression: when there is no benefit in avoiding AIDS and infectiousness is unchanged.
Maintenance on ART and PrEP is re-assessed at periodic 'Decision Events', at which it is determined if the person
  will attend the "next" HSI for continuation of the service; and if not, they are removed from that service and "stop
  treatment". If a stock-out or non-availability of health system resources prevent treatment continuation, the person
  "stops treatment". Stopping treatment leads to a new AIDS Event being scheduled. Persons can restart treatment. If a
  person has developed AIDS, starts treatment and then defaults from treatment, their 'original' AIDS Death Event will
  still occur.
If PrEP is not available due to limitations in the HealthSystem, the person defaults to not being on PrEP.
# Things to note:
    * Need to incorporate testing for HIV at first ANC appointment (as it does in generic HSI)
    * Need to incorporate testing for infants born to HIV-positive mothers (currently done in on_birth here).
    * Need to incorporate cotrim for infants born to HIV-positive mothers (not done here)
    * Cotrimoxazole is not included - either in effect of consumption of the drug (because the effect is not known).
    * Calibration has not been done: most things look OK - except HIV-AIDS deaths
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd

from tlo import DAYS_IN_YEAR, Date, DateOffset, Module, Parameter, Property, Types, logging
from tlo.analysis.utils import (
    flatten_multi_index_series_into_dict_for_logging,
    get_counts_by_sex_and_age_group,
)
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, demography, tb
from tlo.methods.causes import Cause
from tlo.methods.dxmanager import DxTest
from tlo.methods.hsi_event import HSI_Event
from tlo.methods.hsi_generic_first_appts import GenericFirstAppointmentsMixin
from tlo.methods.symptommanager import Symptom
from tlo.util import create_age_range_lookup, read_csv_files

if TYPE_CHECKING:
    from tlo.methods.hsi_generic_first_appts import HSIEventScheduler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Hiv(Module, GenericFirstAppointmentsMixin):
    """
    The HIV Disease Module
    """

    def __init__(self, name=None, run_with_checks=False):
        super().__init__(name)

        assert isinstance(run_with_checks, bool)
        self.run_with_checks = run_with_checks

        self.stored_test_numbers = []  # create empty list for storing hiv test numbers
        self.stored_tdf_numbers = 0  # create container for storing numbers of tdf tests
        self.stored_selftest_numbers = 0 # create container for storing numbers of hiv self tests

        # hiv outputs needed for calibration
        keys = ["date",
                "hiv_prev_adult_1549",
                "hiv_adult_inc_1549",
                "hiv_prev_child",
                "population"
                ]
        # initialise empty dict with set keys
        self.hiv_outputs = {k: [] for k in keys}

        self.daly_wts = dict()
        self.lm = dict()
        self.item_codes_for_consumables_required = dict()
        # create dict of years (keys) where VL testing is available (values=True / False)
        self.vl_testing_available_by_year: dict[int, bool] = {}

    INIT_DEPENDENCIES = {"Demography", "HealthSystem", "Lifestyle", "SymptomManager"}

    OPTIONAL_INIT_DEPENDENCIES = {"HealthBurden"}

    ADDITIONAL_DEPENDENCIES = {'Tb', 'NewbornOutcomes'}

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN,
        Metadata.REPORTS_DISEASE_NUMBERS
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        "AIDS_non_TB": Cause(gbd_causes="HIV/AIDS", label="AIDS"),
        "AIDS_TB": Cause(gbd_causes="HIV/AIDS", label="AIDS"),
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        "HIV": Cause(gbd_causes="HIV/AIDS", label="AIDS"),
    }

    PROPERTIES = {
        # --- Core Properties
        "hv_inf": Property(
            Types.BOOL,
            "Is person currently infected with HIV (NB. AIDS status is determined by presence of the AIDS Symptom.",
        ),
        "hv_art": Property(
            Types.CATEGORICAL,
            "ART status of person, whether on ART or not; and whether viral load is suppressed or not if on ART.",
            categories=["not", "on_VL_suppressed", "on_not_VL_suppressed"],
        ),
        "hv_on_cotrimoxazole": Property(
            Types.BOOL,
            "Whether the person is currently taking and receiving a malaria-protective effect from cotrimoxazole",
        ),
        "hv_is_on_prep_oral": Property(
            Types.BOOL,
            "Whether the person is currently taking and receiving a protective effect "
            "from oral Pre-Exposure Prophylaxis",
        ),
        "hv_is_on_prep_inj": Property(
            Types.BOOL,
            "Whether the person is currently taking and receiving a protective effect "
            "from injectable Pre-Exposure Prophylaxis",
        ),
        "hv_behaviour_change": Property(
            Types.BOOL,
            "Has this person been exposed to HIV prevention counselling following a negative HIV test result",
        ),
        "hv_diagnosed": Property(
            Types.BOOL, "Knows that they are HIV+: i.e. is HIV+ and tested as HIV+"
        ),
        "hv_number_tests": Property(Types.INT, "Number of HIV tests ever taken"),
        "hv_arv_dispensing_interval": Property(Types.REAL,
                                               "Number of months of ARVs dispensed in current prescription"),
        # --- Dates on which things have happened:
        "hv_last_test_date": Property(Types.DATE, "Date of last HIV test"),
        "hv_date_inf": Property(Types.DATE, "Date infected with HIV"),
        "hv_date_treated": Property(Types.DATE, "date hiv treatment started"),
        "hv_date_last_ART": Property(Types.DATE, "date of last ART dispensation"),
        # --- Counters for logging:
        "hv_VMMC_in_last_year": Property(Types.BOOL, "whether VMMC was performed in the last year"),
        "hv_days_on_oral_prep_AGYW": Property(Types.INT,
                                              "number of days spent on oral prep per "
                                              "logger time interval for AGYW"),
        "hv_days_on_oral_prep_FSW": Property(Types.INT,
                                             "number of days spent on oral prep per "
                                             "logger time interval for FSW"),
        "hv_days_on_inj_prep_AGYW": Property(Types.INT, "number of days spent on inj prep per "
                                                        "logger time interval for AGYW"),
        "hv_days_on_inj_prep_FSW": Property(Types.INT, "number of days spent on inj prep per "
                                                       "logger time interval for FSW"),
    }

    PARAMETERS = {
        # Baseline characteristics
        "time_inf": Parameter(
            Types.DATA_FRAME, "prob of time since infection for baseline adult pop"
        ),
        "art_coverage": Parameter(Types.DATA_FRAME, "coverage of ART at baseline"),
        "treatment_cascade": Parameter(Types.DATA_FRAME, "spectrum estimates of treatment cascade"),
        # Natural history - transmission - overall rates
        "beta": Parameter(Types.REAL, "Transmission rate"),
        "unaids_prevalence_adjustment_factor": Parameter(
            Types.REAL, "adjustment for baseline age-specific prevalence values to give correct population prevalence"
        ),
        "prob_mtct_untreated": Parameter(
            Types.REAL, "Probability of mother to child transmission"
        ),
        "prob_mtct_treated": Parameter(
            Types.REAL, "Probability of mother to child transmission, mother on ART"
        ),
        "prob_mtct_incident_preg": Parameter(
            Types.REAL,
            "Probability of mother to child transmission, mother infected during pregnancy",
        ),
        "monthly_prob_mtct_bf_untreated": Parameter(
            Types.REAL,
            "Probability of mother to child transmission during breastfeeding",
        ),
        "monthly_prob_mtct_bf_treated": Parameter(
            Types.REAL,
            "Probability of mother to child transmission, mother infected during breastfeeding",
        ),
        # Natural history - transmission - relative risk of HIV acquisition (non-intervention)
        "rr_fsw": Parameter(Types.REAL, "Relative risk of HIV with female sex work"),
        "rr_circumcision": Parameter(
            Types.REAL, "Relative risk of HIV with circumcision"
        ),
        "rr_rural": Parameter(Types.REAL, "Relative risk of HIV in rural location"),
        "rr_windex_poorer": Parameter(
            Types.REAL, "Relative risk of HIV with wealth level poorer"
        ),
        "rr_windex_middle": Parameter(
            Types.REAL, "Relative risk of HIV with wealth level middle"
        ),
        "rr_windex_richer": Parameter(
            Types.REAL, "Relative risk of HIV with wealth level richer"
        ),
        "rr_windex_richest": Parameter(
            Types.REAL, "Relative risk of HIV with wealth level richest"
        ),
        "rr_sex_f": Parameter(Types.REAL, "Relative risk of HIV if female"),
        "rr_edlevel_primary": Parameter(
            Types.REAL, "Relative risk of HIV with primary education"
        ),
        "rr_edlevel_secondary": Parameter(
            Types.REAL, "Relative risk of HIV with secondary education"
        ),
        "rr_edlevel_higher": Parameter(
            Types.REAL, "Relative risk of HIV with higher education"
        ),
        "rr_schisto": Parameter(
            Types.REAL, "Relative risk of HIV with high intensity S. haematobium infection"
        ),
        # Natural history - transmission - relative risk of HIV acquisition (interventions)
        "rr_behaviour_change": Parameter(
            Types.REAL, "Relative risk of HIV with behaviour modification"
        ),
        "proportion_reduction_in_risk_of_hiv_aq_if_on_oral_prep": Parameter(
            Types.REAL,
            "Proportion reduction in risk of HIV acquisition if on oral PrEP. "
            "0 for no efficacy; 1.0 for perfect efficacy.",
        ),
        "proportion_reduction_in_risk_of_hiv_aq_if_on_inj_prep": Parameter(
            Types.REAL,
            "Proportion reduction in risk of HIV acquisition if on inj PrEP. "
            "0 for no efficacy; 1.0 for perfect efficacy.",
        ),
        # Natural history - survival (adults)
        "min_months_between_aids_and_death": Parameter(
            Types.INT,
            "Minimum number of months for the time between AIDS and AIDS Death",
        ),
        "mean_months_between_aids_and_death": Parameter(
            Types.REAL,
            "Mean number of months (distributed exponentially) for the time between AIDS and AIDS Death",
        ),
        "mean_months_between_aids_and_death_infant": Parameter(
            Types.REAL,
            "Mean number of months for the time between AIDS and AIDS Death for infants",
        ),
        "infection_to_death_weibull_shape_1519": Parameter(
            Types.REAL,
            "Shape parameter for Weibull describing time between infection and death for 15-19 yo (units: years)",
        ),
        "infection_to_death_weibull_shape_2024": Parameter(
            Types.REAL,
            "Shape parameter for Weibull describing time between infection and death for 20-24 yo (units: years)",
        ),
        "infection_to_death_weibull_shape_2529": Parameter(
            Types.REAL,
            "Shape parameter for Weibull describing time between infection and death for 25-29 yo (units: years)",
        ),
        "infection_to_death_weibull_shape_3034": Parameter(
            Types.REAL,
            "Shape parameter for Weibull describing time between infection and death for 30-34 yo (units: years)",
        ),
        "infection_to_death_weibull_shape_3539": Parameter(
            Types.REAL,
            "Shape parameter for Weibull describing time between infection and death for 35-39 yo (units: years)",
        ),
        "infection_to_death_weibull_shape_4044": Parameter(
            Types.REAL,
            "Shape parameter for Weibull describing time between infection and death for 40-44 yo (units: years)",
        ),
        "infection_to_death_weibull_shape_4549": Parameter(
            Types.REAL,
            "Shape parameter for Weibull describing time between infection and death for 45-49 yo (units: years)",
        ),
        "infection_to_death_weibull_scale_1519": Parameter(
            Types.REAL,
            "Scale parameter for Weibull describing time between infection and death for 15-19 yo (units: years)",
        ),
        "infection_to_death_weibull_scale_2024": Parameter(
            Types.REAL,
            "Scale parameter for Weibull describing time between infection and death for 20-24 yo (units: years)",
        ),
        "infection_to_death_weibull_scale_2529": Parameter(
            Types.REAL,
            "Scale parameter for Weibull describing time between infection and death for 25-29 yo (units: years)",
        ),
        "infection_to_death_weibull_scale_3034": Parameter(
            Types.REAL,
            "Scale parameter for Weibull describing time between infection and death for 30-34 yo (units: years)",
        ),
        "infection_to_death_weibull_scale_3539": Parameter(
            Types.REAL,
            "Scale parameter for Weibull describing time between infection and death for 35-39 yo (units: years)",
        ),
        "infection_to_death_weibull_scale_4044": Parameter(
            Types.REAL,
            "Scale parameter for Weibull describing time between infection and death for 40-44 yo (units: years)",
        ),
        "infection_to_death_weibull_scale_4549": Parameter(
            Types.REAL,
            "Scale parameter for Weibull describing time between infection and death for 45-49 yo (units: years)",
        ),
        "art_default_to_aids_mean_years": Parameter(
            Types.REAL,
            "Mean years between when a person (any change) stops being on treatment to when AIDS is onset (if the "
            "absence of resuming treatment).",
        ),
        "prop_delayed_aids_onset": Parameter(
            Types.REAL,
            "Proportion of PLHIV that will have delayed AIDS onset to compensate for AIDS-TB",
        ),
        # Natural history - survival (children)
        "mean_survival_for_infants_infected_prior_to_birth": Parameter(
            Types.REAL,
            "Exponential rate parameter for mortality in infants who are infected before birth",
        ),
        "infection_to_death_infant_infection_after_birth_weibull_scale": Parameter(
            Types.REAL,
            "Weibull scale parameter for mortality in infants who are infected after birth",
        ),
        "infection_to_death_infant_infection_after_birth_weibull_shape": Parameter(
            Types.REAL,
            "Weibull shape parameter for mortality in infants who are infected after birth",
        ),
        # Uptake of Interventions
        "hiv_testing_rates": Parameter(
            Types.DATA_FRAME, "annual rates of testing for children and adults"
        ),
        "rr_test_hiv_positive": Parameter(
            Types.REAL,
            "relative likelihood of having HIV test for people with HIV",
        ),
        "hiv_testing_rate_adjustment": Parameter(
            Types.REAL,
            "adjustment to current testing rates to account for multiple routes into HIV testing",
        ),
        "prob_hiv_test_at_anc_or_delivery": Parameter(
            Types.REAL,
            "probability of a women having hiv test at anc or following delivery",
        ),
        "prob_hiv_test_for_newborn_infant": Parameter(
            Types.REAL,
            "probability of a newborn infant having HIV test pre-discharge",
        ),
        "prob_start_art_or_vs": Parameter(
            Types.REAL,
            "Probability that a person will start treatment and be virally suppressed following testing, "
            "from 2016 inputs reduced viral suppression rates to account for viral load testing effect",
        ),
        "prob_behav_chg_after_hiv_test": Parameter(
            Types.REAL,
            "Probability that a person will change risk behaviours, if HIV-negative, following testing",
        ),
        "prob_prep_for_fsw_after_hiv_test": Parameter(
            Types.REAL,
            "Probability that a FSW will start PrEP, if HIV-negative, following testing",
        ),
        "prob_prep_for_agyw": Parameter(
            Types.REAL,
            "Probability that adolescent girls / young women will start PrEP",
        ),
        "prob_circ_after_hiv_test": Parameter(
            Types.REAL,
            "Probability that a male will be circumcised, if HIV-negative, following testing",
        ),
        "increase_in_prob_circ_2019": Parameter(
            Types.REAL,
            "increase in probability that a male will be circumcised, if HIV-negative, following testing"
            "from 2019 onwards",
        ),
        "prob_circ_for_child_before_2020": Parameter(
            Types.REAL,
            "Probability that a male aging <15 yrs will be circumcised before year 2020",
        ),
        "prob_circ_for_child_from_2020": Parameter(
            Types.REAL,
            "Probability that a male aging <15 yrs will be circumcised from year 2020, "
            "which is different from before 2020 as children vmmc policy/fund/cases has changed, "
            "according to PEPFAR 2020 Country Operational Plan and DHIS2 data",
        ),
        "probability_of_being_retained_on_prep_every_3_months": Parameter(
            Types.REAL,
            "Probability that someone who has initiated on prep will attend an appointment and be on prep "
            "for the next 3 months, until the next appointment.",
        ),
        "probability_of_being_retained_on_art_every_3_months": Parameter(
            Types.REAL,
            "Probability that someone who has initiated on treatment will attend an appointment and be on "
            "treatment for next 3 months, until the next appointment.",
        ),
        "probability_of_seeking_further_art_appointment_if_drug_not_available": Parameter(
            Types.REAL,
            "Probability that a person who 'should' be on art will seek another appointment (the following "
            "day and try for each of the next 7 days) if drugs were not available.",
        ),
        "probability_of_seeking_further_art_appointment_if_appointment_not_available": Parameter(
            Types.REAL,
            "Probability that a person who 'should' be on art will seek another appointment if the health-"
            "system has not been able to provide them with an appointment",
        ),
        "prep_start_year": Parameter(Types.REAL, "Year from which PrEP is available"),
        "ART_age_cutoff_young_child": Parameter(
            Types.INT, "Age cutoff for ART regimen for young children"
        ),
        "ART_age_cutoff_older_child": Parameter(
            Types.INT, "Age cutoff for ART regimen for older children"
        ),
        "rel_probability_art_baseline_aids": Parameter(
            Types.REAL,
            "relative probability of person with HIV infection over 10 years being on ART at baseline",
        ),
        "aids_tb_death_rate_with_tb_treatment": Parameter(
            Types.REAL,
            "probability of death if aids and tb, person on treatment for tb",
        ),
        "aids_tb_death_rate_no_tb_treatment": Parameter(
            Types.REAL,
            "probability of death if aids and tb, person not on treatment for tb",
        ),
        "hiv_healthseekingbehaviour_cap": Parameter(
            Types.INT,
            "number of repeat visits assumed for healthcare services",
        ),
        "initial_dispensation_period_months": Parameter(
            Types.INT,
            "length of prescription for ARVs in months for prescriptions pre-2021",
        ),
        "dispensation_period_months": Parameter(
            Types.DATA_FRAME,
            "prescription length probabilities for ARVs in months, varies by age/sex/year",
        ),
        "dispensation_min_days": Parameter(
            Types.REAL,
            "Minimum days to attain ARV dispensation",
        ),
        "length_of_inpatient_stay_if_terminal": Parameter(
            Types.LIST,
            "length in days of inpatient stay for end-of-life HIV patients: list has two elements [low-bound-inclusive,"
            " high-bound-exclusive]",
        ),
        "aids_death_beddays_default": Parameter(
            Types.INT,
            "Default number of bed days for end-of-life AIDS care"
        ),
        # ------------------ scale-up parameters for scenario analysis ------------------ #
        "type_of_scaleup": Parameter(
            Types.STRING, "argument to determine type scale-up of program which will be implemented, "
                          "can be 'none', 'target' or 'max'",
        ),
        "scaleup_start_year": Parameter(
            Types.INT,
            "the year when the scale-up starts (it will occur on 1st January of that year)"
        ),
        "scaleup_parameters": Parameter(
            Types.DATA_FRAME,
            "the parameters and values changed in scenario analysis"
        ),
        # ------------------ program-related parameters ------------------ #
        "interval_for_viral_load_measurement_months": Parameter(
            Types.REAL,
            " the interval for viral load monitoring in months"
        ),
        "annual_rate_selftest": Parameter(
            Types.REAL,
            "the annual rate of having an HIV self-test for those not on ART"
        ),
        "selftest_start_year": Parameter(
            Types.REAL,
            "the year in which HIV self-tests become available"
        ),
        "selftest_available": Parameter(
            Types.BOOL,
            "whether self-tests for HIV are available from the scale-up start date"
        ),
        "young_adult_vls_factor": Parameter(
            Types.REAL,
            "adjustment factor for viral load suppression probability for young adults compared to older adults"
        ),
        "proportion_young_adult_on_art": Parameter(
            Types.REAL,
            "proportion of adults on ART who are aged 15-29"
        ),
        "prob_receive_viral_load_test_result": Parameter(
            Types.REAL,
            "the probability of receiving viral load test result"
        ),
        "sensitivity_viral_load_test": Parameter(
            Types.REAL,
            "sensitivity of a viral load test"
        ),
        "prob_of_viral_suppression_following_VL_test": Parameter(
            Types.REAL,
            "the probability of viral suppression following a viral load test"
        ),
        "viral_load_testing_start_year": Parameter(
            Types.INT,
            "the year when the viral load testing starts (it will occur on 1st January of that year"
        ),
        "delay_from_viral_load_to_result_months_min": Parameter(
            Types.REAL,
            "the minimum delay in months from viral load test to receiving result, months"
        ),
        "delay_from_viral_load_to_result_months_max": Parameter(
            Types.REAL,
            "the maximum delay in months from viral load test to receiving result, months"
        ),
        "tdf_test_replace_vl_test": Parameter(
            Types.BOOL,
            "whether urine TDF test should replace viral load monitoring during follow-up appts"
        ),
        "targeted_adherence_monitoring": Parameter(
            Types.BOOL,
            "whether viral load monitoring or TDF testing should be targeted to specific groups"
        ),
        "p_tdf_positive_given_suppressed": Parameter(
            Types.REAL,
            "probability of a urine TDF test returning positive if person virally suppressed"
        ),
        "p_tdf_positive_given_not_suppressed": Parameter(
            Types.REAL,
            "probability of a urine TDF test returning positive if person not virally suppressed"
        ),
        "switch_vl_test_to_tdf": Parameter(
            Types.BOOL,
            "whether TDF urine test is being used in place of VL testing"
        ),
        "injectable_prep_allowed": Parameter(
            Types.BOOL,
            "whether injectable prep is allowed"
        ),
        "prob_injectable_prep_vs_oral": Parameter(
            Types.REAL,
            "probability of injectable prep vs oral prep, given if injectable_prep_allowed=True"
        ),
        "linked_to_care_after_selftest": Parameter(
            Types.REAL,
            "probability a person who has had a self-test will be linked to care"
        ),
        "selftest_sensitivity": Parameter(
            Types.REAL,
            "sensitivity of an HIV self-test"
        ),
        "delay_from_selftest_to_facility_weibull_shape": Parameter(
            Types.REAL,
            "Weibull distribution shape parameter for number of days between a self-test result and "
            "confirmatory test"
        ),
        "delay_from_selftest_to_facility_weibull_scale": Parameter(
            Types.REAL,
            "Weibull distribution scale parameter for number of days between a self-test result and "
            "confirmatory test"
        ),
        "proportion_on_treatment_not_lost_to_follow_up": Parameter(
            Types.REAL,
            "proportion of people who are lost to follow up due to not attending follow-up appointments"
        ),
        # ------------------ Module design parameters ------------------ #
        "hiv_regular_polling_event_frequency_months": Parameter(
            Types.INT,
            "Frequency in months for HivRegularPollingEvent"
        ),
        "regular_polling_event_initialisation_delay_days": Parameter(
            Types.INT,
            "Delay in days for initial HivRegularPollingEvent"
        ),
        "hiv_check_properties_event_delay_months": Parameter(
            Types.INT,
            "Delay in months for HivCheckPropertiesEvent, which checks configuration of all properties"
        ),
        # ------------------ Time periods and data availability ------------------ #
        "baseline_year": Parameter(
            Types.INT,
            "Baseline year at which simulation starts"
        ),
        "end_year_circumcision_data": Parameter(
            Types.INT,
            "End year for circumcision data availability"
        ),
        "end_year_treatment_data": Parameter(
            Types.INT,
            "End year for treatment data availability"
        ),
        "end_year_testing_data": Parameter(
            Types.INT,
            "End year for HIV testing data availability"
        ),
        "years_since_infection_threshold_for_higher_art_probability": Parameter(
            Types.INT,
            "Minimum years since HIV infection required to assign higher ART probability at baseline"
        ),
        "years_offset_for_dummy_test_date_at_baseline": Parameter(
            Types.INT,
            "Years prior to simulation start date to assign test date at baseline; "
            "Required for logical consistency so that all persons on ART have been tested and diagnosed, "
        ),
        # ------------------ Age categorization parameters ------------------ #
        "age_threshold_child_adult_distinction": Parameter(
            Types.INT,
            "Age threshold for child/adult distinction"
        ),
        "age_max_adult": Parameter(
            Types.INT,
            "Maximum age for adult categorization"
        ),
        "age_max_infant": Parameter(
            Types.INT,
            "Maximum age for infant categorization"
        ),
        "age_default_adult": Parameter(
            Types.INT,
            "Default age for adult viral suppression calculations"
        ),
        "age_default_children": Parameter(
            Types.INT,
            "Default age for children viral suppression calculations"
        ),
        "age_min_agyw": Parameter(
            Types.INT,
            "Minimum age for adolescent girls and young women (AGYW) category"
        ),
        "age_max_agyw": Parameter(
            Types.INT,
            "Maximum age for adolescent girls and young women (AGYW) category"
        ),
        "age_max_prep": Parameter(
            Types.INT,
            "Maximum age for PrEP eligibility"
        ),
        "age_max_hiv_early_infant_test_years": Parameter(
            Types.REAL,
            "Maximum age in years for early infant HIV test eligibility"
        ),
        "age_max_prophylaxis_years": Parameter(
            Types.REAL,
            "Maximum age in years for infant HIV prophylaxis"
        ),
        # ------------------ Disease progression ------------------ #
        "hiv_infection_to_aids_untreated_threshold_years": Parameter(
            Types.INT,
            "Number of years after HIV infection at which a person is assumed to have AIDS "
            "if not virally suppressed on ART"
        ),
        "prop_aids_death_is_tb_coinfection": Parameter(
            Types.REAL,
            "Proportion of AIDS deaths that have TB co-infection"
        ),
        # ------------------ Testing parameters ------------------ #
        "hiv_early_infant_test_sensitivity": Parameter(
            Types.REAL,
            "Sensitivity of early infant HIV test"
        ),
        "hiv_early_infant_test_specificity": Parameter(
            Types.REAL,
            "Specificity of early infant HIV test"
        ),
        "hv_min_test_interval_days": Parameter(
            Types.INT,
            "Minimum interval in days between HIV tests"
        ),
        "infant_hiv_test_first_weeks": Parameter(
            Types.INT,
            "Timing of first infant HIV test in weeks after birth in weeks"
        ),
        "infant_hiv_test_second_months": Parameter(
            Types.INT,
            "Timing of second infant HIV test in months after birth in months"
        ),
        "infant_hiv_test_third_months": Parameter(
            Types.INT,
            "Timing of third infant HIV test in months after birth in months"
        ),
        "hiv_testing_initial_window_for_individuals_with_aids_at_init_days": Parameter(
            Types.INT,
            "Initial window in days for HIV testing via HSI event HSI_Test_and_Refer "
            "for individuals with AIDS at initialization"
        ),
        "hiv_testing_close_window_for_individuals_with_aids_at_init_days": Parameter(
            Types.INT,
            "Close window in days for HIV testing via HSI event HSI_Test_and_Refer"
            "for individuals with AIDS at initialization"
        ),
        # ------------------ Health-seeking behavior ------------------ #
        "odds_ratio_health_seeking_aids_symptoms_adults": Parameter(
            Types.REAL,
            "Odds ratio for health seeking behavior in adults with AIDS symptoms"
        ),
        "odds_ratio_health_seeking_aids_symptoms_children": Parameter(
            Types.REAL,
            "Odds ratio for health seeking behavior in children with AIDS symptoms"
        ),
        # ------------------ PrEP-related parameters ------------------ #
        "hiv_prep_refill_appointment_window_days": Parameter(
            Types.INT,
            "Window in days for PrEP refill appointments, checked if retained on prep every 3 months"
        ),
        "prep_continuation_reevaluation_period_months": Parameter(
            Types.INT,
            "Months between PrEP continuation re-evaluation appointments"
        ),
        "prep_cons_notavailable_retry_days": Parameter(
            Types.INT,
            "Days to retry PrEP appointment if consumables not available"
        ),
        # ------------------ ART appointment timing ------------------ #
        "art_continuation_appointment_window_days": Parameter(
            Types.INT,
            "Window in days for ART continuation appointments"
        ),
        "art_restart_appointment_delay_months": Parameter(
            Types.INT,
            "Delay in months for scheduling ART restart appointment"
        ),
        "art_drug_notavailable_default_retry_months": Parameter(
            Types.INT,
            "Default months to retry ART appointment if drugs not available"
        ),
        "art_level1a_healthseekingbehaviour_cap": Parameter(
            Types.INT,
            "Maximum number of retry attempts at level 1a facility before referral"
        ),
        "art_level1a_appointment_delay_months": Parameter(
            Types.INT,
            "Delay in months for ART appointment retry at level 1a facility"
        ),
        "art_level1b_appointment_delay_days": Parameter(
            Types.INT,
            "Delay in days for ART appointment at level 1b facility after referral"
        ),
        "seek_further_art_appointment_if_appointment_not_available_min_delay_days": Parameter(
            Types.INT,
            "Minimum delay in days to seek further ART appointment if appointment not available"
        ),
        "seek_further_art_appointment_if_appointment_not_available_max_delay_days": Parameter(
            Types.INT,
            "Maximum delay in days to seek further ART appointment if appointment not available"
        ),
        # ------------------ Circumcision-related parameters ------------------ #
        "hiv_circ_procedure_first_followup_days": Parameter(
            Types.INT,
            "Days after circumcision procedure for first follow-up appointment"
        ),
        "hiv_circ_procedure_second_followup_days": Parameter(
            Types.INT,
            "Days after circumcision procedure for second follow-up appointment"
        ),
        "hiv_circ_cons_notavailable_retry_procedure_weeks": Parameter(
            Types.INT,
            "Weeks to retry circumcision procedure if consumables not available"
        ),
        # ------------------ Infant prophylaxis parameters ------------------ #
        "infant_prophylaxis_followup_months": Parameter(
            Types.INT,
            "Months after starting infant prophylaxis for follow-up appointment"
        ),
        "infant_prophylaxis_cons_notavailable_retry_days": Parameter(
            Types.INT,
            "Days to retry infant prophylaxis if consumables not available"
        ),
    }

    def read_parameters(self, resourcefilepath: Optional[Path] = None):
        """
        * 1) Reads the ResourceFiles
        * 2) Declare the Symptoms
        """

        # 1) Read the ResourceFiles

        # Shortcut to parameters dict
        p = self.parameters

        workbook = read_csv_files(resourcefilepath / 'ResourceFile_HIV', files=None)
        self.load_parameters_from_dataframe(workbook["parameters"])

        # Load data on HIV prevalence
        p["hiv_prev"] = workbook["hiv_prevalence"]

        # Load assumed time since infected at baseline (year 2010)
        p["time_inf"] = workbook["time_since_infection_at_baselin"]

        # Load reported hiv testing rates
        p["hiv_testing_rates"] = workbook["MoH_numbers_tests"]

        # Load assumed ART coverage at baseline (year 2010)
        p["art_coverage"] = workbook["art_coverage"]

        # Load probability of art / viral suppression start after positive HIV test
        p["prob_start_art_or_vs"] = workbook["spectrum_treatment_cascade"]
        # adjust the reported viral suppression rates for historical impact of VL testing
        self.adjust_viral_load_suppression_rates()

        # Load ARV dispensation schedule
        p["dispensation_period_months"] = workbook["arv_dispensation_schedule"]

        # Load spectrum estimates of treatment cascade
        p["treatment_cascade"] = workbook["spectrum_treatment_cascade"]

        # load parameters for scale-up projections
        p['scaleup_parameters'] = workbook["scaleup_parameters"]

        # create nested dict for ARV dispensation schedule
        self.setup_art_dispensation_lookup()

        # DALY weights
        # get the DALY weight that this module will use from the weight database (these codes are just random!)
        if "HealthBurden" in self.sim.modules.keys():
            # Symptomatic HIV without anemia
            self.daly_wts["hiv_infection_but_not_aids"] = self.sim.modules[
                "HealthBurden"
            ].get_daly_weight(17)

            # AIDS with antiretroviral treatment without anemia
            self.daly_wts["hiv_infection_on_ART"] = self.sim.modules[
                "HealthBurden"
            ].get_daly_weight(20)

            #  AIDS without anti-retroviral treatment without anemia
            self.daly_wts["aids"] = self.sim.modules["HealthBurden"].get_daly_weight(19)

        # 2) Declare the Symptoms.
        self.sim.modules["SymptomManager"].register_symptom(
            Symptom(
                name="aids_symptoms",
                odds_ratio_health_seeking_in_adults=p["odds_ratio_health_seeking_aids_symptoms_adults"],
                odds_ratio_health_seeking_in_children=p["odds_ratio_health_seeking_aids_symptoms_children"],
            )
        )

    def adjust_viral_load_suppression_rates(self):
        """
        Adjusts the 'virally_suppressed_on_art' column in a DataFrame
        using the VL testing adjustment formula.
        """
        p = self.parameters
        tests_per_year = min(1.0, 12.0 / p["interval_for_viral_load_measurement_months"])
        f_benefit = tests_per_year * p["prob_receive_viral_load_test_result"]

        def adjust(p_base):
            if pd.isna(p_base):
                return np.nan
            p_adj = p_base + f_benefit * (1 - p_base) * p["prob_of_viral_suppression_following_VL_test"]
            return max(0.0, min(1.0, p_adj))

        p["prob_start_art_or_vs"]["adjusted_viral_suppression_on_art"] = (
            p["prob_start_art_or_vs"]["overall_adjusted_viral_suppression_on_art"].apply(adjust)
        )

    def pre_initialise_population(self):
        """Do things required before the population is created
        * Build the LinearModels"""
        self._build_linear_models()

    def _build_linear_models(self):
        """Establish the Linear Models"""

        p = self.parameters

        # ---- LINEAR MODELS -----
        # LinearModel for the relative risk of becoming infected during the simulation
        # N.B. age assumed not to have an effect on incidence
        predictors = [
            Predictor("age_years").when("<15", 0.0).when("<49", 1.0).otherwise(0.0),
            Predictor("sex").when("F", p["rr_sex_f"]),
            Predictor("li_is_circ").when(True, p["rr_circumcision"]),
            Predictor("hv_is_on_prep_oral").
            when(True, 1.0 - p["proportion_reduction_in_risk_of_hiv_aq_if_on_oral_prep"]),
            Predictor("hv_is_on_prep_inj").
            when(True, 1.0 - p["proportion_reduction_in_risk_of_hiv_aq_if_on_inj_prep"]),
            Predictor("li_urban").when(False, p["rr_rural"]),
            Predictor("li_wealth", conditions_are_mutually_exclusive=True)
            .when(2, p["rr_windex_poorer"])
            .when(3, p["rr_windex_middle"])
            .when(4, p["rr_windex_richer"])
            .when(5, p["rr_windex_richest"]),
            Predictor("li_ed_lev", conditions_are_mutually_exclusive=True)
            .when(2, p["rr_edlevel_primary"])
            .when(3, p["rr_edlevel_secondary"]),
            Predictor("hv_behaviour_change").when(True, p["rr_behaviour_change"])
        ]

        conditional_predictors = [
            Predictor().when(
                '(ss_sh_infection_status == "High-infection") &'
                '(sex == "F")',
                p["rr_schisto"]
            ),
        ] if "Schisto" in self.sim.modules else []

        self.lm["rr_of_infection"] = LinearModel.multiplicative(
            *(predictors + conditional_predictors))

        # LinearModels to give the shape and scale for the Weibull distribution describing time from infection to death
        self.lm["scale_parameter_for_infection_to_death"] = LinearModel.multiplicative(
            Predictor(
                "age_years",
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when("==0", p["mean_survival_for_infants_infected_prior_to_birth"])
            .when(
                ".between(1,4)",
                p["infection_to_death_infant_infection_after_birth_weibull_scale"],
            )
            .when(".between(5, 19)", p["infection_to_death_weibull_scale_1519"])
            .when(".between(20, 24)", p["infection_to_death_weibull_scale_2024"])
            .when(".between(25, 29)", p["infection_to_death_weibull_scale_2529"])
            .when(".between(30, 34)", p["infection_to_death_weibull_scale_3034"])
            .when(".between(35, 39)", p["infection_to_death_weibull_scale_3539"])
            .when(".between(40, 44)", p["infection_to_death_weibull_scale_4044"])
            .when(".between(45, 49)", p["infection_to_death_weibull_scale_4549"])
            .when(">= 50", p["infection_to_death_weibull_scale_4549"])
        )

        self.lm["shape_parameter_for_infection_to_death"] = LinearModel.multiplicative(
            Predictor(
                "age_years",
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True)
            .when("==0", 1)  # Weibull with shape=1 equivalent to exponential distribution
            .when(".between(1,4)", p["infection_to_death_infant_infection_after_birth_weibull_shape"])
            .when(".between(5, 19)", p["infection_to_death_weibull_shape_1519"])
            .when(".between(20, 24)", p["infection_to_death_weibull_shape_2024"])
            .when(".between(25, 29)", p["infection_to_death_weibull_shape_2529"])
            .when(".between(30, 34)", p["infection_to_death_weibull_shape_3034"])
            .when(".between(35, 39)", p["infection_to_death_weibull_shape_3539"])
            .when(".between(40, 44)", p["infection_to_death_weibull_shape_4044"])
            .when(".between(45, 49)", p["infection_to_death_weibull_shape_4549"])
            .when(">= 50", p["infection_to_death_weibull_shape_4549"])
        )

        # -- Linear Model to give the mean months between aids and death depending on age
        self.lm["offset_parameter_for_months_from_aids_to_death"] = (
            LinearModel.multiplicative(
                Predictor(
                    "age_years",
                    conditions_are_mutually_exclusive=True,
                    conditions_are_exhaustive=True,
                )
                .when("<5", p["mean_months_between_aids_and_death_infant"])
                .when(">=5", p["mean_months_between_aids_and_death"])
            )
        )

        # -- Linear Models for the Uptake of Services
        # Linear model that give the increase in likelihood of seeking a 'Spontaneous' Test for HIV
        # condition must be not on ART for test
        # allow children to be tested without symptoms
        # previously diagnosed can be re-tested
        self.lm["lm_spontaneous_test_12m"] = LinearModel.multiplicative(
            Predictor("hv_inf").when(True, p["rr_test_hiv_positive"]).otherwise(1.0),
            Predictor("hv_art").when("not", 1.0).otherwise(0.0),
        )

        # Linear model for changing behaviour following an HIV-negative test
        self.lm["lm_behavchg"] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p["prob_behav_chg_after_hiv_test"],
            Predictor("hv_inf").when(False, 1.0).otherwise(0.0),
        )

        # Linear model for starting PrEP (if F/sex-workers), following when the person has tested HIV -ve:
        self.lm["lm_prep"] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p["prob_prep_for_fsw_after_hiv_test"],
            Predictor("hv_inf").when(False, 1.0).otherwise(0.0),
            Predictor("sex").when("F", 1.0).otherwise(0.0),
            Predictor("li_is_sexworker").when(True, 1.0).otherwise(0.0),
        )

        # Linear model for starting PrEP (if AGYW), given through regular poll:
        self.lm["lm_prep_agyw"] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p["prob_prep_for_agyw"],
            Predictor("sex").when("F", 1.0).otherwise(0.0),
            Predictor("hv_inf").when(False, 1.0).otherwise(0.0),
            Predictor("age_years")
            .when("<15", 0.0)
            .when(">25", 0.0)
            .otherwise(1.0),
            Predictor("year",
                      external=True,
                      conditions_are_mutually_exclusive=True,
                      conditions_are_exhaustive=True)
            .when("<2021", 0)
            .otherwise(1.0),
        )

        # Linear model for circumcision (if M) following when the person has been tested:
        self.lm["lm_circ"] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p["prob_circ_after_hiv_test"],
            Predictor("hv_inf").when(False, 1.0).otherwise(0.0),
            Predictor("sex").when("M", 1.0).otherwise(0.0),
            Predictor("year",
                      external=True,
                      conditions_are_mutually_exclusive=True).when("<2019", 1)
            .otherwise(p["increase_in_prob_circ_2019"])
        )

        # Linear model for circumcision for male and aging <15 yrs who spontaneously presents for VMMC
        # This is to increase the VMMC cases/visits for <15 yrs males, which should account for about
        # 40% of total VMMC cases according to UNAIDS & WHO/DHIS2 2015-2019 data.
        self.lm["lm_circ_child"] = LinearModel.multiplicative(
            Predictor("sex").when("M", 1.0).otherwise(0.0),
            Predictor("age_years").when("<15", 1.0).otherwise(0.0),
            Predictor("year",
                      external=True,
                      conditions_are_mutually_exclusive=True,
                      conditions_are_exhaustive=True).when("<2020", p["prob_circ_for_child_before_2020"])
            .otherwise(p["prob_circ_for_child_from_2020"])
        )

    def initialise_population(self, population):
        """Set our property values for the initial population."""

        df = population.props

        # --- Current status
        df.loc[df.is_alive, "hv_inf"] = False
        df.loc[df.is_alive, "hv_art"] = "not"
        df.loc[df.is_alive, "hv_is_on_prep_oral"] = False
        df.loc[df.is_alive, "hv_is_on_prep_inj"] = False
        df.loc[df.is_alive, "hv_behaviour_change"] = False
        df.loc[df.is_alive, "hv_diagnosed"] = False
        df.loc[df.is_alive, "hv_number_tests"] = 0
        df.loc[df.is_alive, "hv_arv_dispensing_interval"] = np.nan

        # --- Dates on which things have happened
        df.loc[df.is_alive, "hv_date_inf"] = pd.NaT
        df.loc[df.is_alive, "hv_last_test_date"] = pd.NaT
        df.loc[df.is_alive, "hv_date_treated"] = pd.NaT
        df.loc[df.is_alive, "hv_date_last_ART"] = pd.NaT

        # --- Counters
        df.loc[df.is_alive, "hv_VMMC_in_last_year"] = False
        df.loc[df.is_alive, "hv_days_on_oral_prep_AGYW"] = 0
        df.loc[df.is_alive, "hv_days_on_oral_prep_FSW"] = 0
        df.loc[df.is_alive, "hv_days_on_inj_prep_AGYW"] = 0
        df.loc[df.is_alive, "hv_days_on_inj_prep_FSW"] = 0

        # Launch sub-routines for allocating the right number of people into each category
        self.initialise_baseline_prevalence(population)  # allocate baseline prevalence

        self.initialise_baseline_art(population)  # allocate baseline art coverage
        self.initialise_baseline_tested(population)  # allocate baseline testing coverage

    def initialise_baseline_prevalence(self, population):
        """
        Assign baseline HIV prevalence, according to age, sex and key other variables (established in analysis of DHS
        data).
        """

        params = self.parameters
        df = population.props

        # prob of infection based on age and sex in baseline year (baseline_year):
        prevalence_db = params["hiv_prev"]
        prev_baseline = prevalence_db.loc[
            prevalence_db.year == params['baseline_year'], ["age_from", "sex", "prop_hiv_mw2021_v14"]
        ]

        prev_baseline = prev_baseline.rename(columns={"age_from": "age_years"})
        prob_of_infec = df.loc[df.is_alive, ["age_years", "sex"]].merge(
            prev_baseline, on=["age_years", "sex"], how="left"
        )["prop_hiv_mw2021_v14"]

        # probability of being hiv-positive based on risk factors
        rel_prob_by_risk_factor = LinearModel.multiplicative(
            Predictor("li_is_sexworker").when(True, params["rr_fsw"]),
            Predictor("li_is_circ").when(True, params["rr_circumcision"]),
            Predictor("li_urban").when(False, params["rr_rural"]),
            Predictor("li_wealth", conditions_are_mutually_exclusive=True)
            .when(2, params["rr_windex_poorer"])
            .when(3, params["rr_windex_middle"])
            .when(4, params["rr_windex_richer"])
            .when(5, params["rr_windex_richest"]),
            Predictor("li_ed_lev", conditions_are_mutually_exclusive=True)
            .when(2, params["rr_edlevel_primary"])
            .when(3, params["rr_edlevel_secondary"]),
        ).predict(df.loc[df.is_alive])

        # Rescale relative probability of infection so that its average is 1.0 within each age/sex group
        p = pd.DataFrame(
            {
                "age_years": df["age_years"],
                "sex": df["sex"],
                "prob_of_infec": prob_of_infec,
                "rel_prob_by_risk_factor": rel_prob_by_risk_factor,
            }
        )

        p["mean_of_rel_prob_within_age_sex_group"] = p.groupby(["age_years", "sex"])[
            "rel_prob_by_risk_factor"
        ].transform("mean")
        p["scaled_rel_prob_by_risk_factor"] = (
            p["rel_prob_by_risk_factor"] / p["mean_of_rel_prob_within_age_sex_group"]
        )
        # add scaling factor 1.1 to match overall unaids prevalence
        # different assumptions on pop size result in slightly different overall prevalence so use adjustment factor
        p["overall_prob_of_infec"] = (
            p["scaled_rel_prob_by_risk_factor"] * p["prob_of_infec"] * params["unaids_prevalence_adjustment_factor"]
        )
        # this needs to be series of True/False
        infec = (
                    self.rng.random_sample(len(p["overall_prob_of_infec"]))
                    < p["overall_prob_of_infec"]) & df.is_alive

        # Assign the designated person as infected in the population.props dataframe:
        df.loc[infec, "hv_inf"] = True

        # Assign date that persons were infected by drawing from assumed distribution (for adults)
        # Clipped to prevent dates of infection before the person was born.
        time_inf = params["time_inf"]
        years_ago_inf = self.rng.choice(
            time_inf["year"],
            size=len(infec),
            replace=True,
            p=time_inf["scaled_prob"],
        )

        hv_date_inf = pd.Series(
            self.sim.date - pd.to_timedelta(years_ago_inf * DAYS_IN_YEAR, unit="d")
        )
        df.loc[infec, "hv_date_inf"] = hv_date_inf.clip(lower=df.date_of_birth)

    def initialise_baseline_art(self, population):
        """assign initial art coverage levels
        also assign hiv test properties if allocated ART
        probability of being on ART scaled by length of time infected (>10 years)
        """
        df = population.props
        params = self.parameters

        # 1) Determine who is currently on ART
        # this is updated for revised UNAIDS estimates
        worksheet = self.parameters["art_coverage"]

        art_data = worksheet.loc[
            worksheet.year == (params['baseline_year']-1), ["year", "single_age", "sex", "prop_coverage_reduced"]
        ]

        # merge all susceptible individuals with their coverage probability based on sex and age
        prob_art = df.loc[df.is_alive, ["age_years", "sex"]].merge(
            art_data,
            left_on=["age_years", "sex"],
            right_on=["single_age", "sex"],
            how="left",
        )["prop_coverage_reduced"]

        # make a series with relative risks of art which depends on >10 years infected (5x higher)
        rr_art = pd.Series(1, index=df.index)
        rr_art.loc[
            df.is_alive & (
                df.hv_date_inf < (
                    self.sim.date - pd.DateOffset(
                        years=params["years_since_infection_threshold_for_higher_art_probability"]
                    )
                )
            )
        ] = params["rel_probability_art_baseline_aids"]

        # Rescale relative probability of infection so that its average is 1.0 within each age/sex group
        p = pd.DataFrame(
            {
                "age_years": df["age_years"],
                "sex": df["sex"],
                "prob_art": prob_art,
                "rel_prob_art_by_time_infected": rr_art,
            }
        )

        p["mean_of_rel_prob_within_age_sex_group"] = p.groupby(["age_years", "sex"])[
            "rel_prob_art_by_time_infected"
        ].transform("mean")
        p["scaled_rel_prob_by_time_infected"] = (
            p["rel_prob_art_by_time_infected"]
            / p["mean_of_rel_prob_within_age_sex_group"]
        )
        p["overall_prob_of_art"] = p["scaled_rel_prob_by_time_infected"] * p["prob_art"]
        random_draw = self.rng.random_sample(size=len(df))

        art_idx = df.index[
            (random_draw < p["overall_prob_of_art"]) & df.is_alive & df.hv_inf
            ]

        # 2) Determine adherence levels for those currently on ART, for each of adult men, adult women and children
        adult_f_art_idx = df.loc[
            (df.index.isin(art_idx) & (df.sex == "F") &
             (df.age_years >= params["age_threshold_child_adult_distinction"]))
        ].index
        adult_m_art_idx = df.loc[
            (df.index.isin(art_idx) & (df.sex == "M") &
             (df.age_years >= params["age_threshold_child_adult_distinction"]))
        ].index
        child_art_idx = df.loc[
            (df.index.isin(art_idx) &
             (df.age_years < params["age_threshold_child_adult_distinction"]))
        ].index

        suppr = list()  # list of all indices for persons on ART and suppressed
        notsuppr = list()  # list of all indices for persons on ART and not suppressed

        def split_into_vl_and_notvl(all_idx, prob):
            vl_suppr = self.rng.random_sample(len(all_idx)) < prob
            suppr.extend(all_idx[vl_suppr])
            notsuppr.extend(all_idx[~vl_suppr])

        # get expected viral suppression rates by age and year
        prob_vs_adult = self.prob_viral_suppression(
            df=df, year=self.sim.date.year, age_of_person=params["age_default_adult"],
        )
        prob_vs_child = self.prob_viral_suppression(
            df=df, year=self.sim.date.year, age_of_person=params["age_default_children"],
        )

        split_into_vl_and_notvl(adult_f_art_idx, prob_vs_adult)
        split_into_vl_and_notvl(adult_m_art_idx, prob_vs_adult)
        split_into_vl_and_notvl(child_art_idx, prob_vs_child)

        # Set ART status:
        df.loc[suppr, "hv_art"] = "on_VL_suppressed"
        df.loc[notsuppr, "hv_art"] = "on_not_VL_suppressed"

        # check that everyone on ART is labelled as such
        assert not (df.loc[art_idx, "hv_art"] == "not").any()

        # for logical consistency, ensure that all persons on ART have been tested and diagnosed
        # set last test date prior to baseline_year so not counted as a current new test
        # don't record individual property hv_number_tests for logging (occurred prior to baseline_year)
        df.loc[art_idx, "hv_last_test_date"] = self.sim.date - pd.DateOffset(
            years=params["years_offset_for_dummy_test_date_at_baseline"]
        )
        df.loc[art_idx, "hv_diagnosed"] = True

        # all those on ART need to have event scheduled for continuation/cessation of treatment
        # this window is 1-90 days (3-monthly prescribing)
        for person in art_idx:
            days = self.rng.randint(low=params['dispensation_min_days'],
                                    high=params['initial_dispensation_period_months'] * 30.5, dtype=np.int64)

            date_treated = (params['initial_dispensation_period_months'] * 30.5) - days
            df.at[person, "hv_date_treated"] = self.sim.date - pd.to_timedelta(date_treated, unit="days")
            df.at[person, "hv_date_last_ART"] = self.sim.date - pd.to_timedelta(date_treated, unit="days")
            df.at[person, "hv_arv_dispensing_interval"] = self.parameters['initial_dispensation_period_months']

            self.sim.schedule_event(
                Hiv_DecisionToContinueTreatment(person_id=person, module=self),
                self.sim.date + pd.to_timedelta(days, unit="days"),
            )

    def initialise_baseline_tested(self, population):
        """assign initial hiv testing levels, only for adults
        all who have been allocated ART will already have property hv_diagnosed=True
        use the spectrum proportion PLHIV who know status to assign remaining tests
        """
        df = population.props
        p = self.parameters

        # get proportion plhiv who know staus from spectum estimates
        worksheet = p["treatment_cascade"]
        testing_data = worksheet.loc[
            worksheet.year == p['baseline_year'], ["year", "age", "know_status"]
        ]
        adult_know_status = testing_data.loc[(testing_data.age == "adults"), "know_status"].values[0] / 100
        children_know_status = testing_data.loc[(testing_data.age == "children"), "know_status"].values[0] / 100

        # ADULTS
        # find proportion of adult PLHIV diagnosed (currently on ART)
        adults_diagnosed = len(df[df.is_alive
                                  & df.hv_diagnosed
                                  & (df.age_years >= p["age_threshold_child_adult_distinction"])])

        adults_infected = len(df[df.is_alive
                                 & df.hv_inf
                                 & (df.age_years >= p["age_threshold_child_adult_distinction"])])

        prop_currently_diagnosed = adults_diagnosed / adults_infected if adults_infected > 0 else 0
        hiv_test_deficit = adult_know_status - prop_currently_diagnosed
        number_deficit = int(hiv_test_deficit * adults_infected)

        adult_test_index = []
        if hiv_test_deficit > 0:
            # sample number_deficit from remaining undiagnosed pop
            adult_undiagnosed = df.loc[df.is_alive
                                       & df.hv_inf
                                       & ~df.hv_diagnosed
                                       & (df.age_years >= p["age_threshold_child_adult_distinction"])].index

            adult_test_index = self.rng.choice(adult_undiagnosed, size=number_deficit, replace=False)

        # CHILDREN
        # find proportion of adult PLHIV diagnosed (currently on ART)
        children_diagnosed = len(df[df.is_alive
                                    & df.hv_diagnosed
                                    & (df.age_years < p["age_threshold_child_adult_distinction"])])

        children_infected = len(df[df.is_alive
                                   & df.hv_inf
                                   & (df.age_years < p["age_threshold_child_adult_distinction"])])

        prop_currently_diagnosed = children_diagnosed / children_infected if children_infected > 0 else 0
        hiv_test_deficit = children_know_status - prop_currently_diagnosed
        number_deficit = int(hiv_test_deficit * children_infected)

        child_test_index = []
        if hiv_test_deficit > 0:
            child_undiagnosed = df.loc[df.is_alive
                                       & df.hv_inf
                                       & ~df.hv_diagnosed
                                       & (df.age_years < p["age_threshold_child_adult_distinction"])].index

            child_test_index = self.rng.choice(child_undiagnosed, size=number_deficit, replace=False)

        # join indices
        test_index = list(adult_test_index) + list(child_test_index)

        df.loc[df.index.isin(test_index), "hv_diagnosed"] = True
        # dummy date for date last hiv test (before sim start), otherwise see big spike in testing 01-01-2010
        df.loc[test_index, "hv_last_test_date"] = self.sim.date - pd.DateOffset(
            years=p["years_offset_for_dummy_test_date_at_baseline"]
        )

    def initialise_simulation(self, sim):
        """
        * 1) Schedule the Main HIV Regular Polling Event
        * 2) Schedule the Logging Event
        * 3) Determine who has AIDS and impose the Symptoms 'aids_symptoms'
        * 4) Schedule the AIDS onset events and AIDS death event for those infected already
        * 5) (Optionally) Schedule the event to check the configuration of all properties
        * 6) Define the DxTests
        * 7) Look-up and save the codes for consumables
        """
        df = sim.population.props
        p = self.parameters

        # 1) Schedule the Main HIV Regular Polling Event
        sim.schedule_event(
            HivRegularPollingEvent(self), sim.date +
            DateOffset(days=p['regular_polling_event_initialisation_delay_days'])
        )

        # 2) Schedule the Logging Event
        sim.schedule_event(HivLoggingEvent(self), sim.date + DateOffset(years=1))

        # Optional: Schedule the scale-up of programs
        if self.parameters["type_of_scaleup"] != 'none':
            scaleup_start_date = Date(self.parameters["scaleup_start_year"], 1, 1)
            assert scaleup_start_date >= self.sim.start_date, f"Date {scaleup_start_date} is before simulation starts."
            sim.schedule_event(HivScaleUpEvent(self), scaleup_start_date)

        # 3) Determine who has AIDS and impose the Symptoms 'aids_symptoms'

        # Those on ART currently (will not get any further events scheduled):
        on_art_idx = df.loc[df.is_alive & df.hv_inf & (df.hv_art == "on_VL_suppressed")].index

        # Those that lived more than threshold years and not currently on ART are assumed to currently have AIDS
        #  (will have AIDS Death event scheduled)
        has_aids_idx = df.loc[
            df.is_alive
            & df.hv_inf
            & ((self.sim.date - df.hv_date_inf).dt.days > p["hiv_infection_to_aids_untreated_threshold_years"] * 365)
            & (df.hv_art != "on_VL_suppressed")
            ].index

        # Those that are in neither category are "before AIDS" (will have AIDS Onset Event scheduled)
        before_aids_idx = df.loc[df.is_alive & df.hv_inf].index.difference(has_aids_idx).difference(on_art_idx)

        # Impose the symptom to those that have AIDS (the symptom is the definition of having AIDS)
        self.sim.modules["SymptomManager"].change_symptom(
            person_id=has_aids_idx.tolist(),
            symptom_string="aids_symptoms",
            add_or_remove="+",
            disease_module=self,
        )

        # 4) Schedule the AIDS onset events and AIDS death event for those infected already
        # AIDS Onset Event for those who are infected but not yet AIDS and have not ever started ART
        # NB. This means that those on ART at the start of the simulation may not have an AIDS event --
        # like it happened at some point in the past
        scale, shape, offset = self.get_time_from_infection_to_aids_distribution_parameters(before_aids_idx)
        days_infection_to_aids = self.sample_time_from_infection_to_aids_given_parameters(scale, shape, offset)
        days_since_infection = (self.sim.date - df.loc[before_aids_idx, "hv_date_inf"])
        # If any days_since_infection >= days_infection_to_aids are negative resample
        # these values until all are positive
        days_until_aids_is_negative = days_since_infection >= days_infection_to_aids
        while np.any(days_until_aids_is_negative):
            days_infection_to_aids[days_until_aids_is_negative] = (
                self.sample_time_from_infection_to_aids_given_parameters(
                    scale[days_until_aids_is_negative],
                    shape[days_until_aids_is_negative],
                    offset[days_until_aids_is_negative],
                )
            )
            days_until_aids_is_negative = days_since_infection >= days_infection_to_aids
        days_until_aids = days_infection_to_aids - days_since_infection
        date_onset_aids = self.sim.date + pd.to_timedelta(days_until_aids, unit='D')
        for person_id, date in zip(before_aids_idx, date_onset_aids):
            sim.schedule_event(
                HivAidsOnsetEvent(person_id=person_id, module=self, cause='AIDS_non_TB'),
                date=date,
            )

        # Schedule the AIDS death events for those who have got AIDS already
        for person_id in has_aids_idx:
            # schedule a HSI_Test_and_Refer otherwise initial AIDS rates and deaths are far too high
            date_test = self.sim.date + pd.DateOffset(
                days=self.rng.randint(0, p["hiv_testing_initial_window_for_individuals_with_aids_at_init_days"])
            )
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                hsi_event=HSI_Hiv_TestAndRefer(
                    person_id=person_id, module=self, referred_from="initialise_simulation"),
                priority=1,
                topen=date_test,
                tclose=self.sim.date + pd.DateOffset(
                    days=p["hiv_testing_close_window_for_individuals_with_aids_at_init_days"]
                ),
            )

            date_aids_death = self.sim.date + pd.DateOffset(
                months=self.rng.randint(
                    low=p["min_months_between_aids_and_death"],
                    high=p["mean_months_between_aids_and_death"],
                )
            )

            # proportion of AIDS deaths that have TB co-infection
            cause_of_death = self.rng.choice(
                a=["AIDS_non_TB", "AIDS_TB"],
                size=1,
                p=[1 - p['prop_aids_death_is_tb_coinfection'], p['prop_aids_death_is_tb_coinfection']]
            )

            sim.schedule_event(
                HivAidsDeathEvent(person_id=person_id, module=self, cause=cause_of_death),
                date=date_aids_death,
            )

            # schedule hospital stay for end of life care if untreated
            beddays = self.rng.randint(
                low=p['length_of_inpatient_stay_if_terminal'][0],
                high=p['length_of_inpatient_stay_if_terminal'][1])
            date_admission = date_aids_death - pd.DateOffset(days=beddays)
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                hsi_event=HSI_Hiv_EndOfLifeCare(
                    person_id=person_id, module=self, beddays=beddays
                ),
                priority=0,
                topen=(
                    date_admission
                    if (date_admission >= self.sim.date)
                    else self.sim.date
                ),
            )

        # 5) (Optionally) Schedule the event to check the configuration of all properties
        if self.run_with_checks:
            sim.schedule_event(
                HivCheckPropertiesEvent(self),
                sim.date + pd.DateOffset(months=p["hiv_check_properties_event_delay_months"])
            )

        # 6) Store codes for the consumables needed
        hs = self.sim.modules["HealthSystem"]

        # updated consumables listing
        # blood tube and gloves are optional items
        self.item_codes_for_consumables_required['hiv_rapid_test'] = \
            hs.get_item_code_from_item_name("Test, HIV EIA Elisa")

        self.item_codes_for_consumables_required['hiv_early_infant_test'] = \
            hs.get_item_code_from_item_name("Test, HIV EIA Elisa")

        self.item_codes_for_consumables_required['blood_tube'] = \
            hs.get_item_code_from_item_name("Blood collecting tube, 5 ml")

        self.item_codes_for_consumables_required['gloves'] = \
            hs.get_item_code_from_item_name("Disposables gloves, powder free, 100 pieces per box")

        self.item_codes_for_consumables_required['vl_measurement'] = \
            hs.get_item_codes_from_package_name("Viral Load")

        self.item_codes_for_consumables_required['circ'] = \
            hs.get_item_codes_from_package_name("Male circumcision ")

        # adult prep: 1 tablet daily
        self.item_codes_for_consumables_required['prep'] = \
            hs.get_item_code_from_item_name("Tenofovir (TDF)/Emtricitabine (FTC), tablet, 300/200 mg")

        # infant NVP 1.5mg daily for birth weight 2500g or above, for 6 weeks
        self.item_codes_for_consumables_required['infant_prep'] = \
            hs.get_item_code_from_item_name("Nevirapine, oral solution, 10 mg/ml")

        # First - line ART for adults(age > "ART_age_cutoff_older_child")
        # TDF/3TC/DTG 120/60/50mg, 1 tablet per day
        # cotrim adult tablet, 1 tablet per day, units specified in mg * dispensation days
        self.item_codes_for_consumables_required['First-line ART regimen: adult'] = \
            hs.get_item_code_from_item_name("First-line ART regimen: adult")
        self.item_codes_for_consumables_required['First-line ART regimen: adult: cotrimoxazole'] = \
            hs.get_item_code_from_item_name("Cotrimoxizole, 960mg pppy")

        # ART for older children aged ("ART_age_cutoff_younger_child" < age <= "ART_age_cutoff_older_child"):
        # ABC/3TC/DTG 120/60/50mg, 3 tablets per day
        # cotrim paediatric tablet, 4 tablets per day, units specified in mg * dispensation days
        self.item_codes_for_consumables_required['First line ART regimen: older child'] = \
            hs.get_item_code_from_item_name("First line ART regimen: older child")
        self.item_codes_for_consumables_required['First line ART regimen: older child: cotrimoxazole'] = \
            hs.get_item_code_from_item_name("Sulfamethoxazole + trimethropin, oral suspension, 240 mg, 100 ml")

        # ART for younger children aged (age < "ART_age_cutoff_younger_child"):
        # ABC/3TC/DTG 120/60/10mg, 2 tablets per day
        # cotrim paediatric tablet, 2 tablets per day, units specified in mg * dispensation days
        self.item_codes_for_consumables_required['First line ART regimen: young child'] = \
            hs.get_item_code_from_item_name("First line ART regimen: young child")
        self.item_codes_for_consumables_required['First line ART regimen: young child: cotrimoxazole'] = \
            hs.get_item_code_from_item_name("Sulfamethoxazole + trimethropin, oral suspension, 240 mg, 100 ml")

        # 7) Define the DxTests
        # HIV Rapid Diagnostic Test:
        # NB. The rapid test is assumed to be 100% specific and sensitive. This is used to guarantee that all persons
        #  that start ART are truly HIV-pos.
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            hiv_rapid_test=DxTest(
                property='hv_inf',
                item_codes=self.item_codes_for_consumables_required['hiv_rapid_test'],
                optional_item_codes=[
                    self.item_codes_for_consumables_required['blood_tube'],
                    self.item_codes_for_consumables_required['gloves']]
            )
        )

        # Test for Early Infect Diagnosis
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            hiv_early_infant_test=DxTest(
                property='hv_inf',
                sensitivity=p["hiv_early_infant_test_sensitivity"],
                specificity=p["hiv_early_infant_test_specificity"],
                item_codes=self.item_codes_for_consumables_required['hiv_early_infant_test'],
                optional_item_codes=[
                    self.item_codes_for_consumables_required['blood_tube'],
                    self.item_codes_for_consumables_required['gloves']]
            )
        )

        # 8) define the start date for viral load testing
        self.vl_testing_available_by_year = {
            year: year >= p["viral_load_testing_start_year"] for year in range(2010, sim.end_date.year + 1)
        }

    def update_parameters_for_program_change(self):
        """
        options for program scale-up are 'target' or 'max'
        other options switch on / off various interventions
        """

        p = self.parameters
        scaled_params_workbook = p["scaleup_parameters"]

        if p['type_of_scaleup'] == 'target':
            scaled_params = scaled_params_workbook.set_index('parameter_name')['value'].to_dict()
        else:
            scaled_params = scaled_params_workbook.set_index('parameter_name')['prior_max'].to_dict()

        # scale-up HIV program
        # reduce risk of HIV - applies to whole adult population
        p["beta"] = p["beta"] * scaled_params["reduction_in_hiv_beta"]

        # increase PrEP coverage for FSW after HIV test
        p["prob_prep_for_fsw_after_hiv_test"] = scaled_params["prob_prep_for_fsw_after_hiv_test"]

        # prep poll for AGYW - target to the highest risk
        # increase retention to 75% for FSW and AGYW
        p["prob_prep_for_agyw"] = scaled_params["prob_prep_for_agyw"]
        p["probability_of_being_retained_on_prep_every_3_months"] = scaled_params[
            "probability_of_being_retained_on_prep_every_3_months"
        ]

        # perfect retention on ART
        p["probability_of_being_retained_on_art_every_3_months"] = scaled_params[
            "probability_of_being_retained_on_art_every_3_months"
        ]

        # increase probability of VMMC after hiv test
        p["prob_circ_after_hiv_test"] = scaled_params["prob_circ_after_hiv_test"]

        # increase testing/diagnosis rates, default 2020 0.03/0.25 -> 93% dx
        p["hiv_testing_rates"]["annual_testing_rate_adults"] = scaled_params["annual_testing_rate_adults"]

        # ANC testing - value for mothers and infants testing
        p["prob_hiv_test_at_anc_or_delivery"] = scaled_params["prob_hiv_test_at_anc_or_delivery"]
        p["prob_hiv_test_for_newborn_infant"] = scaled_params["prob_hiv_test_for_newborn_infant"]

        # viral suppression rates
        # adults adjusted rates at 89% in 2025 -> 95% when VL monitoring incorporated
        # change all column values
        p["prob_start_art_or_vs"]["virally_suppressed_on_art"] = scaled_params["virally_suppressed_on_art"]
        self.adjust_viral_load_suppression_rates()  # then adjust these rates for VL testing effects

        # update exising linear models to use new scaled-up parameters
        self._build_linear_models()

    def on_birth(self, mother_id, child_id):
        """
        * Initialise our properties for a newborn individual;
        * schedule testing;
        * schedule infection during breastfeeding
        """
        p = self.parameters
        df = self.sim.population.props

        # Default Settings:
        # --- Current status
        df.at[child_id, "hv_inf"] = False
        df.at[child_id, "hv_art"] = "not"
        df.at[child_id, "hv_on_cotrimoxazole"] = False
        df.at[child_id, "hv_is_on_prep_oral"] = False
        df.at[child_id, "hv_is_on_prep_inj"] = False
        df.at[child_id, "hv_behaviour_change"] = False
        df.at[child_id, "hv_diagnosed"] = False
        df.at[child_id, "hv_number_tests"] = 0
        df.at[child_id, "hv_arv_dispensing_interval"] = np.nan

        # --- Dates on which things have happened
        df.at[child_id, "hv_date_inf"] = pd.NaT
        df.at[child_id, "hv_last_test_date"] = pd.NaT
        df.at[child_id, "hv_date_treated"] = pd.NaT
        df.at[child_id, "hv_date_last_ART"] = pd.NaT

        # --- Counters
        df.at[child_id, "hv_VMMC_in_last_year"] = False
        df.at[child_id, "hv_days_on_oral_prep_AGYW"] = 0
        df.at[child_id, "hv_days_on_oral_prep_FSW"] = 0
        df.at[child_id, "hv_days_on_inj_prep_AGYW"] = 0
        df.at[child_id, "hv_days_on_inj_prep_FSW"] = 0

        # ----------------------------------- MTCT - AT OR PRIOR TO BIRTH --------------------------
        #  DETERMINE IF THE CHILD IS INFECTED WITH HIV FROM THEIR MOTHER DURING PREGNANCY / DELIVERY
        mother = df.loc[abs(mother_id)]  # Not interested whether true or direct birth

        mother_infected_prior_to_pregnancy = mother.hv_inf & (
            mother.hv_date_inf <= mother.date_of_last_pregnancy
        )
        mother_infected_during_pregnancy = mother.hv_inf & (
            mother.hv_date_inf > mother.date_of_last_pregnancy
        )

        if mother_infected_prior_to_pregnancy:
            if mother.hv_art == "on_VL_suppressed":
                #  mother has existing infection, mother ON ART and VL suppressed at time of delivery
                child_infected = self.rng.random_sample() < p["prob_mtct_treated"]
            else:
                # mother was infected prior to pregnancy but is not on VL suppressed at time of delivery
                child_infected = (
                    self.rng.random_sample() < p["prob_mtct_untreated"]
                )

        elif mother_infected_during_pregnancy:
            #  mother has incident infection during pregnancy, NO ART
            child_infected = (
                self.rng.random_sample() < p["prob_mtct_incident_preg"]
            )

        else:
            # mother is not infected
            child_infected = False

        if child_infected:
            self.do_new_infection(child_id)

        # ----------------------------------- MTCT - DURING BREASTFEEDING --------------------------
        # If child is not infected and is being breastfed, then expose them to risk of MTCT through breastfeeding
        if (
            (not child_infected)
            and (df.at[child_id, "nb_breastfeeding_status"] != "none")
            and mother.hv_inf
        ):
            # Pass mother's id, whether from true or direct birth
            self.mtct_during_breastfeeding(abs(mother_id), child_id)

        # ----------------------------------- HIV testing --------------------------
        if "CareOfWomenDuringPregnancy" not in self.sim.modules:
            # If mother's HIV status not known, schedule test at delivery
            # (usually performed by care_of_women_during_pregnancy module)
            if (
                (not mother.hv_diagnosed)
                and mother.is_alive
                and (self.rng.random_sample() < p["prob_hiv_test_at_anc_or_delivery"])
            ):
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    hsi_event=HSI_Hiv_TestAndRefer(
                        person_id=abs(mother_id),  # mother's id (true or direct birth)
                        module=self,
                        referred_from="ANC_routine",
                    ),
                    priority=1,
                    topen=self.sim.date,
                    tclose=None,
                )

        # if mother known HIV+, schedule virological test for infant and give prep
        if mother.hv_diagnosed and df.at[child_id, "is_alive"]:
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                hsi_event=HSI_Hiv_StartInfantProphylaxis(
                    person_id=child_id,
                    module=self,
                    referred_from="on_birth",
                    repeat_visits=0),
                priority=1,
                topen=self.sim.date,
                tclose=None,
            )

            if "newborn_outcomes" not in self.sim.modules and (
                    self.rng.random_sample() < p['prob_hiv_test_for_newborn_infant']):
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    hsi_event=HSI_Hiv_TestAndRefer(
                        person_id=child_id,
                        module=self,
                        referred_from='Infant_testing'),
                    priority=1,
                    topen=self.sim.date + pd.DateOffset(weeks=p["infant_hiv_test_first_weeks"]),
                    tclose=None,
                )

            # these later infant tests are not in newborn_outcomes
            if self.rng.random_sample() < p['prob_hiv_test_for_newborn_infant']:
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    hsi_event=HSI_Hiv_TestAndRefer(
                        person_id=child_id,
                        module=self,
                        referred_from='Infant_testing'),
                    priority=1,
                    topen=self.sim.date + pd.DateOffset(months=p["infant_hiv_test_second_months"]),
                    tclose=None,
                )

                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    hsi_event=HSI_Hiv_TestAndRefer(
                        person_id=child_id,
                        module=self,
                        referred_from='Infant_testing'),
                    priority=1,
                    topen=self.sim.date + pd.DateOffset(months=p["infant_hiv_test_third_months"]),
                    tclose=None,
                )

    def report_daly_values(self):
        """Report DALYS for HIV, based on current symptomatic state of persons."""
        df = self.sim.population.props

        dalys = pd.Series(data=0, index=df.loc[df.is_alive].index, dtype=float)

        # All those infected get the 'infected but not AIDS' daly_wt:
        dalys.loc[df.hv_inf & (df.hv_art == "not")] = self.daly_wts["hiv_infection_but_not_aids"]

        # infected and on ART and virally suppressed
        dalys.loc[df.hv_inf & (df.hv_art == "on_VL_suppressed")] = self.daly_wts["hiv_infection_on_ART"]

        # Overwrite the value for those that currently have symptoms of AIDS with the 'AIDS' daly_wt:
        dalys.loc[
            self.sim.modules["SymptomManager"].who_has("aids_symptoms")
        ] = self.daly_wts["aids"]

        return dalys

    def report_summary_stats(self):
        # This reports age- and sex-specific prevalence of HIV for all individuals
        df = self.sim.population.props
        number_by_age_group_sex = get_counts_by_sex_and_age_group(df, 'hv_inf')
        return {'number_living_with_hiv': number_by_age_group_sex}

    def mtct_during_breastfeeding(self, mother_id, child_id):
        """
        Compute risk of mother-to-child transmission and schedule HivInfectionDuringBreastFeedingEvent.
        If the child is breastfeeding currently, consider the time-until-infection assuming a constantly monthly risk of
         transmission. If the breastfeeding has ceased by the time of the scheduled infection, then it will not run.
        (This means that this event can be run at birth or at the time of the mother's infection without the need for
        further polling etc.)
        """

        df = self.sim.population.props
        params = self.parameters

        if df.at[mother_id, "hv_art"] == "on_VL_suppressed":
            monthly_prob_mtct_bf = params["monthly_prob_mtct_bf_treated"]
        else:
            monthly_prob_mtct_bf = params["monthly_prob_mtct_bf_untreated"]

        if monthly_prob_mtct_bf > 0.0:
            months_to_infection = int(self.rng.exponential(1 / monthly_prob_mtct_bf))
            date_of_infection = self.sim.date + pd.DateOffset(
                months=months_to_infection
            )
            self.sim.schedule_event(
                HivInfectionDuringBreastFeedingEvent(person_id=child_id, module=self),
                date_of_infection,
            )

    def do_new_infection(self, person_id):
        """
        Enact that this person is infected with HIV
        * Update their hv_inf status and hv_date_inf
        * Schedule the AIDS onset event for this person
        """
        df = self.sim.population.props

        # Update HIV infection status for this person
        df.at[person_id, "hv_inf"] = True
        df.at[person_id, "hv_date_inf"] = self.sim.date

        # Schedule AIDS onset events for this person
        parameters = self.get_time_from_infection_to_aids_distribution_parameters(
            [person_id]
        )
        date_onset_aids = (
            self.sim.date
            + self.sample_time_from_infection_to_aids_given_parameters(*parameters)
        ).iloc[0]
        self.sim.schedule_event(
            event=HivAidsOnsetEvent(self, person_id, cause='AIDS_non_TB'),
            date=date_onset_aids,
        )

    def sample_time_from_infection_to_aids_given_parameters(self, scale, shape, offset):
        """Generate time(s) between onset of infection and AIDS as Pandas time deltas.

        The times are generated from translated Weibull distributions discretised to
        an integer number of months.

        :param scale: Scale parameters of Weibull distributions (unit: years).
        :param shape: Shape parameters of Weibull distributions.
        :param offset: Offset to (negatively) shift Weibull variable by (unit: months).

        :return: Generated time deltas.
        """

        months_to_death = self.rng.weibull(shape) * scale * 12
        months_to_aids = np.round(months_to_death - offset).clip(0).astype(int)

        return pd.to_timedelta(months_to_aids * 30.5, unit='D')

    def get_time_from_infection_to_aids_distribution_parameters(self, person_ids):
        """Compute per-person parameters of distribution of time from infection to aids.

        Evaluates three linear models which output age specific scale, shape and offset
        parameters for the (translated) Weibull distribution used to generate the time
        from infection to aids for an individual.

        For those infected prior to, or at, birth, a Weibull distribution with shape
        parameter 1 (equivalent to an exponential distribution) is used.

        For those infected after birth a Weibull distribution with both shape and
        scale depending on age is used.

        :param person_ids: Iterable of ID indices of individuals to get parameters for.

        :return: Per-person parameters as a 3-tuple ``(scale, shape, offset)`` of
            ``pandas.Series`` objects.
        """
        subpopulation = self.sim.population.props.loc[person_ids]
        # get the scale parameters (unit: years)
        scale = self.lm["scale_parameter_for_infection_to_death"].predict(subpopulation)
        # get the shape parameter
        shape = self.lm["shape_parameter_for_infection_to_death"].predict(subpopulation)
        # get the mean months between aids and death (unit: months)
        offset = self.lm["offset_parameter_for_months_from_aids_to_death"].predict(
            subpopulation
        )
        return scale, shape, offset

    def get_time_from_aids_to_death(self):
        """Gives time between onset of AIDS and death, returning a pd.DateOffset.
        Assumes that the time between onset of AIDS symptoms and deaths is exponentially distributed.
        """
        mean = self.parameters["mean_months_between_aids_and_death"]
        draw_number_of_months = int(np.round(self.rng.exponential(mean)))
        return pd.DateOffset(months=(draw_number_of_months + 1))

    def do_when_hiv_diagnosed(self, person_id):
        """Things to do when a person has been tested and found (newly) to be HIV-positive:.
        * Consider if ART should be initiated, and schedule HSI if so.
        The person should not yet be on ART.
        """
        df = self.sim.population.props
        p = self.parameters

        if not (df.loc[person_id, "hv_art"] == "not"):
            logger.warning(
                key="message",
                data="This event should not be running. do_when_diagnosed is for newly diagnosed persons.",
            )

        # Consider if the person will be referred to start ART
        if df.loc[person_id, "age_years"] <= p["age_threshold_child_adult_distinction"]:
            starts_art = True
        else:
            starts_art = self.rng.random_sample() < self.prob_art_start_after_test(self.sim.date.year)

        if starts_art:
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Hiv_StartOrContinueTreatment(person_id=person_id, module=self,
                                                 facility_level_of_this_hsi="1a"),
                topen=self.sim.date,
                tclose=None,
                priority=0,
            )

    def prob_art_start_after_test(self, year):
        """ returns the probability of starting ART after a positive HIV test
        this value for initiation can be higher than the current reported coverage levels
        to account for defaulters
        """
        p = self.parameters
        prob_art = p["prob_start_art_or_vs"]
        current_year = year if year <= p["end_year_treatment_data"] else p["end_year_treatment_data"]

        # use iloc to index by position as index will change by year
        return_prob = prob_art.loc[
                          (prob_art.year == current_year) &
                          (prob_art.age == "adults"),
                          "prob_art_if_dx_updated"].values[0]

        return return_prob

    def prob_viral_suppression(self, df, year, age_of_person):
        """ returns the probability of viral suppression once on ART
        data from 2010-2025 from spectrum
        time-series ends at 2025
        """

        p = self.parameters
        prob_vs = self.parameters["prob_start_art_or_vs"]
        current_year = min(year, p["end_year_treatment_data"])

        age_of_person = age_of_person
        age_group = "adults" if age_of_person >= p['age_threshold_child_adult_distinction'] else "children"

        base = prob_vs.loc[
            (prob_vs.year == current_year) &
            (prob_vs.age == age_group),
            "adjusted_viral_suppression_on_art"].values[0]

        assert base is not None

        # Children: return baseline
        if age_of_person < 15:
            return max(0.0, min(1.0, base))

        # Adults: apply young adult gap and rebalance 30+ to preserve adult average
        gap = self.parameters["young_adult_vls_factor"]  # absolute fraction, 0.08

        # Adults (15+) currently on ART
        mask_adult_on_art = (df["age_years"] >= p['age_threshold_child_adult_distinction']) & (df["hv_art"].ne("not"))
        den = mask_adult_on_art.sum()

        # Proportion of young adults (15–29) among adults on ART
        if den:
            w = df.loc[mask_adult_on_art, "age_years"].between(
                p['age_threshold_child_adult_distinction'],
                29,
                inclusive="both"
            ).mean()
        else:
            w = self.parameters["proportion_young_adult_on_art"]

        # Feasibility so both strata remain in [0,1]
        # young = base - gap >= 0 → gap ≤ base
        # old = base + (w/(1-w))·gap ≤ 1 → gap ≤ (1-w)(1-base)/w
        max_gap_young = base
        max_gap_old = ((1.0 - w) * (1.0 - base)) / w
        gap = max(0.0, min(gap, max_gap_young, max_gap_old))

        if 15 <= age_of_person <= 29:
            return max(0.0, min(1.0, base - gap))
        else:
            return max(0.0, min(1.0, base + (w / (1.0 - w)) * gap))

    def update_viral_suppression_status(self):
        """Helper function to determine whether person who is currently not virally suppressed
        will become virally suppressed following viral load test"
        """
        p = self.parameters

        return (
            "on_VL_suppressed"
            if self.rng.random_sample() < p["prob_of_viral_suppression_following_VL_test"]
            else "on_not_VL_suppressed"
        )

    def setup_art_dispensation_lookup(self):
        """Preprocess the dispensation probability table into a nested dict for fast lookup."""
        df = self.parameters["dispensation_period_months"]
        lookup = {}

        for (sub_group, year), group_df in df.groupby(["sub_group", "year"]):
            lengths = group_df["length_of_dispensation_months"].values
            probs = group_df["probability"].values
            lookup[(sub_group, year)] = (lengths, probs)

        self._art_dispensation_lookup = lookup

    def get_art_dispensation_length(self, year, person, currently_breastfeeding):
        """
        Return ART dispensation duration (in months) based on year and person attributes
        """
        # subgroup assignment
        if person["age_years"] < 15:
            sub_group = "child"
        elif person["sex"] == "F":
            if person["is_pregnant"]:
                sub_group = "pregnant"
            elif currently_breastfeeding:
                sub_group = "breastfeeding"
            else:
                sub_group = "adult_female"
        else:
            sub_group = "adult_male"

        # Restrict to year
        if year < 2021:
            return self.parameters["initial_dispensation_period_months"]
        year = min(year, 2025)

        key = (sub_group, year)
        try:
            lengths, probs = self._art_dispensation_lookup[key]
            return np.random.choice(lengths, p=probs)
        except KeyError:
            raise ValueError(f"No ART dispensation data for sub_group={sub_group} and year={year}")

    def stops_treatment(self, person_id):
        """Helper function that is called when someone stops being on ART.
        Sets the flag for ART status. If the person was already on ART, it schedules a new AIDSEvent
        """

        df = self.sim.population.props

        # Schedule a new AIDS onset event if the person was on ART up until now
        if df.at[person_id, "hv_art"] == "on_VL_suppressed":
            months_to_aids = int(
                np.floor(
                    self.rng.exponential(
                        scale=self.parameters["art_default_to_aids_mean_years"]
                    )
                    * 12.0
                )
            )
            self.sim.schedule_event(
                event=HivAidsOnsetEvent(person_id=person_id, module=self, cause='AIDS_non_TB'),
                date=self.sim.date + pd.DateOffset(months=months_to_aids),
            )

        # Set that the person is no longer on ART or cotrimoxazole
        df.at[person_id, "hv_art"] = "not"
        df.at[person_id, "hv_on_cotrimoxazole"] = False

    def per_capita_testing_rate(self):
        """This calculates the numbers of hiv tests performed in each time period.
        It looks at the cumulative number of tests ever performed and subtracts the
        number calculated at the last time point.
        Values are converted to per capita testing rates.
        This function is called by the logger and can be called at any frequency
        """

        df = self.sim.population.props

        if not self.stored_test_numbers:
            # If it's the first year, set previous_test_numbers to 0
            previous_test_numbers = 0
        else:
            # For subsequent years, retrieve the last stored number
            previous_test_numbers = self.stored_test_numbers[-1]

        # Calculate number of tests now performed - cumulative, include those who have died
        number_tests_new = df.hv_number_tests.sum()

        # Number of tests performed in the last time period
        number_tests_in_last_period = number_tests_new - previous_test_numbers

        # Store the number of tests performed in this year for future reference
        self.stored_test_numbers.append(number_tests_in_last_period)

        # per-capita testing rate
        per_capita_testing = number_tests_in_last_period / len(df[df.is_alive])

        # return updated value for time-period
        return per_capita_testing

    def decide_whether_hiv_test_for_mother(self, person_id, referred_from) -> bool:
        """
        This will return a True/False for whether an HIV test should be scheduled for a mother; and schedule the HIV
        Test if a test should be scheduled.
        This is called from `labour.py` under `interventions_delivered_pre_discharge` and
        `care_of_women_during_pregnancy.py`.
        Mothers who are not already diagnosed will have an HIV test with a certain probability defined by a parameter;
         mothers who are diagnosed already will not have another HIV test.
        """
        df = self.sim.population.props

        if not df.at[person_id, 'hv_diagnosed'] and (
                self.rng.random_sample() < self.parameters['prob_hiv_test_at_anc_or_delivery']):

            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Hiv_TestAndRefer(
                    person_id=person_id,
                    module=self,
                    referred_from=referred_from),
                topen=self.sim.date,
                tclose=None,
                priority=0)

            return True

        else:
            return False

    def decide_whether_hiv_test_for_infant(self, mother_id, child_id) -> None:
        """ This will schedule an HIV testing HSI for a child under certain conditions.
        It is called from newborn_outcomes.py under hiv_screening_for_at_risk_newborns.
        """

        df = self.sim.population.props
        p = self.parameters
        mother_id = mother_id
        child_id = child_id

        if (
            not df.at[child_id, "hv_diagnosed"]
            and df.at[mother_id, "hv_diagnosed"]
            and (df.at[child_id, "nb_pnc_check"] == 1)
            and (self.rng.random_sample() < self.parameters["prob_hiv_test_for_newborn_infant"])
        ):
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Hiv_TestAndRefer(
                    person_id=child_id,
                    module=self,
                    referred_from="newborn_outcomes"),
                topen=self.sim.date + pd.DateOffset(weeks=p["infant_hiv_test_first_weeks"]),
                tclose=None,
                priority=0
            )

    def perform_tdf_test(self, is_suppressed: bool) -> bool:
        """
        Simulate TDF urine test result based on known viral suppression status.

        Parameters:
        - is_suppressed (bool): True if the person is virally suppressed, False if not.

        Returns:
        - true (positive) or false (negative) based on test result
        """
        if is_suppressed:
            prob_positive = self.parameters["p_tdf_positive_given_suppressed"]
        else:
            prob_positive = self.parameters["p_tdf_positive_given_not_suppressed"]

        return self.rng.random_sample() < prob_positive

    def check_config_of_properties(self):
        """check that the properties are currently configured correctly"""
        df = self.sim.population.props
        df_alive = df.loc[df.is_alive]

        # basic check types of columns and dtypes
        orig = self.sim.population.new_row
        assert (df.dtypes == orig.dtypes).all()

        def is_subset(col_for_set, col_for_subset):
            # Confirms that the series of col_for_subset is true only for a subset of the series for col_for_set
            return set(col_for_subset.loc[col_for_subset].index).issubset(
                col_for_set.loc[col_for_set].index
            )

        # Check that core properties of current status are never None/NaN/NaT
        assert not df_alive.hv_inf.isna().any()
        assert not df_alive.hv_art.isna().any()
        assert not df_alive.hv_behaviour_change.isna().any()
        assert not df_alive.hv_diagnosed.isna().any()
        assert not df_alive.hv_number_tests.isna().any()

        # Check that the core HIV properties are 'nested' in the way expected.
        assert is_subset(
            col_for_set=df_alive.hv_inf, col_for_subset=df_alive.hv_diagnosed
        )
        assert is_subset(
            col_for_set=df_alive.hv_diagnosed, col_for_subset=(df_alive.hv_art != "not")
        )

        # Check that if person is not infected, the dates of HIV events are None/NaN/NaT
        assert df_alive.loc[~df_alive.hv_inf, "hv_date_inf"].isna().all()

        # Check that dates consistent for those infected with HIV
        assert not df_alive.loc[df_alive.hv_inf].hv_date_inf.isna().any()
        assert (
            df_alive.loc[df_alive.hv_inf].hv_date_inf
            >= df_alive.loc[df_alive.hv_inf].date_of_birth
        ).all()

        # Check alignment between AIDS Symptoms and status and infection and ART status
        has_aids_symptoms = set(
            self.sim.modules["SymptomManager"].who_has("aids_symptoms")
        )
        assert has_aids_symptoms.issubset(
            df_alive.loc[df_alive.is_alive & df_alive.hv_inf].index
        )

        # a person can have AIDS onset if they're virally suppressed if had active TB
        # if cured of TB and still virally suppressed, AIDS symptoms are removed
        assert 0 == len(
            has_aids_symptoms.intersection(
                df_alive.loc[
                    df_alive.is_alive
                    & (df_alive.hv_art == "on_VL_suppressed")
                    & (df_alive.tb_inf == "uninfected")
                    ].index
            )
        )

    def do_at_generic_first_appt(
        self,
        person_id: int,
        symptoms: List[str],
        schedule_hsi_event: HSIEventScheduler,
        **kwargs,
    ) -> None:
        # 'Automatic' testing for HIV for everyone attending care with AIDS symptoms:
        #  - suppress the footprint (as it done as part of another appointment)
        #  - do not do referrals if the person is HIV negative (assumed not time for counselling etc).
        if "aids_symptoms" in symptoms:
            event = HSI_Hiv_TestAndRefer(
                person_id=person_id,
                module=self,
                referred_from="hsi_generic_first_appt",
                suppress_footprint=True,
                do_not_refer_if_neg=True,
            )
            schedule_hsi_event(event, priority=0, topen=self.sim.date)


# ---------------------------------------------------------------------------
#   Main Polling Event
# ---------------------------------------------------------------------------


class HivRegularPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """ The HIV Regular Polling Events
    * Schedules persons becoming newly infected through horizontal transmission
    * Schedules who will present for voluntary ("spontaneous") testing
    """

    def __init__(self, module):
        super().__init__(
            module, frequency=DateOffset(months=module.parameters["hiv_regular_polling_event_frequency_months"])
        )  # repeats based on frequency of 'hiv_regular_polling_event_frequency_months', but this can be changed

    def apply(self, population):

        df = population.props
        p = self.module.parameters
        rng = self.module.rng

        fraction_of_year_between_polls = self.frequency.months / 12
        beta = p["beta"] * fraction_of_year_between_polls

        # ----------------------------------- HORIZONTAL TRANSMISSION -----------------------------------
        def horizontal_transmission(to_sex, from_sex):
            # Count current number of alive 15-80 year-olds at risk of transmission
            # (= those infected and not VL suppressed):
            n_infectious = len(
                df.loc[
                    df.is_alive
                    & df.age_years.between(p['age_threshold_child_adult_distinction'], p['age_max_adult'])
                    & df.hv_inf
                    & (df.hv_art != "on_VL_suppressed")
                    & (df.sex == from_sex)
                    ]
            )

            if n_infectious > 0:

                # Get Susceptible (non-infected alive 15-80 year-old) persons:
                susc_idx = df.loc[
                    df.is_alive
                    & ~df.hv_inf
                    & df.age_years.between(p['age_threshold_child_adult_distinction'], p['age_max_adult'])
                    & (df.sex == to_sex)
                    ].index
                n_susceptible = len(susc_idx)

                # Compute chance that each susceptible person becomes infected:
                #  - relative chance of infection (acts like a scaling-factor on 'beta')
                rr_of_infection = self.module.lm["rr_of_infection"].predict(
                    df.loc[susc_idx]
                )

                #  - probability of infection = beta * I/N
                p_infection = (
                    rr_of_infection * beta * (n_infectious / (n_infectious + n_susceptible))
                )

                # New infections:
                will_be_infected = (
                    self.module.rng.random_sample(len(p_infection)) < p_infection
                )
                idx_new_infection = will_be_infected[will_be_infected].index

                idx_new_infection_fsw = []

                # additional risk for fsw
                if to_sex == "F":
                    fsw_at_risk = df.loc[
                        df.is_alive
                        & ~df.hv_inf
                        & df.li_is_sexworker
                        & ~(df.hv_is_on_prep_oral | df.hv_is_on_prep_inj)
                        & df.age_years.between(p['age_threshold_child_adult_distinction'], p['age_max_adult'])
                        ].index

                    #  - probability of infection - relative risk applies only to fsw
                    p_infection_fsw = (
                        self.module.parameters["rr_fsw"] * beta * (n_infectious / (n_infectious + n_susceptible))
                    )

                    fsw_infected = (
                        self.module.rng.random_sample(len(fsw_at_risk)) < p_infection_fsw
                    )
                    idx_new_infection_fsw = fsw_at_risk[fsw_infected]

                idx_new_infection = list(idx_new_infection) + list(idx_new_infection_fsw)

                # Schedule the date of infection for each new infection:
                for idx in idx_new_infection:
                    date_of_infection = self.sim.date + pd.DateOffset(
                        days=self.module.rng.randint(0, 365 * fraction_of_year_between_polls)
                    )
                    self.sim.schedule_event(
                        HivInfectionEvent(self.module, idx), date_of_infection
                    )

        # ----------------------------------- SPONTANEOUS TESTING -----------------------------------
        def spontaneous_testing(current_year):

            # extract annual testing rates from MoH Reports
            test_rates = p["hiv_testing_rates"]

            testing_rate_adults = test_rates.loc[
                                      test_rates.year == current_year, "annual_testing_rate_adults"
                                  ].values[0] * p["hiv_testing_rate_adjustment"]

            # adult testing trends also informed by demographic characteristics
            # relative probability of testing - this may skew testing rates higher or lower than moh reports
            rr_of_test = self.module.lm["lm_spontaneous_test_12m"].predict(
                df[df.is_alive & (df.age_years >= p['age_threshold_child_adult_distinction'])])
            mean_prob_test = (rr_of_test * testing_rate_adults).mean()
            scaled_prob_test = (rr_of_test * testing_rate_adults) / mean_prob_test
            overall_prob_test = scaled_prob_test * testing_rate_adults

            random_draw = rng.random_sample(size=len(df[df.is_alive &
                                                        (df.age_years >= p['age_threshold_child_adult_distinction'])]))
            adult_tests_idx = df.loc[df.is_alive & (df.age_years >= p['age_threshold_child_adult_distinction']) &
                                     (random_draw < overall_prob_test)].index

            idx_will_test = adult_tests_idx

            for person_id in idx_will_test:
                date_test = self.sim.date + pd.DateOffset(
                    days=self.module.rng.randint(0, 365 * fraction_of_year_between_polls)
                )
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    hsi_event=HSI_Hiv_TestAndRefer(person_id=person_id, module=self.module, referred_from='HIV_poll'),
                    priority=1,
                    topen=date_test,
                    tclose=date_test + pd.DateOffset(
                        months=self.frequency.months
                    ),  # (to occur before next polling)
                )

        # ----------------------------------- SELF-TEST -----------------------------------
        def self_tests():
            # scaled in same way as general testing using linear model
            # can be a repeat test if diagnosed already
            # the same people may have been selected from general testing above

            if p["selftest_available"] & (self.sim.date.year >= p["selftest_start_year"]):

                test_rates = p["annual_rate_selftest"]

                # adult testing trends also informed by demographic characteristics
                # relative probability of testing - this may skew testing rates higher or lower than moh reports
                rr_of_test = self.module.lm["lm_spontaneous_test_12m"].predict(
                    df[df.is_alive & (df.age_years >= 15)])
                mean_prob_test = (rr_of_test * test_rates).mean()
                scaled_prob_test = (rr_of_test * test_rates) / mean_prob_test
                overall_prob_test = scaled_prob_test * test_rates

                random_draw = rng.random_sample(size=len(df[df.is_alive & (df.age_years >= 15)]))
                self_tests_idx = df.loc[
                    df.is_alive & (df.age_years >= 15) & (random_draw < overall_prob_test)].index

                for person_id in self_tests_idx:
                    date_test = self.sim.date + pd.DateOffset(
                        days=self.module.rng.randint(0, 365 * fraction_of_year_between_polls)
                    )

                    self.sim.modules["HealthSystem"].schedule_hsi_event(
                        hsi_event=HSI_Hiv_SelfTest(
                            person_id=person_id,
                            module=self.module,
                        ),
                        priority=1,
                        topen=date_test,
                        tclose=date_test + pd.DateOffset(months=self.frequency.months),  # before next polling
                    )

        # ----------------------------------- PrEP poll for AGYW and pregnant women -----------------------------------
        def prep_for_agyw():

            if self.sim.date.year >= p["prep_start_year"]:

                # select highest risk agyw
                # Common mask for alive, undiagnosed, female, not on PrEP
                common = (
                    df.is_alive
                    & ~df.hv_diagnosed
                    & (df.sex == "F")
                    & ~(df.hv_is_on_prep_oral | df.hv_is_on_prep_inj)
                )

                # AGYW 15–25
                agyw_idx = df.loc[common & df.age_years.between(p['age_min_agyw'], p['age_max_agyw'])].index

                # Pregnant women (any age)
                pregnant_idx = df.loc[common & df.is_pregnant].index

                # Combine and drop duplicates
                eligible_idx = agyw_idx.union(pregnant_idx)

                if len(eligible_idx) == 0:
                    return  # no eligible AGYW

                # calculate relative risk of infection
                rr_of_infection = self.module.lm["rr_of_infection"].predict(df.loc[eligible_idx])
                mean_risk = rr_of_infection.mean()
                scaled_risk = rr_of_infection / mean_risk

                # get probabilities for receiving prep from linear model
                prob_prep = self.module.lm["lm_prep_agyw"].predict(df.loc[eligible_idx],
                                                                   year=self.sim.date.year,
                                                                   )

                # number of AGYW expected to get PrEP = mean(prob_prep) * total AGYW
                expected_n = int(round(prob_prep.mean() * len(eligible_idx)))

                # subset to top 50% by risk
                top_risk_threshold = np.percentile(scaled_risk, 50)
                high_risk_mask = scaled_risk >= top_risk_threshold

                high_risk_idx = eligible_idx[high_risk_mask]
                high_risk_probs = prob_prep[high_risk_mask]

                if len(high_risk_idx) == 0:
                    return  # no one in top risk group

                # rescale probabilities so total matches expected_n
                rescaled_probs = (
                    high_risk_probs * (expected_n / high_risk_probs.sum())
                    if expected_n > 0 and high_risk_probs.sum() > 0
                    else pd.Series(0.0, index=high_risk_probs.index)
                )

                # cap at 1
                rescaled_probs = np.minimum(rescaled_probs, 1.0)

                # assign PrEP probabilistically
                give_prep = high_risk_idx[
                    self.module.rng.random_sample(len(high_risk_idx)) < rescaled_probs
                    ]

                # dates are scattered throughout year
                for person in give_prep:
                    if (self.module.parameters['injectable_prep_allowed'] &
                            (self.sim.date.year >= 2025)):

                        prob_injectable = self.module.parameters['prob_injectable_prep_vs_oral']

                        type_of_prep = self.module.rng.choice(['injectable', 'oral'],
                                                              p=[prob_injectable, 1 - prob_injectable])
                    else:
                        type_of_prep = 'oral'

                    date_prep = self.sim.date + pd.DateOffset(
                        days=self.module.rng.randint(0, 365 * fraction_of_year_between_polls)
                    )

                    self.sim.modules["HealthSystem"].schedule_hsi_event(
                        hsi_event=HSI_Hiv_StartOrContinueOnPrep(person_id=person,
                                                                module=self.module,
                                                                type_of_prep=type_of_prep),
                        priority=1,
                        topen=date_prep,
                        tclose=date_prep + pd.DateOffset(
                            months=self.frequency.months
                        )
                    )

        # ----------------------------------- PrEP poll for FSW -----------------------------------
        def prep_for_fsw():
            if self.sim.date.year >= p["prep_start_year"]:

                random_draw = self.module.rng.random_sample(size=len(df))
                eligible_fsw_idx = df.loc[df.is_alive &
                                          ~df.hv_diagnosed &
                                          df.li_is_sexworker &
                                          ~(df.hv_is_on_prep_oral | df.hv_is_on_prep_inj) &
                                          (random_draw < p["prob_prep_for_fsw_after_hiv_test"])].index

                for person in eligible_fsw_idx:
                    if (self.module.parameters['injectable_prep_allowed'] &
                            (self.sim.date.year >= 2025)):

                        prob_injectable = self.module.parameters['prob_injectable_prep_vs_oral']

                        type_of_prep = self.module.rng.choice(['injectable', 'oral'],
                                                              p=[prob_injectable, 1 - prob_injectable])
                    else:
                        type_of_prep = 'oral'

                    date_prep = self.sim.date + pd.DateOffset(
                        days=self.module.rng.randint(0, 365 * fraction_of_year_between_polls)
                    )

                    self.sim.modules["HealthSystem"].schedule_hsi_event(
                        hsi_event=HSI_Hiv_StartOrContinueOnPrep(person_id=person,
                                                                module=self.module,
                                                                type_of_prep=type_of_prep),
                        priority=1,
                        topen=date_prep,
                        tclose=date_prep + pd.DateOffset(
                            months=self.frequency.months
                        )
                    )

        # ----------------------------------- SPONTANEOUS VMMC FOR <15 YRS -----------------------------------
        def vmmc_for_child():
            """schedule the HSI_Hiv_Circ for <15 yrs males according to his age, circumcision status
            and the probability of being circumcised"""
            # work out who will be circumcised.
            will_go_to_circ = self.module.lm["lm_circ_child"].predict(
                df.loc[
                    df.is_alive
                    & (df.sex == "M")
                    & (df.age_years < p["age_threshold_child_adult_distinction"])
                    & (~df.li_is_circ)
                    ],
                self.module.rng,
                year=self.sim.date.year,
            )

            # schedule the HSI based on the probability
            for person_id in will_go_to_circ.loc[will_go_to_circ].index:
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    HSI_Hiv_Circ(person_id=person_id, module=self.module),
                    topen=self.sim.date,
                    tclose=None,
                    priority=0,
                )

        # Horizontal transmission: Male --> Female
        horizontal_transmission(from_sex="M", to_sex="F")

        # Horizontal transmission: Female --> Male
        horizontal_transmission(from_sex="F", to_sex="M")

        # testing
        # if year later than 2020, set testing rates to those reported in 2020
        if self.sim.date.year <= p['end_year_testing_data']:
            current_year = self.sim.date.year
        else:
            current_year = p['end_year_testing_data']
        spontaneous_testing(current_year=current_year)

        # self-test
        self_tests()

        # PrEP for AGYW
        prep_for_agyw()

        # PrEP for FSW
        prep_for_fsw()

        # VMMC for <15 yrs in the population
        vmmc_for_child()


# ---------------------------------------------------------------------------
#   Natural History Events
# ---------------------------------------------------------------------------

class HivInfectionEvent(Event, IndividualScopeEventMixin):
    """ This person will become infected.
    * Do the infection process
    * Check for onward transmission through MTCT if the infection is to a mother who is currently breastfeeding.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props

        # Check person is_alive
        if not df.at[person_id, "is_alive"]:
            return

        # Onset the infection for this person (which will schedule progression etc)
        self.module.do_new_infection(person_id)

        # Consider mother-to-child-transmission (MTCT) from this person to their children:
        children_of_this_person_being_breastfed = df.loc[
            (df.mother_id == person_id) & (df.nb_breastfeeding_status != "none")
            ].index
        # - Do the MTCT routine for each child:
        for child_id in children_of_this_person_being_breastfed:
            self.module.mtct_during_breastfeeding(person_id, child_id)


class HivInfectionDuringBreastFeedingEvent(Event, IndividualScopeEventMixin):
    """ This person will become infected during breastfeeding
    * Do the infection process
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props

        # Check person is_alive
        if not df.at[person_id, "is_alive"]:
            return

        # Check person is breastfed currently
        if df.at[person_id, "nb_breastfeeding_status"] == "none":
            return

        # If child is on NVP for HIV prophylaxis, then do not let the infection occur
        # (the prophylaxis is assumed to be perfectly effective in blocking transmission)
        if df.at[person_id, "hv_is_on_prep_oral"]:
            return

        # Onset the infection for this person (which will schedule progression etc)
        self.module.do_new_infection(person_id)


class HivAidsOnsetEvent(Event, IndividualScopeEventMixin):
    """ This person has developed AIDS.
    * Update their symptomatic status
    * Record the date at which AIDS onset
    * Schedule the AIDS death
    """

    def __init__(self, module, person_id, cause):
        super().__init__(module, person_id=person_id)

        self.cause = cause

    def apply(self, person_id):

        df = self.sim.population.props
        p = self.module.parameters

        # Return if person is dead or no HIV+
        if not df.at[person_id, "is_alive"] or not df.at[person_id, "hv_inf"]:
            return

        # eligible for AIDS onset:
        # not VL suppressed no active tb
        # not VL suppressed active tb
        # VL suppressed active tb

        # Do nothing if person is now on ART and VL suppressed (non-VL suppressed has no effect)
        # if cause is TB, allow AIDS onset
        if (df.at[person_id, "hv_art"] == "on_VL_suppressed") and (self.cause != 'AIDS_TB'):
            return

        # need to delay onset of AIDS (non-tb) to compensate for AIDS-TB
        if (self.cause == "AIDS_non_TB") and (
                self.sim.modules["Hiv"].rng.rand() < self.sim.modules["Hiv"].parameters["prop_delayed_aids_onset"]):

            # redraw time to aids and reschedule
            months_to_aids = int(
                np.floor(
                    self.sim.modules["Hiv"].rng.exponential(
                        scale=self.sim.modules["Hiv"].parameters["art_default_to_aids_mean_years"]
                    )
                    * 12.0
                )
            )

            self.sim.schedule_event(
                event=HivAidsOnsetEvent(person_id=person_id, module=self.sim.modules["Hiv"], cause='AIDS_non_TB'),
                date=self.sim.date + pd.DateOffset(months=months_to_aids),
            )

        # else assign aids onset and schedule aids death
        else:

            # if eligible for aids onset (not treated with ART or currently has active TB):
            # Update Symptoms
            self.sim.modules["SymptomManager"].change_symptom(
                person_id=person_id,
                symptom_string="aids_symptoms",
                add_or_remove="+",
                disease_module=self.sim.modules["Hiv"],
            )

            # Schedule AidsDeath
            date_of_aids_death = self.sim.date + self.sim.modules["Hiv"].get_time_from_aids_to_death()

            if self.cause == "AIDS_non_TB":

                # cause is HIV
                self.sim.schedule_event(
                    event=HivAidsDeathEvent(
                        person_id=person_id, module=self.sim.modules["Hiv"], cause=self.cause
                    ),
                    date=date_of_aids_death,
                )
                # schedule hospital stay
                beddays = self.sim.modules["Hiv"].rng.randint(
                    low=p['length_of_inpatient_stay_if_terminal'][0], high=p['length_of_inpatient_stay_if_terminal'][1])
                date_admission = date_of_aids_death - pd.DateOffset(days=beddays)
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    hsi_event=HSI_Hiv_EndOfLifeCare(
                        person_id=person_id,
                        module=self.sim.modules["Hiv"],
                        beddays=beddays,
                    ),
                    priority=0,
                    topen=(
                        date_admission
                        if (date_admission > self.sim.date)
                        else self.sim.date
                    ),
                    tclose=date_of_aids_death,
                )

            else:
                # cause is active TB
                self.sim.schedule_event(
                    event=HivAidsTbDeathEvent(
                        person_id=person_id, module=self.sim.modules["Hiv"], cause=self.cause
                    ),
                    date=date_of_aids_death,
                )
                # schedule hospital stay
                beddays = self.sim.modules["Hiv"].rng.randint(
                    low=p['length_of_inpatient_stay_if_terminal'][0], high=p['length_of_inpatient_stay_if_terminal'][1])
                date_admission = date_of_aids_death - pd.DateOffset(days=beddays)
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    hsi_event=HSI_Hiv_EndOfLifeCare(
                        person_id=person_id,
                        module=self.sim.modules["Hiv"],
                        beddays=beddays,
                    ),
                    priority=0,
                    topen=(
                        date_admission
                        if (date_admission >= self.sim.date)
                        else self.sim.date
                    ),
                    tclose=date_of_aids_death,
                )


class HivAidsDeathEvent(Event, IndividualScopeEventMixin):
    """
    Causes someone to die of AIDS, if they are not VL suppressed on ART.
    if death scheduled by tb-aids, death event is HivAidsTbDeathEvent
    if death scheduled by hiv but person also has active TB, cause of
    death is AIDS_TB
    """

    def __init__(self, module, person_id, cause):
        super().__init__(module, person_id=person_id)

        self.cause = cause
        assert self.cause in ["AIDS_non_TB", "AIDS_TB"]

    def apply(self, person_id):
        df = self.sim.population.props

        # Check person is_alive
        if not df.at[person_id, "is_alive"]:
            return

        # Do nothing if person is now on ART and VL suppressed (non VL suppressed has no effect)
        # only if no current TB infection
        if (df.at[person_id, "hv_art"] == "on_VL_suppressed") and (
                df.at[person_id, "tb_inf"] != "active"):
            return

        # off ART, no TB infection
        if (df.at[person_id, "hv_art"] != "on_VL_suppressed") and (
                df.at[person_id, "tb_inf"] != "active"):
            # cause is HIV (no TB)
            self.sim.modules["Demography"].do_death(
                individual_id=person_id,
                cause="AIDS_non_TB",
                originating_module=self.module,
            )

        # on or off ART, active TB infection, schedule AidsTbDeathEvent
        if df.at[person_id, "tb_inf"] == "active":
            # cause is active TB
            self.sim.schedule_event(
                event=HivAidsTbDeathEvent(
                    person_id=person_id, module=self.module, cause="AIDS_TB"
                ),
                date=self.sim.date,
            )


class HivAidsTbDeathEvent(Event, IndividualScopeEventMixin):
    """
    This event is caused by someone co-infected with HIV and active TB
    it causes someone to die of AIDS-TB, death dependent on tb treatment status
    and not affected by ART status
    can be called by Tb or Hiv module
    if the random draw doesn't result in AIDS-TB death, an AIDS death (HivAidsDeathEvent) will be scheduled
    """

    def __init__(self, module, person_id, cause):
        super().__init__(module, person_id=person_id)

        self.cause = cause

    def apply(self, person_id):
        df = self.sim.population.props
        p = self.sim.modules["Hiv"].parameters

        # Check person is_alive
        if not df.at[person_id, "is_alive"]:
            return

        if df.at[person_id, 'tb_on_treatment']:

            risk_of_death = p["aids_tb_death_rate_with_tb_treatment"]

            if "CardioMetabolicDisorders" in self.sim.modules:
                if df.at[person_id, "nc_diabetes"]:
                    risk_of_death *= self.sim.modules["Tb"].parameters["rr_death_diabetes"]

            # treatment adjustment reduces probability of death
            if self.module.rng.rand() < risk_of_death:
                self.sim.modules["Demography"].do_death(
                    individual_id=person_id,
                    cause="AIDS_TB",
                    originating_module=self.module,
                )
            else:
                # if they survive, reschedule the aids death event
                # module calling rescheduled AIDS death should be Hiv (not TB)
                date_of_aids_death = self.sim.date + self.sim.modules["Hiv"].get_time_from_aids_to_death()

                self.sim.schedule_event(
                    event=HivAidsDeathEvent(
                        person_id=person_id,
                        module=self.sim.modules["Hiv"],
                        cause="AIDS_non_TB"
                    ),
                    date=date_of_aids_death,
                )
                # schedule hospital stay
                beddays = self.module.rng.randint(
                    low=p['length_of_inpatient_stay_if_terminal'][0], high=p['length_of_inpatient_stay_if_terminal'][1])
                date_admission = date_of_aids_death - pd.DateOffset(days=beddays)
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    hsi_event=HSI_Hiv_EndOfLifeCare(
                        person_id=person_id,
                        module=self.sim.modules["Hiv"],
                        beddays=beddays,
                    ),
                    priority=0,
                    topen=(
                        date_admission
                        if (date_admission >= self.sim.date)
                        else self.sim.date
                    ),
                    tclose=date_of_aids_death,
                )

        # aids-tb and not on tb treatment
        elif not df.at[person_id, 'tb_on_treatment']:
            risk_of_death = p["aids_tb_death_rate_no_tb_treatment"]
            if self.module.rng.rand() < risk_of_death:
                # Cause the death to happen immediately, cause defined by TB status
                self.sim.modules["Demography"].do_death(
                    individual_id=person_id, cause="AIDS_TB", originating_module=self.module
                )


class Hiv_DecisionToContinueOnPrEP(Event, IndividualScopeEventMixin):
    """Helper event that is used to 'decide' if someone on PrEP should continue on PrEP.
    This event is scheduled by 'HSI_Hiv_StartOrContinueOnPrep' 3 months after it is run.
    """

    def __init__(self, module, person_id, type_of_prep):
        super().__init__(module, person_id=person_id)
        self.type_of_prep = type_of_prep

    def apply(self, person_id):
        df = self.sim.population.props
        p = self.module.parameters
        person = df.loc[person_id]
        m = self.module
        property = 'hv_is_on_prep_oral' if self.type_of_prep == 'oral' else 'hv_is_on_prep_inj'

        # If the person is no longer alive or has been diagnosed with hiv, they will not continue on PrEP
        if (
            (not person["is_alive"])
            or (person["hv_diagnosed"])
        ):
            return

        # Check that they are on PrEP currently:
        if not person[property]:
            logger.warning(
                key="message",
                data="This event should not be running: Hiv_DecisionToContinueOnPrEP is for those currently on prep")

        # check still eligible, person must be <25 years old or a fsw
        if (person["age_years"] < p['age_max_prep']) or person["li_is_sexworker"]:

            # Determine if this appointment is actually attended by the person who has already started on PrEP
            if (
                m.rng.random_sample()
                < m.parameters["probability_of_being_retained_on_prep_every_3_months"]
            ):
                # Continue on PrEP - and schedule an HSI for a refill appointment today
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    HSI_Hiv_StartOrContinueOnPrep(person_id=person_id, module=m, type_of_prep=self.type_of_prep),
                    topen=self.sim.date,
                    tclose=self.sim.date + pd.DateOffset(days=p["hiv_prep_refill_appointment_window_days"]),
                    priority=0,
                )

            else:
                # Defaults to being off PrEP - reset flag and take no further action
                df.at[person_id, property] = False


class Hiv_DecisionToContinueTreatment(Event, IndividualScopeEventMixin):
    """Helper event that is used to 'decide' if someone on Treatment should continue on Treatment.
    This event is scheduled by 'HSI_Hiv_StartOrContinueTreatment' 3 months after it is run.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        p = self.module.parameters
        person = df.loc[person_id]
        m = self.module

        if not person["is_alive"]:
            return

        # Check that they are on Treatment currently:
        if person["hv_art"] not in ["on_VL_suppressed", "on_not_VL_suppressed"]:
            logger.warning(
                key="message",
                data="This event should not be running, Hiv_DecisionToContinueTreatment is for those already on tx")

        # Determine if person who has already started on ART will return for next appointment
        length_of_current_dispensation = person["hv_arv_dispensing_interval"]

        # Convert 3-month retention probability to equivalent retention over a different number of months,
        # assuming constant hazard (i.e.exponential decay)
        prob_retention = m.parameters["probability_of_being_retained_on_art_every_3_months"] ** (
                length_of_current_dispensation / 3)

        if (
            m.rng.random_sample()
            < prob_retention
        ):
            # Continue on Treatment - and schedule an HSI for a continuation appointment today
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Hiv_StartOrContinueTreatment(person_id=person_id, module=m,
                                                 facility_level_of_this_hsi="1a"),
                topen=self.sim.date,
                tclose=self.sim.date + pd.DateOffset(days=p["art_continuation_appointment_window_days"]),
                priority=0,
            )

        else:
            # Defaults to being off Treatment
            m.stops_treatment(person_id)

            # decide whether long-term LTFU or return within 3 months
            # change return to 1 month
            if m.rng.random_sample() < p['proportion_on_treatment_not_lost_to_follow_up']:

                # refer for another treatment again in 3 months
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    HSI_Hiv_StartOrContinueTreatment(person_id=person_id, module=m,
                                                     facility_level_of_this_hsi="1a"),
                    topen=self.sim.date + pd.DateOffset(months=3),
                    tclose=None,
                    priority=0,
                )


class Hiv_AdherenceCounselling(Event, IndividualScopeEventMixin):
    """Helper event that is used to 'decide' if someone on Treatment who is not currently
    virally suppressed will become suppressed
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        person = df.loc[person_id]

        if not person["is_alive"]:
            return

        df.at[person_id, "hv_art"] = self.module.update_viral_suppression_status()


class HivScaleUpEvent(Event, PopulationScopeEventMixin):
    """ This event exists to change parameters or functions
    depending on the scenario for projections which has been set
    It only occurs once on date: scaleup_start_date,
    called by initialise_simulation
    """

    def __init__(self, module):
        super().__init__(module)

    def apply(self, population):

        self.module.update_parameters_for_program_change()


# ---------------------------------------------------------------------------
#   Health System Interactions (HSI)
# ---------------------------------------------------------------------------

class HSI_Hiv_SelfTest(HSI_Event, IndividualScopeEventMixin):
    """
    This is the HIV Self-test HSI.
    No consumables currently logged for HIV self-tests so counter can record numbers of tests given

    Self-tests are available from 2023 onwards and can be switched off via parameter selfttest_available
    Self-testing is scheduled via the regular poll and a positive result prompts a referral for a
    confirmatory test at a facility
    """

    def __init__(
        self, module, person_id,
    ):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Hiv)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Hiv_Test_Selftest"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"ConWithDCSA": 1})
        self.ACCEPTED_FACILITY_LEVEL = "0"

    def apply(self, person_id, squeeze_factor):
        """Do the testing and referring to other services"""

        df = self.sim.population.props
        person = df.loc[person_id]
        p = self.module.parameters

        if not person["is_alive"]:
            return

        self.module.stored_selftest_numbers += 1

        # refer for confirmatory test if positive
        if person["hv_inf"] and p["selftest_sensitivity"] and p["linked_to_care_after_selftest"]:
            delay_days = round(
                int(
                    self.module.rng.weibull(p["delay_from_selftest_to_facility_weibull_shape"])
                    * p["delay_from_selftest_to_facility_weibull_scale"]
                )
            )

            self.sim.modules["HealthSystem"].schedule_hsi_event(
                hsi_event=HSI_Hiv_TestAndRefer(
                    person_id=person_id, module=self.module, referred_from="selftest"),
                topen=self.sim.date + pd.DateOffset(days=delay_days),
                tclose=None,
                priority=0,
            )


class HSI_Hiv_TestAndRefer(HSI_Event, IndividualScopeEventMixin):
    """
    This is the Test-and-Refer HSI. Individuals may seek an HIV test at any time. From this, they can be referred on to
    other services.
    This event is scheduled by:
        * the main event poll,
        * when someone presents for any care through a Generic HSI.
        * when an infant is born to an HIV-positive mother
    Following the test, they may or may not go on to present for uptake an HIV service: ART (if HIV-positive), VMMC (if
    HIV-negative and male) or PrEP (if HIV-negative and a female sex worker).
    If this event is called within another HSI, it may be desirable to limit the functionality of the HSI: do this
    using the arguments:
        * do_not_refer_if_neg=False : if the person is HIV-neg they will not be referred to VMMC or PrEP
        * suppress_footprint=True : the HSI will not have any footprint
    """

    def __init__(
        self, module, person_id, do_not_refer_if_neg=False, suppress_footprint=False, referred_from=None,
    ):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Hiv)

        assert isinstance(do_not_refer_if_neg, bool)
        self.do_not_refer_if_neg = do_not_refer_if_neg

        assert isinstance(suppress_footprint, bool)
        self.suppress_footprint = suppress_footprint

        self.referred_from = referred_from

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Hiv_Test"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"VCTNegative": 1})
        self.ACCEPTED_FACILITY_LEVEL = "1a"
        self.counter_for_test_not_available = 0

    def apply(self, person_id, squeeze_factor):
        """Do the testing and referring to other services"""

        df = self.sim.population.props
        p = self.module.parameters
        person = df.loc[person_id]

        if not person["is_alive"]:
            return

        # If person is diagnosed and on treatment do nothing do not occupy any resources
        if person["hv_diagnosed"] and (person["hv_art"] != "not"):
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        # if person has had test in last week, do not repeat test
        if person["hv_last_test_date"] >= (self.sim.date - DateOffset(days=p['hv_min_test_interval_days'])):
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        # Run test
        if person["age_years"] < p['age_max_hiv_early_infant_test_years']:
            test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                dx_tests_to_run="hiv_early_infant_test", hsi_event=self
            )
        else:
            test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                dx_tests_to_run="hiv_rapid_test", hsi_event=self
            )

        if test_result is not None:

            # Update number of tests:
            df.at[person_id, "hv_number_tests"] += 1
            df.at[person_id, "hv_last_test_date"] = self.sim.date

            # Log the test: line-list of summary information about each test
            person_details_for_test = {
                'age': person['age_years'],
                'hiv_status': person['hv_inf'],
                'hiv_diagnosed': person['hv_diagnosed'],
                'referred_from': self.referred_from,
                'person_id': person_id
            }
            logger.info(key='hiv_test', data=person_details_for_test)

            # Offer services as needed:
            if test_result:
                # The test_result is HIV positive
                ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint({"VCTPositive": 1})

                # Update diagnosis if the person is indeed HIV positive;
                if person["hv_inf"]:
                    df.at[person_id, "hv_diagnosed"] = True
                    self.module.do_when_hiv_diagnosed(person_id=person_id)

            else:
                # The test_result is HIV negative
                ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint({"VCTNegative": 1})

                if not self.do_not_refer_if_neg:
                    # The test was negative: make referrals to other services:

                    # Consider if the person's risk will be reduced by behaviour change counselling
                    if self.module.lm["lm_behavchg"].predict(
                        df.loc[[person_id]], self.module.rng
                    ):
                        df.at[person_id, "hv_behaviour_change"] = True

                    # If person is a man, and not circumcised, then consider referring to VMMC
                    if (person["sex"] == "M") and (not person["li_is_circ"]):
                        x = self.module.lm["lm_circ"].predict(
                            df.loc[[person_id]], self.module.rng,
                            year=self.sim.date.year,
                        )
                        if x:
                            self.sim.modules["HealthSystem"].schedule_hsi_event(
                                HSI_Hiv_Circ(person_id=person_id, module=self.module),
                                topen=self.sim.date,
                                tclose=None,
                                priority=0,
                            )

                    # If person is a woman and FSW, and not currently on PrEP then consider referring to PrEP
                    # numbers available 2018 onwards
                    if (
                        (person["sex"] == "F")
                        and person["li_is_sexworker"]
                        and not (person["hv_is_on_prep_oral"] or person["hv_is_on_prep_inj"])
                        and (self.sim.date.year >= self.module.parameters["prep_start_year"])
                    ):
                        if self.module.lm["lm_prep"].predict(df.loc[[person_id]], self.module.rng):
                            if (
                                self.module.parameters["injectable_prep_allowed"]
                                and (self.sim.date.year >= 2025)
                            ):
                                prob_injectable = self.module.parameters["prob_injectable_prep_vs_oral"]

                                type_of_prep = self.module.rng.choice(
                                    ["injectable", "oral"],
                                    p=[prob_injectable, 1 - prob_injectable],
                                )
                            else:
                                type_of_prep = "oral"

                            self.sim.modules["HealthSystem"].schedule_hsi_event(
                                HSI_Hiv_StartOrContinueOnPrep(
                                    person_id=person_id,
                                    module=self.module,
                                    type_of_prep=type_of_prep,
                                ),
                                topen=self.sim.date,
                                tclose=None,
                                priority=0,
                            )

        else:
            # Test was not possible, set blank footprint and schedule another test
            ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint({})

            # set cap for number of repeat tests
            self.counter_for_test_not_available += 1  # The current appointment is included in the count.

            if (
                self.counter_for_test_not_available
                <= self.module.parameters["hiv_healthseekingbehaviour_cap"]
            ):
                # repeat appt for HIV test
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    self,
                    topen=self.sim.date + pd.DateOffset(days=p['hv_min_test_interval_days']),
                    tclose=None,
                    priority=0,
                )

        # Return the footprint. If it should be suppressed, return a blank footprint.
        if self.suppress_footprint:
            return self.make_appt_footprint({})
        else:
            return ACTUAL_APPT_FOOTPRINT


class HSI_Hiv_Circ(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "Hiv_Prevention_Circumcision"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"MaleCirc": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'
        self.number_of_occurrences = 0

    def apply(self, person_id, squeeze_factor):
        """ Do the circumcision for this man. If he is already circumcised, this is a follow-up appointment."""
        self.number_of_occurrences += 1  # The current appointment is included in the count.
        df = self.sim.population.props  # shortcut to the dataframe
        p = self.module.parameters

        person = df.loc[person_id]

        # Do not run if the person is not alive
        if not person["is_alive"]:
            return

        # get confirmatory test
        test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
            dx_tests_to_run="hiv_rapid_test", hsi_event=self
        )
        if test_result is not None:
            df.at[person_id, "hv_number_tests"] += 1
            df.at[person_id, "hv_last_test_date"] = self.sim.date

        # if person not circumcised, perform the procedure
        if not person["li_is_circ"]:
            # Check/log use of consumables, if materials available, do circumcision and schedule follow-up appts
            # If materials not available, repeat the HSI, i.e., first appt.
            if self.get_consumables(item_codes=self.module.item_codes_for_consumables_required['circ']):
                # Update circumcision state
                df.at[person_id, "li_is_circ"] = True
                # record for logger if VMMC performed
                df.at[person_id, "hv_VMMC_in_last_year"] = True

                # Add used equipment
                self.add_equipment({'Drip stand', 'Stool, adjustable height', 'Autoclave',
                                    'Bipolar Diathermy Machine', 'Bed, adult', 'Trolley, patient'})

                # Schedule follow-up appts
                # schedule first follow-up appt, 3 days from procedure;
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    self,
                    topen=self.sim.date + DateOffset(days=p['hiv_circ_procedure_first_followup_days']),
                    tclose=None,
                    priority=0,
                )
                # schedule second follow-up appt, 7 days from procedure;
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    self,
                    topen=self.sim.date + DateOffset(days=p['hiv_circ_procedure_second_followup_days']),
                    tclose=None,
                    priority=0,
                )
            else:
                # schedule repeating appt when consumables not available
                if (
                    self.number_of_occurrences
                    <= self.module.parameters["hiv_healthseekingbehaviour_cap"]
                ):
                    self.sim.modules["HealthSystem"].schedule_hsi_event(
                        self,
                        topen=self.sim.date + DateOffset(weeks=p['hiv_circ_cons_notavailable_retry_procedure_weeks']),
                        tclose=None,
                        priority=0,
                    )


class HSI_Hiv_StartInfantProphylaxis(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id, referred_from, repeat_visits):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Hiv)

        self.TREATMENT_ID = "Hiv_Prevention_Infant"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Peds": 1, "VCTNegative": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.referred_from = referred_from
        self.repeat_visits = repeat_visits

    def apply(self, person_id, squeeze_factor):
        """
        Start infant prophylaxis for this infant lasting for duration of breastfeeding
        or up to 18 months
        """

        df = self.sim.population.props
        p = self.module.parameters
        person = df.loc[person_id]

        # Do not run if the child is not alive or is diagnosed with hiv
        if not person["is_alive"] or person["hv_diagnosed"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        # if breastfeeding has ceased or child >18 months, no further prophylaxis required
        if (df.at[person_id, "nb_breastfeeding_status"] == "none") \
                or (df.at[person_id, "age_years"] >= p['age_max_prophylaxis_years']):
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        # Check that infant prophylaxis is available and if it is, initiate:
        if self.get_consumables(
            item_codes={self.module.item_codes_for_consumables_required['infant_prep']: 63}
        ):
            df.at[person_id, "hv_is_on_prep_oral"] = True

            # Schedule follow-up visit for 3 months time
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                hsi_event=HSI_Hiv_StartInfantProphylaxis(
                    person_id=person_id,
                    module=self.module,
                    referred_from="repeat3months",
                    repeat_visits=0),
                priority=1,
                topen=self.sim.date + DateOffset(months=p['infant_prophylaxis_followup_months']),
                tclose=None,
            )

        else:
            # infant does not get NVP now but has repeat visit scheduled up to 5 times
            df.at[person_id, "hv_is_on_prep_oral"] = False

            if (
                self.repeat_visits
                <= self.module.parameters["hiv_healthseekingbehaviour_cap"]
            ):
                self.repeat_visits += 1

                # Schedule repeat visit for one week's time
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    hsi_event=HSI_Hiv_StartInfantProphylaxis(
                        person_id=person_id,
                        module=self.module,
                        referred_from="repeatNoCons",
                        repeat_visits=self.repeat_visits),
                    priority=1,
                    topen=self.sim.date + pd.DateOffset(days=p['infant_prophylaxis_cons_notavailable_retry_days']),
                    tclose=None,
                )

    def never_ran(self, *args, **kwargs):
        """This is called if this HSI was never run.
        Default the person to being off PrEP"""

        self.sim.population.props.at[self.target, "hv_is_on_prep_oral"] = False


class HSI_Hiv_StartOrContinueOnPrep(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id, type_of_prep):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Hiv)

        self.TREATMENT_ID = "Hiv_Prevention_Prep"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"PharmDispensing": 1, "VCTNegative": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.counter_for_drugs_not_available = 0
        self.type_of_prep = type_of_prep

    def apply(self, person_id, squeeze_factor):
        """Start PrEP for this person; or continue them on PrEP for 3 more months"""

        df = self.sim.population.props
        p = self.module.parameters
        person = df.loc[person_id]
        property = 'hv_is_on_prep_oral' if self.type_of_prep == 'oral' else 'hv_is_on_prep_inj'
        days_on_prep_property = 'hv_days_on_oral_prep' if self.type_of_prep == 'oral' else 'hv_days_on_inj_prep'

        # Do not run if the person is not alive or is diagnosed with hiv
        if (
            (not person["is_alive"])
            or (person["hv_diagnosed"])
        ):
            return

        # Run an HIV test
        test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
            dx_tests_to_run="hiv_rapid_test", hsi_event=self
        )
        df.at[person_id, "hv_number_tests"] += 1
        df.at[person_id, "hv_last_test_date"] = self.sim.date

        # If test is positive, flag as diagnosed and refer to ART
        if test_result is True:
            # label as diagnosed
            df.at[person_id, "hv_diagnosed"] = True

            # Do actions for when a person has been diagnosed with HIV
            self.module.do_when_hiv_diagnosed(person_id=person_id)

            return self.make_appt_footprint({"Over5OPD": 1, "VCTPositive": 1})

        # HIV test is negative - check that PrEP is available and if it is, initiate or continue PrEP:
        # 90 days supply oral tablets, injection lasts 8 weeks
        else:
            days_on_prep = self.module.parameters[
                               'initial_dispensation_period_months'] * 90 if self.type_of_prep == 'oral' else 56
            if self.get_consumables(
                item_codes={self.module.item_codes_for_consumables_required['prep']: days_on_prep}
            ):
                df.at[person_id, property] = True

                if df.at[person_id, "li_is_sexworker"]:
                    df.at[person_id, f'{days_on_prep_property}_FSW'] += days_on_prep

                elif (
                    (df.at[person_id, "sex"] == "F")
                    and (p['age_min_agyw'] <= df.at[person_id, "age_years"] <= p['age_max_agyw'])
                ):
                    df.at[person_id, f'{days_on_prep_property}_AGYW'] += days_on_prep

                # Schedule 'decision about whether to continue on PrEP'
                self.sim.schedule_event(
                    Hiv_DecisionToContinueOnPrEP(person_id=person_id, module=self.module,
                                                 type_of_prep=self.type_of_prep),
                    self.sim.date + pd.DateOffset(days=days_on_prep),
                )

            else:
                # If PrEP is not available, the person will default and not be on PrEP
                df.at[person_id, property] = False

                self.counter_for_drugs_not_available += (
                    1  # The current appointment is included in the count.
                )

                if (
                    self.counter_for_drugs_not_available
                    <= self.module.parameters["hiv_healthseekingbehaviour_cap"]
                ):
                    # Schedule repeat visit for one week's time
                    self.sim.modules["HealthSystem"].schedule_hsi_event(
                        self,
                        priority=1,
                        topen=self.sim.date + pd.DateOffset(days=p['prep_cons_notavailable_retry_days']),
                        tclose=None,
                    )

    def never_ran(self):
        """This is called if this HSI was never run.
        Default the person to being off PrEP"""
        self.sim.population.props.at[self.target, property] = False


class HSI_Hiv_StartOrContinueTreatment(HSI_Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Hiv)

        self.TREATMENT_ID = "Hiv_Treatment"
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.counter_for_drugs_not_available = 0
        self.counter_for_did_not_run = 0

    def apply(self, person_id, squeeze_factor):
        """This is a Health System Interaction Event - start or continue HIV treatment for 6 more months"""

        df = self.sim.population.props
        person = df.loc[person_id]
        p = self.module.parameters
        art_status_at_beginning_of_hsi = person["hv_art"]
        self.dispensation_interval = p['initial_dispensation_period_months']  # default

        if not person["is_alive"]:
            return

        # Confirm that the person is diagnosed (this should not run if they are not)
        assert person["hv_diagnosed"]

        # check whether person had Rx at least 3 months ago and is now due repeat prescription
        # alternate routes into testing/tx may mean person already has recent ARV dispensation
        if pd.notna(person["hv_arv_dispensing_interval"]):
            if person["hv_date_last_ART"] > (
                self.sim.date
                - pd.DateOffset(months=person["hv_arv_dispensing_interval"])
            ):
                return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        # ------------------------- give ART ------------------------- #

        if art_status_at_beginning_of_hsi == "not":

            assert person[
                "hv_inf"
            ]  # after the test results, it can be guaranteed that the person is HIV-pos.

            # Try to initiate the person onto ART:
            drugs_were_available = self.do_at_initiation(person_id)
        else:
            # Try to continue the person on ART:
            drugs_were_available = self.do_at_continuation(person_id)

        # if ART is available (1st item in drugs_were_available dict)
        if drugs_were_available.get('art', False):
            df.at[person_id, 'hv_date_last_ART'] = self.sim.date

            # If person has been placed/continued on ART, schedule decision about whether to continue on Treatment
            self.sim.schedule_event(
                Hiv_DecisionToContinueTreatment(
                    person_id=person_id, module=self.module
                ),
                self.sim.date + pd.DateOffset(months=self.dispensation_interval),
            )

        else:
            # ------------------------- if ART not available ------------------------- #

            # logger for drugs not available
            # person_details_for_tx = {
            #     'age': person['age_years'],
            #     'facility_level': self.ACCEPTED_FACILITY_LEVEL,
            #     'number_appts': self.counter_for_drugs_not_available,
            #     'district': person['district_of_residence'],
            #     'person_id': person_id,
            #     'drugs_available': drugs_were_available,
            # }
            # logger.info(key='hiv_arv_NA', data=person_details_for_tx)

            # As drugs were not available, the person will default to being off ART (...if they were on ART at the
            # beginning of the HSI.)
            # NB. If the person was not on ART at the beginning of the HSI, then there is no need to stop them (which
            #  causes a new AIDSOnsetEvent to be scheduled.)
            self.counter_for_drugs_not_available += 1  # The current appointment is included in the count.

            if art_status_at_beginning_of_hsi != "not":
                self.module.stops_treatment(person_id)

            if (self.module.rng.random_sample() <=
                    p['probability_of_seeking_further_art_appointment_if_drug_not_available']):

                # add in referral straight back to tx
                # if defaulting, seek another treatment appointment in 6 months
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    hsi_event=HSI_Hiv_StartOrContinueTreatment(
                        person_id=person_id,
                        module=self.module,
                        facility_level_of_this_hsi="1a",
                    ),
                    topen=self.sim.date + pd.DateOffset(months=p['art_drug_notavailable_default_retry_months']),
                    priority=0,
                )

            else:
                # If person 'decides to' seek another treatment appointment,
                # schedule a new HSI appointment for next month
                # NB. With a probability of 1.0, this will keep occurring,
                # if person has already tried unsuccessfully to get ART at level 1a 2 times
                #  then refer to level 1b
                if self.counter_for_drugs_not_available <= p['art_level1a_healthseekingbehaviour_cap']:
                    # repeat attempt for ARVs at level 1a
                    self.sim.modules["HealthSystem"].schedule_hsi_event(
                        self,
                        topen=self.sim.date + pd.DateOffset(months=p['art_level1a_appointment_delay_months']),
                        priority=0,
                    )

                else:
                    # refer to higher facility level
                    self.sim.modules["HealthSystem"].schedule_hsi_event(
                        hsi_event=HSI_Hiv_StartOrContinueTreatment(
                            person_id=person_id,
                            module=self.module,
                            facility_level_of_this_hsi="2",
                        ),
                        topen=self.sim.date + pd.DateOffset(days=p['art_level1b_appointment_delay_days']),
                        priority=0,
                    )

        # also screen for tb
        if "Tb" in self.sim.modules and not person["tb_diagnosed"]:
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                tb.HSI_Tb_ScreeningAndRefer(
                    person_id=person_id, module=self.sim.modules["Tb"]
                ),
                topen=self.sim.date,
                tclose=None,
                priority=0,
            )

    def do_at_initiation(self, person_id):
        """Things to do when this the first appointment ART"""
        df = self.sim.population.props
        person = df.loc[person_id]

        # assign dispensation period of 1 month for first prescription
        self.dispensation_interval = 1

        # Check if drugs are available, and provide drugs
        # this will return a dict where the first item is ART and the second is cotrimoxazole
        drugs_available = self.get_drugs(age_of_person=person["age_years"],
                                         dispensation_interval=self.dispensation_interval)

        # ART is first item in drugs_available dict
        if drugs_available.get('art', False):

            # get confirmatory test
            test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                dx_tests_to_run="hiv_rapid_test", hsi_event=self
            )
            if test_result is not None:
                df.at[person_id, "hv_number_tests"] += 1
                df.at[person_id, "hv_last_test_date"] = self.sim.date

            # Assign person to be suppressed or un-suppressed viral load
            # (If person is VL suppressed This will prevent the Onset of AIDS, or an AIDS death if AIDS has already
            # onset)
            vl_status = self.determine_vl_status(
                age_of_person=person["age_years"], df=df
            )

            df.at[person_id, "hv_art"] = vl_status
            df.at[person_id, "hv_date_treated"] = self.sim.date
            df.at[person_id, "hv_arv_dispensing_interval"] = self.dispensation_interval

            # If VL suppressed, remove any symptoms caused by this module
            if vl_status == "on_VL_suppressed":
                self.sim.modules["SymptomManager"].clear_symptoms(
                    person_id=person_id, disease_module=self.module
                )

        # if cotrimoxazole is available
        if drugs_available.get('cotrim', False):
            df.at[person_id, "hv_on_cotrimoxazole"] = True

        # Consider if TB preventive therapy should start
        self.consider_tb(person_id)

        return drugs_available

    def do_at_continuation(self, person_id):
        """Things to do when the person is already on ART"""

        df = self.sim.population.props
        person = df.loc[person_id]
        p = self.module.parameters
        is_suppressed = person["hv_art"] == "on_VL_suppressed"  # returns true if suppressed

        # default to person stopping cotrimoxazole
        df.at[person_id, "hv_on_cotrimoxazole"] = False

        # numbers look up dispensation length, DSD from 2021 onwards, else stick with default
        currently_breastfeeding = (
            (df["mother_id"] == person_id) & (df["nb_breastfeeding_status"] != "none")
        ).any()

        if self.sim.date.year >= 2021:
            self.dispensation_interval = self.module.get_art_dispensation_length(
                year=self.sim.date.year,
                person=person,
                currently_breastfeeding=currently_breastfeeding)

        # Check if drugs are available and provide drugs:
        drugs_available = self.get_drugs(age_of_person=person["age_years"],
                                         dispensation_interval=self.dispensation_interval)

        if not drugs_available.get('art', False):
            # if ART not available for full dispensation length, check if one month available
            self.dispensation_interval = 1
            drugs_available = self.get_drugs(age_of_person=person["age_years"],
                                             dispensation_interval=self.dispensation_interval)

        # if cotrimoxazole is available, update person's property
        if drugs_available.get('cotrim', False):
            df.at[person_id, "hv_on_cotrimoxazole"] = True

        # store new dispensation length for person
        df.at[person_id, 'hv_arv_dispensing_interval'] = self.dispensation_interval

        # Viral Load Monitoring (or TDF test if switching)
        if drugs_available.get('art', False) and self.module.vl_testing_available_by_year[self.sim.date.year]:

            test_probability, test_type = self.decide_test_policy(person_id)

            if self.module.rng.random_sample() < test_probability:

                if test_type == 'TDF':
                    tdf_result = self.module.perform_tdf_test(is_suppressed=is_suppressed)

                    if not tdf_result and person["hv_art"] == "on_not_VL_suppressed":
                        # schedule Adherence Counselling - no delay
                        self.sim.schedule_event(
                            Hiv_AdherenceCounselling(person_id=person_id, module=self.module),
                            self.sim.date,
                        )
                    self.module.stored_tdf_numbers += 1

                elif test_type == 'VL' and self.get_consumables(
                        self.module.item_codes_for_consumables_required['vl_measurement']):

                    # logic around unsuppressed person receiving and acting on result
                    if person["hv_art"] == "on_not_VL_suppressed":

                        q = p["prob_receive_viral_load_test_result"] * p["sensitivity_viral_load_test"]

                        if self.module.rng.random_sample() < q:

                            delay_months = self.module.rng.randint(3, 7)

                            self.sim.schedule_event(
                                Hiv_AdherenceCounselling(person_id=person_id, module=self.module),
                                self.sim.date + pd.DateOffset(months=delay_months),
                            )

                    logger.info(
                        key='hiv_VLtest',
                        data={
                            'adult': person['age_years'] >= 15,
                            'person_id': person_id
                        }
                    )

        return drugs_available

    def determine_vl_status(self, age_of_person, df):
        """Helper function to determine the VL status that the person will have.
        Return what will be the status of "hv_art"
        """
        prob_vs = self.module.prob_viral_suppression(df=df, year=self.sim.date.year, age_of_person=age_of_person)

        return (
            "on_VL_suppressed"
            if (self.module.rng.random_sample() < prob_vs)
            else "on_not_VL_suppressed"
        )

    def decide_test_policy(self, person_id):
        """
        Decide (a) the probability a person is offered a VL test this encounter,
        and (b) which test ('VL', 'TDF').
        """

        df = self.sim.population.props
        person = df.loc[person_id]
        p = self.module.parameters

        # 1) Choose test type
        test_type = 'TDF' if p['tdf_test_replace_vl_test'] else 'VL'

        # 2) Base probability from dispensing / VL interval
        test_prob = person['hv_arv_dispensing_interval'] / p['interval_for_viral_load_measurement_months']
        # Clamp probability to [0,1]
        test_prob = max(0.0, min(1.0, test_prob))

        if p['targeted_adherence_monitoring']:
            if person['age_years'] < 30 or person['is_pregnant']:
                test_prob = test_prob
            else:
                test_prob = 0

        return test_prob, test_type

    def get_drugs(self, age_of_person, dispensation_interval):
        """Helper function to get the ART according to the age of the person being treated. Returns dict to indicate
        whether individual drugs were available"""

        p = self.module.parameters
        dispensation_days = 30 * dispensation_interval

        if age_of_person < p["ART_age_cutoff_young_child"]:
            # Formulation for young children
            drugs_available = self.get_consumables(
                item_codes={
                    self.module.item_codes_for_consumables_required[
                        "First line ART regimen: young child"
                    ]: dispensation_days * 2,
                },
                optional_item_codes={
                    self.module.item_codes_for_consumables_required[
                        "First line ART regimen: young child: cotrimoxazole"
                    ]: dispensation_days * 240,
                },
                return_individual_results=True,
            )

        elif age_of_person <= p["ART_age_cutoff_older_child"]:
            # Formulation for older children
            drugs_available = self.get_consumables(
                item_codes={
                    self.module.item_codes_for_consumables_required[
                                'First line ART regimen: older child'
                    ]: dispensation_days * 3
                },
                optional_item_codes={
                    self.module.item_codes_for_consumables_required[
                        'First line ART regimen: older child: cotrimoxazole'
                    ]: dispensation_days * 480
                },
                return_individual_results=True)

        else:
            # Formulation for adults
            drugs_available = self.get_consumables(
                item_codes={
                    self.module.item_codes_for_consumables_required[
                                'First-line ART regimen: adult'
                    ]: dispensation_days
                },
                optional_item_codes={
                    self.module.item_codes_for_consumables_required[
                                         'First-line ART regimen: adult: cotrimoxazole'
                    ]: dispensation_days * 960
                },
                return_individual_results=True)

        # add drug names to dict
        drugs_available = {
            'art': list(drugs_available.values())[0],
            'cotrim': list(drugs_available.values())[1]
        }

        return drugs_available

    def consider_tb(self, person_id):
        """
        screen for tb
        Consider whether IPT is needed at this time. This is run only when treatment is initiated.
        """

        if "Tb" in self.sim.modules:
            self.sim.modules["Tb"].consider_ipt_for_those_initiating_art(
                person_id=person_id
            )

    def never_ran(self):
        """This is called if this HSI was never run.
        * Default the person to being off ART.
        * Determine if they will re-seek care themselves in the future:
        """
        # stop treatment for this person
        person_id = self.target
        self.module.stops_treatment(person_id)
        p = self.module.parameters

        # sample whether person will seek further appt
        if self.module.rng.random_sample() < self.module.parameters[
            "probability_of_seeking_further_art_appointment_if_appointment_not_available"
        ]:
            # schedule HSI
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Hiv_StartOrContinueTreatment(
                    person_id=person_id, module=self.module, facility_level_of_this_hsi="1a"
                ),
                topen=self.sim.date + pd.DateOffset(
                    days=p['seek_further_art_appointment_if_appointment_not_available_min_delay_days']
                ),
                tclose=self.sim.date + pd.DateOffset(
                    days=p['seek_further_art_appointment_if_appointment_not_available_max_delay_days']
                ),
                priority=1,
            )

    @property
    def EXPECTED_APPT_FOOTPRINT(self):
        """Returns the appointment footprint for this person according to their current status:
         * `NewAdult` for an adult, newly starting treatment
         * `EstNonCom` for an adult, re-starting treatment or already on treatment
         (NB. This is an appointment type that assumes that the patient does not have complications.)
         * `Peds` for a child - whether newly starting or already on treatment
        """
        person_id = self.target
        p = self.module.parameters

        if self.sim.population.props.at[person_id, 'age_years'] < p['age_threshold_child_adult_distinction']:
            return self.make_appt_footprint({"Peds": 1})  # Child

        if (self.sim.population.props.at[person_id, 'hv_art'] == "not") & (
            pd.isna(self.sim.population.props.at[person_id, 'hv_date_treated'])
        ):
            return self.make_appt_footprint({"NewAdult": 1})  # Adult newly starting treatment
        else:
            return self.make_appt_footprint({"EstNonCom": 1})  # Adult already on treatment


class HSI_Hiv_EndOfLifeCare(HSI_Event, IndividualScopeEventMixin):
    """
    this is a hospital stay for terminally-ill patients with AHD
    it does not affect disability weight or probability of death
    no consumables are logged but health system capacity (HR) is allocated
    there are no consequences if hospital bed is not available as person has scheduled death
    already within 2 weeks
    """

    def __init__(self, module, person_id, beddays=None):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Hiv)

        self.TREATMENT_ID = "Hiv_PalliativeCare"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
        self.ACCEPTED_FACILITY_LEVEL = "2"
        if beddays is None:
            beddays = module.parameters['aids_death_beddays_default']
        self.beddays = beddays
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({"general_bed": self.beddays})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, "is_alive"]:
            return hs.get_blank_appt_footprint()

        if df.at[person_id, "hv_art"] == "virally_suppressed":
            return hs.get_blank_appt_footprint()

        logger.debug(
            key="message",
            data=f"HSI_Hiv_EndOfLifeCare: inpatient admission for {person_id}",
        )


# ---------------------------------------------------------------------------
#   Logging
# ---------------------------------------------------------------------------


class HivLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """ Log Current status of the population, every year
        """

        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props
        now = self.sim.date

        # ------------------------------------ SUMMARIES ------------------------------------
        # population
        pop_male_15plus = len(
            df[df.is_alive & (df.age_years >= 15) & (df.sex == "M")]
        )
        pop_female_15plus = len(
            df[df.is_alive & (df.age_years >= 15) & (df.sex == "F")]
        )

        # plhiv
        male_plhiv = len(
            df[df.hv_inf & df.is_alive & (df.age_years >= 15) & (df.sex == "M")]
        )
        female_plhiv = len(
            df[df.hv_inf & df.is_alive & (df.age_years >= 15) & (df.sex == "F")]
        )
        child_plhiv = len(df[df.hv_inf & df.is_alive & (df.age_years < 15)])
        total_plhiv = len(df[df.hv_inf & df.is_alive])

        # adult prevalence
        adult_prev_15plus = len(
            df[df.hv_inf & df.is_alive & (df.age_years >= 15)]
        ) / len(df[df.is_alive & (df.age_years >= 15)]) if len(df[df.is_alive & (df.age_years >= 15)]) else 0

        adult_prev_1549 = len(
            df[df.hv_inf & df.is_alive & df.age_years.between(15, 49)]
        ) / len(df[df.is_alive & df.age_years.between(15, 49)]) if len(
            df[df.is_alive & df.age_years.between(15, 49)]) else 0

        # child prevalence
        child_prev = len(
            df[df.hv_inf & df.is_alive & (df.age_years < 15)]
        ) / len(df[df.is_alive & (df.age_years < 15)]
                ) if len(df[df.is_alive & (df.age_years < 15)]) else 0

        # incidence in the period since the last log for 15+ and 15-49 year-olds (denominator is approximate)
        n_new_infections_adult_15plus = len(
            df.loc[
                (df.age_years >= 15)
                & df.is_alive
                & (df.hv_date_inf >= (now - DateOffset(months=self.repeat)))
                ]
        )
        denom_adults_15plus = len(df[df.is_alive & (df.age_years >= 15)])
        adult_inc_15plus = n_new_infections_adult_15plus / denom_adults_15plus if denom_adults_15plus else 0

        n_new_infections_adult_1549 = len(
            df.loc[
                df.age_years.between(15, 49)
                & df.is_alive
                & (df.hv_date_inf >= (now - DateOffset(months=self.repeat)))
                ]
        )
        denom_adults_1549 = len(df[df.is_alive & df.age_years.between(15, 49)])
        adult_inc_1549 = n_new_infections_adult_1549 / denom_adults_1549 if denom_adults_1549 else 0

        # incidence in the period since the last log for 0-14 year-olds (denominator is approximate)
        n_new_infections_children = len(
            df.loc[
                (df.age_years < 15)
                & df.is_alive
                & (df.hv_date_inf >= (now - DateOffset(months=self.repeat)))
                ]
        )
        denom_children = len(df[df.is_alive & (df.age_years < 15)])
        child_inc = n_new_infections_children / denom_children if denom_children else 0

        # hiv prev among female sex workers (aged 15-49)
        n_fsw = len(
            df.loc[
                df.is_alive
                & df.li_is_sexworker
                & (df.sex == "F")
                & df.age_years.between(15, 49)
                ]
        )
        prev_hiv_fsw = (
            0
            if n_fsw == 0
            else len(
                df.loc[
                    df.is_alive
                    & df.hv_inf
                    & df.li_is_sexworker
                    & (df.sex == "F")
                    & df.age_years.between(15, 49)
                    ]
            ) / n_fsw
        )

        total_population = len(df.loc[df.is_alive])

        logger.info(
            key="summary_inc_and_prev_for_adults_and_children_and_fsw",
            description="Summary of HIV among adult (15+ and 15-49) and children (0-14s) and female sex workers"
                        " (15-49)",
            data={
                "pop_male_15plus": pop_male_15plus,
                "pop_female_15plus": pop_female_15plus,
                "pop_child": denom_children,
                "pop_total": total_population,
                "male_plhiv_15plus": male_plhiv,
                "female_plhiv_15plus": female_plhiv,
                "child_plhiv": child_plhiv,
                "total_plhiv": total_plhiv,
                "hiv_prev_adult_15plus": adult_prev_15plus,
                "hiv_prev_adult_1549": adult_prev_1549,
                "hiv_prev_child": child_prev,
                "hiv_adult_inc_15plus": adult_inc_15plus,
                "n_new_infections_adult_1549": n_new_infections_adult_1549,
                "hiv_adult_inc_1549": adult_inc_1549,
                "hiv_child_inc": child_inc,
                "hiv_prev_fsw": prev_hiv_fsw,
            },
        )

        # ------------------------------------ PREVALENCE BY AGE and SEX  ------------------------------------

        # Prevalence by Age/Sex (to make every category be output, do separately by 'sex')
        prev_by_age_and_sex = {}
        for sex in ["F", "M"]:
            n_hiv = df.loc[df.sex == sex].groupby(by=["age_range"])["hv_inf"].sum()
            n_pop = df.loc[df.sex == sex].groupby(by=["age_range"])["hv_inf"].count()
            prev_by_age_and_sex[sex] = (n_hiv / n_pop).to_dict()

        logger.info(
            key="prev_by_age_and_sex",
            data=prev_by_age_and_sex,
            description="Prevalence of HIV split by age and sex",
        )

        male_prev_1524 = len(
            df[df.hv_inf & df.is_alive & df.age_years.between(15, 24) & (df.sex == "M")]
        ) / len(df[df.is_alive & df.age_years.between(15, 24) & (df.sex == "M")]) if len(
            df[df.is_alive & df.age_years.between(15, 24) & (df.sex == "M")]) else 0

        male_prev_2549 = len(
            df[df.hv_inf & df.is_alive & df.age_years.between(25, 49) & (df.sex == "M")]
        ) / len(df[df.is_alive & df.age_years.between(25, 49) & (df.sex == "M")]) if len(
            df[df.is_alive & df.age_years.between(25, 49) & (df.sex == "M")]) else 0

        female_prev_1524 = len(
            df[df.hv_inf & df.is_alive & df.age_years.between(15, 24) & (df.sex == "F")]
        ) / len(df[df.is_alive & df.age_years.between(15, 24) & (df.sex == "F")]) if len(
            df[df.is_alive & df.age_years.between(15, 24) & (df.sex == "F")]) else 0

        female_prev_2549 = len(
            df[df.hv_inf & df.is_alive & df.age_years.between(25, 49) & (df.sex == "F")]
        ) / len(df[df.is_alive & df.age_years.between(25, 49) & (df.sex == "F")]) if len(
            df[df.is_alive & df.age_years.between(25, 49) & (df.sex == "F")]) else 0

        total_prev = len(
            df[df.hv_inf & df.is_alive]
        ) / len(df[df.is_alive])

        # incidence by age-group and sex
        n_new_infections_male_1524 = len(
            df.loc[
                df.age_years.between(15, 24)
                & (df.sex == "M")
                & df.is_alive
                & (df.hv_date_inf >= (now - DateOffset(months=self.repeat)))
                ]
        )
        denom_male_1524 = len(df[
                                  df.is_alive
                                  & (df.sex == "M")
                                  & df.age_years.between(15, 24)])
        male_inc_1524 = n_new_infections_male_1524 / denom_male_1524 if denom_male_1524 else 0

        n_new_infections_male_2549 = len(
            df.loc[
                df.age_years.between(25, 49)
                & (df.sex == "M")
                & df.is_alive
                & (df.hv_date_inf >= (now - DateOffset(months=self.repeat)))
                ]
        )
        denom_male_2549 = len(df[
                                  df.is_alive
                                  & (df.sex == "M")
                                  & df.age_years.between(25, 49)])
        male_inc_2549 = n_new_infections_male_2549 / denom_male_2549 if denom_male_2549 else 0

        n_new_infections_male_1549 = len(
            df.loc[
                df.age_years.between(15, 49)
                & (df.sex == "M")
                & df.is_alive
                & (df.hv_date_inf >= (now - DateOffset(months=self.repeat)))
                ]
        )

        n_new_infections_female_1524 = len(
            df.loc[
                df.age_years.between(15, 24)
                & (df.sex == "F")
                & df.is_alive
                & (df.hv_date_inf >= (now - DateOffset(months=self.repeat)))
                ]
        )
        denom_female_1524 = len(df[
                                    df.is_alive
                                    & (df.sex == "F")
                                    & df.age_years.between(15, 24)])
        female_inc_1524 = n_new_infections_female_1524 / denom_female_1524 if denom_female_1524 else 0

        n_new_infections_female_2549 = len(
            df.loc[
                df.age_years.between(25, 49)
                & (df.sex == "F")
                & df.is_alive
                & (df.hv_date_inf >= (now - DateOffset(months=self.repeat)))
                ]
        )
        denom_female_2549 = len(df[
                                    df.is_alive
                                    & (df.sex == "F")
                                    & df.age_years.between(25, 49)])
        female_inc_2549 = n_new_infections_female_2549 / denom_female_2549 if denom_female_2549 else 0

        logger.info(
            key="infections_by_2age_groups_and_sex",
            data={
                "male_prev_1524": male_prev_1524,
                "male_prev_2549": male_prev_2549,
                "female_prev_1524": female_prev_1524,
                "female_prev_2549": female_prev_2549,
                "total_prev": total_prev,
                "n_new_infections_male_1524": n_new_infections_male_1524,
                "n_new_infections_male_2549": n_new_infections_male_2549,
                "n_new_infections_male_1549": n_new_infections_male_1549,
                "n_new_infections_female_1524": n_new_infections_female_1524,
                "n_new_infections_female_2549": n_new_infections_female_2549,
                "male_inc_1524": male_inc_1524,
                "male_inc_2549": male_inc_2549,
                "female_inc_1524": female_inc_1524,
                "female_inc_2549": female_inc_2549,
            },
            description="HIV infections split by 2 age-groups age and sex",
        )
        logger.info(
            key="prev_by_age_and_sex",
            data=prev_by_age_and_sex,
            description="Prevalence of HIV split by age and sex",
        )

        # ------------------------------------ TESTING ------------------------------------
        # testing can happen through lm[spontaneous_testing] or symptom-driven or ANC or TB

        # proportion of adult population tested in past year
        n_tested = len(
            df.loc[
                df.is_alive
                & (df.hv_number_tests > 0)
                & (df.age_years >= 15)
                & (df.hv_last_test_date >= (now - DateOffset(months=self.repeat)))
                ]
        )
        n_pop = len(df.loc[df.is_alive & (df.age_years >= 15)])
        tested = n_tested / n_pop if n_pop else 0

        # proportion of adult population tested in past year by sex
        testing_by_sex = {}
        for sex in ["F", "M"]:
            n_tested = len(
                df.loc[
                    (df.sex == sex)
                    & (df.hv_number_tests > 0)
                    & (df.age_years >= 15)
                    & (df.hv_last_test_date >= (now - DateOffset(months=self.repeat)))
                    ]
            )
            n_pop = len(df.loc[(df.sex == sex) & (df.age_years >= 15)])
            testing_by_sex[sex] = n_tested / n_pop if n_pop else 0

        # per_capita_testing_rate: number of tests administered divided by population
        current_testing_rate = self.module.per_capita_testing_rate()

        # testing yield: number positive results divided by number tests performed
        # if person has multiple tests in one year, will only count 1
        total_tested = len(
            df.loc[(df.hv_number_tests > 0)
                   & (df.hv_last_test_date >= (now - DateOffset(months=self.repeat)))
                   ]
        )
        total_tested_hiv_positive = len(
            df.loc[df.hv_inf
                   & (df.hv_number_tests > 0)
                   & (df.hv_last_test_date >= (now - DateOffset(months=self.repeat)))
                   ]
        )
        testing_yield = total_tested_hiv_positive / total_tested if total_tested > 0 else 0

        # ------------------------------------ TREATMENT ------------------------------------
        def treatment_counts(subset):
            # total number of subset (subset is a true/false series)
            count = sum(subset)
            # proportion of subset living with HIV that are diagnosed:
            proportion_diagnosed = (
                sum(subset & df.hv_diagnosed) / count if count > 0 else 0.0
            )
            # proportions of subset living with HIV on treatment:
            art = sum(subset & (df.hv_art != "not"))
            art_cov = art / count if count > 0 else 0.0

            # proportion of subset on treatment that have good VL suppression
            art_vs = sum(subset & (df.hv_art == "on_VL_suppressed"))
            art_cov_vs = art_vs / art if art > 0 else 0.0
            return proportion_diagnosed, art_cov, art_cov_vs

        alive_infected = df.is_alive & df.hv_inf
        dx_adult, art_cov_adult, art_cov_vs_adult = treatment_counts(
            alive_infected & (df.age_years >= 15)
        )
        dx_children, art_cov_children, art_cov_vs_children = treatment_counts(
            alive_infected & (df.age_years < 15)
        )

        n_on_art_male_15plus = len(
            df.loc[
                (df.age_years >= 15)
                & (df.sex == "M")
                & df.is_alive
                & (df.hv_art != "not")
                ]
        )

        n_on_art_female_15plus = len(
            df.loc[
                (df.age_years >= 15)
                & (df.sex == "F")
                & df.is_alive
                & (df.hv_art != "not")
                ]
        )

        n_on_art_children = len(
            df.loc[
                (df.age_years < 15)
                & df.is_alive
                & (df.hv_art != "not")
                ]
        )

        n_on_art_total = n_on_art_male_15plus + n_on_art_female_15plus + n_on_art_children

        # ------------------------------------ BEHAVIOUR CHANGE ------------------------------------

        # proportion of adults (15+) exposed to behaviour change intervention
        prop_adults_exposed_to_behav_intv = len(
            df[df.is_alive & df.hv_behaviour_change & (df.age_years >= 15)]
        ) / len(df[df.is_alive & (df.age_years >= 15)]) if len(df[df.is_alive & (df.age_years >= 15)]) else 0

        # ------------------------------------ PREP AMONG FSW and AGYW ------------------------------------
        prop_fsw_on_prep = (
            0
            if n_fsw == 0
            else len(
                df[
                    df.is_alive
                    & df.li_is_sexworker
                    & (df.age_years >= 15)
                    & (df.hv_is_on_prep_oral | df.hv_is_on_prep_inj)
                    ]
            ) / len(df[df.is_alive & df.li_is_sexworker & (df.age_years >= 15)])
        ) if len(df[df.is_alive & df.li_is_sexworker & (df.age_years >= 15)]) else 0

        PY_PREP_ORAL_AGYW = df["hv_days_on_oral_prep_AGYW"].sum() / 365
        PY_PREP_ORAL_FSW = df["hv_days_on_oral_prep_FSW"].sum() / 365

        PY_PREP_INJ_AGYW = df["hv_days_on_inj_prep_AGYW"].sum() / 365
        PY_PREP_INJ_FSW = df["hv_days_on_inj_prep_FSW"].sum() / 365

        # ------------------------------------ MALE CIRCUMCISION ------------------------------------
        # NB. Among adult men
        num_men_circ = len(
            df[df.is_alive & (df.sex == "M") & (df.age_years >= 15) & df.li_is_circ]
        )

        prop_men_circ = len(
            df[df.is_alive & (df.sex == "M") & (df.age_years >= 15) & df.li_is_circ]
        ) / len(df[df.is_alive & (df.sex == "M") & (df.age_years >= 15)]) if len(
            df[df.is_alive & (df.sex == "M") & (df.age_years >= 15)]) else 0

        N_NewVMMC = len(df[df.hv_VMMC_in_last_year & (df.age_years >= 15)])

        logger.info(
            key="hiv_program_coverage",
            description="Coverage of interventions for HIV among adult (15+) and children (0-14s)",
            data={
                "number_adults_tested": n_tested,
                "prop_tested_adult": tested,
                "prop_tested_adult_male": testing_by_sex["M"],
                "prop_tested_adult_female": testing_by_sex["F"],
                "per_capita_testing_rate": current_testing_rate,
                "testing_yield": testing_yield,
                "dx_adult": dx_adult,
                "dx_childen": dx_children,
                "art_coverage_adult": art_cov_adult,
                "art_coverage_adult_VL_suppression": art_cov_vs_adult,
                "art_coverage_child": art_cov_children,
                "art_coverage_child_VL_suppression": art_cov_vs_children,
                "n_on_art_total": n_on_art_total,
                "n_on_art_male_15plus": n_on_art_male_15plus,
                "n_on_art_female_15plus": n_on_art_female_15plus,
                "n_on_art_children": n_on_art_children,
                "n_tdf_tests_performed": self.module.stored_tdf_numbers,
                "n_selftests_performed": self.module.stored_selftest_numbers,
                "prop_adults_exposed_to_behav_intv": prop_adults_exposed_to_behav_intv,
                "prop_fsw_on_prep": prop_fsw_on_prep,
                "PY_PREP_ORAL_AGYW": PY_PREP_ORAL_AGYW,
                "PY_PREP_ORAL_FSW": PY_PREP_ORAL_FSW,
                "PY_PREP_INJ_AGYW": PY_PREP_INJ_AGYW,
                "PY_PREP_INJ_FSW": PY_PREP_INJ_FSW,
                "prop_men_circ": prop_men_circ,
                "N_NewVMMC": N_NewVMMC,
                "Total_M_circ": num_men_circ,
            },
        )
        # after logger, reset yearly logged properties
        df["hv_VMMC_in_last_year"] = False
        df["hv_days_on_oral_prep_AGYW"] = 0
        df["hv_days_on_oral_prep_AGYW"] = 0
        df["hv_days_on_inj_prep_AGYW"] = 0
        df["hv_days_on_inj_prep_AGYW"] = 0
        self.module.stored_tdf_numbers = 0
        self.module.stored_selftest_numbers = 0

        # ------------------------------------ Multi-month dispensing ------------------------------------
        # Filter denominator: alive, HIV-infected, and on ART
        denominator = df[df.is_alive & df.hv_inf & (df.hv_art != 'not')].copy()

        # Identify breastfeeding mothers
        breastfeeding_mother_ids = (
            df.loc[df['nb_breastfeeding_status'] != 'none', 'mother_id']
            .dropna()
            .unique()
        )

        # Define group masks
        group_masks = {
            "child": denominator["age_years"] < 15,
            "pregnant_women": (
                (denominator["sex"] == "F")
                & (denominator["age_years"] >= 15)
                & (denominator["is_pregnant"])
            ),
            "breastfeeding_women": (
                (denominator["sex"] == "F")
                & (denominator["age_years"] >= 15)
                & denominator.index.isin(breastfeeding_mother_ids)
            ),
        }

        # Exclude pregnant/breastfeeding women for adult_female group
        excluded_ids = set(
            denominator.loc[group_masks['pregnant_women'] | group_masks['breastfeeding_women']].index
        )
        group_masks['adult_female'] = (
            (denominator['sex'] == 'F') & (denominator['age_years'] >= 15) &
            ~denominator.index.isin(excluded_ids)
        )
        group_masks['adult_male'] = (denominator['sex'] == 'M') & (denominator['age_years'] >= 15)

        # Assign group labels
        group_labels = pd.Series(index=denominator.index, dtype='object')
        for group, mask in group_masks.items():
            group_labels.loc[mask] = group
        group_labels = group_labels.dropna()

        # Filter and annotate the working dataframe
        grouped_df = (
            denominator.loc[group_labels.index]
            .assign(group=group_labels)
            .copy()
        )

        # Define dispensing interval bins and labels
        dispense_bins = [-float('inf'), 3, 6, float('inf')]
        dispense_labels = ['<3', '3-5', '6+']

        # Ensure 'interval' is a Categorical with all possible levels
        grouped_df['interval'] = pd.Categorical(
            pd.cut(
                grouped_df['hv_arv_dispensing_interval'],
                bins=dispense_bins,
                labels=dispense_labels,
                right=False
            ),
            categories=dispense_labels
        )

        # Define full group list to ensure consistent logging keys across time
        all_groups = ['child', 'pregnant_women', 'breastfeeding_women', 'adult_female', 'adult_male']
        full_index = pd.MultiIndex.from_product(
            [all_groups, dispense_labels],
            names=['group', 'interval']
        )

        # Crosstab with full index reindexing
        prop_series = (
            pd.crosstab(
                grouped_df['group'],
                grouped_df['interval'],
                normalize='index'
            )
            .reindex(columns=dispense_labels, fill_value=0)
            .stack()
            .reindex(full_index, fill_value=0)
        )

        # Log results
        logger.info(
            key="arv_dispensing_intervals",
            data=flatten_multi_index_series_into_dict_for_logging(prop_series),
            description="Proportion of people by ARV dispensing interval categories (<3, 3-5, 6+ months)"
        )


# ---------------------------------------------------------------------------
#   Debugging / Checking Events
# ---------------------------------------------------------------------------


class HivCheckPropertiesEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))  # runs every month

    def apply(self, population):
        self.module.check_config_of_properties()


# ---------------------------------------------------------------------------
#   Helper functions for analysing outputs
# ---------------------------------------------------------------------------


def set_age_group(ser):
    AGE_RANGE_CATEGORIES, AGE_RANGE_LOOKUP = create_age_range_lookup(
        min_age=demography.MIN_AGE_FOR_RANGE,
        max_age=demography.MAX_AGE_FOR_RANGE,
        range_size=demography.AGE_RANGE_SIZE,
    )
    ser = ser.astype("category")
    AGE_RANGE_CATEGORIES_filtered = [a for a in AGE_RANGE_CATEGORIES if a in ser.values]
    return ser.cat.reorder_categories(AGE_RANGE_CATEGORIES_filtered)


def map_to_age_group(ser):
    AGE_RANGE_CATEGORIES, AGE_RANGE_LOOKUP = create_age_range_lookup(
        min_age=demography.MIN_AGE_FOR_RANGE,
        max_age=demography.MAX_AGE_FOR_RANGE,
        range_size=demography.AGE_RANGE_SIZE,
    )
    ser = ser.map(AGE_RANGE_LOOKUP)
    ser = set_age_group(ser)
    return ser


def unpack_raw_output_dict(raw_dict):
    x = pd.DataFrame.from_dict(data=raw_dict, orient="index")
    x.reset_index(inplace=True)
    x.rename(columns={"index": "age_group", 0: "value"}, inplace=True)
    x["age_group"] = set_age_group(x["age_group"])
    return x


# ---------------------------------------------------------------------------
#   Dummy Version of the Module
# ---------------------------------------------------------------------------


class DummyHivModule(Module):
    """Dummy HIV Module - it's only job is to create and maintain the 'hv_inf' and 'hv_art' properties.
    This can be used in test files."""

    INIT_DEPENDENCIES = {"Demography"}
    ALTERNATIVE_TO = {"Hiv"}

    PROPERTIES = {
        "hv_inf": Property(
            Types.BOOL,
            "DUMMY version of the property for hv_inf"),
        "hv_art": Property(
            Types.CATEGORICAL,
            "DUMMY version of the property for hv_art.",
            categories=["not", "on_VL_suppressed", "on_not_VL_suppressed"]),
        "hv_diagnosed": Property(
            Types.BOOL,
            "DUMMY version of the property for hv_diagnosed.",
            categories=["not", "on_VL_suppressed", "on_not_VL_suppressed"]),
    }

    def __init__(self, name=None, hiv_prev=0.1, art_cov=0.75):
        super().__init__(name)
        self.hiv_prev = hiv_prev
        self.art_cov = art_cov

    def read_parameters(self, resourcefilepath: Optional[Path] = None):
        pass

    def initialise_population(self, population):
        df = population.props
        df.loc[df.is_alive, "hv_inf"] = self.rng.rand(sum(df.is_alive)) < self.hiv_prev
        df.loc[(df.is_alive & df.hv_inf), "hv_art"] = pd.Series(
            self.rng.rand(sum(df.is_alive & df.hv_inf)) < self.art_cov).replace(
            {True: "on_VL_suppressed", False: "not"}).values
        df.loc[(df.is_alive & df.hv_inf), "hv_diagnosed"] = (
            self.rng.random_sample(len(df.loc[(df.is_alive & df.hv_inf)])) < 0.5)

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother, child):
        df = self.sim.population.props
        df.at[child, "hv_inf"] = self.rng.rand() < self.hiv_prev

        if df.at[child, "hv_inf"]:
            df.at[child, "hv_art"] = "on_VL_suppressed" if self.rng.rand() < self.art_cov else "not"
            df.at[child, "hv_diagnosed"] = self.rng.rand() < 0.5

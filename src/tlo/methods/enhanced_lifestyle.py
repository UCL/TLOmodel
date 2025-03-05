"""
Lifestyle module
Documentation: 04 - Methods Repository/Method_Lifestyle.xlsx
"""
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from tlo import Date, DateOffset, Module, Parameter, Property, Types, logging
from tlo.analysis.utils import flatten_multi_index_series_into_dict_for_logging
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.logging.helpers import grouped_counts_with_all_combinations
from tlo.util import get_person_id_to_inherit_from, read_csv_files

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Lifestyle(Module):
    """
    Lifestyle module provides properties that are used by all disease modules if they are affected
    by urban/rural, wealth, tobacco usage etc.
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath: Path = resourcefilepath

        # a pointer to the linear models class
        self.models = None

    INIT_DEPENDENCIES = {'Demography'}

    # Declare Metadata
    METADATA = {}

    PARAMETERS = {
        # -------- list of parameters ---------------------------------------------------------------
        'init_p_urban': Parameter(Types.REAL, 'initial proportion urban'),
        'init_p_wealth_urban': Parameter(Types.LIST, 'List of probabilities of category given urban'),
        'init_p_wealth_rural': Parameter(Types.LIST, 'List of probabilities of category given rural'),
        'init_p_bmi_urban_m_not_high_sugar_age1529_not_tob_wealth1': Parameter(
            Types.LIST,
            'List of probabilities of  bmi categories '
            'for urban men age 15-29 with not high sugar, not tobacco, wealth level 1',
        ),
        'init_or_higher_bmi_f': Parameter(Types.REAL, 'odds ratio higher BMI if female'),
        'init_or_higher_bmi_rural': Parameter(Types.REAL, 'odds ratio higher BMI if rural'),
        'init_or_higher_bmi_high_sugar': Parameter(Types.REAL, 'odds ratio higher BMI if high sugar intake'),
        'init_or_higher_bmi_age3049': Parameter(Types.REAL, 'odds ratio higher BMI if age 30-49'),
        'init_or_higher_bmi_agege50': Parameter(Types.REAL, 'odds ratio higher BMI if age ge 50'),
        'init_or_higher_bmi_tob': Parameter(Types.REAL, 'odds ratio higher BMI if use tobacco'),
        'init_or_higher_bmi_per_higher_wealth': Parameter(Types.REAL, 'odds ratio higher BMI per higer wealth level'),
        'init_or_higher_bmi_per_higher_wealth_level': Parameter(
            Types.REAL, 'odds ratio for higher initial bmi category per higher wealth level'
        ),
        'init_p_high_sugar': Parameter(Types.REAL, 'initital proportion with high sugar intake'),
        'init_p_high_salt_urban': Parameter(Types.REAL, 'initital proportion with high salt intake'),
        'init_or_high_salt_rural': Parameter(Types.REAL, 'odds ratio high salt if rural'),
        'init_p_ex_alc_m': Parameter(Types.REAL, 'initital proportion of men with excess alcohol use'),
        'init_p_ex_alc_f': Parameter(Types.REAL, 'initital proportion of women with excess alcohol use'),
        'init_p_low_ex_urban_m': Parameter(Types.REAL, 'initital proportion of men with low exercise urban'),
        'init_or_low_ex_rural': Parameter(Types.REAL, 'odds ratio low exercise rural'),
        'init_or_low_ex_f': Parameter(Types.REAL, 'odds ratio low exercise female'),
        'init_p_tob_age1519_m_wealth1': Parameter(
            Types.REAL, 'initial proportion of 15-19 year old men using tobacco, wealth level 1 '
        ),
        'init_or_tob_f': Parameter(Types.REAL, 'odds ratio tobacco use females'),
        'init_or_tob_age2039_m': Parameter(Types.REAL, 'odds ratio tobacco use age2039 in men'),
        'init_or_tob_agege40_m': Parameter(Types.REAL, 'odds ratio tobacco use age40+ in men'),
        'init_or_tob_wealth2': Parameter(Types.REAL, 'odds ratio tobacco use wealth level 2'),
        'init_or_tob_wealth3': Parameter(Types.REAL, 'odds ratio tobacco use wealth level 3'),
        'init_or_tob_wealth4': Parameter(Types.REAL, 'odds ratio tobacco use wealth level 4'),
        'init_or_tob_wealth5': Parameter(Types.REAL, 'odds ratio tobacco use wealth level 5'),
        'init_dist_mar_stat_age1520': Parameter(Types.LIST, 'proportions never, current, div_wid age 15-20 baseline'),
        'init_dist_mar_stat_age2030': Parameter(Types.LIST, 'proportions never, current, div_wid age 20-30 baseline'),
        'init_dist_mar_stat_age3040': Parameter(Types.LIST, 'proportions never, current, div_wid age 30-40 baseline'),
        'init_dist_mar_stat_age4050': Parameter(Types.LIST, 'proportions never, current, div_wid age 40-50 baseline'),
        'init_dist_mar_stat_age5060': Parameter(Types.LIST, 'proportions never, current, div_wid age 50-60 baseline'),
        'init_dist_mar_stat_agege60': Parameter(Types.LIST, 'proportions never, current, div_wid age 60+ baseline'),
        'init_age2030_w5_some_ed': Parameter(
            Types.REAL, 'proportions of low wealth 20-30 year olds with some education at baseline'
        ),
        'init_or_some_ed_age0513': Parameter(Types.REAL, 'odds ratio of some education at baseline age 5-13'),
        'init_or_some_ed_age1320': Parameter(Types.REAL, 'odds ratio of some education at baseline age 13-20'),
        'init_or_some_ed_age2030': Parameter(Types.REAL, 'odds ratio of some education at baseline age 20-30'),
        'init_or_some_ed_age3040': Parameter(Types.REAL, 'odds ratio of some education at baseline age 30-40'),
        'init_or_some_ed_age4050': Parameter(Types.REAL, 'odds ratio of some education at baseline age 40-50'),
        'init_or_some_ed_age5060': Parameter(Types.REAL, 'odds ratio of some education at baseline age 50-60'),
        'init_or_some_ed_per_higher_wealth': Parameter(
            Types.REAL, 'odds ratio of some education at baseline per higher wealth level'
        ),
        'init_prop_age2030_w5_some_ed_sec': Parameter(
            Types.REAL,
            'proportion of low wealth aged 20-30 with some education who have secondary education at baseline',
        ),
        'init_or_some_ed_sec_age1320': Parameter(Types.REAL, 'odds ratio of secondary education age 13-20'),
        'init_or_some_ed_sec_age3040': Parameter(Types.REAL, 'odds ratio of secondary education age 30-40'),
        'init_or_some_ed_sec_age4050': Parameter(Types.REAL, 'odds ratio of secondary education age 40-50'),
        'init_or_some_ed_sec_age5060': Parameter(Types.REAL, 'odds ratio of secondary education age 50-60'),
        'init_or_some_ed_sec_agege60': Parameter(Types.REAL, 'odds ratio of secondary education age 60+'),
        'init_or_some_ed_sec_per_higher_wealth': Parameter(
            Types.REAL, 'odds ratio of secondary education per higher wealth level'
        ),
        'init_p_unimproved_sanitation_urban': Parameter(
            Types.REAL, 'initial probability of unimproved_sanitation given urban'
        ),
        # note that init_p_unimproved_sanitation is also used as the one-off probability of unimproved_sanitation '
        #                                                     'true to false upon move from rural to urban'
        'init_or_unimproved_sanitation_rural': Parameter(
            Types.REAL, 'initial odds ratio of unimproved_sanitation if rural'
        ),
        'init_p_no_clean_drinking_water_urban': Parameter(
            Types.REAL, 'initial probability of no_clean_drinking_water given urban'
        ),
        # note that init_p_no_clean_drinking_water is also used as the one-off probability of no_clean_drinking_water '
        #                                                     'true to false upon move from rural to urban'
        'init_or_no_clean_drinking_water_rural': Parameter(
            Types.REAL, 'initial odds ratio of no clean drinking_water if rural'
        ),
        'init_p_wood_burn_stove_urban': Parameter(Types.REAL, 'initial probability of wood_burn_stove given urban'),
        # note that init_p_wood_burn_stove is also used as the one-off probability of wood_burn_stove '
        #                                                     'true to false upon move from rural to urban'
        'init_or_wood_burn_stove_rural': Parameter(Types.REAL, 'initial odds ratio of wood_burn_stove if rural'),
        'init_p_no_access_handwashing_wealth1': Parameter(
            Types.REAL, 'initial probability of no_access_handwashing given wealth 1'
        ),
        'init_or_no_access_handwashing_per_lower_wealth': Parameter(
            Types.REAL, 'initial odds ratio of no_access_handwashing per lower wealth level'
        ),
        'init_prop_primary_edu': Parameter(Types.REAL, 'proportion of starting primary education at baseline age 5-12'),
        'init_rp_some_ed_age0513': Parameter(Types.REAL, 'relative prevalence of some education at baseline age 5-13'),
        'init_rp_some_ed_age1320': Parameter(Types.REAL, 'relative prevalence of some education at baseline age 13-20'),
        'init_rp_some_ed_age2030': Parameter(Types.REAL, 'relative prevalence of some education at baseline age 20-30'),
        'init_rp_some_ed_age3040': Parameter(Types.REAL, 'relative prevalence of some education at baseline age 30-40'),
        'init_rp_some_ed_age4050': Parameter(Types.REAL, 'relative prevalence of some education at baseline age 40-50'),
        'init_rp_some_ed_age5060': Parameter(Types.REAL, 'relative prevalence of some education at baseline age 50-60'),
        'init_rp_some_ed_per_higher_wealth': Parameter(
            Types.REAL, 'relative prevalence of some education at baseline per higher wealth level'
        ),
        'init_rp_some_ed_sec_age1320': Parameter(Types.REAL, 'relative prevalence of secondary education age 15-20'),
        'init_rp_some_ed_sec_age3040': Parameter(Types.REAL, 'relative prevalence of secondary education age 30-40'),
        'init_rp_some_ed_sec_age4050': Parameter(Types.REAL, 'relative prevalence of secondary education age 40-50'),
        'init_rp_some_ed_sec_age5060': Parameter(Types.REAL, 'relative prevalence of secondary education age 50-60'),
        'init_rp_some_ed_sec_agege60': Parameter(Types.REAL, 'relative prevalence of secondary education age 60+'),
        # Note: Added this to the properties and parameters tab of the resource file excel (init_rp_some_ed_agege60)
        # Did have a value in the parameter_values tabs but may need updating in other documents?
        'init_rp_some_ed_agege60': Parameter(
            Types.REAL, 'relative prevalence of some education at baseline age age 60+'
        ),
        'init_rp_some_ed_sec_per_higher_wealth': Parameter(
            Types.REAL, 'relative prevalence of secondary education per higher wealth level'
        ),
        # ------------ parameters relating to updating of property values over time ------------------------
        'r_urban': Parameter(Types.REAL, 'probability per 3 months of change from rural to urban'),
        'r_rural': Parameter(Types.REAL, 'probability per 3 months of change from urban to rural'),
        'r_higher_bmi': Parameter(
            Types.REAL,
            'probability per 3 months of increase in bmi category if rural male age'
            '15-29 not using tobacoo with wealth level 1 with not high sugar intake',
        ),
        'rr_higher_bmi_urban': Parameter(Types.REAL, 'probability per 3 months of increase in bmi category if '),
        'rr_higher_bmi_f': Parameter(Types.REAL, 'rate ratio for increase in bmi category for females'),
        'rr_higher_bmi_age3049': Parameter(Types.REAL, 'rate ratio for increase in bmi category for age 30-49'),
        'rr_higher_bmi_agege50': Parameter(Types.REAL, 'rate ratio for increase in bmi category for age ge 50'),
        'rr_higher_bmi_tob': Parameter(Types.REAL, 'rate ratio for increase in bmi category for tobacco users'),
        'rr_higher_bmi_per_higher_wealth': Parameter(
            Types.REAL, 'rate ratio for increase in bmi category per higher wealth level'
        ),
        'rr_higher_bmi_high_sugar': Parameter(
            Types.REAL, 'rate ratio for increase in bmi category for high sugar intake'
        ),
        'r_lower_bmi': Parameter(
            Types.REAL, 'probability per 3 months of decrease in bmi category in non tobacco users'
        ),
        'rr_lower_bmi_tob': Parameter(Types.REAL, 'rate ratio for lower bmi category for tobacco users'),
        'rr_lower_bmi_pop_advice_weight': Parameter(
            Types.REAL,
            'probability per 3 months of decrease in bmi category given population advice/campaign on weight',
        ),
        'r_high_salt_urban': Parameter(Types.REAL, 'probability per 3 months of high salt intake if urban'),
        'rr_high_salt_rural': Parameter(Types.REAL, 'rate ratio for high salt if rural'),
        'r_not_high_salt': Parameter(Types.REAL, 'probability per 3 months of not high salt intake'),
        'rr_not_high_salt_pop_advice_salt': Parameter(
            Types.REAL, 'probability per 3 months of not high salt given population advice/campaign on salt'
        ),
        'r_high_sugar': Parameter(Types.REAL, 'probability per 3 months of high sugar intake'),
        'r_not_high_sugar': Parameter(Types.REAL, 'probability per 3 months of not high sugar intake'),
        'rr_not_high_sugar_pop_advice_sugar': Parameter(
            Types.REAL, 'probability per 3 months of not high sugar given population advice/campaign on sugar'
        ),
        'r_low_ex': Parameter(Types.REAL, 'probability per 3 months of change from not low exercise to low exercise'),
        'r_not_low_ex': Parameter(
            Types.REAL, 'probability per 3 months of change from low exercise to not low exercie'
        ),
        'rr_not_low_ex_pop_advice_exercise': Parameter(
            Types.REAL, 'probability per 3 months of not low exercise population advice/campaign on exercise'
        ),
        'rr_low_ex_f': Parameter(Types.REAL, 'risk ratio for becoming low exercise if female rather than male'),
        'rr_low_ex_urban': Parameter(Types.REAL, 'risk ratio for becoming low exercise if urban rather than rural'),
        'r_tob': Parameter(
            Types.REAL,
            'probability per 3 months of change from not using tobacco to using '
            'tobacco if male age 15-19 wealth level 1',
        ),
        'r_not_tob': Parameter(
            Types.REAL, 'probability per 3 months of change from tobacco using to not tobacco using'
        ),
        'rr_tob_f': Parameter(Types.REAL, 'rate ratio tobacco if female'),
        'rr_tob_age2039': Parameter(Types.REAL, 'risk ratio for tobacco using if age 20-39 compared with 15-19'),
        'rr_tob_agege40': Parameter(Types.REAL, 'risk ratio for tobacco using if age >= 40 compared with 15-19'),
        'rr_tob_wealth': Parameter(
            Types.REAL, 'risk ratio for tobacco using per 1 higher wealth level (higher wealth level = lower wealth)'
        ),
        'rr_not_tob_pop_advice_tobacco': Parameter(
            Types.REAL, 'probability per 3 months of quitting tobacco given population advice/campaign on tobacco'
        ),
        'r_ex_alc': Parameter(
            Types.REAL, 'probability per 3 months of change from not excess alcohol to excess alcohol'
        ),
        'r_not_ex_alc': Parameter(
            Types.REAL, 'probability per 3 months of change from excess alcohol to not excess alcohol'
        ),
        'rr_ex_alc_f': Parameter(Types.REAL, 'risk ratio for becoming excess alcohol if female rather than male'),
        'rr_not_ex_alc_pop_advice_alcohol': Parameter(
            Types.REAL, 'probability per 3 months of not excess alcohol given population advice/campaign on alcohol'
        ),
        'r_mar': Parameter(Types.REAL, 'probability per 3 months of marriage when age 15-30'),
        'r_div_wid': Parameter(
            Types.REAL, 'probability per 3 months of becoming divorced or widowed, amongst those married'
        ),
        'r_stop_ed': Parameter(Types.REAL, 'probabilities per 3 months of stopping education if wealth level 5'),
        'rr_stop_ed_lower_wealth': Parameter(
            Types.REAL, 'relative rate of stopping education per 1 lower wealth quantile'
        ),
        'p_ed_primary': Parameter(Types.REAL, 'probability at age 5 that start primary education if wealth level 5'),
        'rp_ed_primary_higher_wealth': Parameter(
            Types.REAL, 'relative probability of starting school per 1 higher wealth level'
        ),
        'p_ed_secondary': Parameter(
            Types.REAL,
            'probability at age 13 that start secondary education at 13 if in primary education and wealth level 5',
        ),
        'rp_ed_secondary_higher_wealth': Parameter(
            Types.REAL, 'relative probability of starting secondary school per 1 higher wealth level'
        ),
        'r_improved_sanitation': Parameter(
            Types.REAL, 'probability per 3 months of change from unimproved_sanitation true to false'
        ),
        'r_clean_drinking_water': Parameter(
            Types.REAL, 'probability per 3 months of change from drinking_water true to false'
        ),
        'r_non_wood_burn_stove': Parameter(
            Types.REAL, 'probability per 3 months of change from wood_burn_stove true to false'
        ),
        'r_access_handwashing': Parameter(
            Types.REAL, 'probability per 3 months of change from no access hand washing true to false'
        ),
        'start_date_campaign_exercise_increase': Parameter(
            Types.DATE, 'Date of campaign start for increased exercise'
        ),
        'start_date_campaign_quit_smoking': Parameter(
            Types.DATE, 'Date of campaign start to quit smoking'
        ),
        'start_date_campaign_alcohol_reduction': Parameter(
            Types.DATE, 'Date of campaign start for alcohol reduction'
        ),
        'proportion_of_men_that_are_assumed_to_be_circumcised_at_birth': Parameter(
            Types.REAL, 'Proportion of men that are assumed to be circumcised at birth.'
                        'The men are assumed to be circumcised at birth.'
        ),
        'proportion_of_men_circumcised_at_initiation': Parameter(
            Types.REAL, 'Proportion of men (of all ages) that are assumed to be circumcised at the initiation of the'
                        'simulation.'
        ),
        "proportion_female_sex_workers": Parameter(
            Types.REAL, "proportion of women aged 15-49 years who are sex workers"
        ),
        "fsw_transition": Parameter(
            Types.REAL, "proportion of sex workers that stop being a sex worker each year"
        )
    }

    # Properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'li_urban': Property(Types.BOOL, 'Currently urban'),
        'li_wealth': Property(Types.CATEGORICAL, 'wealth level: 1 (high) to 5 (low)', categories=[1, 2, 3, 4, 5]),
        'li_bmi': Property(
            Types.CATEGORICAL, 'bmi: 1 (<18) 2 (18-24.9)  3 (25-29.9) 4 (30-34.9) 5 (35+)'
                               'bmi is np.nan until age 15', categories=[1, 2, 3, 4, 5], ordered=True
        ),
        'li_exposed_to_campaign_weight_reduction': Property(
            Types.BOOL, 'currently exposed to population campaign for weight reduction if BMI >= 25'
        ),
        'li_low_ex': Property(Types.BOOL, 'currently low exercise'),
        'li_exposed_to_campaign_exercise_increase': Property(
            Types.BOOL, 'currently exposed to population campaign for increase exercise if low ex'
        ),
        'li_high_salt': Property(Types.BOOL, 'currently high salt intake'),
        'li_exposed_to_campaign_salt_reduction': Property(
            Types.BOOL, 'currently exposed to population campaign for salt reduction if high salt'
        ),
        'li_high_sugar': Property(Types.BOOL, 'currently high sugar intake'),
        'li_exposed_to_campaign_sugar_reduction': Property(
            Types.BOOL, 'currently exposed to population campaign for sugar reduction if high sugar'
        ),
        'li_tob': Property(Types.BOOL, 'current using tobacco'),
        'li_date_not_tob': Property(Types.DATE, 'date last transitioned from tob to not tob'),
        'li_exposed_to_campaign_quit_smoking': Property(
            Types.BOOL, 'currently exposed to population campaign to quit smoking if tob'
        ),
        'li_ex_alc': Property(Types.BOOL, 'current excess alcohol'),
        'li_exposed_to_campaign_alcohol_reduction': Property(
            Types.BOOL, 'currently exposed to population campaign for alcohol reduction if ex alc'
        ),
        'li_mar_stat': Property(
            Types.CATEGORICAL, 'marital status {1:never, 2:current, 3:past (widowed or divorced)}', categories=[1, 2, 3]
        ),
        'li_in_ed': Property(Types.BOOL, 'currently in education'),
        'li_ed_lev': Property(Types.CATEGORICAL, 'education level achieved as of now', categories=[1, 2, 3],
                              ordered=True),
        'li_unimproved_sanitation': Property(
            Types.BOOL, 'uninproved sanitation - anything other than own or shared latrine'
        ),
        'li_no_access_handwashing': Property(
            Types.BOOL, 'no_access_handwashing - no water, no soap, no other cleaning agent - as in DHS'
        ),
        'li_no_clean_drinking_water': Property(Types.BOOL, 'no drinking water from an improved source'),
        'li_wood_burn_stove': Property(Types.BOOL, 'wood (straw / crop)-burning stove'),
        'li_date_trans_to_urban': Property(Types.DATE, 'date transition to urban'),
        'li_date_acquire_improved_sanitation': Property(Types.DATE, 'date transition to urban'),
        'li_date_acquire_access_handwashing': Property(Types.DATE, 'date acquire access to handwashing'),
        'li_date_acquire_clean_drinking_water': Property(Types.DATE, 'date acquire clean drinking water'),
        'li_date_acquire_non_wood_burn_stove': Property(Types.DATE, 'date acquire non-wood burning stove'),
        "li_is_sexworker": Property(Types.BOOL, "Is the person a sex worker"),
        "li_is_circ": Property(Types.BOOL, "Is the person circumcised if they are male (False for all females)"),
    }

    def read_parameters(self, data_folder):
        p = self.parameters
        dataframes = read_csv_files(
            Path(self.resourcefilepath) / 'ResourceFile_Lifestyle_Enhanced',
            files=["parameter_values", "urban_rural_by_district"],
        )
        self.load_parameters_from_dataframe(dataframes["parameter_values"])
        p['init_p_urban'] = (
            dataframes["urban_rural_by_district"].drop(
                columns=["rural", "urban", "prop_rural"], axis=1
            ).set_index("district").to_dict()
        )

        # Manually set dates for campaign starts
        # Todo: adjust these to better represent the number of people who have transitioned
        p['start_date_campaign_exercise_increase'] = Date(2010, 7, 1)
        p['start_date_campaign_quit_smoking'] = Date(2010, 7, 1)
        p['start_date_campaign_alcohol_reduction'] = Date(2010, 7, 1)

    def pre_initialise_population(self):
        """Initialise the linear model class"""
        self.models = LifestyleModels(self)

    def initialise_population(self, population):
        """Set our property values for the initial population.
        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals
        # todo: express all rates per year and divide by 4 inside program

        # initialise all properties using linear models
        self.models.initialise_all_properties(df)

    def initialise_simulation(self, sim):
        """Add lifestyle events to the simulation
        """
        event = LifestyleEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=3))

        event = LifestylesLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=0))

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props

        # Determine id from which characteristics that inherited (from mother, or if no
        # mother, from a randomly selected alive person that is not child themself)
        _id_inherit_from = get_person_id_to_inherit_from(
            child_id, mother_id, df, self.rng
        )

        df.at[child_id, 'li_urban'] = df.at[_id_inherit_from, 'li_urban']
        df.at[child_id, 'li_wealth'] = df.at[_id_inherit_from, 'li_wealth']
        df.at[child_id, 'li_bmi'] = np.nan
        df.at[child_id, 'li_exposed_to_campaign_weight_reduction'] = False
        df.at[child_id, 'li_low_ex'] = False
        df.at[child_id, 'li_exposed_to_campaign_exercise_increase'] = False
        df.at[child_id, 'li_high_salt'] = df.at[_id_inherit_from, 'li_high_salt']
        df.at[child_id, 'li_exposed_to_campaign_salt_reduction'] = False
        df.at[child_id, 'li_high_sugar'] = df.at[_id_inherit_from, 'li_high_sugar']
        df.at[child_id, 'li_exposed_to_campaign_sugar_reduction'] = False
        df.at[child_id, 'li_tob'] = False
        df.at[child_id, 'li_date_not_tob'] = pd.NaT
        df.at[child_id, 'li_exposed_to_campaign_quit_smoking'] = False
        df.at[child_id, 'li_ex_alc'] = False
        df.at[child_id, 'li_exposed_to_campaign_alcohol_reduction'] = False
        df.at[child_id, 'li_mar_stat'] = 1
        df.at[child_id, 'li_in_ed'] = False
        df.at[child_id, 'li_ed_lev'] = 1
        df.at[child_id, 'li_unimproved_sanitation'] = df.at[_id_inherit_from, 'li_unimproved_sanitation']
        df.at[child_id, 'li_no_access_handwashing'] = df.at[_id_inherit_from, 'li_no_access_handwashing']
        df.at[child_id, 'li_no_clean_drinking_water'] = df.at[_id_inherit_from, 'li_no_clean_drinking_water']
        df.at[child_id, 'li_wood_burn_stove'] = df.at[_id_inherit_from, 'li_wood_burn_stove']
        df.at[child_id, 'li_date_trans_to_urban'] = pd.NaT
        df.at[child_id, 'li_date_acquire_improved_sanitation'] = pd.NaT
        df.at[child_id, 'li_date_acquire_access_handwashing'] = pd.NaT
        df.at[child_id, 'li_date_acquire_clean_drinking_water'] = pd.NaT
        df.at[child_id, 'li_date_acquire_non_wood_burn_stove'] = pd.NaT
        df.at[child_id, 'li_is_sexworker'] = False
        df.at[child_id, 'li_is_circ'] = (
            self.rng.rand() < self.parameters['proportion_of_men_that_are_assumed_to_be_circumcised_at_birth']
        )


class EduPropertyInitialiser:
    """ a class that will initialise education property in the population dataframe. it is mimicing the
    Linear model as its predict method is expected to behave like a linear model when called inside
    Lifestyle class """

    def __init__(self, module, _property: str, ):
        # this property will be used to return the prefered series
        self.edu_property = _property
        self.module = module

    def predict(self, df, rng=None, **kwargs) -> pd.Series:
        """ use output from education linear models to set education levels and status

        :param self: a reference to EduProperties class
        :param df: population dataframe
        :param rng: random number generator """
        # create a new dataframe to hold results
        edu_df = pd.DataFrame(data=df[['li_in_ed', 'li_ed_lev']].copy(), index=df.index)
        edu_df['li_ed_lev'] = 1

        # select individuals who are alive and 5+ years
        age_gte5 = df.index[(df.age_years >= 5) & df.is_alive]

        # store population eligible for education
        edu_pop = df.loc[(df.age_years >= 5) & df.is_alive]
        rnd_draw = pd.Series(data=rng.random_sample(size=len(age_gte5)), index=age_gte5, dtype=float)

        # make some predictions
        education_linear_models = self.module.models.education_linear_models()
        p_some_ed = education_linear_models['some_edu_linear_model'].predict(edu_pop)
        p_ed_lev_3 = education_linear_models['level_3_edu_linear_model'].predict(edu_pop)

        dfx = pd.concat([(1 - p_ed_lev_3), (1 - p_some_ed)], axis=1)
        dfx.columns = ['cut_off_ed_levl_3', 'p_ed_lev_1']

        dfx['li_ed_lev'] = 2
        dfx.loc[dfx['cut_off_ed_levl_3'] < rnd_draw, 'li_ed_lev'] = 3
        dfx.loc[dfx['p_ed_lev_1'] > rnd_draw, 'li_ed_lev'] = 1

        edu_df.loc[age_gte5, 'li_ed_lev'] = dfx['li_ed_lev']

        # ---- PRIMARY EDUCATION
        # get index of all individuals alive and aged between 5 and 12 years old
        age512 = edu_df.index[df.age_years.between(5, 12) & (edu_df.li_ed_lev == 2) & df.is_alive]

        # create a series to hold the probablity of being in primary education for all individuals aged 5 to 12.
        # Here we assume a 50-50 chance of being in education
        prob_primary_edu = pd.Series(data=self.module.parameters['init_prop_primary_edu'], index=age512, dtype=float)

        # randomly select some individuals to be in education
        age512_in_edu = rng.random_sample(len(age512)) < prob_primary_edu

        # for the selected individuals, set their in education property to true
        edu_df.loc[age512[age512_in_edu], 'li_in_ed'] = True

        # --- SECONDARY EDUCATION
        edu_df.loc[df.age_years.between(13, 19) & (edu_df.li_ed_lev == 3) & df.is_alive, 'li_in_ed'] = True

        # return results based on the selected property
        return edu_df.li_in_ed if self.edu_property == 'li_in_ed' else edu_df.li_ed_lev


class BmiPropertyInitialiser:
    """ a class that will be used to initialise the bmi property in the population dataframe. it is receiving
    predictions from bmi linear models defined in LifestyleModels class and predict different bmi categories.
    This class is mimicing the Linear model as its predict method is expected to behave like a linear model
    when called inside Lifestyle class """

    def __init__(self, module):
        # initialise some parameters here
        self.module = module

    def predict(self, df, rng=None, **kwargs) -> pd.Series:
        """ set property for BMI in population dataframe

        :param df: population dataframe
        :param rng: random number generator    """

        # create a series that will hold all bmi categories
        bmi_cat_series = pd.Series(data=np.nan, index=df.index, dtype=int)

        # get indexes of population alive and 15+ years
        age_ge15_idx = df.index[df.is_alive & (df.age_years >= 15)]
        prop_df = df.loc[df.is_alive & (df.age_years >= 15)]

        # only relevant if at least one individual with age >= 15 years present
        if len(age_ge15_idx) > 0:
            # this below is the approach to apply the effect of contributing determinants on bmi levels at baseline
            # create bmi probabilities dataframe using bmi linear model and normalise to sum to 1
            df_lm = pd.DataFrame()
            bmi_pow = [-2, -1, 0, 1, 2]

            for index, power in enumerate(bmi_pow):
                df_lm[index + 1] = self.module.models.bmi_linear_model(index, power).predict(prop_df)

            dfxx = df_lm.div(df_lm.sum(axis=1), axis=0)

            # for each row, make a choice
            bmi_cat = dfxx.apply(lambda p_bmi: rng.choice(dfxx.columns, p=p_bmi), axis=1)

            bmi_cat_series.loc[age_ge15_idx] = bmi_cat
            return bmi_cat_series


class LifestyleModels:
    """Helper class to store all linear models for the Lifestyle module. We have used three types of linear models
    namely logistic, multiplicative and custom linear models. We currently have defined linear models for
    the following;

            1.  urban rural status
            2.  wealth level
            3.  low exercise
            4.  tobacco use
            5.  excessive alcohol
            6.  marital status
            7.  education
            8.  unimproved sanitation
            9.  no clean drinking water
            10. wood burn stove
            11. no access hand washing
            12. salt intake
            13. sugar intake
            14. bmi
            15. male circumcision
            16. female sex workers """

    def __init__(self, module):
        # initialise variables
        self.module = module
        self.rng = self.module.rng
        self.params = module.parameters
        self.date_last_run = module.sim.date

        # define a dictionary that wll hold results from education linear models
        self.edu_lm_res = dict()

        # create all linear models dictionary for use in both initialisation and update of properties
        self._models = {
            'li_urban': {
                'init': self.rural_urban_linear_model(),
                'update': self.update_rural_urban_property_linear_model()
            },
            'li_wealth': {
                'init': self.wealth_level_linear_model(),
                'update': None
            },
            'li_low_ex': {
                'init': self.low_exercise_linear_model(),
                'update': self.update_exercise_property_linear_model()
            },
            'li_tob': {
                'init': self.tobacco_use_linear_model(),
                'update': self.update_tobacco_use_property_linear_model()
            },
            'li_ex_alc': {
                'init': self.excessive_alcohol_linear_model(),
                'update': self.update_excess_alcohol_property_linear_model()
            },
            'li_mar_stat': {
                'init': self.marital_status_linear_model(),
                'update': self.update_marital_status_linear_model()
            },
            'li_in_ed': {
                'init': EduPropertyInitialiser(self.module, 'li_in_ed'),
                'update': self.update_education_status_linear_model('li_in_ed')
            },
            'li_ed_lev': {
                'init': EduPropertyInitialiser(self.module, 'li_ed_lev'),
                'update': self.update_education_status_linear_model('li_ed_lev')
            },
            'li_unimproved_sanitation': {
                'init': self.unimproved_sanitation_linear_model(),
                'update': self.update_unimproved_sanitation_status_linear_model()
            },
            'li_no_clean_drinking_water': {
                'init': self.no_clean_drinking_water_linear_model(),
                'update': self.update_no_clean_drinking_water_linear_model()
            },
            'li_wood_burn_stove': {
                'init': self.wood_burn_stove_linear_model(),
                'update': self.update_wood_burn_stove_linear_model()
            },
            'li_no_access_handwashing': {
                'init': self.no_access_hand_washing(),
                'update': self.update_no_access_hand_washing_status_linear_model()
            },
            'li_high_salt': {
                'init': self.salt_intake_linear_model(),
                'update': self.update_high_salt_property_linear_model()
            },
            'li_high_sugar': {
                'init': self.sugar_intake_linear_model(),
                'update': self.update_high_sugar_property_linear_model()
            },
            'li_bmi': {
                'init': BmiPropertyInitialiser(self.module),
                'update': self.update_bmi_categories_linear_model()
            },
            'li_is_circ': {
                'init': self.male_circumcision_property_linear_model(),
                'update': None
            },
            'li_is_sexworker': {
                'init': self.female_sex_workers(),
                'update': self.female_sex_workers()
            }
        }

    def is_edu_dictionary_empty(self):
        """ a function to check if education dictionary is empty or not """
        return self.edu_lm_res

    def get_lm_keys(self):
        """ a function to return all linear model keys as defined in models dictionary """
        return self._models.keys()

    def initialise_all_properties(self, df):
        """ initialise population properties using linear models defined in LifestyleModels class.

        :param df: The population dataframe """
        # loop through linear models dictionary and initialise each property in the population dataframe
        for _property_name, _model in self._models.items():
            df.loc[df.is_alive, _property_name] = _model['init'].predict(
                df.loc[df.is_alive], rng=self.rng, other=self.module.sim.date, months_since_last_poll=0)

    def update_all_properties(self, df):
        """update population properties using linear models defined in LifestyleModels class. This function is to be
         called by the Lifestyle Event class

        :param df: The population dataframe """
        # get months since last poll
        now = self.module.sim.date
        months_since_last_poll = round((now - self.date_last_run) / np.timedelta64(1, "M"))
        # loop through linear models dictionary and initialise each property in the population dataframe
        for _property_name, _model in self._models.items():
            if _model['update'] is not None:
                df.loc[df.is_alive, _property_name] = _model['update'].predict(
                    df.loc[df.is_alive], rng=self.rng, other=self.module.sim.date,
                    months_since_last_poll=months_since_last_poll)
        # update date last event run
        self.date_last_run = now

    def rural_urban_linear_model(self) -> LinearModel:
        """ a function to create linear model for rural urban properties. Here we are using custom linear model and
        assigning individuals rural urban status based on their district of origin """

        def predict_rural_urban_status(self, df, rng=None, **externals) -> pd.Series:
            """ a function to assign individuals rural, urban status

            :param  self:   a reference to the calling custom linear model
            :param  df:     population dataframe
            :param  rng:    a random number generator
            :param  externals: a dict containing any other variables passed on to this function """
            # get access to module parameters
            p = self.parameters

            # generate a radom number
            rnd_draw = pd.Series(data=rng.random_sample(size=len(df)), index=df.index, dtype=float)

            # map district of residence to a series containing probabilities of becoming rural or urban
            rural_urban_props = df['district_of_residence'].map(p['init_p_urban']['prop_urban'])

            # check all districts have been corectly mapped to their rural urban proportions
            assert not rural_urban_props.isnull().any(), 'some districts are not mapped to their rural urban values'
            # check urban rural proportion is greater or equal to 0 but less or equal to 1
            assert rural_urban_props.apply(lambda x: 0.0 <= x <= 1.0).any(), 'proportion less than 0 or greater than 1'

            # get individual's rural urban status
            rural_urban = rural_urban_props > rnd_draw

            # return individual rural urban status
            return rural_urban

        # create and return wealth levels linear model
        rural_urban_lm = LinearModel.custom(predict_rural_urban_status, parameters=self.params)
        return rural_urban_lm

    def wealth_level_linear_model(self) -> LinearModel:
        """ a function to create linear model for wealth level property. Here are using custom linear model
        and are setting probabilities based on whether the individual is urban or rural

        wealth level categories here are defined as follows;

        Urban                               |         Rural
        ------------------------------------|----------------------------------------------
        leve 1 = 75% wealth level           |  level 1 = 11% wealth level.
        level 2 = 16% wealth level          |  level 2 = 21% wealth level.
        level 3 = 5% wealth level           |  level 3 = 23% wealth level.
        level 4 = 2% wealth level           |  level 4 = 23% wealth level.
        level 5 = 2% wealth level           |  level 5 = 23% wealth level

        """

        def predict_wealth_levels(self, df, rng=None, **externals) -> pd.Series:
            """ a function to assign individuals different wealth levels

            :param  self:   a reference to the calling custom linear model
            :param  df:     population dataframe
            :param  rng:    a random number generator
            :param  externals: a dict containing any other variables passed on to this function """
            # get access to module parameters
            p = self.parameters

            # create a new series to hold different wealth levels
            li_wealth_dtype = df.li_wealth.dtype
            res = pd.Series(index=df.index, dtype=li_wealth_dtype)
            num_urban = df.li_urban.sum()
            num_rural = len(df) - num_urban
            res[df.li_urban] = rng.choice(li_wealth_dtype.categories, p=p['init_p_wealth_urban'], size=num_urban)
            res[~df.li_urban] = rng.choice(li_wealth_dtype.categories, p=p['init_p_wealth_rural'], size=num_rural)
            # return wealth level linear model
            return res

        # create and return wealth levels linear model
        wealth_lev_lm = LinearModel.custom(predict_wealth_levels, parameters=self.params)
        return wealth_lev_lm

    def low_exercise_linear_model(self) -> LinearModel:
        """A function to create a linear model for lower exercise property of Lifestyle module. Here we are using
        Logistic linear model and we are looking at an individual's probability of being low exercise based on
        gender and rural or urban based. Finally, we return a dictionary containing low exercise linear models """

        # get baseline odds for exercise
        init_odds_low_ex_urban_m: float = (self.params['init_p_low_ex_urban_m']
                                           / (1 - self.params['init_p_low_ex_urban_m']))

        # create low exercise linear model
        low_exercise_lm = LinearModel(LinearModelType.LOGISTIC,
                                      init_odds_low_ex_urban_m,
                                      Predictor('age_years').when('<15', 0),
                                      Predictor('sex').when('F', self.params['init_or_low_ex_f']),
                                      Predictor('li_urban').when(False, self.params['init_or_low_ex_rural'])
                                      )

        # return low exercise dictionary
        return low_exercise_lm

    def tobacco_use_linear_model(self) -> LinearModel:
        """A function to create a Linear Model for tobacco use. Here we are using Logistic Linear Model. it is
        designed to accept the outputs of standard logistic regression, which are based on 'odds'. The relationship
        between odds and probabilities are as follows: odds = prob / (1 - prob); prob = odds / (1 + odds) The
        intercept is the baseline odds and effects are the Odds Ratio. After everything we are returning the Linear
        Model to predict final outcomes based on population properties """

        # get the baseline odds
        tobacco_use_baseline_odds: float = (self.params['init_p_tob_age1519_m_wealth1'] /
                                            (1 - self.params['init_p_tob_age1519_m_wealth1']))

        # get population properties and apply effects
        tobacco_use_linear_model = LinearModel(LinearModelType.LOGISTIC,
                                               tobacco_use_baseline_odds,
                                               Predictor('age_years').when('<15', 0.0),
                                               Predictor('sex').when('F', self.params['init_or_tob_f']),
                                               Predictor().when(
                                                   '(sex == "M") & (age_years >= 20) & (age_years < 40)',
                                                   self.params['init_or_tob_age2039_m']),
                                               Predictor().when('(sex == "M") & (age_years >= 40)',
                                                                self.params['init_or_tob_agege40_m']),
                                               Predictor('li_wealth').when(2, 2)
                                               .when(3, 3)
                                               .when(4, 4)
                                               .when(5, 5)
                                               )
        # return a Linear Model object
        return tobacco_use_linear_model

    def excessive_alcohol_linear_model(self) -> LinearModel:
        """ a function to create linear model for excessive alcohol property. Here we are using additive linear model
        and are looking at individual probabilities of excessive alcohol based on their gender. In this model we are
        considering gender of an individual as either Male or Female """

        # set baseline probability for all gender male or female
        base_prob = 0.0

        # define excessive alcohol linear model
        excessive_alc_lm = LinearModel(LinearModelType.ADDITIVE,
                                       base_prob,
                                       Predictor().when('(age_years >= 15) & (sex == "M")',
                                                        self.params['init_p_ex_alc_m'])
                                       .when('(age_years >= 15) & (sex == "F")',
                                             self.params['init_p_ex_alc_f']),
                                       )
        # return excessive alcohol linear model
        return excessive_alc_lm

    def marital_status_linear_model(self) -> LinearModel:
        """A function to create linear model for individual's marital status. Here, We are using a custom
        linear model and we are assigning individual's marital status based on their age group.In this module,
        marital status is in three categories;
                1.  Never Married
                2.  Currently Married
                3   Divorced or Widowed
        """

        def init_marital_status(self, df, rng=None, **externals) -> pd.Series:
            """ a function to assign marital status to individuals of different age groups

            :param self: a reference to cutom linear model
            :param df:  population dataframe
            :param  rng:
            :param  externals a dict containing any other variables passed on to this function """
            # get access to Lifestyle module parameters
            p = self.parameters

            li_mar_stat_dtype = df.li_mar_stat.dtype
            mar_stat = pd.Series(data=1, index=df.index, dtype=li_mar_stat_dtype )
            # select individuals of different age category
            age_ranges = [(15, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, np.inf)]
            for lower_age, upper_age in age_ranges:
                subpopulation = df.index[
                    df.age_years.between(lower_age, upper_age, inclusive="left")
                    & df.is_alive
                ]
                parameters_key = (
                    f"init_dist_mar_stat_age{lower_age}{upper_age}"
                    if upper_age != np.inf else
                    f"init_dist_mar_stat_agege{lower_age}"
                )
                mar_stat[subpopulation] = rng.choice(
                    li_mar_stat_dtype.categories,
                    p=p[parameters_key],
                    size=len(subpopulation)
                )
            # return marital status series
            return mar_stat

        # create and return marital status linear model
        mar_status_lm = LinearModel.custom(init_marital_status, parameters=self.params)
        return mar_status_lm

    def education_linear_models(self) -> Dict[str, LinearModel]:
        """A function to create linear models for education properties of Lifestyle module. In this case we choose
        Multiplicative linear model as our linear model Type, create two linear models one for education for all
        individuals over 5 years old and another for level 3 education and Finally, we return a linear model
        dictionary to help predict property values of a given population """

        # define a dictionary that will hold all linear models in this function
        education_lm_dict: Dict[str, LinearModel] = dict()

        # get the baseline of education for all individuals over 5 years old
        p_some_ed = self.params['init_age2030_w5_some_ed']

        # get the baseline of education level 3
        p_ed_lev_3 = self.params['init_prop_age2030_w5_some_ed_sec']

        some_education_linear_model = LinearModel(LinearModelType.MULTIPLICATIVE,  # choose linear model type
                                                  p_some_ed,  # intercept (default probability for all individuals)

                                                  # adjust probability of some education based on age
                                                  Predictor('age_years').when('<13',
                                                                              self.params['init_rp_some_ed_age0513'])
                                                  .when('.between(13, 19)', self.params['init_rp_some_ed_age1320'])
                                                  .when('.between(30, 39)', self.params['init_rp_some_ed_age3040'])
                                                  .when('.between(40, 49)', self.params['init_rp_some_ed_age4050'])
                                                  .when('.between(50, 59)', self.params['init_rp_some_ed_age5060'])
                                                  .when('>=60', self.params['init_rp_some_ed_agege60']),

                                                  # adjust probability of some education based on wealth
                                                  Predictor('li_wealth').when(1, self.params[
                                                      'init_rp_some_ed_per_higher_wealth'] ** 4)
                                                  .when(2, self.params[
                                                      'init_rp_some_ed_per_higher_wealth'] ** 3)
                                                  .when(3, self.params[
                                                      'init_rp_some_ed_per_higher_wealth'] ** 2)
                                                  .when(4, self.params[
                                                      'init_rp_some_ed_per_higher_wealth'] ** 1)
                                                  .when(5, self.params[
                                                      'init_rp_some_ed_per_higher_wealth'] ** 0)
                                                  )

        # calculate baseline of education level 3, and adjust for age and wealth
        level_3_education_linear_model = LinearModel(LinearModelType.MULTIPLICATIVE,
                                                     p_ed_lev_3,

                                                     # adjust baseline of education level 3 for age
                                                     Predictor('age_years').when('<13', 0.0)
                                                     .when('.between(13, 19)',
                                                           self.params['init_rp_some_ed_sec_age1320'])
                                                     .when('.between(30, 39)',
                                                           self.params['init_rp_some_ed_sec_age3040'])
                                                     .when('.between(40, 49)',
                                                           self.params['init_rp_some_ed_sec_age4050'])
                                                     .when('.between(50, 59)',
                                                           self.params['init_rp_some_ed_sec_age5060'])
                                                     .when('>=60',
                                                           self.params['init_rp_some_ed_sec_agege60']),

                                                     # adjust baseline of education level 3 for wealth
                                                     Predictor('li_wealth').when(1, (
                                                         self.params['init_rp_some_ed_sec_per_higher_wealth'] ** 4))
                                                     .when(2, (self.params[
                                                                   'init_rp_some_ed_sec_per_higher_wealth'] ** 3))
                                                     .when(3, (self.params[
                                                                   'init_rp_some_ed_sec_per_higher_wealth'] ** 2))
                                                     .when(4, (self.params[
                                                                   'init_rp_some_ed_sec_per_higher_wealth'] ** 1))
                                                     .when(5, (self.params[
                                                                   'init_rp_some_ed_sec_per_higher_wealth'] ** 0))
                                                     )

        # update education linear models dictionary with all defined linear models( some education linear model and
        # level 3 education linear model)
        education_lm_dict['some_edu_linear_model'] = some_education_linear_model
        education_lm_dict['level_3_edu_linear_model'] = level_3_education_linear_model

        # return a linear model dictionary
        return education_lm_dict

    def unimproved_sanitation_linear_model(self) -> LinearModel:
        """A function to create linear model for unimproved sanitation. Here, We are using a Logistic linear model
        and we are looking at an individual's probability of unimproved sanitation based on whether they are rural or
        urban based. Finally we return a linear model """

        # get the baseline odds for unimproved sanitation
        init_odds_un_imp_san: float = self.params['init_p_unimproved_sanitation_urban'] / (
            1 - self.params['init_p_unimproved_sanitation_urban'])

        # create an unimproved sanitation linear model
        un_imp_san_lm = LinearModel(LinearModelType.LOGISTIC,
                                    init_odds_un_imp_san,

                                    # update odds according to determinants of unimproved sanitation (rural status
                                    # the only determinant)
                                    Predictor('li_urban').when(False, self.params[
                                        'init_or_unimproved_sanitation_rural'])
                                    )

        # return unimproved sanitation linear model
        return un_imp_san_lm

    def no_clean_drinking_water_linear_model(self) -> LinearModel:
        """This function creates a linear model for no clean drinking water property. Here, we are using Logistic
        linear model and looking at individual's probability to have no clean drinking water based on whether they
        are rural or urban. """

        # get baseline odds for no clean drinking water
        init_odds_no_clean_drinking_water: float = self.params['init_p_no_clean_drinking_water_urban'] / (
            1 - self.params['init_p_no_clean_drinking_water_urban']
        )

        # create no clean drinking water linear model
        no_clean_drinking_water_lm = LinearModel(LinearModelType.LOGISTIC,
                                                 init_odds_no_clean_drinking_water,
                                                 Predictor('li_urban').when(False, self.params[
                                                     'init_or_no_clean_drinking_water_rural'])
                                                 )

        # return no clean drinking water linear model
        return no_clean_drinking_water_lm

    def wood_burn_stove_linear_model(self) -> LinearModel:
        """This function create a linear model for wood burn stove property. Here, we are using logistic linear model
        and looking at individual's probability of using wood burn stove based on whether they are rural or urban  """

        # get baseline odds for wood burn stove
        init_odds_wood_burn_stove: float = self.params['init_p_wood_burn_stove_urban'] / (
            1 - self.params['init_p_wood_burn_stove_urban'])

        # create wood burn stove linear model
        wood_burn_stove_lm = LinearModel(LinearModelType.LOGISTIC,
                                         init_odds_wood_burn_stove,
                                         Predictor('li_urban').when(False,
                                                                    self.params['init_or_wood_burn_stove_rural'])
                                         )

        # return wood burn stove linear model
        return wood_burn_stove_lm

    def no_access_hand_washing(self) -> LinearModel:
        """This function creates a linear model for no access to hand-washing property.  Here, we are using Logistic
        linear model and looking at individual's probability of having no access to hand washing based on their
        wealth level/status. Finally, we return a no access to hand-washing linear model """

        # get baseline odds for individuals with no access to hand-washing
        odds_no_access_hand_washing: float = 1 / (1 - self.params['init_p_no_access_handwashing_wealth1'])

        # create linear model for no access to hand-washing
        no_access_hand_washing_lm = LinearModel(LinearModelType.LOGISTIC,
                                                odds_no_access_hand_washing,
                                                Predictor('li_wealth').when(2, self.params[
                                                    'init_or_no_access_handwashing_per_lower_wealth'])
                                                .when(3, self.params[
                                                    'init_or_no_access_handwashing_per_lower_wealth'] ** 2)
                                                .when(4, self.params[
                                                    'init_or_no_access_handwashing_per_lower_wealth'] ** 3)
                                                .when(5, self.params[
                                                    'init_or_no_access_handwashing_per_lower_wealth'] ** 4)
                                                )

        # return no access hand-washing linear model
        return no_access_hand_washing_lm

    def salt_intake_linear_model(self) -> LinearModel:
        """"A function to create a linear model for salt intake property. Here, we are using logistic linear model
        and looking at individual's probability of high salt intake based on whether they are rural or urban """

        # get a baseline odds for salt intake
        odds_high_salt: float = self.params['init_p_high_salt_urban'] / (1 -
                                                                         self.params['init_p_high_salt_urban'])

        # create salt intake linear model
        high_salt_lm = LinearModel(LinearModelType.LOGISTIC,
                                   odds_high_salt,
                                   Predictor('li_urban').when(False, self.params[
                                       'init_or_high_salt_rural'])
                                   )
        # return salt intake linear model
        return high_salt_lm

    def sugar_intake_linear_model(self) -> LinearModel:
        """ a function to create linear model for sugar intake property. Here we are using additive linear model and
        have no predictors hence the base probability is used as the final value for all individuals """

        # set baseline probability
        base_prob = self.params['init_p_high_sugar']
        sugar_intake_lm = LinearModel(LinearModelType.ADDITIVE,
                                      base_prob
                                      )
        # return sugar intake linear model
        return sugar_intake_lm

    def bmi_linear_model(self, index, bmi_power) -> LinearModel:
        """ a function to create linear model for bmi. here we are using Logistic model and are
        looking at individual probabilities of a particular bmi level based on the following;
                1.  sex
                2.  age group
                3.  sugar intake
                3.  tobacco usage
                4.  rural or urban
                5.  wealth level    """

        # get bmi baseline
        init_odds_bmi_urban_m_not_high_sugar_age1529_not_tob_wealth1: List[float] = [
            i / (1 - i) for i in self.params['init_p_bmi_urban_m_not_high_sugar_age1529_not_tob_wealth1']
        ]

        # create bmi linear model
        bmi_lm = LinearModel(LinearModelType.LOGISTIC,
                             init_odds_bmi_urban_m_not_high_sugar_age1529_not_tob_wealth1[index],
                             Predictor('sex').when('F', self.params['init_or_higher_bmi_f'] ** bmi_power),
                             Predictor('li_urban').when(False, self.params['init_or_higher_bmi_rural'] ** bmi_power),
                             Predictor('li_high_sugar').when(True,
                                                             self.params['init_or_higher_bmi_high_sugar'] ** bmi_power),
                             Predictor('age_years').when('.between(30, 50, inclusive="right")',
                                                         self.params['init_or_higher_bmi_age3049'] ** bmi_power)
                             .when('>= 50', self.params['init_or_higher_bmi_agege50'] ** bmi_power),
                             Predictor('li_tob').when(True, self.params['init_or_higher_bmi_tob'] ** bmi_power),
                             Predictor('li_wealth').when(2, (
                                 self.params['init_or_higher_bmi_per_higher_wealth_level'] ** 2) ** bmi_power)
                             .when(3, (self.params['init_or_higher_bmi_per_higher_wealth_level'] ** 3) ** bmi_power)
                             .when(4, (self.params['init_or_higher_bmi_per_higher_wealth_level'] ** 4) ** bmi_power)
                             .when(5, (self.params['init_or_higher_bmi_per_higher_wealth_level'] ** 5) ** bmi_power)
                             )

        return bmi_lm

    def female_sex_workers(self) -> LinearModel:
        """ a function that is used to initialise and update the sex workers property in the population dataframe. it
        predicts the number of women who will be sex workers at both population initialisation and in Lifestyle event.
        Here we are using custom linear model """

        def predict_female_sex_workers(self, df, rng=None, **externals) -> pd.Series:
            """Determine which women will be sex workers.
            This is called by initialise_population and the LifestyleEvent.
            Subject to the constraints:
            (i) Women who are in sex work may stop and there is a proportion that stop during each year.
            (ii) New women are 'recruited' to start sex work such that the overall proportion of women who are
                sex workers does not fall below a given level.

            :param df: population dataframe
            :param rng: random number generator
            :param externals: any other variable passed on to this function """

            p = self.parameters
            # create a series that will return all female sexworkers
            f_sex_worker = df.li_is_sexworker.copy()

            # Select current sex workers to stop being a sex worker
            sw_idx = df.loc[df.is_alive & df.li_is_sexworker].index
            proportion_that_will_stop_being_sexworker = p['fsw_transition'] * externals['months_since_last_poll'] / 12
            will_stop = sw_idx[rng.rand(len(sw_idx)) < proportion_that_will_stop_being_sexworker]
            f_sex_worker.loc[will_stop] = False

            # Select women to start sex worker (irrespective of any characteristic)
            # eligible to become a sex worker: alive, unmarried, women aged 15-49 who are not currently sex worker
            eligible_idx = df.loc[df.is_alive & (df.sex == 'F') & ~df.li_is_sexworker & df.age_years.between(15, 49)
                                  & (df.li_mar_stat != 2)].index

            n_sw = len(df.loc[df.is_alive & df.li_is_sexworker].index)
            target_n_sw = int(np.round(len(df.loc[df.is_alive & (df.sex == 'F') & df.age_years.between(15, 49)].index)
                                       * p["proportion_female_sex_workers"]
                                       )
                              )
            deficit = target_n_sw - n_sw
            if deficit > 0:
                if deficit < len(eligible_idx):
                    # randomly select women to start sex work:
                    will_start_sw_idx = rng.choice(eligible_idx, deficit, replace=False)
                else:
                    # select all eligible women to start sex work:
                    will_start_sw_idx = eligible_idx
                # make is_sexworker for selected women:
                f_sex_worker.loc[will_start_sw_idx] = True

            # return updated female sexworker series
            return f_sex_worker

        # return female sex worker linear model
        f_sex_worker_lm = LinearModel.custom(predict_female_sex_workers, parameters=self.params)
        return f_sex_worker_lm

    def male_circumcision_property_linear_model(self) -> LinearModel:
        """A function to create linear model for initialising the male circumcision property. Here, we are using
        custom linear model """

        def handle_male_circumcision_prop(self, df, rng=None, **externals) -> pd.Series:
            """ # determine the proportion of men that are circumcised at initiation
            # NB. this is determined with respect to any characteristics (eg. ethnicity or religion)
            Steps:
                1.  accept a population dataframe from the predict method of a custom linear model
                2.  create a new series using `li_is_circ` column data in the population dataframe
                3.  update the series using different probabilities
                4.  return the updated series

            :param self: a reference to the calling custom linear model
            :param df: population dataframe
            :param rng: random number generator
            :param  externals a dict containing any other variables passed on to this function """

            # get module parameters
            p = self.parameters

            # create a new male circumcision series
            male_circ = pd.Series(data=False, index=df.index, dtype=bool)

            # select a population of men to be circumcised
            men = df.loc[df.is_alive & (df.sex == 'M')]
            will_be_circ = rng.rand(len(men)) < p['proportion_of_men_circumcised_at_initiation']
            male_circ.loc[men[will_be_circ].index] = True

            # return male circumcision series
            return male_circ

        # return male circumcision linear model
        male_circ_lm = LinearModel.custom(handle_male_circumcision_prop, parameters=self.params)
        return male_circ_lm

    # --------------------- LINEAR MODELS FOR UPDATING POPULATION PROPERTIES ------------------------------ #
    # todo: make exposed to campaign `_property` reflect index of individuals who have transitioned

    def update_rural_urban_property_linear_model(self) -> LinearModel:
        """A function to create linear model for updating the rural urban status of an individual. Here, we are using
        custom linear model """

        def handle_rural_urban_transitions(self, df, rng=None, **externals) -> pd.Series:
            """ a function that will return a series containing rural urban transitions in individuals
            Steps:
                1.  accept a population dataframe from the predict method of a custom linear model
                2.  create a new series using `li_urban` column data in the population dataframe
                3.  update the series using different rural urban probabilities
                4.  return the updated series

            :param self: a reference to the calling custom linear model
            :param df: population dataframe
            :param rng: random number generator
            :param  externals: a dict containing any other variables passed on to this function """

            # get module parameters
            p = self.parameters

            # create a new series that will hold all rural urban transitions
            rural_urban_trans = df.li_urban.copy()

            # get index of current urban/rural status
            currently_rural = df.index[~df.li_urban & df.is_alive]
            currently_urban = df.index[df.li_urban & df.is_alive]

            # handle new transitions
            rural_to_urban = currently_rural[rng.random_sample(size=len(currently_rural)) < p['r_urban']]
            rural_urban_trans.loc[rural_to_urban] = True

            # handle new transitions to rural
            urban_to_rural = currently_urban[rng.random_sample(size=len(currently_urban)) < p['r_rural']]
            rural_urban_trans.loc[urban_to_rural] = False

            # return updated rural urban series
            return rural_urban_trans

        # return a rural urban linear model
        rural_urban_lm = LinearModel.custom(handle_rural_urban_transitions, parameters=self.params)
        return rural_urban_lm

    def update_exercise_property_linear_model(self) -> LinearModel:
        """ A function to create linear model for updating the exercise property. Here we are using custom
        linear model and are looking at rate of transitions from low exercise to not low exercise and vice versa """

        def handle_low_exercise_transitions(self, df, rng=None, **externals) -> pd.Series:
            """ a function that will return a series containing low exercise transitions in individuals
            Steps:
                1.  accept a population dataframe from the predict method of a custom linear model
                2.  create a new series using low excerise column data in the population dataframe
                3.  update the series using diferent low exercise probabilities
                4.  return the updated series

            :param self: a reference to the calling custom linear model
            :param df: population dataframe
            :param rng: random number generator
            :param externals: an external variable passed on to this function """

            # get module parameters
            p = self.parameters

            # create a new series that will hold all low exercise transitions
            low_ex_trans = df.li_low_ex.copy()

            # transition new individuals from not low exercise to low exercise
            adults_not_low_ex = df.index[~df.li_low_ex & df.is_alive & (df.age_years >= 15)]
            eff_p_low_ex = pd.Series(data=p['r_low_ex'], index=adults_not_low_ex, dtype=float)
            eff_p_low_ex.loc[df.sex == 'F'] *= p['rr_low_ex_f']
            eff_p_low_ex.loc[df.li_urban] *= p['rr_low_ex_urban']
            low_ex_trans.loc[adults_not_low_ex] = rng.random_sample(len(adults_not_low_ex)) < eff_p_low_ex

            # transition from low exercise to not low exercise
            low_ex_idx = df.index[df.li_low_ex & df.is_alive]
            eff_rate_not_low_ex = pd.Series(data=p['r_not_low_ex'], index=low_ex_idx, dtype=float)
            eff_rate_not_low_ex.loc[df.li_exposed_to_campaign_exercise_increase] *= (
                p['rr_not_low_ex_pop_advice_exercise']
            )
            random_draw = rng.random_sample(len(low_ex_idx))
            newly_not_low_ex_idx = low_ex_idx[random_draw < eff_rate_not_low_ex]
            low_ex_trans.loc[newly_not_low_ex_idx] = False

            # setting below properties direclty here. perhaps I can do beter?
            all_idx_campaign_exercise_increase = df.index[
                df.is_alive & (externals['other'] == p['start_date_campaign_exercise_increase'])
                ]
            df.loc[all_idx_campaign_exercise_increase, 'li_exposed_to_campaign_exercise_increase'] = True

            # return an updated low exercise serie
            return low_ex_trans

        # create and return a custom linear model
        low_ex_lm = LinearModel.custom(handle_low_exercise_transitions, parameters=self.params)
        return low_ex_lm

    def update_tobacco_use_property_linear_model(self) -> LinearModel:
        """A function to create linear model for tobacco use property. Here we are using custom linear model
        and are looking at transitions from not tobacco use to tobacco use and vice versa"""

        def handle_all_tob_tansitions(self, df, rng=None, **externals) -> pd.Series:
            """ a function that will return a serie containg tobacco transitions in individuals
            Steps:
                1.  accept a population dataframe from the predict method of a custom linear model
                2.  create a new series using `li_tob` column data in the population dataframe
                3.  update the series using different tobacco use properties
                4.  return the updated series

            :param self: a reference to the calling custom linear model
            :param df: population dataframe
            :param rng: random number generator
            :param externals: any external variable passed on to this function """
            # get module parameters
            p = self.parameters

            # create a new serie that will hold all tobacco use transitions
            tob_trans = df.li_tob.copy()

            # select indiviuals 15+ who are not smoking tobacco
            adults_not_tob = df.index[(df.age_years >= 15) & df.is_alive & ~df.li_tob]

            # start tobacco use
            eff_p_tob = pd.Series(data=p['r_tob'], index=adults_not_tob, dtype=float)
            eff_p_tob.loc[(df.sex == 'M') & (df.age_years >= 20) & (df.age_years < 40)] *= p['rr_tob_age2039']
            eff_p_tob.loc[(df.sex == 'M') & df.age_years >= 40] *= p['rr_tob_agege40']
            eff_p_tob.loc[df.sex == 'F'] *= p['rr_tob_f']
            eff_p_tob *= p['rr_tob_wealth'] ** (pd.to_numeric(df.loc[adults_not_tob, 'li_wealth']) - 1)

            tob_trans.loc[adults_not_tob] = rng.random_sample(len(adults_not_tob)) < eff_p_tob

            # transition from tobacco to no tobacco
            tob_idx = df.index[df.li_tob & df.is_alive]
            eff_rate_not_tob = pd.Series(data=p['r_not_tob'], index=tob_idx, dtype=float)
            eff_rate_not_tob.loc[df.li_exposed_to_campaign_quit_smoking] *= (p['rr_not_tob_pop_advice_tobacco'])
            random_draw = rng.random_sample(len(tob_idx))
            newly_not_tob_idx = tob_idx[random_draw < eff_rate_not_tob]
            tob_trans.loc[newly_not_tob_idx] = False

            # setting below properties directly here. perhaps I can do beter?
            df.loc[newly_not_tob_idx, 'li_date_not_tob'] = externals['other']
            all_idx_campaign_quit_smoking = df.index[df.is_alive &
                                                     (externals['other'] == p['start_date_campaign_quit_smoking'])]
            df.loc[all_idx_campaign_quit_smoking, 'li_exposed_to_campaign_quit_smoking'] = True

            # return updated tobacco serie
            return tob_trans

        tob_lm = LinearModel.custom(handle_all_tob_tansitions, parameters=self.params)
        return tob_lm

    def update_excess_alcohol_property_linear_model(self) -> LinearModel:
        """ a function tp create linear model for excess alcohol property. Here we are using custom linear
        model and are looking at individuals transition from either excess alcohol to not excess alcohol or vice
        versa """

        def handle_excess_alcohol_transitions(self, df, rng=None, **externals) -> pd.Series:
            """ a function that will return a serie containg excess alcohol transitions in individuals
                        Steps:
                            1.  accept a population dataframe from the predict method of a custom linear model
                            2.  create a new series using `li_ex_alc` column data in the population dataframe
                            3.  update the series using different excess alcohol properties
                            4.  return the updated series. This will then be used to update `li_ex_alc` column
                                in the dataframe

                        :param self: a reference to the calling custom linear model
                        :param df: population dataframe
                        :param rng: random number generator
                        :param externals: any external variable passed on to this function """
            # get module parameters
            p = self.parameters

            # create a new serie that will hold all excess alcohol transitions
            li_ex_alc_trans = df.li_ex_alc.copy()

            # select index of individuals of different categories
            not_ex_alc_f = df.index[~df.li_ex_alc & df.is_alive & (df.sex == 'F') & (df.age_years >= 15)]
            not_ex_alc_m = df.index[~df.li_ex_alc & df.is_alive & (df.sex == 'M') & (df.age_years >= 15)]
            now_ex_alc = df.index[df.li_ex_alc & df.is_alive]

            # randomly select individuals to be excess alcohol or to stop being excess alcohol
            li_ex_alc_trans.loc[not_ex_alc_f] = (rng.random_sample(len(not_ex_alc_f))
                                                 < p['r_ex_alc'] * p['rr_ex_alc_f'])
            li_ex_alc_trans.loc[not_ex_alc_m] = rng.random_sample(len(not_ex_alc_m)) < p['r_ex_alc']
            li_ex_alc_trans.loc[now_ex_alc] = ~(rng.random_sample(len(now_ex_alc)) < p['r_not_ex_alc'])

            # transition from excess alcohol to not excess alcohol
            ex_alc_idx = df.index[df.li_ex_alc & df.is_alive]
            eff_rate_not_ex_alc = pd.Series(data=p['r_not_ex_alc'], index=ex_alc_idx, dtype=float)
            eff_rate_not_ex_alc.loc[df.li_exposed_to_campaign_alcohol_reduction] *= p[
                'rr_not_ex_alc_pop_advice_alcohol']
            random_draw = rng.random_sample(len(ex_alc_idx))
            newly_not_ex_alc_idx = ex_alc_idx[random_draw < eff_rate_not_ex_alc]
            li_ex_alc_trans.loc[newly_not_ex_alc_idx] = False

            # setting below properties direclty here. perhaps I can do beter?
            all_idx_campaign_alcohol_reduction = df.index[
                df.is_alive & (externals['other'] == p['start_date_campaign_alcohol_reduction'])
                ]
            df.loc[all_idx_campaign_alcohol_reduction, 'li_exposed_to_campaign_alcohol_reduction'] = True

            # return updated excess alcohol serie
            return li_ex_alc_trans

        # return excess alcohol linear model
        ex_alc_lm = LinearModel.custom(handle_excess_alcohol_transitions, parameters=self.params)
        return ex_alc_lm

    def update_marital_status_linear_model(self) -> LinearModel:
        """A function to create linear model for marital status property. Here we are using custom linear
        model and are looking at individuals ability to transition into different marital statuses """

        def handle_marital_status_transitions(self, df, rng=None, **externals) -> pd.Series:
            """ a function that will return a serie containing marital status transitions in individuals
                        Steps:
                            1.  accept a population dataframe from the predict method of a custom linear model
                            2.  create a new series using `li_mar_stat` column data in the population dataframe
                            3.  update the series using different marital status probabilities
                            4.  return the updated series. This will then be used to update `li_mar_stat` column
                                in the dataframe

                        :param self: a reference to the calling custom linear model
                        :param df: population dataframe
                        :param rng: random number generator
                        :param externals: any external variable passed on to this function """
            # get module parameters
            p = self.parameters

            # create a new serie that will hold all marital status transitions
            mar_stat_trans = df.li_mar_stat.copy()

            curr_never_mar = df.index[df.is_alive & df.age_years.between(15, 29) & (df.li_mar_stat == 1)]
            curr_mar = df.index[df.is_alive & (df.li_mar_stat == 2)]

            # update if now married
            now_mar = rng.random_sample(len(curr_never_mar)) < p['r_mar']
            mar_stat_trans.loc[curr_never_mar[now_mar]] = 2

            # update if now divorced/widowed
            now_div_wid = rng.random_sample(len(curr_mar)) < p['r_div_wid']
            mar_stat_trans.loc[curr_mar[now_div_wid]] = 3
            return mar_stat_trans

        # return marital status linear model
        mar_stat_lm = LinearModel.custom(handle_marital_status_transitions, parameters=self.params)
        return mar_stat_lm

    def update_education_status_linear_model(self, _property: str) -> LinearModel:
        """ a function to create a linear model for for education prperty. here we are using custom linear
        model and are looking at individuals ability to transition from different education levels

        :param _property: This should be either `li_ed_lev` or `li_in_ed` i.e. The two education properties """

        def handle_edu_transitions(self, df, rng=None, **externals) -> pd.Series:
            """ a function that will return a serie containing education transitions in individuals
                        Steps:
                            1.  accept a population dataframe from the predict method of a custom linear model
                            2.  create a new dataframe using data from two columns(`li_ed_lev` and `li_in_ed`)
                                in the population dataframe
                            3.  update the dataframe using different education probabilities
                            4.  return the updated column based on what argument is provided in
                                `update_education_status_linear_model` function

                        :param self: a reference to the calling custom linear model
                        :param df: population dataframe
                        :param rng: random number generator
                        :param externals: any external variable passed on to this function """
            # get module parameters
            p = self.parameters

            # create a new dataframe that will hold all education transitions
            edu_trans = df[['li_in_ed', 'li_ed_lev']].copy()

            # get all individuals currently in education
            in_ed = edu_trans.index[df.is_alive & df.li_in_ed]

            # ---- PRIMARY EDUCATION
            # get index of all children who are alive and between 5 and 5.25 years old
            age5 = edu_trans.index[(df.age_exact_years >= 5) & (df.age_exact_years < 5.25) & df.is_alive]

            # by default, these children are not in education and have education level 1
            edu_trans.loc[age5, 'li_ed_lev'] = 1
            edu_trans.loc[age5, 'li_in_ed'] = False

            # create a series to hold the probablity of primary education for children at age 5
            prob_primary = pd.Series(data=p['p_ed_primary'], index=age5, dtype=float)
            prob_primary *= p['rp_ed_primary_higher_wealth'] ** (5 - pd.to_numeric(df.loc[age5, 'li_wealth']))

            # randomly select some to have primary education
            age5_in_primary = rng.random_sample(len(age5)) < prob_primary
            edu_trans.loc[age5[age5_in_primary], 'li_ed_lev'] = 2
            edu_trans.loc[age5[age5_in_primary], 'li_in_ed'] = True
            # ---- SECONDARY EDUCATION

            # get thirteen-year-olds that are in primary education, any wealth level
            age13_in_primary = df.index[(df.age_years == 13) & df.is_alive & df.li_in_ed & (df.li_ed_lev == 2)]

            # they have a probability of gaining secondary education (level 3), based on wealth
            prob_secondary = pd.Series(data=p['p_ed_secondary'], index=age13_in_primary, dtype=float)
            prob_secondary *= (p['rp_ed_secondary_higher_wealth'] **
                               (5 - pd.to_numeric(df.loc[age13_in_primary, 'li_wealth'])))

            # randomly select some to get secondary education
            age13_to_secondary = rng.random_sample(len(age13_in_primary)) < prob_secondary
            edu_trans.loc[age13_in_primary[age13_to_secondary], 'li_ed_lev'] = 3

            # those who did not go on to secondary education are no longer in education
            edu_trans.loc[age13_in_primary[~age13_to_secondary], 'li_in_ed'] = False

            # ---- DROP OUT OF EDUCATION

            # baseline rate of leaving education then adjust for wealth level
            p_leave_ed = pd.Series(data=p['r_stop_ed'], index=in_ed, dtype=float)
            p_leave_ed *= (p['rr_stop_ed_lower_wealth'] ** (pd.to_numeric(df.loc[in_ed, 'li_wealth']) - 1))

            # randomly select some individuals to leave education
            now_not_in_ed = rng.random_sample(len(in_ed)) < p_leave_ed
            edu_trans.loc[in_ed[now_not_in_ed], 'li_in_ed'] = False

            # everyone leaves education at age 20
            edu_trans.loc[df.is_alive & df.li_in_ed & (df.age_years == 20), 'li_in_ed'] = False

            # return education serie based on the argument passed to `update_education_status_linear_model` function
            return edu_trans.li_in_ed if _property == 'li_in_ed' else edu_trans.li_ed_lev

        # return education linear model
        edu_lm = LinearModel.custom(handle_edu_transitions, parameters=self.params)
        return edu_lm

    def update_unimproved_sanitation_status_linear_model(self) -> LinearModel:
        """ A function to create a linear model for updating unimproved sanitation property. here we are using
        custom linear model and are looking at individual's ability to transition from unimproved sanitation to
        improved sanitation or vice versa """

        def handle_unimproved_sanitation_transitions(self, df, rng=None, **externals) -> pd.Series:
            """ a function that will return a series containing unimproved sanitation transitions in individuals
                        Steps:
                            1.  accept a population dataframe from the predict method of a custom linear model
                            2.  create a new series using `li_unimproved_sanitation` column data in the dataframe
                            3.  update the series using different unimproved sanitation probability
                            4.  return the newely created serie. This will be used to update `li_unimproved_sanitation`
                                column in the dataframe

                        :param self: a reference to the calling custom linear model
                        :param df: population dataframe
                        :param rng: random number generator
                        :param externals: any external variable passed on to this function """
            # get module parameters
            p = self.parameters

            # create a new series that will hold all unimproved sanitation transitions
            unimproved_san_trans = df.li_unimproved_sanitation.copy()

            # probability of improved sanitation at all follow-up times
            unimproved_sanitaton_idx = df.index[df.li_unimproved_sanitation & df.is_alive]

            eff_rate_improved_sanitation = pd.Series(data=p['r_improved_sanitation'], index=unimproved_sanitaton_idx,
                                                     dtype=float)

            random_draw = rng.random_sample(len(unimproved_sanitaton_idx))

            newly_improved_sanitation_idx = unimproved_sanitaton_idx[random_draw < eff_rate_improved_sanitation]
            unimproved_san_trans.loc[newly_improved_sanitation_idx] = False

            # probability of improved sanitation upon moving to urban from rural
            unimproved_sanitation_newly_urban_idx = df.index[
                df.li_unimproved_sanitation & df.is_alive & (df.li_date_trans_to_urban == externals['other'])
                ]

            random_draw = rng.random_sample(len(unimproved_sanitation_newly_urban_idx))

            eff_prev_unimproved_sanitation_urban = pd.Series(
                data=p['init_p_unimproved_sanitation_urban'], index=unimproved_sanitation_newly_urban_idx, dtype=float
            )

            unimproved_san_trans.loc[unimproved_sanitation_newly_urban_idx] = (
                random_draw < eff_prev_unimproved_sanitation_urban
            )

            # return unimproved sanitation series
            return unimproved_san_trans

        # return unimproved sanitation linear model
        unimproved_san_lm = LinearModel.custom(handle_unimproved_sanitation_transitions, parameters=self.params)
        return unimproved_san_lm

    def update_no_access_hand_washing_status_linear_model(self) -> LinearModel:
        """ a function to create linear model for updating no access to hand washing property. Here we are using
        custom linear models and are looking at individual ability to transition from no access to hand
        washing to having access to hand washing """

        def handle_no_access_hand_washing_transitions(self, df, rng=None, **externals) -> pd.Series:
            """ a function that will return a series containing no access handwashing transitions in individuals
                        Steps:
                            1.  accept a population dataframe from the predict method of a custom linear model
                            2.  create a new series using `li_no_access_handwashing` column data in the dataframe
                            3.  update the dataframe using different no access handwashing probabilities
                            4.  return the updated no access handwashing series

                        :param self: a reference to the calling custom linear model
                        :param df: population dataframe
                        :param rng: random number generator
                        :param externals: any external variable passed on to this function """
            # get module parameters
            p = self.parameters

            # create a new dataframe that will hold all no access handwashing transitions
            trans_no_access_handwashing = df.li_no_access_handwashing.copy()

            # probability of moving to access to handwashing at all follow-up times
            no_access_handwashing_idx = df.index[df.li_no_access_handwashing & df.is_alive]

            eff_rate_access_handwashing = pd.Series(data=p['r_access_handwashing'], index=no_access_handwashing_idx,
                                                    dtype=float)

            random_draw = rng.random_sample(len(no_access_handwashing_idx))

            newly_access_handwashing_idx = no_access_handwashing_idx[random_draw < eff_rate_access_handwashing]
            trans_no_access_handwashing.loc[newly_access_handwashing_idx] = False

            # return the updated no access handwashing series
            return trans_no_access_handwashing

        # return no access handwashing linear model
        no_access_handwashing_lm = LinearModel.custom(handle_no_access_hand_washing_transitions, parameters=self.params)
        return no_access_handwashing_lm

    def update_no_clean_drinking_water_linear_model(self) -> LinearModel:
        """ a function to create linear model for updating no clean drinking water property. Here we are using
        custom linear model and are looking at individuals ability to transition from no access to clean
        drinking water to having access to drinking water """

        def handle_no_clean_drinking_water_transitions(self, df, rng=None, **externals) -> pd.Series:
            """ a function that will return a series containing no clean drinking water transitions in individuals
                        Steps:
                            1.  accept a population dataframe from the predict method of a custom linear model
                            2.  create a new series using data from column `li_no_clean_drinking_water` in the dataframe
                            3.  update the dataframe using different probabilities
                            4.  return the updated series

                        :param self: a reference to the calling custom linear model
                        :param df: population dataframe
                        :param rng: random number generator
                        :param externals: any external variable passed on to this function """
            # get module parameters
            p = self.parameters

            # create a new series that will hold all no clean drinking water transitions
            trans_no_clean_drinking_water = df.li_no_clean_drinking_water.copy()

            # probability of moving to clean drinking water at all follow-up times
            no_clean_drinking_water_idx = df.index[df.li_no_clean_drinking_water & df.is_alive]

            eff_rate_clean_drinking_water = pd.Series(data=p['r_clean_drinking_water'],
                                                      index=no_clean_drinking_water_idx, dtype=float)

            random_draw = rng.random_sample(len(no_clean_drinking_water_idx))

            newly_clean_drinking_water_idx = no_clean_drinking_water_idx[random_draw < eff_rate_clean_drinking_water]
            trans_no_clean_drinking_water.loc[newly_clean_drinking_water_idx] = False

            # probability of no clean drinking water upon moving to urban from rural
            no_clean_drinking_water_newly_urban_idx = df.index[
                df.li_no_clean_drinking_water & df.is_alive & (df.li_date_trans_to_urban == externals['other'])
                ]

            random_draw = rng.random_sample(len(no_clean_drinking_water_newly_urban_idx))

            eff_prev_no_clean_drinking_water_urban = pd.Series(
                data=p['init_p_no_clean_drinking_water_urban'], index=no_clean_drinking_water_newly_urban_idx,
                dtype=float
            )

            trans_no_clean_drinking_water.loc[no_clean_drinking_water_newly_urban_idx] = (
                random_draw < eff_prev_no_clean_drinking_water_urban
            )

            # return updated series
            return trans_no_clean_drinking_water

        # return no access clean drinking water linear model
        no_clean_drinking_water_lm = LinearModel.custom(handle_no_clean_drinking_water_transitions,
                                                        parameters=self.params)
        return no_clean_drinking_water_lm

    def update_wood_burn_stove_linear_model(self) -> LinearModel:
        """ a function to create linear model for updating wood burn stove property. Here we are using
        custom linear model and are looking at individual's ability to transition from wood burn stove to non
        wood burn stove """

        def handle_wood_burn_stove_transitions(self, df, rng=None, **externals) -> pd.Series:
            """ a function that will return a series containing wood burn stove transitions in individuals
                        Steps:
                            1.  accept a population dataframe from the predict method of a custom linear model
                            2.  create a new series using data from column `li_wood_burn_stove` in the  dataframe
                            3.  update the series using different probabilities
                            4.  return the updated series

                        :param self: a reference to the calling custom linear model
                        :param df: population dataframe
                        :param rng: random number generator
                        :param externals: any external variable passed on to this function """
            # get module parameters
            p = self.parameters

            # create a new series that will hold allwood burn stove transitions
            trans_wood_burn_stove = df.li_wood_burn_stove.copy()

            # probability of moving to non wood burn stove at all follow-up times
            wood_burn_stove_idx = df.index[df.li_wood_burn_stove & df.is_alive]

            eff_rate_non_wood_burn_stove = pd.Series(data=p['r_non_wood_burn_stove'], index=wood_burn_stove_idx,
                                                     dtype=float)

            random_draw = rng.random_sample(len(wood_burn_stove_idx))

            newly_non_wood_burn_stove_idx = wood_burn_stove_idx[random_draw < eff_rate_non_wood_burn_stove]
            trans_wood_burn_stove.loc[newly_non_wood_burn_stove_idx] = False

            # probability of moving to wood burn stove upon moving to urban from rural
            wood_burn_stove_newly_urban_idx = df.index[
                df.li_wood_burn_stove & df.is_alive & (df.li_date_trans_to_urban == externals['other'])
                ]

            random_draw = rng.random_sample(len(wood_burn_stove_newly_urban_idx))

            eff_prev_wood_burn_stove_urban = pd.Series(
                data=p['init_p_wood_burn_stove_urban'], index=wood_burn_stove_newly_urban_idx, dtype=float
            )

            trans_wood_burn_stove.loc[wood_burn_stove_newly_urban_idx] = \
                random_draw < eff_prev_wood_burn_stove_urban

            # return the preferred column in the dataframe
            return trans_wood_burn_stove

        # return no access clean drinking water linear model
        wood_burn_stove_lm = LinearModel.custom(handle_wood_burn_stove_transitions, parameters=self.params)
        return wood_burn_stove_lm

    def update_high_salt_property_linear_model(self) -> LinearModel:
        """ a function to create linear model for updating high salt property. here we are using custom
        linear model and are looking at individuals ability to transition from high salt to not high salt """

        def handle_high_salt_transitions(self, df, rng=None, **externals) -> pd.Series:
            """ a function that will return a series containing high salt transitions in individuals
                        Steps:
                            1.  accept a population dataframe from the predict method of a custom linear model
                            2.  create a new series using `li_high_salt` column data in the population dataframe
                            3.  update the series using different probabilities
                            4.  return the updated series

                        :param self: a reference to the calling custom linear model
                        :param df: population dataframe
                        :param rng: random number generator
                        :param externals: any external variable passed on to this function """
            # get module parameters
            p = self.parameters

            # create a new series that will hold all high salt transitions
            trans_high_salt = df.li_high_salt.copy()

            # select individuals who are not high salt and make some to be high salt
            not_high_salt_idx = df.index[~df.li_high_salt & df.is_alive]
            eff_rate_high_salt = pd.Series(data=p['r_high_salt_urban'], index=not_high_salt_idx, dtype=float)
            eff_rate_high_salt[df.li_urban] *= p['rr_high_salt_rural']
            random_draw = rng.random_sample(len(not_high_salt_idx))
            newly_high_salt = random_draw < eff_rate_high_salt
            newly_high_salt_idx = not_high_salt_idx[newly_high_salt]
            trans_high_salt.loc[newly_high_salt_idx] = True

            # transition from high salt to not high salt
            high_salt_idx = df.index[df.li_high_salt & df.is_alive]
            eff_rate_not_high_salt = pd.Series(data=p['r_not_high_salt'], index=high_salt_idx, dtype=float)
            eff_rate_not_high_salt.loc[df.li_exposed_to_campaign_salt_reduction] *= \
                p['rr_not_high_salt_pop_advice_salt']
            random_draw = rng.random_sample(len(high_salt_idx))
            newly_not_high_salt_idx = high_salt_idx[random_draw < eff_rate_not_high_salt]
            trans_high_salt.loc[newly_not_high_salt_idx] = False

            # setting `li_exposed_to_campaign_salt_reduction` directly here
            all_idx_campaign_salt_reduction = df.index[df.is_alive & (externals['other'] == Date(2010, 7, 1))]
            df.loc[all_idx_campaign_salt_reduction, 'li_exposed_to_campaign_salt_reduction'] = True

            # return high salt transitions series
            return trans_high_salt

        # return high salt linear model
        high_salt_lm = LinearModel.custom(handle_high_salt_transitions, parameters=self.params)
        return high_salt_lm

    def update_high_sugar_property_linear_model(self) -> LinearModel:
        """ a function to create linear model for updating high sugar property. Here we are using multiplicative
        linear model and are looking at individuals ability to transition from high sugar to not high sugar """

        def handle_high_sugar_transitions(self, df, rng=None, **externals) -> pd.Series:
            """ a function that will return a series containing high sugar transitions in individuals
                       Steps:
                           1.  accept a population dataframe from the predict method of a custom linear model
                           2.  create a new series using `li_high_sugar` column data in the population dataframe
                           3.  update the series using different probabilities
                           4.  return the updated series

                       :param self: a reference to the calling custom linear model
                       :param df: population dataframe
                       :param rng: random number generator
                       :param externals: any external variable passed on to this function """
            # get module parameters
            p = self.parameters

            # create a new series that will hold all high sugar transitions
            trans_high_sugar = df.li_high_sugar.copy()

            not_high_sugar_idx = df.index[~df.li_high_sugar & df.is_alive]
            eff_p_high_sugar = pd.Series(data=p['r_high_sugar'], index=not_high_sugar_idx, dtype=float)
            random_draw = rng.random_sample(len(not_high_sugar_idx))
            newly_high_sugar_idx = not_high_sugar_idx[random_draw < eff_p_high_sugar]
            trans_high_sugar.loc[newly_high_sugar_idx] = True

            # transition from high sugar to not high sugar
            high_sugar_idx = df.index[df.li_high_sugar & df.is_alive]
            eff_rate_not_high_sugar = pd.Series(data=p['r_not_high_sugar'], index=high_sugar_idx, dtype=float)
            eff_rate_not_high_sugar.loc[
                df.li_exposed_to_campaign_sugar_reduction
            ] *= p['rr_not_high_sugar_pop_advice_sugar']
            random_draw = rng.random_sample(len(high_sugar_idx))
            newly_not_high_sugar_idx = high_sugar_idx[random_draw < eff_rate_not_high_sugar]
            trans_high_sugar.loc[newly_not_high_sugar_idx] = False

            # make exposed to campaing sugar reduction reflect index of individuals who have transition
            # from high sugar to not high sugar. setting `li_exposed_to_campaign_sugar_reduction` directly here,
            all_idx_campaign_sugar_reduction = df.index[df.is_alive & (externals['other'] == Date(2010, 7, 1))]
            df.loc[all_idx_campaign_sugar_reduction, 'li_exposed_to_campaign_sugar_reduction'] = True

            # return series containing high sugar transitions
            return trans_high_sugar

        # return high sugar linear model
        high_sugar_lm = LinearModel.custom(handle_high_sugar_transitions, parameters=self.params)
        return high_sugar_lm

    def update_bmi_categories_linear_model(self) -> LinearModel:
        """ a function to create linear model for updating bmi categories. here we are using multiplicative linear
        model and are looking at individual's ability to transition from different bmi categories """

        def handle_bmi_transitions(self, df, rng=None, **externals) -> pd.Series:
            """ a function that will return a serie containing bmi transitions in individuals
                        Steps:
                            1.  accept a population dataframe from the predict method of a custom linear model
                            2.  create a new series using `li_bmi` column data in the population dataframe
                            3.  update the series using different bmi probabilities
                            4.  return the updated series

                        :param self: a reference to the calling custom linear model
                        :param df: population dataframe
                        :param rng: random number generator
                        :param externals: any external variable passed on to this function """
            # get module parameters
            p = self.parameters

            # create a new series that will hold all bmi transitions
            trans_bmi = pd.Series(data=df.li_bmi.copy(), index=df.index, dtype=float)

            # those reaching age 15 allocated bmi 2
            age15_idx = df.index[df.is_alive & (df.age_exact_years >= 15) & (df.age_exact_years < 15.25)]
            trans_bmi.loc[age15_idx] = 2

            # possible increase in category of bmi
            bmi_cat_1_to_4_idx = df.index[df.is_alive & (df.age_years >= 15) & df.li_bmi.isin(range(1, 5))]
            eff_rate_higher_bmi = pd.Series(data=p['r_higher_bmi'], index=bmi_cat_1_to_4_idx, dtype=float)
            eff_rate_higher_bmi[df.li_urban] *= p['rr_higher_bmi_urban']
            eff_rate_higher_bmi[df.sex == 'F'] *= p['rr_higher_bmi_f']
            eff_rate_higher_bmi[df.age_years.between(30, 49)] *= p['rr_higher_bmi_age3049']
            eff_rate_higher_bmi[df.age_years >= 50] *= p['rr_higher_bmi_agege50']
            eff_rate_higher_bmi[df.li_tob] *= p['rr_higher_bmi_tob']
            eff_rate_higher_bmi[df.li_wealth == 2] *= p['rr_higher_bmi_per_higher_wealth'] ** 2
            eff_rate_higher_bmi[df.li_wealth == 3] *= p['rr_higher_bmi_per_higher_wealth'] ** 3
            eff_rate_higher_bmi[df.li_wealth == 4] *= p['rr_higher_bmi_per_higher_wealth'] ** 4
            eff_rate_higher_bmi[df.li_wealth == 5] *= p['rr_higher_bmi_per_higher_wealth'] ** 5
            eff_rate_higher_bmi[df.li_high_sugar] *= p['rr_higher_bmi_high_sugar']

            random_draw = rng.random_sample(len(bmi_cat_1_to_4_idx))
            newly_increase_bmi_cat_idx = bmi_cat_1_to_4_idx[random_draw < eff_rate_higher_bmi]
            # increase bmi
            trans_bmi.loc[newly_increase_bmi_cat_idx] += 1
            trans_bmi.loc[newly_increase_bmi_cat_idx].fillna(1)

            # possible decrease in category of bmi
            bmi_cat_3_to_5_idx = df.index[df.is_alive & df.li_bmi.isin(range(3, 6)) & (df.age_years >= 15)]
            eff_rate_lower_bmi = pd.Series(data=p['r_lower_bmi'], index=bmi_cat_3_to_5_idx, dtype=float)
            eff_rate_lower_bmi[df.li_urban] *= p['rr_lower_bmi_tob']
            eff_rate_lower_bmi.loc[df.li_exposed_to_campaign_weight_reduction] *= p['rr_lower_bmi_pop_advice_weight']
            random_draw = rng.random_sample(len(bmi_cat_3_to_5_idx))
            newly_decrease_bmi_cat_idx = bmi_cat_3_to_5_idx[random_draw < eff_rate_lower_bmi]
            # decrease bmi
            trans_bmi.loc[newly_decrease_bmi_cat_idx] -= 1

            # make exposed to campaing weigh reduction reflect individuals who have decreased their bmi category.
            # setting `li_exposed_to_campaign_weight_reduction` directly here,
            all_idx_campaign_weight_reduction = df.index[df.is_alive & (externals['other'] == Date(2010, 7, 1)
                                                                        )]
            df.loc[all_idx_campaign_weight_reduction, 'li_exposed_to_campaign_weight_reduction'] = True

            # return updated bmi categories
            return trans_bmi

        # return bmi categories linear model
        bmi_lm = LinearModel.custom(handle_bmi_transitions, parameters=self.params)
        return bmi_lm


class LifestyleEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all lifestyle properties for population
    """

    def __init__(self, module):
        """schedule to run every 3 months
        note: if change this offset from 3 months need to consider code conditioning on age.years_exact
        :param module: the module that created this event
        """
        self.repeat_months = 3
        self.module = module
        super().__init__(module, frequency=DateOffset(months=self.repeat_months))
        assert isinstance(module, Lifestyle)

    def apply(self, population):
        """ Apply this event to the population.
        :param population: the current population
        """
        df = self.module.sim.population.props
        # update all properties on Lifestyle Event run
        self.module.models.update_all_properties(df)


class LifestylesLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """Handles lifestyle logging"""

    def __init__(self, module):
        """schedule logging to repeat every 3 months
        """
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, Lifestyle)

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        # get some summary statistics
        df = population.props
        all_lm_keys = self.module.models.get_lm_keys()
        # log summary of each lifestyle property
        # NB: In addition to logging properties by sex and age groups, there are some properties that requires
        # individual's urban or rural status. define and log these properties separately
        cat_by_rural_urban_props = ['li_wealth', 'li_bmi', 'li_low_ex', 'li_ex_alc', 'li_wood_burn_stove',
                                    'li_unimproved_sanitation', 'li_no_clean_drinking_water']
        # these properties are applicable to individuals 15+ years
        log_by_age_15up = ['li_low_ex', 'li_mar_stat', 'li_ex_alc', 'li_bmi', 'li_tob']

        for _property in all_lm_keys:
            if _property in log_by_age_15up:
                if _property in cat_by_rural_urban_props:
                    data = grouped_counts_with_all_combinations(
                        df.loc[df.is_alive & (df.age_years >= 15)],
                        ["li_urban", "sex", _property, "age_range"]
                    )
                else:
                    data = grouped_counts_with_all_combinations(
                        df.loc[df.is_alive & (df.age_years >= 15)],
                        ["sex", _property, "age_range"]
                    )
            elif _property == 'li_in_ed':
                data = grouped_counts_with_all_combinations(
                    df.loc[df.is_alive & df.age_years.between(5, 19)],
                    ["sex", "li_wealth", "li_in_ed", "age_years"],
                    {"age_years": range(5, 20)}
                )
            elif _property == 'li_ed_lev':
                data = grouped_counts_with_all_combinations(
                    df.loc[df.is_alive & df.age_years.between(15, 49)],
                    ["sex", "li_wealth", "li_ed_lev", "age_years"],
                    {"age_years": range(15, 50)}
                )
            elif _property == 'li_is_sexworker':
                data = grouped_counts_with_all_combinations(
                    df.loc[df.is_alive & (df.age_years.between(15, 49))],
                    ["sex", "li_is_sexworker", "age_range"],
                )
            elif _property in cat_by_rural_urban_props:
                # log all properties that are also categorised by rural or urban in addition to ex and age groups
                data = grouped_counts_with_all_combinations(
                    df.loc[df.is_alive], ["li_urban", "sex", _property, "age_range"]
                )
            else:
                # log all other remaining properties
                data = grouped_counts_with_all_combinations(
                    df.loc[df.is_alive], ["sex", _property, "age_range"]
                )
            # log data
            logger.info(
                key=_property,
                data=flatten_multi_index_series_into_dict_for_logging(data)
            )

        # ---------------------- log properties associated with WASH

        # unimproved sanitation
        # NOTE: True = no sanitation
        li_no_clean_drinking_water = len(
            df[df.li_unimproved_sanitation & df.is_alive & (df.age_years < 5)]
        ) / len(df[df.is_alive & (df.age_years < 5)]
                ) if len(df[df.is_alive & (df.age_years < 5)]) else 0

        no_sanitation_SAC = len(
            df[df.li_unimproved_sanitation & df.is_alive & df.age_years.between(5, 15)]
        ) / len(df[df.is_alive & df.age_years.between(5, 15)]) if len(
            df[df.is_alive & df.age_years.between(5, 15)]) else 0

        no_sanitation_ALL = len(
            df[df.li_unimproved_sanitation & df.is_alive]
        ) / len(df[df.is_alive]
                ) if len(df[df.is_alive]) else 0


        # no access hand-washing
        # NOTE: True = no access hand-washing
        no_handwashing_PSAC = len(
            df[df.li_no_access_handwashing & df.is_alive & (df.age_years < 5)]
        ) / len(df[df.is_alive & (df.age_years < 5)]
                ) if len(df[df.is_alive & (df.age_years < 5)]) else 0

        no_handwashing_SAC = len(
            df[df.li_no_access_handwashing & df.is_alive & df.age_years.between(5, 15)]
        ) / len(df[df.is_alive & df.age_years.between(5, 15)]) if len(
            df[df.is_alive & df.age_years.between(5, 15)]) else 0

        no_handwashing_ALL = len(
            df[df.li_no_access_handwashing & df.is_alive]
        ) / len(df[df.is_alive]
                ) if len(df[df.is_alive]) else 0


        # no clean drinking water
        # NOTE: True = no clean drinking water
        no_drinkingwater_PSAC = len(
            df[df.li_no_clean_drinking_water & df.is_alive & (df.age_years < 5)]
        ) / len(df[df.is_alive & (df.age_years < 5)]
                ) if len(df[df.is_alive & (df.age_years < 5)]) else 0

        no_drinkingwater_SAC = len(
            df[df.li_no_clean_drinking_water & df.is_alive & df.age_years.between(5, 15)]
        ) / len(df[df.is_alive & df.age_years.between(5, 15)]) if len(
            df[df.is_alive & df.age_years.between(5, 15)]) else 0

        no_drinkingwater_ALL = len(
            df[df.li_no_clean_drinking_water & df.is_alive]
        ) / len(df[df.is_alive]
                ) if len(df[df.is_alive]) else 0

        logger.info(
            key="summary_WASH_properties",
            description="Summary of current status of WASH properties",
            data={
                "no_sanitation_PSAC": li_no_clean_drinking_water,
                "no_sanitation_SAC": no_sanitation_SAC,
                "no_sanitation_ALL": no_sanitation_ALL,
                "no_handwashing_PSAC": no_handwashing_PSAC,
                "no_handwashing_SAC": no_handwashing_SAC,
                "no_handwashing_ALL": no_handwashing_ALL,
                "no_drinkingwater_PSAC": no_drinkingwater_PSAC,
                "no_drinkingwater_SAC": no_drinkingwater_SAC,
                "no_drinkingwater_ALL": no_drinkingwater_ALL,
            },
        )

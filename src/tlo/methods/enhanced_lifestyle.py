"""
Lifestyle module
Documentation: 04 - Methods Repository/Method_Lifestyle.xlsx
"""
import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# todo: Note: bmi category at turning age 15 needs to be made dependent on malnutrition in childhood when that
# todo: module is coded.


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
            Types.REAL, 'initital proportion of 15-19 year old men using ' 'tobacco, wealth level 1 '
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
            Types.REAL, 'proportions of low wealth 20-30 year olds with some ' 'education at baseline'
        ),
        'init_or_some_ed_age0513': Parameter(Types.REAL, 'odds ratio of some education at baseline age 5-13'),
        'init_or_some_ed_age1320': Parameter(Types.REAL, 'odds ratio of some education at baseline age 13-20'),
        'init_or_some_ed_age2030': Parameter(Types.REAL, 'odds ratio of some education at baseline age 20-30'),
        'init_or_some_ed_age3040': Parameter(Types.REAL, 'odds ratio of some education at baseline age 30-40'),
        'init_or_some_ed_age4050': Parameter(Types.REAL, 'odds ratio of some education at baseline age 40-50'),
        'init_or_some_ed_age5060': Parameter(Types.REAL, 'odds ratio of some education at baseline age 50-60'),
        'init_or_some_ed_per_higher_wealth': Parameter(
            Types.REAL, 'odds ratio of some education at baseline ' 'per higher wealth level'
        ),
        'init_prop_age2030_w5_some_ed_sec': Parameter(
            Types.REAL,
            'proportion of low wealth aged 20-30 with some education who ' 'have secondary education at baseline',
        ),
        'init_or_some_ed_sec_age1320': Parameter(Types.REAL, 'odds ratio of secondary education age 13-20'),
        'init_or_some_ed_sec_age3040': Parameter(Types.REAL, 'odds ratio of secondary education age 30-40'),
        'init_or_some_ed_sec_age4050': Parameter(Types.REAL, 'odds ratio of secondary education age 40-50'),
        'init_or_some_ed_sec_age5060': Parameter(Types.REAL, 'odds ratio of secondary education age 50-60'),
        'init_or_some_ed_sec_agege60': Parameter(Types.REAL, 'odds ratio of secondary education age 60+'),
        'init_or_some_ed_sec_per_higher_wealth': Parameter(
            Types.REAL, 'odds ratio of secondary education ' 'per higher wealth level'
        ),
        'init_p_unimproved_sanitation_urban': Parameter(
            Types.REAL, 'initial probability of unimproved_sanitation ' 'given urban'
        ),
        # note that init_p_unimproved_sanitation is also used as the one-off probability of unimproved_sanitation '
        #                                                     'true to false upon move from rural to urban'
        'init_or_unimproved_sanitation_rural': Parameter(
            Types.REAL, 'initial odds ratio of unimproved_sanitation if ' 'rural'
        ),
        'init_p_no_clean_drinking_water_urban': Parameter(
            Types.REAL, 'initial probability of no_clean_drinking_water given urban'
        ),
        # note that init_p_no_clean_drinking_water is also used as the one-off probability of no_clean_drinking_water '
        #                                                     'true to false upon move from rural to urban'
        'init_or_no_clean_drinking_water_rural': Parameter(
            Types.REAL, 'initial odds ratio of no clean drinking_water ' 'if rural'
        ),
        'init_p_wood_burn_stove_urban': Parameter(Types.REAL, 'initial probability of wood_burn_stove given urban'),
        # note that init_p_wood_burn_stove is also used as the one-off probability of wood_burn_stove '
        #                                                     'true to false upon move from rural to urban'
        'init_or_wood_burn_stove_rural': Parameter(Types.REAL, 'initial odds ratio of wood_burn_stove if rural'),
        'init_p_no_access_handwashing_wealth1': Parameter(
            Types.REAL, 'initial probability of no_access_handwashing given wealth 1'
        ),
        'init_or_no_access_handwashing_per_lower_wealth': Parameter(
            Types.REAL, 'initial odds ratio of no_' 'access_handwashing per lower wealth ' 'level'
        ),
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
            Types.REAL, 'rate ratio for increase in bmi category per higher ' 'wealth level'
        ),
        'rr_higher_bmi_high_sugar': Parameter(
            Types.REAL, 'rate ratio for increase in bmi category for high sugar ' 'intake'
        ),
        'r_lower_bmi': Parameter(
            Types.REAL, 'probability per 3 months of decrease in bmi category in non tobacco users'
        ),
        'rr_lower_bmi_tob': Parameter(Types.REAL, 'rate ratio for lower bmi category for tobacco users'),
        'rr_lower_bmi_pop_advice_weight': Parameter(
            Types.REAL,
            'probability per 3 months of decrease in bmi category ' 'given population advice/campaign on weight',
        ),
        'r_high_salt_urban': Parameter(Types.REAL, 'probability per 3 months of high salt intake if urban'),
        'rr_high_salt_rural': Parameter(Types.REAL, 'rate ratio for high salt if rural'),
        'r_not_high_salt': Parameter(Types.REAL, 'probability per 3 months of not high salt intake'),
        'rr_not_high_salt_pop_advice_salt': Parameter(
            Types.REAL, 'probability per 3 months of not high salt given' 'population advice/campaign on salt'
        ),
        'r_high_sugar': Parameter(Types.REAL, 'probability per 3 months of high sugar intake'),
        'r_not_high_sugar': Parameter(Types.REAL, 'probability per 3 months of not high sugar intake'),
        'rr_not_high_sugar_pop_advice_sugar': Parameter(
            Types.REAL, 'probability per 3 months of not high sugar given' 'population advice/campaign on sugar'
        ),
        'r_low_ex': Parameter(Types.REAL, 'probability per 3 months of change from not low exercise to low exercise'),
        'r_not_low_ex': Parameter(
            Types.REAL, 'probability per 3 months of change from low exercise to not low exercie'
        ),
        'rr_not_low_ex_pop_advice_exercise': Parameter(
            Types.REAL, 'probability per 3 months of not low exercise' 'population advice/campaign on exercise'
        ),
        'rr_low_ex_f': Parameter(Types.REAL, 'risk ratio for becoming low exercise if female rather than male'),
        'rr_low_ex_urban': Parameter(Types.REAL, 'risk ratio for becoming low exercise if urban rather than rural'),
        'r_tob': Parameter(
            Types.REAL,
            'probability per 3 months of change from not using tobacco to using '
            'tobacco if male age 15-19 wealth level 1',
        ),
        'r_not_tob': Parameter(
            Types.REAL, 'probability per 3 months of change from tobacco using to ' 'not tobacco using'
        ),
        'rr_tob_f': Parameter(Types.REAL, 'rate ratio tobacco if female'),
        'rr_tob_age2039': Parameter(Types.REAL, 'risk ratio for tobacco using if age 20-39 compared with 15-19'),
        'rr_tob_agege40': Parameter(Types.REAL, 'risk ratio for tobacco using if age >= 40 compared with 15-19'),
        'rr_tob_wealth': Parameter(
            Types.REAL, 'risk ratio for tobacco using per 1 higher wealth level ' '(higher wealth level = lower wealth)'
        ),
        'rr_not_tob_pop_advice_tobacco': Parameter(
            Types.REAL, 'probability per 3 months of quitting tobacco given' 'population advice/campaign on tobacco'
        ),
        'r_ex_alc': Parameter(
            Types.REAL, 'probability per 3 months of change from not excess alcohol to ' 'excess alcohol'
        ),
        'r_not_ex_alc': Parameter(
            Types.REAL, 'probability per 3 months of change from excess alcohol to ' 'not excess alcohol'
        ),
        'rr_ex_alc_f': Parameter(Types.REAL, 'risk ratio for becoming excess alcohol if female rather than male'),
        'rr_not_ex_alc_pop_advice_alcohol': Parameter(
            Types.REAL, 'probability per 3 months of not excess alcohol given' 'population advice/campaign on alcohol'
        ),
        'r_mar': Parameter(Types.REAL, 'probability per 3 months of marriage when age 15-30'),
        'r_div_wid': Parameter(
            Types.REAL, 'probability per 3 months of becoming divorced or widowed, ' 'amongst those married'
        ),
        'r_stop_ed': Parameter(Types.REAL, 'probabilities per 3 months of stopping education if wealth level 5'),
        'rr_stop_ed_lower_wealth': Parameter(
            Types.REAL, 'relative rate of stopping education per ' '1 lower wealth quintile'
        ),
        'p_ed_primary': Parameter(Types.REAL, 'probability at age 5 that start primary education if wealth level 5'),
        'rp_ed_primary_higher_wealth': Parameter(
            Types.REAL, 'relative probability of starting school per 1 ' 'higher wealth level'
        ),
        'p_ed_secondary': Parameter(
            Types.REAL,
            'probability at age 13 that start secondary education at 13 ' 'if in primary education and wealth level 5',
        ),
        'rp_ed_secondary_higher_wealth': Parameter(
            Types.REAL, 'relative probability of starting secondary ' 'school per 1 higher wealth level'
        ),
        'r_improved_sanitation': Parameter(
            Types.REAL, 'probability per 3 months of change from ' 'unimproved_sanitation true to false'
        ),
        'r_clean_drinking_water': Parameter(
            Types.REAL, 'probability per 3 months of change from ' 'drinking_water true to false'
        ),
        'r_non_wood_burn_stove': Parameter(
            Types.REAL, 'probability per 3 months of change from ' 'wood_burn_stove true to false'
        ),
        'r_access_handwashing': Parameter(
            Types.REAL, 'probability per 3 months of change from ' 'no_access_handwashing true to false'
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
        # todo make BMI property categorical
        'li_bmi': Property(
            Types.INT, 'bmi: 1 (<18) 2 (18-24.9)  3 (25-29.9) 4 (30-34.9) 5 (35+)' 'bmi is 0 until age 15'
        ),
        'li_exposed_to_campaign_weight_reduction': Property(
            Types.BOOL, 'currently exposed to population campaign for ' 'weight reduction if BMI >= 25'
        ),
        'li_low_ex': Property(Types.BOOL, 'currently low exercise'),
        'li_exposed_to_campaign_exercise_increase': Property(
            Types.BOOL, 'currently exposed to population campaign for ' 'increase exercise if low ex'
        ),
        'li_high_salt': Property(Types.BOOL, 'currently high salt intake'),
        'li_exposed_to_campaign_salt_reduction': Property(
            Types.BOOL, 'currently exposed to population campaign for ' 'salt reduction if high salt'
        ),
        'li_high_sugar': Property(Types.BOOL, 'currently high sugar intake'),
        'li_exposed_to_campaign_sugar_reduction': Property(
            Types.BOOL, 'currently exposed to population campaign for ' 'sugar reduction if high sugar'
        ),
        'li_tob': Property(Types.BOOL, 'current using tobacco'),
        'li_date_not_tob': Property(Types.DATE, 'date last transitioned from tob to not tob'),
        'li_exposed_to_campaign_quit_smoking': Property(
            Types.BOOL, 'currently exposed to population campaign to' 'quit smoking if tob'
        ),
        'li_ex_alc': Property(Types.BOOL, 'current excess alcohol'),
        'li_exposed_to_campaign_alcohol_reduction': Property(
            Types.BOOL, 'currently exposed to population campaign for ' 'alcohol reduction if ex alc'
        ),
        'li_mar_stat': Property(
            Types.CATEGORICAL, 'marital status {1:never, 2:current, 3:past (widowed or divorced)}', categories=[1, 2, 3]
        ),
        'li_in_ed': Property(Types.BOOL, 'currently in education'),
        'li_ed_lev': Property(Types.CATEGORICAL, 'education level achieved as of now', categories=[1, 2, 3]),
        'li_unimproved_sanitation': Property(
            Types.BOOL, 'uninproved sanitation - anything other than own or ' 'shared latrine'
        ),
        'li_no_access_handwashing': Property(
            Types.BOOL, 'no_access_handwashing - no water, no soap, no other ' 'cleaning agent - as in DHS'
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
        dfd = pd.read_excel(
            Path(self.resourcefilepath) / 'ResourceFile_Lifestyle_Enhanced.xlsx', sheet_name='parameter_values'
        )

        self.load_parameters_from_dataframe(dfd)
        # Manually set dates for campaign starts for now todo - fix this
        p['start_date_campaign_exercise_increase'] = datetime.date(2010, 7, 1)
        p['start_date_campaign_quit_smoking'] = datetime.date(2010, 7, 1)
        p['start_date_campaign_alcohol_reduction'] = datetime.date(2010, 7, 1)

    def pre_initialise_population(self):
        """Initialise the linear model class"""
        self.models = LifestyleModels(self)

    def initialise_population(self, population):
        """Set our property values for the initial population.
        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals
        alive_idx = df.index[df.is_alive]

        # -------------------- DEFAULTS ------------------------------------------------------------

        df.loc[df.is_alive, 'li_urban'] = False
        df.loc[df.is_alive, 'li_wealth'] = 3
        df.loc[df.is_alive, 'li_bmi'] = 0
        df.loc[df.is_alive, 'li_exposed_to_campaign_weight_reduction'] = False
        df.loc[df.is_alive, 'li_low_ex'] = False
        df.loc[df.is_alive, 'li_exposed_to_campaign_exercise_increase'] = False
        df.loc[df.is_alive, 'li_high_salt'] = False
        df.loc[df.is_alive, 'li_exposed_to_campaign_salt_reduction'] = False
        df.loc[df.is_alive, 'li_high_sugar'] = False
        df.loc[df.is_alive, 'li_exposed_to_campaign_sugar_reduction'] = False
        df.loc[df.is_alive, 'li_tob'] = False
        df.loc[df.is_alive, 'li_date_not_tob'] = pd.NaT
        df.loc[df.is_alive, 'li_exposed_to_campaign_quit_smoking'] = False
        df.loc[df.is_alive, 'li_ex_alc'] = False
        df.loc[df.is_alive, 'li_exposed_to_campaign_alcohol_reduction'] = False
        df.loc[df.is_alive, 'li_mar_stat'] = 1
        df.loc[df.is_alive, 'li_in_ed'] = False
        df.loc[df.is_alive, 'li_ed_lev'] = 1
        df.loc[df.is_alive, 'li_unimproved_sanitation'] = True
        df.loc[df.is_alive, 'li_no_access_handwashing'] = True
        df.loc[df.is_alive, 'li_no_clean_drinking_water'] = True
        df.loc[df.is_alive, 'li_wood_burn_stove'] = True
        df.loc[df.is_alive, 'li_date_trans_to_urban'] = pd.NaT
        df.loc[df.is_alive, 'li_date_acquire_improved_sanitation'] = pd.NaT
        df.loc[df.is_alive, 'li_date_acquire_access_handwashing'] = pd.NaT
        df.loc[df.is_alive, 'li_date_acquire_clean_drinking_water'] = pd.NaT
        df.loc[df.is_alive, 'li_date_acquire_non_wood_burn_stove'] = pd.NaT
        df.loc[df.is_alive, 'li_is_sexworker'] = False
        df.loc[df.is_alive, 'li_is_circ'] = False
        # todo: express all rates per year and divide by 4 inside program

        # initialise all properties using linear models
        self.models.initialise_all_properties(df)

        # -------------------- SEX WORKER ----------------------------------------------------------
        # determine which women will be sex worker
        self.determine_who_will_be_sexworker(months_since_last_poll=0)

        # -------------------- MALE CIRCUMCISION ----------------------------------------------------------
        # determine the proportion of men that are circumcised at initiation
        # NB. this is determined with respect to any characteristics (eg. ethnicity or religion)
        men = df.loc[df.is_alive & (df.sex == 'M')]
        will_be_circ = self.rng.rand(len(men)) < self.parameters['proportion_of_men_circumcised_at_initiation']
        df.loc[men[will_be_circ].index, 'li_is_circ'] = True

    def init_edu_bmi_properties(self, lifestyle_property):
        """ a function to initialise education and bmi properties """
        if lifestyle_property == 'li_in_ed':
            self.init_education_properties()
        else:
            self.init_bmi_property()

    def init_education_properties(self):
        """ use output from education linear models to set education levels and status
        :param df: population dataframe """

        df = self.sim.population.props
        age_gte5 = df.index[(df.age_years >= 5) & df.is_alive]

        # store population eligible for education
        edu_pop = df.loc[(df.age_years >= 5) & df.is_alive]

        rnd_draw = pd.Series(self.rng.random_sample(size=len(age_gte5)), index=age_gte5)

        # make some predictions
        p_some_ed = self.models.education_linear_models()['some_edu_linear_model'].predict(edu_pop)
        p_ed_lev_3 = self.models.education_linear_models()['level_3_edu_linear_model'].predict(edu_pop)

        dfx = pd.concat([(1 - p_ed_lev_3), (1 - p_some_ed)], axis=1)
        dfx.columns = ['cut_off_ed_levl_3', 'p_ed_lev_1']

        dfx['li_ed_lev'] = 2
        dfx.loc[dfx['cut_off_ed_levl_3'] < rnd_draw, 'li_ed_lev'] = 3
        dfx.loc[dfx['p_ed_lev_1'] > rnd_draw, 'li_ed_lev'] = 1

        df.loc[age_gte5, 'li_ed_lev'] = dfx['li_ed_lev']

        df.loc[df.age_years.between(5, 12) & (df['li_ed_lev'] == 2) & df.is_alive, 'li_in_ed'] = True
        df.loc[df.age_years.between(13, 19) & (df['li_ed_lev'] == 3) & df.is_alive, 'li_in_ed'] = True

    def init_bmi_property(self):
        """ set property for BMI in population dataframe
        :param df: population dataframe """

        df = self.sim.population.props
        # get indexes of population alive and 15+ years
        age_ge15_idx = df.index[df.is_alive & (df.age_years >= 15)]
        prop_df = df.loc[df.is_alive & (df.age_years >= 15)]

        # only relevant if at least one individual with age >= 15 years present
        if len(age_ge15_idx) > 0:
            # this below is the approach to apply the effect of contributing determinants on bmi levels at baseline
            # create bmi probabilities dataframe using bmi linear model and normalise to sum to 1
            df_lm = pd.DataFrame()
            bmi_pow = [-2, -1, 0, 1, 2]

            for index_ in range(0, 5):
                df_lm[index_ + 1] = LifestyleModels(self).bmi_linear_model(index_, bmi_pow[index_]).predict(prop_df)

            dfxx = df_lm.div(df_lm.sum(axis=1), axis=0)

            # for each row, make a choice
            bmi_cat = dfxx.apply(lambda p_bmi: self.rng.choice(dfxx.columns, p=p_bmi), axis=1)

            df.loc[age_ge15_idx, 'li_bmi'] = bmi_cat

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

        # Determine id from which characteristics that inherited (from mother, or if no mother, from a randomly
        # selected person.)
        _id_inherit_from = mother_id if mother_id != -1 else self.rng.choice(df.index[df.is_alive])

        df.at[child_id, 'li_urban'] = df.at[_id_inherit_from, 'li_urban']
        df.at[child_id, 'li_wealth'] = df.at[_id_inherit_from, 'li_wealth']
        df.at[child_id, 'li_bmi'] = 0
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

    def determine_who_will_be_sexworker(self, months_since_last_poll):
        """Determine which women will be sex workers.
        This is called by initialise_population and the LifestyleEvent.
        Subject to the constraints:
        (i) Women who are in sex work may stop and there is a proportion that stop during each year.
        (ii) New women are 'recruited' to start sex work such that the overall proportion of women who are sex workers
            does not fall below a given level.
        """

        df = self.sim.population.props
        params = self.parameters
        rng = self.rng

        # Select current sex workers to stop being a sex worker
        sw_idx = df.loc[df.is_alive & df.li_is_sexworker].index
        proportion_that_will_stop_being_sexworker = params['fsw_transition'] * months_since_last_poll / 12
        will_stop = sw_idx[rng.rand(len(sw_idx)) < proportion_that_will_stop_being_sexworker]
        df.loc[will_stop, 'li_is_sexworker'] = False

        # Select women to start sex worker (irrespective of any characteristic)
        # eligible to become a sex worker: alive, unmarried, women aged 15-49 who are not currently sex worker
        eligible_idx = df.loc[
            df.is_alive &
            (df.sex == 'F') &
            ~df.li_is_sexworker &
            df.age_years.between(15, 49) &
            (df.li_mar_stat != 2)
            ].index

        n_sw = len(df.loc[df.is_alive & df.li_is_sexworker].index)
        target_n_sw = int(np.round(len(df.loc[
                                           df.is_alive &
                                           (df.sex == 'F') &
                                           df.age_years.between(15, 49)
                                           ].index) * params["proportion_female_sex_workers"]
                                   ))
        deficit = target_n_sw - n_sw
        if deficit > 0:
            if deficit < len(eligible_idx):
                # randomly select women to start sex work:
                will_start_sw_idx = rng.choice(eligible_idx, deficit, replace=False)
            else:
                # select all eligible women to start sex work:
                will_start_sw_idx = eligible_idx
            # make is_sexworker for selected women:
            df.loc[will_start_sw_idx, 'li_is_sexworker'] = True

    def compute_bmi_proportions_of_interest(self):
        """This is called by the logger and computes summary statistics about the bmi"""
        df = self.sim.population.props

        n_agege15 = (df.is_alive & (df.age_years >= 15)).sum()
        n_agege15_f = (df.is_alive & (df.age_years >= 15) & (df.sex == 'F')).sum()
        n_agege15_m = (df.is_alive & (df.age_years >= 15) & (df.sex == 'M')).sum()
        n_agege15_urban = (
            df.is_alive & (df.age_years >= 15) & df.li_urban
        ).sum()
        n_agege15_rural = (df.is_alive & (df.age_years >= 15) & ~df.li_urban).sum()
        n_agege15_wealth1 = (df.is_alive & (df.age_years >= 15) & (df.li_wealth == 1)).sum()
        n_agege15_wealth5 = (df.is_alive & (df.age_years >= 15) & (df.li_wealth == 5)).sum()

        n_bmi_1 = (df.is_alive & (df.age_years >= 15) & (df.li_bmi == 1)).sum()
        prop_bmi_1 = n_bmi_1 / n_agege15
        n_bmi_2 = (df.is_alive & (df.age_years >= 15) & (df.li_bmi == 2)).sum()
        prop_bmi_2 = n_bmi_2 / n_agege15
        n_bmi_3 = (df.is_alive & (df.age_years >= 15) & (df.li_bmi == 3)).sum()
        prop_bmi_3 = n_bmi_3 / n_agege15
        n_bmi_4 = (df.is_alive & (df.age_years >= 15) & (df.li_bmi == 4)).sum()
        prop_bmi_4 = n_bmi_4 / n_agege15
        n_bmi_5 = (df.is_alive & (df.age_years >= 15) & (df.li_bmi == 5)).sum()
        prop_bmi_5 = n_bmi_5 / n_agege15

        n_bmi_45_f = (df.is_alive & (df.age_years >= 15) & (df.li_bmi >= 4) & (df.sex == 'F')).sum()
        prop_bmi_45_f = n_bmi_45_f / n_agege15_f
        n_bmi_45_m = (df.is_alive & (df.age_years >= 15) & (df.li_bmi >= 4) & (df.sex == 'M')).sum()
        prop_bmi_45_m = n_bmi_45_m / n_agege15_m
        n_bmi_45_urban = (df.is_alive & (df.age_years >= 15) & (df.li_bmi >= 4) & df.li_urban).sum()
        n_bmi_45_rural = (df.is_alive & (df.age_years >= 15) & (df.li_bmi >= 4) & ~df.li_urban).sum()
        prop_bmi_45_urban = n_bmi_45_urban / n_agege15_urban
        prop_bmi_45_rural = n_bmi_45_rural / n_agege15_rural
        n_bmi_45_wealth1 = (df.is_alive & (df.age_years >= 15) & (df.li_bmi >= 4) & (df.li_wealth == 1)).sum()
        prop_bmi_45_wealth1 = n_bmi_45_wealth1 / n_agege15_wealth1
        n_bmi_45_wealth5 = (df.is_alive & (df.age_years >= 15) & (df.li_bmi >= 4) & (df.li_wealth == 5)).sum()
        prop_bmi_45_wealth5 = n_bmi_45_wealth5 / n_agege15_wealth5

        n_urban_m_not_high_sugar_age1529_not_tob_wealth1 = (
            df.is_alive
            & (df.sex == 'M')
            & ~df.li_high_sugar
            & df.age_years.between(15, 24)
            & ~df.li_tob
            & (df.li_wealth == 1)
        ).sum()

        n_bmi_5_urban_m_not_high_sugar_age1529_not_tob_wealth1 = (
            df.is_alive
            & (df.sex == 'M')
            & ~df.li_high_sugar
            & df.age_years.between(15, 24)
            & ~df.li_tob
            & (df.li_wealth == 1)
            & (df.li_bmi == 5)
        ).sum()

        prop_bmi_5_urban_m_not_high_sugar_age1529_not_tob_wealth1 = (
            n_bmi_5_urban_m_not_high_sugar_age1529_not_tob_wealth1 / n_urban_m_not_high_sugar_age1529_not_tob_wealth1
        )

        if prop_bmi_5_urban_m_not_high_sugar_age1529_not_tob_wealth1 > 0:
            bmi_proportions = {
                'prop_bmi_1': prop_bmi_1,
                'prop_bmi_2': prop_bmi_2,
                'prop_bmi_3': prop_bmi_3,
                'prop_bmi_4': prop_bmi_4,
                'prop_bmi_5': prop_bmi_5,
                'prop_bmi_45_f': prop_bmi_45_f,
                # prop_bmi_45_m is a rare event and is non-zero with 10,000 population sizes
                'prop_bmi_45_m': prop_bmi_45_m,
                'prop_bmi_45_urban': prop_bmi_45_urban,
                'prop_bmi_45_rural': prop_bmi_45_rural,
                'prop_bmi_45_wealth1': prop_bmi_45_wealth1,
                'prop_bmi_45_wealth5': prop_bmi_45_wealth5,
                'prop_bmi_5_urban_m_not_high_sugar_age1529_not_tob_wealth1':
                    prop_bmi_5_urban_m_not_high_sugar_age1529_not_tob_wealth1
            }
        else:
            bmi_proportions = {
                'prop_bmi_1': prop_bmi_1,
                'prop_bmi_2': prop_bmi_2,
                'prop_bmi_3': prop_bmi_3,
                'prop_bmi_4': prop_bmi_4,
                'prop_bmi_5': prop_bmi_5,
                'prop_bmi_45_f': prop_bmi_45_f,
                # prop_bmi_45_m is a rare event and is non-zero with 10,000 population sizes
                'prop_bmi_45_m': prop_bmi_45_m,
                'prop_bmi_45_urban': prop_bmi_45_urban,
                'prop_bmi_45_rural': prop_bmi_45_rural,
                'prop_bmi_45_wealth1': prop_bmi_45_wealth1,
                'prop_bmi_45_wealth5': prop_bmi_45_wealth5,
                'prop_bmi_5_urban_m_not_high_sugar_age1529_not_tob_wealth1':
                    0
            }

        # Screen for null values before placing in the logger
        for k, v in bmi_proportions.items():
            if np.isnan(v):
                bmi_proportions[k] = 0.0

        return bmi_proportions


def get_multinomial_probabilities(_df, *linear_models):
    """returns a dataframe of probabilities for each outcome
    columns are in the same order as function arguments
    last column is base outcome
    """
    ratios = pd.concat([lm.predict(_df) for lm in linear_models], axis=1)
    ratios.insert(len(ratios.columns), len(ratios.columns), 1.0)
    denom = ratios.sum(axis=1)
    return ratios.div(denom, axis=0)


class LifestyleModels:
    """Helper class to store all linear models for the Lifestyle module. We have used two types of linear models
    namely logistic and multiplicative linear models. We currently have defined linear models for the following;
            1.  urban rural status
            2.  wealth level
            3.  low exercise
            4.  tobacco use
            5.  excessive alcohol
            6.  marital status
            7.  education
            8.  unimproved sanitation
            9.  no clean drinking water
            10.  wood burn stove
            11.  no access hand washing
            12.  salt intake
            13  sugar intake
            14. bmi """

    def __init__(self, module):
        # initialise variables
        self.module = module
        self.rng = self.module.rng
        self.params = module.parameters
        # create all linear models dictionary for use in both initialisation and update of properties
        self.models = {
            'li_urban': {
                'init': self.rural_urban_linear_model(),
                # 'update': self.update_rural_urban_property_linear_model()
            },
            'li_wealth': {
                'init': self.wealth_level_linear_model()
            },
            'li_low_ex': {
                'init': self.low_exercise_linear_model(),
                # 'update': self.update_exercise_property_linear_model()
            },
            'li_tob': {
                'init': self.tobacco_use_linear_model(),
                # 'update': self.update_tobacco_use_property_linear_model()
            },
            'li_ex_alc': {
                'init': self.excessive_alcohol_linear_model(),
                # 'update': self.update_excess_alcohol_property_linear_model()
            },
            'li_mar_stat': {
                'init': self.marital_status_linear_model(),
                # 'update': self.update_marital_status_linear_model()
            },
            'li_in_ed': {
                'init': 0,
                # 'update': self.update_education_status_linear_model()
            },
            'li_unimproved_sanitation': {
                'init': self.unimproved_sanitation_linear_model(),
                # 'update': self.update_unimproved_sanitation_status_linear_model()
            },
            'li_no_clean_drinking_water': {
                'init': self.no_clean_drinking_water_linear_model(),
                # 'update': self.update_no_clean_drinking_water_linear_model()
            },
            'li_wood_burn_stove': {
                'init': self.wood_burn_stove_linear_model(),
                # 'update': self.update_wood_burn_stove_linear_model()
            },
            'li_no_access_handwashing': {
                'init': self.no_access_hand_washing(),
                # 'update': self.update_no_access_hand_washing_status_linear_model()
            },
            'li_high_salt': {
                'init': self.salt_intake_linear_model(),
                # 'update': self.update_high_salt_property_linear_model()
            },
            'li_high_sugar': {
                'init': self.sugar_intake_linear_model(),
                # 'update': self.update_high_sugar_property_linear_model()
            },
            'li_bmi': {
                'init': 0,
                # 'update': self.update_bmi_categories_linear_model()
            }
        }

    def initialise_all_properties(self, df):
        """initialise population properties using linear models defined in LifestyleModels class.

        :param df: The population dataframe """
        # loop through linear models dictionary and initialise each property in the population dataframe
        for _property_name, _model in self.models.items():
            if _property_name in ['li_wealth', 'li_mar_stat']:
                df.loc[df.is_alive, _property_name] = _model['init'].predict(df.loc[df.is_alive])

            elif _property_name in ['li_in_ed', 'li_bmi']:
                self.module.init_edu_bmi_properties(_property_name)

            else:
                df.loc[df.is_alive, _property_name] = _model['init'].predict(df.loc[df.is_alive], self.rng)

    def rural_urban_linear_model(self) -> LinearModel:
        """ a function to create linear model for rural urban properties. Here we are using additive linear model and
        have no predictors hence the base probability is used as the final value for all individuals """

        # set baseline probability for all individuals
        base_prob = self.params['init_p_urban']

        # create linear model
        rural_urban_lm = LinearModel(LinearModelType.MULTIPLICATIVE,
                                     base_prob
                                     )

        # return rural urban linear model
        return rural_urban_lm

    def wealth_level_linear_model(self) -> LinearModel:
        """ a function to create linear model for wealth level property. Here are using multiplicative linear model
        and are setting probabilities based on whether the individual is urban or rural """

        wealth_level_lm = LinearModel.multiplicative(
            Predictor('li_urban').when(True, self.rng.choice(
                [1, 2, 3, 4, 5], p=self.params['init_p_wealth_urban']
            ))
                .otherwise(self.rng.choice(
                [1, 2, 3, 4, 5], p=self.params['init_p_wealth_rural']))
        )
        # return wealth level linear model
        return wealth_level_lm

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
        """A function to create linear model for individual's marital status. Here, We are using a multiplicative
        linear model and we are assigning individual's marital status based on their age group.In this module,
        marital status is in three categories;
                1.  Never Married
                2.  Currently Married
                3   Divorced or Widowed
        """

        # create marital status linear model
        mar_status_lm = LinearModel.multiplicative(
            Predictor('age_years').when('.between(15, 19)', self.rng.choice(
                [1, 2, 3], p=self.params['init_dist_mar_stat_age1520']))
                .when('.between(20, 29)', self.rng.choice(
                [1, 2, 3], p=self.params['init_dist_mar_stat_age2030']))
                .when('.between(30, 39)', self.rng.choice(
                [1, 2, 3], p=self.params['init_dist_mar_stat_age3040']))
                .when('.between(40, 49)', self.rng.choice(
                [1, 2, 3], p=self.params['init_dist_mar_stat_age4050']))
                .when('.between(50, 59)', self.rng.choice(
                [1, 2, 3], p=self.params['init_dist_mar_stat_age5060']))
                .when('>= 60', self.rng.choice(
                [1, 2, 3], p=self.params['init_dist_mar_stat_agege60'])),
        )

        # return marital status linear model
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
                                                                              self.params[
                                                                                  'init_rp_some_ed_age0513'])
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

    # --------------------- LINEAR MODELS FOR UPDATING POPULATION PROPERTIES ------------------------------ #
    def update_rural_urban_property_linear_model(self) -> Dict[str, LinearModel]:
        """A function to create linear model for updating the rural urban status of an individual. Here, we are using
        multiplicative linear model """

        # create rural urban linear model
        rural_urban_transition_lm = LinearModel.multiplicative(
            Predictor('li_urban').when(False, self.params['r_urban'])
            .otherwise(0.0)
        )
        urban_rural_transition_lm = LinearModel.multiplicative(
            Predictor('li_urban').when(True, self.params['r_urban'])
            .otherwise(0.0)
        )
        urban_rural_models = {
            'rural_urban_transition_lm': rural_urban_transition_lm,
            'urban_rural_transition_lm': urban_rural_transition_lm
        }
        # return linear model
        return urban_rural_models

    def update_exercise_property_linear_model(self) -> Dict[str, LinearModel]:
        """ A function to create linear model for updating the exercise property. Here we are using multiplicative
        linear model and are looking at rate of transitions from low exercise to not low exercise and vice versa """

        # get base probability
        base_prob = self.params['r_low_ex']

        # create exercise linear model
        update_exercise_status_lm = LinearModel(LinearModelType.MULTIPLICATIVE,
                                                base_prob,
                                                Predictor('sex').when('F', self.params['rr_low_ex_f']),
                                                Predictor('li_urban').when(True, self.params['rr_low_ex_urban'])
                                                )

        # handle transitions

        # get transition baseline probability
        trans_base_prob = self.params['r_not_low_ex']

        trans_exercise_status_lm = LinearModel(LinearModelType.MULTIPLICATIVE,
                                               trans_base_prob,
                                               Predictor('li_low_ex').when(False, 0),
                                               Predictor('li_exposed_to_campaign_exercise_increase')
                                               .when(True, self.params['rr_not_low_ex_pop_advice_exercise']))

        # return all update exercise linear models
        exercise_dict = {
            'update_lm': update_exercise_status_lm,
            'trans_lm': trans_exercise_status_lm
        }
        return exercise_dict

    def update_tobacco_use_property_linear_model(self) -> Dict[str, LinearModel]:
        """A function to create linear model for tobacco use property. Here we are using multiplicative linear model
        and are looking at transitions from not tobacco use to tobacco use and vice versa"""

        # define start tobacco use baseline
        base_prob = self.params['r_tob']

        # start tobacco linear model
        start_tob_lm = LinearModel(LinearModelType.MULTIPLICATIVE,
                                   base_prob,
                                   Predictor('age_years').when('.between(20, 40, inclusive="right")',
                                                               self.params['rr_tob_age2039'])
                                   .when('>40', self.params['rr_tob_agege40']),
                                   Predictor('sex').when('F', self.params['rr_tob_f']),
                                   Predictor('li_wealth').when(2, self.params['rr_tob_wealth'])
                                   .when(3, self.params['rr_tob_wealth'] ** 2)
                                   .when(4, self.params['rr_tob_wealth'] ** 3)
                                   .when(5, self.params['rr_tob_wealth'] ** 4)
                                   )

        # handle tobacco use transitions
        # define transition baseline
        trans_base_prob = self.params['r_not_tob']

        # create tobacco transition linear model
        trans_tob_lm = LinearModel(LinearModelType.MULTIPLICATIVE,
                                   trans_base_prob,
                                   Predictor('li_exposed_to_campaign_quit_smoking')
                                   .when(True, self.params['rr_not_tob_pop_advice_tobacco'])
                                   )

        # return a dictionary of all linear models
        tob_update_lm_dict = {
            'start_tob': start_tob_lm,
            'trans_tob': trans_tob_lm
        }
        return tob_update_lm_dict

    def update_excess_alcohol_property_linear_model(self) -> Dict[str, LinearModel]:
        """ a function tp create linear model for excess alcohol property. Here we are using multiplicative linear
        model and are looking at individuals transition from either excess alcohol to not excess alcohol or vice
        versa """

        # define excessive alcohol linear model
        not_ex_alc_lm = LinearModel.multiplicative(
            Predictor().when('(age_years >= 15) & (sex == "M") & (li_ex_alc == False)',
                             self.params['r_ex_alc'])
                .when('(age_years >= 15) & (sex == "F") & (li_ex_alc == False)',
                      (self.params['r_ex_alc'] * self.params['rr_ex_alc_f']))
                 .otherwise(0.0)
        )
        now_ex_alc_lm = LinearModel.multiplicative(
            Predictor().when('(age_years >= 15) & (li_ex_alc == True)',
                             self.params['r_not_ex_alc'])
        )

        # handle transitions
        trans_ex_alc_lm = LinearModel.multiplicative(
            Predictor().when('li_ex_alc == True', self.params['r_not_ex_alc'])
                .when('(li_ex_alc == True) & (li_exposed_to_campaign_alcohol_reduction == True)',
                      self.params['rr_not_ex_alc_pop_advice_alcohol'])
                .otherwise(0.0)
        )

        all_models_dict = {
            'not_ex_alc': not_ex_alc_lm,
            'now_ex_alc': now_ex_alc_lm,
            'trans_ex_alc': trans_ex_alc_lm
        }
        # return all linear models
        return all_models_dict

    def update_marital_status_linear_model(self) -> LinearModel:
        """A function to create linear models for marital status property. Here we are using multiplicative linear
        model and are looking at individuals ability to transition into different marital status """

        # create marital status linear model
        mar_status_lm = LinearModel.multiplicative(
            Predictor('li_mar_stat').when(2, self.params['r_mar'])
                .when(3, self.params['r_div_wid'])
                .otherwise(0.0),
        )

        # return marital status linear model
        return mar_status_lm

    def update_education_status_linear_model(self) -> Dict[str, LinearModel]:
        """ a function to create linear models for for education prperty. here we are using multiplicative linear
        model and are looking at individuals ability to transition from different education levels """
        # create education linear model
        update_primary_edu_lm = LinearModel.multiplicative(
            Predictor().when('(age_exact_years == .between(5, 5.25, inclusive=right))', self.params['p_ed_primary'])
                .when('(li_wealth == 4)', self.params['rp_ed_primary_higher_wealth'])
                .when('(li_wealth == 3)', (self.params['rp_ed_primary_higher_wealth'] ** 2))
                .when('(li_wealth == 2)', (self.params['rp_ed_primary_higher_wealth'] ** 3))
                .when('(li_wealth == 1)', (self.params['rp_ed_primary_higher_wealth'] ** 4))
                .otherwise(0.0)

        )
        # update secondary education linear model
        update_sec_edu_lm = LinearModel.multiplicative(
            Predictor().when('(age_years == 13) & (li_in_ed == True) & (li_ed_lev == 2)', self.params['p_ed_secondary'])
                .when('(li_wealth == 4)', self.params['rp_ed_primary_higher_wealth'])
                .when('(li_wealth == 3)', (self.params['rp_ed_primary_higher_wealth'] ** 2))
                .when('(li_wealth == 2)', (self.params['rp_ed_primary_higher_wealth'] ** 3))
                .when('(li_wealth == 1)', (self.params['rp_ed_primary_higher_wealth'] ** 4))
                .otherwise(0.0)
        )

        # create education drop outs linear model
        drop_edu_lm = LinearModel.multiplicative(
            Predictor().when('li_in_ed == True', self.params['r_stop_ed'])
                .when('(li_wealth == 5)', (self.params['rr_stop_ed_lower_wealth'] ** 4))
                .when('(li_wealth == 4)', (self.params['rr_stop_ed_lower_wealth'] ** 3))
                .when('(li_wealth == 3)', (self.params['rr_stop_ed_lower_wealth'] ** 2))
                .when('(li_wealth == 2)', self.params['rr_stop_ed_lower_wealth'])
                .otherwise(0.0)
        )

        # create a dictionary that will contain education linear models
        edu_models = {
            'update_primary_edu_lm': update_primary_edu_lm,
            'update_sec_edu_lm': update_sec_edu_lm,
            'drop_edu_lm': drop_edu_lm,

        }

        # return all linear models
        return edu_models

    def update_unimproved_sanitation_status_linear_model(self) -> Dict[str, LinearModel]:
        """ A function to create linear models for updating unimproved sanitation property. here we are using
        multinomial linear model and are looking at individual's ability to transition from unimproved sanitation to
        improved sanitation or vice versa """
        # create unimproved sanitation linear model
        unimproved_san_lm = LinearModel.multiplicative(
            Predictor('li_unimproved_sanitation').when(True, self.params['r_improved_sanitation'])
                .otherwise(0.0)
        )
        # create a linear model that will contain probability of improved sanitation upon moving to urban from rural
        unimproved_san_urban_lm = LinearModel.multiplicative(
            Predictor().when('(li_unimproved_sanitation == True) & (li_date_trans_to_urban == self.sim.date)',
                             self.params['init_p_unimproved_sanitation_urban'])
                .otherwise(0.0)
        )

        # create a dictionary that will contain all linear models
        unimproved_san_models = {
            'unimproved_san_lm': unimproved_san_lm,
            'unimproved_san_urban_lm': unimproved_san_urban_lm
        }
        # return the dictionary
        return unimproved_san_models

    def update_no_access_hand_washing_status_linear_model(self) -> LinearModel:
        """ a function to create linear models for updating no access to hand washing property. Here we are using
        multiplicative linear models and are looking at individual ability to transition from no access to hand
        washing to having access to hand washing """
        # create a linear model that will update individual's no access to handwashing property
        update_no_acc_hand_washing_lm = LinearModel.multiplicative(
            Predictor('li_no_access_handwashing').when(True, self.params['r_access_handwashing'])
                .otherwise(0.0)
        )

        # return the linear model
        return update_no_acc_hand_washing_lm

    def update_no_clean_drinking_water_linear_model(self) -> Dict[str, LinearModel]:
        """ a function to create linear models for updating no clean drinking water property. Here we are using
        multiplicative linear model and are looking at individuals ability to transition from no access to clean
        drinking water to having an access to drinking water """
        # probability of moving to clean drinking water at all follow-up times
        prob_clean_drinking_water_lm = LinearModel.multiplicative(
            Predictor('li_no_clean_drinking_water').when(True, self.params['r_clean_drinking_water'])
                .otherwise(0.0)
        )

        # probability of no clean drinking water upon moving to urban from rural
        no_clean_drinking_water_trans_urban_lm = LinearModel.multiplicative(
            Predictor().when('(li_no_clean_drinking_water == True) & (li_date_trans_to_urban == self.sim.date)',
                             self.params['init_p_no_clean_drinking_water_urban'])
                .otherwise(0.0)
        )

        # create a dictionary that will contain no clean drinking water linear models
        no_clean_drinking_water_models = {
            'prob_clean_drinking_water_lm': prob_clean_drinking_water_lm,
            'no_clean_drinking_water_trans_urban_lm': no_clean_drinking_water_trans_urban_lm
        }
        # return the no clean drinking water linear models
        return no_clean_drinking_water_models

    def update_wood_burn_stove_linear_model(self) -> Dict[str, LinearModel]:
        """ a function to create linear models for updating wood burn stove property. Here we are using
        multiplicative linear model and are looking at individual's ability to transition from wood burn stove to non
        wood burn stove """
        # probability of moving to non wood burn stove at all follow-up times
        non_wood_burn_stove_lm = LinearModel.multiplicative(
            Predictor('li_wood_burn_stove').when(True, self.params['r_non_wood_burn_stove'])
                .otherwise(0.0)
        )
        # probability of moving to wood burn stove upon moving to urban from rural
        wood_burn_stove_urban_lm = LinearModel.multiplicative(
            Predictor().when('(li_wood_burn_stove == True) & (li_date_trans_to_urban == self.sim.date)',
                             self.params['init_p_wood_burn_stove_urban'])
        )
        # all wood burn stove linear_models dictionary
        wood_burn_stove_models = {
            'non_wood_burn_stove_lm': non_wood_burn_stove_lm,
            'wood_burn_stove_urban_lm': wood_burn_stove_urban_lm
        }
        # return all linear models
        return wood_burn_stove_models

    def update_high_salt_property_linear_model(self) -> Dict[str, LinearModel]:
        """ a function to create linear models for updating high salt property. here we are using multiplicative
        linear model and are looking at individuals ability to transition from high salt to not high salt """
        # create a linear model with not high salt probabilities
        not_high_salt_lm = LinearModel.multiplicative(
            Predictor().when('(li_high_salt == False)', self.params['r_high_salt_urban'])
                .when('(li_urban == True)', self.params['rr_high_salt_rural'])
                .otherwise(0.0)
        )
        # create a linear model that handles transitions from high salt to not high salt
        trans_not_high_salt_lm = LinearModel.multiplicative(
            Predictor().when('li_high_salt', self.params['r_not_high_salt'])
                .when('li_exposed_to_campaign_salt_reduction', self.params['rr_not_high_salt_pop_advice_salt'])
                .otherwise(0.0)
        )

        # a dictionary that contains all high salt linear models
        high_salt_models = {
            'not_high_salt_lm': not_high_salt_lm,
            'trans_not_high_salt_lm': trans_not_high_salt_lm
        }
        # return all high salt linear models
        return high_salt_models

    def update_high_sugar_property_linear_model(self) -> Dict[str, LinearModel]:
        """ a function to create linear model for updating high sugar property. Here we are using multiplicative
        linear model and are looking at individuals ability to transition from high sugar to not high sugar """
        # a linear model with probabilities of not high sugar
        not_high_sugar_lm = LinearModel.multiplicative(
            Predictor('li_high_sugar').when(True, self.params['r_high_sugar'])
                .otherwise(0.0)
        )

        # handles transitions from high sugar to not high sugar
        trans_not_high_sugar = LinearModel.multiplicative(
            Predictor().when('(li_high_sugar == True)', self.params['r_not_high_sugar'])
                .when('(li_exposed_to_campaign_sugar_reduction == True)',
                      self.params['rr_not_high_sugar_pop_advice_sugar'])
                .otherwise(0.0)
        )

        # create a dictionary that contains all high sugar property linear model
        high_sugar_models = {
            'not_high_sugar_lm': not_high_sugar_lm,
            'trans_not_high_sugar': trans_not_high_sugar
        }

        # return all high sugar linear models
        return high_sugar_models

    def update_bmi_categories_linear_model(self) -> Dict[str, LinearModel]:
        """ a function to create linear model for updating bmi categories. here we are using multiplicative linear
        model and are looking at individual's ability to transition from different bmi categories """

        # create a linear model for possible increase in category of bmi
        bmi_cat_1_to_4_idx_lm = LinearModel.multiplicative(
            Predictor().when('(age_years == >= 15) & (li_bmi == .between(1, 4))', self.params['r_higher_bmi'])
                .when('(li_urban == True)', self.params['rr_higher_bmi_urban'])
                .when('(sex == 'F')', self.params['rr_higher_bmi_f'])
                .when('(age_years == .between(30, 49)', self.params['rr_higher_bmi_age3049'])
                .when('(age_years >= 50)', self.params['rr_higher_bmi_agege50'])
                .when('(li_tob == True)', self.params['rr_higher_bmi_tob'])
                .when('(li_wealth == 2)', self.params['rr_higher_bmi_per_higher_wealth'] ** 2)
                .when('(li_wealth == 3)', self.params['rr_higher_bmi_per_higher_wealth'] ** 3)
                .when('(li_wealth == 4)', self.params['rr_higher_bmi_per_higher_wealth'] ** 4)
                .when('(li_wealth == 5)', self.params['rr_higher_bmi_per_higher_wealth'] ** 5)
                .when('(li_high_sugar == True)', self.params['rr_higher_bmi_high_sugar'])
                .otherwise(0.0)
        )

        # create a linear model for possible decrease in category of bmi
        dec_bmi_lm = LinearModel.multiplicative(
            Predictor().when('li_bmi == .between(3, 5)) & (age_years >= 15)', self.params['r_lower_bmi'])
                .when('(li_urban == True)', self.params['rr_lower_bmi_tob'])
                .when('(li_exposed_to_campaign_weight_reduction == True)',
                      self.params['rr_lower_bmi_pop_advice_weight'])
                .otherwise(0.0)
        )

        # a dictionary that contains bmi_categories linear models
        bmi_cat_models = {
            'bmi_cat_1_to_4_idx_lm': bmi_cat_1_to_4_idx_lm,
            'dec_bmi_lm': dec_bmi_lm
        }
        # return all bmi categories linear models
        return bmi_cat_models


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
        super().__init__(module, frequency=DateOffset(months=self.repeat_months))
        assert isinstance(module, Lifestyle)

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        # call a function to handle all transitions
        self.handle_all_transitions()

    def handle_all_transitions(self):
        df = self.sim.population.props
        m = self.module

        # handle rural urban transitions
        self.update_rural_urban_property(df, m)

        # handle low exercise transitions
        self.update_low_exercise_property(df, m)

        # handle tobacco use transitions
        self.update_tobacco_use_property(df, m)

        # handle excessive alcohol transitions
        self.update_excessive_alcohol_property(df, m)

        # handle marital status transitions
        self.update_marital_status_property(df)

        # handle education transitions
        self.update_education_property(df)

        # handle unimproved sanitation transitions
        self.unimproved_sanitation_property(df, m)

        # handle no clean drinking water
        self.update_no_access_handwashing(df, m)

        # handle no clean drinking water
        self.update_no_clean_drinking_water(df, m)

        # handle wood burn stove transitions
        self.update_wood_burn_stove(df, m)

        # handle high salt transitions
        self.update_high_salt_property(df, m)

        # handle high sugar transitions
        self.update_high_sugar(df, m)

        # handle bmi transitions
        self.update_bmi_categories(df, m)

        # --- FSW ---
        self.module.determine_who_will_be_sexworker(months_since_last_poll=self.repeat_months)

    def update_rural_urban_property(self, df, m):
        """ a function to handle individual transitions from rural to urban and vice versa
        :param df: population dataframe
        :param m: a pointer to Lifestyle module """
        # get index of current urban/rural status
        currently_rural = df.index[~df.li_urban & df.is_alive]
        currently_urban = df.index[df.li_urban & df.is_alive]

        # handle new transitions
        rural_to_urban = currently_rural[self.module.rng.random_sample(size=len(currently_rural)) < m.parameters[
            'r_urban']]
        df.loc[rural_to_urban, 'li_urban'] = True

        # handle new transitions to rural
        urban_to_rural = currently_urban[
            self.module.rng.random_sample(size=len(currently_urban)) < m.parameters['r_rural']]
        df.loc[urban_to_rural, 'li_urban'] = False

    def update_low_exercise_property(self, df, m):
        """ a function to handle transitions from rural to urban and vice versa
        :param df: population dataframe
        :param m: a pointer to Lifestyle module """
        # get indexes of individuals 15+ years who are not low exercise
        adults_not_low_ex = df.index[~df.li_low_ex & df.is_alive & (df.age_years >= 15)]
        eff_p_low_ex = pd.Series(m.parameters['r_low_ex'], index=adults_not_low_ex)
        eff_p_low_ex.loc[df.sex == 'F'] *= m.parameters['rr_low_ex_f']
        eff_p_low_ex.loc[df.li_urban] *= m.parameters['rr_low_ex_urban']
        df.loc[adults_not_low_ex, 'li_low_ex'] = m.rng.random_sample(len(adults_not_low_ex)) < eff_p_low_ex

        # transition from low exercise to not low exercise
        low_ex_idx = df.index[df.li_low_ex & df.is_alive]
        eff_rate_not_low_ex = pd.Series(m.parameters['r_not_low_ex'], index=low_ex_idx)
        eff_rate_not_low_ex.loc[df.li_exposed_to_campaign_exercise_increase] *= (
            m.parameters['rr_not_low_ex_pop_advice_exercise']
        )
        random_draw = m.rng.random_sample(len(low_ex_idx))
        newly_not_low_ex_idx = low_ex_idx[random_draw < eff_rate_not_low_ex]
        df.loc[newly_not_low_ex_idx, 'li_low_ex'] = False

        # todo: this line below to start a general population campaign
        #  to increase exercise not working yet (same for others below)
        all_idx_campaign_exercise_increase = df.index[
            df.is_alive & (self.sim.date == m.parameters['start_date_campaign_exercise_increase'])
            ]
        df.loc[all_idx_campaign_exercise_increase, 'li_exposed_to_campaign_exercise_increase'] = True

    def update_tobacco_use_property(self, df, m):
        """a function to handle tobacco use transitions
        :param df: population dataframe
        :param m: a pointer to Lifestyle module """

        adults_not_tob = df.index[(df.age_years >= 15) & df.is_alive & ~df.li_tob]

        # start tobacco use
        eff_p_tob = pd.Series(m.parameters['r_tob'], index=adults_not_tob)
        eff_p_tob.loc[(df.age_years >= 20) & (df.age_years < 40)] *= m.parameters['rr_tob_age2039']
        eff_p_tob.loc[df.age_years >= 40] *= m.parameters['rr_tob_agege40']
        eff_p_tob.loc[df.sex == 'F'] *= m.parameters['rr_tob_f']
        eff_p_tob *= m.parameters['rr_tob_wealth'] ** (pd.to_numeric(df.loc[adults_not_tob, 'li_wealth']) - 1)

        df.loc[adults_not_tob, 'li_tob'] = m.rng.random_sample(len(adults_not_tob)) < eff_p_tob

        # transition from tobacco to no tobacco
        tob_idx = df.index[df.li_tob & df.is_alive]
        eff_rate_not_tob = pd.Series(m.parameters['r_not_tob'], index=tob_idx)
        eff_rate_not_tob.loc[df.li_exposed_to_campaign_quit_smoking] *= (
            m.parameters['rr_not_tob_pop_advice_tobacco']
        )
        random_draw = m.rng.random_sample(len(tob_idx))
        newly_not_tob_idx = tob_idx[random_draw < eff_rate_not_tob]
        df.loc[newly_not_tob_idx, 'li_tob'] = False
        df.loc[newly_not_tob_idx, 'li_date_not_tob'] = self.sim.date

        all_idx_campaign_quit_smoking = df.index[
            df.is_alive & (self.sim.date == m.parameters['start_date_campaign_quit_smoking'])
            ]
        df.loc[all_idx_campaign_quit_smoking, 'li_exposed_to_campaign_quit_smoking'] = True

    def update_excessive_alcohol_property(self, df, m):
        """ a function to handle excessive alcohol transitions
        :param df: population dataframe
        :param m: a pointer to Lifestyle module """
        # get indexes of individuals who are 15+ years old
        not_ex_alc_f = df.index[~df.li_ex_alc & df.is_alive & (df.sex == 'F') & (df.age_years >= 15)]
        not_ex_alc_m = df.index[~df.li_ex_alc & df.is_alive & (df.sex == 'M') & (df.age_years >= 15)]
        now_ex_alc = df.index[df.li_ex_alc & df.is_alive]

        df.loc[not_ex_alc_f, 'li_ex_alc'] = (
            m.rng.random_sample(len(not_ex_alc_f))
            < m.parameters['r_ex_alc'] * m.parameters['rr_ex_alc_f']
        )
        df.loc[not_ex_alc_m, 'li_ex_alc'] = m.rng.random_sample(len(not_ex_alc_m)) < m.parameters['r_ex_alc']
        df.loc[now_ex_alc, 'li_ex_alc'] = ~(m.rng.random_sample(len(now_ex_alc)) < m.parameters['r_not_ex_alc'])

        # transition from excess alcohol to not excess alcohol
        ex_alc_idx = df.index[df.li_ex_alc & df.is_alive]
        eff_rate_not_ex_alc = pd.Series(m.parameters['r_not_ex_alc'], index=ex_alc_idx)
        eff_rate_not_ex_alc.loc[
            df.li_exposed_to_campaign_alcohol_reduction
        ] *= m.parameters['rr_not_ex_alc_pop_advice_alcohol']
        random_draw = m.rng.random_sample(len(ex_alc_idx))
        newly_not_ex_alc_idx = ex_alc_idx[random_draw < eff_rate_not_ex_alc]
        df.loc[newly_not_ex_alc_idx, 'li_ex_alc'] = False

        all_idx_campaign_alcohol_reduction = df.index[
            df.is_alive & (self.sim.date == m.parameters['start_date_campaign_alcohol_reduction'])
            ]
        df.loc[all_idx_campaign_alcohol_reduction, 'li_exposed_to_campaign_alcohol_reduction'] = True

    def update_marital_status_property(self, df):
        """ a function to handle marital status transitions
        :param df: population dataframe
        :param m: a pointer to Lifestyle module """
        # get index of individuals aged between 15 to 29 and whose marital status is 1
        curr_never_mar = df.index[df.is_alive & df.age_years.between(15, 29) & (df.li_mar_stat == 1)]
        # get index of individuals who are currently married
        curr_mar = df.index[df.is_alive & (df.li_mar_stat == 2)]

        # update if now married
        now_mar = self.module.rng.random_sample(len(curr_never_mar)) < self.module.parameters['r_mar']
        df.loc[curr_never_mar[now_mar], 'li_mar_stat'] = 2

        # update if now divorced/widowed
        now_div_wid = self.module.rng.random_sample(len(curr_mar)) < self.module.parameters['r_div_wid']
        df.loc[curr_mar[now_div_wid], 'li_mar_stat'] = 3

    def update_education_property(self, df):
        """ a function to handle education transitions
        :param df: population dataframe """

        # get all individuals currently in education
        in_ed = df.index[df.is_alive & df.li_in_ed]

        # ---- PRIMARY EDUCATION

        # get index of all children who are alive and between 5 and 5.25 years old
        age5 = df.index[(df.age_exact_years >= 5) & (df.age_exact_years < 5.25) & df.is_alive]

        # by default, these children are not in education and have education level 1
        df.loc[age5, 'li_ed_lev'] = 1
        df.loc[age5, 'li_in_ed'] = False

        # create a series to hold the probability of primary education for children at age 5
        prob_primary = pd.Series(self.module.parameters['p_ed_primary'], index=age5)
        prob_primary *= self.module.parameters['rp_ed_primary_higher_wealth'] ** (
            5 - pd.to_numeric(df.loc[age5, 'li_wealth']))

        # randomly select some to have primary education
        age5_in_primary = self.module.rng.random_sample(len(age5)) < prob_primary
        df.loc[age5[age5_in_primary], 'li_ed_lev'] = 2
        df.loc[age5[age5_in_primary], 'li_in_ed'] = True

        # ---- SECONDARY EDUCATION

        # get thirteen year olds that are in primary education, any wealth level
        age13_in_primary = df.index[(df.age_years == 13) & df.is_alive & df.li_in_ed & (df.li_ed_lev == 2)]

        # they have a probability of gaining secondary education (level 3), based on wealth
        prob_secondary = pd.Series(self.module.parameters['p_ed_secondary'], index=age13_in_primary)
        prob_secondary *= (
            self.module.parameters['rp_ed_secondary_higher_wealth']
            ** (5 - pd.to_numeric(df.loc[age13_in_primary, 'li_wealth']))
        )

        # randomly select some to get secondary education
        age13_to_secondary = self.module.rng.random_sample(len(age13_in_primary)) < prob_secondary
        df.loc[age13_in_primary[age13_to_secondary], 'li_ed_lev'] = 3

        # those who did not go on to secondary education are no longer in education
        df.loc[age13_in_primary[~age13_to_secondary], 'li_in_ed'] = False

        # ---- DROP OUT OF EDUCATION

        # baseline rate of leaving education then adjust for wealth level
        p_leave_ed = pd.Series(self.module.parameters['r_stop_ed'], index=in_ed)
        p_leave_ed *= (
            self.module.parameters['rr_stop_ed_lower_wealth']
            ** (pd.to_numeric(df.loc[in_ed, 'li_wealth']) - 1)
        )

        # randomly select some individuals to leave education
        now_not_in_ed = self.module.rng.random_sample(len(in_ed)) < p_leave_ed

        df.loc[in_ed[now_not_in_ed], 'li_in_ed'] = False

        # everyone leaves education at age 20
        df.loc[df.is_alive & df.li_in_ed & (df.age_years == 20), 'li_in_ed'] = False

    def unimproved_sanitation_property(self, df, m):
        """ a function to handle unimproved sanitation transitions
        :param df: population dataframe
        :param m: a pointer to Lifestyle module """
        # probability of improved sanitation at all follow-up times
        unimproved_sanitation_idx = df.index[df.li_unimproved_sanitation & df.is_alive]

        eff_rate_improved_sanitation = pd.Series(
            m.parameters['r_improved_sanitation'], index=unimproved_sanitation_idx
        )

        random_draw = m.rng.random_sample(len(unimproved_sanitation_idx))

        newly_improved_sanitation_idx = unimproved_sanitation_idx[random_draw < eff_rate_improved_sanitation]
        df.loc[newly_improved_sanitation_idx, 'li_unimproved_sanitation'] = False
        df.loc[newly_improved_sanitation_idx, 'li_date_acquire_improved_sanitation'] = self.sim.date

        # probability of improved sanitation upon moving to urban from rural
        unimproved_sanitation_newly_urban_idx = df.index[
            df.li_unimproved_sanitation & df.is_alive & (df.li_date_trans_to_urban == self.sim.date)
            ]

        random_draw = m.rng.random_sample(len(unimproved_sanitation_newly_urban_idx))

        eff_prev_unimproved_sanitation_urban = pd.Series(
            m.parameters['init_p_unimproved_sanitation_urban'], index=unimproved_sanitation_newly_urban_idx
        )

        df.loc[unimproved_sanitation_newly_urban_idx, 'li_unimproved_sanitation'] = (
            random_draw < eff_prev_unimproved_sanitation_urban
        )

    def update_no_access_handwashing(self, df, m):
        """" a function to handle no access hand washing transitions
        :param df: population dataframe
        :param m: a pointer to Lifestyle module """
        # probability of moving to access to handwashing at all follow-up times
        no_access_handwashing_idx = df.index[df.li_no_access_handwashing & df.is_alive]

        eff_rate_access_handwashing = pd.Series(m.parameters['r_access_handwashing'], index=no_access_handwashing_idx)

        random_draw = m.rng.random_sample(len(no_access_handwashing_idx))

        newly_access_handwashing_idx = no_access_handwashing_idx[random_draw < eff_rate_access_handwashing]
        df.loc[newly_access_handwashing_idx, 'li_no_access_handwashing'] = False
        df.loc[newly_access_handwashing_idx, 'li_date_acquire_access_handwashing'] = self.sim.date

    def update_no_clean_drinking_water(self, df, m):
        """ a function to handle no clean drinking water transitions
        :param df: population dataframe
        :param m: a pointer to Lifestyle module """
        # probability of moving to clean drinking water at all follow-up times
        no_clean_drinking_water_idx = df.index[df.li_no_clean_drinking_water & df.is_alive]

        eff_rate_clean_drinking_water = pd.Series(
            m.parameters['r_clean_drinking_water'], index=no_clean_drinking_water_idx
        )

        random_draw = m.rng.random_sample(len(no_clean_drinking_water_idx))

        newly_clean_drinking_water_idx = no_clean_drinking_water_idx[random_draw < eff_rate_clean_drinking_water]
        df.loc[newly_clean_drinking_water_idx, 'li_no_clean_drinking_water'] = False
        df.loc[newly_clean_drinking_water_idx, 'li_date_acquire_clean_drinking_water'] = self.sim.date

        # probability of no clean drinking water upon moving to urban from rural
        no_clean_drinking_water_newly_urban_idx = df.index[
            df.li_no_clean_drinking_water & df.is_alive & (df.li_date_trans_to_urban == self.sim.date)
            ]

        random_draw = m.rng.random_sample(len(no_clean_drinking_water_newly_urban_idx))

        eff_prev_no_clean_drinking_water_urban = pd.Series(
            m.parameters['init_p_no_clean_drinking_water_urban'], index=no_clean_drinking_water_newly_urban_idx
        )

        df.loc[no_clean_drinking_water_newly_urban_idx, 'li_no_clean_drinking_water'] = (
            random_draw < eff_prev_no_clean_drinking_water_urban
        )

    def update_wood_burn_stove(self, df, m):
        """ a function to handle wood burn stove transitions
        :param df: population dataframe
        :param m: a pointer to lifestyle module """
        # probability of moving to non wood burn stove at all follow-up times
        wood_burn_stove_idx = df.index[df.li_wood_burn_stove & df.is_alive]

        eff_rate_non_wood_burn_stove = pd.Series(m.parameters['r_non_wood_burn_stove'], index=wood_burn_stove_idx)

        random_draw = m.rng.random_sample(len(wood_burn_stove_idx))

        newly_non_wood_burn_stove_idx = wood_burn_stove_idx[random_draw < eff_rate_non_wood_burn_stove]
        df.loc[newly_non_wood_burn_stove_idx, 'li_wood_burn_stove'] = False
        df.loc[newly_non_wood_burn_stove_idx, 'li_date_acquire_non_wood_burn_stove'] = self.sim.date

        # probability of moving to wood burn stove upon moving to urban from rural
        wood_burn_stove_newly_urban_idx = df.index[
            df.li_wood_burn_stove & df.is_alive & (df.li_date_trans_to_urban == self.sim.date)
            ]

        random_draw = m.rng.random_sample(len(wood_burn_stove_newly_urban_idx))

        eff_prev_wood_burn_stove_urban = pd.Series(
            m.parameters['init_p_wood_burn_stove_urban'], index=wood_burn_stove_newly_urban_idx
        )

        df.loc[wood_burn_stove_newly_urban_idx, 'li_wood_burn_stove'] = random_draw < eff_prev_wood_burn_stove_urban

    def update_high_salt_property(self, df, m):
        """ a function to handle high salt property
         :param df: population dataframe
        :param m: a pointer to lifestyle module """

        not_high_salt_idx = df.index[~df.li_high_salt & df.is_alive]
        eff_rate_high_salt = pd.Series(m.parameters['r_high_salt_urban'], index=not_high_salt_idx)
        eff_rate_high_salt[df.li_urban] *= m.parameters['rr_high_salt_rural']
        random_draw = m.rng.random_sample(len(not_high_salt_idx))
        newly_high_salt = random_draw < eff_rate_high_salt
        newly_high_salt_idx = not_high_salt_idx[newly_high_salt]
        df.loc[newly_high_salt_idx, 'li_high_salt'] = True

        # transition from high salt to not high salt
        high_salt_idx = df.index[df.li_high_salt & df.is_alive]
        eff_rate_not_high_salt = pd.Series(m.parameters['r_not_high_salt'], index=high_salt_idx)
        eff_rate_not_high_salt.loc[
            df.li_exposed_to_campaign_salt_reduction
        ] *= m.parameters['rr_not_high_salt_pop_advice_salt']
        random_draw = m.rng.random_sample(len(high_salt_idx))
        newly_not_high_salt_idx = high_salt_idx[random_draw < eff_rate_not_high_salt]
        df.loc[newly_not_high_salt_idx, 'li_high_salt'] = False

        all_idx_campaign_salt_reduction = df.index[df.is_alive & (self.sim.date == datetime.date(2010, 7, 1))]
        df.loc[all_idx_campaign_salt_reduction, 'li_exposed_to_campaign_salt_reduction'] = True

    def update_high_sugar(self, df, m):
        """ a function to handle high sugar transitions
         :param df: population dataframe
        :param m: a pointer to lifestyle module """
        # get index of individuals who are high sugar
        not_high_sugar_idx = df.index[~df.li_high_sugar & df.is_alive]
        eff_p_high_sugar = pd.Series(m.parameters['r_high_sugar'], index=not_high_sugar_idx)
        random_draw = m.rng.random_sample(len(not_high_sugar_idx))
        newly_high_sugar_idx = not_high_sugar_idx[random_draw < eff_p_high_sugar]
        df.loc[newly_high_sugar_idx, 'li_high_sugar'] = True

        # transition from high sugar to not high sugar
        high_sugar_idx = df.index[df.li_high_sugar & df.is_alive]
        eff_rate_not_high_sugar = pd.Series(m.parameters['r_not_high_sugar'], index=high_sugar_idx)
        eff_rate_not_high_sugar.loc[
            df.li_exposed_to_campaign_sugar_reduction
        ] *= m.parameters['rr_not_high_sugar_pop_advice_sugar']
        random_draw = m.rng.random_sample(len(high_sugar_idx))
        newly_not_high_sugar_idx = high_sugar_idx[random_draw < eff_rate_not_high_sugar]
        df.loc[newly_not_high_sugar_idx, 'li_high_sugar'] = False

        all_idx_campaign_sugar_reduction = df.index[df.is_alive & (self.sim.date == datetime.date(2010, 7, 1))]
        df.loc[all_idx_campaign_sugar_reduction, 'li_exposed_to_campaign_sugar_reduction'] = True

    def update_bmi_categories(self, df, m):
        """ a function to handle bmi transitions
         :param df: population dataframe
        :param m: a pointer to lifestyle module """

        # those reaching age 15 allocated bmi 3
        age15_idx = df.index[df.is_alive & (df.age_exact_years >= 15) & (df.age_exact_years < 15.25)]
        df.loc[age15_idx, 'li_bmi'] = 3

        # possible increase in category of bmi
        bmi_cat_1_to_4_idx = df.index[df.is_alive & (df.age_years >= 15) & df.li_bmi.between(1, 4)]
        eff_rate_higher_bmi = pd.Series(m.parameters['r_higher_bmi'], index=bmi_cat_1_to_4_idx)
        eff_rate_higher_bmi[df.li_urban] *= m.parameters['rr_higher_bmi_urban']
        eff_rate_higher_bmi[df.sex == 'F'] *= m.parameters['rr_higher_bmi_f']
        eff_rate_higher_bmi[df.age_years.between(30, 49)] *= m.parameters['rr_higher_bmi_age3049']
        eff_rate_higher_bmi[df.age_years >= 50] *= m.parameters['rr_higher_bmi_agege50']
        eff_rate_higher_bmi[df.li_tob] *= m.parameters['rr_higher_bmi_tob']
        eff_rate_higher_bmi[df.li_wealth == 2] *= m.parameters['rr_higher_bmi_per_higher_wealth'] ** 2
        eff_rate_higher_bmi[df.li_wealth == 3] *= m.parameters['rr_higher_bmi_per_higher_wealth'] ** 3
        eff_rate_higher_bmi[df.li_wealth == 4] *= m.parameters['rr_higher_bmi_per_higher_wealth'] ** 4
        eff_rate_higher_bmi[df.li_wealth == 5] *= m.parameters['rr_higher_bmi_per_higher_wealth'] ** 5
        eff_rate_higher_bmi[df.li_high_sugar] *= m.parameters['rr_higher_bmi_high_sugar']

        random_draw = m.rng.random_sample(len(bmi_cat_1_to_4_idx))
        newly_increase_bmi_cat_idx = bmi_cat_1_to_4_idx[random_draw < eff_rate_higher_bmi]
        df.loc[newly_increase_bmi_cat_idx, 'li_bmi'] = df['li_bmi'] + 1

        # possible decrease in category of bmi

        bmi_cat_3_to_5_idx = df.index[df.is_alive & df.li_bmi.between(3, 5) & (df.age_years >= 15)]
        eff_rate_lower_bmi = pd.Series(m.parameters['r_lower_bmi'], index=bmi_cat_3_to_5_idx)
        eff_rate_lower_bmi[df.li_urban] *= m.parameters['rr_lower_bmi_tob']
        eff_rate_lower_bmi.loc[
            df.li_exposed_to_campaign_weight_reduction
        ] *= m.parameters['rr_lower_bmi_pop_advice_weight']
        random_draw = m.rng.random_sample(len(bmi_cat_3_to_5_idx))
        newly_decrease_bmi_cat_idx = bmi_cat_3_to_5_idx[random_draw < eff_rate_lower_bmi]
        df.loc[newly_decrease_bmi_cat_idx, 'li_bmi'] = df['li_bmi'] - 1

        all_idx_campaign_weight_reduction = df.index[df.is_alive & (self.sim.date == datetime.date(2010, 7, 1))]
        df.loc[all_idx_campaign_weight_reduction, 'li_exposed_to_campaign_weight_reduction'] = True


def compute_tobacco_use_by_age(pop) -> Dict[str, Any]:
    """called by the logger to computer tobacco use by age """
    # a dictionary to stoke tobacco use statistics
    tob_age_dict: Dict[str, Any] = dict()

    # get tobacco use in individuals aged between 15 to 19
    tob_age15_19 = pop.loc[pop.is_alive & pop.age_years.between(15, 19), 'li_tob']
    # get tobacco use in individuals aged between 20 to 39
    tob_age20_39 = pop.loc[pop.is_alive & pop.age_years.between(20, 39), 'li_tob']
    # get tobacco use in individuals aged 40 and above
    tob_age40 = pop.loc[pop.is_alive & (pop.age_years >= 40), 'li_tob']

    tob_age_dict.update({
        'tob1519': tob_age15_19.mean(),
        'tob2039': tob_age20_39.mean(),
        'tob40': tob_age40.mean(),
    })

    return tob_age_dict


def compute_currently_in_education_individuals_by_age(pop) -> Dict[str, Any]:
    """get a summary of individuals who are currently in education by age groups. the age groups are as follows;
            1.  less than 13 years old
            2.  13 - 20 years
            3.  20 - 30 years
            4.  30 - 39 years
            5.  40 - 49 years
            6.  49 - 59 years
            7.  60+ years
    """

    # a dictionary to store all individuals currently in education
    cur_in_ed_dict: Dict[str, Any] = dict()

    # get individuals currently in education aged 15-19
    cur_ed_l13 = pop.loc[pop.is_alive & pop.age_years < 13, 'li_in_ed']

    # get individuals currently in education and aged between 13 - 20
    cur_ed1320 = pop.loc[pop.is_alive & pop.age_years.between(13, 20), 'li_in_ed']

    # get individuals currently in education and aged between 20 - 29
    cur_ed2029 = pop.loc[pop.is_alive & pop.age_years.between(20, 29), 'li_in_ed']

    # get individuals currently in education and aged between 30 - 39
    cur_ed3039 = pop.loc[pop.is_alive & pop.age_years.between(30, 39), 'li_in_ed']

    # get individuals currently in education and aged between 40 - 49
    cur_ed4049 = pop.loc[pop.is_alive & pop.age_years.between(40, 49), 'li_in_ed']

    # get individuals currently in education and aged between 13 - 20
    cur_ed15059 = pop.loc[pop.is_alive & pop.age_years.between(50, 59), 'li_in_ed']

    # get individuals currently in education and aged 60+
    cur_ed60 = pop.loc[pop.is_alive & pop.age_years >= 60, 'li_in_ed']

    # todo: update dictionary with only rows that sum up to a value > 0
    cur_in_ed_dict.update({
        'cur_ed_l13': cur_ed_l13.mean(),
        'cur_ed1320': cur_ed1320.mean(),
        'cur_ed2029': cur_ed2029.mean(),
        'cur_ed3039': cur_ed3039.mean(),
        'cur_ed4049': cur_ed4049.mean(),
        'cur_ed5059': cur_ed15059.mean(),
        'cur_ed60': cur_ed60.mean(),
    })

    return cur_in_ed_dict


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

        # TODO *** THIS HAS TROUBLE BE PARSED ON LONG RUNS BY PARSE_OUTPUT: CHANGING KEYS DUE TO GROUPBY? \
        #  NEED TO USE UNSTACK?!?!?
        """
        def flatten_tuples_in_keys(d1):
            d2 = dict()
            for k in d1.keys():
                d2['_'.join([str(y) for y in k])] = d1[k]
            return d2

        logger.info(key='li_urban', data=df[df.is_alive].groupby('li_urban').size().to_dict())
        logger.info(key='li_wealth', data=df[df.is_alive].groupby('li_wealth').size().to_dict())
        logger.info(key='li_tob', data=flatten_tuples_in_keys(
            df[df.is_alive].groupby(['sex', 'li_tob']).size().to_dict())
                    )
        logger.info(key='li_ed_lev_by_age',
                    data=flatten_tuples_in_keys(
                        df[df.is_alive].groupby(['age_range', 'li_in_ed', 'li_ed_lev']).size().to_dict())
                    )
        logger.info(
            key='bmi_proportions',
            data=self.module.compute_bmi_proportions_of_interest()
        )
        logger.info(key='li_low_ex', data=flatten_tuples_in_keys(
            df[df.is_alive].groupby(['sex', 'li_low_ex']).size().to_dict())
                    )
        """
        # for _property in ['properties']:
        #     logger.info(
        #         key=_property,
        #         data=flatten_multi_index_series_into_dict_for_logging(df.loc[df.is_alive, _property].groupby(
        #             ['sex', 'age_group']).apply("find the proportion in each group"))
        #     )

        # log summary of individuals living in both rural and urban
        logger.info(key='urban_rural_pop',
                    data=df.loc[
                        df.is_alive, 'li_urban'
                    ].value_counts().sort_index().to_dict(),
                    description='Urban and rural population')

        # log summary of tobacco use by gender
        logger.info(
            key='tobacco_use',
            data=df.loc[df.is_alive & df.li_tob, 'sex'].value_counts().sort_index().to_dict(),
            description='tobacco use by gender'
        )

        # log summary of tobacco use by age
        logger.info(
            key='tobacco_use_age_range',
            data=compute_tobacco_use_by_age(df),
            description='tobacco use by age range'
        )

        # log summary of males and females currently in education
        logger.info(
            key='cur_in_ed',
            data=df.loc[df.is_alive & df.li_in_ed, 'sex'].value_counts().sort_index().to_dict(),
            description='male and female individuals current in education'
        )

        # log summary of individuals by age group currently in education
        logger.info(
            key='age_group_cur_in_ed',
            data=compute_currently_in_education_individuals_by_age(df),
            description='age group of individuals currently in education'
        )

        # log summary of males circumcised
        logger.info(
            key='prop_adult_men_circumcised',
            data=[df.loc[df.is_alive & (df.sex == 'M') & (df.age_years >= 15)].li_is_circ.mean()]
        )

        women_1549 = df.is_alive & (df.sex == "F") & df.age_years.between(15, 49)

        if sum(women_1549) > 0:
            logger.info(
                key='proportion_1549_women_sexworker',
                data=[sum(women_1549 & df.li_is_sexworker) / sum(women_1549)]
            )
        else:
            logger.info(
                key='proportion_1549_women_sexworker',
                data=[0]
            )

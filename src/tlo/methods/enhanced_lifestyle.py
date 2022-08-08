"""
Lifestyle module
Documentation: 04 - Methods Repository/Method_Lifestyle.xlsx
"""
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.util import get_person_id_to_inherit_from

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

    def initialise_population(self, population):
        """Set our property values for the initial population.
        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals
        m = self
        rng = m.rng
        alive_idx = df.index[df.is_alive]

        # -------------------- DEFAULTS ------------------------------------------------------------

        df['li_urban'] = False
        df['li_wealth'].values[:] = 3
        df['li_bmi'] = 0
        df['li_exposed_to_campaign_weight_reduction'] = False
        df['li_low_ex'] = False
        df['li_exposed_to_campaign_exercise_increase'] = False
        df['li_high_salt'] = False
        df['li_exposed_to_campaign_salt_reduction'] = False
        df['li_high_sugar'] = False
        df['li_exposed_to_campaign_sugar_reduction'] = False
        df['li_tob'] = False
        df['li_date_not_tob'] = pd.NaT
        df['li_exposed_to_campaign_quit_smoking'] = False
        df['li_ex_alc'] = False
        df['li_exposed_to_campaign_alcohol_reduction'] = False
        df['li_mar_stat'].values[:] = 1
        df['li_in_ed'] = False
        df['li_ed_lev'].values[:] = 1
        df['li_unimproved_sanitation'] = True
        df['li_no_access_handwashing'] = True
        df['li_no_clean_drinking_water'] = True
        df['li_wood_burn_stove'] = True
        df['li_date_trans_to_urban'] = pd.NaT
        df['li_date_acquire_improved_sanitation'] = pd.NaT
        df['li_date_acquire_access_handwashing'] = pd.NaT
        df['li_date_acquire_clean_drinking_water'] = pd.NaT
        df['li_date_acquire_non_wood_burn_stove'] = pd.NaT
        df['li_is_sexworker'] = False
        df['li_is_circ'] = False
        # todo: express all rates per year and divide by 4 inside program

        # -------------------- URBAN-RURAL STATUS --------------------------------------------------

        # todo: urban rural depends on district of residence

        # randomly selected some individuals as urban
        df.loc[alive_idx, 'li_urban'] = (
            rng.random_sample(size=len(alive_idx)) < m.parameters['init_p_urban']
        )

        # get the indices of all individuals who are urban or rural
        urban_index = df.index[df.is_alive & df.li_urban]
        rural_index = df.index[df.is_alive & ~df.li_urban]

        # randomly sample wealth category according to urban/rural wealth probs
        df.loc[urban_index, 'li_wealth'] = rng.choice(
            [1, 2, 3, 4, 5], size=len(urban_index), p=m.parameters['init_p_wealth_urban']
        )
        df.loc[rural_index, 'li_wealth'] = rng.choice(
            [1, 2, 3, 4, 5], size=len(rural_index), p=m.parameters['init_p_wealth_rural']
        )

        # -------------------- LOW EXERCISE --------------------------------------------------------

        age_ge15_idx = df.index[df.is_alive & (df.age_years >= 15)]

        init_odds_low_ex_urban_m = (
            m.parameters['init_p_low_ex_urban_m']
            / (1 - m.parameters['init_p_low_ex_urban_m'])
        )

        odds_low_ex = pd.Series(init_odds_low_ex_urban_m, index=age_ge15_idx)

        odds_low_ex.loc[df.sex == 'F'] *= m.parameters['init_or_low_ex_f']
        odds_low_ex.loc[~df.li_urban] *= m.parameters['init_or_low_ex_rural']

        low_ex_probs = odds_low_ex / (1 + odds_low_ex)

        random_draw = rng.random_sample(size=len(age_ge15_idx))
        df.loc[age_ge15_idx, 'li_low_ex'] = random_draw < low_ex_probs

        # -------------------- TOBACCO USE ---------------------------------------------------------

        init_odds_tob_age1519_m_wealth1 = (
            m.parameters['init_p_tob_age1519_m_wealth1']
            / (1 - m.parameters['init_p_tob_age1519_m_wealth1'])
        )

        odds_tob = pd.Series(init_odds_tob_age1519_m_wealth1, index=age_ge15_idx)

        odds_tob.loc[df.sex == 'F'] *= m.parameters['init_or_tob_f']
        odds_tob.loc[
            (df.sex == 'M') & (df.age_years >= 20) & (df.age_years < 40)
        ] *= m.parameters['init_or_tob_age2039_m']
        odds_tob.loc[
            (df.sex == 'M') & (df.age_years >= 40)
        ] *= m.parameters['init_or_tob_agege40_m']
        odds_tob.loc[df.li_wealth == 2] *= 2
        odds_tob.loc[df.li_wealth == 3] *= 3
        odds_tob.loc[df.li_wealth == 4] *= 4
        odds_tob.loc[df.li_wealth == 5] *= 5

        tob_probs = odds_tob / (1 + odds_tob)

        random_draw = rng.random_sample(size=len(age_ge15_idx))

        # decide on tobacco use based on the individual probability is greater than random draw
        # this is a list of True/False. assign to li_tob
        df.loc[age_ge15_idx, 'li_tob'] = random_draw < tob_probs

        # -------------------- EXCESSIVE ALCOHOL ---------------------------------------------------

        agege15_m_idx = df.index[df.is_alive & (df.age_years >= 15) & (df.sex == 'M')]
        agege15_f_idx = df.index[df.is_alive & (df.age_years >= 15) & (df.sex == 'F')]

        df.loc[agege15_m_idx, 'li_ex_alc'] = (
            rng.random_sample(size=len(agege15_m_idx)) < m.parameters['init_p_ex_alc_m']
        )
        df.loc[agege15_f_idx, 'li_ex_alc'] = (
            rng.random_sample(size=len(agege15_f_idx)) < m.parameters['init_p_ex_alc_f']
        )

        # -------------------- MARITAL STATUS ------------------------------------------------------

        age_15_19 = df.index[df.age_years.between(15, 19) & df.is_alive]
        age_20_29 = df.index[df.age_years.between(20, 29) & df.is_alive]
        age_30_39 = df.index[df.age_years.between(30, 39) & df.is_alive]
        age_40_49 = df.index[df.age_years.between(40, 49) & df.is_alive]
        age_50_59 = df.index[df.age_years.between(50, 59) & df.is_alive]
        age_gte60 = df.index[(df.age_years >= 60) & df.is_alive]

        df.loc[age_15_19, 'li_mar_stat'] = rng.choice(
            [1, 2, 3], size=len(age_15_19), p=m.parameters['init_dist_mar_stat_age1520']
        )
        df.loc[age_20_29, 'li_mar_stat'] = rng.choice(
            [1, 2, 3], size=len(age_20_29), p=m.parameters['init_dist_mar_stat_age2030']
        )
        df.loc[age_30_39, 'li_mar_stat'] = rng.choice(
            [1, 2, 3], size=len(age_30_39), p=m.parameters['init_dist_mar_stat_age3040']
        )
        df.loc[age_40_49, 'li_mar_stat'] = rng.choice(
            [1, 2, 3], size=len(age_40_49), p=m.parameters['init_dist_mar_stat_age4050']
        )
        df.loc[age_50_59, 'li_mar_stat'] = rng.choice(
            [1, 2, 3], size=len(age_50_59), p=m.parameters['init_dist_mar_stat_age5060']
        )
        df.loc[age_gte60, 'li_mar_stat'] = rng.choice(
            [1, 2, 3], size=len(age_gte60), p=m.parameters['init_dist_mar_stat_agege60']
        )

        # -------------------- EDUCATION -----------------------------------------------------------

        age_gte5 = df.index[(df.age_years >= 5) & df.is_alive]

        # calculate the probability of education for all individuals over 5 years old
        p_some_ed = pd.Series(m.parameters['init_age2030_w5_some_ed'], index=age_gte5)

        # adjust probability of some education based on age
        p_some_ed.loc[df.age_years < 13] *= m.parameters['init_rp_some_ed_age0513']
        p_some_ed.loc[df.age_years.between(13, 19)] *= m.parameters['init_rp_some_ed_age1320']
        p_some_ed.loc[df.age_years.between(30, 39)] *= m.parameters['init_rp_some_ed_age3040']
        p_some_ed.loc[df.age_years.between(40, 49)] *= m.parameters['init_rp_some_ed_age4050']
        p_some_ed.loc[df.age_years.between(50, 59)] *= m.parameters['init_rp_some_ed_age5060']
        p_some_ed.loc[(df.age_years >= 60)] *= m.parameters['init_rp_some_ed_agege60']

        # adjust probability of some education based on wealth
        p_some_ed *= m.parameters['init_rp_some_ed_per_higher_wealth'] ** (
            5 - pd.to_numeric(df.loc[age_gte5, 'li_wealth'])
        )

        # calculate baseline of education level 3, and adjust for age and wealth
        p_ed_lev_3 = pd.Series(m.parameters['init_prop_age2030_w5_some_ed_sec'], index=age_gte5)

        p_ed_lev_3.loc[(df.age_years < 13)] = 0
        p_ed_lev_3.loc[df.age_years.between(13, 19)] *= m.parameters['init_rp_some_ed_sec_age1320']
        p_ed_lev_3.loc[df.age_years.between(30, 39)] *= m.parameters['init_rp_some_ed_sec_age3040']
        p_ed_lev_3.loc[df.age_years.between(40, 49)] *= m.parameters['init_rp_some_ed_sec_age4050']
        p_ed_lev_3.loc[df.age_years.between(50, 59)] *= m.parameters['init_rp_some_ed_sec_age5060']
        p_ed_lev_3.loc[(df.age_years >= 60)] *= m.parameters['init_rp_some_ed_sec_agege60']
        p_ed_lev_3 *= m.parameters['init_rp_some_ed_sec_per_higher_wealth'] ** (
            5 - pd.to_numeric(df.loc[age_gte5, 'li_wealth'])
        )

        rnd_draw = pd.Series(rng.random_sample(size=len(age_gte5)), index=age_gte5)

        dfx = pd.concat([p_ed_lev_3, p_some_ed, rnd_draw], axis=1)
        dfx.columns = ['eff_prob_ed_lev_3', 'eff_prob_some_ed', 'random_draw_01']

        dfx['p_ed_lev_1'] = 1 - dfx['eff_prob_some_ed']
        dfx['p_ed_lev_3'] = dfx['eff_prob_ed_lev_3']
        dfx['cut_off_ed_levl_3'] = 1 - dfx['eff_prob_ed_lev_3']

        dfx['li_ed_lev'] = 2
        dfx.loc[dfx['cut_off_ed_levl_3'] < rnd_draw, 'li_ed_lev'] = 3
        dfx.loc[dfx['p_ed_lev_1'] > rnd_draw, 'li_ed_lev'] = 1

        df.loc[age_gte5, 'li_ed_lev'] = dfx['li_ed_lev']

        df.loc[df.age_years.between(5, 12) & (df['li_ed_lev'] == 1) & df.is_alive, 'li_in_ed'] = False
        df.loc[df.age_years.between(5, 12) & (df['li_ed_lev'] == 2) & df.is_alive, 'li_in_ed'] = True
        df.loc[df.age_years.between(13, 19) & (df['li_ed_lev'] == 3) & df.is_alive, 'li_in_ed'] = True

        # -------------------- UNIMPROVED SANITATION ---------------------------------------------------
        init_odds_unimproved_sanitation = m.parameters['init_p_unimproved_sanitation_urban'] / (
            1 - m.parameters['init_p_unimproved_sanitation_urban'])

        # create a series with odds of unimproved sanitation for base group (urban)
        odds_unimproved_sanitation = pd.Series(init_odds_unimproved_sanitation, index=alive_idx)

        # update odds according to determinants of unimproved sanitation (rural status the only determinant)
        odds_unimproved_sanitation.loc[df.is_alive & ~df.li_urban] *= (
            m.parameters['init_or_unimproved_sanitation_rural']
        )

        prob_unimproved_sanitation = odds_unimproved_sanitation / (1 + odds_unimproved_sanitation)

        random_draw = pd.Series(rng.random_sample(size=len(alive_idx)), index=alive_idx)

        df.loc[alive_idx, 'li_unimproved_sanitation'] = random_draw < prob_unimproved_sanitation

        # -------------------- NO CLEAN DRINKING WATER ---------------------------------------------------
        init_odds_no_clean_drinking_water = m.parameters['init_p_no_clean_drinking_water_urban'] / (
            1 - m.parameters['init_p_no_clean_drinking_water_urban']
        )

        odds_no_clean_drinking_water = pd.Series(init_odds_no_clean_drinking_water, index=alive_idx)

        odds_no_clean_drinking_water.loc[df.is_alive & ~df.li_urban] *= (
            m.parameters['init_or_no_clean_drinking_water_rural']
        )

        prob_no_clean_drinking_water = odds_no_clean_drinking_water / (1 + odds_no_clean_drinking_water)

        random_draw = pd.Series(rng.random_sample(size=len(alive_idx)), index=alive_idx)

        df.loc[alive_idx, 'li_no_clean_drinking_water'] = random_draw < prob_no_clean_drinking_water

        # -------------------- WOOD BURN STOVE ---------------------------------------------------

        init_odds_wood_burn_stove = (
            m.parameters['init_p_wood_burn_stove_urban']
            / (1 - m.parameters['init_p_wood_burn_stove_urban'])
        )

        odds_wood_burn_stove = pd.Series(init_odds_wood_burn_stove, index=alive_idx)

        odds_wood_burn_stove.loc[df.is_alive & ~df.li_urban] *= (
            m.parameters['init_or_wood_burn_stove_rural']
        )

        prob_wood_burn_stove = odds_wood_burn_stove / (1 + odds_wood_burn_stove)

        random_draw = pd.Series(rng.random_sample(size=len(alive_idx)), index=alive_idx)

        df.loc[alive_idx, 'li_wood_burn_stove'] = random_draw < prob_wood_burn_stove

        # -------------------- NO ACCESS HANDWASHING ---------------------------------------------------

        wealth2_idx = df.index[df.is_alive & (df.li_wealth == 2)]
        wealth3_idx = df.index[df.is_alive & (df.li_wealth == 3)]
        wealth4_idx = df.index[df.is_alive & (df.li_wealth == 4)]
        wealth5_idx = df.index[df.is_alive & (df.li_wealth == 5)]

        odds_no_access_handwashing = pd.Series(
            1 / (1 - m.parameters['init_p_no_access_handwashing_wealth1']), index=alive_idx
        )

        odds_no_access_handwashing.loc[wealth2_idx] *= (
            m.parameters['init_or_no_access_handwashing_per_lower_wealth']
        )
        odds_no_access_handwashing.loc[wealth3_idx] *= (
            m.parameters['init_or_no_access_handwashing_per_lower_wealth'] ** 2
        )
        odds_no_access_handwashing.loc[wealth4_idx] *= (
            m.parameters['init_or_no_access_handwashing_per_lower_wealth'] ** 3
        )
        odds_no_access_handwashing.loc[wealth5_idx] *= (
            m.parameters['init_or_no_access_handwashing_per_lower_wealth'] ** 4
        )

        prob_no_access_handwashing = odds_no_access_handwashing / (1 + odds_no_access_handwashing)

        random_draw = pd.Series(rng.random_sample(size=len(alive_idx)), index=alive_idx)

        df.loc[alive_idx, 'li_no_access_handwashing'] = random_draw < prob_no_access_handwashing

        # -------------------- SALT INTAKE ----------------------------------------------------------

        # create a series with odds of unimproved sanitation for base group (urban)
        odds_high_salt = pd.Series(
            m.parameters['init_p_high_salt_urban']
            / (1 - m.parameters['init_p_high_salt_urban']),
            index=alive_idx
        )

        # update odds according to determinants of unimproved sanitation (rural status the only determinant)
        odds_high_salt.loc[df.is_alive & ~df.li_urban] *= m.parameters['init_or_high_salt_rural']

        prob_high_salt = (odds_high_salt / (1 + odds_high_salt))

        random_draw = pd.Series(rng.random_sample(size=len(alive_idx)), index=alive_idx)

        df.loc[alive_idx, 'li_high_salt'] = random_draw < prob_high_salt

        # -------------------- SUGAR INTAKE ----------------------------------------------------------

        # no determinants of sugar intake hence dont need to convert to odds to apply odds ratios

        random_draw = pd.Series(rng.random_sample(size=len(alive_idx)), index=alive_idx)

        df.loc[alive_idx, 'li_high_sugar'] = random_draw < m.parameters['init_p_high_sugar']

        # -------------------- WEALTH LEVEL ----------------------------------------------------------

        urban_idx = df.index[df.is_alive & df.li_urban]
        rural_idx = df.index[df.is_alive & ~df.li_urban]

        # allocate wealth level for urban
        df.loc[urban_idx, 'li_wealth'] = rng.choice(
            [1, 2, 3, 4, 5], size=len(urban_idx), p=m.parameters['init_p_wealth_urban']
        )

        # allocate wealth level for rural
        df.loc[rural_idx, 'li_wealth'] = rng.choice(
            [1, 2, 3, 4, 5], size=len(rural_idx), p=m.parameters['init_p_wealth_rural']
        )

        # -------------------- BMI CATEGORIES ----------------------------------------------------------

        # only relevant if at least one individual with age >= 15 years present
        if len(age_ge15_idx) > 0:

            agege15_w_idx = df.index[df.is_alive & (df.sex == 'F') & (df.age_years >= 15)]
            agege15_rural_idx = df.index[df.is_alive & ~df.li_urban & (df.age_years >= 15)]
            agege15_high_sugar_idx = df.index[df.is_alive & df.li_high_sugar & (df.age_years >= 15)]
            agege3049_idx = df.index[df.is_alive & (df.age_years >= 30) & (df.age_years < 50)]
            agege50_idx = df.index[df.is_alive & (df.age_years >= 50)]
            agege15_tob_idx = df.index[df.is_alive & df.li_tob & (df.age_years >= 15)]
            agege15_wealth2_idx = df.index[df.is_alive & (df.li_wealth == 2) & (df.age_years >= 15)]
            agege15_wealth3_idx = df.index[df.is_alive & (df.li_wealth == 3) & (df.age_years >= 15)]
            agege15_wealth4_idx = df.index[df.is_alive & (df.li_wealth == 4) & (df.age_years >= 15)]
            agege15_wealth5_idx = df.index[df.is_alive & (df.li_wealth == 5) & (df.age_years >= 15)]

            # this below is the approach to apply the effect of contributing determinants on bmi levels at baseline
            # transform to odds, apply the odds ratio for the effect, transform back to probabilities and normalise
            # to sum to 1
            init_odds_bmi_urban_m_not_high_sugar_age1529_not_tob_wealth1 = [
                i / (1 - i) for i in m.parameters['init_p_bmi_urban_m_not_high_sugar_age1529_not_tob_wealth1']
            ]

            df_odds_probs_bmi_levels = pd.DataFrame(
                data=[init_odds_bmi_urban_m_not_high_sugar_age1529_not_tob_wealth1],
                columns=['1', '2', '3', '4', '5'],
                index=age_ge15_idx,
            )

            def update_df_odds_bmi(bmi: str, power: int):
                """Update specified bmi column using pattern and the power given"""
                df_odds_probs_bmi_levels.loc[agege15_w_idx, bmi] *= (
                    m.parameters['init_or_higher_bmi_f'] ** power
                )
                df_odds_probs_bmi_levels.loc[agege15_rural_idx, bmi] *= (
                    m.parameters['init_or_higher_bmi_rural'] ** power
                )
                df_odds_probs_bmi_levels.loc[agege15_high_sugar_idx, bmi] *= (
                    m.parameters['init_or_higher_bmi_high_sugar'] ** power
                )
                df_odds_probs_bmi_levels.loc[agege3049_idx, bmi] *= (
                    m.parameters['init_or_higher_bmi_age3049'] ** power
                )
                df_odds_probs_bmi_levels.loc[agege50_idx, bmi] *= (
                    m.parameters['init_or_higher_bmi_agege50'] ** power
                )
                df_odds_probs_bmi_levels.loc[agege15_tob_idx, bmi] *= (
                    m.parameters['init_or_higher_bmi_tob'] ** power
                )
                df_odds_probs_bmi_levels.loc[agege15_wealth2_idx, bmi] *= (
                    m.parameters['init_or_higher_bmi_per_higher_wealth_level'] ** 2) ** power
                df_odds_probs_bmi_levels.loc[agege15_wealth3_idx, bmi] *= (
                    m.parameters['init_or_higher_bmi_per_higher_wealth_level'] ** 3) ** power
                df_odds_probs_bmi_levels.loc[agege15_wealth4_idx, bmi] *= (
                    m.parameters['init_or_higher_bmi_per_higher_wealth_level'] ** 4) ** power
                df_odds_probs_bmi_levels.loc[agege15_wealth5_idx, bmi] *= (
                    m.parameters['init_or_higher_bmi_per_higher_wealth_level'] ** 5) ** power

            update_df_odds_bmi('1', -2)
            update_df_odds_bmi('2', -1)
            update_df_odds_bmi('3', 0)
            update_df_odds_bmi('4', 1)
            update_df_odds_bmi('5', 2)

            for bmi in range(1, 6):
                bmi = str(bmi)
                df_odds_probs_bmi_levels[f'prob {bmi}'] = df_odds_probs_bmi_levels.apply(
                    lambda row: row[bmi] / (1 + row[bmi]), axis=1
                )
            # normalise probabilities
            df_odds_probs_bmi_levels['sum_probs'] = df_odds_probs_bmi_levels.apply(
                lambda row: row['prob 1'] + row['prob 2'] + row['prob 3'] + row['prob 4'] + row['prob 5'], axis=1
            )
            for bmi in range(1, 6):
                df_odds_probs_bmi_levels[bmi] = df_odds_probs_bmi_levels.apply(
                    lambda row: row[f'prob {bmi}'] / row['sum_probs'], axis=1
                )

            dfxx = df_odds_probs_bmi_levels[[1, 2, 3, 4, 5]]

            # for each row, make a choice
            bmi_cat = dfxx.apply(lambda p_bmi: rng.choice(dfxx.columns, p=p_bmi), axis=1)

            df.loc[age_ge15_idx, 'li_bmi'] = bmi_cat

        # -------------------- SEX WORKER ----------------------------------------------------------
        # determine which women will be sex worker
        self.determine_who_will_be_sexworker(months_since_last_poll=0)

        # -------------------- MALE CIRCUMCISION ----------------------------------------------------------
        # determine the proportion of men that are circumcised at initiation
        # NB. this is determined with respect to any characteristics (eg. ethnicity or religion)
        men = df.loc[df.is_alive & (df.sex == 'M')]
        will_be_circ = self.rng.rand(len(men)) < self.parameters['proportion_of_men_circumcised_at_initiation']
        df.loc[men[will_be_circ].index, 'li_is_circ'] = True

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
        df = population.props
        m = self.module
        rng = m.rng

        # -------------------- URBAN-RURAL STATUS --------------------------------------------------

        # get index of current urban/rural status
        currently_rural = df.index[~df.li_urban & df.is_alive]
        currently_urban = df.index[df.li_urban & df.is_alive]

        # handle new transitions
        rural_to_urban = currently_rural[m.rng.random_sample(size=len(currently_rural)) < m.parameters['r_urban']]
        df.loc[rural_to_urban, 'li_urban'] = True

        # handle new transitions to rural
        urban_to_rural = currently_urban[rng.random_sample(size=len(currently_urban)) < m.parameters['r_rural']]
        df.loc[urban_to_rural, 'li_urban'] = False

        # -------------------- LOW EXERCISE --------------------------------------------------------

        adults_not_low_ex = df.index[~df.li_low_ex & df.is_alive & (df.age_years >= 15)]
        eff_p_low_ex = pd.Series(m.parameters['r_low_ex'], index=adults_not_low_ex)
        eff_p_low_ex.loc[df.sex == 'F'] *= m.parameters['rr_low_ex_f']
        eff_p_low_ex.loc[df.li_urban] *= m.parameters['rr_low_ex_urban']
        df.loc[adults_not_low_ex, 'li_low_ex'] = rng.random_sample(len(adults_not_low_ex)) < eff_p_low_ex

        # transition from low exercise to not low exercise
        low_ex_idx = df.index[df.li_low_ex & df.is_alive]
        eff_rate_not_low_ex = pd.Series(m.parameters['r_not_low_ex'], index=low_ex_idx)
        eff_rate_not_low_ex.loc[df.li_exposed_to_campaign_exercise_increase] *= (
            m.parameters['rr_not_low_ex_pop_advice_exercise']
        )
        random_draw = rng.random_sample(len(low_ex_idx))
        newly_not_low_ex_idx = low_ex_idx[random_draw < eff_rate_not_low_ex]
        df.loc[newly_not_low_ex_idx, 'li_low_ex'] = False

        # todo: this line below to start a general population campaign
        #  to increase exercise not working yet (same for others below)
        all_idx_campaign_exercise_increase = df.index[
            df.is_alive & (self.sim.date == m.parameters['start_date_campaign_exercise_increase'])
        ]
        df.loc[all_idx_campaign_exercise_increase, 'li_exposed_to_campaign_exercise_increase'] = True

        # -------------------- TOBACCO USE ---------------------------------------------------------

        adults_not_tob = df.index[(df.age_years >= 15) & df.is_alive & ~df.li_tob]

        # start tobacco use
        eff_p_tob = pd.Series(m.parameters['r_tob'], index=adults_not_tob)
        eff_p_tob.loc[(df.age_years >= 20) & (df.age_years < 40)] *= m.parameters['rr_tob_age2039']
        eff_p_tob.loc[df.age_years >= 40] *= m.parameters['rr_tob_agege40']
        eff_p_tob.loc[df.sex == 'F'] *= m.parameters['rr_tob_f']
        eff_p_tob *= m.parameters['rr_tob_wealth'] ** (pd.to_numeric(df.loc[adults_not_tob, 'li_wealth']) - 1)

        df.loc[adults_not_tob, 'li_tob'] = rng.random_sample(len(adults_not_tob)) < eff_p_tob

        # transition from tobacco to no tobacco
        tob_idx = df.index[df.li_tob & df.is_alive]
        eff_rate_not_tob = pd.Series(m.parameters['r_not_tob'], index=tob_idx)
        eff_rate_not_tob.loc[df.li_exposed_to_campaign_quit_smoking] *= (
            m.parameters['rr_not_tob_pop_advice_tobacco']
        )
        random_draw = rng.random_sample(len(tob_idx))
        newly_not_tob_idx = tob_idx[random_draw < eff_rate_not_tob]
        df.loc[newly_not_tob_idx, 'li_tob'] = False
        df.loc[newly_not_tob_idx, 'li_date_not_tob'] = self.sim.date

        all_idx_campaign_quit_smoking = df.index[
            df.is_alive & (self.sim.date == m.parameters['start_date_campaign_quit_smoking'])
        ]
        df.loc[all_idx_campaign_quit_smoking, 'li_exposed_to_campaign_quit_smoking'] = True

        # -------------------- EXCESSIVE ALCOHOL ---------------------------------------------------

        not_ex_alc_f = df.index[~df.li_ex_alc & df.is_alive & (df.sex == 'F') & (df.age_years >= 15)]
        not_ex_alc_m = df.index[~df.li_ex_alc & df.is_alive & (df.sex == 'M') & (df.age_years >= 15)]
        now_ex_alc = df.index[df.li_ex_alc & df.is_alive]

        df.loc[not_ex_alc_f, 'li_ex_alc'] = (
            rng.random_sample(len(not_ex_alc_f))
            < m.parameters['r_ex_alc'] * m.parameters['rr_ex_alc_f']
        )
        df.loc[not_ex_alc_m, 'li_ex_alc'] = rng.random_sample(len(not_ex_alc_m)) < m.parameters['r_ex_alc']
        df.loc[now_ex_alc, 'li_ex_alc'] = ~(rng.random_sample(len(now_ex_alc)) < m.parameters['r_not_ex_alc'])

        # transition from excess alcohol to not excess alcohol
        ex_alc_idx = df.index[df.li_ex_alc & df.is_alive]
        eff_rate_not_ex_alc = pd.Series(m.parameters['r_not_ex_alc'], index=ex_alc_idx)
        eff_rate_not_ex_alc.loc[
            df.li_exposed_to_campaign_alcohol_reduction
        ] *= m.parameters['rr_not_ex_alc_pop_advice_alcohol']
        random_draw = rng.random_sample(len(ex_alc_idx))
        newly_not_ex_alc_idx = ex_alc_idx[random_draw < eff_rate_not_ex_alc]
        df.loc[newly_not_ex_alc_idx, 'li_ex_alc'] = False

        all_idx_campaign_alcohol_reduction = df.index[
            df.is_alive & (self.sim.date == m.parameters['start_date_campaign_alcohol_reduction'])
        ]
        df.loc[all_idx_campaign_alcohol_reduction, 'li_exposed_to_campaign_alcohol_reduction'] = True

        # -------------------- MARITAL STATUS ------------------------------------------------------

        curr_never_mar = df.index[df.is_alive & df.age_years.between(15, 29) & (df.li_mar_stat == 1)]
        curr_mar = df.index[df.is_alive & (df.li_mar_stat == 2)]

        # update if now married
        now_mar = rng.random_sample(len(curr_never_mar)) < m.parameters['r_mar']
        df.loc[curr_never_mar[now_mar], 'li_mar_stat'] = 2

        # update if now divorced/widowed
        now_div_wid = rng.random_sample(len(curr_mar)) < m.parameters['r_div_wid']
        df.loc[curr_mar[now_div_wid], 'li_mar_stat'] = 3

        # -------------------- EDUCATION -----------------------------------------------------------

        # get all individuals currently in education
        in_ed = df.index[df.is_alive & df.li_in_ed]

        # ---- PRIMARY EDUCATION

        # get index of all children who are alive and between 5 and 5.25 years old
        age5 = df.index[(df.age_exact_years >= 5) & (df.age_exact_years < 5.25) & df.is_alive]

        # by default, these children are not in education and have education level 1
        df.loc[age5, 'li_ed_lev'] = 1
        df.loc[age5, 'li_in_ed'] = False

        # create a series to hold the probablity of primary education for children at age 5
        prob_primary = pd.Series(m.parameters['p_ed_primary'], index=age5)
        prob_primary *= m.parameters['rp_ed_primary_higher_wealth'] ** (5 - pd.to_numeric(df.loc[age5, 'li_wealth']))

        # randomly select some to have primary education
        age5_in_primary = rng.random_sample(len(age5)) < prob_primary
        df.loc[age5[age5_in_primary], 'li_ed_lev'] = 2
        df.loc[age5[age5_in_primary], 'li_in_ed'] = True

        # ---- SECONDARY EDUCATION

        # get thirteen year olds that are in primary education, any wealth level
        age13_in_primary = df.index[(df.age_years == 13) & df.is_alive & df.li_in_ed & (df.li_ed_lev == 2)]

        # they have a probability of gaining secondary education (level 3), based on wealth
        prob_secondary = pd.Series(m.parameters['p_ed_secondary'], index=age13_in_primary)
        prob_secondary *= (
            m.parameters['rp_ed_secondary_higher_wealth']
            ** (5 - pd.to_numeric(df.loc[age13_in_primary, 'li_wealth']))
        )

        # randomly select some to get secondary education
        age13_to_secondary = rng.random_sample(len(age13_in_primary)) < prob_secondary
        df.loc[age13_in_primary[age13_to_secondary], 'li_ed_lev'] = 3

        # those who did not go on to secondary education are no longer in education
        df.loc[age13_in_primary[~age13_to_secondary], 'li_in_ed'] = False

        # ---- DROP OUT OF EDUCATION

        # baseline rate of leaving education then adjust for wealth level
        p_leave_ed = pd.Series(m.parameters['r_stop_ed'], index=in_ed)
        p_leave_ed *= (
            m.parameters['rr_stop_ed_lower_wealth']
            ** (pd.to_numeric(df.loc[in_ed, 'li_wealth']) - 1)
        )

        # randomly select some individuals to leave education
        now_not_in_ed = rng.random_sample(len(in_ed)) < p_leave_ed

        df.loc[in_ed[now_not_in_ed], 'li_in_ed'] = False

        # everyone leaves education at age 20
        df.loc[df.is_alive & df.li_in_ed & (df.age_years == 20), 'li_in_ed'] = False

        # -------------------- UNIMPROVED SANITATION --------------------------------------------------------

        # probability of improved sanitation at all follow-up times
        unimproved_sanitaton_idx = df.index[df.li_unimproved_sanitation & df.is_alive]

        eff_rate_improved_sanitation = pd.Series(
            m.parameters['r_improved_sanitation'], index=unimproved_sanitaton_idx
        )

        random_draw = rng.random_sample(len(unimproved_sanitaton_idx))

        newly_improved_sanitation_idx = unimproved_sanitaton_idx[random_draw < eff_rate_improved_sanitation]
        df.loc[newly_improved_sanitation_idx, 'li_unimproved_sanitation'] = False
        df.loc[newly_improved_sanitation_idx, 'li_date_acquire_improved_sanitation'] = self.sim.date

        # probability of improved sanitation upon moving to urban from rural
        unimproved_sanitation_newly_urban_idx = df.index[
            df.li_unimproved_sanitation & df.is_alive & (df.li_date_trans_to_urban == self.sim.date)
            ]

        random_draw = rng.random_sample(len(unimproved_sanitation_newly_urban_idx))

        eff_prev_unimproved_sanitation_urban = pd.Series(
            m.parameters['init_p_unimproved_sanitation_urban'], index=unimproved_sanitation_newly_urban_idx
        )

        df.loc[unimproved_sanitation_newly_urban_idx, 'li_unimproved_sanitation'] = (
            random_draw < eff_prev_unimproved_sanitation_urban
        )

        # -------------------- NO ACCESS HANDWASHING --------------------------------------------------------

        # probability of moving to access to handwashing at all follow-up times
        no_access_handwashing_idx = df.index[df.li_no_access_handwashing & df.is_alive]

        eff_rate_access_handwashing = pd.Series(m.parameters['r_access_handwashing'], index=no_access_handwashing_idx)

        random_draw = rng.random_sample(len(no_access_handwashing_idx))

        newly_access_handwashing_idx = no_access_handwashing_idx[random_draw < eff_rate_access_handwashing]
        df.loc[newly_access_handwashing_idx, 'li_no_access_handwashing'] = False
        df.loc[newly_access_handwashing_idx, 'li_date_acquire_access_handwashing'] = self.sim.date

        # -------------------- NO CLEAN DRINKING WATER  --------------------------------------------------------

        # probability of moving to clean drinking water at all follow-up times
        no_clean_drinking_water_idx = df.index[df.li_no_clean_drinking_water & df.is_alive]

        eff_rate_clean_drinking_water = pd.Series(
            m.parameters['r_clean_drinking_water'], index=no_clean_drinking_water_idx
        )

        random_draw = rng.random_sample(len(no_clean_drinking_water_idx))

        newly_clean_drinking_water_idx = no_clean_drinking_water_idx[random_draw < eff_rate_clean_drinking_water]
        df.loc[newly_clean_drinking_water_idx, 'li_no_clean_drinking_water'] = False
        df.loc[newly_clean_drinking_water_idx, 'li_date_acquire_clean_drinking_water'] = self.sim.date

        # probability of no clean drinking water upon moving to urban from rural
        no_clean_drinking_water_newly_urban_idx = df.index[
            df.li_no_clean_drinking_water & df.is_alive & (df.li_date_trans_to_urban == self.sim.date)
            ]

        random_draw = rng.random_sample(len(no_clean_drinking_water_newly_urban_idx))

        eff_prev_no_clean_drinking_water_urban = pd.Series(
            m.parameters['init_p_no_clean_drinking_water_urban'], index=no_clean_drinking_water_newly_urban_idx
        )

        df.loc[no_clean_drinking_water_newly_urban_idx, 'li_no_clean_drinking_water'] = (
            random_draw < eff_prev_no_clean_drinking_water_urban
        )

        # -------------------- WOOD BURN STOVE -------------------------------------------------------------

        # probability of moving to non wood burn stove at all follow-up times
        wood_burn_stove_idx = df.index[df.li_wood_burn_stove & df.is_alive]

        eff_rate_non_wood_burn_stove = pd.Series(m.parameters['r_non_wood_burn_stove'], index=wood_burn_stove_idx)

        random_draw = rng.random_sample(len(wood_burn_stove_idx))

        newly_non_wood_burn_stove_idx = wood_burn_stove_idx[random_draw < eff_rate_non_wood_burn_stove]
        df.loc[newly_non_wood_burn_stove_idx, 'li_wood_burn_stove'] = False
        df.loc[newly_non_wood_burn_stove_idx, 'li_date_acquire_non_wood_burn_stove'] = self.sim.date

        # probability of moving to wood burn stove upon moving to urban from rural
        wood_burn_stove_newly_urban_idx = df.index[
            df.li_wood_burn_stove & df.is_alive & (df.li_date_trans_to_urban == self.sim.date)
            ]

        random_draw = rng.random_sample(len(wood_burn_stove_newly_urban_idx))

        eff_prev_wood_burn_stove_urban = pd.Series(
            m.parameters['init_p_wood_burn_stove_urban'], index=wood_burn_stove_newly_urban_idx
        )

        df.loc[wood_burn_stove_newly_urban_idx, 'li_wood_burn_stove'] = random_draw < eff_prev_wood_burn_stove_urban

        # -------------------- HIGH SALT ----------------------------------------------------------

        not_high_salt_idx = df.index[~df.li_high_salt & df.is_alive]
        eff_rate_high_salt = pd.Series(m.parameters['r_high_salt_urban'], index=not_high_salt_idx)
        eff_rate_high_salt[df.li_urban] *= m.parameters['rr_high_salt_rural']
        random_draw = rng.random_sample(len(not_high_salt_idx))
        newly_high_salt = random_draw < eff_rate_high_salt
        newly_high_salt_idx = not_high_salt_idx[newly_high_salt]
        df.loc[newly_high_salt_idx, 'li_high_salt'] = True

        # transition from high salt to not high salt
        high_salt_idx = df.index[df.li_high_salt & df.is_alive]
        eff_rate_not_high_salt = pd.Series(m.parameters['r_not_high_salt'], index=high_salt_idx)
        eff_rate_not_high_salt.loc[
            df.li_exposed_to_campaign_salt_reduction
        ] *= m.parameters['rr_not_high_salt_pop_advice_salt']
        random_draw = rng.random_sample(len(high_salt_idx))
        newly_not_high_salt_idx = high_salt_idx[random_draw < eff_rate_not_high_salt]
        df.loc[newly_not_high_salt_idx, 'li_high_salt'] = False

        all_idx_campaign_salt_reduction = df.index[df.is_alive & (self.sim.date == datetime.date(2010, 7, 1))]
        df.loc[all_idx_campaign_salt_reduction, 'li_exposed_to_campaign_salt_reduction'] = True

        # -------------------- HIGH SUGAR ----------------------------------------------------------

        not_high_sugar_idx = df.index[~df.li_high_sugar & df.is_alive]
        eff_p_high_sugar = pd.Series(m.parameters['r_high_sugar'], index=not_high_sugar_idx)
        random_draw = rng.random_sample(len(not_high_sugar_idx))
        newly_high_sugar_idx = not_high_sugar_idx[random_draw < eff_p_high_sugar]
        df.loc[newly_high_sugar_idx, 'li_high_sugar'] = True

        # transition from high sugar to not high sugar
        high_sugar_idx = df.index[df.li_high_sugar & df.is_alive]
        eff_rate_not_high_sugar = pd.Series(m.parameters['r_not_high_sugar'], index=high_sugar_idx)
        eff_rate_not_high_sugar.loc[
            df.li_exposed_to_campaign_sugar_reduction
        ] *= m.parameters['rr_not_high_sugar_pop_advice_sugar']
        random_draw = rng.random_sample(len(high_sugar_idx))
        newly_not_high_sugar_idx = high_sugar_idx[random_draw < eff_rate_not_high_sugar]
        df.loc[newly_not_high_sugar_idx, 'li_high_sugar'] = False

        all_idx_campaign_sugar_reduction = df.index[df.is_alive & (self.sim.date == datetime.date(2010, 7, 1))]
        df.loc[all_idx_campaign_sugar_reduction, 'li_exposed_to_campaign_sugar_reduction'] = True

        # -------------------- BMI ----------------------------------------------------------

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

        random_draw = rng.random_sample(len(bmi_cat_1_to_4_idx))
        newly_increase_bmi_cat_idx = bmi_cat_1_to_4_idx[random_draw < eff_rate_higher_bmi]
        df.loc[newly_increase_bmi_cat_idx, 'li_bmi'] = df['li_bmi'] + 1

        # possible decrease in category of bmi

        bmi_cat_3_to_5_idx = df.index[df.is_alive & df.li_bmi.between(3, 5) & (df.age_years >= 15)]
        eff_rate_lower_bmi = pd.Series(m.parameters['r_lower_bmi'], index=bmi_cat_3_to_5_idx)
        eff_rate_lower_bmi[df.li_urban] *= m.parameters['rr_lower_bmi_tob']
        eff_rate_lower_bmi.loc[
            df.li_exposed_to_campaign_weight_reduction
        ] *= m.parameters['rr_lower_bmi_pop_advice_weight']
        random_draw = rng.random_sample(len(bmi_cat_3_to_5_idx))
        newly_decrease_bmi_cat_idx = bmi_cat_3_to_5_idx[random_draw < eff_rate_lower_bmi]
        df.loc[newly_decrease_bmi_cat_idx, 'li_bmi'] = df['li_bmi'] - 1

        all_idx_campaign_weight_reduction = df.index[df.is_alive & (self.sim.date == datetime.date(2010, 7, 1))]
        df.loc[all_idx_campaign_weight_reduction, 'li_exposed_to_campaign_weight_reduction'] = True

        # --- FSW ---
        self.module.determine_who_will_be_sexworker(months_since_last_poll=self.repeat_months)


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

        # TODO *** THIS HAS TROUBLE BE PARESED ON LONG RUNS BY PARSE_OUTPUT: CHANGING KEYS DUE TO GROUPBY? \
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

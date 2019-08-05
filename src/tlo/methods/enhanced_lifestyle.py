
"""
Lifestyle module
Documentation: 04 - Methods Repository/Method_Lifestyle.xlsx
"""
import logging

import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# todo: Note: bmi category at turning age 15 needs to be made dependent on malnutrition in childhood when that is coded.

class Lifestyle(Module):
    """
    Lifestyle module provides properties that are used by all disease modules if they are affected
    by urban/rural, wealth, tobacco usage etc.
    """
    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    PARAMETERS = {

        # -------- list of parameters -----------------------------------------------------------------------------------

        'init_p_urban': Parameter(Types.REAL, 'initial proportion urban'),
        'init_p_wealth_urban': Parameter(Types.LIST, 'List of probabilities of category given urban'),
        'init_p_wealth_rural': Parameter(Types.LIST, 'List of probabilities of category given rural'),
        'init_p_bmi_urban_m_not_high_sugar_age1529_not_tob_wealth1': Parameter(Types.LIST, 'List of probabilities of '
                                                                                           'bmi categories for urban '
                                                                                           'men age 15-29 with not high'
                                                                                           'sugar, not tobacco, '
                                                                                           'wealth level 1'),
        'init_or_higher_bmi_f': Parameter(Types.REAL, 'odds ratio higher BMI if female'),
        'init_or_higher_bmi_rural': Parameter(Types.REAL, 'odds ratio higher BMI if rural'),
        'init_or_higher_bmi_high_sugar': Parameter(Types.REAL, 'odds ratio higher BMI if high sugar intake'),
        'init_or_higher_bmi_age3049': Parameter(Types.REAL, 'odds ratio higher BMI if age 30-49'),
        'init_or_higher_bmi_agege50': Parameter(Types.REAL, 'odds ratio higher BMI if age ge 50'),
        'init_or_higher_bmi_tob': Parameter(Types.REAL, 'odds ratio higher BMI if use tobacco'),
        'init_or_higher_bmi_per_higher_wealth': Parameter(Types.REAL, 'odds ratio higher BMI per higer wealth level'),
        'init_p_high_sugar': Parameter(Types.REAL, 'initital proportion with high sugar intake'),
        'init_p_high_salt_urban': Parameter(Types.REAL, 'initital proportion with high salt intake'),
        'init_or_high_salt_rural': Parameter(Types.REAL, 'odds ratio high salt if rural'),
        'init_p_ex_alc_m': Parameter(Types.REAL, 'initital proportion of men with excess alcohol use'),
        'init_p_ex_alc_f': Parameter(Types.REAL, 'initital proportion of women with excess alcohol use'),
        'init_dist_mar_stat_age1520': Parameter(Types.LIST, 'proportions never, current, div_wid age 15-20 baseline'),
        'init_dist_mar_stat_age2030': Parameter(Types.LIST, 'proportions never, current, div_wid age 20-30 baseline'),
        'init_dist_mar_stat_age3040': Parameter(Types.LIST, 'proportions never, current, div_wid age 30-40 baseline'),
        'init_dist_mar_stat_age4050': Parameter(Types.LIST, 'proportions never, current, div_wid age 40-50 baseline'),
        'init_dist_mar_stat_age5060': Parameter(Types.LIST, 'proportions never, current, div_wid age 50-60 baseline'),
        'init_dist_mar_stat_agege60': Parameter(Types.LIST, 'proportions never, current, div_wid age 60+ baseline'),
        'init_age2030_w5_some_ed': Parameter(Types.REAL, 'proportions of low wealth 20-30 year olds with some '
                                                         'education at baseline'),
        'init_or_some_ed_age0513': Parameter(Types.REAL, 'odds ratio of some education at baseline age 5-13'),
        'init_or_some_ed_age1320': Parameter(Types.REAL, 'odds ratio of some education at baseline age 13-20'),
        'init_or_some_ed_age2030': Parameter(Types.REAL, 'odds ratio of some education at baseline age 20-30'),
        'init_or_some_ed_age3040': Parameter(Types.REAL, 'odds ratio of some education at baseline age 30-40'),
        'init_or_some_ed_age4050': Parameter(Types.REAL, 'odds ratio of some education at baseline age 40-50'),
        'init_or_some_ed_age5060': Parameter(Types.REAL, 'odds ratio of some education at baseline age 50-60'),
        'init_or_some_ed_per_higher_wealth': Parameter(Types.REAL, 'odds ratio of some education at baseline '
                                                                   'per higher wealth level'),
        'init_prop_age2030_w5_some_ed_sec': Parameter(Types.REAL,
                                                      'proportion of low wealth aged 20-30 with some education who '
                                                      'have secondary education at baseline'),
        'init_or_some_ed_sec_age1320': Parameter(Types.REAL, 'odds ratio of secondary education age 13-20'),
        'init_or_some_ed_sec_age3040': Parameter(Types.REAL, 'odds ratio of secondary education age 30-40'),
        'init_or_some_ed_sec_age4050': Parameter(Types.REAL, 'odds ratio of secondary education age 40-50'),
        'init_or_some_ed_sec_age5060': Parameter(Types.REAL, 'odds ratio of secondary education age 50-60'),
        'init_or_some_ed_sec_agege60': Parameter(Types.REAL, 'odds ratio of secondary education age 60+'),
        'init_or_some_ed_sec_per_higher_wealth': Parameter(Types.REAL, 'odds ratio of secondary education '
                                                                       'per higher wealth level'),
        'init_p_unimproved_sanitation': Parameter(Types.REAL, 'initial probability of unimproved_sanitation '
                                                              'given urban'),
        # note that init_p_unimproved_sanitation is also used as the one-off probability of unimproved_sanitation '
        #                                                     'true to false upon move from rural to urban'
        'init_or_unimproved_sanitation_rural': Parameter(Types.REAL,
                                                         'initial odds ratio of unimproved_sanitation if '
                                                         'rural'),
        'init_p_no_clean_drinking_water': Parameter(Types.REAL,
                                                    'initial probability of no_clean_drinking_water given urban'),
        # note that init_p_no_clean_drinking_water is also used as the one-off probability of no_clean_drinking_water '
        #                                                     'true to false upon move from rural to urban'
        'init_or_no_clean_drinking_water_rural': Parameter(Types.REAL,
                                                           'initial odds ratio of no clean drinking_water '
                                                           'if rural'),
        'init_p_wood_burn_stove': Parameter(Types.REAL,
                                            'initial probability of wood_burn_stove given urban'),
        # note that init_p_wood_burn_stove is also used as the one-off probability of wood_burn_stove '
        #                                                     'true to false upon move from rural to urban'
        'init_or_wood_burn_stove_rural': Parameter(Types.REAL,
                                                   'initial odds ratio of wood_burn_stove if rural'),
        'init_p_no_access_handwashing': Parameter(Types.REAL,
                                                  'initial probability of no_access_handwashing given wealth 1'),
        'init_or_no_access_handwashing_per_lower_wealth': Parameter(Types.REAL, 'initial odds ratio of no_'
                                                                                'access_handwashing per lower wealth '
                                                                                'level'),

        # ------------ parameters relating to updating of property values over time ------------------------

        'r_urban': Parameter(Types.REAL, 'probability per 3 months of change from rural to urban'),
        'r_rural': Parameter(Types.REAL, 'probability per 3 months of change from urban to rural'),
        'r_higher_bmi': Parameter(Types.REAL, 'probability per 3 months of increase in bmi category if rural male age'
                                              '15-29 not using tobacoo with wealth level 1 with not high sugar intake'),
        'rr_higher_bmi_urban': Parameter(Types.REAL, 'probability per 3 months of increase in bmi category if '),
        'rr_higher_bmi_f': Parameter(Types.REAL, 'rate ratio for increase in bmi category for females'),
        'rr_higher_bmi_age3049': Parameter(Types.REAL, 'rate ratio for increase in bmi category for age 30-49'),
        'rr_higher_bmi_agege50': Parameter(Types.REAL, 'rate ratio for increase in bmi category for age ge 50'),
        'rr_higher_bmi_tob': Parameter(Types.REAL, 'rate ratio for increase in bmi category for tobacco users'),
        'rr_higher_bmi_per_higher_wealth': Parameter(Types.REAL, 'rate ratio for increase in bmi category per higher '
                                                                 'wealth'),
        'rr_higher_bmi_high_sugar': Parameter(Types.REAL, 'rate ratio for increase in bmi category for high sugar '
                                                          'intake'),
        'r_lower_bmi': Parameter(Types.REAL, 'probability per 3 months of decrease in bmi category in non tobacco users'),
        'rr_lower_bmi_pop_advice_weight': Parameter(Types.REAL, 'probability per 3 months of decrease in bmi category '
                                                                'given population advice/campaign on weight'),
        'rr_lower_bmi_tob': Parameter(Types.REAL, 'rate ratio for lower bmi category for tobacco users'),
        'r_high_salt_urban': Parameter(Types.REAL, 'probability per 3 months of high salt intake if urban'),
        'rr_high_salt_rural': Parameter(Types.REAL, 'rate ratio for high salt if rural'),
        'r_not_high_salt': Parameter(Types.REAL, 'probability per 3 months of not high salt intake'),
        'rr_not_high_salt_pop_advice_salt': Parameter(Types.REAL, 'probability per 3 months of not high salt given'
                                                                  'population advice/campaign on salt'),
        'r_high_sugar': Parameter(Types.REAL, 'probability per 3 months of high sugar intake'),
        'r_not_high_sugar': Parameter(Types.REAL, 'probability per 3 months of not high sugar intake'),
        'r_low_ex': Parameter(Types.REAL, 'probability per 3 months of change from not low exercise to low exercise'),
        'r_not_low_ex': Parameter(Types.REAL, 'probability per 3 months of change from low exercise to not low exercie'),
        'rr_not_high_sugar_pop_advice_salt': Parameter(Types.REAL, 'probability per 3 months of not high sugar given'
                                                                   'population advice/campaign on sugar'),
        'rr_low_ex_f': Parameter(Types.REAL, 'risk ratio for becoming low exercise if female rather than male'),
        'rr_low_ex_urban': Parameter(Types.REAL, 'risk ratio for becoming low exercise if urban rather than rural'),
        'r_tob': Parameter(Types.REAL, 'probability per 3 months of change from not using tobacco to using '
                                       'tobacco if male age 15-19 wealth level 1'),
        'r_not_tob': Parameter(Types.REAL, 'probability per 3 months of change from tobacco using to '
                                           'not tobacco using'),
        'rr_tob_age2039': Parameter(Types.REAL, 'risk ratio for tobacco using if age 20-39 compared with 15-19'),
        'rr_tob_agege40': Parameter(Types.REAL, 'risk ratio for tobacco using if age >= 40 compared with 15-19'),
        'rr_tob_f': Parameter(Types.REAL, 'risk ratio for tobacco using if female'),
        'rr_tob_wealth': Parameter(Types.REAL, 'risk ratio for tobacco using per 1 higher wealth level '
                                               '(higher wealth level = lower wealth)'),
        'rr_not_tob_pop_advice_tobacco': Parameter(Types.REAL, 'probability per 3 months of quitting tobacco given'
                                                               'population advice/campaign on tobacco'),
        'r_ex_alc': Parameter(Types.REAL, 'probability per 3 months of change from not excess alcohol to '
                                          'excess alcohol'),
        'r_not_ex_alc': Parameter(Types.REAL, 'probability per 3 months of change from excess alcohol to '
                                              'not excess alcohol'),
        'rr_ex_alc_f': Parameter(Types.REAL, 'risk ratio for becoming excess alcohol if female rather than male'),
        'rr_not_tob_pop_advice_alc': Parameter(Types.REAL, 'probability per 3 months of not excess alcohol given'
                                                           'population advice/campaign on alcohol'),
        'r_mar': Parameter(Types.REAL, 'probability per 3 months of marriage when age 15-30'),
        'r_div_wid': Parameter(Types.REAL, 'probability per 3 months of becoming divorced or widowed, '
                                           'amongst those married'),
        'r_stop_ed': Parameter(Types.REAL, 'probabilities per 3 months of stopping education if wealth level 5'),
        'rr_stop_ed_lower_wealth': Parameter(Types.REAL, 'relative rate of stopping education per '
                                                         '1 lower wealth quintile'),
        'p_ed_primary': Parameter(Types.REAL, 'probability at age 5 that start primary education if wealth level 5'),
        'rp_ed_primary_higher_wealth': Parameter(Types.REAL, 'relative probability of starting school per 1 '
                                                             'higher wealth level'),
        'p_ed_secondary': Parameter(Types.REAL, 'probability at age 13 that start secondary education at 13 '
                                                'if in primary education and wealth level 5'),
        'rp_ed_secondary_higher_wealth': Parameter(Types.REAL, 'relative probability of starting secondary '
                                                               'school per 1 higher wealth level'),
        'r_improved_sanitation': Parameter(Types.REAL, 'probability per 3 months of change from '
                                                       'unimproved_sanitation true to false'),
        'r_clean_drinking_water': Parameter(Types.REAL, 'probability per 3 months of change from '
                                                        'drinking_water true to false'),
        'r_non_wood_burn_stove': Parameter(Types.REAL, 'probability per 3 months of change from '
                                                       'wood_burn_stove true to false'),
        'r_access_handwashing': Parameter(Types.REAL, 'probability per 3 months of change from '
                                                      'no_access_handwashing true to false')
    }

    # Properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'li_urban': Property(Types.BOOL, 'Currently urban'),
        'li_date_trans_to_urban': Property(Types.DATE, 'date of transition to urban'),
        'li_wealth': Property(Types.CATEGORICAL, 'wealth level: 1 (high) to 5 (low)', categories=[1, 2, 3, 4, 5]),
        'li_bmi': Property(Types.CATEGORICAL, 'bmi category: 1 (high) to 5 (low)', categories=[1, 2, 3, 4, 5]),
        'li_low_ex': Property(Types.BOOL, 'currently low exercise'),
        'li_high_salt': Property(Types.BOOL, 'currently high salt intake'),
        'li_high_sugar': Property(Types.BOOL, 'currently high sugar intake'),
        'li_date_no_longer_low_ex': Property(Types.DATE, 'li_date_no_longer_low_ex'),
        'li_tob': Property(Types.BOOL, 'current using tobacco'),
        'li_date_quit_tob': Property(Types.DATE, 'li_date_quit_tob'),
        'li_ex_alc': Property(Types.BOOL, 'current excess alcohol'),
        'li_date_no_longer_ex_alc': Property(Types.DATE, 'li_date_no_longer_ex_alc'),
        'li_mar_stat': Property(Types.CATEGORICAL,
                                'marital status {1:never, 2:current, 3:past (widowed or divorced)}',
                                categories=[1, 2, 3]),
        'li_in_ed': Property(Types.BOOL, 'currently in education'),
        'li_ed_lev': Property(Types.CATEGORICAL, 'education level achieved as of now', categories=[1, 2, 3]),
        'li_unimproved_sanitation': Property(Types.BOOL, 'uninproved sanitation - anything other than own or '
                                                         'shared latrine'),
        'li_no_access_handwashing': Property(Types.BOOL, 'no_access_handwashing - no water, no soap, no other '
                                                         'cleaning agent - as in DHS'),
        'li_no_clean_drinking_water': Property(Types.BOOL, 'no drinking water from an improved source'),
        'li_wood_burn_stove': Property(Types.BOOL, 'wood (straw / crop)-burning stove')
    }


    def read_parameters(self, data_folder):
        p = self.parameters
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_LIfestyle_Enhanced.xlsx', sheet_name='parameter_values')

        dfd.set_index('parameter_name', inplace=True)

        p['init_p_urban'] = dfd.loc['init_p_urban', 'value1']
        p['init_p_wealth_urban'] = \
                    [dfd.loc['init_p_wealth_urban', 'value1'], dfd.loc['init_p_wealth_urban', 'value2'],
                     dfd.loc['init_p_wealth_urban', 'value3'], dfd.loc['init_p_wealth_urban', 'value4'],
                     dfd.loc['init_p_wealth_urban', 'value5']]
        p['init_p_wealth_urban'] = \
                    [dfd.loc['init_p_wealth_rural', 'value1'], dfd.loc['init_p_wealth_rural', 'value2'],
                     dfd.loc['init_p_wealth_rural', 'value3'], dfd.loc['init_p_wealth_rural', 'value4'],
                     dfd.loc['init_p_wealth_rural', 'value5']]
        p['init_p_bmi_urban_m_not_high_sugar_age1529_not_tob_wealth1'] = \
                    [dfd.loc['init_p_bmi_urban_m_not_high_sugar_age1529_not_tob_wealth1', 'value1'],
                     dfd.loc['init_p_bmi_urban_m_not_high_sugar_age1529_not_tob_wealth1', 'value2'],
                     dfd.loc['init_p_bmi_urban_m_not_high_sugar_age1529_not_tob_wealth1', 'value3'],
                     dfd.loc['init_p_bmi_urban_m_not_high_sugar_age1529_not_tob_wealth1', 'value4'],
                     dfd.loc['init_p_bmi_urban_m_not_high_sugar_age1529_not_tob_wealth1', 'value5']]
        p['init_or_higher_bmi_f'] = dfd.loc['init_or_higher_bmi_f', 'value1']
        p['init_or_higher_bmi_rural'] = dfd.loc['init_or_higher_bmi_rural', 'value1']
        p['init_or_higher_bmi_high_sugar'] = dfd.loc['init_or_higher_bmi_high_sugar', 'value1']
        p['init_or_higher_bmi_age3049'] = dfd.loc['init_or_higher_bmi_age3049', 'value1']
        p['init_or_higher_bmi_agege50'] = dfd.loc['init_or_higher_bmi_agege50', 'value1']
        p['init_or_higher_bmi_tob'] = dfd.loc['init_or_higher_bmi_tob', 'value1']
        p['init_or_higher_bmi_per_higher_wealth'] = dfd.loc['init_or_higher_bmi_per_higher_wealth', 'value1']
        p['init_p_high_sugar'] = dfd.loc['init_p_high_sugar', 'value1']
        p['init_p_high_salt_urban'] = dfd.loc['init_p_high_salt_urban', 'value1'],
        p['init_or_high_salt_rural'] = dfd.loc['init_or_high_salt_rural', 'value1']
        p['init_p_ex_alc_m'] = dfd.loc['init_p_ex_alc_m', 'value1']
        p['init_p_ex_alc_f'] = dfd.loc['init_p_ex_alc_f', 'value1']
        p['init_dist_mar_stat_age1520'] = [dfd.loc['init_dist_mar_stat_age1520', 'value1'],
                                           dfd.loc['init_dist_mar_stat_age1520', 'value2'],
                                           dfd.loc['init_dist_mar_stat_age1520', 'value3']]
        p['init_dist_mar_stat_age2030'] = [dfd.loc['init_dist_mar_stat_age2030', 'value1'],
                                           dfd.loc['init_dist_mar_stat_age2030', 'value2'],
                                           dfd.loc['init_dist_mar_stat_age2030', 'value3']]
        p['init_dist_mar_stat_age3040'] = [dfd.loc['init_dist_mar_stat_age3040', 'value1'],
                                           dfd.loc['init_dist_mar_stat_age3040', 'value2'],
                                           dfd.loc['init_dist_mar_stat_age3040', 'value3']]
        p['init_dist_mar_stat_age4050'] = [dfd.loc['init_dist_mar_stat_age4050', 'value1'],
                                           dfd.loc['init_dist_mar_stat_age4050', 'value2'],
                                           dfd.loc['init_dist_mar_stat_age4050', 'value3']]
        p['init_dist_mar_stat_age5060'] = [dfd.loc['init_dist_mar_stat_age5060', 'value1'],
                                           dfd.loc['init_dist_mar_stat_age5060', 'value2'],
                                           dfd.loc['init_dist_mar_stat_age5060', 'value3']]
        p['init_dist_mar_stat_agege60'] = [dfd.loc['init_dist_mar_stat_agege60', 'value1'],
                                           dfd.loc['init_dist_mar_stat_agege60', 'value2'],
                                           dfd.loc['init_dist_mar_stat_agege60', 'value3']]
        p['init_age2030_w5_some_ed'] = dfd.loc['init_age2030_w5_some_ed', 'value1'],
        p['init_or_some_ed_age0513'] = dfd.loc['init_or_some_ed_age0513', 'value1'],
        p['init_or_some_ed_age1320'] = dfd.loc['init_or_some_ed_age1320', 'value1'],
        p['init_or_some_ed_age2030'] = dfd.loc['init_or_some_ed_age2030', 'value1'],
        p['init_or_some_ed_age3040'] = dfd.loc['init_or_some_ed_age3040', 'value1'],
        p['init_or_some_ed_age4050'] = dfd.loc['init_or_some_ed_age4050', 'value1'],
        p['init_or_some_ed_age5060'] = dfd.loc['init_or_some_ed_age5060', 'value1'],
        p['init_or_some_ed_per_higher_wealth'] = dfd.loc['init_or_some_ed_per_higher_wealth', 'value1'],
        p['init_prop_age2030_w5_some_ed_sec'] = dfd.loc['init_prop_age2030_w5_some_ed_sec', 'value1'],
        p['init_or_some_ed_sec_age1320'] = dfd.loc['init_or_some_ed_sec_age1320', 'value1'],
        p['init_or_some_ed_sec_age3040'] = dfd.loc['init_or_some_ed_sec_age3040', 'value1'],
        p['init_or_some_ed_sec_age4050'] = dfd.loc['init_or_some_ed_sec_age4050', 'value1'],
        p['init_or_some_ed_sec_age5060'] = dfd.loc['init_or_some_ed_sec_age5060', 'value1'],
        p['init_or_some_ed_sec_agege60'] = dfd.loc['init_or_some_ed_sec_agegr60', 'value1'],
        p['init_or_some_ed_sec_per_higher_wealth'] = dfd.loc['init_or_some_ed_sec_per_higher_wealth', 'value1'],
        p['init_p_unimproved_sanitation'] = dfd.loc['init_p_unimproved_sanitation', 'value1'],
        p['init_or_unimproved_sanitation_rural'] = dfd.loc['init_or_unimproved_sanitation_rural', 'value1'],
        p['init_p_no_clean_drinking_water'] = dfd.loc['init_p_no_clean_drinking_water', 'value1'],
        p['init_or_no_clean_drinking_water_rural'] = dfd.loc['init_or_no_clean_drinking_water_rural', 'value1'],
        p['init_p_wood_burn_stove'] = dfd.loc['init_p_wood_burn_stove', 'value1'],
        p['init_or_wood_burn_stove_rural'] = dfd.loc['init_or_wood_burn_stove_rural', 'value1'],
        p['init_p_no_access_handwashing'] = dfd.loc['init_p_no_access_handwashing', 'value1'],
        p['init_or_no_access_handwashing_per_lower_wealth'] = dfd.loc['init_or_no_access_handwashing_per_lower_wealth', 'value1'],

        p['r_urban'] = dfd.loc['r_urban', 'value1']
        p['r_rural'] = dfd.loc['r_rural', 'value1']
        p['r_higher_bmi'] = dfd.loc['r_higher_bmi', 'value1']
        p['rr_higher_bmi_urban'] = dfd.loc['rr_higher_bmi_urban', 'value1']
        p['rr_higher_bmi_f'] = dfd.loc['rr_higher_bmi_f', 'value1']
        p['rr_higher_bmi_age3049'] = dfd.loc['rr_higher_bmi_age3049', 'value1']
        p['rr_higher_bmi_agege50'] = dfd.loc['rr_higher_bmi_agege50', 'value1']
        p['rr_higher_bmi_tob'] = dfd.loc['rr_higher_bmi_tob', 'value1']
        p['rr_higher_bmi_per_higher_wealth'] = dfd.loc['rr_higher_bmi_per_higher_wealth', 'value1']
        p['rr_higher_bmi_high_sugar'] = dfd.loc['rr_higher_bmi_high_sugar', 'value1']




        'r_lower_bmi': Parameter(Types.REAL, 'probability per 3 months of decrease in bmi category in non tobacco users'),
        'rr_lower_bmi_pop_advice_weight': Parameter(Types.REAL, 'probability per 3 months of decrease in bmi category '
                                                                'given population advice/campaign on weight'),
        'rr_lower_bmi_tob': Parameter(Types.REAL, 'rate ratio for lower bmi category for tobacco users'),
        'r_high_salt_urban': Parameter(Types.REAL, 'probability per 3 months of high salt intake if urban'),
        'rr_high_salt_rural': Parameter(Types.REAL, 'rate ratio for high salt if rural'),
        'r_not_high_salt': Parameter(Types.REAL, 'probability per 3 months of not high salt intake'),
        'rr_not_high_salt_pop_advice_salt': Parameter(Types.REAL, 'probability per 3 months of not high salt given'
                                                                  'population advice/campaign on salt'),
        'r_high_sugar': Parameter(Types.REAL, 'probability per 3 months of high sugar intake'),
        'r_not_high_sugar': Parameter(Types.REAL, 'probability per 3 months of not high sugar intake'),
        'r_low_ex': Parameter(Types.REAL, 'probability per 3 months of change from not low exercise to low exercise'),
        'r_not_low_ex': Parameter(Types.REAL, 'probability per 3 months of change from low exercise to not low exercie'),
        'rr_not_high_sugar_pop_advice_salt': Parameter(Types.REAL, 'probability per 3 months of not high sugar given'
                                                                   'population advice/campaign on sugar'),
        'rr_low_ex_f': Parameter(Types.REAL, 'risk ratio for becoming low exercise if female rather than male'),
        'rr_low_ex_urban': Parameter(Types.REAL, 'risk ratio for becoming low exercise if urban rather than rural'),
        'r_tob': Parameter(Types.REAL, 'probability per 3 months of change from not using tobacco to using '
                                       'tobacco if male age 15-19 wealth level 1'),
        'r_not_tob': Parameter(Types.REAL, 'probability per 3 months of change from tobacco using to '
                                           'not tobacco using'),
        'rr_tob_age2039': Parameter(Types.REAL, 'risk ratio for tobacco using if age 20-39 compared with 15-19'),
        'rr_tob_agege40': Parameter(Types.REAL, 'risk ratio for tobacco using if age >= 40 compared with 15-19'),
        'rr_tob_f': Parameter(Types.REAL, 'risk ratio for tobacco using if female'),
        'rr_tob_wealth': Parameter(Types.REAL, 'risk ratio for tobacco using per 1 higher wealth level '
                                               '(higher wealth level = lower wealth)'),
        'rr_not_tob_pop_advice_tobacco': Parameter(Types.REAL, 'probability per 3 months of quitting tobacco given'
                                                               'population advice/campaign on tobacco'),
        'r_ex_alc': Parameter(Types.REAL, 'probability per 3 months of change from not excess alcohol to '
                                          'excess alcohol'),
        'r_not_ex_alc': Parameter(Types.REAL, 'probability per 3 months of change from excess alcohol to '
                                              'not excess alcohol'),
        'rr_ex_alc_f': Parameter(Types.REAL, 'risk ratio for becoming excess alcohol if female rather than male'),
        'rr_not_tob_pop_advice_alc': Parameter(Types.REAL, 'probability per 3 months of not excess alcohol given'
                                                           'population advice/campaign on alcohol'),
        'r_mar': Parameter(Types.REAL, 'probability per 3 months of marriage when age 15-30'),
        'r_div_wid': Parameter(Types.REAL, 'probability per 3 months of becoming divorced or widowed, '
                                           'amongst those married'),
        'r_stop_ed': Parameter(Types.REAL, 'probabilities per 3 months of stopping education if wealth level 5'),
        'rr_stop_ed_lower_wealth': Parameter(Types.REAL, 'relative rate of stopping education per '
                                                         '1 lower wealth quintile'),
        'p_ed_primary': Parameter(Types.REAL, 'probability at age 5 that start primary education if wealth level 5'),
        'rp_ed_primary_higher_wealth': Parameter(Types.REAL, 'relative probability of starting school per 1 '
                                                             'higher wealth level'),
        'p_ed_secondary': Parameter(Types.REAL, 'probability at age 13 that start secondary education at 13 '
                                                'if in primary education and wealth level 5'),
        'rp_ed_secondary_higher_wealth': Parameter(Types.REAL, 'relative probability of starting secondary '
                                                               'school per 1 higher wealth level'),
        'r_improved_sanitation': Parameter(Types.REAL, 'probability per 3 months of change from '
                                                       'unimproved_sanitation true to false'),
        'r_clean_drinking_water': Parameter(Types.REAL, 'probability per 3 months of change from '
                                                        'drinking_water true to false'),
        'r_non_wood_burn_stove': Parameter(Types.REAL, 'probability per 3 months of change from '
                                                       'wood_burn_stove true to false'),
        'r_access_handwashing': Parameter(Types.REAL, 'probability per 3 months of change from '
                                                      'no_access_handwashing true to false')













    def initialise_population(self, population):
        """Set our property values for the initial population.
        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals
        m = self
        rng = m.rng

        # -------------------- DEFAULTS ------------------------------------------------------------

        df['li_urban'] = False  # default: all individuals rural
        df['li_date_trans_to_urban'] = pd.NaT
        df['li_wealth'].values[:] = 3  # default: all individuals wealth 3
        df['li_overwt'] = False  # default: all not overwt
        df['li_low_ex'] = False  # default all not low ex
        df['li_tob'] = False  # default all not tob
        df['li_ex_alc'] = False  # default all not ex alc
        df['li_mar_stat'].values[:] = 1  # default: all individuals never married
        df['li_on_con'] = False  # default: all not on contraceptives
        # default: call contraceptive type 1, but when li_on_con = False this property becomes most
        # recent contraceptive used
        df['li_con_t'].values[:] = 1
        df['li_in_ed'] = False   # default: not in education
        df['li_ed_lev'].values[:] = 1   # default: education level = 1 - no education
        df['li_unimproved_sanitation'] = True  # default: unimproved_sanitation
        df['li_date_acquire_improved_sanitation'] = pd.NaT
        df['li_no_access_handwashing'] = True  # default: no_access_handwashing
        df['li_date_acquire_access_handwashing'] = pd.NaT
        df['li_no_clean_drinking_water'] = True  # default: unimproved_sanitation
        df['li_date_acquire_clean_drinking_water'] = pd.NaT
        df['li_wood_burn_stove'] = True  # default: li_wood_burn_stove
        df['li_date_acquire_non_wood_burn_stove'] = pd.NaT


        # todo: odds bmi 1 x (1.1**2), odds bmi 2 x (1.1**1), odds bmi 3 x 1.1**0, odds bmi 4 x 1.1**(-1),
        # todo: odds bmi 5 x 1.1**(-2) - then re-normalise to sum proportions to 1


        # todo: express all rates per year and divide by 4 inside program


        # -------------------- URBAN-RURAL STATUS --------------------------------------------------

        # randomly selected some individuals as urban
        df['li_urban'] = (rng.random_sample(size=len(df)) < m.init_p_urban)

        # get the indices of all individuals who are urban or rural
        urban_index = df.index[df.is_alive & df.li_urban]
        rural_index = df.index[df.is_alive & ~df.li_urban]

        # randomly sample wealth category according to urban/rural wealth probs
        df.loc[urban_index, 'li_wealth'] = rng.choice([1, 2, 3, 4, 5], size=len(urban_index), p=m.init_p_wealth_urban)
        df.loc[rural_index, 'li_wealth'] = rng.choice([1, 2, 3, 4, 5], size=len(rural_index), p=m.init_p_wealth_rural)

        # -------------------- OVERWEIGHT ----------------------------------------------------------

        # get indices of all individuals over 15 years
        age_gte15 = df.index[df.is_alive & (df.age_years >= 15)]

        overweight_lookup = pd.DataFrame(data=[('M', True, 0.46),
                                               ('M', False, 0.27),
                                               ('F', True, 0.32),
                                               ('F', False, 0.17)],
                                         columns=['sex', 'is_urban', 'p_ow'])

        overweight_probs = df.loc[age_gte15, ['sex', 'li_urban']].merge(overweight_lookup,
                                                                        left_on=['sex', 'li_urban'],
                                                                        right_on=['sex', 'is_urban'],
                                                                        how='inner')['p_ow']
        assert len(overweight_probs) == len(age_gte15)

        random_draw = rng.random_sample(size=len(age_gte15))
        df.loc[age_gte15, 'li_overwt'] = (random_draw < overweight_probs.values)

        # -------------------- LOW EXERCISE --------------------------------------------------------

        low_ex_lookup = pd.DataFrame(data=[('M', True, 0.32),
                                           ('M', False, 0.11),
                                           ('F', True, 0.18),
                                           ('F', False, 0.07)],
                                     columns=['sex', 'is_urban', 'p_low_ex'])

        low_ex_probs = df.loc[age_gte15, ['sex', 'li_urban']].merge(low_ex_lookup,
                                                                    left_on=['sex', 'li_urban'],
                                                                    right_on=['sex', 'is_urban'],
                                                                    how='inner')['p_low_ex']
        assert len(low_ex_probs) == len(age_gte15)

        random_draw = rng.random_sample(size=len(age_gte15))
        df.loc[age_gte15, 'li_low_ex'] = (random_draw < low_ex_probs.values)

        # -------------------- TOBACCO USE ---------------------------------------------------------

        tob_lookup = pd.DataFrame([('M', '15-19', 0.01),
                                   ('M', '20-24', 0.04),
                                   ('M', '25-29', 0.04),
                                   ('M', '30-34', 0.04),
                                   ('M', '35-39', 0.04),
                                   ('M', '40-44', 0.06),
                                   ('M', '45-49', 0.06),
                                   ('M', '50-54', 0.06),
                                   ('M', '55-59', 0.06),
                                   ('M', '60-64', 0.06),
                                   ('M', '65-69', 0.06),
                                   ('M', '70-74', 0.06),
                                   ('M', '75-79', 0.06),
                                   ('M', '80-84', 0.06),
                                   ('M', '85-89', 0.06),
                                   ('M', '90-94', 0.06),
                                   ('M', '95-99', 0.06),
                                   ('M', '100+',  0.06),

                                   ('F', '15-19', 0.002),
                                   ('F', '20-24', 0.002),
                                   ('F', '25-29', 0.002),
                                   ('F', '30-34', 0.002),
                                   ('F', '35-39', 0.002),
                                   ('F', '40-44', 0.002),
                                   ('F', '45-49', 0.002),
                                   ('F', '50-54', 0.002),
                                   ('F', '55-59', 0.002),
                                   ('F', '60-64', 0.002),
                                   ('F', '65-69', 0.002),
                                   ('F', '70-74', 0.002),
                                   ('F', '75-79', 0.002),
                                   ('F', '80-84', 0.002),
                                   ('F', '85-89', 0.002),
                                   ('F', '90-94', 0.002),
                                   ('F', '95-99', 0.002),
                                   ('F', '100+',  0.002)],
                                  columns=['sex', 'age_range', 'p_tob'])

        # join the population-with-age dataframe with the tobacco use lookup table (join on sex and age_range)
        tob_probs = df.loc[age_gte15].merge(tob_lookup,
                                            left_on=['sex', 'age_range'],
                                            right_on=['sex', 'age_range'],
                                            how='inner')
        assert len(age_gte15) == len(tob_probs)

        # each individual has a baseline probability
        # multiply this probability by the wealth level. wealth is a category, so convert to integer
        tob_probs = pd.to_numeric(tob_probs['li_wealth']) * tob_probs['p_tob']

        # we now have the probability of tobacco use for each individual where age >= 15
        # draw a random number between 0 and 1 for all of them
        random_draw = rng.random_sample(size=len(age_gte15))

        # decide on tobacco use based on the individual probability is greater than random draw
        # this is a list of True/False. assign to li_tob
        df.loc[age_gte15, 'li_tob'] = (random_draw < tob_probs.values)

        # -------------------- EXCESSIVE ALCOHOL ---------------------------------------------------

        m_gte15 = df.index[df.is_alive & (df.age_years >= 15) & (df.sex == 'M')]
        f_gte15 = df.index[df.is_alive & (df.age_years >= 15) & (df.sex == 'F')]

        df.loc[m_gte15, 'li_ex_alc'] = rng.random_sample(size=len(m_gte15)) < m.init_p_ex_alc_m
        df.loc[f_gte15, 'li_ex_alc'] = rng.random_sample(size=len(f_gte15)) < m.init_p_ex_alc_f

        # -------------------- MARITAL STATUS ------------------------------------------------------

        age_15_19 = df.index[df.age_years.between(15, 19) & df.is_alive]
        age_20_29 = df.index[df.age_years.between(20, 29) & df.is_alive]
        age_30_39 = df.index[df.age_years.between(30, 39) & df.is_alive]
        age_40_49 = df.index[df.age_years.between(40, 49) & df.is_alive]
        age_50_59 = df.index[df.age_years.between(50, 59) & df.is_alive]
        age_gte60 = df.index[(df.age_years >= 60) & df.is_alive]

        df.loc[age_15_19, 'li_mar_stat'] = rng.choice([1, 2, 3], size=len(age_15_19), p=m.init_dist_mar_stat_age1520)
        df.loc[age_20_29, 'li_mar_stat'] = rng.choice([1, 2, 3], size=len(age_20_29), p=m.init_dist_mar_stat_age2030)
        df.loc[age_30_39, 'li_mar_stat'] = rng.choice([1, 2, 3], size=len(age_30_39), p=m.init_dist_mar_stat_age3040)
        df.loc[age_40_49, 'li_mar_stat'] = rng.choice([1, 2, 3], size=len(age_40_49), p=m.init_dist_mar_stat_age4050)
        df.loc[age_50_59, 'li_mar_stat'] = rng.choice([1, 2, 3], size=len(age_50_59), p=m.init_dist_mar_stat_age5060)
        df.loc[age_gte60, 'li_mar_stat'] = rng.choice([1, 2, 3], size=len(age_gte60), p=m.init_dist_mar_stat_agege60)

        # -------------------- CONTRACEPTION STATUS ------------------------------------------------

        f_age_1550 = df.index[df.age_years.between(15, 49) & df.is_alive & (df.sex == 'F')]
        df.loc[f_age_1550, 'li_on_con'] = (rng.random_sample(size=len(f_age_1550)) < m.init_p_on_contrac)

        f_age_1550_on_con = df.index[df.age_years.between(14, 49) & df.is_alive & (df.sex == 'F') & df.li_on_con]
        df.loc[f_age_1550_on_con, 'li_con_t'] = rng.choice([1, 2, 3, 4, 5, 6],
                                                           size=len(f_age_1550_on_con),
                                                           p=m.init_dist_con_t)

        # -------------------- EDUCATION -----------------------------------------------------------

        age_gte5 = df.index[(df.age_years >= 5) & df.is_alive]

        # calculate the probability of education for all individuals over 5 years old
        p_some_ed = pd.Series(m.init_age2030_w5_some_ed, index=age_gte5)


        #todo  note now using odds ratio rather than relative prevalence so small amendments needed


        # adjust probability of some education based on age
        p_some_ed.loc[df.age_years < 13] *= m.init_rp_some_ed_age0513
        p_some_ed.loc[df.age_years.between(13, 19)] *= m.init_rp_some_ed_age1320
        p_some_ed.loc[df.age_years.between(30, 39)] *= m.init_rp_some_ed_age3040
        p_some_ed.loc[df.age_years.between(40, 49)] *= m.init_rp_some_ed_age4050
        p_some_ed.loc[df.age_years.between(50, 59)] *= m.init_rp_some_ed_age5060
        p_some_ed.loc[(df.age_years >= 60)] *= m.init_rp_some_ed_agege60

        # adjust probability of some education based on wealth
        p_some_ed *= m.init_rp_some_ed_per_higher_wealth**(5 - pd.to_numeric(df.loc[age_gte5, 'li_wealth']))

        # calculate baseline of education level 3, and adjust for age and wealth
        p_ed_lev_3 = pd.Series(m.init_prop_age2030_w5_some_ed_sec, index=age_gte5)

        p_ed_lev_3.loc[(df.age_years < 13)] *= 0
        p_ed_lev_3.loc[df.age_years.between(13, 19)] *= m.init_rp_some_ed_sec_age1320
        p_ed_lev_3.loc[df.age_years.between(30, 39)] *= m.init_rp_some_ed_sec_age3040
        p_ed_lev_3.loc[df.age_years.between(40, 49)] *= m.init_rp_some_ed_sec_age4050
        p_ed_lev_3.loc[df.age_years.between(50, 59)] *= m.init_rp_some_ed_sec_age5060
        p_ed_lev_3.loc[(df.age_years >= 60)] *= m.init_rp_some_ed_sec_agege60
        p_ed_lev_3 *= m.init_rp_some_ed_sec_per_higher_wealth**(5 - pd.to_numeric(df.loc[age_gte5, 'li_wealth']))

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

        rural_idx = df.index[df.is_alive & ~df.li_urban]
        all_idx = df.index[df.is_alive]

        eff_prev_unimproved_sanitation = pd.Series(m.init_p_unimproved_sanitation,
                                                   index=df.index[df.is_alive])

        eff_prev_unimproved_sanitation.loc[rural_idx] *= m.init_rp_unimproved_sanitation_rural

        random_draw = pd.Series(rng.random_sample(size=len(all_idx)), index=df.index[df.is_alive])

        df.loc[all_idx, 'li_unimproved_sanitation'] = random_draw < eff_prev_unimproved_sanitation

        # -------------------- NO CLEAN DRINKING WATER ---------------------------------------------------

        rural_idx = df.index[df.is_alive & ~df.li_urban]
        all_idx = df.index[df.is_alive]

        eff_prev_no_clean_drinking_water = pd.Series(m.init_p_no_clean_drinking_water, index=df.index[df.is_alive])

        eff_prev_no_clean_drinking_water.loc[rural_idx] *= m.init_rp_no_clean_drinking_water

        random_draw = pd.Series(rng.random_sample(size=len(all_idx)), index=df.index[df.is_alive])

        df.loc[all_idx, 'li_no_clean_drinking_water'] = random_draw < eff_prev_no_clean_drinking_water

        # -------------------- WOOD BURN STOVE ---------------------------------------------------

        rural_idx = df.index[df.is_alive & ~df.li_urban]
        all_idx = df.index[df.is_alive]

        eff_prev_wood_burn_stove = pd.Series(m.init_p_wood_burn_stove, index=df.index[df.is_alive])

        eff_prev_wood_burn_stove.loc[rural_idx] *= m.init_rp_wood_burn_stove

        random_draw = pd.Series(rng.random_sample(size=len(all_idx)), index=df.index[df.is_alive])

        df.loc[all_idx, 'li_wood_burn_stove'] = random_draw < eff_prev_wood_burn_stove

        # -------------------- NO ACCESS HANDWASHING ---------------------------------------------------

        all_idx = df.index[df.is_alive]
        wealth2_idx = df.index[df.is_alive & (df.li_wealth == 2)]
        wealth3_idx = df.index[df.is_alive & (df.li_wealth == 3)]
        wealth4_idx = df.index[df.is_alive & (df.li_wealth == 4)]
        wealth5_idx = df.index[df.is_alive & (df.li_wealth == 5)]

        eff_prev_no_access_handwashing = pd.Series(m.init_p_no_access_handwashing, index=df.index[df.is_alive])

        eff_prev_no_access_handwashing.loc[wealth2_idx] *= m.init_rp_no_access_handwashing_per_lower_wealth
        eff_prev_no_access_handwashing.loc[wealth3_idx] *= (m.init_rp_no_access_handwashing_per_lower_wealth ** 2)
        eff_prev_no_access_handwashing.loc[wealth4_idx] *= (m.init_rp_no_access_handwashing_per_lower_wealth ** 3)
        eff_prev_no_access_handwashing.loc[wealth5_idx] *= (m.init_rp_no_access_handwashing_per_lower_wealth ** 4)

        random_draw = pd.Series(rng.random_sample(size=len(all_idx)), index=df.index[df.is_alive])

        df.loc[all_idx, 'li_no_access_handwashing'] = random_draw < eff_prev_no_access_handwashing

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

        df.at[child_id, 'li_urban'] = df.at[mother_id, 'li_urban']
        df.at[child_id, 'li_date_trans_to_urban'] = pd.NaT
        df.at[child_id, 'li_wealth'] = df.at[mother_id, 'li_wealth']
        df.at[child_id, 'li_overwt'] = False
        df.at[child_id, 'li_low_ex'] = False
        df.at[child_id, 'li_tob'] = False
        df.at[child_id, 'li_ex_alc'] = False
        df.at[child_id, 'li_mar_stat'] = 1
        df.at[child_id, 'li_on_con'] = False
        df.at[child_id, 'li_con_t'] = 1
        df.at[child_id, 'li_in_ed'] = False
        df.at[child_id, 'li_ed_lev'] = 1
        df.at[child_id, 'li_unimproved_sanitation'] = df.at[mother_id, 'li_unimproved_sanitation']
        df.at[child_id, 'li_no_access_handwashing'] = df.at[mother_id, 'li_no_access_handwashing']
        df.at[child_id, 'li_no_clean_drinking_water'] = df.at[mother_id, 'li_no_clean_drinking_water']
        df.at[child_id, 'li_wood_burn_stove'] = df.at[mother_id, 'li_wood_burn_stove']


class LifestyleEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all lifestyle properties for population
    """
    def __init__(self, module):
        """schedule to run every 3 months
        note: if change this offset from 3 months need to consider code conditioning on age.years_exact
        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months=3))

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
        now_urban: pd.Series = m.rng.random_sample(size=len(currently_rural)) < m.r_urban
        urban_idx = currently_rural[now_urban]
        df.loc[urban_idx, 'li_urban'] = True
        df.loc[urban_idx, 'li_date_trans_to_urban'] = self.sim.date

        # handle new transitions to rural
        now_rural: pd.Series = rng.random_sample(size=len(currently_urban)) < m.r_rural
        df.loc[currently_urban[now_rural], 'li_urban'] = False

        # -------------------- OVERWEIGHT ----------------------------------------------------------

        # get all adult who are not overweight
        adults_not_ow = df.index[~df.li_overwt & df.is_alive & (df.age_years >= 15)]

        # calculate the effective prob of becoming overweight; use the index of adults not ow
        eff_p_ow = pd.Series(m.r_overwt, index=adults_not_ow)
        eff_p_ow.loc[df.sex == 'F'] *= m.rr_overwt_f
        eff_p_ow.loc[df.li_urban] *= m.rr_overwt_urban

        # random draw and start of overweight status
        df.loc[adults_not_ow, 'li_overwt'] = (rng.random_sample(len(adults_not_ow)) < eff_p_ow)

        # transition from over weight to not over weight
        overwt_idx = df.index[df.li_overwt & df.is_alive]
        eff_rate_not_overwt = pd.Series(m.r_not_overwt, index=overwt_idx)
        random_draw = rng.random_sample(len(overwt_idx))
        newly_not_overwt: pd.Series = random_draw < eff_rate_not_overwt
        newly_not_overwt_idx = overwt_idx[newly_not_overwt]
        df.loc[newly_not_overwt_idx, 'li_overwt'] = False
        df.loc[newly_not_overwt_idx, 'li_date_no_longer_overwt'] = self.sim.date

        # -------------------- LOW EXERCISE --------------------------------------------------------

        adults_not_low_ex = df.index[~df.li_low_ex & df.is_alive & (df.age_years >= 15)]

        eff_p_low_ex = pd.Series(m.r_low_ex, index=adults_not_low_ex)
        eff_p_low_ex.loc[df.sex == 'F'] *= m.rr_low_ex_f
        eff_p_low_ex.loc[df.li_urban] *= m.rr_low_ex_urban

        df.loc[adults_not_low_ex, 'li_low_ex'] = (rng.random_sample(len(adults_not_low_ex)) < eff_p_low_ex)

        # transition from low exercise to not low exercise
        low_ex_idx = df.index[df.li_low_ex & df.is_alive]
        eff_rate_not_low_ex = pd.Series(m.r_not_low_ex, index=low_ex_idx)
        random_draw = rng.random_sample(len(low_ex_idx))
        newly_not_low_ex: pd.Series = random_draw < eff_rate_not_low_ex
        newly_not_low_ex_idx = low_ex_idx[newly_not_low_ex]
        df.loc[newly_not_low_ex_idx, 'li_low_ex'] = False
        df.loc[newly_not_low_ex_idx, 'li_date_no_longer_low_ex'] = self.sim.date

        # -------------------- TOBACCO USE ---------------------------------------------------------

        adults_not_tob = df.index[(df.age_years >= 15) & df.is_alive & ~df.li_tob]
        currently_tob = df.index[df.li_tob & df.is_alive]

        # start tobacco use
        eff_p_tob = pd.Series(m.r_tob, index=adults_not_tob)
        eff_p_tob.loc[(df.age_years >= 20) & (df.age_years < 40)] *= m.rr_tob_age2039
        eff_p_tob.loc[df.age_years >= 40] *= m.rr_tob_agege40
        eff_p_tob.loc[df.sex == 'F'] *= m.rr_tob_f
        eff_p_tob *= m.rr_tob_wealth ** (pd.to_numeric(df.loc[adults_not_tob, 'li_wealth']) - 1)

        df.loc[adults_not_tob, 'li_tob'] = (rng.random_sample(len(adults_not_tob)) < eff_p_tob)

        # transition from tobacco to no tobacco
        tob_idx = df.index[df.li_tob & df.is_alive]
        eff_rate_not_tob = pd.Series(m.r_not_tob, index=tob_idx)
        random_draw = rng.random_sample(len(tob_idx))
        newly_not_tob: pd.Series = random_draw < eff_rate_not_tob
        newly_not_tob_idx = tob_idx[newly_not_tob]
        df.loc[newly_not_tob_idx, 'li_tob'] = False
        df.loc[newly_not_tob_idx, 'li_date_quit_tob'] = self.sim.date

        # -------------------- EXCESSIVE ALCOHOL ---------------------------------------------------

        not_ex_alc_f = df.index[~df.li_ex_alc & df.is_alive & (df.sex == 'F') & (df.age_years >= 15)]
        not_ex_alc_m = df.index[~df.li_ex_alc & df.is_alive & (df.sex == 'M') & (df.age_years >= 15)]
        now_ex_alc = df.index[df.li_ex_alc & df.is_alive]

        df.loc[not_ex_alc_f, 'li_ex_alc'] = rng.random_sample(len(not_ex_alc_f)) < m.r_ex_alc * m.rr_ex_alc_f
        df.loc[not_ex_alc_m, 'li_ex_alc'] = rng.random_sample(len(not_ex_alc_m)) < m.r_ex_alc
        df.loc[now_ex_alc, 'li_ex_alc'] = ~(rng.random_sample(len(now_ex_alc)) < m.r_not_ex_alc)

        # transition from excess alcohol to not excess alcohol
        ex_alc_idx = df.index[df.li_ex_alc & df.is_alive]
        eff_rate_not_ex_alc = pd.Series(m.r_not_ex_alc, index=ex_alc_idx)
        random_draw = rng.random_sample(len(ex_alc_idx))
        newly_not_ex_alc: pd.Series = random_draw < eff_rate_not_ex_alc
        newly_not_ex_alc_idx = ex_alc_idx[newly_not_ex_alc]
        df.loc[newly_not_ex_alc_idx, 'li_ex_alc'] = False
        df.loc[newly_not_ex_alc_idx, 'li_date_no_longer_ex_alc'] = self.sim.date

        # -------------------- MARITAL STATUS ------------------------------------------------------

        curr_never_mar = df.index[df.is_alive & df.age_years.between(15, 29) & (df.li_mar_stat == 1)]
        curr_mar = df.index[df.is_alive & (df.li_mar_stat == 2)]

        # update if now married
        now_mar = rng.random_sample(len(curr_never_mar)) < m.r_mar
        df.loc[curr_never_mar[now_mar], 'li_mar_stat'] = 2

        # update if now divorced/widowed
        now_div_wid = rng.random_sample(len(curr_mar)) < m.r_div_wid
        df.loc[curr_mar[now_div_wid], 'li_mar_stat'] = 3

        # -------------------- CONTRACEPTION USE ---------------------------------------------------

        possibly_using = df.is_alive & (df.sex == 'F') & df.age_years.between(15, 49)
        curr_not_on_con = df.index[possibly_using & ~df.li_on_con]
        curr_on_con = df.index[possibly_using & df.li_on_con]

        # currently not on contraceptives -> start using contraceptives
        now_on_con = rng.random_sample(size=len(curr_not_on_con)) < m.r_contrac
        df.loc[curr_not_on_con[now_on_con], 'li_on_con'] = True

        # currently using contraceptives -> interrupted
        now_not_on_con = rng.random_sample(size=len(curr_on_con)) < m.r_contrac_int
        df.loc[curr_on_con[now_not_on_con], 'li_on_con'] = False

        # everyone stops using contraceptives at age 50
        f_age_50 = df.index[(df.age_years == 50) & df.li_on_con]
        df.loc[f_age_50, 'li_on_con'] = False

        # contraceptive method transitions
        # note: transitions contr. type for those already using, not those who just started in this event
        def con_method_transition(con_type, rates):
            curr_on_con_type = df.index[curr_on_con & (df.li_con_t == con_type)]
            df.loc[curr_on_con_type, 'li_con_t'] = rng.choice([1, 2, 3, 4, 5, 6], size=len(curr_on_con_type), p=rates)

#       con_method_transition(1, m.r_con_from_1)
#       con_method_transition(2, m.r_con_from_2)
#       con_method_transition(3, m.r_con_from_3)
#       con_method_transition(4, m.r_con_from_4)
#       con_method_transition(5, m.r_con_from_5)
#       con_method_transition(6, m.r_con_from_6)

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
        prob_primary = pd.Series(m.p_ed_primary, index=age5)
        prob_primary *= m.rp_ed_primary_higher_wealth**(5 - pd.to_numeric(df.loc[age5, 'li_wealth']))

        # randomly select some to have primary education
        age5_in_primary = rng.random_sample(len(age5)) < prob_primary
        df.loc[age5[age5_in_primary], 'li_ed_lev'] = 2
        df.loc[age5[age5_in_primary], 'li_in_ed'] = True

        # ---- SECONDARY EDUCATION

        # get thirteen year olds that are in primary education, any wealth level
        age13_in_primary = df.index[(df.age_years == 13) & df.is_alive & df.li_in_ed & (df.li_ed_lev == 2)]

        # they have a probability of gaining secondary education (level 3), based on wealth
        prob_secondary = pd.Series(m.p_ed_secondary, index=age13_in_primary)
        prob_secondary *= m.rp_ed_secondary_higher_wealth**(5 - pd.to_numeric(df.loc[age13_in_primary, 'li_wealth']))

        # randomly select some to get secondary education
        age13_to_secondary = rng.random_sample(len(age13_in_primary)) < prob_secondary
        df.loc[age13_in_primary[age13_to_secondary], 'li_ed_lev'] = 3

        # those who did not go on to secondary education are no longer in education
        df.loc[age13_in_primary[~age13_to_secondary], 'li_in_ed'] = False

        # ---- DROP OUT OF EDUCATION

        # baseline rate of leaving education then adjust for wealth level
        p_leave_ed = pd.Series(m.r_stop_ed, index=in_ed)
        p_leave_ed *= m.rr_stop_ed_lower_wealth**(pd.to_numeric(df.loc[in_ed, 'li_wealth']) - 1)

        # randomly select some individuals to leave education
        now_not_in_ed = rng.random_sample(len(in_ed)) < p_leave_ed

        df.loc[in_ed[now_not_in_ed], 'li_in_ed'] = False

        # everyone leaves education at age 20
        df.loc[df.is_alive & df.li_in_ed & (df.age_years == 20), 'li_in_ed'] = False

        # -------------------- UNIMPROVED SANITATION --------------------------------------------------------

        # probability of improved sanitation at all follow-up times
        unimproved_sanitaton_idx = df.index[df.li_unimproved_sanitation & df.is_alive]

        eff_rate_improved_sanitation = pd.Series(m.r_improved_sanitation, index=unimproved_sanitaton_idx)

        random_draw = rng.random_sample(len(unimproved_sanitaton_idx))

        newly_improved_sanitation: pd.Series = random_draw < eff_rate_improved_sanitation
        newly_improved_sanitation_idx = unimproved_sanitaton_idx[newly_improved_sanitation]
        df.loc[newly_improved_sanitation_idx, 'li_unimproved_sanitation'] = False
        df.loc[newly_improved_sanitation_idx, 'li_date_acquire_improved_sanitation'] = self.sim.date

        # probability of improved sanitation upon moving to urban from rural
        unimproved_sanitation_newly_urban_idx = df.index[df.li_unimproved_sanitation & df.is_alive &
                                                         df.li_date_trans_to_urban == self.sim.date]

        random_draw = rng.random_sample(len(unimproved_sanitation_newly_urban_idx))

        eff_prev_unimproved_sanitation_urban = pd.Series(m.init_p_unimproved_sanitation,
                                                         index=unimproved_sanitation_newly_urban_idx)

        df.loc[unimproved_sanitation_newly_urban_idx, 'li_unimproved_sanitation'] \
            = random_draw < eff_prev_unimproved_sanitation_urban

        # -------------------- NO ACCESS HANDWASHING --------------------------------------------------------

        # probability of moving to access to handwashing at all follow-up times
        no_access_handwashing_idx = df.index[df.li_no_access_handwashing & df.is_alive]

        eff_rate_access_handwashing = pd.Series(m.r_access_handwashing, index=no_access_handwashing_idx)

        random_draw = rng.random_sample(len(no_access_handwashing_idx))

        newly_access_handwashing: pd.Series = random_draw < eff_rate_access_handwashing
        newly_access_handwashing_idx = no_access_handwashing_idx[newly_access_handwashing]
        df.loc[newly_access_handwashing_idx, 'li_no_access_handwashing'] = False
        df.loc[newly_access_handwashing_idx, 'li_date_acquire_access_handwashing'] = self.sim.date

        # -------------------- NO CLEAN DRINKING WATER  --------------------------------------------------------

        # probability of moving to clean drinking water at all follow-up times
        no_clean_drinking_water_idx = df.index[df.li_no_clean_drinking_water & df.is_alive]

        eff_rate_clean_drinking_water = pd.Series(m.r_clean_drinking_water, index=no_clean_drinking_water_idx)

        random_draw = rng.random_sample(len(no_clean_drinking_water_idx))

        newly_clean_drinking_water: pd.Series = random_draw < eff_rate_clean_drinking_water
        newly_clean_drinking_water_idx = no_clean_drinking_water_idx[newly_clean_drinking_water]
        df.loc[newly_clean_drinking_water_idx, 'li_no_clean_drinking_water'] = False
        df.loc[newly_clean_drinking_water_idx, 'li_date_acquire_clean_drinking_water'] = self.sim.date

        # probability of no clean drinking water upon moving to urban from rural
        no_clean_drinking_water_newly_urban_idx = df.index[df.li_no_clean_drinking_water & df.is_alive &
                                                           df.li_date_trans_to_urban == self.sim.date]

        random_draw = rng.random_sample(len(no_clean_drinking_water_newly_urban_idx))

        eff_prev_no_clean_drinking_water_urban = pd.Series(m.init_p_no_clean_drinking_water,
                                                           index=no_clean_drinking_water_newly_urban_idx)

        df.loc[no_clean_drinking_water_newly_urban_idx, 'li_no_clean_drinking_water'] \
            = random_draw < eff_prev_no_clean_drinking_water_urban

        # -------------------- WOOD BURN STOVE -------------------------------------------------------------

        # probability of moving to non wood burn stove at all follow-up times
        wood_burn_stove_idx = df.index[df.li_wood_burn_stove & df.is_alive]

        eff_rate_non_wood_burn_stove = pd.Series(m.r_non_wood_burn_stove, index=wood_burn_stove_idx)

        random_draw = rng.random_sample(len(wood_burn_stove_idx))

        newly_non_wood_burn_stove: pd.Series = random_draw < eff_rate_non_wood_burn_stove
        newly_non_wood_burn_stove_idx = wood_burn_stove_idx[newly_non_wood_burn_stove]
        df.loc[newly_non_wood_burn_stove_idx, 'li_wood_burn_stove'] = False
        df.loc[newly_non_wood_burn_stove_idx, 'li_date_acquire_non_wood_burn_stove'] = self.sim.date

        # probability of moving to wood burn stove upon moving to urban from rural
        wood_burn_stove_newly_urban_idx = df.index[df.li_wood_burn_stove & df.is_alive &
                                                   df.li_date_trans_to_urban == self.sim.date]

        random_draw = rng.random_sample(len(wood_burn_stove_newly_urban_idx))

        eff_prev_wood_burn_stove_urban = pd.Series(m.init_p_wood_burn_stove,
                                                   index=wood_burn_stove_newly_urban_idx)

        df.loc[wood_burn_stove_newly_urban_idx, 'li_wood_burn_stove'] \
            = random_draw < eff_prev_wood_burn_stove_urban


class LifestylesLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """Handles lifestyle logging"""
    def __init__(self, module):
        """schedule logging to repeat every 3 months
        """
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        # get some summary statistics
        df = population.props

        #   logger.info('%s|li_wealth li_no_access_handwashing|%s',
        #               self.sim.date,
        #               df[df.is_alive].groupby(['li_wealth', 'li_no_access_handwashing']).size().to_dict())

        logger.debug('%s|person_one|%s',
                     self.sim.date,
                     df.loc[0].to_dict())

        """
        logger.info('%s|li_urban|%s',
                    self.sim.date,
                    df[df.is_alive].groupby('li_urban').size().to_dict())

        logger.info('%s|li_wealth|%s',
                    self.sim.date,
                    df[df.is_alive].groupby('li_wealth').size().to_dict())
        logger.info('%s|li_overwt|%s',
                    self.sim.date,
                    df[df.is_alive].groupby(['sex', 'li_overwt']).size().to_dict())
        logger.info('%s|li_low_ex|%s',
                    self.sim.date,
                    df[df.is_alive].groupby(['sex', 'li_low_ex']).size().to_dict())
        logger.info('%s|li_tob|%s',
                    self.sim.date,
                    df[df.is_alive].groupby(['sex', 'li_tob']).size().to_dict())
        logger.info('%s|li_ed_lev_by_age|%s',
                    self.sim.date,
                    df[df.is_alive].groupby(['age_range', 'li_in_ed', 'li_ed_lev']).size().to_dict())
        """

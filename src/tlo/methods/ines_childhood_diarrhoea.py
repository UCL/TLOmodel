"""
Childhood diarrhoea module
Documentation: 04 - Methods Repository/Method_Child_RespiratoryInfection.xlsx
"""
import logging

import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ChildhoodDiarrhoea(Module):
    PARAMETERS = {
        'base_prev_dysentery': Parameter
        (Types.REAL,
         'initial prevalence of dysentery, among children aged 0-11 months,'
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no access to clean water, no improved sanitation, wealth level 3'
         ),
        'rp_dysentery_agelt11mo': Parameter
        (Types.REAL,
         'relative prevalence of dysentery for age < 11 months'
         ),
        'rp_dysentery_age12to23mo': Parameter
        (Types.REAL,
         'relative prevalence of dysentery for age 12 to 23 months'
         ),
        'rp_dysentery_age24to59mo': Parameter
        (Types.REAL,
         'relative prevalence of dysentery for age 24 to 59 months'
         ),
        'rp_dysentery_HIV': Parameter
        (Types.REAL,
         'relative prevalence of dysentery for HIV positive'
         ),
        'rp_dysentery_SAM': Parameter
        (Types.REAL,
         'relative prevalence of dysentery for severe acute malnutrition'
         ),
        'rp_dysentery_excl_breast': Parameter
        (Types.REAL,
         'relative prevalence of dysentery for exclusive breastfeeding upto 6 months'
         ),
        'rp_dysentery_cont_breast': Parameter
        (Types.REAL,
         'relative prevalence of dysentery for continued breastfeeding upto 23 months'
         ),
        'rp_dysentery_HHhandwashing': Parameter
        (Types.REAL,
         'relative prevalence of dysentery for household handwashing'
         ),
        'rp_pneumonia_clean_water': Parameter
        (Types.REAL,
         'relative prevalence of dysentery for access to clean water'
         ),
        'rp_pneumonia_improved_sanitation': Parameter
        (Types.REAL,
         'relative prevalence of dysentery for improved sanitation'
         ),
        'rp_dysentery_wealth1': Parameter
        (Types.REAL,
         'relative prevalence of dysentery for wealth level 1'
         ),
        'rp_dysentery_wealth2': Parameter
        (Types.REAL,
         'relative prevalence of dysentery for wealth level 2'
         ),
        'rp_dysentery_wealth4': Parameter
        (Types.REAL,
         'relative prevalence of dysentery for wealth level 4'
         ),
        'rp_dysentery_wealth5': Parameter
        (Types.REAL,
         'relative prevalence of dysentery for wealth level 5'
         ),
        'base_incidence_dysentery': Parameter
        (Types.REAL,
         'baseline incidence of dysentery, among children aged < 11 months, '
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no clean water source, no improved sanitation, wealth level 3'
         ),
        'rr_dysentery_agelt11mo': Parameter
        (Types.REAL,
         'relative rate of dysentery for age < 11 months'
         ),
        'rr_dysentery_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of dysentery for age 12 to 23 months'
         ),
        'rr_dysentery_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of dysentery for age 24 to 59 months'
         ),
        'rr_dysentery_HIV': Parameter
        (Types.REAL,
         'relative rate of dysentery for HIV positive'
         ),
        'rr_dysentery_SAM': Parameter
        (Types.REAL,
         'relative rate of dysentery for severe acute malnutrition'
         ),
        'rr_dysentery_excl_breast': Parameter
        (Types.REAL,
         'relative rate of dysentery for exclusive breastfeeding upto 6 months'
         ),
        'rr_dysentery_cont_breast': Parameter
        (Types.REAL,
         'relative rate of dysentery for continued breastfeeding 6 months to 2 years'
         ),
        'rr_dysentery_HHhandwashing': Parameter
        (Types.REAL,
         'relative rate of dysentery for household handwashing'
         ),
        'rr_dysentery_clean_water': Parameter
        (Types.REAL,
         'relative rate of dysentery for access to clean water'
         ),
        'rr_dysentery_improved_sanitation': Parameter
        (Types.REAL,
         'relative rate of dysentery for improved sanitation'
         ),
        'rr_dysentery_wealth1': Parameter
        (Types.REAL,
         'relative rate of dysentery for wealth level 1'
         ),
        'rr_dysentery_wealth2': Parameter
        (Types.REAL,
         'relative rate of dysentery for wealth level 2'
         ),
        'rr_dysentery_wealth4': Parameter
        (Types.REAL,
         'relative rate of dysentery for wealth level 4'
         ),
        'rr_dysentery_wealth5': Parameter
        (Types.REAL,
         'relative rate of dysentery for wealth level 5'
         ),
        'base_prev_acute_diarrhoea': Parameter
        (Types.REAL,
         'initial prevalence of acute watery diarrhoea, among children aged < 11 months, '
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no clean water source, no improved sanitation, wealth level 3'
         ),
        'rp_acute_diarrhoea_agelt11mo': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for age < 11 months'
         ),
        'rp_acute_diarrhoea_age12to23mo': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for age 12 to 23 months'
         ),
        'rp_acute_diarrhoea_age24to59mo': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for age 24 to 59 months'
         ),
        'rp_acute_diarrhoea_HIV': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for HIV positive'
         ),
        'rp_acute_diarrhoea_SAM': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for severe acute malnutrition'
         ),
        'rp_acute_diarrhoea_excl_breast': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for exclusive breastfeeding upto 6 months'
         ),
        'rp_acute_diarrhoea_cont_breast': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for continued breastfeeding 6 months to 2 years'
         ),
        'rp_acute_diarrhoea_HHhandwashing': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for household handwashing'
         ),
        'rp_acute_diarrhoea_clean_water': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for access to clean water'
         ),
        'rp_acute_diarrhoea_improved_sanitation': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for improved sanitation'
         ),
        'rp_acute_diarrhoea_wealth1': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for wealth level 1'
         ),
        'rp_acute_diarrhoea_wealth2': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for wealth level 2'
         ),
        'rp_acute_diarrhoea_wealth4': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for wealth level 4'
         ),
        'rp_acute_diarrhoea_wealth5': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for wealth level 5'
         ),
        'base_incidence_acute_diarrhoea': Parameter
        (Types.REAL,
         'baseline incidence of acute watery diarrhoea, among children aged < 11 months, '
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no clean water source, no improved sanitation, wealth level 3'
         ),
        'rr_acute_diarrhoea_agelt11mo': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for age < 11 months'
         ),
        'rr_acute_diarrhoea_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for age 12 to 23 months'
         ),
        'rr_acute_diarrhoea_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for age 24 to 59 months'
         ),
        'rr_acute_diarrhoea_HIV': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for HIV positive'
         ),
        'rr_acute_diarrhoea_SAM': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for severe acute malnutrition'
         ),
        'rr_acute_diarrhoea_excl_breast': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for exclusive breastfeeding upto 6 months'
         ),
        'rr_acute_diarrhoea_cont_breast': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for continued breastfeeding 6 months to 2 years'
         ),
        'rr_acute_diarrhoea_HHhandwashing': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for household handwashing'
         ),
        'rr_acute_diarrhoea_clean_water': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for access to clean water'
         ),
        'rr_acute_diarrhoea_improved_sanitation': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for improved sanitation'
         ),
        'rr_acute_diarrhoea_wealth1': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for wealth level 1'
         ),
        'rr_acute_diarrhoea_wealth2': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for wealth level 2'
         ),
        'rr_acute_diarrhoea_wealth4': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for wealth level 4'
         ),
        'rr_acute_diarrhoea_wealth5': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for wealth level 5'
         ),
        'base_prev_persistent_diarrhoea': Parameter
        (Types.REAL,
         'initial prevalence of persistent diarrhoea, among children aged < 11 months,'
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no access to clean water, no improved sanitation, wealth level 3'
         ),
        'rp_persistent_diarrhoea_agelt11mo': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for age < 11 months'
         ),
        'rp_persistent_diarrhoea_age12to23mo': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for age 12 to 23 months'
         ),
        'rp_persistent_diarrhoea_age24to59mo': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for age 24 to 59 months'
         ),
        'rp_persistent_diarrhoea_HIV': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for HIV positive'
         ),
        'rp_persistent_diarrhoea_SAM': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for severe acute malnutrition'
         ),
        'rp_persistent_diarrhoea_excl_breast': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for exclusive breastfeeding upto 6 months'
         ),
        'rp_persistent_diarrhoea_cont_breast': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for continued breastfeeding 6 months to 2 years'
         ),
        'rp_persistent_diarrhoea_HHhandwashing': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for household handwashing'
         ),
        'rp_persistent_diarrhoea_clean_water': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for access to clean water'
         ),
        'rp_persistent_diarrhoea_improved_sanitation': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for improved sanitation'
         ),
        'rp_persistent_diarrhoea_wealth1': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for wealth level 1'
         ),
        'rp_persistent_diarrhoea_wealth2': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for wealth level 2'
         ),
        'rp_persistent_diarrhoea_wealth4': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for wealth level 4'
         ),
        'rp_persistent_diarrhoea_wealth5': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for wealth level 5'
         ),
        'base_incidence_persistent_diarrhoea': Parameter
        (Types.REAL,
         'initial prevalence of persistent diarrhoea, among children aged < 11 months,'
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no access to clean water, no improved sanitation, wealth level 3'
         ),
        'rr_persistent_diarrhoea_agelt11mo': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for age < 11 months'
         ),
        'rr_persistent_diarrhoea_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for age 12 to 23 months'
         ),
        'rr_persistent_diarrhoea_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for age 24 to 59 months'
         ),
        'rr_persistent_diarrhoea_HIV': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for HIV positive'
         ),
        'rr_persistent_diarrhoea_SAM': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for severe acute malnutrition'
         ),
        'rr_persistent_diarrhoea_excl_breast': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for exclusive breastfeeding upto 6 months'
         ),
        'rr_persistent_diarrhoea_cont_breast': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for continued breastfeeding 6 months to 2 years'
         ),
        'rr_persistent_diarrhoea_HHhandwashing': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for household handwashing'
         ),
        'rr_persistent_diarrhoea_clean_water': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for access to clean water'
         ),
        'rr_persistent_diarrhoea_improved_sanitation': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for improved sanitation'
         ),
        'rr_persistent_diarrhoea_wealth1': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for wealth level 1'
         ),
        'rr_persistent_diarrhoea_wealth2': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for wealth level 2'
         ),
        'rr_persistent_diarrhoea_wealth4': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for wealth level 4'
         ),
        'rr_persistent_diarrhoea_wealth5': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for wealth level 5'
         ),
        'r_progress_to_persistent_diar': Parameter
        (Types.REAL,
         'probability of progressing from acute watery diarrhoea to persistent diarrhoea among children < 11 months, '
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no access to clean water, no improved sanitation, wealth level 3'
         ),
        'rr_progress_persistent_diar_agelt11mo': Parameter
        (Types.REAL,
         'relative rate of progression to persistent diarrhoea for age <11 months'
         ),
        'rr_progress_persistent_diar_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of progression to persistent diarrhoea for age 12 to 23 months'
         ),
        'rr_progress_persistent_diar_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of progression to persistent diarrhoea for age 24 to 59 months'
         ),
        'rr_progress_persistent_diar_HIV': Parameter
        (Types.REAL,
         'relative rate of progression to persistent diarrhoea for HIV positive status'
         ),
        'rr_progress_persistent_diar_SAM': Parameter
        (Types.REAL,
         'relative rate of progression to persistent diarrhoea for severe acute malnutrition'
         ),
        'rr_progress_persistent_diar_excl_breast': Parameter
        (Types.REAL,
         'relative rate of progression to persistent diarrhoea for exclusive breastfeeding'
         ),
        'rr_progress_persistent_diar_cont_breast': Parameter
        (Types.REAL,
         'relative rate of progression to persistent diarrhoea for continued breastfeeding'
         ),
        'rr_progress_persistent_diar_HHhandwashing': Parameter
        (Types.REAL,
         'relative rate of progression to persistent diarrhoea for household handwashing'
         ),
        'rr_progress_persistent_diar_clean_water': Parameter
        (Types.REAL,
         'relative rate of progression to persistent diarrhoea for access to clean water'
         ),
        'rr_progress_persistent_diar_improved_sanitation': Parameter
        (Types.REAL,
         'relative rate of progression to persistent diarrhoea for improved sanitation'
         ),
        'rr_progress_persistent_diar_wealth1': Parameter
        (Types.REAL,
         'relative rate of progression to persistent diarrhoea for wealth level 1'
         ),
        'rr_progress_persistent_diar_wealth2': Parameter
        (Types.REAL,
         'relative rate of progression to persistent diarrhoea for wealth level 2'
         ),
        'rr_progress_persistent_diar_wealth4': Parameter
        (Types.REAL,
         'relative rate of progression to persistent diarrhoea for wealth level 4'
         ),
        'rr_progress_persistent_diar_wealth5': Parameter
        (Types.REAL,
         'relative rate of progression to persistent diarrhoea for wealth level 5'
         ),
        'r_death_dysentery': Parameter
        (Types.REAL,
         'death rate from dysentery among children aged less than 11 months, '
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no access to clean water, no improved sanitation, wealth level 3'
         ),
        'rr_death_dysentery_agelt11mo': Parameter
        (Types.REAL,
         'relative rate of death from dysentery for age < 11 months'
         ),
        'rr_death_dysentery_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of death from dysentery for age 12 to 23 months'
         ),
        'rr_death_dysentery_diar_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of death from dysentery for age 24 to 59 months'
         ),
        'rr_death_dysentery_diar_HIV': Parameter
        (Types.REAL,
         'relative rate of death from dysentery for HIV positive status'
         ),
        'rr_death_dysentery_diar_SAM': Parameter
        (Types.REAL,
         'relative rate of death from dysentery for severe acute malnutrition'
         ),
        'rr_death_dysentery_diar_excl_breast': Parameter
        (Types.REAL,
         'relative rate of death from dysentery for exclusive breastfeeding'
         ),
        'rr_death_dysentery_diar_cont_breast': Parameter
        (Types.REAL,
         'relative rate of death from dysentery for continued breastfeeding'
         ),
        'rr_death_dysentery_diar_wealth1': Parameter
        (Types.REAL,
         'relative rate of death from dysentery for wealth level 1'
         ),
        'rr_death_dysentery_diar_wealth2': Parameter
        (Types.REAL,
         'relative rate of death from dysentery for wealth level 2'
         ),
        'rr_death_dysentery_diar_wealth4': Parameter
        (Types.REAL,
         'relative rate of death from dysentery for wealth level 4'
         ),
        'rr_death_dysentery_diar_wealth5': Parameter
        (Types.REAL,
         'relative rate of death from dysentery for wealth level 5'
         ),
        'r_recovery_dysentery': Parameter
        (Types.REAL,
         'recovery rate from dysentery among children aged 2-11 months, '
         'HIV negative, no SAM,  '
         ),
        'rr_recovery_dysentery_agelt11mo': Parameter
        (Types.REAL,
         'relative rate of recovery from dysentery for age < 11 months'
         ),
        'rr_recovery_pneumonia_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of recovery from dysentery for age between 12 to 23 months'
         ),
        'rr_recovery_dysentery_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of recovery from dysentery for age between 24 to 59 months'
         ),
        'rr_recovery_dysentery_HIV': Parameter
        (Types.REAL,
         'relative rate of recovery from dysentery for HIV positive status'
         ),
        'rr_recovery_dysentery_SAM': Parameter
        (Types.REAL,
         'relative rate of recovery from dysentery for severe acute malnutrition'
         ),
        'r_recovery_acute_diarrhoea': Parameter
        (Types.REAL,
         'baseline recovery rate from acute watery diarrhoea among children ages 2 to 11 months, '
         'HIV negative, no SAM'
         ),
        'rr_recovery_acute_diarrhoea_agelt11mo': Parameter
        (Types.REAL,
         'relative rate of recovery from acute watery diarrhoea for age <11 months'
         ),
        'rr_recovery_acute_diarrhoea_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of recovery from acute watery diarrhoea for age between 12 to 23 months'
         ),
        'rr_recovery_acute_diarrhoea_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of recovery from acute watery diarrhoea for age between 24 to 59 months'
         ),
        'rr_recovery_acute_diarrhoea_HIV': Parameter
        (Types.REAL,
         'relative rate of recovery from acute watery diarrhoea for HIV positive status'
         ),
        'rr_recovery_acute_diarrhoea_SAM': Parameter
        (Types.REAL,
         'relative rate of recovery from acute watery diarrhoea for severe acute malnutrition'
         ),
        'r_recovery_persistent_diarrhoea': Parameter
        (Types.REAL,
         'baseline recovery rate from persistent diarrhoea among children ages 2 to 11 months, '
         'HIV negative, no SAM'
         ),
        'rr_recovery_persistent_diarrhoea_agelt11mo': Parameter
        (Types.REAL,
         'relative rate of recovery from acute watery diarrhoea for age <11 months'
         ),
        'rr_recovery_persistent_diarrhoea_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of recovery from acute watery diarrhoea for age between 12 to 23 months'
         ),
        'rr_recovery_persistent_diarrhoea_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of recovery from acute watery diarrhoea for age between 24 to 59 months'
         ),
        'rr_recovery_persistent_diarrhoea_HIV': Parameter
        (Types.REAL,
         'relative rate of recovery from acute watery diarrhoea for HIV positive status'
         ),
        'rr_recovery_persistent_diarrhoea_SAM': Parameter
        (Types.REAL,
         'relative rate of recovery from acute watery diarrhoea for severe acute malnutrition'
         ),
        'init_prop_pneumonia_status': Parameter
        (Types.LIST,
         'initial proportions in ri_pneumonia_status categories '
         'for children aged 2-11 months, HIV negative, no SAM, '
         'not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no indoor air pollution, wealth level 3'
         )
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'ei_diarrhoea_status': Property(Types.CATEGORICAL, 'enteric infections - diarrhoea status',
                                        categories=['none', 'dysentery', 'acute watery diarrhoea',
                                                    'persistent diarrhoea']),
        'has_hiv': Property(Types.BOOL, 'temporary property - has hiv'),
        'malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'indoor_air_pollution': Property(Types.BOOL, 'temporary property - indoor air pollution'),
        'exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
        'HHhandwashing': Property(Types.BOOL, 'temporary property - household handwashing'),
        'clean_water': Property(Types.BOOL, 'temporary property - access to clean water sources'),
        'improved_sanitation': Property(Types.BOOL, 'temporary property - improved sanitation')
    }

    def read_parameters(self, data_folder):
        """ Setup parameters values used by the module
        """
        p = self.parameters

        p['base_prev_dysentery'] = 0.2
        p['rp_dysentery_agelt11mo'] = 1.2
        p['rp_dysentery_age12to23mo'] = 0.8
        p['rp_dysentery_age24to59mo'] = 0.5
        p['rp_dysentery_HIV'] = 1.4
        p['rp_dysentery_SAM'] = 1.25
        p['rp_dysentery_excl_breast'] = 0.5
        p['rp_dysentery_cont_breast'] = 0.7
        p['rp_dysentery_HHhandwashing'] = 0.5
        p['rp_dysentery_clean_water'] = 0.6
        p['rp_dysentery_improved_sanitation'] = 1.1
        p['rp_dysentery_wealth1'] = 0.8
        p['rp_dysentery_wealth2'] = 0.9
        p['rp_dysentery_wealth4'] = 1.2
        p['rp_dysentery_wealth5'] = 1.3
        p['base_incidence_dysentery'] = 0.2
        p['rr_dysentery_agelt11mo'] = 1.2
        p['rr_dysentery_age12to23mo'] = 0.8
        p['rr_dysentery_age24to59mo'] = 0.5
        p['rr_dysentery_HIV'] = 1.4
        p['rr_dysentery_SAM'] = 1.25
        p['rr_dysentery_excl_breast'] = 0.5
        p['rr_dysentery_cont_breast'] = 0.7
        p['rr_dysentery_HHhandwashing'] = 0.5
        p['rr_dysentery_clean_water'] = 0.6
        p['rr_dysentery_improved_sanitation'] = 1.1
        p['rr_dysentery_wealth1'] = 0.8
        p['rr_dysentery_wealth2'] = 0.9
        p['rr_dysentery_wealth4'] = 1.2
        p['rr_dysentery_wealth5'] = 1.3
        p['base_prev_acute_diarrhoea'] = 0.4
        p['rp_acute_diarrhoea_agelt11mo'] = 1.3
        p['rp_acute_diarrhoea_age12to23mo'] = 0.8
        p['rp_acute_diarrhoea_age24to59mo'] = 0.5
        p['rp_acute_diarrhoea_HIV'] = 1.3
        p['rp_acute_diarrhoea_SAM'] = 1.3
        p['rp_acute_diarrhoea_excl_breast'] = 0.5
        p['rp_acute_diarrhoea_cont_breast'] = 0.7
        p['rp_acute_diarrhoea_HHhandwashing'] = 0.8
        p['rp_acute_diarrhoea_clean_water'] = 0.6
        p['rp_acute_diarrhoea_improved_sanitation'] = 1.1
        p['rp_acute_diarrhoea_wealth1'] = 0.8
        p['rp_acute_diarrhoea_wealth2'] = 0.9
        p['rp_acute_diarrhoea_wealth4'] = 1.2
        p['rp_acute_diarrhoea_wealth5'] = 1.3
        p['base_incidence_acute_diarrhoea'] = 0.5
        p['rr_acute_diarrhoea_agelt11mo'] = 1.3
        p['rr_acute_diarrhoea_age12to23mo'] = 0.8
        p['rr_acute_diarrhoea_age24to59mo'] = 0.5
        p['rr_acute_diarrhoea_HIV'] = 1.3
        p['rr_acute_diarrhoea_SAM'] = 1.3
        p['rr_acute_diarrhoea_excl_breast'] = 0.5
        p['rr_acute_diarrhoea_cont_breast'] = 0.7
        p['rr_acute_diarrhoea_HHhandwashing'] = 0.8
        p['rr_acute_diarrhoea_clean_water'] = 0.6
        p['rr_acute_diarrhoea_improved_sanitation'] = 1.1
        p['rr_acute_diarrhoea_wealth1'] = 0.8
        p['rr_acute_diarrhoea_wealth2'] = 0.9
        p['rr_acute_diarrhoea_wealth4'] = 1.2
        p['rr_acute_diarrhoea_wealth5'] = 1.3
        p['base_prev_persistent_diarrhoea'] = 0.2
        p['rp_persistent_diarrhoea_agelt11mo'] = 1.3
        p['rp_persistent_diarrhoea_age12to23mo'] = 0.8
        p['rp_persistent_diarrhoea_age24to59mo'] = 0.5
        p['rp_persistent_diarrhoea_HIV'] = 1.3
        p['rp_persistent_diarrhoea_SAM'] = 1.3
        p['rp_persistent_diarrhoea_excl_breast'] = 0.5
        p['rp_persistent_diarrhoea_cont_breast'] = 0.7
        p['rp_persistent_diarrhoea_HHhandwashing'] = 0.8
        p['rp_persistent_diarrhoea_clean_water'] = 0.6
        p['rp_persistent_diarrhoea_improved_sanitation'] = 1.1
        p['rp_persistent_diarrhoea_wealth1'] = 0.8
        p['rp_persistent_diarrhoea_wealth2'] = 0.9
        p['rp_persistent_diarrhoea_wealth4'] = 1.2
        p['rp_persistent_diarrhoea_wealth5'] = 1.3
        p['base_incidence_persistent_diarrhoea'] = 0.5
        p['rr_persistent_diarrhoea_agelt11mo'] = 1.3
        p['rr_persistent_diarrhoea_age12to23mo'] = 0.8
        p['rr_persistent_diarrhoea_age24to59mo'] = 0.5
        p['rr_persistent_diarrhoea_HIV'] = 1.3
        p['rr_persistent_diarrhoea_SAM'] = 1.3
        p['rr_persistent_diarrhoea_excl_breast'] = 0.5
        p['rr_persistent_diarrhoea_cont_breast'] = 0.7
        p['rr_persistent_diarrhoea_HHhandwashing'] = 0.8
        p['rr_persistent_diarrhoea_clean_water'] = 0.6
        p['rr_persistent_diarrhoea_improved_sanitation'] = 1.1
        p['rr_persistent_diarrhoea_wealth1'] = 0.8
        p['rr_persistent_diarrhoea_wealth2'] = 0.9
        p['rr_persistent_diarrhoea_wealth4'] = 1.2
        p['rr_persistent_diarrhoea_wealth5'] = 1.3
        p['init_prop_diarrhoea_status'] = [0.2, 0.2, 0.2]
        p['r_recovery_dysentery'] = 0.9
        p['rr_recovery_dysentery_agelt11mo'] = 0.7
        p['rr_recovery_dysentery_age12to23mo'] = 0.9
        p['rr_recovery_dysentery_age24to59mo'] = 1.1
        p['rr_recovery_dysentery_HIV'] = 0.5
        p['rr_recovery_dysentery_SAM'] = 0.5
        p['r_recovery_acute_diarrhoea'] = 0.6
        p['rr_recovery_acute_diarrhoea_agelt11mo'] = 0.7
        p['rr_recovery_acute_diarrhoea_age12to23mo'] = 0.9
        p['rr_recovery_acute_diarrhoea_age24to59mo'] = 1.2
        p['rr_recovery_acute_diarrhoea_HIV'] = 0.6
        p['rr_recovery_acute_diarrhoea_SAM'] = 0.3
        p['r_recovery_persistent_diarrhoea'] = 0.8
        p['rr_recovery_persistent_diarrhoea_agelt11mo'] = 0.8
        p['rr_recovery_persistent_diarrhoea_age12to23mo'] = 1.2
        p['rr_recovery_persistent_diarrhoea_age24to59mo'] = 1.3
        p['rr_recovery_persistent_diarrhoea_HIV'] = 0.6
        p['rr_recovery_persistent_diarrhoea_SAM'] = 0.5

    def initialise_population(self, population):
        """Set our property values for the initial population.
        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.
        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals
        m = self
        rng = m.rng

        # -------------------- DEFAULTS ------------------------------------------------------------

        df['ei_diarrhoea_status'] = 'none'
        df['malnutrition'] = False
        df['has_HIV'] = False
        df['HHhandwashing'] = False
        df['exclusive_breastfeeding'] = False
        df['continued_breastfeeding'] = False
        df['clean_water'] = False
        df['improved_sanitation'] = False

        # -------------------- ASSIGN VALUES OF ENTERIC INFECTION STATUS AT BASELINE -----------

        under5_idx = df.index[(df.age_years < 5) & df.is_alive]

        # create data-frame of the probabilities of ei_diarrhoea_status for children
        # aged 2-11 months, HIV negative, no SAM,
        p_dysentery_status = pd.Series(self.init_prop_diarrhoea_status[0], index=under5_idx)
        p_acute_diarrhoea_status = pd.Series(self.init_prop_diarrhoea_status[1], index=under5_idx)
        p_persistent_diarrhoea_status = pd.Series(self.init_prop_diarrhoea_status[2], index=under5_idx)

        # create probabilities of dysentery  for all age under 5
        p_dysentery_status.loc[
            (df.age_exact_years < 1) & df.is_alive] *= self.rp_dysentery_agelt11mo
        p_dysentery_status.loc[
            (df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] *= self.rp_dysentery_age12to23mo
        p_dysentery_status.loc[
            (df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] *= self.rp_dysentery_age24to59mo
        p_dysentery_status.loc[
            (df.has_hiv == True) & (df.age_years < 5) & df.is_alive] *= self.rp_dysentery_HIV
        p_dysentery_status.loc[
            (df.malnutrition == True) & (df.age_years < 5) & df.is_alive] *= self.rp_dysentery_SAM
        p_dysentery_status.loc[
            (df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5) & df.is_alive] \
            *= self.rp_dysentery_excl_breast
        p_dysentery_status.loc[
            (df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) & (df.age_exact_years < 2) &
            df.is_alive] *= self.rp_dysentery_cont_breast
        p_dysentery_status.loc[
            (df.clean_water == True) & (df.age_years < 5) & df.is_alive] *= self.rp_dysentery_clean_water
        p_dysentery_status.loc[
            (df.improved_sanitation == True) & (df.age_years < 5) &
            df.is_alive] *= self.rp_dysentery_improved_sanitation
        p_dysentery_status.loc[
            (df.li_wealth == 1) & (df.age_years < 5) & df.is_alive] *= self.rp_dysentery_wealth1
        p_dysentery_status.loc[
            (df.li_wealth == 2) & (df.age_years < 5) & df.is_alive] *= self.rp_dysentery_wealth2
        p_dysentery_status.loc[
            (df.li_wealth == 4) & (df.age_years < 5) & df.is_alive] *= self.rp_dysentery_wealth4
        p_dysentery_status.loc[
            (df.li_wealth == 5) & (df.age_years < 5) & df.is_alive] *= self.rp_dysentery_wealth5

        # create probabilities of acute watery diarrhoea for all age under 5
        p_acute_diarrhoea_status.loc[
            (df.age_exact_years < 1) & df.is_alive] *= self.rp_acute_diarrhoea_agelt11mo
        p_acute_diarrhoea_status.loc[
            (df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] *= self.rp_acute_diarrhoea_age12to23mo
        p_acute_diarrhoea_status.loc[
            (df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] *= self.rp_acute_diarrhoea_age24to59mo
        p_acute_diarrhoea_status.loc[
            (df.has_hiv == True) & (df.age_years < 5) & df.is_alive] *= self.rp_acute_diarrhoea_HIV
        p_acute_diarrhoea_status.loc[
            (df.malnutrition == True) & (df.age_years < 5) & df.is_alive] *= self.rp_acute_diarrhoea_SAM
        p_acute_diarrhoea_status.loc[
            (df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5) & df.is_alive] \
            *= self.rp_acute_diarrhoea_excl_breast
        p_acute_diarrhoea_status.loc[
            (df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) & (df.age_exact_years < 2) &
            df.is_alive] *= self.rp_acute_diarrhoea_cont_breast
        p_acute_diarrhoea_status.loc[
            (df.clean_water == True) & (df.age_years < 5) & df.is_alive] *= self.rp_acute_diarrhoea_clean_water
        p_acute_diarrhoea_status.loc[
            (df.improved_sanitation == True) & (df.age_years < 5) &
            df.is_alive] *= self.rp_acute_diarrhoea_improved_sanitation
        p_acute_diarrhoea_status.loc[
            (df.li_wealth == 1) & (df.age_years < 5) & df.is_alive] *= self.rp_acute_diarrhoea_wealth1
        p_acute_diarrhoea_status.loc[
            (df.li_wealth == 2) & (df.age_years < 5) & df.is_alive] *= self.rp_acute_diarrhoea_wealth2
        p_acute_diarrhoea_status.loc[
            (df.li_wealth == 4) & (df.age_years < 5) & df.is_alive] *= self.rp_acute_diarrhoea_wealth4
        p_acute_diarrhoea_status.loc[
            (df.li_wealth == 5) & (df.age_years < 5) & df.is_alive] *= self.rp_acute_diarrhoea_wealth5

        # create probabilities of persistent diarrhoea for all age under 5
        p_persistent_diarrhoea_status.loc[
            (df.age_exact_years < 1) & df.is_alive] *= self.rp_persistent_diarrhoea_agelt11mo
        p_persistent_diarrhoea_status.loc[
            (df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] *= self.rp_persistent_diarrhoea_age12to23mo
        p_persistent_diarrhoea_status.loc[
            (df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] *= self.rp_persistent_diarrhoea_age24to59mo
        p_persistent_diarrhoea_status.loc[
            (df.has_hiv == True) & (df.age_years < 5) & df.is_alive] *= self.rp_persistent_diarrhoea_HIV
        p_persistent_diarrhoea_status.loc[
            (df.malnutrition == True) & (df.age_years < 5) & df.is_alive] *= self.rp_persistent_diarrhoea_SAM
        p_persistent_diarrhoea_status.loc[
            (df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5) & df.is_alive] \
            *= self.rp_persistent_diarrhoea_excl_breast
        p_persistent_diarrhoea_status.loc[
            (df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) & (df.age_exact_years < 2) &
            df.is_alive] *= self.rp_persistent_diarrhoea_cont_breast
        p_persistent_diarrhoea_status.loc[
            (df.clean_water == True) & (df.age_years < 5) & df.is_alive] *= self.rp_persistent_diarrhoea_clean_water
        p_persistent_diarrhoea_status.loc[
            (df.improved_sanitation == True) & (df.age_years < 5) &
            df.is_alive] *= self.rp_persistent_diarrhoea_improved_sanitation
        p_persistent_diarrhoea_status.loc[
            (df.li_wealth == 1) & (df.age_years < 5) & df.is_alive] *= self.rp_persistent_diarrhoea_wealth1
        p_persistent_diarrhoea_status.loc[
            (df.li_wealth == 2) & (df.age_years < 5) & df.is_alive] *= self.rp_persistent_diarrhoea_wealth2
        p_persistent_diarrhoea_status.loc[
            (df.li_wealth == 4) & (df.age_years < 5) & df.is_alive] *= self.rp_persistent_diarrhoea_wealth4
        p_persistent_diarrhoea_status.loc[
            (df.li_wealth == 5) & (df.age_years < 5) & df.is_alive] *= self.rp_persistent_diarrhoea_wealth5

        random_draw = pd.Series(rng.random_sample(size=len(under5_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive])

        # create a temporary dataframe called dfx to hold values of probabilities and random draw
        dfx = pd.concat([p_dysentery_status, p_acute_diarrhoea_status, p_persistent_diarrhoea_status, random_draw], axis=1)
        dfx.columns = ['p_dysentery', 'p_acute_diarrhoea', 'p_persistent_diarrhoea', 'random_draw']

        dfx['p_none'] = 1 - (dfx.p_dysentery + dfx.p_acute_diarrhoea + dfx.p_persistent_diarrhoea)

        # based on probabilities of being in each category, define cut-offs to determine status from
        # random draw uniform(0,1)

        # assign baseline values of ri_resp_infection_stat based on probabilities and value of random draw

        idx_none = dfx.index[dfx.p_none > dfx.random_draw]
        idx_dysentery = dfx.index[(dfx.p_none < dfx.random_draw) & ((dfx.p_none + dfx.p_dysentery) > dfx.random_draw)]
        idx_acute_diarrhoea = dfx.index[((dfx.p_none + dfx.p_dysentery) < dfx.random_draw) &
                                         (dfx.p_none + dfx.p_dysentery + dfx.p_acute_diarrhoea) > dfx.random_draw]
        idx_persistent_diarrhoea = dfx.index[((dfx.p_none + dfx.p_dysentery + dfx.p_acute_diarrhoea) < dfx.random_draw) &
                                         (dfx.p_none + dfx.p_dysentery + dfx.p_acute_diarrhoea +
                                          dfx.p_persistent_diarrhoea) > dfx.random_draw]

        df.loc[idx_none, 'ei_diarrhoea_status'] = 'none'
        df.loc[idx_dysentery, 'ei_diarrhoea_status'] = 'dysentery'
        df.loc[idx_acute_diarrhoea, 'ei_diarrhoea_status'] = 'acute watery diarrhoea'
        df.loc[idx_persistent_diarrhoea, 'ei_diarrhoea_status'] = 'persistent diarrhoea'


    def initialise_simulation(self, sim):
        """
        Get ready for simulation start.
        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event
        event = EntericInfectionEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(weeks=1))

        # add an event to log to screen
        sim.schedule_event(EntericInfectionLoggingEvent(self), sim.date + DateOffset(weeks=1))

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        df.at[child_id, 'ei_diarrhoea_status'] = 'none'


class EntericInfectionEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all Respiratory Infection properties for population
    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module):
        """schedule to run every 7 days
        note: if change this offset from 1 week need to consider code conditioning on age.years_exact
        We need to pass the frequency at which we want to occur to the base class
        constructor using super(). We also pass the module that created this event,
        so that random number generators can be scoped per-module.
        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(days=14))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = m.rng

        # ------------------- UPDATING OF LOWER RESPIRATORY INFECTION - PNEUMONIA STATUS OVER TIME -------------------

        # updating for children under 5 with current status 'none'

        di_current_none_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'none') & (df.age_years < 5)]
        di_current_none_agelt11mo_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'none') & (df.age_exact_years < 1)]
        di_current_none_age12to23mo_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                     (df.age_exact_years >= 1) & (df.age_exact_years < 2)]
        di_current_none_age24to59mo_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                     (df.age_exact_years >= 2) & (df.age_exact_years < 5)]
        di_current_none_handwashing_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                     df.HHhandwashing & (df.age_years < 5)]
        di_current_none_HIV_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                     (df.has_hiv) & (df.age_years < 5)]
        di_current_none_SAM_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                     df.malnutrition & (df.age_years < 5)]
        di_current_none_excl_breast_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                     df.exclusive_breastfeeding & (df.age_exact_years <= 0.5)]
        di_current_none_cont_breast_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                     df.continued_breastfeeding & (df.age_exact_years > 0.5) & (df.age_exact_years < 2)]
        di_current_none_clean_water_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                     df.clean_water & (df.age_years < 5)]
        di_current_none_improved_sanitation_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'none') & df.improved_sanitation & (df.age_years < 5)]
        di_current_none_wealth1_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                     (df.li_wealth == 1) & (df.age_years < 5)]
        di_current_none_wealth2_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                     (df.li_wealth == 2) & (df.age_years < 5)]
        di_current_none_wealth4_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                     (df.li_wealth == 4) & (df.age_years < 5)]
        di_current_none_wealth5_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                     (df.li_wealth == 5) & (df.age_years < 5)]

        eff_prob_ei_dysentery = pd.Series(m.base_incidence_dysentery,
                                          index=df.index[
                                              df.is_alive & (df.ei_diarrhoea_status == 'none') & (
                                                  df.age_years < 5)])

        eff_prob_ei_dysentery.loc[di_current_none_agelt11mo_idx] *= m.rr_dysentery_agelt11mo
        eff_prob_ei_dysentery.loc[di_current_none_age12to23mo_idx] *= m.rr_dysentery_age12to23mo
        eff_prob_ei_dysentery.loc[di_current_none_age24to59mo_idx] *= m.rr_dysentery_age24to59mo
        eff_prob_ei_dysentery.loc[di_current_none_handwashing_idx] *= m.rr_dysentery_HHhandwashing
        eff_prob_ei_dysentery.loc[di_current_none_HIV_idx] *= m.rr_dysentery_HIV
        eff_prob_ei_dysentery.loc[di_current_none_SAM_idx] *= m.rr_dysentery_SAM
        eff_prob_ei_dysentery.loc[di_current_none_excl_breast_idx] *= m.rr_dysentery_excl_breast
        eff_prob_ei_dysentery.loc[di_current_none_cont_breast_idx] *= m.rr_dysentery_cont_breast
        eff_prob_ei_dysentery.loc[di_current_none_clean_water_idx] *= m.rr_dysentery_clean_water
        eff_prob_ei_dysentery.loc[di_current_none_improved_sanitation_idx] *= m.rr_dysentery_improved_sanitation
        eff_prob_ei_dysentery.loc[di_current_none_wealth1_idx] *= m.rr_dysentery_wealth1
        eff_prob_ei_dysentery.loc[di_current_none_wealth2_idx] *= m.rr_dysentery_wealth2
        eff_prob_ei_dysentery.loc[di_current_none_wealth4_idx] *= m.rr_dysentery_wealth4
        eff_prob_ei_dysentery.loc[di_current_none_wealth5_idx] *= m.rr_dysentery_wealth5

        eff_prob_ei_acute_diarrhoea = pd.Series(m.base_incidence_acute_diarrhoea,
                                          index=df.index[
                                              df.is_alive & (df.ei_diarrhoea_status == 'none') & (
                                                  df.age_years < 5)])
        eff_prob_ei_acute_diarrhoea.loc[di_current_none_agelt11mo_idx] *= m.rr_acute_diarrhoea_agelt11mo
        eff_prob_ei_acute_diarrhoea.loc[di_current_none_age12to23mo_idx] *= m.rr_acute_diarrhoea_age12to23mo
        eff_prob_ei_acute_diarrhoea.loc[di_current_none_age24to59mo_idx] *= m.rr_acute_diarrhoea_age24to59mo
        eff_prob_ei_acute_diarrhoea.loc[di_current_none_handwashing_idx] *= m.rr_acute_diarrhoea_HHhandwashing
        eff_prob_ei_acute_diarrhoea.loc[di_current_none_HIV_idx] *= m.rr_acute_diarrhoea_HIV
        eff_prob_ei_acute_diarrhoea.loc[di_current_none_SAM_idx] *= m.rr_acute_diarrhoea_SAM
        eff_prob_ei_acute_diarrhoea.loc[di_current_none_excl_breast_idx] *= m.rr_acute_diarrhoea_excl_breast
        eff_prob_ei_acute_diarrhoea.loc[di_current_none_cont_breast_idx] *= m.rr_acute_diarrhoea_cont_breast
        eff_prob_ei_acute_diarrhoea.loc[di_current_none_clean_water_idx] *= m.rr_acute_diarrhoea_clean_water
        eff_prob_ei_acute_diarrhoea.loc[di_current_none_improved_sanitation_idx] *= m.rr_acute_diarrhoea_improved_sanitation
        eff_prob_ei_acute_diarrhoea.loc[di_current_none_wealth1_idx] *= m.rr_acute_diarrhoea_wealth1
        eff_prob_ei_acute_diarrhoea.loc[di_current_none_wealth2_idx] *= m.rr_acute_diarrhoea_wealth2
        eff_prob_ei_acute_diarrhoea.loc[di_current_none_wealth4_idx] *= m.rr_acute_diarrhoea_wealth4
        eff_prob_ei_acute_diarrhoea.loc[di_current_none_wealth5_idx] *= m.rr_acute_diarrhoea_wealth5

        eff_prob_ei_persistent_diarrhoea = pd.Series(m.base_incidence_persistent_diarrhoea,
                                                index=df.index[
                                                    df.is_alive & (df.ei_diarrhoea_status == 'none') & (
                                                        df.age_years < 5)])

        eff_prob_ei_persistent_diarrhoea.loc[di_current_none_agelt11mo_idx] *= m.rr_persistent_diarrhoea_agelt11mo
        eff_prob_ei_persistent_diarrhoea.loc[di_current_none_age12to23mo_idx] *= m.rr_persistent_diarrhoea_age12to23mo
        eff_prob_ei_persistent_diarrhoea.loc[di_current_none_age24to59mo_idx] *= m.rr_persistent_diarrhoea_age24to59mo
        eff_prob_ei_persistent_diarrhoea.loc[di_current_none_handwashing_idx] *= m.rr_persistent_diarrhoea_HHhandwashing
        eff_prob_ei_persistent_diarrhoea.loc[di_current_none_HIV_idx] *= m.rr_persistent_diarrhoea_HIV
        eff_prob_ei_persistent_diarrhoea.loc[di_current_none_SAM_idx] *= m.rr_persistent_diarrhoea_SAM
        eff_prob_ei_persistent_diarrhoea.loc[di_current_none_excl_breast_idx] *= m.rr_persistent_diarrhoea_excl_breast
        eff_prob_ei_persistent_diarrhoea.loc[di_current_none_cont_breast_idx] *= m.rr_persistent_diarrhoea_cont_breast
        eff_prob_ei_persistent_diarrhoea.loc[di_current_none_clean_water_idx] *= m.rr_persistent_diarrhoea_clean_water
        eff_prob_ei_persistent_diarrhoea.loc[
            di_current_none_improved_sanitation_idx] *= m.rr_persistent_diarrhoea_improved_sanitation
        eff_prob_ei_persistent_diarrhoea.loc[di_current_none_wealth1_idx] *= m.rr_persistent_diarrhoea_wealth1
        eff_prob_ei_persistent_diarrhoea.loc[di_current_none_wealth2_idx] *= m.rr_persistent_diarrhoea_wealth2
        eff_prob_ei_persistent_diarrhoea.loc[di_current_none_wealth4_idx] *= m.rr_persistent_diarrhoea_wealth4
        eff_prob_ei_persistent_diarrhoea.loc[di_current_none_wealth5_idx] *= m.rr_persistent_diarrhoea_wealth5

        random_draw_01 = pd.Series(rng.random_sample(size=len(di_current_none_idx)),
                                   index=df.index[
                                       (df.age_years < 5) & df.is_alive & (df.ei_diarrhoea_status == 'none')])

        dfx = pd.concat([eff_prob_ei_dysentery, eff_prob_ei_acute_diarrhoea, eff_prob_ei_persistent_diarrhoea,
                         random_draw_01], axis=1)
        dfx.columns = ['eff_prob_ei_dysentery', 'eff_prob_ei_acute_diarrhoea', 'eff_prob_ei_persistent_diarrhoea',
                       'random_draw_01']

        dfx['di_none'] = 1 - (dfx.eff_prob_ei_dysentery + dfx.eff_prob_ei_acute_diarrhoea +
                              dfx.eff_prob_ei_persistent_diarrhoea)

        idx_incident_di_none = dfx.index[dfx.di_none > dfx.random_draw_01]
        idx_incident_dysentery = dfx.index[
            (dfx.di_none < dfx.random_draw_01) & ((dfx.di_none + dfx.eff_prob_ei_dysentery) > dfx.random_draw_01)]
        idx_incident_acute_diarrhoea = dfx.index[((dfx.di_none + dfx.eff_prob_ei_dysentery) < dfx.random_draw_01) &
                                                  (dfx.di_none + dfx.eff_prob_ei_dysentery +
                                                   dfx.eff_prob_ei_acute_diarrhoea) > dfx.random_draw_01]
        idx_incident_persistent_diarrhoea = dfx.index[((dfx.di_none + dfx.eff_prob_ei_dysentery +
                                                   dfx.eff_prob_ei_acute_diarrhoea) < dfx.random_draw_01) &
                                                      (dfx.di_none + dfx.eff_prob_ei_dysentery +
                                                       dfx.eff_prob_ei_acute_diarrhoea + dfx.eff_prob_ei_persistent_diarrhoea)
                                                      > dfx.random_draw_01]

        df.loc[idx_incident_di_none, 'ei_diarrhoea_status'] = 'none'
        df.loc[idx_incident_dysentery, 'ei_diarrhoea_status'] = 'dysentery'
        df.loc[idx_incident_acute_diarrhoea, 'ei_diarrhoea_status'] = 'acute watery diarrhoea'
        df.loc[idx_incident_persistent_diarrhoea, 'ei_diarrhoea_status'] = 'persistent diarrhoea'

        # -------------------- UPDATING OF EI_DIARRHOEA_STATUS RECOVERY OVER TIME --------------------------------
        # recovery from dysentery

        di_current_dysentery_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'dysentery') & df.age_years < 5]
        di_current_dysentery_agelt11mo_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'dysentery') & (df.age_exact_years > 0)
                     & (df.age_exact_years < 1)]
        di_current_dysentery_age12to23mo_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'dysentery') &
                     (df.age_exact_years >= 1) & (df.age_exact_years < 2)]
        di_current_dysentery_age24to59mo_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'dysentery') &
                     (df.age_exact_years >= 2) & (df.age_exact_years < 5)]
        di_current_dysentery_HIV_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'dysentery') &
                     (df.has_hiv) & (df.age_exact_years < 5)]
        di_current_dysentery_SAM_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'dysentery') &
                     df.malnutrition & (df.age_exact_years < 5)]

        eff_prob_recovery_dysentery = pd.Series(m.r_recovery_dysentery,
                                                index=df.index[df.is_alive & (df.ei_diarrhoea_status == 'dysentery')
                                                               & (df.age_exact_years < 5)])
        eff_prob_recovery_dysentery.loc[di_current_dysentery_agelt11mo_idx] *= \
             m.rr_recovery_dysentery_agelt11mo
        eff_prob_recovery_dysentery.loc[di_current_dysentery_age12to23mo_idx] *= \
            m.rr_recovery_dysentery_age12to23mo
        eff_prob_recovery_dysentery.loc[di_current_dysentery_age24to59mo_idx] *= \
            m.rr_recovery_dysentery_age24to59mo
        eff_prob_recovery_dysentery.loc[di_current_dysentery_HIV_idx] *= \
            m.rr_recovery_dysentery_HIV
        eff_prob_recovery_dysentery.loc[di_current_dysentery_SAM_idx] *= \
            m.rr_recovery_dysentery_SAM#

        random_draw_02 = pd.Series(rng.random_sample(size=len(di_current_dysentery_idx)),
                                   index=df.index[df.is_alive & (df.ei_diarrhoea_status == 'dysentery') &
                                                                 df.age_years < 5])

        dfx = pd.concat([eff_prob_recovery_dysentery, random_draw_02], axis=1)
        dfx.columns = ['eff_prob_recovery_dysentery', 'random_draw_02']
        idx_recovery_dysentery = dfx.index[dfx.eff_prob_recovery_dysentery > dfx.random_draw_02]
        df.loc[idx_recovery_dysentery, 'ei_diarrhoea_status'] = 'none'

        # recovery from acute watery diarrhoea

        di_current_acute_diarrhoea_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'acute watery diarrhoea') & df.age_exact_years < 5]
        di_current_acute_diarrhoea_agelt11mo_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'acute watery diarrhoea') & df.age_exact_years < 1]
        di_current_acute_diarrhoea_age12to23mo_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'acute watery diarrhoea') &
                     (df.age_exact_years >= 1) & (df.age_exact_years < 2)]
        di_current_acute_diarrhoea_age24to59mo_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'acute watery diarrhoea') &
                     (df.age_exact_years >= 2) & (df.age_exact_years < 5)]
        di_current_acute_diarrhoea_HIV_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'acute watery diarrhoea') &
                     (df.has_hiv) & (df.age_exact_years < 5)]
        di_current_acute_diarrhoea_SAM_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'acute watery diarrhoea') &
                     df.malnutrition & (df.age_exact_years < 5)]

        eff_prob_recovery_acute_diarrhoea = \
            pd.Series(m.r_recovery_acute_diarrhoea,
                      index=df.index[df.is_alive & (df.ei_diarrhoea_status == 'acute watery diarrhoea') &
                                     (df.age_exact_years < 5)])

 #      eff_prob_recovery_acute_diarrhoea.loc[di_current_acute_diarrhoea_agelt11mo_idx] *= \
 #           m.rr_recovery_acute_diarrhoea_agelt11mo
        eff_prob_recovery_acute_diarrhoea.loc[di_current_acute_diarrhoea_age12to23mo_idx] *= \
            m.rr_recovery_acute_diarrhoea_age12to23mo
        eff_prob_recovery_acute_diarrhoea.loc[di_current_acute_diarrhoea_age24to59mo_idx] *= \
            m.rr_recovery_acute_diarrhoea_age24to59mo
        eff_prob_recovery_acute_diarrhoea.loc[di_current_acute_diarrhoea_HIV_idx] *= \
            m.rr_recovery_acute_diarrhoea_HIV
        eff_prob_recovery_acute_diarrhoea.loc[di_current_acute_diarrhoea_SAM_idx] *= \
            m.rr_recovery_acute_diarrhoea_SAM

        random_draw_03 = pd.Series(rng.random_sample(size=len(di_current_acute_diarrhoea_idx)),
                                   index=df.index[df.is_alive & (df.ei_diarrhoea_status == 'acute watery diarrhoea')
                                                  & df.age_exact_years < 5])

        dfx = pd.concat([eff_prob_recovery_acute_diarrhoea, random_draw_03], axis=1)
        dfx.columns = ['eff_prob_recovery_acute_diarrhoea', 'random_draw_03']
        idx_recovery_acute_diarrhoea = dfx.index[dfx.eff_prob_recovery_acute_diarrhoea > dfx.random_draw_03]
        df.loc[idx_recovery_acute_diarrhoea, 'ei_diarrhoea_status'] = 'none'

        # recovery from persistent diarrhoea

        di_current_persistent_diarrhoea_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'persistent diarrhoea') & df.age_years < 5]
        di_current_persistent_diarrhoea_agelt11mo_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'persistent diarrhoea') & df.age_exact_years < 1]
        di_current_persistent_diarrhoea_age12to23mo_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'persistent diarrhoea') &
                     (df.age_exact_years >= 1) & (df.age_exact_years < 2)]
        di_current_persistent_diarrhoea_age24to59mo_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'persistent diarrhoea') &
                     (df.age_exact_years >= 2) & (df.age_exact_years < 5)]
        di_current_persistent_diarrhoea_HIV_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'persistent diarrhoea') &
                     (df.has_hiv) & (df.age_exact_years < 5)]
        di_current_persistent_diarrhoea_SAM_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'persistent diarrhoea') &
                     df.malnutrition & (df.age_exact_years < 5)]

        eff_prob_recovery_persistent_diarrhoea = \
            pd.Series(m.r_recovery_persistent_diarrhoea,
                      index=df.index[df.is_alive & (df.ei_diarrhoea_status == 'persistent diarrhoea') &
                                     (df.age_exact_years < 5)])

#       eff_prob_recovery_persistent_diarrhoea.loc[di_current_persistent_diarrhoea_agelt11mo_idx] *= \
#          m.rr_recovery_persistent_diarrhoea_agelt11mo
        eff_prob_recovery_persistent_diarrhoea.loc[di_current_persistent_diarrhoea_age12to23mo_idx] *= \
            m.rr_recovery_persistent_diarrhoea_age12to23mo
        eff_prob_recovery_persistent_diarrhoea.loc[di_current_persistent_diarrhoea_age24to59mo_idx] *= \
            m.rr_recovery_persistent_diarrhoea_age24to59mo
        eff_prob_recovery_persistent_diarrhoea.loc[di_current_persistent_diarrhoea_HIV_idx] *= \
            m.rr_recovery_persistent_diarrhoea_HIV
        eff_prob_recovery_persistent_diarrhoea.loc[di_current_persistent_diarrhoea_SAM_idx] *= \
            m.rr_recovery_persistent_diarrhoea_SAM

        random_draw_04 = pd.Series(rng.random_sample(size=len(di_current_persistent_diarrhoea_idx)),
                                   index=df.index[df.is_alive & (df.ei_diarrhoea_status == 'persistent diarrhoea')
                                                  & df.age_years < 5])

        dfx = pd.concat([eff_prob_recovery_persistent_diarrhoea, random_draw_04], axis=1)
        dfx.columns = ['eff_prob_recovery_persistent_diarrhoea', 'random_draw_04']
        idx_recovery_persistent_diarrhoea = dfx.index[dfx.eff_prob_recovery_persistent_diarrhoea > dfx.random_draw_04]
        df.loc[idx_recovery_persistent_diarrhoea, 'ei_diarrhoea_status'] = 'none'

        # ---------------------------- DEATH FROM PNEUMONIA DISEASE ---------------------------------------

#       eff_prob_death_pneumonia = \
#           pd.Series(m.r_death_pneumonia,
#                     index=df.index[df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') & (df.age_years < 5)])
#       eff_prob_death_pneumonia.loc[pn_current_severe_pneum_agelt2mo_idx] *= \
#           m.rr_death_pneumonia_agelt2mo
#       eff_prob_death_pneumonia.loc[pn_current_severe_pneum_age12to23mo_idx] *= \
#           m.rr_death_pneumonia_age12to23mo
#       eff_prob_death_pneumonia.loc[pn_current_severe_pneum_age24to59mo_idx] *= \
#           m.rr_death_pneumonia_age24to59mo
#       eff_prob_death_pneumonia.loc[pn_current_severe_pneum_HIV_idx] *= \
#           m.rr_death_pneumonia_HIV
#       eff_prob_death_pneumonia.loc[pn_current_severe_pneum_SAM_idx] *= \
#           m.rr_death_pneumonia_SAM

#       random_draw = pd.Series(rng.random_sample(size=len(pn_current_severe_pneumonia_idx)),
#                               index=df.index[(df.age_years < 5) & df.is_alive &
#                                              (df.ri_pneumonia_status == 'severe pneumonia')])

#       dfx = pd.concat([eff_prob_death_pneumonia, random_draw], axis=1)
#       dfx.columns = ['eff_prob_death_pneumonia', 'random_draw']
#       idx_incident_death = dfx.index[dfx.eff_prob_death_pneumonia > dfx.random_draw]
#       df.loc[idx_incident_death, 'ri_pneumonia_death'] = True


class EntericInfectionLoggingEvent(RegularEvent, PopulationScopeEventMixin):
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

        logger.debug('%s|person_one|%s', self.sim.date,
                       df.loc[0].to_dict())
